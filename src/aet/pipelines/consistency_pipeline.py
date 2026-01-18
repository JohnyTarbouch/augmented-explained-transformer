from __future__ import annotations

import csv
import json
import random
from difflib import SequenceMatcher
from pathlib import Path

import numpy as np

from aet.data.augment import augment_text, get_wordnet_synonyms
from aet.data.datasets import load_sst2
from aet.explain.integrated_gradients import compute_integrated_gradients, predict_label
from aet.metrics.consistency import cosine_similarity, kendall_tau, rank_values, top_k_overlap
from aet.models.distilbert import load_model_and_tokenizer
from aet.utils.device import resolve_device
from aet.utils.logging import get_logger
from aet.utils.seed import set_seed


def _normalize_token(token: str) -> str:
    return token.lower().strip(".,!?;:\"'()[]{}")


def _align_tokens(
    orig_words: list[str],
    aug_words: list[str],
) -> list[tuple[int, int]]:
    norm_orig = [_normalize_token(token) for token in orig_words]
    norm_aug = [_normalize_token(token) for token in aug_words]
    matcher = SequenceMatcher(None, norm_orig, norm_aug)

    pairs: list[tuple[int, int]] = []
    matched_orig: set[int] = set()
    matched_aug: set[int] = set()
    for match in matcher.get_matching_blocks():
        for i in range(match.size):
            orig_idx = match.a + i
            aug_idx = match.b + i
            pairs.append((orig_idx, aug_idx))
            matched_orig.add(orig_idx)
            matched_aug.add(aug_idx)

    unmatched_orig = [i for i in range(len(orig_words)) if i not in matched_orig]
    unmatched_aug = [j for j in range(len(aug_words)) if j not in matched_aug]

    for orig_idx in unmatched_orig:
        if not norm_orig[orig_idx]:
            continue
        syns = get_wordnet_synonyms(norm_orig[orig_idx])
        if not syns:
            continue
        for aug_idx in list(unmatched_aug):
            if norm_aug[aug_idx] in syns:
                pairs.append((orig_idx, aug_idx))
                unmatched_aug.remove(aug_idx)
                break

    pairs.sort()
    return pairs


def _save_histogram(values: list[float], path: Path, title: str, xlabel: str) -> bool:
    try:
        import matplotlib.pyplot as plt
    except Exception:
        return False

    if not values:
        return False

    path.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(6, 4))
    plt.hist(values, bins=20, color="#2b6cb0", edgecolor="#1a202c")
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel("Count")
    plt.tight_layout()
    plt.savefig(path)
    plt.close()
    return True


def run(cfg: dict) -> None:
    logger = get_logger(__name__)
    logger.info("Running consistency pipeline (baseline).")

    seed = cfg.get("project", {}).get("seed", 42)
    set_seed(seed)

    data_cfg = cfg.get("data", {})
    model_cfg = cfg.get("model", {})
    training_cfg = cfg.get("training", {})
    explain_cfg = cfg.get("explain", {})
    aug_cfg = cfg.get("augmentation", {})
    cons_cfg = cfg.get("consistency", {})

    cache_dir = data_cfg.get("cache_dir")
    max_length = data_cfg.get("max_length", 128)
    split = cons_cfg.get("split", "validation")
    max_samples = int(cons_cfg.get("max_samples", 200))
    output_dir = Path(cons_cfg.get("output_dir", "reports/metrics"))
    figures_dir = Path(cons_cfg.get("figures_dir", "reports/figures"))
    output_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    replace_prob = float(aug_cfg.get("replace_prob", 0.1))
    n_steps = int(explain_cfg.get("n_steps", 50))
    top_k = int(explain_cfg.get("top_k", 10))

    model_path = cons_cfg.get("model_path")
    if model_path:
        model_id = str(model_path)
    else:
        output_dir_cfg = training_cfg.get("output_dir")
        config_path = Path(output_dir_cfg) / "config.json" if output_dir_cfg else None
        if config_path and config_path.exists():
            model_id = str(output_dir_cfg)
        else:
            model_id = model_cfg.get("name", "distilbert-base-uncased")

    device = resolve_device(cons_cfg.get("device", training_cfg.get("device", "auto")))

    dataset = load_sst2(cache_dir=cache_dir)
    if split not in dataset:
        raise ValueError(f"Split '{split}' not found in dataset.")
    split_ds = dataset[split]

    rng = random.Random(seed)
    indices = rng.sample(range(len(split_ds)), k=min(max_samples, len(split_ds)))

    tokenizer, model = load_model_and_tokenizer(
        model_id,
        num_labels=model_cfg.get("num_labels", 2),
    )
    model.to(device)
    model.eval()

    rows: list[dict[str, object]] = []
    metrics_tau: list[float] = []
    metrics_topk: list[float] = []
    metrics_cos: list[float] = []
    aligned_counts: list[int] = []

    total = 0
    used = 0

    for idx in indices:
        total += 1
        text = split_ds[idx]["sentence"]
        aug_text = augment_text(text, replace_prob=replace_prob, seed=seed + idx)

        pred_orig = predict_label(
            model,
            tokenizer,
            text,
            device=device,
            max_length=max_length,
        )
        pred_aug = predict_label(
            model,
            tokenizer,
            aug_text,
            device=device,
            max_length=max_length,
        )
        if pred_orig != pred_aug:
            continue

        result_orig = compute_integrated_gradients(
            model,
            tokenizer,
            text,
            target_label=pred_orig,
            n_steps=n_steps,
            device=device,
            max_length=max_length,
        )
        result_aug = compute_integrated_gradients(
            model,
            tokenizer,
            aug_text,
            target_label=pred_orig,
            n_steps=n_steps,
            device=device,
            max_length=max_length,
        )

        pairs = _align_tokens(result_orig.words, result_aug.words)
        if len(pairs) < 2:
            continue

        aligned_orig = np.array([result_orig.word_attributions[i] for i, _ in pairs], dtype=float)
        aligned_aug = np.array([result_aug.word_attributions[j] for _, j in pairs], dtype=float)

        ranks_orig = rank_values(np.abs(aligned_orig))
        ranks_aug = rank_values(np.abs(aligned_aug))
        tau = kendall_tau(ranks_orig, ranks_aug)
        topk = top_k_overlap(aligned_orig, aligned_aug, k=min(top_k, len(aligned_orig)))
        cos = cosine_similarity(aligned_orig, aligned_aug)

        metrics_tau.append(tau)
        metrics_topk.append(topk)
        metrics_cos.append(cos)
        aligned_counts.append(len(pairs))
        used += 1

        rows.append(
            {
                "id": int(idx),
                "pred_label": int(pred_orig),
                "orig_len": len(result_orig.words),
                "aug_len": len(result_aug.words),
                "aligned_tokens": len(pairs),
                "kendall_tau": tau,
                "top_k_overlap": topk,
                "cosine_similarity": cos,
                "text": text,
                "aug_text": aug_text,
            }
        )

    csv_path = output_dir / "consistency_baseline.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "id",
                "pred_label",
                "orig_len",
                "aug_len",
                "aligned_tokens",
                "kendall_tau",
                "top_k_overlap",
                "cosine_similarity",
                "text",
                "aug_text",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)

    summary = {
        "total_samples": total,
        "used_samples": used,
        "mean_kendall_tau": float(np.mean(metrics_tau)) if metrics_tau else 0.0,
        "std_kendall_tau": float(np.std(metrics_tau)) if metrics_tau else 0.0,
        "mean_top_k_overlap": float(np.mean(metrics_topk)) if metrics_topk else 0.0,
        "std_top_k_overlap": float(np.std(metrics_topk)) if metrics_topk else 0.0,
        "mean_cosine_similarity": float(np.mean(metrics_cos)) if metrics_cos else 0.0,
        "std_cosine_similarity": float(np.std(metrics_cos)) if metrics_cos else 0.0,
        "mean_aligned_tokens": float(np.mean(aligned_counts)) if aligned_counts else 0.0,
    }
    summary_path = output_dir / "consistency_baseline_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    _save_histogram(
        metrics_tau,
        figures_dir / "consistency_kendall_tau_hist.png",
        "Kendall Tau Consistency",
        "Kendall Tau",
    )
    _save_histogram(
        metrics_topk,
        figures_dir / "consistency_topk_hist.png",
        "Top-k Overlap Consistency",
        "Top-k Overlap",
    )
    _save_histogram(
        metrics_cos,
        figures_dir / "consistency_cosine_hist.png",
        "Cosine Similarity Consistency",
        "Cosine Similarity",
    )

    logger.info("Saved consistency report to %s", csv_path)
    logger.info("Summary: %s", summary)
