from __future__ import annotations

"""
LIME consistency pipeline.

Generates LIME explanations for original vs augmented texts and compares
stability using Kendall tau, top-k overlap, and cosine similarity.
"""

import csv
import json
import random
from difflib import SequenceMatcher
from pathlib import Path

import numpy as np
from datasets import load_dataset

from aet.data.augment import augment_text, get_wordnet_synonyms
from aet.data.datasets import load_sst2
from aet.explain.integrated_gradients import predict_label
from aet.explain.lime_explainer import compute_lime_explanation
from aet.metrics.consistency import cosine_similarity, kendall_tau, rank_values, top_k_overlap
from aet.models.distilbert import load_model_and_tokenizer
from aet.utils.device import resolve_device
from aet.utils.logging import get_logger
from aet.utils.paths import resolve_model_id, with_run_id
from aet.utils.seed import set_seed


def _normalize_token(token: str) -> str:
    """Normalize token for alignment (lowercase + strip punctuation).

    Improves matching of tokens that differ only by casing or punctuation.
    """
    return token.lower().strip(".,!?;:\"'()[]{}")


def _align_tokens(orig_words: list[str], aug_words: list[str]) -> list[tuple[int, int]]:
    """Align tokens between original and augmented texts.

    Uses sequence matching first, then attempts synonym matches for leftovers.
    """
    norm_orig = [_normalize_token(token) for token in orig_words]
    norm_aug = [_normalize_token(token) for token in aug_words]
    matcher = SequenceMatcher(None, norm_orig, norm_aug)
    # First pass: direct matches
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
    # Second pass: synonym matches for unmatched tokens
    unmatched_orig = [i for i in range(len(orig_words)) if i not in matched_orig]
    unmatched_aug = [j for j in range(len(aug_words)) if j not in matched_aug]
    # Try to match unmatched original tokens to augmented tokens via synonyms
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
    """Save a histogram if matplotlib is available."""
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


def _resolve_model_id(
    model_cfg: dict,
    training_cfg: dict,
    lime_cfg: dict,
    run_id: str | None,
    seed: int | None,
) -> str:
    """Resolve model id/path to load for LIME evaluation."""
    return resolve_model_id(
        model_path=lime_cfg.get("model_path"),
        training_output_dir=training_cfg.get("output_dir"),
        model_name=model_cfg.get("name", "distilbert-base-uncased"),
        run_id=run_id,
        seed=seed,
    )


def run(cfg: dict) -> None:
    """Run LIME consistency and write per-sample + summary outputs.

    Compares LIME explanations for original vs augmented texts, grouped by
    whether the model prediction flips under augmentation.
    """
    logger = get_logger(__name__)
    logger.info("Running LIME consistency pipeline.")
    # set seed
    project_cfg = cfg.get("project", {})
    seed = project_cfg.get("seed", 42)
    set_seed(seed)
    # extract configs
    data_cfg = cfg.get("data", {})
    model_cfg = cfg.get("model", {})
    training_cfg = cfg.get("training", {})
    aug_cfg = cfg.get("augmentation", {})
    lime_cfg = cfg.get("lime", {})
    # output dirs
    run_id = project_cfg.get("run_id")
    output_dir = with_run_id(lime_cfg.get("output_dir", "reports/metrics"), run_id)
    figures_dir = with_run_id(lime_cfg.get("figures_dir", "reports/figures"), run_id)
    output_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)
    # data params
    split = lime_cfg.get("split", "validation")
    max_samples = int(lime_cfg.get("max_samples", 200))
    num_samples = int(lime_cfg.get("num_samples", 500))
    max_features = lime_cfg.get("max_features")
    max_features = int(max_features) if max_features is not None else None
    top_k = int(lime_cfg.get("top_k", 10))
    class_names = lime_cfg.get("class_names") or ["negative", "positive"]

    cache_dir = data_cfg.get("cache_dir")
    max_length = data_cfg.get("max_length", 128)
    replace_prob = float(aug_cfg.get("replace_prob", 0.1))
    method = aug_cfg.get("method", "wordnet")
    backtranslation_cfg = aug_cfg.get("backtranslation", {})
    # load dataset
    data_path = lime_cfg.get("data_path")
    text_column = lime_cfg.get("text_column", "sentence")
    if data_path:
        csv_dataset = load_dataset("csv", data_files=str(data_path))
        split_ds = csv_dataset["train"]
        if text_column not in split_ds.column_names:
            raise ValueError(f"Column '{text_column}' not found in {data_path}.")
    else:
        dataset = load_sst2(cache_dir=cache_dir)
        if split not in dataset:
            raise ValueError(f"Split '{split}' not found in dataset.")
        split_ds = dataset[split]

    rng = random.Random(seed)  # TODO: reproducible sampling for multiseeds
    indices = rng.sample(range(len(split_ds)), k=min(max_samples, len(split_ds)))
    # load model
    model_id = _resolve_model_id(model_cfg, training_cfg, lime_cfg, run_id, seed)
    device = resolve_device(lime_cfg.get("device", training_cfg.get("device", "auto")))
    tokenizer, model = load_model_and_tokenizer(
        model_id,
        num_labels=model_cfg.get("num_labels", 2),
    )
    model.to(device)
    model.eval()
    # process samples
    rows: list[dict[str, object]] = []
    metrics_tau: dict[str, list[float]] = {"no_flip": [], "flip": []}
    metrics_topk: dict[str, list[float]] = {"no_flip": [], "flip": []}
    metrics_cos: dict[str, list[float]] = {"no_flip": [], "flip": []}
    aligned_counts: dict[str, list[int]] = {"no_flip": [], "flip": []}
    total = 0
    used = 0
    used_by_group = {"no_flip": 0, "flip": 0}
    flip_count = 0
    # Iterate over sampled indices, compute explanations and metrics
    for idx in indices:
        total += 1
        text = split_ds[idx][text_column]
        aug_text = augment_text(
            text,
            replace_prob=replace_prob,
            seed=seed + idx,
            method=method,
            backtranslation_cfg=backtranslation_cfg,
        )

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
        flip = pred_orig != pred_aug
        if flip:
            flip_count += 1

        lime_orig = compute_lime_explanation(
            model,
            tokenizer,
            text,
            num_samples=num_samples,
            max_features=max_features,
            seed=seed + idx,
            device=device,
            max_length=max_length,
            class_names=class_names,
        )
        lime_aug = compute_lime_explanation(
            model,
            tokenizer,
            aug_text,
            num_samples=num_samples,
            max_features=max_features,
            seed=seed + idx + 1,
            device=device,
            max_length=max_length,
            class_names=class_names,
        )

        # Compare only aligned tokens (unchanged or synonym-matched).
        pairs = _align_tokens(lime_orig.tokens, lime_aug.tokens)
        if len(pairs) < 2:
            continue

        aligned_orig = np.array([lime_orig.token_weights[i] for i, _ in pairs], dtype=float)
        aligned_aug = np.array([lime_aug.token_weights[j] for _, j in pairs], dtype=float)
        # Compute metrics
        ranks_orig = rank_values(np.abs(aligned_orig))
        ranks_aug = rank_values(np.abs(aligned_aug))
        tau = kendall_tau(ranks_orig, ranks_aug)
        topk = top_k_overlap(aligned_orig, aligned_aug, k=min(top_k, len(aligned_orig)))
        cos = cosine_similarity(aligned_orig, aligned_aug)
        # Record metrics
        group = "flip" if flip else "no_flip"
        metrics_tau[group].append(tau)
        metrics_topk[group].append(topk)
        metrics_cos[group].append(cos)
        aligned_counts[group].append(len(pairs))
        used += 1
        used_by_group[group] += 1

        rows.append(
            {
                "id": int(idx),
                "pred_label": int(pred_orig),
                "orig_len": len(lime_orig.tokens),
                "aug_len": len(lime_aug.tokens),
                "aligned_tokens": len(pairs),
                "kendall_tau": tau,
                "top_k_overlap": topk,
                "cosine_similarity": cos,
                "flip": flip,
                "text": text,
                "aug_text": aug_text,
            }
        )

    csv_path = output_dir / "lime_consistency.csv"
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
                "flip",
                "text",
                "aug_text",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)

    # Summary aggregates per-group metrics and flip rates.
    summary = {
        "check": "consistency",
        "model_id": model_id,
        "total_samples": total,
        "used_samples": used,
        "flip_count": flip_count,
        "flip_rate": float(flip_count / total) if total else 0.0,
        "groups": {},
        "replace_prob": replace_prob,
        "num_samples": num_samples,
        "max_features": max_features,
    }
    for group in ("no_flip", "flip"):
        summary["groups"][group] = {
            "used_samples": used_by_group[group],
            "mean_kendall_tau": float(np.mean(metrics_tau[group])) if metrics_tau[group] else 0.0,
            "std_kendall_tau": float(np.std(metrics_tau[group])) if metrics_tau[group] else 0.0,
            "mean_top_k_overlap": float(np.mean(metrics_topk[group])) if metrics_topk[group] else 0.0,
            "std_top_k_overlap": float(np.std(metrics_topk[group])) if metrics_topk[group] else 0.0,
            "mean_cosine_similarity": float(np.mean(metrics_cos[group])) if metrics_cos[group] else 0.0,
            "std_cosine_similarity": float(np.std(metrics_cos[group])) if metrics_cos[group] else 0.0,
            "mean_aligned_tokens": float(np.mean(aligned_counts[group]))
            if aligned_counts[group]
            else 0.0,
        }
    summary_path = output_dir / "lime_consistency_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    _save_histogram(
        metrics_tau["no_flip"],
        figures_dir / "lime_kendall_tau_hist.png",
        "Kendall Tau (LIME Consistency, no-flip)",
        "Kendall Tau",
    )
    _save_histogram(
        metrics_topk["no_flip"],
        figures_dir / "lime_topk_hist.png",
        "Top-k Overlap (LIME Consistency, no-flip)",
        "Top-k Overlap",
    )
    _save_histogram(
        metrics_cos["no_flip"],
        figures_dir / "lime_cosine_hist.png",
        "Cosine Similarity (LIME Consistency, no-flip)",
        "Cosine Similarity",
    )
    if metrics_tau["flip"]:
        _save_histogram(
            metrics_tau["flip"],
            figures_dir / "lime_kendall_tau_hist_flip.png",
            "Kendall Tau (LIME Consistency, flip)",
            "Kendall Tau",
        )
    if metrics_topk["flip"]:
        _save_histogram(
            metrics_topk["flip"],
            figures_dir / "lime_topk_hist_flip.png",
            "Top-k Overlap (LIME Consistency, flip)",
            "Top-k Overlap",
        )
    if metrics_cos["flip"]:
        _save_histogram(
            metrics_cos["flip"],
            figures_dir / "lime_cosine_hist_flip.png",
            "Cosine Similarity (LIME Consistency, flip)",
            "Cosine Similarity",
        )

    logger.info("Saved LIME consistency report to %s", csv_path)
    logger.info("Summary: %s", summary)
