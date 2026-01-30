from __future__ import annotations

"""Consistency pipeline (baseline IG stability).

Generates IG explanations for original vs augmented texts and measures
stability using Kendall tau, top-k overlap, and cosine similarity. (metrics/consistency.py)
"""

import csv
import json
import random
from difflib import SequenceMatcher

import numpy as np

from aet.data.augment import augment_text, get_wordnet_synonyms
from aet.data.datasets import load_sst2
from aet.explain.integrated_gradients import compute_integrated_gradients, predict_label
from aet.metrics.consistency import cosine_similarity, kendall_tau, rank_values, top_k_overlap
from aet.models.distilbert import load_model_and_tokenizer
from aet.utils.device import resolve_device
from aet.utils.logging import get_logger
from aet.utils.paths import resolve_model_id, with_run_id
from aet.utils.seed import set_seed


def _normalize_token(token: str) -> str:
    """Normalize token for alignment (lowercase + strip punctuation)."""
    return token.lower().strip(".,!?;:\"'()[]{}")


def _align_tokens(
    orig_words: list[str],
    aug_words: list[str],
) -> list[tuple[int, int]]:
    """Align tokens between original and augmented texts.

    Uses sequence matching first, then attempts synonym matches for rest.
    Why we do this: Augmentations may change some words to synonyms,
    so exact matching may miss these alignments.
    
    Args:
        orig_words (list[str]): Tokens from original text.
        aug_words (list[str]): Tokens from augmented text.  
    returns:
        list[tuple[int, int]]: List of (orig_idx, aug_idx) token index pairs.
    """
    norm_orig = [_normalize_token(token) for token in orig_words]
    norm_aug = [_normalize_token(token) for token in aug_words]
    matcher = SequenceMatcher(None, norm_orig, norm_aug)
    # first exact matches from sequence matcher
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
    # synonym matches for unmatched tokens
    unmatched_orig = [i for i in range(len(orig_words)) if i not in matched_orig]
    unmatched_aug = [j for j in range(len(aug_words)) if j not in matched_aug]
    # try to match remaining tokens via synonyms
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
    """Save a histogram"""
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
    """Run IG consistency checks and write per-sample + summary outputs."""
    logger = get_logger(__name__)
    logger.info("Running consistency pipeline (baseline).")

    # Extract configs and set seed for reproducibility.
    project_cfg = cfg.get("project", {})
    seed = project_cfg.get("seed", 42)
    set_seed(seed)
    # Extract other configs.
    data_cfg = cfg.get("data", {})
    model_cfg = cfg.get("model", {})
    training_cfg = cfg.get("training", {})
    explain_cfg = cfg.get("explain", {})
    aug_cfg = cfg.get("augmentation", {})
    cons_cfg = cfg.get("consistency", {})
    run_id = project_cfg.get("run_id")

    cache_dir = data_cfg.get("cache_dir")
    max_length = data_cfg.get("max_length", 128)
    split = cons_cfg.get("split", "validation")
    max_samples = int(cons_cfg.get("max_samples", 200))
    output_dir = with_run_id(cons_cfg.get("output_dir", "reports/metrics"), run_id)
    figures_dir = with_run_id(cons_cfg.get("figures_dir", "reports/figures"), run_id)
    output_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    replace_prob = float(aug_cfg.get("replace_prob", 0.1))
    method = aug_cfg.get("method", "wordnet")
    backtranslation_cfg = aug_cfg.get("backtranslation", {})
    n_steps = int(explain_cfg.get("n_steps", 50))
    top_k = int(explain_cfg.get("top_k", 10))
    # Resolve model ID (trained or basel;ine)
    model_id = resolve_model_id(
        model_path=cons_cfg.get("model_path"),
        training_output_dir=training_cfg.get("output_dir"),
        model_name=model_cfg.get("name", "distilbert-base-uncased"),
        run_id=run_id,
        seed=seed,
    )

    device = resolve_device(cons_cfg.get("device", training_cfg.get("device", "auto")))

    # Load dataset (SST-2 only for baseline consistency).
    dataset = load_sst2(cache_dir=cache_dir)
    if split not in dataset:
        raise ValueError(f"Split '{split}' not found in dataset.")
    split_ds = dataset[split]

    # Deterministic sampling for reproducibility.
    rng = random.Random(seed)
    indices = rng.sample(range(len(split_ds)), k=min(max_samples, len(split_ds)))

    tokenizer, model = load_model_and_tokenizer(
        model_id,
        num_labels=model_cfg.get("num_labels", 2),
    )
    model.to(device)
    model.eval()
    # Process samples
    rows: list[dict[str, object]] = []
    metrics_tau: dict[str, list[float]] = {"no_flip": [], "flip": []}
    metrics_topk: dict[str, list[float]] = {"no_flip": [], "flip": []}
    metrics_cos: dict[str, list[float]] = {"no_flip": [], "flip": []}
    aligned_counts: dict[str, list[int]] = {"no_flip": [], "flip": []}

    total = 0
    used = 0
    used_by_group = {"no_flip": 0, "flip": 0}
    flip_count = 0
    # Iterate over samples
    for idx in indices:
        total += 1
        text = split_ds[idx]["sentence"]
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
            target_label=pred_aug,
            n_steps=n_steps,
            device=device,
            max_length=max_length,
        )

        # Compare only aligned tokens (unchanged or synonym-matched).
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
                "orig_len": len(result_orig.words),
                "aug_len": len(result_aug.words),
                "aligned_tokens": len(pairs),
                "kendall_tau": tau,
                "top_k_overlap": topk,
                "cosine_similarity": cos,
                "flip": flip,
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
                "flip",
                "text",
                "aug_text",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)

    summary = {
        "total_samples": total,
        "used_samples": used,
        "flip_count": flip_count,
        "flip_rate": float(flip_count / total) if total else 0.0,
        "groups": {},
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
    summary_path = output_dir / "consistency_baseline_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    _save_histogram(
        metrics_tau["no_flip"],
        figures_dir / "consistency_kendall_tau_hist.png",
        "Kendall Tau Consistency (no-flip)",
        "Kendall Tau",
    )
    _save_histogram(
        metrics_topk["no_flip"],
        figures_dir / "consistency_topk_hist.png",
        "Top-k Overlap Consistency (no-flip)",
        "Top-k Overlap",
    )
    _save_histogram(
        metrics_cos["no_flip"],
        figures_dir / "consistency_cosine_hist.png",
        "Cosine Similarity Consistency (no-flip)",
        "Cosine Similarity",
    )
    if metrics_tau["flip"]:
        _save_histogram(
            metrics_tau["flip"],
            figures_dir / "consistency_kendall_tau_hist_flip.png",
            "Kendall Tau Consistency (flip)",
            "Kendall Tau",
        )
    if metrics_topk["flip"]:
        _save_histogram(
            metrics_topk["flip"],
            figures_dir / "consistency_topk_hist_flip.png",
            "Top-k Overlap Consistency (flip)",
            "Top-k Overlap",
        )
    if metrics_cos["flip"]:
        _save_histogram(
            metrics_cos["flip"],
            figures_dir / "consistency_cosine_hist_flip.png",
            "Cosine Similarity Consistency (flip)",
            "Cosine Similarity",
        )

    logger.info("Saved consistency report to %s", csv_path)
    logger.info("Summary: %s", summary)
