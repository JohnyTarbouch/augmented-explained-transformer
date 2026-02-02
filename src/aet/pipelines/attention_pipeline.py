from __future__ import annotations

"""
Attention analysis pipeline.

Computes attention-based token scores and evaluates consistency under text
augmentation, plus alignment with Integrated Gradients (IG).
"""

import csv
import json
import random
from difflib import SequenceMatcher
from pathlib import Path

import numpy as np

from aet.data.augment import augment_text, get_wordnet_synonyms
from aet.data.datasets import load_sst2
from aet.explain.attention import compute_attention_scores
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


def _align_tokens(orig_words: list[str], aug_words: list[str]) -> list[tuple[int, int]]:
    """Align tokens between original and augmented texts.

    - Use SequenceMatcher on normalized tokens to get exact/near-exact matches.
    - For leftover unmatched original tokens, try a synonym match (WordNet).
    """
    norm_orig = [_normalize_token(token) for token in orig_words]
    norm_aug = [_normalize_token(token) for token in aug_words]
    matcher = SequenceMatcher(None, norm_orig, norm_aug)

    pairs: list[tuple[int, int]] = []
    matched_orig: set[int] = set()
    matched_aug: set[int] = set()
    
    # 1) Exact sequence matches from difflib (fast + robust to small edits)
    for match in matcher.get_matching_blocks():
        for i in range(match.size):
            orig_idx = match.a + i
            aug_idx = match.b + i
            pairs.append((orig_idx, aug_idx))
            matched_orig.add(orig_idx)
            matched_aug.add(aug_idx)

    # Compute leftover indices not covered by matching blocks
    unmatched_orig = [i for i in range(len(orig_words)) if i not in matched_orig]
    unmatched_aug = [j for j in range(len(aug_words)) if j not in matched_aug]

    for orig_idx in unmatched_orig:
        if not norm_orig[orig_idx]:
            continue
        syns = get_wordnet_synonyms(norm_orig[orig_idx])
        if not syns:
            continue

        # 2) Synonym matching for leftovers (useful for WordNet-based augmentation)
        for aug_idx in list(unmatched_aug):
            if norm_aug[aug_idx] in syns:
                pairs.append((orig_idx, aug_idx))
                unmatched_aug.remove(aug_idx) # avoid double-matching augmented tokens
                break

    # Keep deterministic order (helps debugging / reproducibility)
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
    attn_cfg: dict,
    run_id: str | None,
    seed: int | None,
) -> str:
    """Resolve model id/path to load for attention analysis."""
    return resolve_model_id(
        model_path=attn_cfg.get("model_path"),
        training_output_dir=training_cfg.get("output_dir"),
        model_name=model_cfg.get("name", "distilbert-base-uncased"),
        run_id=run_id,
        seed=seed,
    )


def run(cfg: dict) -> None:
    """Run attention consistency + IG alignment checks and write reports."""
    logger = get_logger(__name__)
    logger.info("Running attention analysis pipeline.")

    # Extract configs and set seed for reproducibility.
    project_cfg = cfg.get("project", {})
    seed = project_cfg.get("seed", 42)
    set_seed(seed)

    data_cfg = cfg.get("data", {})
    model_cfg = cfg.get("model", {})
    training_cfg = cfg.get("training", {})
    explain_cfg = cfg.get("explain", {})
    aug_cfg = cfg.get("augmentation", {})
    attn_cfg = cfg.get("attention", {})

    run_id = project_cfg.get("run_id")
    output_dir = with_run_id(attn_cfg.get("output_dir", "reports/metrics"), run_id)
    figures_dir = with_run_id(attn_cfg.get("figures_dir", "reports/figures"), run_id)
    output_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)
    # Pipeline params
    split = attn_cfg.get("split", "validation")
    max_samples = int(attn_cfg.get("max_samples", 200))
    layer = attn_cfg.get("layer", "last")
    checks = attn_cfg.get("checks", ["consistency", "ig_alignment"])
    top_k = int(attn_cfg.get("top_k", explain_cfg.get("top_k", 10)))
    # Data + augmentation param
    cache_dir = data_cfg.get("cache_dir")
    max_length = data_cfg.get("max_length", 128)
    replace_prob = float(aug_cfg.get("replace_prob", 0.1))
    method = aug_cfg.get("method", "wordnet")
    backtranslation_cfg = aug_cfg.get("backtranslation", {})

    # Load dataset (SST-2 only for attention pipeline).
    dataset = load_sst2(cache_dir=cache_dir)
    if split not in dataset:
        raise ValueError(f"Split '{split}' not found in dataset.")
    split_ds = dataset[split]

    # Deterministic sampling for reproducibility.
    rng = random.Random(seed)
    indices = rng.sample(range(len(split_ds)), k=min(max_samples, len(split_ds)))
    #Load model + tokenizer
    model_id = _resolve_model_id(model_cfg, training_cfg, attn_cfg, run_id, seed)
    device = resolve_device(attn_cfg.get("device", training_cfg.get("device", "auto")))
    tokenizer, model = load_model_and_tokenizer(
        model_id,
        num_labels=model_cfg.get("num_labels", 2),
    )
    model.to(device)
    model.eval()

    # ============================================================
    # 1) Consistency check: attention(orig) vs attention(augmented)
    # ============================================================
    if "consistency" in checks:
        # Compare attention scores on original vs augmented texts.
        rows: list[dict[str, object]] = []
        metrics_tau: dict[str, list[float]] = {"no_flip": [], "flip": []}
        metrics_topk: dict[str, list[float]] = {"no_flip": [], "flip": []}
        metrics_cos: dict[str, list[float]] = {"no_flip": [], "flip": []}
        aligned_counts: dict[str, list[int]] = {"no_flip": [], "flip": []}
        total = 0
        used = 0
        used_by_group = {"no_flip": 0, "flip": 0}
        flip_count = 0

        for idx in indices:
            total += 1
            text = split_ds[idx]["sentence"]
            # Create an augmented version of the text (WordNet/backtranslation/etc.)

            aug_text = augment_text(
                text,
                replace_prob=replace_prob,
                seed=seed + idx,
                method=method,
                backtranslation_cfg=backtranslation_cfg,
            )
            # Determine if the model prediction changes under augmentation
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

            # Compute attention-based word scores for both texts
            attn_orig = compute_attention_scores(
                model,
                tokenizer,
                text,
                device=device,
                max_length=max_length,
                layer=layer,
            )
            attn_aug = compute_attention_scores(
                model,
                tokenizer,
                aug_text,
                device=device,
                max_length=max_length,
                layer=layer,
            )

            # Align words between original and augmented (handles minor edits / synonyms)
            pairs = _align_tokens(attn_orig.words, attn_aug.words)
            if len(pairs) < 2:
                continue # not enough aligned items to compute rank correlations

            aligned_orig = np.array([attn_orig.word_scores[i] for i, _ in pairs], dtype=float)
            aligned_aug = np.array([attn_aug.word_scores[j] for _, j in pairs], dtype=float)

            ranks_orig = rank_values(np.abs(aligned_orig))
            ranks_aug = rank_values(np.abs(aligned_aug))
            tau = kendall_tau(ranks_orig, ranks_aug)
            topk = top_k_overlap(aligned_orig, aligned_aug, k=min(top_k, len(aligned_orig)))
            cos = cosine_similarity(aligned_orig, aligned_aug)
            # Compare rankings by absolute magnitude (importance-style comparison)
            group = "flip" if flip else "no_flip"
            metrics_tau[group].append(tau)
            metrics_topk[group].append(topk)
            metrics_cos[group].append(cos)
            aligned_counts[group].append(len(pairs))
            used += 1
            used_by_group[group] += 1
            # Store per-example details for analysis/debugging
            rows.append(
                {
                    "id": int(idx),
                    "pred_label": int(pred_orig),
                    "orig_len": len(attn_orig.words),
                    "aug_len": len(attn_aug.words),
                    "aligned_tokens": len(pairs),
                    "kendall_tau": tau,
                    "top_k_overlap": topk,
                    "cosine_similarity": cos,
                    "flip": flip,
                    "text": text,
                    "aug_text": aug_text,
                }
            )

        csv_path = output_dir / "attention_consistency.csv"
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
            "check": "consistency",
            "model_id": model_id,
            "layer": layer,
            "total_samples": total,
            "used_samples": used,
            "flip_count": flip_count,
            "flip_rate": float(flip_count / total) if total else 0.0,
            "groups": {},
            "replace_prob": replace_prob,
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
        # Aggregate stats into a small JSON summary
        summary_path = output_dir / "attention_consistency_summary.json"
        summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

        _save_histogram(
            metrics_tau["no_flip"],
            figures_dir / "attention_kendall_tau_hist.png",
            "Kendall Tau (Attention Consistency, no-flip)",
            "Kendall Tau",
        )
        _save_histogram(
            metrics_topk["no_flip"],
            figures_dir / "attention_topk_hist.png",
            "Top-k Overlap (Attention Consistency, no-flip)",
            "Top-k Overlap",
        )
        _save_histogram(
            metrics_cos["no_flip"],
            figures_dir / "attention_cosine_hist.png",
            "Cosine Similarity (Attention Consistency, no-flip)",
            "Cosine Similarity",
        )
        if metrics_tau["flip"]:
            _save_histogram(
                metrics_tau["flip"],
                figures_dir / "attention_kendall_tau_hist_flip.png",
                "Kendall Tau (Attention Consistency, flip)",
                "Kendall Tau",
            )
        if metrics_topk["flip"]:
            _save_histogram(
                metrics_topk["flip"],
                figures_dir / "attention_topk_hist_flip.png",
                "Top-k Overlap (Attention Consistency, flip)",
                "Top-k Overlap",
            )
        if metrics_cos["flip"]:
            _save_histogram(
                metrics_cos["flip"],
                figures_dir / "attention_cosine_hist_flip.png",
                "Cosine Similarity (Attention Consistency, flip)",
                "Cosine Similarity",
            )

        logger.info("Saved attention consistency report to %s", csv_path)
        logger.info("Summary: %s", summary)
    # ============================================================
    # 2) IG alignment: attention(text) vs IG(text) on same example
    # ============================================================
    if "ig_alignment" in checks:
        # Compare attention scores to IG attributions on the same text.
        rows: list[dict[str, object]] = []
        metrics_tau = []
        metrics_topk = []
        metrics_cos = []
        aligned_counts = []
        total = 0
        used = 0

        for idx in indices:
            total += 1
            text = split_ds[idx]["sentence"]
            # Explain with respect to predicted label (common interpretability setup)
            pred = predict_label(
                model,
                tokenizer,
                text,
                device=device,
                max_length=max_length,
            )
            # Compute both attribution signals on the same input
            attn = compute_attention_scores(
                model,
                tokenizer,
                text,
                device=device,
                max_length=max_length,
                layer=layer,
            )
            ig = compute_integrated_gradients(
                model,
                tokenizer,
                text,
                target_label=pred,
                n_steps=int(explain_cfg.get("n_steps", 50)),
                device=device,
                max_length=max_length,
            )

            # Align word lists (should usually match, but keep consistent with other checks)
            pairs = _align_tokens(attn.words, ig.words)
            if len(pairs) < 2:
                continue

            aligned_attn = np.array([attn.word_scores[i] for i, _ in pairs], dtype=float)
            aligned_ig = np.array([ig.word_attributions[j] for _, j in pairs], dtype=float)

            # Compare ranks by absolute magnitude (importance agreement)
            ranks_attn = rank_values(np.abs(aligned_attn))
            ranks_ig = rank_values(np.abs(aligned_ig))
            tau = kendall_tau(ranks_attn, ranks_ig)
            topk = top_k_overlap(aligned_attn, aligned_ig, k=min(top_k, len(aligned_attn)))
            cos = cosine_similarity(aligned_attn, aligned_ig)

            metrics_tau.append(tau)
            metrics_topk.append(topk)
            metrics_cos.append(cos)
            aligned_counts.append(len(pairs))
            used += 1

            rows.append(
                {
                    "id": int(idx),
                    "pred_label": int(pred),
                    "num_words": len(attn.words),
                    "aligned_tokens": len(pairs),
                    "kendall_tau": tau,
                    "top_k_overlap": topk,
                    "cosine_similarity": cos,
                    "text": text,
                }
            )
        
        # Write per-example CSV
        csv_path = output_dir / "attention_ig_alignment.csv"
        with csv_path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(
                handle,
                fieldnames=[
                    "id",
                    "pred_label",
                    "num_words",
                    "aligned_tokens",
                    "kendall_tau",
                    "top_k_overlap",
                    "cosine_similarity",
                    "text",
                ],
            )
            writer.writeheader()
            writer.writerows(rows)

        # Summary JSON for quick reporting
        summary = {
            "check": "ig_alignment",
            "model_id": model_id,
            "layer": layer,
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
        summary_path = output_dir / "attention_ig_alignment_summary.json"
        summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
        # Optional plots
        _save_histogram(
            metrics_tau,
            figures_dir / "attention_ig_kendall_tau_hist.png",
            "Kendall Tau (Attention vs IG)",
            "Kendall Tau",
        )
        _save_histogram(
            metrics_topk,
            figures_dir / "attention_ig_topk_hist.png",
            "Top-k Overlap (Attention vs IG)",
            "Top-k Overlap",
        )
        _save_histogram(
            metrics_cos,
            figures_dir / "attention_ig_cosine_hist.png",
            "Cosine Similarity (Attention vs IG)",
            "Cosine Similarity",
        )

        logger.info("Saved attention-IG alignment report to %s", csv_path)
        logger.info("Summary: %s", summary)
