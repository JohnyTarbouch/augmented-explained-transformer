from __future__ import annotations

"""
Faithfulness pipeline using AOPC (comprehensiveness/sufficiency).

We remove or keep top-ranked tokens (by IG) and track confidence drops.
This tests whether explanations are causally meaningful.
"""

import json
import math
import random

import numpy as np
import torch
from datasets import load_dataset

from aet.data.datasets import load_sst2
from aet.explain.integrated_gradients import compute_integrated_gradients, predict_label
from aet.models.distilbert import load_model_and_tokenizer
from aet.utils.device import resolve_device
from aet.utils.logging import get_logger
from aet.utils.paths import resolve_model_id, with_run_id
from aet.utils.seed import set_seed


def _predict_prob(
    model,
    tokenizer,
    text: str,
    target_label: int,
    *,
    device: str,
    max_length: int,
) -> float:
    """Return model probability for a target label on a single text."""
    # Tokenize single example (no padding needed here).
    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        max_length=max_length,
    )
    # Move tensors to the same device as the model.
    inputs = {k: v.to(device) for k, v in inputs.items()}
    # Inference-only: no gradients required for probability queries.
    with torch.no_grad():
        logits = model(**inputs).logits
    # Convert logits -> probabilities; take the probability of target_label.
    probs = logits.softmax(dim=-1)  # shape (1, num_labels)
    return float(probs[0, target_label].item())


def _mask_words(
    words: list[str],
    indices: set[int],
    *,
    strategy: str,
    mask_token: str,
    keep: bool,
) -> str:
    """Mask or keep selected words based on strategy.

    When keep=True, selected words are preserved and others are masked.
    When keep=False, selected words are masked and others are preserved.
    """
    output: list[str] = []
    # Build perturbed text by iterating word-by-word.
    for idx, word in enumerate(words):
        selected = idx in indices
        if keep:
            # Sufficiency-style: keep top words, mask the rest.
            if selected:
                output.append(word)
            elif strategy == "mask":
                output.append(mask_token)
        else:
            # Comprehensiveness-style: remove top words, keep the rest.
            if selected:
                if strategy == "mask":
                    output.append(mask_token)
            else:
                output.append(word)
    # Join back into a single string; strip avoids leading/trailing whitespace artifacts.
    return " ".join(output).strip()


def _fractions(step_fraction: float, max_steps: int | None) -> list[float]:
    """Compute monotonic fraction steps for perturbation curves."""
    # Fractions control how aggressively we perturb text over the curve (e.g., 0.1, 0.2, ..., 1.0).
    if step_fraction <= 0:
        raise ValueError("step_fraction must be positive")
    # If max_steps is not provided, cover [step_fraction, 1.0] in uniform steps.
    steps = max_steps or int(math.ceil(1.0 / step_fraction))
    fractions = [min(1.0, step_fraction * i) for i in range(1, steps + 1)]
    # Ensure strictly increasing fractions (avoid duplicates due to rounding).
    unique: list[float] = []
    for frac in fractions:
        if not unique or frac > unique[-1] + 1e-9:
            unique.append(frac)
    return unique


def _curve_for_order(
    order: list[int],
    *,
    words: list[str],
    fractions: list[float],
    orig_prob: float,
    target_label: int,
    model,
    tokenizer,
    device: str,
    max_length: int,
    mask_strategy: str,
    mask_token: str,
    keep: bool,
) -> list[float]:
    """Compute confidence drop curve for a given token order.

    This supports comprehensiveness (keep=False) and sufficiency (keep=True).
    """
    drops: list[float] = []
    n = len(words)
    for frac in fractions:
        # Determine how many tokens to mask/keep at this fraction.
        # We force at least 1 token and cap at n to keep behavior well-defined.
        k = max(1, int(math.ceil(frac * n)))
        k = min(k, n)

        # Select top-k indices according to the provided ordering (IG-ranked or random).
        idx = set(order[:k])

        # Build the perturbed text by masking/keeping according to the setting.
        perturbed = _mask_words(
            words,
            idx,
            strategy=mask_strategy,
            mask_token=mask_token,
            keep=keep,
        )

        # Confidence drop relative to original probability.
        # Positive drop => model is less confident after perturbation (desired for faithful top tokens).
        prob = _predict_prob(
            model,
            tokenizer,
            perturbed,
            target_label,
            device=device,
            max_length=max_length,
        )
        drops.append(orig_prob - prob)
    return drops


def _aopc(drops: list[float]) -> float:
    """Area over perturbation curve (AOPC)."""
    # AOPC is just the mean drop across the curve steps.
    if not drops:
        return 0.0
    return float(sum(drops) / len(drops))


def run(cfg: dict) -> None:
    """Run faithfulness evaluation and write JSONL + summary + plots.

    Uses IG to rank tokens and evaluates comprehensiveness/sufficiency by
    masking/keeping top-ranked tokens at multiple fractions.
    """
    # Similar to robustness pipeline and other pipelines
    logger = get_logger(__name__)
    logger.info("Running faithfulness pipeline (AOPC).")

    # --- Reproducibility: seed affects sampling + any RNG-based baselines ---
    project_cfg = cfg.get("project", {})
    seed = project_cfg.get("seed", 42)
    set_seed(seed)

    # --- Config blocks ---
    data_cfg = cfg.get("data", {})
    model_cfg = cfg.get("model", {})
    training_cfg = cfg.get("training", {})
    explain_cfg = cfg.get("explain", {})
    faith_cfg = cfg.get("faithfulness", {})

    # --- Output dirs (optionally namespaced by run_id) ---
    run_id = project_cfg.get("run_id")
    output_dir = with_run_id(faith_cfg.get("output_dir", "reports/metrics"), run_id)
    figures_dir = with_run_id(faith_cfg.get("figures_dir", "reports/figures"), run_id)
    output_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    # --- Evaluation parameters ---
    split = faith_cfg.get("split", "validation")
    max_samples = int(faith_cfg.get("max_samples", 200))
    step_fraction = float(faith_cfg.get("step_fraction", 0.1))
    random_repeats = int(faith_cfg.get("random_repeats", 5))
    mask_strategy = str(faith_cfg.get("mask_strategy", "mask"))
    max_steps = faith_cfg.get("max_steps")
    max_steps = int(max_steps) if max_steps is not None else None

    # IG integration steps (fallback to explain_cfg if not set under faithfulness).
    n_steps = int(faith_cfg.get("n_steps", explain_cfg.get("n_steps", 50)))

    # --- Data parameters ---
    cache_dir = data_cfg.get("cache_dir")
    max_length = data_cfg.get("max_length", 128)

    # --- Dataset selection: custom CSV or SST-2 ---
    data_path = faith_cfg.get("data_path")
    text_column = faith_cfg.get("text_column", "sentence")
    if data_path:
        csv_dataset = load_dataset("csv", data_files=str(data_path))
        dataset = csv_dataset["train"]
    else:
        dataset_dict = load_sst2(cache_dir=cache_dir)
        if split not in dataset_dict:
            raise ValueError(f"Split '{split}' not found in dataset.")
        dataset = dataset_dict[split]

    # Validate expected schema early.
    if text_column not in dataset.column_names:
        raise ValueError(f"Column '{text_column}' not found in dataset.")

    # Deterministic sampling for reproducibility.
    rng = random.Random(seed)
    indices = rng.sample(range(len(dataset)), k=min(max_samples, len(dataset)))

    # --- Model loading ---
    model_id = resolve_model_id(
        model_path=faith_cfg.get("model_path"),
        training_output_dir=training_cfg.get("output_dir"),
        model_name=model_cfg.get("name", "distilbert-base-uncased"),
        run_id=run_id,
        seed=seed,
    )

    device = resolve_device(faith_cfg.get("device", training_cfg.get("device", "auto")))
    tokenizer, model = load_model_and_tokenizer(model_id, num_labels=model_cfg.get("num_labels", 2))
    model.to(device)
    model.eval()

    # Use model mask token (or fallback) for deletion-style perturbations.
    mask_token = tokenizer.mask_token or "[MASK]"

    # Fractions define curve x-axis (how much we remove/keep).
    fractions = _fractions(step_fraction, max_steps)

    out_path = output_dir / "faithfulness_aopc.jsonl"

    # Collect per-sample AOPC values to summarize later (IG vs random baseline).
    comp_aopc_vals: list[float] = []
    suff_aopc_vals: list[float] = []
    comp_rand_vals: list[float] = []
    suff_rand_vals: list[float] = []

    # Also accumulate mean curves (sum then divide by used at the end).
    comp_curve_sum = [0.0 for _ in fractions]
    suff_curve_sum = [0.0 for _ in fractions]
    rand_comp_curve_sum = [0.0 for _ in fractions]
    rand_suff_curve_sum = [0.0 for _ in fractions]
    used = 0

    with out_path.open("w", encoding="utf-8") as handle:
        for idx in indices:
            text = dataset[idx][text_column]

            # Predict label to define the "target" class whose confidence we track.
            pred_label = predict_label(
                model,
                tokenizer,
                text,
                device=device,
                max_length=max_length,
            )

            # Compute IG attributions for the predicted label (word-level in result.words).
            ig = compute_integrated_gradients(
                model,
                tokenizer,
                text,
                target_label=pred_label,
                n_steps=n_steps,
                device=device,
                max_length=max_length,
            )
            words = ig.words
            scores = ig.word_attributions

            # Skip degenerate cases where IG is not comparable.
            if len(words) < 2 or len(words) != len(scores):
                continue

            # Original probability baseline for confidence-drop computation.
            orig_prob = _predict_prob(
                model,
                tokenizer,
                text,
                pred_label,
                device=device,
                max_length=max_length,
            )

            # Rank words by absolute IG magnitude (strong positive or negative contributions).
            order = sorted(range(len(words)), key=lambda i: abs(scores[i]), reverse=True)

            # Comprehensiveness: remove top-ranked words; should drop confidence if explanation is faithful.
            comp_drops = _curve_for_order(
                order,
                words=words,
                fractions=fractions,
                orig_prob=orig_prob,
                target_label=pred_label,
                model=model,
                tokenizer=tokenizer,
                device=device,
                max_length=max_length,
                mask_strategy=mask_strategy,
                mask_token=mask_token,
                keep=False,
            )

            # Sufficiency: keep only top-ranked words; model should retain confidence if explanation is sufficient.
            suff_drops = _curve_for_order(
                order,
                words=words,
                fractions=fractions,
                orig_prob=orig_prob,
                target_label=pred_label,
                model=model,
                tokenizer=tokenizer,
                device=device,
                max_length=max_length,
                mask_strategy=mask_strategy,
                mask_token=mask_token,
                keep=True,
            )

            # Scalar AOPC summaries for this sample.
            comp_aopc = _aopc(comp_drops)
            suff_aopc = _aopc(suff_drops)

            # Random baseline: shuffle the same index set to estimate "chance" faithfulness.
            rand_comp_aopc = 0.0
            rand_suff_aopc = 0.0
            rand_comp_curve = [0.0 for _ in fractions]
            rand_suff_curve = [0.0 for _ in fractions]

            if random_repeats > 0:
                for _ in range(random_repeats):
                    random_order = order[:]
                    rng.shuffle(random_order)

                    rc = _curve_for_order(
                        random_order,
                        words=words,
                        fractions=fractions,
                        orig_prob=orig_prob,
                        target_label=pred_label,
                        model=model,
                        tokenizer=tokenizer,
                        device=device,
                        max_length=max_length,
                        mask_strategy=mask_strategy,
                        mask_token=mask_token,
                        keep=False,
                    )
                    rs = _curve_for_order(
                        random_order,
                        words=words,
                        fractions=fractions,
                        orig_prob=orig_prob,
                        target_label=pred_label,
                        model=model,
                        tokenizer=tokenizer,
                        device=device,
                        max_length=max_length,
                        mask_strategy=mask_strategy,
                        mask_token=mask_token,
                        keep=True,
                    )

                    rand_comp_aopc += _aopc(rc)
                    rand_suff_aopc += _aopc(rs)
                    rand_comp_curve = [a + b for a, b in zip(rand_comp_curve, rc)]
                    rand_suff_curve = [a + b for a, b in zip(rand_suff_curve, rs)]

                # Average across random repeats.
                rand_comp_aopc /= random_repeats
                rand_suff_aopc /= random_repeats
                rand_comp_curve = [v / random_repeats for v in rand_comp_curve]
                rand_suff_curve = [v / random_repeats for v in rand_suff_curve]

            # Persist per-sample record for later inspection/analysis.
            record = {
                "id": int(idx),
                "pred_label": int(pred_label),
                "num_words": len(words),
                "fractions": fractions,
                "comprehensiveness_drops": comp_drops,
                "sufficiency_drops": suff_drops,
                "aopc_comprehensiveness": comp_aopc,
                "aopc_sufficiency": suff_aopc,
                "random_aopc_comprehensiveness": rand_comp_aopc,
                "random_aopc_sufficiency": rand_suff_aopc,
                "random_comprehensiveness_drops": rand_comp_curve,
                "random_sufficiency_drops": rand_suff_curve,
            }
            handle.write(json.dumps(record, ensure_ascii=True) + "\n")

            # Accumulate summary stats.
            comp_aopc_vals.append(comp_aopc)
            suff_aopc_vals.append(suff_aopc)
            comp_rand_vals.append(rand_comp_aopc)
            suff_rand_vals.append(rand_suff_aopc)
            comp_curve_sum = [a + b for a, b in zip(comp_curve_sum, comp_drops)]
            suff_curve_sum = [a + b for a, b in zip(suff_curve_sum, suff_drops)]
            rand_comp_curve_sum = [a + b for a, b in zip(rand_comp_curve_sum, rand_comp_curve)]
            rand_suff_curve_sum = [a + b for a, b in zip(rand_suff_curve_sum, rand_suff_curve)]
            used += 1

    # Aggregate summary across all used samples.
    summary = {
        "total_samples": len(indices),
        "used_samples": used,
        "fractions": fractions,
        "mask_strategy": mask_strategy,
        "step_fraction": step_fraction,
        "random_repeats": random_repeats,
        "aopc_comprehensiveness_mean": float(np.mean(comp_aopc_vals)) if comp_aopc_vals else 0.0,
        "aopc_comprehensiveness_std": float(np.std(comp_aopc_vals)) if comp_aopc_vals else 0.0,
        "aopc_sufficiency_mean": float(np.mean(suff_aopc_vals)) if suff_aopc_vals else 0.0,
        "aopc_sufficiency_std": float(np.std(suff_aopc_vals)) if suff_aopc_vals else 0.0,
        "random_aopc_comprehensiveness_mean": float(np.mean(comp_rand_vals)) if comp_rand_vals else 0.0,
        "random_aopc_comprehensiveness_std": float(np.std(comp_rand_vals)) if comp_rand_vals else 0.0,
        "random_aopc_sufficiency_mean": float(np.mean(suff_rand_vals)) if suff_rand_vals else 0.0,
        "random_aopc_sufficiency_std": float(np.std(suff_rand_vals)) if suff_rand_vals else 0.0,
        "mean_comprehensiveness_curve": [
            float(v / used) for v in comp_curve_sum
        ]
        if used
        else [],
        "mean_sufficiency_curve": [
            float(v / used) for v in suff_curve_sum
        ]
        if used
        else [],
        "mean_random_comprehensiveness_curve": [
            float(v / used) for v in rand_comp_curve_sum
        ]
        if used
        else [],
        "mean_random_sufficiency_curve": [
            float(v / used) for v in rand_suff_curve_sum
        ]
        if used
        else [],
    }

    summary_path = output_dir / "faithfulness_aopc_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    # Optional plotting (skips gracefully if matplotlib missing).
    try:
        import matplotlib.pyplot as plt

        if used and fractions:
            # Comprehensiveness curve: remove fraction of tokens; higher drop => more faithful ranking.
            plt.figure(figsize=(6, 4))
            plt.plot(fractions, summary["mean_comprehensiveness_curve"], label="IG", color="#2b6cb0")
            if random_repeats > 0:
                plt.plot(
                    fractions,
                    summary["mean_random_comprehensiveness_curve"],
                    label="random",
                    color="#c05621",
                )
            plt.title("Comprehensiveness AOPC Curve")
            plt.xlabel("Fraction removed")
            plt.ylabel("Prob drop")
            plt.legend()
            plt.tight_layout()
            plt.savefig(figures_dir / "faithfulness_comprehensiveness_curve.png")
            plt.close()

            # Sufficiency curve: keep fraction of tokens; lower drop => top tokens are sufficient.
            plt.figure(figsize=(6, 4))
            plt.plot(fractions, summary["mean_sufficiency_curve"], label="IG", color="#2b6cb0")
            if random_repeats > 0:
                plt.plot(
                    fractions,
                    summary["mean_random_sufficiency_curve"],
                    label="random",
                    color="#c05621",
                )
            plt.title("Sufficiency AOPC Curve")
            plt.xlabel("Fraction kept")
            plt.ylabel("Prob drop")
            plt.legend()
            plt.tight_layout()
            plt.savefig(figures_dir / "faithfulness_sufficiency_curve.png")
            plt.close()
    except Exception:
        logger.info("matplotlib not available; skipping AOPC plots.")

    logger.info("Saved faithfulness AOPC report to %s", out_path)
