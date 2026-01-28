from __future__ import annotations

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


def _predict_prob(model, tokenizer, text: str, target_label: int, *, device: str, max_length: int) -> float:
    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        max_length=max_length,
    )
    inputs = {k: v.to(device) for k, v in inputs.items()}
    with torch.no_grad():
        logits = model(**inputs).logits
    probs = logits.softmax(dim=-1)
    return float(probs[0, target_label].item())


def _mask_words(
    words: list[str],
    indices: set[int],
    *,
    strategy: str,
    mask_token: str,
    keep: bool,
) -> str:
    output: list[str] = []
    for idx, word in enumerate(words):
        selected = idx in indices
        if keep:
            if selected:
                output.append(word)
            elif strategy == "mask":
                output.append(mask_token)
        else:
            if selected:
                if strategy == "mask":
                    output.append(mask_token)
            else:
                output.append(word)
    return " ".join(output).strip()


def _fractions(step_fraction: float, max_steps: int | None) -> list[float]:
    if step_fraction <= 0:
        raise ValueError("step_fraction must be positive")
    steps = max_steps or int(math.ceil(1.0 / step_fraction))
    fractions = [min(1.0, step_fraction * i) for i in range(1, steps + 1)]
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
    drops: list[float] = []
    n = len(words)
    for frac in fractions:
        k = max(1, int(math.ceil(frac * n)))
        k = min(k, n)
        idx = set(order[:k])
        perturbed = _mask_words(
            words,
            idx,
            strategy=mask_strategy,
            mask_token=mask_token,
            keep=keep,
        )
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
    if not drops:
        return 0.0
    return float(sum(drops) / len(drops))


def run(cfg: dict) -> None:
    logger = get_logger(__name__)
    logger.info("Running faithfulness pipeline (AOPC).")

    project_cfg = cfg.get("project", {})
    seed = project_cfg.get("seed", 42)
    set_seed(seed)

    data_cfg = cfg.get("data", {})
    model_cfg = cfg.get("model", {})
    training_cfg = cfg.get("training", {})
    explain_cfg = cfg.get("explain", {})
    faith_cfg = cfg.get("faithfulness", {})

    run_id = project_cfg.get("run_id")
    output_dir = with_run_id(faith_cfg.get("output_dir", "reports/metrics"), run_id)
    figures_dir = with_run_id(faith_cfg.get("figures_dir", "reports/figures"), run_id)
    output_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    split = faith_cfg.get("split", "validation")
    max_samples = int(faith_cfg.get("max_samples", 200))
    step_fraction = float(faith_cfg.get("step_fraction", 0.1))
    random_repeats = int(faith_cfg.get("random_repeats", 5))
    mask_strategy = str(faith_cfg.get("mask_strategy", "mask"))
    max_steps = faith_cfg.get("max_steps")
    max_steps = int(max_steps) if max_steps is not None else None

    n_steps = int(faith_cfg.get("n_steps", explain_cfg.get("n_steps", 50)))

    cache_dir = data_cfg.get("cache_dir")
    max_length = data_cfg.get("max_length", 128)

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

    if text_column not in dataset.column_names:
        raise ValueError(f"Column '{text_column}' not found in dataset.")

    rng = random.Random(seed)
    indices = rng.sample(range(len(dataset)), k=min(max_samples, len(dataset)))

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

    mask_token = tokenizer.mask_token or "[MASK]"
    fractions = _fractions(step_fraction, max_steps)

    out_path = output_dir / "faithfulness_aopc.jsonl"
    comp_aopc_vals: list[float] = []
    suff_aopc_vals: list[float] = []
    comp_rand_vals: list[float] = []
    suff_rand_vals: list[float] = []
    comp_curve_sum = [0.0 for _ in fractions]
    suff_curve_sum = [0.0 for _ in fractions]
    rand_comp_curve_sum = [0.0 for _ in fractions]
    rand_suff_curve_sum = [0.0 for _ in fractions]
    used = 0

    with out_path.open("w", encoding="utf-8") as handle:
        for idx in indices:
            text = dataset[idx][text_column]
            pred_label = predict_label(
                model,
                tokenizer,
                text,
                device=device,
                max_length=max_length,
            )

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
            if len(words) < 2 or len(words) != len(scores):
                continue

            orig_prob = _predict_prob(
                model,
                tokenizer,
                text,
                pred_label,
                device=device,
                max_length=max_length,
            )

            order = sorted(range(len(words)), key=lambda i: abs(scores[i]), reverse=True)
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

            comp_aopc = _aopc(comp_drops)
            suff_aopc = _aopc(suff_drops)

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

                rand_comp_aopc /= random_repeats
                rand_suff_aopc /= random_repeats
                rand_comp_curve = [v / random_repeats for v in rand_comp_curve]
                rand_suff_curve = [v / random_repeats for v in rand_suff_curve]

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

            comp_aopc_vals.append(comp_aopc)
            suff_aopc_vals.append(suff_aopc)
            comp_rand_vals.append(rand_comp_aopc)
            suff_rand_vals.append(rand_suff_aopc)
            comp_curve_sum = [a + b for a, b in zip(comp_curve_sum, comp_drops)]
            suff_curve_sum = [a + b for a, b in zip(suff_curve_sum, suff_drops)]
            rand_comp_curve_sum = [a + b for a, b in zip(rand_comp_curve_sum, rand_comp_curve)]
            rand_suff_curve_sum = [a + b for a, b in zip(rand_suff_curve_sum, rand_suff_curve)]
            used += 1

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

    try:
        import matplotlib.pyplot as plt

        if used and fractions:
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
