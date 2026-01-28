from __future__ import annotations

import json
import random

import numpy as np
import torch
from datasets import load_dataset

from aet.data.datasets import load_sst2
from aet.explain.integrated_gradients import compute_integrated_gradients, predict_label
from aet.metrics.consistency import cosine_similarity, kendall_tau, rank_values, top_k_overlap
from aet.models.distilbert import load_model_and_tokenizer
from aet.utils.device import resolve_device
from aet.utils.logging import get_logger
from aet.utils.paths import resolve_model_id, with_run_id
from aet.utils.seed import set_seed


def _reset_module(module: torch.nn.Module) -> None:
    if hasattr(module, "reset_parameters"):
        module.reset_parameters()


def _randomize_classifier(model) -> None:
    for name in ("pre_classifier", "classifier"):
        module = getattr(model, name, None)
        if module is not None:
            _reset_module(module)


def _randomize_last_layers(model, count: int) -> None:
    if count <= 0:
        return
    try:
        layers = model.distilbert.transformer.layer
    except AttributeError:
        return
    for layer in layers[-count:]:
        layer.apply(_reset_module)


def _randomize_embeddings(model) -> None:
    try:
        embeddings = model.distilbert.embeddings
    except AttributeError:
        return
    embeddings.apply(_reset_module)


def _compute_similarity(a: list[float], b: list[float], *, top_k: int) -> tuple[float, float, float] | None:
    if len(a) != len(b) or len(a) < 2:
        return None
    arr_a = np.array(a, dtype=float)
    arr_b = np.array(b, dtype=float)
    ranks_a = rank_values(np.abs(arr_a))
    ranks_b = rank_values(np.abs(arr_b))
    tau = kendall_tau(ranks_a, ranks_b)
    topk = top_k_overlap(arr_a, arr_b, k=min(top_k, len(arr_a)))
    cos = cosine_similarity(arr_a, arr_b)
    return tau, topk, cos


def run(cfg: dict) -> None:
    logger = get_logger(__name__)
    logger.info("Running attribution sanity checks (IG randomization).")

    project_cfg = cfg.get("project", {})
    seed = project_cfg.get("seed", 42)
    set_seed(seed)

    data_cfg = cfg.get("data", {})
    model_cfg = cfg.get("model", {})
    training_cfg = cfg.get("training", {})
    explain_cfg = cfg.get("explain", {})
    sanity_cfg = cfg.get("sanity", {})

    run_id = project_cfg.get("run_id")
    output_dir = with_run_id(sanity_cfg.get("output_dir", "reports/metrics"), run_id)
    figures_dir = with_run_id(sanity_cfg.get("figures_dir", "reports/figures"), run_id)
    output_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    split = sanity_cfg.get("split", "validation")
    max_samples = int(sanity_cfg.get("max_samples", 50))
    top_k = int(sanity_cfg.get("top_k", explain_cfg.get("top_k", 10)))
    n_steps = int(sanity_cfg.get("n_steps", explain_cfg.get("n_steps", 50)))
    include_embeddings = bool(sanity_cfg.get("include_embeddings", False))
    max_layers = sanity_cfg.get("max_layers")

    cache_dir = data_cfg.get("cache_dir")
    max_length = data_cfg.get("max_length", 128)

    data_path = sanity_cfg.get("data_path")
    text_column = sanity_cfg.get("text_column", "sentence")
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
        model_path=sanity_cfg.get("model_path"),
        training_output_dir=training_cfg.get("output_dir"),
        model_name=model_cfg.get("name", "distilbert-base-uncased"),
        run_id=project_cfg.get("run_id"),
        seed=seed,
    )

    device = resolve_device(sanity_cfg.get("device", training_cfg.get("device", "auto")))

    tokenizer, base_model = load_model_and_tokenizer(model_id, num_labels=model_cfg.get("num_labels", 2))
    base_model.to(device)
    base_model.eval()

    base_igs: dict[int, list[float]] = {}
    base_labels: dict[int, int] = {}
    for idx in indices:
        text = dataset[idx][text_column]
        pred_label = predict_label(
            base_model,
            tokenizer,
            text,
            device=device,
            max_length=max_length,
        )
        ig = compute_integrated_gradients(
            base_model,
            tokenizer,
            text,
            target_label=pred_label,
            n_steps=n_steps,
            device=device,
            max_length=max_length,
        )
        base_igs[idx] = ig.word_attributions
        base_labels[idx] = pred_label

    try:
        num_layers = len(base_model.distilbert.transformer.layer)
    except AttributeError:
        num_layers = 0
    if max_layers is None:
        max_layers = num_layers
    else:
        max_layers = min(int(max_layers), num_layers)

    levels: list[tuple[str, int, bool]] = [("trained", 0, False), ("classifier", 0, True)]
    for i in range(1, max_layers + 1):
        levels.append((f"last_{i}", i, True))
    if include_embeddings:
        levels.append(("all_plus_embeddings", max_layers, True))

    per_level_rows: list[dict[str, object]] = []
    sample_rows_path = output_dir / "sanity_ig_randomization_samples.jsonl"
    with sample_rows_path.open("w", encoding="utf-8") as sample_handle:
        for level_idx, (level_name, layer_count, randomize_classifier) in enumerate(levels):
            if level_name == "trained":
                taus = [1.0 for _ in base_igs]
                topks = [1.0 for _ in base_igs]
                coss = [1.0 for _ in base_igs]
            else:
                _, model = load_model_and_tokenizer(model_id, num_labels=model_cfg.get("num_labels", 2))
                if randomize_classifier:
                    _randomize_classifier(model)
                _randomize_last_layers(model, layer_count)
                if include_embeddings and level_name == "all_plus_embeddings":
                    _randomize_embeddings(model)
                model.to(device)
                model.eval()

                taus = []
                topks = []
                coss = []
                for idx in indices:
                    text = dataset[idx][text_column]
                    pred_label = base_labels.get(idx)
                    if pred_label is None:
                        continue
                    ig = compute_integrated_gradients(
                        model,
                        tokenizer,
                        text,
                        target_label=pred_label,
                        n_steps=n_steps,
                        device=device,
                        max_length=max_length,
                    )
                    base_scores = base_igs.get(idx)
                    if base_scores is None:
                        continue
                    sim = _compute_similarity(base_scores, ig.word_attributions, top_k=top_k)
                    if sim is None:
                        continue
                    tau, topk, cos = sim
                    taus.append(tau)
                    topks.append(topk)
                    coss.append(cos)

                    sample_handle.write(
                        json.dumps(
                            {
                                "id": int(idx),
                                "level": level_idx,
                                "level_name": level_name,
                                "kendall_tau": tau,
                                "top_k_overlap": topk,
                                "cosine_similarity": cos,
                            },
                            ensure_ascii=True,
                        )
                        + "\n"
                    )

            per_level_rows.append(
                {
                    "level": level_idx,
                    "level_name": level_name,
                    "num_samples": len(taus),
                    "mean_kendall_tau": float(np.mean(taus)) if taus else 0.0,
                    "mean_top_k_overlap": float(np.mean(topks)) if topks else 0.0,
                    "mean_cosine_similarity": float(np.mean(coss)) if coss else 0.0,
                }
            )

    csv_path = output_dir / "sanity_ig_randomization.csv"
    with csv_path.open("w", encoding="utf-8") as handle:
        handle.write("level,level_name,num_samples,mean_kendall_tau,mean_top_k_overlap,mean_cosine_similarity\n")
        for row in per_level_rows:
            handle.write(
                f"{row['level']},{row['level_name']},{row['num_samples']},"
                f"{row['mean_kendall_tau']},{row['mean_top_k_overlap']},"
                f"{row['mean_cosine_similarity']}\n"
            )

    summary_path = output_dir / "sanity_ig_randomization_summary.json"
    summary_path.write_text(json.dumps(per_level_rows, indent=2), encoding="utf-8")

    try:
        import matplotlib.pyplot as plt

        levels_idx = [row["level"] for row in per_level_rows]
        taus = [row["mean_kendall_tau"] for row in per_level_rows]
        topks = [row["mean_top_k_overlap"] for row in per_level_rows]
        coss = [row["mean_cosine_similarity"] for row in per_level_rows]

        plt.figure(figsize=(6, 4))
        plt.plot(levels_idx, taus, marker="o", label="kendall_tau", color="#2b6cb0")
        plt.plot(levels_idx, topks, marker="o", label="top_k_overlap", color="#c05621")
        plt.plot(levels_idx, coss, marker="o", label="cosine_similarity", color="#2f855a")
        plt.title("IG sanity check: similarity vs randomization")
        plt.xlabel("Randomization level")
        plt.ylabel("Similarity to trained IG")
        plt.xticks(levels_idx, [row["level_name"] for row in per_level_rows], rotation=45, ha="right")
        plt.legend()
        plt.tight_layout()
        plt.savefig(figures_dir / "sanity_ig_randomization.png")
        plt.close()
    except Exception:
        logger.info("matplotlib not available; skipping sanity plots.")

    logger.info("Saved sanity check report to %s", csv_path)
