from __future__ import annotations

"""Integrated Gradients (IG) explanation pipeline.

Samples inputs, computes token-level attributions, and saves JSONL records with
top-k influential tokens plus the full attribution vector for later analysis.
"""

import json
import random

import numpy as np
from datasets import load_dataset

from aet.data.datasets import load_sst2
from aet.explain.integrated_gradients import compute_integrated_gradients
from aet.models.distilbert import load_model_and_tokenizer
from aet.utils.device import resolve_device
from aet.utils.logging import get_logger
from aet.utils.paths import resolve_model_id, with_run_id
from aet.utils.seed import set_seed


def run(cfg: dict) -> None:
    """Run IG explanation generation and write JSONL samples.

    Expected config keys (nested):
      - data: cache_dir, max_length
      - model: name, num_labels
      - training: output_dir, device
      - explain: split, max_samples, n_steps, top_k, output_dir, model_path, device,
        data_path, text_column
      - project: seed, run_id
    """
    logger = get_logger(__name__)
    logger.info("Running explanation pipeline (Integrated Gradients).")

    seed = cfg.get("project", {}).get("seed", 42)
    set_seed(seed)

    data_cfg = cfg.get("data", {})
    model_cfg = cfg.get("model", {})
    training_cfg = cfg.get("training", {})
    explain_cfg = cfg.get("explain", {})
    project_cfg = cfg.get("project", {})

    cache_dir = data_cfg.get("cache_dir")
    max_length = data_cfg.get("max_length", 128)
    split = explain_cfg.get("split", "validation")
    max_samples = int(explain_cfg.get("max_samples", 50))
    n_steps = int(explain_cfg.get("n_steps", 50))
    top_k = int(explain_cfg.get("top_k", 10))
    run_id = project_cfg.get("run_id")
    output_dir = with_run_id(explain_cfg.get("output_dir", "reports/attributions"), run_id)
    output_dir.mkdir(parents=True, exist_ok=True)

    model_id = resolve_model_id(
        model_path=explain_cfg.get("model_path"),
        training_output_dir=training_cfg.get("output_dir"),
        model_name=model_cfg.get("name", "distilbert-base-uncased"),
        run_id=project_cfg.get("run_id"),
        seed=seed,
    )

    device = resolve_device(explain_cfg.get("device", training_cfg.get("device", "auto")))

    data_path = explain_cfg.get("data_path")
    text_column = explain_cfg.get("text_column", "sentence")
    if data_path:
        # CSV input for custom datasets.
        csv_dataset = load_dataset("csv", data_files=str(data_path))
        split_ds = csv_dataset["train"]
        if text_column not in split_ds.column_names:
            raise ValueError(f"Column '{text_column}' not found in {data_path}.")
    else:
        # Default SST-2 split.
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

    out_path = output_dir / "ig_samples.jsonl"
    # Write one record per sampled input (JSONL for easy streaming).
    with out_path.open("w", encoding="utf-8") as handle:
        for idx in indices:
            text = split_ds[idx][text_column]
            result = compute_integrated_gradients(
                model,
                tokenizer,
                text,
                n_steps=n_steps,
                device=device,
                max_length=max_length,
            )

            # Rank words by absolute attribution magnitude.
            scores = np.array(result.word_attributions, dtype=float)
            top_idx = np.argsort(-np.abs(scores))[:top_k].tolist()
            top_tokens = [
                {"token": result.words[i], "score": float(scores[i])} for i in top_idx
            ]

            record = {
                "id": int(idx),
                "text": result.text,
                "pred_label": result.pred_label,
                "top_tokens": top_tokens,
                "word_attributions": result.word_attributions,
                "words": result.words,
            }
            handle.write(json.dumps(record, ensure_ascii=True) + "\n")

    logger.info("Saved IG samples to %s", out_path)
