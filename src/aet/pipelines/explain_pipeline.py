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

    # Reproducibility: seed controls sampling + any randomized components in downstream code.
    seed = cfg.get("project", {}).get("seed", 42)
    set_seed(seed)

    # Pull sub-config blocks (keeps config access tidy and consistent).
    data_cfg = cfg.get("data", {})
    model_cfg = cfg.get("model", {})
    training_cfg = cfg.get("training", {})
    explain_cfg = cfg.get("explain", {})
    project_cfg = cfg.get("project", {})

    # Data / explanation parameters.
    cache_dir = data_cfg.get("cache_dir")
    max_length = data_cfg.get("max_length", 128)           # tokenizer truncation length
    split = explain_cfg.get("split", "validation")         # dataset split for SST-2
    max_samples = int(explain_cfg.get("max_samples", 50))  # how many examples to explain
    n_steps = int(explain_cfg.get("n_steps", 50))          # IG integration steps
    top_k = int(explain_cfg.get("top_k", 10))              # how many top tokens to save
    run_id = project_cfg.get("run_id")

    # Output directory is namespaced by run_id (if present) for experiment tracking.
    output_dir = with_run_id(explain_cfg.get("output_dir", "reports/attributions"), run_id)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Resolve model checkpoint/path (trained output vs explicit model_path vs base model name).
    model_id = resolve_model_id(
        model_path=explain_cfg.get("model_path"),
        training_output_dir=training_cfg.get("output_dir"),
        model_name=model_cfg.get("name", "distilbert-base-uncased"),
        run_id=project_cfg.get("run_id"),
        seed=seed,
    )

    # Select compute device ("auto" typically maps to cuda if available).
    device = resolve_device(explain_cfg.get("device", training_cfg.get("device", "auto")))

    # Data source selection:
    # - if data_path is set: load a CSV dataset (train split by HF convention)
    # - else: load SST-2 from the project dataset utility
    data_path = explain_cfg.get("data_path")
    text_column = explain_cfg.get("text_column", "sentence")
    if data_path:
        # CSV input for custom datasets.
        csv_dataset = load_dataset("csv", data_files=str(data_path))
        split_ds = csv_dataset["train"]
        # Validate the requested text column exists
        if text_column not in split_ds.column_names:
            raise ValueError(f"Column '{text_column}' not found in {data_path}.")
    else:
        # Default SST-2 split.
        dataset = load_sst2(cache_dir=cache_dir)
        if split not in dataset:
            raise ValueError(f"Split '{split}' not found in dataset.")
        split_ds = dataset[split]

    # Deterministic sampling for reproducibility (same seed -> same indices).
    rng = random.Random(seed)
    indices = rng.sample(range(len(split_ds)), k=min(max_samples, len(split_ds)))

    # Load model/tokenizer (supports adapter loading depending on implementation).
    tokenizer, model = load_model_and_tokenizer(
        model_id,
        num_labels=model_cfg.get("num_labels", 2),
    )
    model.to(device)
    model.eval()  # ensures deterministic behavior (e.g., disables dropout)

    out_path = output_dir / "ig_samples.jsonl"

    # Write one record per sampled input (JSONL is convenient for streaming and large outputs).
    with out_path.open("w", encoding="utf-8") as handle:
        for idx in indices:
            # Extract the raw text for this example
            text = split_ds[idx][text_column]

            # Compute IG attributions (implementation returns token + word-level attributions)
            result = compute_integrated_gradients(
                model,
                tokenizer,
                text,
                n_steps=n_steps,
                device=device,
                max_length=max_length,
            )

            # Rank words by absolute attribution magnitude.
            # Using abs() surfaces strongly positive or strongly negative contributions.
            scores = np.array(result.word_attributions, dtype=float)
            top_idx = np.argsort(-np.abs(scores))[:top_k].tolist()

            # Store the top-k as a compact (token, score) list for quick inspection.
            top_tokens = [
                {"token": result.words[i], "score": float(scores[i])} for i in top_idx
            ]

            # Persist both a summary (top-k) and full attribution vector for offline analysis.
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
