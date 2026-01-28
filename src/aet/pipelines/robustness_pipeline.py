from __future__ import annotations

import json
import random
from importlib.util import find_spec

from datasets import load_dataset

from aet.data.datasets import load_sst2
from aet.models.distilbert import load_model_and_tokenizer
from aet.utils.device import resolve_device
from aet.utils.logging import get_logger
from aet.utils.paths import resolve_model_id, with_run_id
from aet.utils.seed import set_seed


def _has_module(name: str) -> bool:
    return find_spec(name) is not None


def _disable_use_constraint(attack) -> bool:
    before = len(attack.constraints)
    attack.constraints = [
        constraint
        for constraint in attack.constraints
        if constraint.__class__.__name__ != "UniversalSentenceEncoder"
    ]
    return len(attack.constraints) != before


def run(cfg: dict) -> None:
    logger = get_logger(__name__)
    logger.info("Running robustness pipeline (TextFooler).")

    try:
        from textattack import Attacker, AttackArgs
        from textattack.attack_recipes import TextFoolerJin2019
        from textattack.attack_results import (
            FailedAttackResult,
            SkippedAttackResult,
            SuccessfulAttackResult,
        )
        from textattack.datasets import Dataset
        from textattack.models.wrappers import HuggingFaceModelWrapper
    except Exception as exc:  # pragma: no cover - optional dependency
        raise RuntimeError(
            "textattack is required. Install with: python -m pip install textattack"
        ) from exc

    project_cfg = cfg.get("project", {})
    seed = project_cfg.get("seed", 42)
    set_seed(seed)

    data_cfg = cfg.get("data", {})
    model_cfg = cfg.get("model", {})
    training_cfg = cfg.get("training", {})
    robust_cfg = cfg.get("robustness", {})

    run_id = project_cfg.get("run_id")
    output_dir = with_run_id(robust_cfg.get("output_dir", "reports/metrics"), run_id)
    output_dir.mkdir(parents=True, exist_ok=True)

    split = robust_cfg.get("split", "validation")
    max_samples = int(robust_cfg.get("max_samples", 200))
    attack_name = str(robust_cfg.get("attack", "textfooler")).lower()
    data_path = robust_cfg.get("data_path")
    text_column = robust_cfg.get("text_column", "sentence")
    label_column = robust_cfg.get("label_column", "label")

    cache_dir = data_cfg.get("cache_dir")
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
    if label_column not in dataset.column_names:
        raise ValueError(f"Column '{label_column}' not found in dataset.")

    rng = random.Random(seed)
    indices = rng.sample(range(len(dataset)), k=min(max_samples, len(dataset)))
    examples = [
        (dataset[idx][text_column], int(dataset[idx][label_column])) for idx in indices
    ]
    attack_dataset = Dataset(examples)

    model_id = resolve_model_id(
        model_path=robust_cfg.get("model_path"),
        training_output_dir=training_cfg.get("output_dir"),
        model_name=model_cfg.get("name", "distilbert-base-uncased"),
        run_id=run_id,
        seed=seed,
    )

    device = resolve_device(robust_cfg.get("device", training_cfg.get("device", "auto")))
    tokenizer, model = load_model_and_tokenizer(
        model_id,
        num_labels=model_cfg.get("num_labels", 2),
    )
    model.to(device)
    model.eval()

    model_wrapper = HuggingFaceModelWrapper(model, tokenizer)
    if attack_name != "textfooler":
        raise ValueError(f"Unsupported attack '{attack_name}'.")
    attack = TextFoolerJin2019.build(model_wrapper)
    use_constraint_enabled = True
    if not (_has_module("tensorflow_hub") and _has_module("tensorflow")):
        if _disable_use_constraint(attack):
            use_constraint_enabled = False
            logger.warning(
                "tensorflow_hub/tensorflow not found; disabling UniversalSentenceEncoder constraint."
            )

    results_path = output_dir / "textfooler_results.csv"
    attack_args = AttackArgs(
        num_examples=len(examples),
        log_to_csv=str(results_path),
        disable_stdout=True,
        random_seed=seed,
        shuffle=False,
    )

    attacker = Attacker(attack, attack_dataset, attack_args)
    results = attacker.attack_dataset()

    success = sum(isinstance(r, SuccessfulAttackResult) for r in results)
    failed = sum(isinstance(r, FailedAttackResult) for r in results)
    skipped = sum(isinstance(r, SkippedAttackResult) for r in results)
    attempted = success + failed

    summary = {
        "attack": "textfooler",
        "model_id": model_id,
        "split": split if not data_path else "csv",
        "total_samples": len(results),
        "attempted": attempted,
        "successful": success,
        "failed": failed,
        "skipped": skipped,
        "attack_success_rate": float(success / attempted) if attempted else 0.0,
        "robust_accuracy": float(failed / attempted) if attempted else 0.0,
        "use_constraint_enabled": use_constraint_enabled,
    }
    summary_path = output_dir / "textfooler_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    logger.info("Saved TextFooler results to %s", results_path)
    logger.info("Summary: %s", summary)
