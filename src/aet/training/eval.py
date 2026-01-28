from __future__ import annotations

import numpy as np
from sklearn.metrics import accuracy_score
from transformers import DataCollatorWithPadding, Trainer, TrainingArguments

from aet.data.datasets import load_sst2, tokenize_sst2
from aet.models.distilbert import load_model_and_tokenizer
from aet.utils.device import resolve_device
from aet.utils.logging import get_logger
from aet.utils.paths import resolve_model_id, with_run_id


def evaluate_model(cfg: dict) -> dict[str, float]:
    logger = get_logger(__name__)

    data_cfg = cfg.get("data", {})
    model_cfg = cfg.get("model", {})
    training_cfg = cfg.get("training", {})
    eval_cfg = cfg.get("evaluation", {})
    project_cfg = cfg.get("project", {})

    cache_dir = data_cfg.get("cache_dir")
    max_length = data_cfg.get("max_length", 128)
    eval_split = eval_cfg.get("split", "validation")
    run_id = project_cfg.get("run_id")
    output_dir = with_run_id(eval_cfg.get("output_dir", "reports/metrics"), run_id)
    output_dir.mkdir(parents=True, exist_ok=True)

    model_id = resolve_model_id(
        model_path=eval_cfg.get("model_path"),
        training_output_dir=training_cfg.get("output_dir"),
        model_name=model_cfg.get("name", "distilbert-base-uncased"),
        run_id=project_cfg.get("run_id"),
        seed=project_cfg.get("seed"),
    )

    dataset = load_sst2(cache_dir=cache_dir)
    if eval_split not in dataset:
        raise ValueError(f"Split '{eval_split}' not found in dataset.")
    if "label" not in dataset[eval_split].column_names:
        raise ValueError(f"Split '{eval_split}' has no labels for evaluation.")

    tokenizer, model = load_model_and_tokenizer(
        model_id,
        num_labels=model_cfg.get("num_labels", 2),
    )
    tokenized = tokenize_sst2(dataset, tokenizer, max_length=max_length)
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    def compute_metrics(eval_pred):
        predictions = eval_pred.predictions if hasattr(eval_pred, "predictions") else eval_pred[0]
        labels = eval_pred.label_ids if hasattr(eval_pred, "label_ids") else eval_pred[1]
        if isinstance(predictions, tuple):
            predictions = predictions[0]
        preds = np.argmax(predictions, axis=-1)
        return {"accuracy": accuracy_score(labels, preds)}

    device = resolve_device(eval_cfg.get("device", training_cfg.get("device", "auto")))

    args = TrainingArguments(
        output_dir=str(output_dir),
        per_device_eval_batch_size=int(eval_cfg.get("batch_size", training_cfg.get("batch_size", 16))),
        do_train=False,
        do_eval=True,
        eval_strategy="no",
        report_to="none",
        no_cuda=device == "cpu",
        use_cpu=device == "cpu",
    )

    trainer = Trainer(
        model=model,
        args=args,
        eval_dataset=tokenized[eval_split],
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )

    logger.info("Evaluating model '%s' on split '%s'.", model_id, eval_split)
    metrics = trainer.evaluate()
    logger.info("Eval metrics: %s", metrics)
    return metrics
