from __future__ import annotations

from pathlib import Path

import numpy as np
from sklearn.metrics import accuracy_score
from transformers import DataCollatorWithPadding, Trainer, TrainingArguments

from aet.data.datasets import load_csv_dataset, load_sst2, tokenize_sst2
from aet.models.distilbert import load_model_and_tokenizer
from aet.utils.device import resolve_device
from aet.utils.logging import get_logger
from aet.utils.seed import set_seed


def _to_int(value: object, default: int) -> int:
    if value is None:
        return default
    return int(value)


def _to_float(value: object, default: float) -> float:
    if value is None:
        return default
    return float(value)


def train_baseline(cfg: dict) -> None:
    logger = get_logger(__name__)
    seed = cfg.get("project", {}).get("seed", 42)
    set_seed(seed)

    data_cfg = cfg.get("data", {})
    training_cfg = cfg.get("training", {})
    model_cfg = cfg.get("model", {})

    cache_dir = data_cfg.get("cache_dir")
    max_length = data_cfg.get("max_length", 128)
    output_dir = Path(training_cfg.get("output_dir", "models/baseline"))
    output_dir.mkdir(parents=True, exist_ok=True)

    tokenizer, model = load_model_and_tokenizer(
        model_cfg.get("name", "distilbert-base-uncased"),
        num_labels=model_cfg.get("num_labels", 2),
    )
    if training_cfg.get("skip_train", False):
        logger.info("Skipping training; saving pretrained model to %s", output_dir)
        model.save_pretrained(str(output_dir))
        tokenizer.save_pretrained(str(output_dir))
        return

    lora_cfg = training_cfg.get("lora", {})
    if lora_cfg.get("enabled", False):
        try:
            from peft import LoraConfig, get_peft_model
        except Exception as exc:  # pragma: no cover
            raise ImportError("peft is required for LoRA training. Install it via pip.") from exc

        target_modules = lora_cfg.get("target_modules", ["q_lin", "k_lin", "v_lin", "out_lin"])
        lora_config = LoraConfig(
            r=_to_int(lora_cfg.get("r"), 8),
            lora_alpha=_to_int(lora_cfg.get("alpha"), 16),
            lora_dropout=_to_float(lora_cfg.get("dropout"), 0.1),
            target_modules=target_modules,
            bias="none",
            task_type="SEQ_CLS",
        )
        model = get_peft_model(model, lora_config)
        if hasattr(model, "print_trainable_parameters"):
            model.print_trainable_parameters()

    train_data_path = training_cfg.get("train_data_path")
    eval_data_path = training_cfg.get("eval_data_path")
    text_column = training_cfg.get("text_column", "sentence")
    label_column = training_cfg.get("label_column", "label")

    if train_data_path:
        dataset = load_csv_dataset(str(train_data_path), str(eval_data_path) if eval_data_path else None)
    else:
        dataset = load_sst2(cache_dir=cache_dir)

    tokenized = tokenize_sst2(
        dataset,
        tokenizer,
        max_length=max_length,
        text_column=text_column,
        label_column=label_column,
    )

    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    def compute_metrics(eval_pred):
        predictions = eval_pred.predictions if hasattr(eval_pred, "predictions") else eval_pred[0]
        labels = eval_pred.label_ids if hasattr(eval_pred, "label_ids") else eval_pred[1]
        if isinstance(predictions, tuple):
            predictions = predictions[0]
        preds = np.argmax(predictions, axis=-1)
        return {"accuracy": accuracy_score(labels, preds)}

    device = resolve_device(training_cfg.get("device", "auto"))

    args = TrainingArguments(
        output_dir=str(output_dir),
        per_device_train_batch_size=_to_int(training_cfg.get("batch_size"), 16),
        per_device_eval_batch_size=_to_int(training_cfg.get("batch_size"), 16),
        num_train_epochs=_to_float(training_cfg.get("epochs"), 3.0),
        learning_rate=_to_float(training_cfg.get("lr"), 2e-5),
        weight_decay=_to_float(training_cfg.get("weight_decay"), 0.0),
        eval_strategy="epoch",
        save_strategy="epoch",
        logging_strategy="steps",
        logging_steps=50,
        save_total_limit=1,
        load_best_model_at_end=True,
        metric_for_best_model="accuracy",
        report_to="none",
        seed=seed,
        no_cuda=device == "cpu",
        use_cpu=device == "cpu",
    )

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=tokenized["train"],
        eval_dataset=tokenized["validation"],
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )

    logger.info("Starting baseline training.")
    trainer.train()
    trainer.save_model()
    tokenizer.save_pretrained(str(output_dir))
    logger.info("Baseline model saved to %s", output_dir)


def train_augmented(cfg: dict) -> None:
    raise NotImplementedError("Implement augmented model training.")
