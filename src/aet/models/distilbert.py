from __future__ import annotations

from pathlib import Path

from transformers import AutoModelForSequenceClassification, AutoTokenizer

try:  # optional for LoRA adapters
    from peft import PeftConfig, PeftModel
except Exception:  # pragma: no cover
    PeftConfig = None
    PeftModel = None


def load_model_and_tokenizer(model_name: str, num_labels: int = 2):
    adapter_path = Path(model_name)
    adapter_config = adapter_path / "adapter_config.json"

    if adapter_config.exists():
        if PeftConfig is None or PeftModel is None:
            raise ImportError("peft is required to load LoRA adapters. Install it via pip.")
        peft_config = PeftConfig.from_pretrained(model_name)
        tokenizer = AutoTokenizer.from_pretrained(peft_config.base_model_name_or_path, use_fast=True)
        base_model = AutoModelForSequenceClassification.from_pretrained(
            peft_config.base_model_name_or_path,
            num_labels=num_labels,
        )
        model = PeftModel.from_pretrained(base_model, model_name)
        return tokenizer, model

    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=num_labels,
    )
    return tokenizer, model
