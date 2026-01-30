'''
We used DistilBERT as the base model for our experiments. Similar to BERT, but smaller and faster to fine-tune.
The code below includes functionality to load LoRA adapters if they are present.
'''
from __future__ import annotations

from pathlib import Path

from transformers import AutoModelForSequenceClassification, AutoTokenizer

try:  # optional for LoRA adapters
    from peft import PeftConfig, PeftModel
except Exception:  # pragma: no cover
    PeftConfig = None
    PeftModel = None


def load_model_and_tokenizer(model_name: str, num_labels: int = 2):
    '''
    Load DistilBERT model and tokenizer, with optional LoRA adapter support.
    Args:
        model_name (str): Name or path of the DistilBERT model or LoRA adapter.
        num_labels (int): Number of labels for sequence classification.
    Returns:
        tokenizer: The loaded tokenizer.
        model: The loaded model.
    '''
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
