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
    # Treat model_name as a local path candidate (works even if it isnt).
    # If its actually a HF hub id, Path(model_name) just won't exist on disk.
    adapter_path = Path(model_name)

    # PEFT adapters typically include adapter_config.json in the adapter folder.
    adapter_config = adapter_path / "adapter_config.json"

    # If adapter_config.json exists, assume this is a LoRA adapter checkpoint.
    if adapter_config.exists():
        # PEFT is an optional dependency; guard with a clear error.
        if PeftConfig is None or PeftModel is None:
            raise ImportError("peft is required to load LoRA adapters. Install it via pip.")

        # Load adapter metadata to find the base model id/path it was trained on.
        peft_config = PeftConfig.from_pretrained(model_name)

        # Tokenizer should come from the base model to ensure vocab + special tokens match.
        tokenizer = AutoTokenizer.from_pretrained(
            peft_config.base_model_name_or_path,
            use_fast=True,
        )

        # Load the base sequence classification model (classifier head sized by num_labels).
        # Important: num_labels should match the training setup / saved head dimensions.
        base_model = AutoModelForSequenceClassification.from_pretrained(
            peft_config.base_model_name_or_path,
            num_labels=num_labels,
        )

        # Attach adapter weights on top of the base model.
        model = PeftModel.from_pretrained(base_model, model_name)

        return tokenizer, model

    # Fallback: load directly as a standard HF model + tokenizer.
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=num_labels,
    )
    return tokenizer, model