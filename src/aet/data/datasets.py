from __future__ import annotations

from typing import Any

from datasets import DatasetDict, load_dataset
from transformers import PreTrainedTokenizerBase


def load_sst2(cache_dir: str | None = None) -> DatasetDict:
    return load_dataset("glue", "sst2", cache_dir=cache_dir)


def tokenize_sst2(
    dataset: DatasetDict,
    tokenizer: PreTrainedTokenizerBase,
    max_length: int = 128,
) -> DatasetDict:
    def tokenize_batch(batch: dict[str, Any]) -> dict[str, Any]:
        return tokenizer(
            batch["sentence"],
            truncation=True,
            max_length=max_length,
        )

    remove_columns = [col for col in ["sentence", "idx"] if col in dataset["train"].column_names]
    tokenized = dataset.map(tokenize_batch, batched=True, remove_columns=remove_columns)
    if "label" in tokenized["train"].column_names:
        tokenized = tokenized.rename_column("label", "labels")
    tokenized.set_format(type="torch")
    return tokenized
