from __future__ import annotations

from typing import Any

from datasets import DatasetDict, load_dataset
from transformers import PreTrainedTokenizerBase


def load_sst2(cache_dir: str | None = None) -> DatasetDict:
    return load_dataset("glue", "sst2", cache_dir=cache_dir)


def load_csv_dataset(
    train_path: str,
    validation_path: str | None = None,
) -> DatasetDict:
    data_files: dict[str, str] = {"train": train_path}
    if validation_path:
        data_files["validation"] = validation_path
    return load_dataset("csv", data_files=data_files)


def tokenize_sst2(
    dataset: DatasetDict,
    tokenizer: PreTrainedTokenizerBase,
    max_length: int = 128,
    text_column: str = "sentence",
    label_column: str = "label",
) -> DatasetDict:
    def tokenize_batch(batch: dict[str, Any]) -> dict[str, Any]:
        return tokenizer(
            batch[text_column],
            truncation=True,
            max_length=max_length,
        )

    remove_columns = [
        col
        for col in [text_column, "idx", "source"]
        if col in dataset["train"].column_names
    ]
    tokenized = dataset.map(tokenize_batch, batched=True, remove_columns=remove_columns)
    if label_column in tokenized["train"].column_names and label_column != "labels":
        tokenized = tokenized.rename_column(label_column, "labels")
    tokenized.set_format(type="torch")
    return tokenized
