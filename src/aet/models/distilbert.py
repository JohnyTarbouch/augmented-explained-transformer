from transformers import AutoModelForSequenceClassification, AutoTokenizer


def load_model_and_tokenizer(model_name: str, num_labels: int = 2):
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=num_labels,
    )
    return tokenizer, model
