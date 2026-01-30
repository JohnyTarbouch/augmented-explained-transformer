"""
LIME explanation methode

Wraps LIME's text explainer to produce token-level weights aligned to a
simple word tokenizer.

Commands:
  - Install extra: `python -m pip install -e ".[lime]"`
  - Run via pipeline: `python -m aet.cli --config configs/base.yaml --stage lime`
"""

from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Callable

import numpy as np
import torch
from transformers import PreTrainedModel, PreTrainedTokenizerBase


@dataclass
class LimeResult:
    text: str
    tokens: list[str]
    token_weights: list[float]
    pred_label: int


def _tokenize_words(text: str) -> list[str]:
    """Tokenize into lowercase word tokens (letters/digits/underscore)"""
    return [tok.lower() for tok in re.findall(r"\w+", text)]


def build_predict_proba(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizerBase,
    *,
    device: str,
    max_length: int,
) -> Callable[[list[str]], np.ndarray]:
    """Build a batched predict_proba callable for LIME"""
    def predict(texts: list[str]) -> np.ndarray:
        # LIME passes a list of raw strings; we must return class probabilities
        inputs = tokenizer(
            texts,
            return_tensors="pt",
            truncation=True,
            padding=True,
            max_length=max_length,
        )
        inputs = {k: v.to(device) for k, v in inputs.items()}
        with torch.no_grad():
            logits = model(**inputs).logits
        probs = torch.softmax(logits, dim=-1).detach().cpu().numpy()
        return probs

    return predict


def compute_lime_explanation(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizerBase,
    text: str,
    *,
    num_samples: int,
    max_features: int | None,
    seed: int,
    device: str,
    max_length: int,
    class_names: list[str] | None = None,
) -> LimeResult:
    """Compute a LIME explanation for a single text

    Args:
        model: Hugging Face classifier.
        tokenizer: Matching tokenizer.
        text: Input text.
        num_samples: LIME perturbation samples.
        max_features: Number of tokens to include (defaults to len(tokens)).
        seed: Random seed for LIME.
        device: Torch device string.
        max_length: Max token length (truncates longer texts).
        class_names: Optional class names for LIME plots.
    """
    try:
        from lime.lime_text import LimeTextExplainer
    except Exception as exc:  # pragma: no cover - optional dependency
        raise RuntimeError("lime is required. Install with: python -m pip install lime") from exc

    model.eval()
    model.to(device)

    # Build a predict_proba wrapper for LIME and get the predicted class
    predict_proba = build_predict_proba(model, tokenizer, device=device, max_length=max_length)
    probs = predict_proba([text])
    pred_label = int(np.argmax(probs, axis=-1)[0])

    explainer = LimeTextExplainer(class_names=class_names, random_state=seed)
    num_features = max_features
    if num_features is None:
        num_features = max(1, len(_tokenize_words(text)))

    explanation = explainer.explain_instance(
        text,
        predict_proba,
        labels=[pred_label],
        num_features=num_features,
        num_samples=num_samples,
    )

    # Convert LIME weights into a dense vector aligned to our tokenization
    weights = explanation.as_list(label=pred_label)
    weight_map = {word.lower(): float(score) for word, score in weights}
    tokens = _tokenize_words(text)
    token_weights = [weight_map.get(token, 0.0) for token in tokens]

    return LimeResult(
        text=text,
        tokens=tokens,
        token_weights=token_weights,
        pred_label=pred_label,
    )
