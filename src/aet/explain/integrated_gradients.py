"""
Integrated Gradients explanation module.
Provides functionality to compute Integrated Gradients (IG) attributions for text classification models.
"""

from __future__ import annotations

from dataclasses import dataclass
import re

import torch
from captum.attr import LayerIntegratedGradients
from transformers import PreTrainedModel, PreTrainedTokenizerBase


@dataclass
class IGResult:
    """Container for Integrated Gradients attributions (token + word level)."""

    text: str
    tokens: list[str]
    token_attributions: list[float]
    words: list[str]
    word_attributions: list[float]
    pred_label: int
    target_label: int


def predict_label(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizerBase,
    text: str,
    device: str,
    max_length: int = 128,
) -> int:
    """Predict the most likely label index for a single text (argmax over logits)."""
    model.eval()
    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        max_length=max_length,
    )
    # Move tokenizer outputs onto the same device as the model (cpu/cuda)
    inputs = {k: v.to(device) for k, v in inputs.items()}
    with torch.no_grad():
        logits = model(**inputs).logits
    return int(torch.argmax(logits, dim=-1).item())


def _word_spans(text: str) -> list[tuple[int, int, str]]:
    """Return (start, end, word) spans for whitespace-delimited chunks in the raw text."""
    return [(m.start(), m.end(), m.group(0)) for m in re.finditer(r"\S+", text)]


def _map_token_to_word(word_spans: list[tuple[int, int, str]], start: int, end: int) -> int | None:
    """Map a token character span to a word index by overlap with word spans."""
    for idx, (w_start, w_end, _) in enumerate(word_spans):
        if start < w_end and end > w_start:  # overlap check for [start, end)
            return idx
    return None


def compute_integrated_gradients(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizerBase,
    text: str,
    *,
    target_label: int | None = None,
    n_steps: int = 50,
    device: str = "cpu",
    max_length: int = 128,
) -> IGResult:
    """
    Compute IG attributions for a single text.

    IG is computed w.r.t. the input embedding layer, then reduced to a scalar per token
    by summing over the embedding dimension. Token attributions are then aggregated to
    word attributions using tokenizer offset mappings.
    """
    model.eval()
    model.to(device)

    # Keep offsets + special token mask so we can map subword tokens back to raw words.
    enc = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        max_length=max_length,
        return_offsets_mapping=True,
        return_special_tokens_mask=True,
    )
    input_ids = enc["input_ids"].to(device)
    attention_mask = enc["attention_mask"].to(device)

    # If no explicit target was provided, explain the model's own prediction.
    pred_label = predict_label(model, tokenizer, text, device=device, max_length=max_length)
    if target_label is None:
        target_label = pred_label

    # Build a baseline sequence (same shape as input_ids).
    # We prefer pad_token_id, then eos/unk, then fall back to 0.
    baseline_id = tokenizer.pad_token_id
    if baseline_id is None:
        baseline_id = tokenizer.eos_token_id
    if baseline_id is None:
        baseline_id = tokenizer.unk_token_id
    if baseline_id is None:
        baseline_id = 0
    baselines = torch.full_like(input_ids, fill_value=baseline_id)

    # Captum calls this forward function with interpolated inputs.
    # It must return logits so Captum can take the target index.
    def forward_func(ids: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        return model(input_ids=ids, attention_mask=mask).logits

    # Attribute w.r.t. the embedding layer to get token-level attributions.
    lig = LayerIntegratedGradients(forward_func, model.get_input_embeddings())
    attributions = lig.attribute(
        inputs=input_ids,
        baselines=baselines,
        additional_forward_args=(attention_mask,),
        target=target_label,
        n_steps=n_steps,
    )

    # attributions: [batch, seq, hidden] -> sum hidden -> [seq]
    token_scores = attributions.sum(dim=-1).squeeze(0).detach().cpu().numpy()

    # Token strings and offset spans align with token_scores length.
    tokens = tokenizer.convert_ids_to_tokens(input_ids.squeeze(0).tolist())
    offsets = enc.get("offset_mapping")
    special_mask = enc.get("special_tokens_mask")

    # Prepare word spans from raw text and aggregate subwords into words.
    word_spans = _word_spans(text)
    words = [span[2] for span in word_spans]
    word_scores = [0.0 for _ in words]

    if offsets is not None:
        offsets_list = offsets.squeeze(0).tolist()
        special_list = special_mask.squeeze(0).tolist() if special_mask is not None else [0] * len(tokens)

        for idx, (offset, score) in enumerate(zip(offsets_list, token_scores)):
            if special_list[idx] == 1:
                continue  # ignore [CLS]/[SEP]/etc.

            start, end = offset
            if start == end:
                continue  # ignore tokens with empty spans

            word_idx = _map_token_to_word(word_spans, start, end)
            if word_idx is None:
                continue

            # Sum subword attributions into the owning word.
            word_scores[word_idx] += float(score)

    return IGResult(
        text=text,
        tokens=tokens,
        token_attributions=[float(v) for v in token_scores.tolist()],
        words=words,
        word_attributions=word_scores,
        pred_label=pred_label,
        target_label=int(target_label),
    )
