"""
Attention-based token attribution utilities.

This module computes attention scores (CLS -> tokens) and aggregates them to
word-level scores using tokenizer offsets.

Commands:
  - Run via pipeline: `python -m aet.cli --config configs/base.yaml --stage attention`
  - This pipeline writes CSV/JSON summaries under `reports/`.
"""

from __future__ import annotations

from dataclasses import dataclass
import re

import torch
from transformers import PreTrainedModel, PreTrainedTokenizerBase


@dataclass
class AttentionResult:
    """Container for attention scores at token and word level"""

    text: str
    tokens: list[str]
    token_scores: list[float]
    words: list[str]
    word_scores: list[float]


def _word_spans(text: str) -> list[tuple[int, int, str]]:
    """Return (start, end, word) spans for non-whitespace tokens."""
    return [(m.start(), m.end(), m.group(0)) for m in re.finditer(r"\S+", text)]


def _map_token_to_word(word_spans: list[tuple[int, int, str]], start: int, end: int) -> int | None:
    """Map a token character span to a word index using overlap"""
    for idx, (w_start, w_end, _) in enumerate(word_spans):
        if start < w_end and end > w_start:
            return idx
    return None


def compute_attention_scores(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizerBase,
    text: str,
    *,
    device: str = "cpu",
    max_length: int = 128,
    layer: str = "last",
) -> AttentionResult:
    """
    Compute attention-based scores for a single text, aggregating token scores to word scores

    Args:
        model: Hugging Face classifier with attention outputs.
        tokenizer: Matching tokenizer.
        text: Input text.
        device: Torch device string.
        max_length: Max token length (truncates longer texts).
        layer: "last" or "first" layer attention to use.
    """
    model.eval()
    model.to(device)

    # Tokenize and keep offsets so we can map subword tokens back to words.
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

    with torch.no_grad():
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_attentions=True,
            return_dict=True,
        )

    attentions = outputs.attentions
    if attentions is None:
        raise RuntimeError("Model did not return attention weights.")

    if layer == "last":
        attn = attentions[-1]
    elif layer == "first":
        attn = attentions[0]
    else:
        raise ValueError(f"Unknown layer selection: {layer}")

    # attn: [batch, heads, seq, seq]
    # Average over heads and take CLS -> token attention as token scores.
    attn_mean = attn.mean(dim=1)  # [batch, seq, seq]
    cls_attention = attn_mean[:, 0, :]  # CLS -> tokens
    token_scores = cls_attention.squeeze(0).detach().cpu().numpy()
    # Map token scores to words using offsets
    tokens = tokenizer.convert_ids_to_tokens(input_ids.squeeze(0).tolist())
    offsets = enc.get("offset_mapping")
    special_mask = enc.get("special_tokens_mask")

    word_spans = _word_spans(text)
    words = [span[2] for span in word_spans]
    word_scores = [0.0 for _ in words]
    # Aggregate token scores to word scores
    if offsets is not None:
        offsets_list = offsets.squeeze(0).tolist()
        special_list = special_mask.squeeze(0).tolist() if special_mask is not None else [0] * len(tokens)
        for idx, (offset, score) in enumerate(zip(offsets_list, token_scores)):
            if special_list[idx] == 1:
                continue
            start, end = offset
            if start == end:
                continue
            word_idx = _map_token_to_word(word_spans, start, end)
            if word_idx is None:
                continue
            # Aggregate token scores to word scores by overlap.
            word_scores[word_idx] += float(score)

    return AttentionResult(
        text=text,
        tokens=tokens,
        token_scores=[float(v) for v in token_scores.tolist()],
        words=words,
        word_scores=word_scores,
    )
