"""
Attention-based token attribution utilities.

This module computes attention scores (CLS -> tokens) and aggregates them to
word-level scores using tokenizer offsets.
"""

from __future__ import annotations

from dataclasses import dataclass
import re

import torch
from transformers import PreTrainedModel, PreTrainedTokenizerBase


@dataclass
class AttentionResult:
    """Container for attention scores at token and word level."""
    text: str
    tokens: list[str]
    token_scores: list[float]
    words: list[str]
    word_scores: list[float]


def _word_spans(text: str) -> list[tuple[int, int, str]]:
    """Return (start, end, word) spans for non-whitespace chunks in the raw text."""
    return [(m.start(), m.end(), m.group(0)) for m in re.finditer(r"\S+", text)]


def _map_token_to_word(word_spans: list[tuple[int, int, str]], start: int, end: int) -> int | None:
    """
    Map a token character span to a word index by overlap.

    We treat a token as belonging to the first word span it overlaps.
    """
    for idx, (w_start, w_end, _) in enumerate(word_spans):
        # overlap check for [start, end) spans
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
    Compute attention-based scores for a single text, aggregating token scores to word scores.

    Uses CLS -> token attention (averaged over heads) as token-level scores.
    Then sums token scores into word scores using tokenizer offset mappings.
    """
    model.eval()
    model.to(device)

    # Keep offsets so we can map subword tokens back to word spans in `text`.
    enc = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        max_length=max_length,
        return_offsets_mapping=True, # char spans per token
        return_special_tokens_mask=True, # marks [CLS], [SEP], etc.
    )
    input_ids = enc["input_ids"].to(device)
    attention_mask = enc["attention_mask"].to(device)

    with torch.no_grad():
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_attentions=True,  # required to get outputs.attentions
            return_dict=True,
        )

    attentions = outputs.attentions
    if attentions is None:
        raise RuntimeError("Model did not return attention weights.")

    # Choose which layer's attention to use.
    if layer == "last":
        attn = attentions[-1]
    elif layer == "first":
        attn = attentions[0]
    else:
        raise ValueError(f"Unknown layer selection: {layer}")

    # attn: [batch, heads, seq, seq] -> mean over heads -> [batch, seq, seq]
    # Average over heads and take CLS -> token attention as token scores.
    attn_mean = attn.mean(dim=1)

    # Take CLS (position 0) attention to all tokens as a per-token score.
    cls_attention = attn_mean[:, 0, :]  # [batch, seq]
    token_scores = cls_attention.squeeze(0).detach().cpu().numpy()

    # Token strings and offset spans align with token_scores length.
    tokens = tokenizer.convert_ids_to_tokens(input_ids.squeeze(0).tolist())
    offsets = enc.get("offset_mapping")
    special_mask = enc.get("special_tokens_mask")

    # Build word spans from the raw text and initialize aggregation buffer.
    word_spans = _word_spans(text)
    words = [span[2] for span in word_spans]
    word_scores = [0.0 for _ in words]

    # Aggregate token scores into word scores using overlap of char spans.
    if offsets is not None:
        offsets_list = offsets.squeeze(0).tolist()
        special_list = special_mask.squeeze(0).tolist() if special_mask is not None else [0] * len(tokens)

        for idx, (offset, score) in enumerate(zip(offsets_list, token_scores)):
            if special_list[idx] == 1:
                continue  # skip special tokens

            start, end = offset
            if start == end:
                continue  # skip tokens with empty spans

            word_idx = _map_token_to_word(word_spans, start, end)
            if word_idx is None:
                continue

            # Sum subword contributions into the owning word.
            word_scores[word_idx] += float(score)

    return AttentionResult(
        text=text,
        tokens=tokens,
        token_scores=[float(v) for v in token_scores.tolist()],
        words=words,
        word_scores=word_scores,
    )
