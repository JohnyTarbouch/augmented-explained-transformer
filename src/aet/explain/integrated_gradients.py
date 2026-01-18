from __future__ import annotations

from dataclasses import dataclass
import re

import torch
from captum.attr import LayerIntegratedGradients
from transformers import PreTrainedModel, PreTrainedTokenizerBase


@dataclass
class IGResult:
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
    model.eval()
    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        max_length=max_length,
    )
    inputs = {k: v.to(device) for k, v in inputs.items()}
    with torch.no_grad():
        logits = model(**inputs).logits
    return int(torch.argmax(logits, dim=-1).item())


def _word_spans(text: str) -> list[tuple[int, int, str]]:
    return [(m.start(), m.end(), m.group(0)) for m in re.finditer(r"\S+", text)]


def _map_token_to_word(word_spans: list[tuple[int, int, str]], start: int, end: int) -> int | None:
    for idx, (w_start, w_end, _) in enumerate(word_spans):
        if start < w_end and end > w_start:
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
    model.eval()
    model.to(device)

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

    pred_label = predict_label(model, tokenizer, text, device=device, max_length=max_length)
    if target_label is None:
        target_label = pred_label

    baseline_id = tokenizer.pad_token_id
    if baseline_id is None:
        baseline_id = tokenizer.eos_token_id
    if baseline_id is None:
        baseline_id = tokenizer.unk_token_id
    if baseline_id is None:
        baseline_id = 0
    baselines = torch.full_like(input_ids, fill_value=baseline_id)

    def forward_func(ids: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        return model(input_ids=ids, attention_mask=mask).logits

    lig = LayerIntegratedGradients(forward_func, model.get_input_embeddings())
    attributions = lig.attribute(
        inputs=input_ids,
        baselines=baselines,
        additional_forward_args=(attention_mask,),
        target=target_label,
        n_steps=n_steps,
    )
    token_scores = attributions.sum(dim=-1).squeeze(0).detach().cpu().numpy()

    tokens = tokenizer.convert_ids_to_tokens(input_ids.squeeze(0).tolist())
    offsets = enc.get("offset_mapping")
    special_mask = enc.get("special_tokens_mask")

    word_spans = _word_spans(text)
    words = [span[2] for span in word_spans]
    word_scores = [0.0 for _ in words]

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
