from __future__ import annotations

from dataclasses import dataclass
import math
import os
from typing import Iterable

os.environ.setdefault("TRANSFORMERS_NO_TF", "1")

import torch
from transformers import MarianMTModel, MarianTokenizer

from aet.utils.logging import get_logger


@dataclass(frozen=True)
class BackTranslationResult:
    text: str
    length_ratio: float
    filtered: bool
    reason: str | None = None


_TRANSLATOR_CACHE: dict[tuple[str, str, str, int, int], "BackTranslator"] = {}


def _normalize_whitespace(text: str) -> str:
    return " ".join(text.split()).strip()


def _length_ratio(source: str, target: str) -> float:
    src_len = len(source.split())
    tgt_len = len(target.split())
    if src_len == 0:
        return 0.0
    return tgt_len / float(src_len)


def _passes_length_filter(ratio: float, min_ratio: float, max_ratio: float) -> bool:
    if min_ratio <= 0 or max_ratio <= 0:
        return True
    return min_ratio <= ratio <= max_ratio


class BackTranslator:
    def __init__(
        self,
        *,
        src_lang: str,
        pivot_lang: str,
        device: str,
        batch_size: int,
        max_length: int,
    ) -> None:
        self.src_lang = src_lang
        self.pivot_lang = pivot_lang
        self.device = device
        self.batch_size = batch_size
        self.max_length = max_length

        self.forward_model_name = f"Helsinki-NLP/opus-mt-{src_lang}-{pivot_lang}"
        self.backward_model_name = f"Helsinki-NLP/opus-mt-{pivot_lang}-{src_lang}"

        self.forward_tokenizer = MarianTokenizer.from_pretrained(self.forward_model_name)
        self.forward_model = MarianMTModel.from_pretrained(self.forward_model_name)
        self.backward_tokenizer = MarianTokenizer.from_pretrained(self.backward_model_name)
        self.backward_model = MarianMTModel.from_pretrained(self.backward_model_name)

        self.forward_model.to(device)
        self.backward_model.to(device)
        self.forward_model.eval()
        self.backward_model.eval()

    def _translate_batch(
        self,
        texts: list[str],
        *,
        tokenizer: MarianTokenizer,
        model: MarianMTModel,
    ) -> list[str]:
        if not texts:
            return []
        batch = tokenizer(
            texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self.max_length,
        )
        batch = {k: v.to(self.device) for k, v in batch.items()}
        with torch.inference_mode():
            generated = model.generate(**batch, max_length=self.max_length)
        return tokenizer.batch_decode(generated, skip_special_tokens=True)

    def back_translate(self, texts: list[str], *, progress_every: int | None = None) -> list[str]:
        outputs: list[str] = []
        total = len(texts)
        if total == 0:
            return outputs
        num_batches = int(math.ceil(total / float(self.batch_size)))
        for batch_idx, i in enumerate(range(0, total, self.batch_size)):
            if progress_every and (
                (batch_idx + 1) % progress_every == 0 or (batch_idx + 1) == num_batches
            ):
                print(f"Back-translation batch {batch_idx + 1}/{num_batches}", flush=True)
            batch = texts[i : i + self.batch_size]
            mid = self._translate_batch(batch, tokenizer=self.forward_tokenizer, model=self.forward_model)
            back = self._translate_batch(mid, tokenizer=self.backward_tokenizer, model=self.backward_model)
            outputs.extend(back)
        return [_normalize_whitespace(text) for text in outputs]


def get_back_translator(
    *,
    src_lang: str,
    pivot_lang: str,
    device: str,
    batch_size: int,
    max_length: int,
) -> BackTranslator:
    key = (src_lang, pivot_lang, device, batch_size, max_length)
    if key not in _TRANSLATOR_CACHE:
        _TRANSLATOR_CACHE[key] = BackTranslator(
            src_lang=src_lang,
            pivot_lang=pivot_lang,
            device=device,
            batch_size=batch_size,
            max_length=max_length,
        )
    return _TRANSLATOR_CACHE[key]


def back_translate_text(
    text: str,
    *,
    src_lang: str = "en",
    pivot_lang: str = "de",
    device: str = "cpu",
    batch_size: int = 8,
    max_length: int = 128,
    min_length_ratio: float = 0.7,
    max_length_ratio: float = 1.3,
) -> BackTranslationResult:
    logger = get_logger(__name__)
    cleaned = _normalize_whitespace(text)
    if not cleaned:
        return BackTranslationResult(text=cleaned, length_ratio=0.0, filtered=True, reason="empty")

    try:
        translator = get_back_translator(
            src_lang=src_lang,
            pivot_lang=pivot_lang,
            device=device,
            batch_size=batch_size,
            max_length=max_length,
        )
        translated = translator.back_translate([cleaned])[0]
    except Exception as exc:  # pragma: no cover - optional dependency/runtime failures
        logger.warning("Back-translation failed; returning original text. Error: %s", exc)
        return BackTranslationResult(text=cleaned, length_ratio=1.0, filtered=True, reason="error")

    ratio = _length_ratio(cleaned, translated)
    if not _passes_length_filter(ratio, min_length_ratio, max_length_ratio):
        return BackTranslationResult(
            text=cleaned,
            length_ratio=ratio,
            filtered=True,
            reason="length_ratio",
        )

    return BackTranslationResult(text=translated, length_ratio=ratio, filtered=False)


def back_translate_batch(
    texts: Iterable[str],
    *,
    src_lang: str = "en",
    pivot_lang: str = "de",
    device: str = "cpu",
    batch_size: int = 8,
    max_length: int = 128,
    progress_every: int | None = None,
) -> list[str]:
    translator = get_back_translator(
        src_lang=src_lang,
        pivot_lang=pivot_lang,
        device=device,
        batch_size=batch_size,
        max_length=max_length,
    )
    return translator.back_translate(
        [_normalize_whitespace(t) for t in texts],
        progress_every=progress_every,
    )
