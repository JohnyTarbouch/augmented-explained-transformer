"""Text augmentation helpers.

Supported methods:
  - wordnet: synonym replacement using NLTK WordNet
  - backtranslation: round-trip translation using MarianMT (optional)

Commands:
  - Download SST-2: `python scripts/download_sst2.py`
  - Create augmented CSVs: `python scripts/augment_sst2.py --split train --combined --augment-fraction 0.1`
  - Back-translation augmentation:
    `python scripts/augment_sst2.py --split train --combined --augment-fraction 0.1 --method backtranslation`
  - Enable WordNet data: `python -m nltk.downloader wordnet`
"""

from __future__ import annotations

import random
import re

try:
    from nltk.corpus import wordnet as wn
except Exception:  # pragma: no cover - optional dependency
    wn = None


_STOPWORDS = {
    "the",
    "a",
    "an",
    "and",
    "or",
    "but",
    "if",
    "while",
    "of",
    "in",
    "on",
    "at",
    "to",
    "for",
    "with",
    "as",
    "by",
    "is",
    "are",
    "was",
    "were",
    "be",
    "been",
    "being",
}


def _ensure_wordnet() -> None:
    """Ensure NLTK WordNet is installed and available."""
    if wn is None:
        raise RuntimeError(
            "NLTK is required for synonym replacement. Install with: pip install nltk "
            "and download data: python -m nltk.downloader wordnet"
        )
    try:
        wn.synsets("good")
    except LookupError as exc:
        raise RuntimeError(
            "NLTK wordnet data not found. Run: python -m nltk.downloader wordnet"
        ) from exc


def _match_case(source: str, target: str) -> str:
    """Match the casing style of source onto target."""
    if source.isupper():
        return target.upper()
    if source[0].isupper():
        return target.capitalize()
    return target


def get_wordnet_synonyms(word: str) -> set[str]:
    """Return a set of single-token alphabetic synonyms for a word."""
    _ensure_wordnet()
    synonyms: set[str] = set()
    for synset in wn.synsets(word):
        for lemma in synset.lemmas():
            candidate = lemma.name().replace("_", " ")
            if " " in candidate:
                continue
            if not candidate.isalpha():
                continue
            if candidate.lower() != word.lower():
                synonyms.add(candidate)
    return synonyms


def augment_text(
    text: str,
    replace_prob: float = 0.1,
    seed: int | None = None,
    *,
    method: str = "wordnet",
    backtranslation_cfg: dict | None = None,
) -> str:
    """Augment a single text using the selected method.

    Args:
        text: Input string to augment.
        replace_prob: Probability of replacing each token (wordnet only).
        seed: RNG seed for deterministic replacements.
        method: "wordnet" or "backtranslation".
        backtranslation_cfg: Optional kwargs for back-translation.
    """
    if method == "wordnet":
        if replace_prob <= 0:
            return text
        _ensure_wordnet()
        rng = random.Random(seed)
        # Keep punctuation by splitting into word/non-word spans.
        tokens = re.findall(r"\w+|\W+", text)
        augmented: list[str] = []

        for token in tokens:
            if not token.isalpha():
                augmented.append(token)
                continue
            if token.lower() in _STOPWORDS or rng.random() > replace_prob:
                augmented.append(token)
                continue

            synonyms = sorted(get_wordnet_synonyms(token))
            if not synonyms:
                augmented.append(token)
                continue

            replacement = rng.choice(synonyms)
            augmented.append(_match_case(token, replacement))

        return "".join(augmented)

    if method == "backtranslation":
        from aet.data.backtranslation import back_translate_text
        from aet.utils.device import resolve_device

        cfg = dict(backtranslation_cfg or {})
        if "device" in cfg and cfg["device"] is not None:
            cfg["device"] = resolve_device(str(cfg["device"]))
        # Back-translation can fail; the helper returns the original text on error.
        result = back_translate_text(text, **cfg)
        return result.text

    raise ValueError(f"Unknown augmentation method: {method}")
