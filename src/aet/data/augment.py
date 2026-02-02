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

# NLTK is optional: only required for the WordNet augmentation method.
try:
    from nltk.corpus import wordnet as wn
except Exception:  # pragma: no cover - optional dependency
    wn = None


# Simple stopword list to avoid replacing extremely common function words.
# (Replacing these often hurts grammaticality and doesn't add useful diversity.)
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
    # If import failed, we can't do WordNet-based augmentation.
    if wn is None:
        raise RuntimeError(
            "NLTK is required for synonym replacement. Install with: pip install nltk "
            "and download data: python -m nltk.downloader wordnet"
        )
    # Even if NLTK is installed, the WordNet corpus data may be missing.
    try:
        wn.synsets("good")
    except LookupError as exc:
        raise RuntimeError(
            "NLTK wordnet data not found. Run: python -m nltk.downloader wordnet"
        ) from exc


def _match_case(source: str, target: str) -> str:
    """Match the casing style of source onto target."""
    # Preserve casing so replacements don't look obviously "synthetic".
    if source.isupper():
        return target.upper()
    if source[0].isupper():
        return target.capitalize()
    return target


def get_wordnet_synonyms(word: str) -> set[str]:
    """Return a set of single-token alphabetic synonyms for a word."""
    _ensure_wordnet()
    synonyms: set[str] = set()

    # WordNet provides synsets; each synset has lemma names (candidate synonyms).
    for synset in wn.synsets(word):
        for lemma in synset.lemmas():
            candidate = lemma.name().replace("_", " ")

            # Keep only single-token replacements (no multi-word phrases).
            if " " in candidate:
                continue

            # Filter non-alphabetic forms (numbers, punctuation, etc.).
            if not candidate.isalpha():
                continue

            # Avoid returning the original word (case-insensitive).
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
        # If replace_prob is zero/negative, return input unchanged.
        if replace_prob <= 0:
            return text

        _ensure_wordnet()
        rng = random.Random(seed)

        # Keep punctuation by splitting into word/non-word spans.
        # Example: "Hello, world!" -> ["Hello", ", ", "world", "!"]
        tokens = re.findall(r"\w+|\W+", text)
        augmented: list[str] = []

        for token in tokens:
            # Only attempt synonym replacement on alphabetic word tokens.
            if not token.isalpha():
                augmented.append(token)
                continue

            # Skip stopwords and apply replacement probability gate.
            if token.lower() in _STOPWORDS or rng.random() > replace_prob:
                augmented.append(token)
                continue

            # Collect candidates; sorted for deterministic choice under the same RNG seed.
            synonyms = sorted(get_wordnet_synonyms(token))
            if not synonyms:
                augmented.append(token)
                continue

            # Sample one synonym and preserve casing style.
            replacement = rng.choice(synonyms)
            augmented.append(_match_case(token, replacement))

        # Join without inserting extra spaces (we preserved original punctuation spans).
        return "".join(augmented)

    if method == "backtranslation":
        # Import locally so users can use WordNet augmentation without MarianMT deps.
        from aet.data.backtranslation import back_translate_text
        from aet.utils.device import resolve_device

        # Copy config dict so we don't mutate caller-provided objects.
        cfg = dict(backtranslation_cfg or {})

        # If a device string is provided (e.g., "auto"), normalize it to an actual device.
        if "device" in cfg and cfg["device"] is not None:
            cfg["device"] = resolve_device(str(cfg["device"]))

        # Back-translation can fail; helper returns original text on error (with filtered=True).
        result = back_translate_text(text, **cfg)
        return result.text

    # Explicit failure for unknown methods (helps config debugging).
    raise ValueError(f"Unknown augmentation method: {method}")
