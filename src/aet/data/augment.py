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
    if source.isupper():
        return target.upper()
    if source[0].isupper():
        return target.capitalize()
    return target


def get_wordnet_synonyms(word: str) -> set[str]:
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


def augment_text(text: str, replace_prob: float = 0.1, seed: int | None = None) -> str:
    if replace_prob <= 0:
        return text
    _ensure_wordnet()
    rng = random.Random(seed)
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
