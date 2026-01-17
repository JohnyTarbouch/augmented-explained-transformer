from __future__ import annotations

import numpy as np


def top_k_overlap(a: np.ndarray, b: np.ndarray, k: int) -> float:
    if k <= 0:
        raise ValueError("k must be positive")
    idx_a = np.argsort(-np.abs(a))[:k]
    idx_b = np.argsort(-np.abs(b))[:k]
    return len(set(idx_a) & set(idx_b)) / float(k)


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    denom = np.linalg.norm(a) * np.linalg.norm(b)
    if denom == 0:
        return 0.0
    return float(np.dot(a, b) / denom)
