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


def rank_values(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values)
    ranks = np.empty_like(order)
    ranks[order] = np.arange(len(values))
    return ranks


def kendall_tau(a: np.ndarray, b: np.ndarray) -> float:
    if len(a) != len(b):
        raise ValueError("Inputs must have the same length")
    n = len(a)
    if n < 2:
        return 0.0
    concordant = 0
    discordant = 0
    for i in range(n - 1):
        for j in range(i + 1, n):
            sign_a = np.sign(a[i] - a[j])
            sign_b = np.sign(b[i] - b[j])
            if sign_a == 0 or sign_b == 0:
                continue
            if sign_a == sign_b:
                concordant += 1
            else:
                discordant += 1
    denom = concordant + discordant
    if denom == 0:
        return 0.0
    return float((concordant - discordant) / denom)
