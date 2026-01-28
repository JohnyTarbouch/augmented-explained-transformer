from __future__ import annotations

from typing import Sequence

import numpy as np


def summarize_with_ci(
    values: Sequence[float],
    *,
    num_bootstrap: int = 1000,
    confidence: float = 0.95,
    seed: int | None = None,
) -> dict[str, float]:
    if num_bootstrap < 0:
        raise ValueError("num_bootstrap must be non-negative")
    if not 0.0 < confidence < 1.0:
        raise ValueError("confidence must be between 0 and 1")

    arr = np.asarray(values, dtype=float)
    n = int(arr.size)
    if n == 0:
        return {
            "mean": 0.0,
            "std": 0.0,
            "ci_low": 0.0,
            "ci_high": 0.0,
            "n": 0,
            "confidence": float(confidence),
            "num_bootstrap": int(num_bootstrap),
        }

    mean = float(np.mean(arr))
    std = float(np.std(arr))

    if n < 2 or num_bootstrap == 0:
        return {
            "mean": mean,
            "std": std,
            "ci_low": mean,
            "ci_high": mean,
            "n": n,
            "confidence": float(confidence),
            "num_bootstrap": int(num_bootstrap),
        }

    rng = np.random.default_rng(seed)
    samples = rng.choice(arr, size=(num_bootstrap, n), replace=True)
    means = samples.mean(axis=1)
    alpha = (1.0 - confidence) / 2.0
    low, high = np.quantile(means, [alpha, 1.0 - alpha])

    return {
        "mean": mean,
        "std": std,
        "ci_low": float(low),
        "ci_high": float(high),
        "n": n,
        "confidence": float(confidence),
        "num_bootstrap": int(num_bootstrap),
    }


def ci_metadata(stats: dict[str, float]) -> dict[str, float]:
    return {
        "low": float(stats["ci_low"]),
        "high": float(stats["ci_high"]),
        "confidence": float(stats["confidence"]),
        "n": int(stats["n"]),
        "num_bootstrap": int(stats["num_bootstrap"]),
    }
