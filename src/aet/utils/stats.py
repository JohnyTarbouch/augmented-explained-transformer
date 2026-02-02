'''
Utility functions for statistical summaries with confidence interval
'''
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
    '''
    Compute mean, std, and bootstrap confidence intervals for a sequence of values.
    Parameters:
        values: Sequence of numerical values.
        num_bootstrap: Number of bootstrap samples to use for CI estimation.
        confidence: Confidence level for the intervals (between 0 and 1).
        seed: Random seed for reproducibility.
    Returns:
        A dictionary with keys: mean, std, ci_low, ci_high, n, confidence, num_bootstrap.
    '''
    # Validate parameters
    if num_bootstrap < 0:
        raise ValueError("num_bootstrap must be non-negative")
    if not 0.0 < confidence < 1.0:
        raise ValueError("confidence must be between 0 and 1")
    # Convert input to numpy array
    arr = np.asarray(values, dtype=float)
    n = int(arr.size)
    # Handle edge cases
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

    # Compute mean and std
    mean = float(np.mean(arr))
    std = float(np.std(arr))
    # If not enough data for bootstrap, return mean and std only
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

    # Bootstrap sampling for confidence intervals
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
    # Extract confidence interval metadata from stats dictionary
    return {
        "low": float(stats["ci_low"]),
        "high": float(stats["ci_high"]),
        "confidence": float(stats["confidence"]),
        "n": int(stats["n"]),
        "num_bootstrap": int(stats["num_bootstrap"]),
    }
