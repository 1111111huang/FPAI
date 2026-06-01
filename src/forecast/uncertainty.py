"""Uncertainty and count-distribution helpers for forecast payloads."""

from __future__ import annotations

import math
from typing import Iterable

import numpy as np


def normalized_entropy_uncertainty(probabilities: Iterable[float]) -> dict[str, float | str]:
    """Return normalized entropy uncertainty for a probability vector."""
    probs = np.asarray(list(probabilities), dtype=float)
    if probs.ndim != 1 or len(probs) < 2:
        raise ValueError("Entropy uncertainty requires at least two probabilities.")
    total = float(probs.sum())
    if total <= 0:
        raise ValueError("Probability vector must have positive mass.")
    probs = np.clip(probs / total, 0.0, 1.0)
    non_zero = probs[probs > 0]
    entropy = -float(np.sum(non_zero * np.log(non_zero)))
    score = entropy / math.log(len(probs))
    if score < 0.40:
        level = "low"
    elif score <= 0.75:
        level = "medium"
    else:
        level = "high"
    return {"method": "entropy", "score": round(float(score), 6), "level": level}


def residual_prediction_interval(
    expected: float,
    lower_residual: float,
    upper_residual: float,
    coverage: float = 0.8,
    minimum: float = 0.0,
) -> dict[str, float | str]:
    """Build a prediction interval from validation residual quantiles."""
    lower = max(minimum, float(expected) + float(lower_residual))
    upper = max(lower, float(expected) + float(upper_residual))
    return {
        "lower": round(lower, 6),
        "upper": round(upper, 6),
        "coverage": float(coverage),
        "method": "validation_residual_quantile",
    }


def poisson_count_distribution(expected: float) -> dict[str, float]:
    """Convert an expected count into 0/1/2/3_plus Poisson buckets."""
    lam = max(0.0, float(expected))
    probs = {
        "0": math.exp(-lam),
        "1": math.exp(-lam) * lam,
        "2": math.exp(-lam) * (lam**2) / 2.0,
    }
    probs["3_plus"] = max(0.0, 1.0 - sum(probs.values()))
    rounded = {key: round(float(value), 6) for key, value in probs.items()}
    rounded["3_plus"] = round(max(0.0, 1.0 - rounded["0"] - rounded["1"] - rounded["2"]), 6)
    return rounded
