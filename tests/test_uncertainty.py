from __future__ import annotations

from pathlib import Path
import sys

import pytest

sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.forecast.uncertainty import (
    normalized_entropy_uncertainty,
    poisson_count_distribution,
    residual_prediction_interval,
)


def test_normalized_entropy_uncertainty_levels() -> None:
    low = normalized_entropy_uncertainty([0.99, 0.01])
    high = normalized_entropy_uncertainty([0.5, 0.5])

    assert low["method"] == "entropy"
    assert low["level"] == "low"
    assert low["score"] < 0.4
    assert high["level"] == "high"
    assert high["score"] == pytest.approx(1.0)


def test_normalized_entropy_uncertainty_for_multiclass() -> None:
    uncertainty = normalized_entropy_uncertainty([1 / 3, 1 / 3, 1 / 3])

    assert uncertainty["score"] == pytest.approx(1.0)
    assert uncertainty["level"] == "high"


def test_residual_prediction_interval_clamps_lower_bound() -> None:
    interval = residual_prediction_interval(expected=0.3, lower_residual=-1.0, upper_residual=1.2)

    assert interval == {
        "lower": 0.0,
        "upper": 1.5,
        "coverage": 0.8,
        "method": "validation_residual_quantile",
    }


def test_poisson_count_distribution_is_stable_bucket_contract() -> None:
    distribution = poisson_count_distribution(1.5)

    assert set(distribution) == {"0", "1", "2", "3_plus"}
    assert sum(distribution.values()) == pytest.approx(1.0)
