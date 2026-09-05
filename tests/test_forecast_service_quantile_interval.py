"""Tests for ForecastService._predict_target's use of per-match prediction
intervals (US#184) -- prefers model.predict_interval() (genuinely
per-match, heteroscedastic) over the existing metadata-driven
residual_prediction_interval() (one fixed width for every match) whenever
the loaded model supports it.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.forecast.forecast_service import ForecastService
from src.logic.target_registry import get_target_definition


class _FakeIntervalModel:
    """Minimal stand-in: predict() + predict_interval(), no training needed."""

    def predict(self, X):
        return np.array([2.5] * len(X))

    def predict_interval(self, X):
        # Deliberately per-row-varying, unlike a fixed residual band.
        lower = np.array([1.0, 0.5])[: len(X)]
        upper = np.array([4.0, 6.0])[: len(X)]
        return lower, upper


class _FakePlainModel:
    """No predict_interval() at all -- must fall back to the existing
    metadata-driven fixed-width mechanism, unchanged."""

    def predict(self, X):
        return np.array([2.5] * len(X))


def _service() -> ForecastService:
    return ForecastService.__new__(ForecastService)


def test_predict_target_uses_model_predict_interval_when_available():
    service = _service()
    definition = get_target_definition("total_corners")
    feature_row = pd.DataFrame({"a": [1.0]})

    result = service._predict_target(definition, _FakeIntervalModel(), {}, feature_row)

    assert result["prediction_interval"]["lower"] == pytest.approx(1.0)
    assert result["prediction_interval"]["upper"] == pytest.approx(4.0)
    assert result["prediction_interval"]["method"] == "quantile_regression"


def test_predict_target_falls_back_to_fixed_width_interval_without_predict_interval():
    service = _service()
    definition = get_target_definition("total_corners")
    feature_row = pd.DataFrame({"a": [1.0]})
    metadata = {"prediction_interval": {"lower_residual": -1.5, "upper_residual": 1.5, "coverage": 0.8}}

    result = service._predict_target(definition, _FakePlainModel(), metadata, feature_row)

    assert result["prediction_interval"]["method"] == "validation_residual_quantile"
    assert result["prediction_interval"]["lower"] == pytest.approx(1.0)
    assert result["prediction_interval"]["upper"] == pytest.approx(4.0)


def test_predict_target_no_interval_at_all_when_neither_available():
    service = _service()
    definition = get_target_definition("total_corners")
    feature_row = pd.DataFrame({"a": [1.0]})

    result = service._predict_target(definition, _FakePlainModel(), {}, feature_row)

    assert "prediction_interval" not in result
