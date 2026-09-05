"""Tests for QuantileIntervalModel (US#184).

Per-match, heteroscedastic prediction intervals for regression targets --
replaces the existing `residual_prediction_interval` mechanism's fixed,
training-time-computed global residual width (every match got the same
band regardless of whether the model was actually more or less certain
about that specific fixture) with genuine per-match bounds from XGBoost's
native quantile regression (`reg:quantileerror`), which vary with input
features the same way the point estimate does.
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.models.quantile_interval_model import QuantileIntervalModel


def _heteroscedastic_data(n: int = 400, seed: int = 0) -> tuple[pd.DataFrame, pd.Series]:
    """y's noise scale depends on 'volatility' -- a real model should give
    tighter intervals for low-volatility rows, wider for high-volatility."""
    rng = np.random.default_rng(seed)
    strength = rng.normal(0, 1, size=n)
    volatility = rng.uniform(0.2, 3.0, size=n)
    y = np.clip(1.5 + strength * 0.5 + rng.normal(0, volatility, size=n), 0, None)
    X = pd.DataFrame({"strength": strength, "volatility": volatility})
    return X, pd.Series(y)


class TestQuantileIntervalModelTrain:
    def test_predict_returns_point_estimate(self):
        model = QuantileIntervalModel()
        X, y = _heteroscedastic_data()
        model.train(X, y)
        preds = model.predict(X)
        assert preds.shape == (len(X),)
        assert np.isfinite(preds).all()

    def test_predict_proba_raises_not_a_classifier(self):
        model = QuantileIntervalModel()
        X, y = _heteroscedastic_data(50)
        model.train(X, y)
        with pytest.raises(TypeError):
            model.predict_proba(X)


class TestQuantileIntervalModelInterval:
    def test_predict_interval_shape_and_ordering(self):
        model = QuantileIntervalModel(coverage=0.8)
        X, y = _heteroscedastic_data()
        model.train(X, y)
        lower, upper = model.predict_interval(X)
        assert lower.shape == (len(X),)
        assert upper.shape == (len(X),)
        assert (lower <= upper).all(), "quantile crossing must be corrected"
        assert (lower >= 0).all(), "goal/corner counts can't be negative"

    def test_interval_width_is_genuinely_heteroscedastic(self):
        """The actual point of this model: interval width must vary
        meaningfully with input features, not be one fixed global number
        the way residual_prediction_interval's config was."""
        model = QuantileIntervalModel(coverage=0.8)
        X, y = _heteroscedastic_data(600, seed=3)
        model.train(X, y)
        lower, upper = model.predict_interval(X)
        widths = upper - lower
        # Low-volatility rows should get a materially tighter interval than
        # high-volatility rows -- not checking an exact number, checking
        # the model actually discriminates.
        low_vol_widths = widths[X["volatility"] < 0.5]
        high_vol_widths = widths[X["volatility"] > 2.5]
        assert low_vol_widths.mean() < high_vol_widths.mean(), (
            f"expected tighter intervals for low-volatility rows: "
            f"low={low_vol_widths.mean():.3f} high={high_vol_widths.mean():.3f}"
        )
        assert widths.std() > 0.01, "widths should vary across matches, not be constant"

    def test_coverage_is_approximately_honored(self):
        """A held-out check: at coverage=0.8, roughly 80% of true values
        should actually fall inside the predicted interval."""
        model = QuantileIntervalModel(coverage=0.8)
        X_train, y_train = _heteroscedastic_data(600, seed=5)
        X_test, y_test = _heteroscedastic_data(300, seed=6)
        model.train(X_train, y_train)
        lower, upper = model.predict_interval(X_test)
        inside = (y_test.to_numpy() >= lower) & (y_test.to_numpy() <= upper)
        coverage_rate = inside.mean()
        assert 0.65 <= coverage_rate <= 0.95, f"coverage {coverage_rate:.2f} far from target 0.8"


class TestQuantileIntervalModelPersistence:
    def test_save_load_roundtrip_predictions_match(self):
        model = QuantileIntervalModel(coverage=0.8)
        X, y = _heteroscedastic_data(200)
        model.train(X, y)
        point_before = model.predict(X)
        lower_before, upper_before = model.predict_interval(X)

        with tempfile.TemporaryDirectory() as tmp:
            path = str(Path(tmp) / "model.joblib")
            model.save(path)
            loaded = QuantileIntervalModel.load(path)
            point_after = loaded.predict(X)
            lower_after, upper_after = loaded.predict_interval(X)

        assert np.allclose(point_before, point_after)
        assert np.allclose(lower_before, lower_after)
        assert np.allclose(upper_before, upper_after)
        assert loaded.coverage == 0.8
