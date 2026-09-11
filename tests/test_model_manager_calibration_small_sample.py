"""Tests for BUG-066: isotonic calibration on a small validation set produces
wide flat plateaus that collapse genuinely different raw probabilities into
byte-identical calibrated output.

Root cause reproduced directly (see docs/bugs.md BUG-066): E0 result_3way's
promoted calibrator mapped every raw away-probability in a real 0.474-0.495
range to one constant 0.426695, confirmed by feeding the exact recorded
mu_home/mu_away pairs from 6 real distinct matches through the live sidecar.

Fix: ModelManager._make_calibrator falls back to sigmoid (Platt) scaling
below _MIN_ISOTONIC_SAMPLES, which cannot produce flat plateaus of this kind
(a logistic curve is strictly monotonic, never locally constant over a real
input range short of numerical saturation).
"""

from __future__ import annotations

from pathlib import Path
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.models.model_manager import ModelManager, _MIN_ISOTONIC_SAMPLES, _make_calibrator
from sklearn.isotonic import IsotonicRegression
from sklearn.calibration import _SigmoidCalibration


def test_make_calibrator_picks_isotonic_above_threshold() -> None:
    assert isinstance(_make_calibrator(_MIN_ISOTONIC_SAMPLES), IsotonicRegression)


def test_make_calibrator_picks_sigmoid_below_threshold() -> None:
    assert isinstance(_make_calibrator(_MIN_ISOTONIC_SAMPLES - 1), _SigmoidCalibration)


def test_small_sample_isotonic_reproduces_bug066_plateau_collapse() -> None:
    """Demonstrates the actual failure mode this fix avoids: with few, narrow-
    range validation points, IsotonicRegression collapses a real spread of
    raw probabilities into one identical output -- exactly what shipped live."""
    rng = np.random.RandomState(0)
    # Mirrors the live shape: ~15 narrowly-clustered validation points, most
    # outcomes negative for this class (away win in a compressed-goals regime).
    x = np.sort(rng.uniform(0.45, 0.55, 15))
    y = np.zeros(15)
    y[-2:] = 1.0  # only the two highest-x points are positive

    isotonic = IsotonicRegression(out_of_bounds="clip")
    isotonic.fit(x, y)

    probe = np.array([0.474, 0.48, 0.485, 0.49, 0.495])
    collapsed = isotonic.predict(probe)
    assert len(set(collapsed.round(6))) == 1, "expected the known plateau-collapse failure mode"


def test_sigmoid_fallback_does_not_collapse_the_same_inputs() -> None:
    rng = np.random.RandomState(0)
    x = np.sort(rng.uniform(0.45, 0.55, 15))
    y = np.zeros(15)
    y[-2:] = 1.0

    sigmoid = _make_calibrator(len(x))
    sigmoid.fit(x, y)

    probe = np.array([0.474, 0.48, 0.485, 0.49, 0.495])
    differentiated = sigmoid.predict(probe)
    assert len(set(differentiated.round(6))) > 1, "sigmoid calibration should preserve real differentiation"


def _train_multiclass_model() -> tuple:
    from src.models.base_model import XGBoostModel

    rng = np.random.RandomState(42)
    n = 90
    f1 = rng.normal(0.0, 1.0, n)
    y = pd.Series(np.where(f1 > 0.5, "home", np.where(f1 < -0.5, "away", "draw")))
    X = pd.DataFrame({"f1": f1, "f2": rng.normal(0.0, 1.0, n)})

    model = XGBoostModel(objective="multi:softprob", eval_metric="mlogloss", num_class=3)
    model.train(X, y, eval_set=[(X, y)])
    return model, X, y


def test_fit_and_save_calibrator_uses_sigmoid_for_small_multiclass_val_set(tmp_path: Path) -> None:
    """n=90 (< _MIN_ISOTONIC_SAMPLES) — every fitted per-class calibrator
    should be the sigmoid fallback, not IsotonicRegression."""
    model, X_val, y_val = _train_multiclass_model()
    model_path = tmp_path / "result_3way_test.joblib"

    result = ModelManager._fit_and_save_calibrator(model, X_val, y_val, model_path)
    assert result is not None

    import joblib
    sidecar_path = model_path.with_suffix(model_path.suffix + ".calibration.pkl")
    sidecar = joblib.load(str(sidecar_path))
    assert all(isinstance(cal, _SigmoidCalibration) for cal in sidecar["calibrator"])
