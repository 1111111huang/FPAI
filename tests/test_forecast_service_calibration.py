"""Tests for ForecastService actually applying a model's own saved
calibrator sidecar at serving time (US#186).

Found live: `ModelManager._fit_and_save_calibrator` computes and saves a
`.calibration.pkl` sidecar for every classifier (logged before/after
log_loss to MLflow), but nothing in the serving path ever reads it back --
confirmed by grep, no consumer existed anywhere in src/forecast/. Every
classifier has been serving raw, uncalibrated probabilities the whole time
a better-calibrated version sat on disk unused.
"""

from __future__ import annotations

import sys
from pathlib import Path

import joblib
import numpy as np
import pytest
from sklearn.isotonic import IsotonicRegression

sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.forecast.forecast_service import ForecastService
from src.logic.target_registry import get_target_definition


def _fit_binary_calibrator() -> IsotonicRegression:
    # Deliberately miscalibrated raw scores (systematically too high) so
    # calibration visibly moves the result -- a monotone identity
    # calibrator wouldn't distinguish "applied" from "not applied".
    raw = np.linspace(0.0, 1.0, 20)
    true_rate = raw * 0.5  # true positive rate is half of the raw score
    cal = IsotonicRegression(out_of_bounds="clip")
    cal.fit(raw, true_rate)
    return cal


def test_apply_calibration_binary_shifts_probabilities():
    from src.forecast.forecast_service import _apply_calibration

    cal = _fit_binary_calibrator()
    sidecar = {"type": "binary", "calibrator": cal}
    raw_proba = np.array([[0.2, 0.8]])

    calibrated = _apply_calibration(raw_proba, sidecar)

    assert calibrated.shape == (1, 2)
    assert calibrated.sum() == pytest.approx(1.0)
    # raw positive-class prob 0.8 should calibrate down toward ~0.4 (half).
    assert calibrated[0, 1] < raw_proba[0, 1]


def test_apply_calibration_multiclass_renormalizes():
    from src.forecast.forecast_service import _apply_calibration

    classes = ["away", "draw", "home"]
    calibrators = []
    for _ in classes:
        cal = IsotonicRegression(out_of_bounds="clip")
        cal.fit([0.0, 0.5, 1.0], [0.0, 0.5, 1.0])  # identity-ish
        calibrators.append(cal)
    sidecar = {"type": "multiclass", "calibrator": calibrators, "classes": classes}
    raw_proba = np.array([[0.2, 0.3, 0.5]])

    calibrated = _apply_calibration(raw_proba, sidecar)

    assert calibrated.shape == (1, 3)
    assert calibrated.sum(axis=1)[0] == pytest.approx(1.0)


def test_apply_calibration_none_sidecar_is_noop():
    from src.forecast.forecast_service import _apply_calibration

    raw_proba = np.array([[0.3, 0.7]])
    result = _apply_calibration(raw_proba, None)
    assert np.array_equal(result, raw_proba)


def test_load_calibrator_sidecar_returns_none_when_absent(tmp_path: Path):
    from src.forecast.forecast_service import ForecastService

    model_path = tmp_path / "btts_xgboost_v1_20260101.joblib"
    model_path.write_bytes(b"fake")

    assert ForecastService._load_calibrator_sidecar(model_path) is None


def test_load_calibrator_sidecar_loads_real_file(tmp_path: Path):
    from src.forecast.forecast_service import ForecastService

    model_path = tmp_path / "btts_xgboost_v1_20260101.joblib"
    model_path.write_bytes(b"fake")
    cal = _fit_binary_calibrator()
    sidecar = {"type": "binary", "calibrator": cal}
    cal_path = model_path.with_suffix(model_path.suffix + ".calibration.pkl")
    joblib.dump(sidecar, str(cal_path))

    loaded = ForecastService._load_calibrator_sidecar(model_path)

    assert loaded is not None
    assert loaded["type"] == "binary"


def test_predict_target_uses_calibrator_when_present(tmp_path: Path):
    """End-to-end through _predict_target: a model + a real saved
    calibration sidecar in metadata must produce calibrated, not raw,
    probabilities."""
    class _FakeBinaryModel:
        classes_ = np.array([0, 1])

        def predict_proba(self, X):
            return np.array([[0.2, 0.8]] * len(X))

        def predict(self, X):
            return np.array([1] * len(X))

    cal = _fit_binary_calibrator()
    sidecar = {"type": "binary", "calibrator": cal}
    cal_path = tmp_path / "btts_fake.joblib.calibration.pkl"
    joblib.dump(sidecar, str(cal_path))

    service = ForecastService.__new__(ForecastService)
    definition = get_target_definition("btts")
    import pandas as pd
    feature_row = pd.DataFrame({"a": [1.0]})
    metadata = {"calibrator": sidecar}

    result = service._predict_target(definition, _FakeBinaryModel(), metadata, feature_row)

    # yes/no probabilities per btts's own class labels; calibrated value
    # should differ from the raw 0.8/0.2 the fake model always returns.
    probs = result["probabilities"]
    assert sum(probs.values()) == pytest.approx(1.0, abs=1e-4)
    assert probs != pytest.approx({"no": 0.2, "yes": 0.8}, abs=1e-6)


def test_predict_target_includes_raw_probabilities_alongside_calibrated(tmp_path: Path):
    """A107 (BUG-066 follow-up): forecast_payload must carry the
    pre-calibration probabilities too, whenever a calibrator actually ran --
    otherwise a future calibration collapse (BUG-066) is invisible in the
    data itself again."""
    class _FakeBinaryModel:
        classes_ = np.array([0, 1])

        def predict_proba(self, X):
            return np.array([[0.2, 0.8]] * len(X))

        def predict(self, X):
            return np.array([1] * len(X))

    cal = _fit_binary_calibrator()
    sidecar = {"type": "binary", "calibrator": cal}

    service = ForecastService.__new__(ForecastService)
    definition = get_target_definition("btts")
    import pandas as pd
    feature_row = pd.DataFrame({"a": [1.0]})
    metadata = {"calibrator": sidecar}

    result = service._predict_target(definition, _FakeBinaryModel(), metadata, feature_row)

    assert result["raw_probabilities"] == pytest.approx({"no": 0.2, "yes": 0.8}, abs=1e-6)
    assert result["raw_probabilities"] != result["probabilities"]


def test_predict_target_omits_raw_probabilities_when_no_calibrator():
    """No calibrator ran -- raw_probabilities would be identical to
    probabilities, so it's omitted rather than stored as a redundant copy."""
    class _FakeBinaryModel:
        classes_ = np.array([0, 1])

        def predict_proba(self, X):
            return np.array([[0.2, 0.8]] * len(X))

        def predict(self, X):
            return np.array([1] * len(X))

    service = ForecastService.__new__(ForecastService)
    definition = get_target_definition("btts")
    import pandas as pd
    feature_row = pd.DataFrame({"a": [1.0]})

    result = service._predict_target(definition, _FakeBinaryModel(), {}, feature_row)

    assert "raw_probabilities" not in result
