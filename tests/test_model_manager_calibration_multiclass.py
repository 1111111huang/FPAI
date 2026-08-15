"""Tests for BUG (found live, 2026-08-14): ModelManager._fit_and_save_calibrator
silently no-ops for multiclass string-labeled targets (result_3way's
'home'/'draw'/'away').

Root cause: y_val_arr = pd.to_numeric(y_val, errors="coerce") turns every
string label into NaN, and the broad `except Exception` swallows the
downstream failure as a quiet "Calibration skipped" log line. Confirmed
pre-existing (no .calibration.pkl sidecar exists for the original SP1
result_3way artifact either) -- not introduced by the sample_weight fix,
just surfaced while verifying it.
"""

from __future__ import annotations

from pathlib import Path
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.models.model_manager import ModelManager
from src.models.base_model import XGBoostModel


def _train_multiclass_model() -> tuple[XGBoostModel, pd.DataFrame, pd.Series]:
    rng = np.random.RandomState(42)
    n = 90
    f1 = rng.normal(0.0, 1.0, n)
    y = pd.Series(np.where(f1 > 0.5, "home", np.where(f1 < -0.5, "away", "draw")))
    X = pd.DataFrame({"f1": f1, "f2": rng.normal(0.0, 1.0, n)})

    model = XGBoostModel(objective="multi:softprob", eval_metric="mlogloss", num_class=3)
    model.train(X, y, eval_set=[(X, y)])
    return model, X, y


def test_calibration_succeeds_for_multiclass_string_labels(tmp_path: Path) -> None:
    model, X_val, y_val = _train_multiclass_model()
    model_path = tmp_path / "result_3way_test.joblib"

    result = ModelManager._fit_and_save_calibrator(model, X_val, y_val, model_path)

    assert result is not None
    assert np.isfinite(result["log_loss_before"])
    assert np.isfinite(result["log_loss_after"])


def test_calibration_writes_sidecar_file_for_multiclass_string_labels(tmp_path: Path) -> None:
    model, X_val, y_val = _train_multiclass_model()
    model_path = tmp_path / "result_3way_test.joblib"

    ModelManager._fit_and_save_calibrator(model, X_val, y_val, model_path)

    sidecar_path = model_path.with_suffix(model_path.suffix + ".calibration.pkl")
    assert sidecar_path.exists()
