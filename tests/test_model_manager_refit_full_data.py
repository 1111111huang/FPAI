"""Tests for refit_on_full_data (direct user decision, 2026-09-16): the
deployed artifact should be a static, season-frozen model trained on
train+val+test combined, while metrics/promotion still use the honest
train-only-vs-held-out split unchanged. See src/models/model_manager.py's
run_pipeline() and CLAUDE.md's own documentation-on-change convention --
covered here, in `documents/user_stories.md` (US#173/US#192/US#193's
"freeze before this season" discussion).
"""

from __future__ import annotations

from pathlib import Path
import sys
from typing import Any

import mlflow
import pandas as pd
import pytest
import yaml

sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.models.model_manager import ModelManager


class _CapturingModel:
    """Minimal FPAIBaseModel stand-in that records every train() call."""

    def __init__(self) -> None:
        self.train_calls: list[dict[str, Any]] = []

    def train(self, X: Any, y: Any, eval_set: Any | None = None, sample_weight: Any | None = None) -> None:
        self.train_calls.append({"n_rows": len(X), "eval_set": eval_set})

    def predict_proba(self, X: Any):
        import numpy as np
        return np.tile(np.array([0.5, 0.5]), (len(X), 1))

    def predict(self, X: Any):
        import numpy as np
        return np.zeros(len(X))

    def save(self, path: str) -> None:
        Path(path).write_text("fake artifact")


def _make_manager(tmp_path: Path, model: Any, refit_on_full_data: bool) -> ModelManager:
    db_path = tmp_path / "test.db"
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        yaml.safe_dump({"paths": {"database_path": str(db_path), "model_dir": str(tmp_path / "models")}}),
        encoding="utf-8",
    )
    return ModelManager(
        model=model,
        config_path=str(config_path),
        target_config={"target": "result_3way"},
        refit_on_full_data=refit_on_full_data,
    )


def _wire_common_mocks(manager: ModelManager, monkeypatch: pytest.MonkeyPatch, n_train: int, n_val: int, n_test: int):
    X_train = pd.DataFrame({"f1": range(n_train)})
    X_val = pd.DataFrame({"f1": range(n_val)})
    X_test = pd.DataFrame({"f1": range(n_test)})
    y_train = pd.Series(["home"] * n_train)
    y_val = pd.Series(["home"] * n_val)
    y_test = pd.Series(["home"] * n_test)
    test_meta = pd.DataFrame(index=y_test.index)
    manager.full_data_cutoff = "2026-05-24T00:00:00"

    monkeypatch.setattr(manager, "prepare_training_data", lambda: (X_train, X_val, X_test, y_train, y_val, y_test, test_meta))
    monkeypatch.setattr(manager, "_load_selected_features", lambda: ["f1"])
    monkeypatch.setattr(manager, "_log_selected_features", lambda *_: None)
    monkeypatch.setattr(manager, "_log_feature_importance", lambda *_: None)
    monkeypatch.setattr(manager, "_evaluate_target", lambda *a, **k: ({"log_loss": 1.0}, None))
    monkeypatch.setattr(manager, "_build_artifact_metadata", lambda *a, **k: {})

    written: dict[str, Any] = {}
    def _capture_write(model_path, metadata):
        written["metadata"] = metadata
        return model_path
    monkeypatch.setattr(manager, "_write_artifact_metadata", _capture_write)

    calibrator_calls: list[Any] = []
    monkeypatch.setattr(manager, "_fit_and_save_calibrator", lambda *a, **k: calibrator_calls.append(1) or None)

    # mlflow: no active run needed for these unit-level assertions.
    for name in ("log_metric", "log_param", "set_tag", "set_tags", "log_artifact"):
        monkeypatch.setattr(mlflow, name, lambda *a, **k: None)

    return written, calibrator_calls


def test_default_still_trains_once_on_train_only(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """refit_on_full_data=False (default) must be byte-identical to pre-existing behavior: one train() call, train-only."""
    model = _CapturingModel()
    manager = _make_manager(tmp_path, model, refit_on_full_data=False)
    _wire_common_mocks(manager, monkeypatch, n_train=10, n_val=3, n_test=3)

    manager.run_pipeline(external_run=True)

    assert len(model.train_calls) == 1
    assert model.train_calls[0]["n_rows"] == 10  # train only


def test_refit_on_full_data_trains_twice_second_pass_uses_everything(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    model = _CapturingModel()
    manager = _make_manager(tmp_path, model, refit_on_full_data=True)
    _wire_common_mocks(manager, monkeypatch, n_train=10, n_val=3, n_test=3)

    manager.run_pipeline(external_run=True)

    assert len(model.train_calls) == 2, "expected train-only fit (for metrics) + full-data refit (for the saved artifact)"
    assert model.train_calls[0]["n_rows"] == 10  # the honest, held-out fit metrics come from
    assert model.train_calls[1]["n_rows"] == 16  # train+val+test combined -- everything
    assert model.train_calls[1]["eval_set"] is None  # nothing left to hold out for early stopping


def test_refit_on_full_data_records_metadata_and_skips_calibration(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    model = _CapturingModel()
    manager = _make_manager(tmp_path, model, refit_on_full_data=True)
    written, calibrator_calls = _wire_common_mocks(manager, monkeypatch, n_train=10, n_val=3, n_test=3)

    manager.run_pipeline(external_run=True)

    assert written["metadata"]["refit_on_full_data"] is True
    assert written["metadata"]["full_data_cutoff"] == "2026-05-24T00:00:00"
    assert calibrator_calls == [], "calibrating against X_val is in-sample once it's folded into the full-data fit"


def test_default_path_still_calibrates_and_has_no_full_data_cutoff(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    model = _CapturingModel()
    manager = _make_manager(tmp_path, model, refit_on_full_data=False)
    written, calibrator_calls = _wire_common_mocks(manager, monkeypatch, n_train=10, n_val=3, n_test=3)

    manager.run_pipeline(external_run=True)

    assert written["metadata"]["refit_on_full_data"] is False
    assert "full_data_cutoff" not in written["metadata"]
    assert calibrator_calls == [1]
