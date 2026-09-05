"""Tests for time-decay (recency) sample weighting (US#189).

Per direct user prioritization ("Do 8, 1, 3, 5, 6 in that order"), item #6:
"Time-decay sample weighting -- currently every model weights an 8-10-year-
old match the same as a recent one." `_compute_sample_weight` already
reweights for class balance but has no notion of match recency at all --
this adds an independent, multiplicative recency weight, off by default
(`time_decay_half_life_days=None` preserves every existing caller's exact
current behavior byte-for-byte), combined with the existing class-balance
weight for classifiers and used alone for regressors (which previously
always got sample_weight=None).
"""

from __future__ import annotations

from pathlib import Path
import sys
from typing import Any

import numpy as np
import pandas as pd
import pytest
import yaml

sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.models.model_manager import ModelManager, _compute_sample_weight, _compute_time_decay_weight


# ---------------------------------------------------------------------------
# _compute_time_decay_weight: pure function, no DB/model needed
# ---------------------------------------------------------------------------

def test_most_recent_match_gets_weight_one():
    dates = pd.Series(pd.to_datetime(["2020-01-01", "2024-01-01", "2026-01-01"]))
    weights = _compute_time_decay_weight(dates, half_life_days=365.0)
    assert weights[-1] == pytest.approx(1.0)


def test_older_matches_get_strictly_smaller_weight():
    dates = pd.Series(pd.to_datetime(["2020-01-01", "2023-01-01", "2026-01-01"]))
    weights = _compute_time_decay_weight(dates, half_life_days=365.0)
    assert weights[0] < weights[1] < weights[2]


def test_half_life_days_ago_gives_half_weight():
    dates = pd.Series(pd.to_datetime(["2025-01-01", "2026-01-01"]))
    weights = _compute_time_decay_weight(dates, half_life_days=365.0)
    assert weights[0] == pytest.approx(0.5, abs=0.01)


def test_all_same_date_gives_uniform_weight_one():
    dates = pd.Series(pd.to_datetime(["2026-01-01", "2026-01-01", "2026-01-01"]))
    weights = _compute_time_decay_weight(dates, half_life_days=365.0)
    assert np.allclose(weights, 1.0)


# ---------------------------------------------------------------------------
# ModelManager.train()/run_pipeline() threading
# ---------------------------------------------------------------------------

class _CapturingModel:
    """Minimal FPAIBaseModel stand-in that records train() kwargs."""

    def __init__(self) -> None:
        self.received_sample_weight: Any = "NOT_CALLED"

    def train(self, X: Any, y: Any, eval_set: Any | None = None, sample_weight: Any | None = None) -> None:
        self.received_sample_weight = sample_weight

    def predict_proba(self, X: Any) -> np.ndarray:
        n = len(X)
        return np.tile(np.array([0.5, 0.5]), (n, 1))

    def predict(self, X: Any) -> np.ndarray:
        return np.zeros(len(X))

    def save(self, path: str) -> None:
        raise NotImplementedError

    @classmethod
    def load(cls, path: str) -> "_CapturingModel":
        raise NotImplementedError


def _make_manager(
    tmp_path: Path, model: Any, target: str, time_decay_half_life_days: float | None = None,
) -> ModelManager:
    db_path = tmp_path / "test.db"
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        yaml.safe_dump({"paths": {"database_path": str(db_path)}}), encoding="utf-8"
    )
    return ModelManager(
        model=model,
        config_path=str(config_path),
        target_config={"target": target},
        time_decay_half_life_days=time_decay_half_life_days,
    )


def test_no_decay_by_default_classifier_gets_same_weight_as_before(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """Default (None) must be a byte-identical no-op vs. the pre-existing behavior."""
    model = _CapturingModel()
    manager = _make_manager(tmp_path, model, "result_3way")

    X = pd.DataFrame({"f1": range(20)})
    y_train = pd.Series(["home"] * 15 + ["draw"] * 5)
    dates = pd.Series(pd.date_range("2020-01-01", periods=20, freq="D"))
    empty_meta = pd.DataFrame(index=y_train.index)
    monkeypatch.setattr(
        manager, "prepare_training_data",
        lambda: (X, X, X, y_train, y_train, y_train, empty_meta),
    )
    monkeypatch.setattr(manager, "_load_selected_features", lambda: ["f1"])
    monkeypatch.setattr(manager, "_log_selected_features", lambda *_: None)
    monkeypatch.setattr(manager, "_log_feature_importance", lambda *_: None)

    manager.train()

    expected = _compute_sample_weight(y_train, "classification")
    np.testing.assert_array_equal(model.received_sample_weight, expected)


def test_decay_combines_multiplicatively_with_class_balance_weight(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    model = _CapturingModel()
    manager = _make_manager(tmp_path, model, "result_3way", time_decay_half_life_days=365.0)

    X = pd.DataFrame({"f1": range(20)})
    y_train = pd.Series(["home"] * 15 + ["draw"] * 5)
    dates = pd.Series(pd.date_range("2020-01-01", periods=20, freq="365D"))
    empty_meta = pd.DataFrame(index=y_train.index)
    monkeypatch.setattr(
        manager, "prepare_training_data",
        lambda: (X, X, X, y_train, y_train, y_train, empty_meta),
    )
    monkeypatch.setattr(manager, "_load_selected_features", lambda: ["f1"])
    monkeypatch.setattr(manager, "_log_selected_features", lambda *_: None)
    monkeypatch.setattr(manager, "_log_feature_importance", lambda *_: None)
    manager.train_dates = dates

    manager.train()

    class_weight = _compute_sample_weight(y_train, "classification")
    decay_weight = _compute_time_decay_weight(dates, 365.0)
    expected = class_weight * decay_weight
    np.testing.assert_allclose(model.received_sample_weight, expected)


def test_decay_used_alone_for_regression_targets(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """Regression targets get sample_weight=None today (no class-balance concept)
    -- with decay configured, they should get the decay weight alone, not None."""
    model = _CapturingModel()
    manager = _make_manager(tmp_path, model, "home_goals", time_decay_half_life_days=365.0)

    X = pd.DataFrame({"f1": range(10)})
    y_train = pd.Series([1.0, 2.0, 0.0, 1.0, 3.0, 2.0, 1.0, 0.0, 2.0, 1.0])
    dates = pd.Series(pd.date_range("2020-01-01", periods=10, freq="365D"))
    empty_meta = pd.DataFrame(index=y_train.index)
    monkeypatch.setattr(
        manager, "prepare_training_data",
        lambda: (X, X, X, y_train, y_train, y_train, empty_meta),
    )
    monkeypatch.setattr(manager, "_load_selected_features", lambda: ["f1"])
    monkeypatch.setattr(manager, "_log_selected_features", lambda *_: None)
    monkeypatch.setattr(manager, "_log_feature_importance", lambda *_: None)
    manager.train_dates = dates

    manager.train()

    expected = _compute_time_decay_weight(dates, 365.0)
    np.testing.assert_allclose(model.received_sample_weight, expected)


def test_no_decay_by_default_regression_still_gets_none(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    model = _CapturingModel()
    manager = _make_manager(tmp_path, model, "home_goals")

    X = pd.DataFrame({"f1": range(10)})
    y_train = pd.Series([1.0, 2.0, 0.0, 1.0, 3.0, 2.0, 1.0, 0.0, 2.0, 1.0])
    empty_meta = pd.DataFrame(index=y_train.index)
    monkeypatch.setattr(
        manager, "prepare_training_data",
        lambda: (X, X, X, y_train, y_train, y_train, empty_meta),
    )
    monkeypatch.setattr(manager, "_load_selected_features", lambda: ["f1"])
    monkeypatch.setattr(manager, "_log_selected_features", lambda *_: None)
    monkeypatch.setattr(manager, "_log_feature_importance", lambda *_: None)

    manager.train()

    assert model.received_sample_weight is None


def test_train_dates_set_by_prepare_training_data(tmp_path: Path):
    """Real integration: prepare_training_data() itself must populate
    self.train_dates (not just tests that monkeypatch it away) -- mirrors
    how self.training_cutoff is already set as a side effect there."""
    import duckdb

    db_path = tmp_path / "test.db"
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        yaml.safe_dump({"paths": {"database_path": str(db_path)}}), encoding="utf-8"
    )
    conn = duckdb.connect(str(db_path))
    conn.execute("""
        CREATE TABLE raw_matches (
            match_id VARCHAR, league VARCHAR, date TIMESTAMP,
            home_team VARCHAR, away_team VARCHAR, fthg INTEGER, ftag INTEGER, odds_h FLOAT
        )
    """)
    conn.execute("CREATE TABLE feature_store (match_id VARCHAR, f1 FLOAT)")
    rng = np.random.default_rng(0)
    for i in range(40):
        conn.execute(
            "INSERT INTO raw_matches VALUES (?, 'E0', ?, 'A', 'B', ?, ?, 2.0)",
            [f"m{i}", pd.Timestamp("2020-01-01") + pd.Timedelta(days=i * 10), int(rng.integers(0, 3)), int(rng.integers(0, 3))],
        )
        conn.execute("INSERT INTO feature_store VALUES (?, ?)", [f"m{i}", float(rng.normal())])
    conn.close()

    from src.models.base_model import XGBoostModel

    manager = ModelManager(
        model=XGBoostModel(early_stopping_rounds=None),
        config_path=str(config_path),
        target_config={"target": "home_win"},
    )
    manager._load_selected_features = lambda: ["f1"]
    manager.prepare_training_data()

    assert manager.train_dates is not None
    assert len(manager.train_dates) > 0
    assert manager.train_dates.max() <= pd.Timestamp(manager.training_cutoff)
