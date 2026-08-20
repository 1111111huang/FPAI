"""Tests for balanced sample_weight computation and threading (draw-blindness fix).

See documents/agent_user_stories.md / conversation history: result_3way's XGBoost
classifier was found to have 2.1% recall on the 'draw' class (SP1) despite draws
being ~25% of real outcomes -- trained with plain unweighted multiclass log-loss.
This adds inverse-class-frequency sample weighting for classification targets,
generic across all classifiers (not result_3way-specific), computed via sklearn's
own compute_sample_weight('balanced', ...) rather than a hand-rolled formula.
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

from src.models.model_manager import ModelManager, _compute_sample_weight
from src.models.base_model import XGBoostModel, XGBoostRegressorModel


# ---------------------------------------------------------------------------
# _compute_sample_weight: pure function, no DB/model needed
# ---------------------------------------------------------------------------

def test_compute_sample_weight_returns_none_for_regression() -> None:
    y = pd.Series([1.0, 2.0, 3.0])
    assert _compute_sample_weight(y, "regression") is None


def test_compute_sample_weight_balances_imbalanced_classes() -> None:
    y = pd.Series(["home"] * 80 + ["away"] * 15 + ["draw"] * 5)
    weights = _compute_sample_weight(y, "classification")

    assert weights is not None
    assert len(weights) == len(y)

    home_weight = weights[y == "home"][0]
    away_weight = weights[y == "away"][0]
    draw_weight = weights[y == "draw"][0]

    # Balanced weighting must upweight the minority class relative to majority.
    assert draw_weight > away_weight > home_weight

    # Definition of 'balanced': each class's total weight mass is equal.
    for label in ("home", "away", "draw"):
        total = weights[y == label].sum()
        assert total == pytest.approx(len(y) / 3, rel=1e-6)


# ---------------------------------------------------------------------------
# ModelManager.train() threading: fake model captures what it was called with
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


def _make_manager(tmp_path: Path, model: Any, target: str) -> ModelManager:
    db_path = tmp_path / "test.db"
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        yaml.safe_dump({"paths": {"database_path": str(db_path)}}), encoding="utf-8"
    )
    return ModelManager(
        model=model,
        config_path=str(config_path),
        target_config={"target": target},
    )


def test_model_manager_train_passes_balanced_sample_weight_for_classifier(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    model = _CapturingModel()
    manager = _make_manager(tmp_path, model, "result_3way")

    X = pd.DataFrame({"f1": range(20)})
    y_train = pd.Series(["home"] * 15 + ["draw"] * 5)
    empty_meta = pd.DataFrame(index=y_train.index)
    monkeypatch.setattr(
        manager,
        "prepare_training_data",
        lambda: (X, X, X, y_train, y_train, y_train, empty_meta),
    )
    monkeypatch.setattr(manager, "_load_selected_features", lambda: ["f1"])
    monkeypatch.setattr(manager, "_log_selected_features", lambda *_: None)
    monkeypatch.setattr(manager, "_log_feature_importance", lambda *_: None)

    manager.train()

    assert model.received_sample_weight is not None
    assert len(model.received_sample_weight) == len(y_train)


def test_model_manager_train_passes_no_sample_weight_for_regression(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    model = _CapturingModel()
    manager = _make_manager(tmp_path, model, "home_goals")

    X = pd.DataFrame({"f1": range(10)})
    y_train = pd.Series([1.0, 2.0, 0.0, 1.0, 3.0, 2.0, 1.0, 0.0, 2.0, 1.0])
    empty_meta = pd.DataFrame(index=y_train.index)
    monkeypatch.setattr(
        manager,
        "prepare_training_data",
        lambda: (X, X, X, y_train, y_train, y_train, empty_meta),
    )
    monkeypatch.setattr(manager, "_load_selected_features", lambda: ["f1"])
    monkeypatch.setattr(manager, "_log_selected_features", lambda *_: None)
    monkeypatch.setattr(manager, "_log_feature_importance", lambda *_: None)

    manager.train()

    assert model.received_sample_weight is None


# ---------------------------------------------------------------------------
# XGBoostModel: sample_weight must actually influence what the model learns
# ---------------------------------------------------------------------------

def _make_imbalanced_dataset(rng: np.random.RandomState, n_majority: int = 120, n_minority: int = 15):
    """3-class synthetic set mirroring the real result_3way shape: 'minority'
    (analog of 'draw') sits ambiguously between the two majority clusters on
    f1 and only weakly separable on f2 (heavy overlap, small mean shift) --
    an unweighted model can minimize loss by mostly ignoring it, same failure
    mode diagnosed live on SP1 (2.1% recall on 'draw'). Heavy sample_weight
    on the minority should force the model to actually use the weak f2 signal."""
    f1_a = rng.normal(-1.0, 1.0, n_majority)
    f2_a = rng.normal(0.0, 1.0, n_majority)
    f1_b = rng.normal(1.0, 1.0, n_majority)
    f2_b = rng.normal(0.0, 1.0, n_majority)
    f1_min = rng.normal(0.0, 1.0, n_minority)
    f2_min = rng.normal(0.5, 1.0, n_minority)

    X = pd.DataFrame({
        "f1": np.concatenate([f1_a, f1_b, f1_min]),
        "f2": np.concatenate([f2_a, f2_b, f2_min]),
    })
    y = pd.Series(["majority_a"] * n_majority + ["majority_b"] * n_majority + ["minority"] * n_minority)
    return X, y


def test_xgboost_model_sample_weight_improves_minority_class_recall() -> None:
    rng = np.random.RandomState(42)
    X, y = _make_imbalanced_dataset(rng)

    multiclass_kwargs = {"objective": "multi:softprob", "eval_metric": "mlogloss", "num_class": 3}

    unweighted = XGBoostModel(**multiclass_kwargs)
    unweighted.train(X, y, eval_set=[(X, y)])
    unweighted_preds = unweighted.predict(X)
    unweighted_minority_recall = ((y == "minority") & (unweighted_preds == "minority")).sum() / (y == "minority").sum()

    from src.models.model_manager import _compute_sample_weight
    weights = _compute_sample_weight(y, "classification")

    weighted = XGBoostModel(**multiclass_kwargs)
    weighted.train(X, y, eval_set=[(X, y)], sample_weight=weights)
    weighted_preds = weighted.predict(X)
    weighted_minority_recall = ((y == "minority") & (weighted_preds == "minority")).sum() / (y == "minority").sum()

    # Unweighted should reproduce the diagnosed real failure mode: mostly ignores
    # the minority class. Weighted should clearly, substantially recover it.
    assert unweighted_minority_recall < 0.5
    assert weighted_minority_recall > unweighted_minority_recall
    assert weighted_minority_recall >= 0.8


def test_compute_sample_weight_alpha_defaults_to_one_unchanged() -> None:
    y = pd.Series(["home"] * 80 + ["away"] * 15 + ["draw"] * 5)
    default_call = _compute_sample_weight(y, "classification")
    explicit_alpha_one = _compute_sample_weight(y, "classification", alpha=1.0)
    np.testing.assert_array_equal(default_call, explicit_alpha_one)


def test_compute_sample_weight_alpha_zero_gives_uniform_weights() -> None:
    y = pd.Series(["home"] * 80 + ["away"] * 15 + ["draw"] * 5)
    weights = _compute_sample_weight(y, "classification", alpha=0.0)
    np.testing.assert_allclose(weights, np.ones(len(y)))


def test_compute_sample_weight_alpha_half_is_between_uniform_and_balanced() -> None:
    y = pd.Series(["home"] * 80 + ["away"] * 15 + ["draw"] * 5)
    balanced = _compute_sample_weight(y, "classification", alpha=1.0)
    dampened = _compute_sample_weight(y, "classification", alpha=0.5)
    uniform = _compute_sample_weight(y, "classification", alpha=0.0)

    draw_mask = (y == "draw").to_numpy()
    # Draw's weight is above 1.0 (uniform) since it's the minority class --
    # dampened must sit strictly between the uniform (1.0) and fully-balanced
    # draw weight, not equal to either.
    assert uniform[draw_mask][0] < dampened[draw_mask][0] < balanced[draw_mask][0]


def test_compute_sample_weight_alpha_still_none_for_regression() -> None:
    y = pd.Series([1.0, 2.0, 3.0])
    assert _compute_sample_weight(y, "regression", alpha=0.5) is None
