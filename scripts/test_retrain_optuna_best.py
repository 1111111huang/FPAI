"""Tests for scripts/retrain_optuna_best.py (US#180).

Focused on the actual new logic this script adds -- merging a fixed
objective/eval_metric onto XGBoostModel (but not XGBoostRegressorModel,
which needs neither) and choosing the right model class -- not
re-testing ModelManager.run_pipeline() itself, which already has its own
extensive coverage elsewhere.
"""

from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

sys.path.append(str(Path(__file__).resolve().parents[1]))

from scripts.retrain_optuna_best import retrain_with_params
from src.models.base_model import XGBoostModel, XGBoostRegressorModel


def _patched(monkeypatch_target: str = "scripts.retrain_optuna_best.ModelManager"):
    fake_manager_instance = MagicMock()
    fake_manager_instance.run_pipeline.return_value = Path("models/fake.joblib")
    fake_manager_cls = MagicMock(return_value=fake_manager_instance)
    return patch(monkeypatch_target, fake_manager_cls), fake_manager_cls, fake_manager_instance


def test_retrain_regressor_gets_no_injected_objective():
    """XGBoostRegressorModel needs neither objective nor eval_metric injected
    -- only the caller's own explicit params should reach it."""
    patcher, fake_manager_cls, fake_instance = _patched()
    with patcher:
        path = retrain_with_params(
            "home_goals", "xgb_regressor", "E0",
            {"n_estimators": 463, "max_depth": 2, "learning_rate": 0.05229},
        )
    assert path == Path("models/fake.joblib")
    fake_instance.run_pipeline.assert_called_once()
    manager_kwargs = fake_manager_cls.call_args.kwargs
    model = manager_kwargs["model"]
    assert isinstance(model, XGBoostRegressorModel)
    assert model.model.get_params()["n_estimators"] == 463
    assert model.model.get_params()["max_depth"] == 2
    assert "objective" not in model.model.get_params() or model.model.get_params()["objective"] != "binary:logistic"


def test_retrain_classifier_gets_correct_objective_for_binary_target():
    """XGBoostModel (btts, binary) gets binary:logistic/logloss injected
    automatically -- the caller shouldn't have to know/supply that."""
    patcher, fake_manager_cls, fake_instance = _patched()
    with patcher:
        retrain_with_params("btts", "xgb", "E0", {"n_estimators": 200, "max_depth": 3})
    model = fake_manager_cls.call_args.kwargs["model"]
    assert isinstance(model, XGBoostModel)
    assert model.model.get_params()["objective"] == "binary:logistic"
    assert model.model.get_params()["eval_metric"] == "logloss"
    assert model.model.get_params()["n_estimators"] == 200


def test_retrain_classifier_gets_correct_objective_for_multiclass_target():
    """result_3way (multiclass) gets multi:softprob/mlogloss + num_class,
    not the binary objective."""
    patcher, fake_manager_cls, fake_instance = _patched()
    with patcher:
        retrain_with_params("result_3way", "xgb", "E0", {"n_estimators": 200})
    model = fake_manager_cls.call_args.kwargs["model"]
    assert model.model.get_params()["objective"] == "multi:softprob"
    assert model.model.get_params()["eval_metric"] == "mlogloss"
    assert model.model.get_params()["num_class"] == 3


def test_retrain_unsupported_model_raises():
    import pytest
    with pytest.raises(ValueError):
        retrain_with_params("home_goals", "not_a_real_model", "E0", {})


def test_retrain_passes_context_and_feature_subset_through():
    patcher, fake_manager_cls, fake_instance = _patched()
    with patch("scripts.retrain_optuna_best.resolve_feature_subset_for_tier", return_value=["A", "B"]):
        with patcher:
            retrain_with_params("total_corners", "xgb_regressor", "SP1", {"n_estimators": 100})
    kwargs = fake_manager_cls.call_args.kwargs
    assert kwargs["context"] == "SP1"
    assert kwargs["competition_id"] == "SP1"
    assert kwargs["feature_subset"] == ["A", "B"]
