"""Tests for GoalStackerModel (US#71)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.models.goal_stacker import GoalStackerModel
from src.models.model_factory import ModelFactory


def _make_data(n: int = 200, n_features: int = 10, seed: int = 0) -> tuple[pd.DataFrame, np.ndarray]:
    rng = np.random.default_rng(seed)
    X = pd.DataFrame(
        rng.standard_normal((n, n_features)),
        columns=[f"feat_{i}" for i in range(n_features)],
    )
    # Simulate Poisson-like count target
    y = rng.poisson(lam=1.5, size=n).astype(float)
    return X, y


class TestGoalStackerInit:
    def test_default_init(self):
        m = GoalStackerModel()
        assert m._xgb_params["objective"] == "count:poisson"
        assert m._glm_params["alpha"] == 0.1
        assert m._ridge_params["alpha"] == 1.0

    def test_custom_params(self):
        m = GoalStackerModel(xgb_params={"n_estimators": 50}, ridge_params={"alpha": 0.5})
        assert m._xgb_params["n_estimators"] == 50
        assert m._ridge_params["alpha"] == 0.5

    def test_factory_registration(self):
        m = ModelFactory.get_model("goal_stacker")
        assert isinstance(m, GoalStackerModel)

    def test_factory_alias(self):
        m = ModelFactory.get_model("stacker")
        assert isinstance(m, GoalStackerModel)


class TestGoalStackerTrainPredict:
    def test_train_and_predict_shape(self):
        X, y = _make_data(200)
        m = GoalStackerModel(xgb_params={"n_estimators": 10, "early_stopping_rounds": None})
        m.train(X, y)
        preds = m.predict(X)
        assert preds.shape == (200,)

    def test_predictions_non_negative(self):
        X, y = _make_data(200)
        m = GoalStackerModel(xgb_params={"n_estimators": 10, "early_stopping_rounds": None})
        m.train(X, y)
        preds = m.predict(X)
        assert (preds >= 0).all()

    def test_train_stores_feature_columns(self):
        X, y = _make_data(50)
        m = GoalStackerModel(xgb_params={"n_estimators": 5, "early_stopping_rounds": None})
        m.train(X, y)
        assert m._feature_columns == list(X.columns)

    def test_predict_proba_raises(self):
        m = GoalStackerModel()
        X, _ = _make_data(10)
        with pytest.raises(TypeError):
            m.predict_proba(X)

    def test_predict_with_subset_columns_reorders(self):
        X, y = _make_data(100)
        m = GoalStackerModel(xgb_params={"n_estimators": 5, "early_stopping_rounds": None})
        m.train(X, y)
        # Shuffle columns — predict should reorder
        X_shuffled = X[list(reversed(X.columns))]
        preds = m.predict(X_shuffled)
        assert preds.shape == (100,)


class TestGoalStackerSaveLoad:
    def test_save_and_load(self, tmp_path):
        X, y = _make_data(100)
        m = GoalStackerModel(xgb_params={"n_estimators": 5, "early_stopping_rounds": None})
        m.train(X, y)
        preds_before = m.predict(X)

        save_path = str(tmp_path / "stacker.joblib")
        m.save(save_path)

        m2 = GoalStackerModel.load(save_path)
        preds_after = m2.predict(X)
        np.testing.assert_allclose(preds_before, preds_after)

    def test_loaded_model_has_correct_type(self, tmp_path):
        X, y = _make_data(60)
        m = GoalStackerModel(xgb_params={"n_estimators": 5, "early_stopping_rounds": None})
        m.train(X, y)
        path = str(tmp_path / "stacker.joblib")
        m.save(path)
        m2 = GoalStackerModel.load(path)
        assert isinstance(m2, GoalStackerModel)
        assert m2._feature_columns == m._feature_columns
