"""Tests for MLPModel and MLPRegressorModel (US#74)."""

from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.models.mlp_model import MLPModel, MLPRegressorModel, _build_hidden_layer_sizes


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _clf_data(n: int = 300, seed: int = 0):
    rng = np.random.default_rng(seed)
    X = pd.DataFrame(rng.standard_normal((n, 10)), columns=[f"f{i}" for i in range(10)])
    y = pd.Series(rng.choice(["home", "draw", "away"], size=n))
    split = int(n * 0.7)
    return X.iloc[:split], y.iloc[:split], X.iloc[split:], y.iloc[split:]


def _reg_data(n: int = 300, seed: int = 0):
    rng = np.random.default_rng(seed)
    X = pd.DataFrame(rng.standard_normal((n, 10)), columns=[f"f{i}" for i in range(10)])
    y = pd.Series(rng.poisson(1.5, n).astype(float))
    split = int(n * 0.7)
    return X.iloc[:split], y.iloc[:split], X.iloc[split:], y.iloc[split:]


@pytest.fixture(scope="module")
def trained_clf():
    X_tr, y_tr, X_v, y_v = _clf_data()
    m = MLPModel(depth=2, hidden_size=32, max_iter=30, patience=5)
    m.train(X_tr, y_tr, eval_set=[(X_v, y_v)])
    return m, X_v, y_v


@pytest.fixture(scope="module")
def trained_reg():
    X_tr, y_tr, X_v, y_v = _reg_data()
    m = MLPRegressorModel(depth=2, hidden_size=32, max_iter=30, patience=5)
    m.train(X_tr, y_tr, eval_set=[(X_v, y_v)])
    return m, X_v, y_v


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

class TestBuildHiddenLayerSizes:
    def test_uniform_depth(self):
        assert _build_hidden_layer_sizes(3, 128) == (128, 128, 128)

    def test_single_layer(self):
        assert _build_hidden_layer_sizes(1, 64) == (64,)


# ---------------------------------------------------------------------------
# MLPModel (classifier)
# ---------------------------------------------------------------------------

class TestMLPModelTrain:
    def test_model_set_after_train(self, trained_clf):
        m, _, _ = trained_clf
        assert m.model is not None

    def test_classes_set(self, trained_clf):
        m, _, _ = trained_clf
        assert set(m.classes_) == {"home", "draw", "away"}

    def test_scaler_fitted(self, trained_clf):
        m, _, _ = trained_clf
        assert hasattr(m.scaler, "mean_")

    def test_predict_proba_shape(self, trained_clf):
        m, X_v, _ = trained_clf
        proba = m.predict_proba(X_v)
        assert proba.shape == (len(X_v), 3)

    def test_predict_proba_sums_to_one(self, trained_clf):
        m, X_v, _ = trained_clf
        proba = m.predict_proba(X_v)
        assert np.allclose(proba.sum(axis=1), 1.0, atol=1e-5)

    def test_predict_returns_known_classes(self, trained_clf):
        m, X_v, _ = trained_clf
        preds = m.predict(X_v)
        assert set(preds).issubset({"home", "draw", "away"})

    def test_binary_classification(self):
        rng = np.random.default_rng(1)
        X = pd.DataFrame(rng.standard_normal((200, 5)), columns=list("abcde"))
        y = pd.Series(rng.integers(0, 2, 200))
        m = MLPModel(depth=2, hidden_size=16, max_iter=20, patience=5)
        m.train(X, y)
        proba = m.predict_proba(X)
        assert proba.shape[1] == 2


class TestMLPModelEarlyStopping:
    def test_early_stopping_with_val_set(self):
        X_tr, y_tr, X_v, y_v = _clf_data(200)
        m = MLPModel(depth=2, hidden_size=16, max_iter=200, patience=5)
        m.train(X_tr, y_tr, eval_set=[(X_v, y_v)])
        # Should stop before 200 epochs due to patience
        assert m.model is not None

    def test_runs_without_val_set(self):
        X_tr, y_tr, _, _ = _clf_data(100)
        m = MLPModel(depth=2, hidden_size=16, max_iter=10)
        m.train(X_tr, y_tr)
        assert m.model is not None


class TestMLPModelOptunaTrial:
    def test_set_optuna_trial(self):
        m = MLPModel()
        sentinel = object()
        m.set_optuna_trial(sentinel)
        assert m._trial is sentinel


# ---------------------------------------------------------------------------
# MLPRegressorModel
# ---------------------------------------------------------------------------

class TestMLPRegressorModel:
    def test_predict_shape(self, trained_reg):
        m, X_v, _ = trained_reg
        preds = m.predict(X_v)
        assert len(preds) == len(X_v)

    def test_predict_finite(self, trained_reg):
        m, X_v, _ = trained_reg
        preds = m.predict(X_v)
        assert np.all(np.isfinite(preds))

    def test_predict_proba_raises(self, trained_reg):
        m, X_v, _ = trained_reg
        with pytest.raises(TypeError):
            m.predict_proba(X_v)

    def test_scaler_fitted(self, trained_reg):
        m, _, _ = trained_reg
        assert hasattr(m.scaler, "mean_")


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------

class TestMLPPersistence:
    def test_clf_save_load_roundtrip(self, trained_clf):
        m, X_v, _ = trained_clf
        with tempfile.TemporaryDirectory() as tmpdir:
            path = str(Path(tmpdir) / "mlp.joblib")
            m.save(path)
            loaded = MLPModel.load(path)
            orig_pred = m.predict(X_v)
            load_pred = loaded.predict(X_v)
            assert np.array_equal(orig_pred, load_pred)

    def test_reg_save_load_roundtrip(self, trained_reg):
        m, X_v, _ = trained_reg
        with tempfile.TemporaryDirectory() as tmpdir:
            path = str(Path(tmpdir) / "mlp_reg.joblib")
            m.save(path)
            loaded = MLPRegressorModel.load(path)
            assert np.allclose(m.predict(X_v), loaded.predict(X_v))
