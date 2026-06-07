"""Tests for the learning curve analysis module."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

from src.utils.learning_curve import (
    LearningCurveAnalyzer,
    _compute_val_metrics,
    _make_model,
    summarise_findings,
)
from src.logic.target_registry import get_target_definition
from src.models.base_model import XGBoostModel, XGBoostRegressorModel


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _dummy_splits(n_train: int = 500, n_val: int = 100, n_feat: int = 10):
    rng = np.random.default_rng(0)
    X = pd.DataFrame(rng.standard_normal((n_train + n_val + 50, n_feat)))
    y_bin = pd.Series(rng.integers(0, 2, n_train + n_val + 50))
    y_reg = pd.Series(rng.integers(0, 5, n_train + n_val + 50).astype(float))
    meta = pd.DataFrame({"match_id": range(50), "odds_h": [1.5] * 50})

    def split(y):
        return (
            X.iloc[:n_train],
            X.iloc[n_train : n_train + n_val],
            X.iloc[n_train + n_val :],
            y.iloc[:n_train],
            y.iloc[n_train : n_train + n_val],
            y.iloc[n_train + n_val :],
            meta,
        )

    return split(y_bin), split(y_reg)


# ---------------------------------------------------------------------------
# _make_model
# ---------------------------------------------------------------------------

class TestMakeModel:
    def test_regression_target_returns_regressor(self):
        defn = get_target_definition("home_goals")
        model = _make_model(defn)
        assert isinstance(model, XGBoostRegressorModel)

    def test_binary_classification_returns_classifier(self):
        defn = get_target_definition("btts")
        model = _make_model(defn)
        assert isinstance(model, XGBoostModel)

    def test_multiclass_classification_returns_classifier(self):
        defn = get_target_definition("result_3way")
        model = _make_model(defn)
        assert isinstance(model, XGBoostModel)


# ---------------------------------------------------------------------------
# _compute_val_metrics
# ---------------------------------------------------------------------------

class TestComputeValMetrics:
    def test_regression_returns_mae_as_primary(self):
        defn = get_target_definition("home_goals")
        model = MagicMock()
        model.predict.return_value = np.array([1.0, 2.0, 1.5])
        y = pd.Series([1.0, 1.5, 2.0])
        X = pd.DataFrame({"a": [1, 2, 3]})
        metrics = _compute_val_metrics(model, defn, X, y)
        assert "mae" in metrics
        assert "primary" in metrics
        assert metrics["mae"] == metrics["primary"]

    def test_binary_classification_returns_log_loss_as_primary(self):
        defn = get_target_definition("btts")
        model = MagicMock()
        model.predict_proba.return_value = np.array([[0.3, 0.7], [0.6, 0.4], [0.4, 0.6]])
        model.predict.return_value = np.array([1, 0, 1])
        model.classes_ = np.array([0, 1])
        y = pd.Series([1, 0, 1])
        X = pd.DataFrame({"a": [1, 2, 3]})
        metrics = _compute_val_metrics(model, defn, X, y)
        assert "log_loss" in metrics
        assert "accuracy" in metrics
        assert metrics["log_loss"] == metrics["primary"]

    def test_regression_mae_is_correct(self):
        defn = get_target_definition("total_goals")
        model = MagicMock()
        model.predict.return_value = np.array([2.0, 2.0])
        y = pd.Series([1.0, 3.0])
        X = pd.DataFrame({"a": [1, 2]})
        metrics = _compute_val_metrics(model, defn, X, y)
        assert abs(metrics["mae"] - 1.0) < 1e-9


# ---------------------------------------------------------------------------
# LearningCurveAnalyzer
# ---------------------------------------------------------------------------

class TestLearningCurveAnalyzer:
    def test_run_structure_binary(self, tmp_path):
        (splits_bin, _) = _dummy_splits()
        analyzer = LearningCurveAnalyzer("btts", output_dir=str(tmp_path))
        with patch.object(analyzer, "_load_splits", return_value=splits_bin):
            result = analyzer.run(fractions=[0.5, 1.0])
        assert result["target"] == "btts"
        assert result["metric"] == "log_loss"
        assert len(result["results"]) == 2
        for row in result["results"]:
            assert "fraction" in row
            assert "train_n" in row
            assert "log_loss" in row

    def test_run_structure_regression(self, tmp_path):
        (_, splits_reg) = _dummy_splits()
        analyzer = LearningCurveAnalyzer("home_goals", output_dir=str(tmp_path))
        with patch.object(analyzer, "_load_splits", return_value=splits_reg):
            result = analyzer.run(fractions=[0.5, 1.0])
        assert result["metric"] == "mae"
        assert all("mae" in row for row in result["results"])

    def test_training_sizes_increase_monotonically(self, tmp_path):
        (splits_bin, _) = _dummy_splits()
        analyzer = LearningCurveAnalyzer("btts", output_dir=str(tmp_path))
        with patch.object(analyzer, "_load_splits", return_value=splits_bin):
            result = analyzer.run(fractions=[0.2, 0.4, 0.6, 0.8, 1.0])
        ns = [r["train_n"] for r in result["results"]]
        assert ns == sorted(ns)

    def test_save_results_writes_csv(self, tmp_path):
        analyzer = LearningCurveAnalyzer("btts", output_dir=str(tmp_path))
        run_output = {
            "target": "btts",
            "metric": "log_loss",
            "total_train_n": 500,
            "total_val_n": 100,
            "results": [
                {"fraction": 0.5, "train_n": 250, "log_loss": 0.68, "accuracy": 0.54},
                {"fraction": 1.0, "train_n": 500, "log_loss": 0.67, "accuracy": 0.55},
            ],
        }
        path = analyzer.save_results(run_output)
        assert path.exists()
        df = pd.read_csv(path)
        assert len(df) == 2
        assert list(df.columns[:2]) == ["fraction", "train_n"]

    def test_save_chart_writes_png(self, tmp_path):
        analyzer = LearningCurveAnalyzer("home_goals", output_dir=str(tmp_path))
        run_output = {
            "target": "home_goals",
            "metric": "mae",
            "total_train_n": 500,
            "total_val_n": 100,
            "results": [
                {"fraction": 0.5, "train_n": 250, "mae": 0.95},
                {"fraction": 1.0, "train_n": 500, "mae": 0.93},
            ],
        }
        path = analyzer.save_chart(run_output)
        assert path is not None and path.exists()

    def test_empty_results_save_chart_returns_none(self, tmp_path):
        analyzer = LearningCurveAnalyzer("btts", output_dir=str(tmp_path))
        result = {"target": "btts", "metric": "log_loss", "total_train_n": 0, "total_val_n": 0, "results": []}
        assert analyzer.save_chart(result) is None


# ---------------------------------------------------------------------------
# summarise_findings
# ---------------------------------------------------------------------------

class TestSummariseFindings:
    def _make_result(self, target, metric, values):
        return {
            "target": target,
            "metric": metric,
            "total_train_n": 500,
            "total_val_n": 100,
            "results": [{"fraction": f, "train_n": int(500 * f), metric: v} for f, v in values],
        }

    def test_plateau_detection(self):
        result = self._make_result("btts", "log_loss", [(0.2, 0.71), (0.4, 0.70), (0.6, 0.685), (0.8, 0.681), (1.0, 0.681)])
        summary = summarise_findings({"btts": result})
        assert "PLATEAU" in summary

    def test_still_improving_detection(self):
        result = self._make_result("home_goals", "mae", [(0.2, 1.1), (0.4, 1.0), (0.6, 0.97), (0.8, 0.95), (1.0, 0.93)])
        summary = summarise_findings({"home_goals": result})
        assert "STILL IMPROVING" in summary
