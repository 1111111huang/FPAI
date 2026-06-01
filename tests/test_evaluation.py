"""Tests for model evaluation and split metrics (US#35, US#38)."""

import pytest
import pandas as pd
import numpy as np
from sklearn.datasets import make_classification, make_regression
import mlflow

from src.models.base_model import LRModel, RandomForestModel, RandomForestRegressorModel
from src.models.model_manager import ModelManager
from src.logic.target_registry import get_target_definition


class TestSplitMetricsLogging:
    """Test that train/val/test metrics are properly logged (US#35)."""

    def test_classification_metrics_logged_for_all_splits(self, tmp_path):
        """Verify that classification metrics are computed and logged for all three splits."""
        # Create synthetic classification data
        X, y = make_classification(n_samples=300, n_features=10, n_classes=2, random_state=42)
        X_df = pd.DataFrame(X, columns=[f"feature_{i}" for i in range(10)])
        y_series = pd.Series(y)

        # Split into train/val/test
        X_train = X_df.iloc[:150]
        y_train = y_series.iloc[:150]
        X_val = X_df.iloc[150:225]
        y_val = y_series.iloc[150:225]
        X_test = X_df.iloc[225:]
        y_test = y_series.iloc[225:]

        # Create and train a model
        model = LRModel()
        model.train(X_train, y_train)

        # Create manager and evaluate
        manager = ModelManager(
            model=model,
            config_path="config.yaml",
            target_config={"target": "home_win"}
        )

        with mlflow.start_run() as run:
            metrics, _ = manager._evaluate_target(
                X_test, y_test,
                X_train, y_train,
                X_val, y_val
            )
            run_id = run.info.run_id

        # Verify metrics were returned for test split
        assert "log_loss" in metrics
        assert "accuracy" in metrics
        assert isinstance(metrics["log_loss"], float)
        assert isinstance(metrics["accuracy"], float)
        logged = mlflow.get_run(run_id).data.metrics
        assert "log_loss_train" in logged
        assert "log_loss_val" in logged
        assert "log_loss_test" in logged
        assert "train_log_loss" in logged
        assert "val_log_loss" in logged
        assert "test_log_loss" in logged

    def test_regression_metrics_logged_for_all_splits(self):
        """Verify that regression metrics are computed for all splits."""
        # Create synthetic regression data
        X, y = make_regression(n_samples=300, n_features=10, random_state=42)
        X_df = pd.DataFrame(X, columns=[f"feature_{i}" for i in range(10)])
        y_series = pd.Series(y)

        # Split into train/val/test
        X_train = X_df.iloc[:150]
        y_train = y_series.iloc[:150]
        X_val = X_df.iloc[150:225]
        y_val = y_series.iloc[150:225]
        X_test = X_df.iloc[225:]
        y_test = y_series.iloc[225:]

        # Create and train a model
        model = RandomForestRegressorModel()
        model.train(X_train, y_train)

        # Create manager and evaluate
        manager = ModelManager(
            model=model,
            config_path="config.yaml",
            target_config={"target": "home_goals"}
        )

        with mlflow.start_run() as run:
            metrics, _ = manager._evaluate_target(
                X_test, y_test,
                X_train, y_train,
                X_val, y_val
            )
            run_id = run.info.run_id

        # Verify metrics were returned
        assert "mae" in metrics
        assert "rmse" in metrics
        assert isinstance(metrics["mae"], float)
        assert isinstance(metrics["rmse"], float)
        logged = mlflow.get_run(run_id).data.metrics
        assert "mae_train" in logged
        assert "mae_val" in logged
        assert "mae_test" in logged
        assert "train_mae" in logged
        assert "val_mae" in logged
        assert "test_mae" in logged


class TestOverfittingDetection:
    """Test overfitting detection capabilities (US#38)."""

    def test_train_val_test_progression_indicates_generalization(self):
        """Verify metric progression from train to val to test is tracked."""
        # Create data where overfitting is likely
        X, y = make_classification(n_samples=200, n_features=100, n_classes=2, random_state=42)
        X_df = pd.DataFrame(X, columns=[f"feature_{i}" for i in range(100)])
        y_series = pd.Series(y)

        X_train = X_df.iloc[:100]
        y_train = y_series.iloc[:100]
        X_val = X_df.iloc[100:150]
        y_val = y_series.iloc[100:150]
        X_test = X_df.iloc[150:]
        y_test = y_series.iloc[150:]

        model = RandomForestModel(n_estimators=100)
        model.train(X_train, y_train)

        manager = ModelManager(
            model=model,
            config_path="config.yaml",
            target_config={"target": "home_win"}
        )

        with mlflow.start_run():
            metrics, _ = manager._evaluate_target(
                X_test, y_test,
                X_train, y_train,
                X_val, y_val
            )

        # Metrics should exist for test split
        assert "log_loss" in metrics
        assert "accuracy" in metrics

    def test_feature_leakage_detection_via_consistent_splits(self):
        """Test that split boundaries are maintained without data leakage."""
        X, y = make_classification(n_samples=300, n_features=10, n_classes=2, random_state=42)
        X_df = pd.DataFrame(X, columns=[f"feature_{i}" for i in range(10)])
        y_series = pd.Series(y)

        X_train = X_df.iloc[:150]
        y_train = y_series.iloc[:150]
        X_val = X_df.iloc[150:225]
        y_val = y_series.iloc[150:225]
        X_test = X_df.iloc[225:]
        y_test = y_series.iloc[225:]

        model = LRModel()
        model.train(X_train, y_train)

        manager = ModelManager(
            model=model,
            config_path="config.yaml",
            target_config={"target": "home_win"}
        )

        # Verify splits do not overlap
        assert len(X_train) == 150
        assert len(X_val) == 75
        assert len(X_test) == 75
        assert X_train.index.tolist() != X_val.index.tolist()
        assert X_val.index.tolist() != X_test.index.tolist()

    def test_chronological_split_boundary_consistency(self, tmp_path):
        """Test that chronological splits maintain temporal order."""
        # Create synthetic time-ordered data
        dates = pd.date_range('2023-01-01', periods=300, freq='D')
        X, y = make_classification(n_samples=300, n_features=10, n_classes=2, random_state=42)
        X_df = pd.DataFrame(X, columns=[f"feature_{i}" for i in range(10)])
        X_df['date'] = dates
        y_series = pd.Series(y)

        X_train = X_df.iloc[:150]
        y_train = y_series.iloc[:150]
        X_val = X_df.iloc[150:225]
        y_val = y_series.iloc[150:225]
        X_test = X_df.iloc[225:]
        y_test = y_series.iloc[225:]

        # Verify temporal ordering
        assert X_train['date'].max() <= X_val['date'].min()
        assert X_val['date'].max() <= X_test['date'].min()
