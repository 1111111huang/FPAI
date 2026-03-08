"""Model management utilities for training, evaluation, and versioned saving."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

import duckdb
import pandas as pd
from sklearn.metrics import accuracy_score, log_loss

from src.models.base_model import FPAIBaseModel
from src.utils.config_loader import AppSettings, load_settings
from src.utils.db_manager import DuckDBManager
from src.utils.logger import get_logger

LOGGER = get_logger(__name__)


class ModelManager:
    """Handle training data preparation, model evaluation, and model versioning."""

    def __init__(self, model: FPAIBaseModel, config_path: str = "config.yaml") -> None:
        """Initialize manager with a model instance and YAML config path."""
        self.model = model
        self.config_path = Path(config_path)
        self.config: AppSettings = load_settings(str(self.config_path))
        self.db_manager = DuckDBManager(config_path=str(self.config_path))
        self.model_dir = Path(self.config.paths.model_dir)
        self.model_dir.mkdir(parents=True, exist_ok=True)
        self.test_size = float(self.config.settings.test_size)

    def prepare_training_data(self) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
        """Build feature matrix and labels, then apply a time-based train/test split."""
        with self.db_manager.connection() as conn:
            df = conn.execute(
                """
                SELECT
                    r.match_id,
                    r.date,
                    r.fthg,
                    r.ftag,
                    f.home_avg_goals_scored,
                    f.home_avg_goals_conceded,
                    f.away_avg_goals_scored,
                    f.away_avg_goals_conceded
                FROM raw_matches r
                INNER JOIN feature_store f ON r.match_id = f.match_id
                ORDER BY r.date, r.match_id
                """
            ).fetchdf()

        if df.empty:
            raise ValueError("No joined training data found in raw_matches and feature_store.")

        # Binary baseline target: 1 if home win, else 0.
        df["target"] = (df["fthg"] > df["ftag"]).astype(int)
        df = df.dropna(
            subset=[
                "home_avg_goals_scored",
                "home_avg_goals_conceded",
                "away_avg_goals_scored",
                "away_avg_goals_conceded",
            ]
        ).reset_index(drop=True)

        if df.empty:
            raise ValueError("No rows left after dropping records with missing feature values.")

        feature_columns = [
            "home_avg_goals_scored",
            "home_avg_goals_conceded",
            "away_avg_goals_scored",
            "away_avg_goals_conceded",
        ]

        X = df[feature_columns]
        y = df["target"]

        split_index = max(1, int(len(df) * (1 - self.test_size)))
        split_index = min(split_index, len(df) - 1)

        X_train = X.iloc[:split_index].copy()
        X_test = X.iloc[split_index:].copy()
        y_train = y.iloc[:split_index].copy()
        y_test = y.iloc[split_index:].copy()

        if y_train.nunique() < 2:
            raise ValueError("Training split has a single class; cannot train Logistic Regression.")

        return X_train, X_test, y_train, y_test

    def run_pipeline(self) -> Path:
        """Train model, evaluate it, and save a timestamped artifact path."""
        try:
            X_train, X_test, y_train, y_test = self.prepare_training_data()

            self.model.train(X_train, y_train)
            probabilities = self.model.predict_proba(X_test)

            if probabilities.ndim == 2 and probabilities.shape[1] > 1:
                positive_proba = probabilities[:, 1]
            else:
                positive_proba = probabilities.ravel()

            predictions = (positive_proba >= 0.5).astype(int)
            accuracy = accuracy_score(y_test, predictions)
            loss = log_loss(y_test, positive_proba, labels=[0, 1])

            LOGGER.info("Accuracy: %.4f", accuracy)
            LOGGER.info("Log Loss: %.4f", loss)

            date_tag = datetime.now().strftime("%Y%m%d")
            model_prefix = self.model.__class__.__name__.lower().replace("model", "")
            save_path = self.model_dir / f"{model_prefix}_v1_{date_tag}.joblib"
            self.model.save(str(save_path))
            return save_path
        except duckdb.Error:
            LOGGER.exception("Database failure during model pipeline.")
            raise
        except Exception:
            LOGGER.exception("Model pipeline failed.")
            raise
