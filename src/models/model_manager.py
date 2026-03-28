"""Model management utilities for training, evaluation, and versioned saving."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

import duckdb
import mlflow
import mlflow.sklearn
import mlflow.xgboost
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, log_loss, precision_score

from src.logic.target_resolver import TargetResolver
from src.models.base_model import FPAIBaseModel, XGBoostModel
from src.strategy.backtester import Backtester
from src.utils.config_loader import AppSettings, load_settings
from src.utils.db_manager import DuckDBManager
from src.utils.logger import get_logger

LOGGER = get_logger(__name__)


class ModelManager:
    """Handle training data preparation, model evaluation, and model versioning."""

    def __init__(
        self,
        model: FPAIBaseModel,
        config_path: str = "config.yaml",
        league_tier: str = "all",
        test_season: str = "time_split",
        feature_version: str = "v1",
        target_config: dict[str, str | float | int] | None = None,
    ) -> None:
        """Initialize manager with a model instance and YAML config path."""
        self.model = model
        self.config_path = Path(config_path)
        self.config: AppSettings = load_settings(str(self.config_path))
        self.db_manager = DuckDBManager(config_path=str(self.config_path))
        self.model_dir = Path(self.config.paths.model_dir)
        self.model_dir.mkdir(parents=True, exist_ok=True)
        self.test_size = float(self.config.settings.test_size)
        self.mlflow_tags = {
            "league_tier": league_tier,
            "test_season": test_season,
            "feature_version": feature_version,
            "target": (target_config or {}).get("target_type", "home_win"),
        }
        self.target_config = target_config or {"target_type": "home_win"}
        mlflow.set_experiment("FPAI_Evolution")

    def prepare_training_data(
        self,
    ) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, pd.DataFrame]:
        """Build feature matrix and labels, then apply a time-based train/test split."""
        with self.db_manager.connection() as conn:
            df = conn.execute(
                """
                SELECT
                    r.match_id,
                    r.date,
                    r.fthg,
                    r.ftag,
                    r.odds_h,
                    f.home_avg_goals_scored,
                    f.home_avg_goals_conceded,
                    f.away_avg_goals_scored,
                    f.away_avg_goals_conceded,
                    f.is_cold_start,
                    f.relative_tier_change,
                    f.market_prob_h,
                    f.elo_rating_diff,
                    f.home_advantage_trend
                FROM raw_matches r
                INNER JOIN feature_store f ON r.match_id = f.match_id
                ORDER BY r.date, r.match_id
                """
            ).fetchdf()

        if df.empty:
            raise ValueError("No joined training data found in raw_matches and feature_store.")

        df["target"] = TargetResolver.get_label(df, self.target_config)
        df = df.dropna(subset=["odds_h"]).reset_index(drop=True)

        if df.empty:
            raise ValueError("No rows left after dropping records with missing odds.")

        feature_columns = [
            "home_avg_goals_scored",
            "home_avg_goals_conceded",
            "away_avg_goals_scored",
            "away_avg_goals_conceded",
            "is_cold_start",
            "relative_tier_change",
            "market_prob_h",
            "elo_rating_diff",
            "home_advantage_trend",
        ]

        X = df[feature_columns]
        y = df["target"]

        split_index = max(1, int(len(df) * (1 - self.test_size)))
        split_index = min(split_index, len(df) - 1)

        X_train = X.iloc[:split_index].copy()
        X_test = X.iloc[split_index:].copy()
        y_train = y.iloc[:split_index].copy()
        y_test = y.iloc[split_index:].copy()
        test_meta = df.iloc[split_index:][["match_id", "odds_h"]].copy()

        # Coerce features to numeric and ensure missing values are np.nan (XGBoost-compatible).
        X_train = X_train.apply(pd.to_numeric, errors="coerce").astype(float)
        X_test = X_test.apply(pd.to_numeric, errors="coerce").astype(float)
        X_train = X_train.replace({pd.NA: np.nan})
        X_test = X_test.replace({pd.NA: np.nan})

        if not isinstance(self.model, XGBoostModel):
            if X_train.isna().any().any() or X_test.isna().any().any():
                raise ValueError(
                    "Missing values detected in features. "
                    "Current model does not support NaNs; use XGBoost or add imputation."
                )

        if y_train.nunique() < 2:
            raise ValueError("Training split has a single class; cannot train Logistic Regression.")

        return X_train, X_test, y_train, y_test, test_meta

    def train(self) -> tuple[pd.Series, pd.DataFrame, pd.Series]:
        """Train the model and return targets, metadata, and positive-class probabilities."""
        X_train, X_test, y_train, y_test, test_meta = self.prepare_training_data()
        self.model.train(X_train, y_train)
        probabilities = self.model.predict_proba(X_test)
        if probabilities.ndim == 2 and probabilities.shape[1] > 1:
            positive_proba = pd.Series(probabilities[:, 1], index=y_test.index)
        else:
            positive_proba = pd.Series(probabilities.ravel(), index=y_test.index)
        return y_test, test_meta, positive_proba

    def run_pipeline(self, external_run: bool = False) -> Path:
        """Train model, evaluate it, and save a timestamped artifact path."""
        try:
            X_train, X_test, y_train, y_test, test_meta = self.prepare_training_data()

            if isinstance(self.model, XGBoostModel):
                mlflow.xgboost.autolog()
            else:
                mlflow.sklearn.autolog()

            def _run_training() -> Path:
                mlflow.set_tags(self.mlflow_tags)
                mlflow.set_tag("primary_metrics", "roi,win_rate,max_drawdown")
                mlflow.log_param("target_type", self.target_config.get("target_type", "home_win"))
                self.model.train(X_train, y_train)
                probabilities = self.model.predict_proba(X_test)

                if probabilities.ndim == 2 and probabilities.shape[1] > 1:
                    positive_proba = probabilities[:, 1]
                else:
                    positive_proba = probabilities.ravel()

                predictions = (positive_proba >= 0.5).astype(int)
                accuracy = accuracy_score(y_test, predictions)
                precision = precision_score(y_test, predictions, zero_division=0)
                loss = log_loss(y_test, positive_proba, labels=[0, 1])
                target_name = str(self.target_config.get("target_type", "home_win")).strip().lower()
                mlflow.log_metric(f"{target_name}_accuracy", float(accuracy))
                mlflow.log_metric(f"{target_name}_precision", float(precision))
                mlflow.log_metric("log_loss", float(loss))

                LOGGER.info("Accuracy: %.4f", accuracy)
                LOGGER.info("Precision: %.4f", precision)
                LOGGER.info("Log Loss: %.4f", loss)

                predictions_df = pd.DataFrame(
                    {
                        "match_id": test_meta["match_id"].values,
                        "predicted_home_win_prob": positive_proba,
                        "odds_h": test_meta["odds_h"].astype(float).values,
                    }
                )
                backtester = Backtester(
                    initial_bankroll=self.config.settings.initial_bankroll,
                    bet_size=10.0,
                    config_path=str(self.config_path),
                )
                backtester.run_simulation(predictions_df, ev_threshold=0.05)
                backtest_metrics = backtester.get_metrics()
                mlflow.log_metric("roi", float(backtest_metrics.total_roi))
                mlflow.log_metric("win_rate", float(backtest_metrics.win_rate))
                mlflow.log_metric("max_drawdown", float(backtest_metrics.max_drawdown))

                date_tag = datetime.now().strftime("%Y%m%d")
                model_prefix = self.model.__class__.__name__.lower().replace("model", "")
                save_path = self.model_dir / f"{model_prefix}_v1_{date_tag}.joblib"
                self.model.save(str(save_path))
                mlflow.log_artifact(str(save_path))
                return save_path

            if external_run:
                return _run_training()

            with mlflow.start_run():
                return _run_training()
        except duckdb.Error:
            LOGGER.exception("Database failure during model pipeline.")
            raise
        except Exception:
            LOGGER.exception("Model pipeline failed.")
            raise
