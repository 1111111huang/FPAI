"""Model management utilities for training, evaluation, and versioned saving."""

from __future__ import annotations

from datetime import datetime
import json
from pathlib import Path
from tempfile import TemporaryDirectory

import duckdb
import mlflow
import mlflow.sklearn
import mlflow.xgboost
import numpy as np
import pandas as pd
import yaml
from sklearn.metrics import accuracy_score, log_loss, mean_absolute_error, mean_squared_error, precision_score

from src.logic.target_resolver import TargetResolver
from src.logic.target_registry import TargetDefinition, get_target_definition
from src.models.base_model import FPAIBaseModel, XGBoostModel, XGBoostRegressorModel
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
            "target": (target_config or {}).get("target") or (target_config or {}).get("target_type", "home_win"),
        }
        self.target_config = target_config or {"target_type": "home_win"}
        self.target_definition = get_target_definition(
            str(self.target_config.get("target") or self.target_config.get("target_type", "home_win"))
        )
        mlflow.set_experiment("FPAI_Evolution")

    def _load_selected_features(self) -> list[str]:
        schema_path = self.config_path.parent / "config" / "schema.yaml"
        if not schema_path.exists():
            raise FileNotFoundError(f"Missing schema file: {schema_path}")
        with schema_path.open("r", encoding="utf-8") as handle:
            schema = yaml.safe_load(handle) or {}
        training_setup = schema.get("training_setup", {})
        selected = training_setup.get("selected_features")
        if not isinstance(selected, list) or not selected:
            raise ValueError("training_setup.selected_features must be a non-empty list in config/schema.yaml.")
        if not all(isinstance(item, str) and item.strip() for item in selected):
            raise ValueError("training_setup.selected_features must contain only non-empty strings.")
        return [item.strip() for item in selected]

    @staticmethod
    def _log_selected_features(selected_features: list[str]) -> None:
        active_run = mlflow.active_run()
        if active_run is None:
            return
        mlflow.log_param("selected_features", ",".join(selected_features))

    @staticmethod
    def _extract_feature_importance(feature_names: list[str], model: FPAIBaseModel) -> pd.DataFrame:
        estimator = getattr(model, "model", None)
        if estimator is None:
            return pd.DataFrame(columns=["feature", "importance"])
        importances = None
        if hasattr(estimator, "feature_importances_"):
            importances = getattr(estimator, "feature_importances_")
        elif hasattr(estimator, "coef_"):
            coef = getattr(estimator, "coef_")
            try:
                importances = np.abs(coef).ravel()
            except Exception:
                importances = None
        if importances is None:
            return pd.DataFrame(columns=["feature", "importance"])
        values = np.asarray(importances, dtype=float)
        if values.ndim > 1:
            values = np.mean(np.abs(values), axis=0)
        if len(values) != len(feature_names):
            values = values.ravel()[: len(feature_names)]
        return pd.DataFrame(
            {"feature": list(feature_names)[: len(values)], "importance": list(values)}
        ).sort_values("importance", ascending=False)

    @staticmethod
    def _log_feature_importance(feature_names: list[str], model: FPAIBaseModel) -> None:
        active_run = mlflow.active_run()
        if active_run is None:
            return
        df = ModelManager._extract_feature_importance(feature_names, model)
        if df.empty:
            return

        with TemporaryDirectory() as tmpdir:
            out_path = Path(tmpdir) / "feature_importance.csv"
            df.to_csv(out_path, index=False)
            mlflow.log_artifact(str(out_path))
            plot_path = Path(tmpdir) / "feature_importance.png"
            top = df.head(20)
            try:
                import matplotlib.pyplot as plt
            except Exception:
                return
            plt.figure(figsize=(8, 6))
            plt.barh(top["feature"][::-1], top["importance"][::-1])
            plt.title("Top 20 Feature Importances")
            plt.xlabel("Importance")
            plt.tight_layout()
            plt.savefig(plot_path)
            plt.close()
            mlflow.log_artifact(str(plot_path))

    def _build_artifact_metadata(
        self,
        model_path: Path,
        feature_names: list[str],
        X_val: pd.DataFrame,
        y_val: pd.Series,
    ) -> dict[str, object]:
        """Build sidecar metadata for forecast-time diagnostics and intervals."""
        created_at = datetime.utcnow().replace(microsecond=0).isoformat() + "Z"
        metadata: dict[str, object] = {
            "target": self.target_definition.name,
            "task_type": self.target_definition.task_type,
            "classes": list(self.target_definition.classes),
            "model_type": self.model.__class__.__name__,
            "feature_schema_version": self.mlflow_tags.get("feature_version", "v1"),
            "feature_names": feature_names,
            "artifact_path": str(model_path),
            "artifact_name": model_path.name,
            "created_at": created_at,
            "training_cutoff": getattr(self, "training_cutoff", None),
            "primary_metric": self.target_definition.primary_metric,
            "secondary_metrics": list(self.target_definition.secondary_metrics),
        }
        if self.target_definition.task_type == "regression":
            validation_predictions = np.asarray(self.model.predict(X_val), dtype=float)
            residuals = pd.to_numeric(y_val, errors="coerce").astype(float).to_numpy() - validation_predictions
            residuals = residuals[~np.isnan(residuals)]
            if len(residuals):
                metadata["prediction_interval"] = {
                    "coverage": 0.8,
                    "lower_residual": float(np.quantile(residuals, 0.10)),
                    "upper_residual": float(np.quantile(residuals, 0.90)),
                    "method": "validation_residual_quantile",
                }
        feature_importance = self._extract_feature_importance(feature_names, self.model)
        metadata["feature_importance"] = feature_importance.head(50).to_dict(orient="records")
        return metadata

    @staticmethod
    def _write_artifact_metadata(model_path: Path, metadata: dict[str, object]) -> Path:
        metadata_path = model_path.with_suffix(model_path.suffix + ".metadata.json")
        with metadata_path.open("w", encoding="utf-8") as handle:
            json.dump(metadata, handle, indent=2, sort_keys=True)
            handle.write("\n")
        if mlflow.active_run() is not None:
            mlflow.log_artifact(str(metadata_path))
        return metadata_path

    def prepare_training_data(
        self,
    ) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, pd.Series, pd.DataFrame]:
        """Build feature matrix and labels, then apply a chronological 70/15/15 split."""
        feature_columns = self._load_selected_features()
        for feature_name in feature_columns:
            if not feature_name.replace("_", "").isalnum():
                raise ValueError(f"Invalid feature name in selected_features: {feature_name}")
        feature_select = ",\n                    ".join(f"f.{name}" for name in feature_columns)
        label_columns = list(dict.fromkeys(self.target_definition.label_columns))
        label_select = ",\n                    ".join(f"r.{name}" for name in label_columns)
        with self.db_manager.connection() as conn:
            df = conn.execute(
                f"""
                SELECT
                    r.match_id,
                    r.date,
                    r.odds_h,
                    {label_select},
                    {feature_select}
                FROM raw_matches r
                INNER JOIN feature_store f ON r.match_id = f.match_id
                ORDER BY r.date, r.match_id
                """
            ).fetchdf()

        if df.empty:
            raise ValueError("No joined training data found in raw_matches and feature_store.")

        df["target"] = TargetResolver.get_label(df, self.target_config)
        required_non_null = ["target", *feature_columns]
        if self.target_definition.name == "home_win":
            required_non_null.append("odds_h")
        df = df.dropna(subset=required_non_null).reset_index(drop=True)

        if df.empty:
            raise ValueError("No rows left after dropping records with missing labels or features.")

        for feature_name in feature_columns:
            if feature_name not in df.columns:
                raise ValueError(f"Missing selected feature in training data: {feature_name}")

        X = df[feature_columns]
        y = df["target"]

        total = len(df)
        train_ratio = float(self.config.settings.train_split)
        val_ratio = float(self.config.settings.val_split)
        test_ratio = float(self.config.settings.test_split)
        ratio_sum = train_ratio + val_ratio + test_ratio
        if ratio_sum <= 0:
            raise ValueError("Train/val/test split ratios must sum to a positive value.")
        train_ratio = train_ratio / ratio_sum
        val_ratio = val_ratio / ratio_sum
        test_ratio = test_ratio / ratio_sum

        train_end = max(1, int(total * train_ratio))
        val_end = max(train_end + 1, int(total * (train_ratio + val_ratio)))
        val_end = min(val_end, total - 1)
        self.training_cutoff = pd.to_datetime(df.iloc[train_end - 1]["date"]).isoformat()

        X_train = X.iloc[:train_end].copy()
        X_val = X.iloc[train_end:val_end].copy()
        X_test = X.iloc[val_end:].copy()
        y_train = y.iloc[:train_end].copy()
        y_val = y.iloc[train_end:val_end].copy()
        y_test = y.iloc[val_end:].copy()
        test_meta = df.iloc[val_end:][["match_id", "odds_h"]].copy()

        # Coerce features to numeric and ensure missing values are np.nan (XGBoost-compatible).
        X_train = X_train.apply(pd.to_numeric, errors="coerce").astype(float)
        X_val = X_val.apply(pd.to_numeric, errors="coerce").astype(float)
        X_test = X_test.apply(pd.to_numeric, errors="coerce").astype(float)
        X_train = X_train.replace({pd.NA: np.nan})
        X_val = X_val.replace({pd.NA: np.nan})
        X_test = X_test.replace({pd.NA: np.nan})

        if not isinstance(self.model, (XGBoostModel, XGBoostRegressorModel)):
            if X_train.isna().any().any() or X_val.isna().any().any() or X_test.isna().any().any():
                raise ValueError(
                    "Missing values detected in features. "
                    "Current model does not support NaNs; use XGBoost or add imputation."
                )

        if self.target_definition.task_type != "regression" and y_train.nunique() < 2:
            raise ValueError("Training split has a single class; cannot train Logistic Regression.")

        return X_train, X_val, X_test, y_train, y_val, y_test, test_meta

    @staticmethod
    def _positive_probability(probabilities: np.ndarray) -> np.ndarray:
        if probabilities.ndim == 2 and probabilities.shape[1] > 1:
            return probabilities[:, 1]
        return probabilities.ravel()

    @staticmethod
    def _classification_loss(
        y_true: pd.Series,
        probabilities: np.ndarray,
        definition: TargetDefinition,
        model: FPAIBaseModel,
    ) -> float:
        if probabilities.ndim == 1:
            classes = list(range(2)) if definition.name in {"home_win", "btts"} else list(definition.classes)
            return float(log_loss(y_true, probabilities, labels=classes))
        estimator = getattr(model, "model", None)
        classes = getattr(model, "classes_", None)
        if classes is None:
            classes = getattr(estimator, "classes_", None)
        labels = list(classes) if classes is not None else list(definition.classes)
        return float(log_loss(y_true, probabilities, labels=labels))

    def _evaluate_target(
        self,
        X_test: pd.DataFrame,
        y_test: pd.Series,
        X_train: pd.DataFrame | None = None,
        y_train: pd.Series | None = None,
        X_val: pd.DataFrame | None = None,
        y_val: pd.Series | None = None,
    ) -> tuple[dict[str, float], np.ndarray]:
        """Evaluate the configured target with registry-defined metrics.
        
        Optionally evaluate on train/val/test splits with explicit split labels.
        When all splits are provided, logs metrics for all three with '_train', '_val', '_test' suffixes.
        """
        def _compute_metrics(X: pd.DataFrame, y: pd.Series) -> dict[str, float]:
            """Compute metrics for a given split."""
            if self.target_definition.task_type == "regression":
                predictions = np.asarray(self.model.predict(X), dtype=float)
                mae = float(mean_absolute_error(y, predictions))
                mse = float(mean_squared_error(y, predictions))
                rmse = float(np.sqrt(mse))
                return {"mae": mae, "rmse": rmse}

            probabilities = np.asarray(self.model.predict_proba(X))
            predictions = np.asarray(self.model.predict(X))
            accuracy = float(accuracy_score(y, predictions))
            metrics = {
                "log_loss": self._classification_loss(y, probabilities, self.target_definition, self.model),
                "accuracy": accuracy,
            }
            if self.target_definition.name in {"home_win", "btts"}:
                positive = self._positive_probability(probabilities)
                metrics["precision"] = float(precision_score(y, positive >= 0.5, zero_division=0))
            return metrics

        # Evaluate test split (required)
        test_metrics = _compute_metrics(X_test, y_test)

        # If all splits provided, evaluate train and val as well, and log all three
        if X_train is not None and y_train is not None and X_val is not None and y_val is not None:
            train_metrics = _compute_metrics(X_train, y_train)
            val_metrics = _compute_metrics(X_val, y_val)
            
            active_run = mlflow.active_run()
            if active_run is not None:
                for split_name, split_metrics in {
                    "train": train_metrics,
                    "val": val_metrics,
                    "test": test_metrics,
                }.items():
                    for metric_name, value in split_metrics.items():
                        mlflow.log_metric(f"{metric_name}_{split_name}", float(value))
                        mlflow.log_metric(f"{split_name}_{metric_name}", float(value))
        
        # Get predictions for test split
        if self.target_definition.task_type == "regression":
            prediction_output = np.asarray(self.model.predict(X_test), dtype=float)
        else:
            prediction_output = np.asarray(self.model.predict_proba(X_test))
        
        return test_metrics, prediction_output

    def train(self) -> tuple[pd.Series, pd.DataFrame, pd.Series]:
        """Train on the chronological train split, tune on val, and return test predictions."""
        selected_features = self._load_selected_features()
        self._log_selected_features(selected_features)
        X_train, X_val, X_test, y_train, y_val, y_test, test_meta = self.prepare_training_data()
        eval_set = [(X_val, y_val)] if isinstance(self.model, (XGBoostModel, XGBoostRegressorModel)) else None
        self.model.train(X_train, y_train, eval_set=eval_set)
        self._log_feature_importance(list(X_train.columns), self.model)
        if isinstance(self.model, (XGBoostModel, XGBoostRegressorModel)):
            estimator = getattr(self.model, "model", None)
            if estimator is not None:
                best_iter = getattr(estimator, "best_iteration", None)
                if best_iter is not None and mlflow.active_run() is not None:
                    mlflow.log_metric("best_iteration", int(best_iter))
                evals = getattr(estimator, "evals_result_", None)
                if isinstance(evals, dict):
                    logloss_hist = evals.get("validation_0", {}).get("logloss", [])
                    if logloss_hist and mlflow.active_run() is not None:
                        mlflow.log_metric("val_logloss", float(logloss_hist[-1]))
        probabilities = self.model.predict_proba(X_test)
        if probabilities.ndim == 2 and probabilities.shape[1] > 1:
            positive_proba = pd.Series(probabilities[:, 1], index=y_test.index)
        else:
            positive_proba = pd.Series(probabilities.ravel(), index=y_test.index)
        return y_test, test_meta, positive_proba

    def run_pipeline(self, external_run: bool = False) -> Path:
        """Train model, evaluate it, and save a timestamped artifact path."""
        try:
            selected_features = self._load_selected_features()
            self._log_selected_features(selected_features)
            X_train, X_val, X_test, y_train, y_val, y_test, test_meta = self.prepare_training_data()

            if isinstance(self.model, (XGBoostModel, XGBoostRegressorModel)):
                mlflow.xgboost.autolog()
            else:
                mlflow.sklearn.autolog()

            def _run_training() -> Path:
                mlflow.set_tags(self.mlflow_tags)
                mlflow.set_tag("primary_metric", self.target_definition.primary_metric)
                mlflow.set_tag("secondary_metrics", ",".join(self.target_definition.secondary_metrics))
                mlflow.log_param("target_type", self.target_definition.name)
                eval_set = [(X_val, y_val)] if isinstance(self.model, (XGBoostModel, XGBoostRegressorModel)) else None
                self.model.train(X_train, y_train, eval_set=eval_set)
                self._log_feature_importance(list(X_train.columns), self.model)
                if isinstance(self.model, (XGBoostModel, XGBoostRegressorModel)):
                    estimator = getattr(self.model, "model", None)
                    if estimator is not None:
                        best_iter = getattr(estimator, "best_iteration", None)
                        if best_iter is not None:
                            mlflow.log_metric("best_iteration", int(best_iter))
                        evals = getattr(estimator, "evals_result_", None)
                        if isinstance(evals, dict):
                            logloss_hist = evals.get("validation_0", {}).get("logloss", [])
                            if logloss_hist:
                                mlflow.log_metric("val_logloss", float(logloss_hist[-1]))
                target_name = self.target_definition.name
                metrics, prediction_output = self._evaluate_target(X_test, y_test, X_train, y_train, X_val, y_val)
                for metric_name, value in metrics.items():
                    mlflow.log_metric(metric_name, float(value))
                    mlflow.log_metric(f"{target_name}_{metric_name}", float(value))
                    LOGGER.info("%s %s: %.4f", target_name, metric_name, value)

                if self.target_definition.name == "home_win":
                    positive_proba = self._positive_probability(np.asarray(prediction_output))
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
                save_path = self.model_dir / f"{target_name}_{model_prefix}_v1_{date_tag}.joblib"
                self.model.save(str(save_path))
                metadata = self._build_artifact_metadata(save_path, selected_features, X_val, y_val)
                metadata["metrics"] = {metric_name: float(value) for metric_name, value in metrics.items()}
                self._write_artifact_metadata(save_path, metadata)
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
