"""Diagnostics for target models and chronological evaluation splits."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.calibration import calibration_curve
from sklearn.metrics import mean_absolute_error, mean_squared_error

from src.logic.target_registry import TargetDefinition, get_target_definition
from src.models import LRModel, ModelFactory, ModelManager, XGBoostModel, XGBoostRegressorModel


class _LoadedEstimatorModel:
    """Adapter for persisted sklearn estimators used by diagnostics."""

    def __init__(self, estimator: Any) -> None:
        self.model = estimator

    def train(self, X: Any, y: Any, eval_set: Any | None = None) -> None:
        raise RuntimeError("Loaded diagnostic models are not trainable.")

    def predict_proba(self, X: Any) -> np.ndarray:
        return np.asarray(self.model.predict_proba(X))

    def predict(self, X: Any) -> np.ndarray:
        return np.asarray(self.model.predict(X))

    def save(self, path: str) -> None:
        raise RuntimeError("Loaded diagnostic models cannot be saved.")

    @classmethod
    def load(cls, path: str) -> "_LoadedEstimatorModel":
        return cls(joblib.load(path))


def _model_type_from_path(model_path: Path, definition: TargetDefinition) -> str:
    """Infer the local wrapper type from artifact naming conventions."""
    name = model_path.name.lower()
    if "xgboostregressor" in name or "xgb_regressor" in name or "xgboost_regressor" in name:
        return "xgboost_regressor"
    if "xgboost" in name or "_xgb" in name:
        return "xgboost"
    if definition.task_type == "regression" and ("randomforestregressor" in name or "rf_regressor" in name):
        return "rf_regressor"
    if definition.task_type == "regression":
        return "rf_regressor"
    return "lr"


def load_model_for_diagnostics(model_path: str | Path, target_name: str):
    """Load a persisted model artifact into the project wrapper interface."""
    path = Path(model_path)
    definition = get_target_definition(target_name)
    model_type = _model_type_from_path(path, definition)
    if model_type == "xgboost":
        return XGBoostModel.load(str(path))
    if model_type == "xgboost_regressor":
        return XGBoostRegressorModel.load(str(path))
    try:
        return _LoadedEstimatorModel.load(str(path))
    except Exception:
        model = ModelFactory.get_model(model_type)
        model.model = joblib.load(path)
        return model


class EvaluationDiagnostics:
    """Generate diagnostic reports for target predictions."""

    def __init__(self, target_name: str) -> None:
        self.definition = get_target_definition(target_name)

    @staticmethod
    def _finite_float(value: float) -> float | None:
        return float(value) if np.isfinite(value) else None

    def residual_summary(
        self,
        y_true: pd.Series,
        predictions: np.ndarray,
        meta: pd.DataFrame | None = None,
    ) -> dict[str, Any]:
        """Summarize regression residuals overall and by odds bins when available."""
        residuals = pd.Series(np.asarray(y_true, dtype=float) - np.asarray(predictions, dtype=float))
        summary: dict[str, Any] = {
            "mean_residual": self._finite_float(float(residuals.mean())),
            "median_residual": self._finite_float(float(residuals.median())),
            "mae": float(mean_absolute_error(y_true, predictions)),
            "rmse": float(np.sqrt(mean_squared_error(y_true, predictions))),
        }
        if meta is not None and "odds_h" in meta.columns:
            frame = pd.DataFrame({"residual": residuals.values, "odds_h": pd.to_numeric(meta["odds_h"], errors="coerce")})
            frame = frame.dropna()
            if len(frame) >= 4 and frame["odds_h"].nunique() >= 2:
                frame["odds_bin"] = pd.qcut(frame["odds_h"], q=min(4, frame["odds_h"].nunique()), duplicates="drop")
                by_bin = {}
                for bin_name, group in frame.groupby("odds_bin", observed=True):
                    by_bin[str(bin_name)] = {
                        "count": int(len(group)),
                        "mean_residual": self._finite_float(float(group["residual"].mean())),
                        "mae": float(group["residual"].abs().mean()),
                    }
                summary["by_home_odds_bin"] = by_bin
        return summary

    def calibration_summary(self, y_true: pd.Series, probabilities: np.ndarray, bins: int = 10) -> dict[str, Any]:
        """Build classifier calibration points for binary targets."""
        if self.definition.task_type != "binary_classification":
            return {"available": False, "reason": "calibration_curve currently applies to binary targets"}
        positive = probabilities[:, 1] if probabilities.ndim == 2 and probabilities.shape[1] > 1 else probabilities.ravel()
        prob_true, prob_pred = calibration_curve(y_true, positive, n_bins=bins, strategy="uniform")
        return {
            "available": True,
            "bins": [
                {"mean_predicted_probability": float(pred), "observed_frequency": float(true)}
                for pred, true in zip(prob_pred, prob_true)
            ],
        }

    def prediction_interval_coverage(
        self,
        y_true: pd.Series,
        predictions: np.ndarray,
        residual_quantiles: dict[str, float] | None,
    ) -> dict[str, Any]:
        """Validate residual-quantile interval coverage when metadata is available."""
        if not residual_quantiles:
            return {"available": False, "reason": "model metadata did not include residual quantiles"}
        lower_q = residual_quantiles.get("lower")
        upper_q = residual_quantiles.get("upper")
        if lower_q is None or upper_q is None:
            return {"available": False, "reason": "residual quantiles missing lower/upper values"}
        lower = np.asarray(predictions, dtype=float) + float(lower_q)
        upper = np.asarray(predictions, dtype=float) + float(upper_q)
        covered = (np.asarray(y_true, dtype=float) >= lower) & (np.asarray(y_true, dtype=float) <= upper)
        return {
            "available": True,
            "coverage": float(np.mean(covered)),
            "mean_width": float(np.mean(upper - lower)),
        }

    def build_report(
        self,
        y_true: pd.Series,
        prediction_output: np.ndarray,
        meta: pd.DataFrame | None = None,
        residual_quantiles: dict[str, float] | None = None,
    ) -> dict[str, Any]:
        """Return a JSON-serializable diagnostics report."""
        if self.definition.task_type == "regression":
            predictions = np.asarray(prediction_output, dtype=float)
            return {
                "target": self.definition.name,
                "task_type": self.definition.task_type,
                "residuals": self.residual_summary(y_true, predictions, meta),
                "prediction_interval_coverage": self.prediction_interval_coverage(y_true, predictions, residual_quantiles),
            }
        probabilities = np.asarray(prediction_output)
        return {
            "target": self.definition.name,
            "task_type": self.definition.task_type,
            "calibration": self.calibration_summary(y_true, probabilities),
        }


def _load_residual_quantiles(model_path: Path) -> dict[str, float] | None:
    metadata_path = model_path.with_suffix(model_path.suffix + ".metadata.json")
    if not metadata_path.exists():
        return None
    with metadata_path.open("r", encoding="utf-8") as handle:
        metadata = json.load(handle)
    interval = metadata.get("prediction_interval", {})
    quantiles = interval.get("residual_quantiles")
    return quantiles if isinstance(quantiles, dict) else None


def run_diagnostics(
    target_name: str,
    model_path: str | Path,
    output_path: str | Path = "reports/diagnostics.json",
    competition_id: str = "E0",
) -> Path:
    """Load a model artifact, evaluate the test split, and write diagnostics JSON.

    US#131: `competition_id` was previously not accepted at all -- ModelManager
    silently defaulted to "E0", so diagnosing any non-E0 artifact (e.g. a
    Sweden model) used E0's feature list and E0's training rows instead of the
    artifact's own. Pass the artifact's actual training context explicitly.
    """
    model_file = Path(model_path)
    model = load_model_for_diagnostics(model_file, target_name)
    manager = ModelManager(model=model, target_config={"target": target_name}, competition_id=competition_id)
    X_train, X_val, X_test, y_train, y_val, y_test, test_meta = manager.prepare_training_data()
    metrics, prediction_output = manager._evaluate_target(X_test, y_test, X_train, y_train, X_val, y_val)

    diagnostics = EvaluationDiagnostics(target_name)
    report = diagnostics.build_report(
        y_test,
        prediction_output,
        test_meta,
        residual_quantiles=_load_residual_quantiles(model_file),
    )
    report["metrics"] = metrics
    report["model_path"] = str(model_file)

    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2, sort_keys=True)
    return destination
