"""Systematic target sweep runner with MLflow logging."""

from __future__ import annotations

import itertools
from pathlib import Path
from typing import Any

import mlflow
import mlflow.sklearn
import mlflow.xgboost
import optuna
import pandas as pd
import yaml

optuna.logging.set_verbosity(optuna.logging.WARNING)

from src.logic.target_registry import get_target_definition
from src.models import ModelFactory, ModelManager, XGBoostModel, XGBoostRegressorModel
from src.utils.logger import get_logger

LOGGER = get_logger(__name__)


def iter_grid_params(grid: dict[str, list[Any]]) -> list[dict[str, Any]]:
    """Expand a parameter grid into stable cartesian-product dictionaries."""
    keys = list(grid.keys())
    return [dict(zip(keys, values)) for values in itertools.product(*(grid[key] for key in keys))]


def forecast_experiment_name(target_name: str, model_type: str, stage: str, version: str) -> str:
    """Return the standard MLflow experiment name for a target sweep."""
    return f"FPAI_{target_name}_{model_type}_{stage}_{version}"


def mlflow_flavor_for_model_type(model_type: str) -> str:
    """Return the MLflow flavor used for a model family."""
    normalized = model_type.strip().lower()
    if normalized in {"xgb", "xgboost", "xgb_regressor", "xgboost_regressor"}:
        return "xgboost"
    return "sklearn"


def log_model_compat(model: Any, flavor: str, name: str = "model") -> None:
    """Log MLflow model across old/new MLflow argument names."""
    if flavor == "xgboost":
        try:
            mlflow.xgboost.log_model(model, name=name)
            return
        except TypeError:
            mlflow.xgboost.log_model(model, name)
            return
    if flavor == "sklearn":
        try:
            mlflow.sklearn.log_model(model, name=name)
            return
        except TypeError:
            mlflow.sklearn.log_model(model, name)
            return
    raise ValueError(f"Unsupported MLflow model flavor: {flavor}")


class SweepRunner:
    """Run target-aware grid sweeps with consistent tags and split metrics."""

    def __init__(
        self,
        target_name: str,
        config_path: str | Path,
        experiment_name: str | None = None,
        max_runs: int | None = None,
        sweep_stage: str | None = None,
    ) -> None:
        self.definition = get_target_definition(target_name)
        self.config_path = Path(config_path)
        self.experiment_name = experiment_name
        self.max_runs = max_runs
        self.sweep_stage = sweep_stage

    def _load_config(self) -> dict[str, Any]:
        if not self.config_path.exists():
            raise FileNotFoundError(f"Experiment config not found: {self.config_path}")
        with self.config_path.open("r", encoding="utf-8") as handle:
            config = yaml.safe_load(handle) or {}
        if self.sweep_stage is not None:
            config["sweep_stage"] = self.sweep_stage
        return config

    @staticmethod
    def _validate_grid(grid: Any) -> dict[str, list[Any]]:
        if not isinstance(grid, dict) or not grid:
            raise ValueError("Experiment config must contain a non-empty grid_search mapping.")
        for key, values in grid.items():
            if not isinstance(values, list) or not values:
                raise ValueError(f"grid_search.{key} must be a non-empty list.")
        return grid

    def _merged_params(self, model_type: str, fixed_params: dict[str, Any], params: dict[str, Any]) -> dict[str, Any]:
        merged = {**fixed_params, **params}
        if model_type in {"xgb", "xgboost"}:
            if self.definition.task_type == "multiclass_classification":
                merged["objective"] = "multi:softprob"
                merged["eval_metric"] = "mlogloss"
                merged["num_class"] = len(self.definition.classes)
            elif self.definition.task_type == "binary_classification":
                merged["objective"] = "binary:logistic"
                merged["eval_metric"] = "logloss"
        elif model_type in {"xgb_regressor", "xgboost_regressor"}:
            merged.setdefault("objective", "reg:squarederror")
            merged.setdefault("eval_metric", "rmse")
        return merged

    def run(self) -> list[dict[str, object]]:
        """Execute the configured sweep and return run metric summaries."""
        config = self._load_config()
        model_type = str(config.get("model_type", "lr")).strip().lower()
        grid = self._validate_grid(config.get("grid_search", {}))
        fixed_params = config.get("fixed_params", {})
        if not isinstance(fixed_params, dict):
            raise ValueError("fixed_params must be a mapping when provided.")

        stage = str(config.get("sweep_stage", "broad")).strip().lower()
        version = str(config.get("version", "v1")).strip().lower()
        resolved_experiment_name = self.experiment_name or forecast_experiment_name(
            self.definition.name,
            model_type,
            stage,
            version,
        )
        mlflow.set_experiment(resolved_experiment_name)
        run_params = iter_grid_params(grid)
        if self.max_runs is not None:
            run_params = run_params[: max(0, int(self.max_runs))]

        LOGGER.info(
            "Running target sweep | experiment=%s | target=%s | model=%s | runs=%s",
            resolved_experiment_name,
            self.definition.name,
            model_type,
            len(run_params),
        )

        results: list[dict[str, object]] = []
        for index, params in enumerate(run_params, start=1):
            merged_params = self._merged_params(model_type, fixed_params, params)
            run_name = f"{self.definition.name}_{model_type}_{stage}_{index:04d}"
            # Re-assert experiment before each run: ModelManager.__init__ resets
            # mlflow.set_experiment("FPAI_Evolution") which would send subsequent
            # runs to the wrong experiment.
            mlflow.set_experiment(resolved_experiment_name)
            with mlflow.start_run(run_name=run_name):
                mlflow.log_dict(config, "experiment_config.yaml")
                mlflow.log_params(merged_params)
                mlflow.log_param("target", self.definition.name)
                mlflow.log_param("model_type", model_type)
                mlflow.set_tags(
                    {
                        "target": self.definition.name,
                        "task_type": self.definition.task_type,
                        "model_family": model_type,
                        "feature_schema_version": str(config.get("feature_schema_version", "v1")),
                        "split_policy": "chronological_70_15_15",
                        "league": str(config.get("league", "all")),
                        "sweep_stage": stage,
                        "experiment_version": version,
                    }
                )
                extra_tags = config.get("tags", {})
                if isinstance(extra_tags, dict):
                    mlflow.set_tags({str(key): str(value) for key, value in extra_tags.items()})

                model = ModelFactory.get_model(model_type, merged_params)
                manager = ModelManager(
                    model=model,
                    league_tier=str(config.get("league_tier", "all")),
                    test_season=str(config.get("test_season", "time_split")),
                    feature_version=str(config.get("feature_schema_version", "v1")),
                    target_config={"target": self.definition.name},
                )
                X_train, X_val, X_test, y_train, y_val, y_test, _ = manager.prepare_training_data()
                eval_set = [(X_val, y_val)] if isinstance(model, (XGBoostModel, XGBoostRegressorModel)) else None
                model.train(X_train, y_train, eval_set=eval_set)
                manager._log_feature_importance(list(X_train.columns), model)
                metrics, _ = manager._evaluate_target(X_test, y_test, X_train, y_train, X_val, y_val)
                mlflow.log_metrics({metric_name: float(value) for metric_name, value in metrics.items()})
                mlflow.set_tag("training_cutoff", getattr(manager, "training_cutoff", ""))
                log_model_compat(model.model, mlflow_flavor_for_model_type(model_type), name="model")
                results.append({**params, **metrics})

        if results:
            primary_metric = self.definition.primary_metric
            ascending = primary_metric in {"log_loss", "mae", "rmse"}
            summary_df = pd.DataFrame(results).sort_values(primary_metric, ascending=ascending).head(5)
            LOGGER.info("Top 5 parameter sets by %s:\n%s", primary_metric, summary_df.to_string(index=False))
        return results


def _suggest_param(trial: optuna.Trial, name: str, spec: Any) -> Any:
    """Suggest a single hyperparameter from a spec.

    Spec formats:
    - list  → categorical choice
    - {"type": "float", "low": x, "high": y, "log": bool}
    - {"type": "int",   "low": x, "high": y}
    """
    if isinstance(spec, list):
        return trial.suggest_categorical(name, spec)
    if isinstance(spec, dict):
        kind = spec.get("type", "float")
        if kind == "float":
            return trial.suggest_float(name, spec["low"], spec["high"], log=spec.get("log", False))
        if kind == "int":
            return trial.suggest_int(name, spec["low"], spec["high"])
    raise ValueError(f"Unsupported Optuna param spec for '{name}': {spec!r}")


class OptunaRunner:
    """Bayesian hyperparameter sweep using Optuna TPE sampler.

    Config YAML keys (in addition to the keys SweepRunner reads):
      optuna_search: dict[str, list | spec_dict]  — search space
      n_trials: int                               — number of trials (default 50)
      direction: "minimize" | "maximize"          — default inferred from primary metric
    """

    def __init__(
        self,
        target_name: str,
        config_path: str | Path,
        experiment_name: str | None = None,
        n_trials: int | None = None,
        sweep_stage: str | None = None,
    ) -> None:
        self.definition = get_target_definition(target_name)
        self.config_path = Path(config_path)
        self.experiment_name = experiment_name
        self.n_trials_override = n_trials
        self.sweep_stage = sweep_stage

    def _load_config(self) -> dict[str, Any]:
        if not self.config_path.exists():
            raise FileNotFoundError(f"Experiment config not found: {self.config_path}")
        with self.config_path.open("r", encoding="utf-8") as handle:
            config = yaml.safe_load(handle) or {}
        if self.sweep_stage is not None:
            config["sweep_stage"] = self.sweep_stage
        return config

    def _merged_params(self, model_type: str, fixed_params: dict[str, Any], params: dict[str, Any]) -> dict[str, Any]:
        merged = {**fixed_params, **params}
        if model_type in {"xgb", "xgboost"}:
            if self.definition.task_type == "multiclass_classification":
                merged["objective"] = "multi:softprob"
                merged["eval_metric"] = "mlogloss"
                merged["num_class"] = len(self.definition.classes)
            elif self.definition.task_type == "binary_classification":
                merged["objective"] = "binary:logistic"
                merged["eval_metric"] = "logloss"
        elif model_type in {"xgb_regressor", "xgboost_regressor"}:
            merged.setdefault("objective", "reg:squarederror")
            merged.setdefault("eval_metric", "rmse")
        return merged

    def run(self) -> list[dict[str, object]]:
        """Execute the Optuna sweep and return trial metric summaries."""
        config = self._load_config()
        model_type = str(config.get("model_type", "lr")).strip().lower()
        search_space: dict[str, Any] = config.get("optuna_search", config.get("grid_search", {}))
        if not search_space:
            raise ValueError("Experiment config must contain 'optuna_search' or 'grid_search'.")
        fixed_params = config.get("fixed_params", {}) or {}
        n_trials = self.n_trials_override or int(config.get("n_trials", 50))

        stage = str(config.get("sweep_stage", "optuna")).strip().lower()
        version = str(config.get("version", "v1")).strip().lower()
        resolved_experiment_name = self.experiment_name or forecast_experiment_name(
            self.definition.name, model_type, stage, version
        )
        primary_metric = self.definition.primary_metric
        direction = config.get("direction") or ("minimize" if primary_metric in {"log_loss", "mae", "rmse"} else "maximize")

        LOGGER.info(
            "Optuna sweep | experiment=%s | target=%s | model=%s | trials=%d | metric=%s | direction=%s",
            resolved_experiment_name, self.definition.name, model_type, n_trials, primary_metric, direction,
        )

        results: list[dict[str, object]] = []

        def objective(trial: optuna.Trial) -> float:
            params = {name: _suggest_param(trial, name, spec) for name, spec in search_space.items()}
            merged_params = self._merged_params(model_type, fixed_params, params)
            run_name = f"{self.definition.name}_{model_type}_{stage}_{trial.number:04d}"
            mlflow.set_experiment(resolved_experiment_name)
            with mlflow.start_run(run_name=run_name):
                mlflow.log_dict(config, "experiment_config.yaml")
                mlflow.log_params(merged_params)
                mlflow.log_param("target", self.definition.name)
                mlflow.log_param("model_type", model_type)
                mlflow.log_param("optuna_trial", trial.number)
                mlflow.set_tags(
                    {
                        "target": self.definition.name,
                        "task_type": self.definition.task_type,
                        "model_family": model_type,
                        "feature_schema_version": str(config.get("feature_schema_version", "v1")),
                        "split_policy": "chronological_70_15_15",
                        "league": str(config.get("league", "all")),
                        "sweep_stage": stage,
                        "experiment_version": version,
                        "sampler": "optuna_tpe",
                    }
                )
                extra_tags = config.get("tags", {})
                if isinstance(extra_tags, dict):
                    mlflow.set_tags({str(k): str(v) for k, v in extra_tags.items()})

                model = ModelFactory.get_model(model_type, merged_params)
                manager = ModelManager(
                    model=model,
                    league_tier=str(config.get("league_tier", "all")),
                    test_season=str(config.get("test_season", "time_split")),
                    feature_version=str(config.get("feature_schema_version", "v1")),
                    target_config={"target": self.definition.name},
                )
                X_train, X_val, X_test, y_train, y_val, y_test, _ = manager.prepare_training_data()
                eval_set = [(X_val, y_val)] if isinstance(model, (XGBoostModel, XGBoostRegressorModel)) else None
                model.train(X_train, y_train, eval_set=eval_set)
                manager._log_feature_importance(list(X_train.columns), model)
                metrics, _ = manager._evaluate_target(X_test, y_test, X_train, y_train, X_val, y_val)
                mlflow.log_metrics({k: float(v) for k, v in metrics.items()})
                mlflow.set_tag("training_cutoff", getattr(manager, "training_cutoff", ""))
                log_model_compat(model.model, mlflow_flavor_for_model_type(model_type), name="model")
                results.append({**params, **metrics})
                return float(metrics.get(primary_metric, float("nan")))

        study = optuna.create_study(
            direction=direction,
            sampler=optuna.samplers.TPESampler(seed=42),
        )
        study.optimize(objective, n_trials=n_trials, show_progress_bar=False)

        if results:
            ascending = direction == "minimize"
            summary_df = pd.DataFrame(results).sort_values(primary_metric, ascending=ascending).head(5)
            LOGGER.info("Top 5 Optuna trials by %s:\n%s", primary_metric, summary_df.to_string(index=False))
        return results
