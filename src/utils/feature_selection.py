"""Feature selection study using stepwise elimination and validation."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import joblib
import mlflow
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, log_loss, mean_absolute_error

from src.logic.target_registry import get_target_definition
from src.models.model_factory import ModelFactory
from src.models.base_model import FPAIBaseModel, XGBoostModel, XGBoostRegressorModel
from src.utils.logger import get_logger

LOGGER = get_logger(__name__)


class FeatureSelectionStudy:
    """Run stepwise feature selection experiments with train/val/test evaluation."""

    def __init__(
        self,
        target_name: str,
        model_type: str,
        config_path: str | Path = "config.yaml",
        output_dir: str | Path = "reports",
    ) -> None:
        """Initialize feature selection study.
        
        Args:
            target_name: Forecast target (e.g., 'result_3way')
            model_type: Model class name (e.g., 'LogisticRegression', 'RandomForestRegressor')
            config_path: Path to config.yaml
            output_dir: Directory for output reports
        """
        self.target_name = target_name
        self.model_type = model_type
        self.config_path = Path(config_path)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.target_definition = get_target_definition(target_name)
        self.results: list[dict[str, Any]] = []

    def run_feature_subset_study(
        self,
        feature_ranking: list[str],
        feature_subsets: list[int] | None = None,
        X_train: pd.DataFrame | None = None,
        X_val: pd.DataFrame | None = None,
        X_test: pd.DataFrame | None = None,
        y_train: pd.Series | None = None,
        y_val: pd.Series | None = None,
        y_test: pd.Series | None = None,
        hyperparams: dict[str, Any] | None = None,
        log_to_mlflow: bool = False,
    ) -> pd.DataFrame:
        """Run experiments with incremental feature subsets.
        
        Args:
            feature_ranking: List of feature names ranked by importance
            feature_subsets: List of subset sizes to test (e.g., [10, 20, 30, all])
                If None, uses [10, 20, 30, 40, all]
            X_train, X_val, X_test: Feature matrices for each split
            y_train, y_val, y_test: Target labels for each split
            hyperparams: Hyperparameters for model creation
            log_to_mlflow: Whether to log results to MLflow
        
        Returns:
            DataFrame with results for each feature subset
        """
        if feature_subsets is None:
            feature_subsets = [10, 20, 30, 40, len(feature_ranking)]
        
        # Ensure 'all' is included
        max_features = len(feature_ranking)
        feature_subsets = sorted(set([min(n, max_features) for n in feature_subsets]))
        
        hyperparams = hyperparams or {}
        results = []
        
        for n_features in feature_subsets:
            if n_features > len(feature_ranking):
                n_features = len(feature_ranking)
            
            # Select top N features
            selected_features = feature_ranking[:n_features]
            LOGGER.info(f"Testing with {n_features} features ({len(selected_features)} selected)")
            
            # Prepare feature subsets
            X_train_sub = X_train[selected_features] if X_train is not None else None
            X_val_sub = X_val[selected_features] if X_val is not None else None
            X_test_sub = X_test[selected_features] if X_test is not None else None
            
            # Create and train model
            try:
                model = ModelFactory.get_model(self.model_type, params=hyperparams)
                
                # Train on train split
                if isinstance(model, (XGBoostModel, XGBoostRegressorModel)):
                    model.train(X_train_sub, y_train, eval_set=[(X_val_sub, y_val)])
                else:
                    model.train(X_train_sub, y_train)
                
                # Evaluate on all splits
                metrics = self._evaluate_model(
                    model=model,
                    X_train=X_train_sub,
                    X_val=X_val_sub,
                    X_test=X_test_sub,
                    y_train=y_train,
                    y_val=y_val,
                    y_test=y_test,
                )
                
                metrics["n_features"] = n_features
                metrics["n_features_selected"] = len(selected_features)
                metrics["top_features"] = ",".join(selected_features)
                
                results.append(metrics)
                
                # Log to MLflow if active
                if log_to_mlflow and mlflow.active_run() is not None:
                    mlflow.log_param(f"features_n_{n_features}", len(selected_features))
                    for key, value in metrics.items():
                        if isinstance(value, (int, float)):
                            mlflow.log_metric(f"features_{n_features}_{key}", value)
                
            except Exception as e:
                LOGGER.error(f"Failed to train with {n_features} features: {e}")
                continue
        
        results_df = pd.DataFrame(results)
        self.results = results
        return results_df

    def _evaluate_model(
        self,
        model: FPAIBaseModel,
        X_train: pd.DataFrame,
        X_val: pd.DataFrame,
        X_test: pd.DataFrame,
        y_train: pd.Series,
        y_val: pd.Series,
        y_test: pd.Series,
    ) -> dict[str, float]:
        """Evaluate model on train/val/test splits."""
        metrics = {}
        
        if self.target_definition.task_type == "regression":
            for split_name, X, y in [
                ("train", X_train, y_train),
                ("val", X_val, y_val),
                ("test", X_test, y_test),
            ]:
                preds = model.predict(X)
                mae = float(mean_absolute_error(y, preds))
                metrics[f"mae_{split_name}"] = mae
        else:
            for split_name, X, y in [
                ("train", X_train, y_train),
                ("val", X_val, y_val),
                ("test", X_test, y_test),
            ]:
                proba = model.predict_proba(X)
                preds = model.predict(X)
                
                # Get classes
                classes = getattr(model, "classes_", None)
                if classes is None:
                    classes = list(self.target_definition.classes)
                
                loss = float(log_loss(y, proba, labels=list(classes)))
                acc = float(accuracy_score(y, preds))
                
                metrics[f"log_loss_{split_name}"] = loss
                metrics[f"accuracy_{split_name}"] = acc
        
        return metrics

    def save_results(
        self,
        results_df: pd.DataFrame,
        filename: str | None = None,
    ) -> Path:
        """Save feature selection results to CSV.
        
        Args:
            results_df: DataFrame from run_feature_subset_study()
            filename: Output filename (default: feature_selection_{target}.csv)
        
        Returns:
            Path to saved file
        """
        if filename is None:
            filename = f"feature_selection_{self.target_name}.csv"
        
        output_path = self.output_dir / filename
        results_df.to_csv(output_path, index=False)
        LOGGER.info(f"Saved feature selection results: {output_path}")
        return output_path

    def recommend_feature_set(
        self,
        results_df: pd.DataFrame,
        improvement_threshold: float = 0.01,
    ) -> dict[str, Any]:
        """Recommend optimal feature set based on improvement over baseline.
        
        Args:
            results_df: DataFrame from run_feature_subset_study()
            improvement_threshold: Minimum relative improvement to justify additional features
        
        Returns:
            Dictionary with recommendation and analysis
        """
        if results_df.empty:
            return {"recommendation": "no_data", "reason": "Empty results"}
        
        if self.target_definition.task_type == "regression":
            metric_col = "mae_test"
        else:
            metric_col = "log_loss_test"
        
        if metric_col not in results_df.columns:
            return {"recommendation": "no_data", "reason": f"Missing metric column {metric_col}"}
        
        # Get baseline (smallest feature set)
        baseline_row = results_df.iloc[0]
        baseline_metric = baseline_row[metric_col]
        
        # Find knee point: where adding features stops improving metric significantly
        best_n = baseline_row["n_features"]
        best_metric = baseline_metric
        
        for _, row in results_df.iloc[1:].iterrows():
            current_metric = row[metric_col]
            improvement = abs(baseline_metric - current_metric) / baseline_metric
            
            if improvement > improvement_threshold:
                if self.target_definition.task_type == "regression":
                    if current_metric < best_metric:
                        best_metric = current_metric
                        best_n = row["n_features"]
                else:
                    if current_metric < best_metric:
                        best_metric = current_metric
                        best_n = row["n_features"]
            else:
                break
        
        recommended_row = results_df[results_df["n_features"] == best_n].iloc[0]
        
        return {
            "recommendation": int(best_n),
            "metric_baseline": float(baseline_row[metric_col]),
            "metric_recommended": float(recommended_row[metric_col]),
            "improvement_pct": float(
                abs(baseline_row[metric_col] - recommended_row[metric_col]) 
                / baseline_row[metric_col] * 100
            ),
            "selected_features": recommended_row["top_features"].split(","),
        }


class BatchFeatureSelection:
    """Run feature selection across multiple targets and model types."""

    def __init__(
        self,
        targets: list[str],
        model_types: dict[str, str],
        config_path: str | Path = "config.yaml",
        output_dir: str | Path = "reports",
    ) -> None:
        """Initialize batch feature selection.
        
        Args:
            targets: List of target names
            model_types: Mapping of target -> model type
            config_path: Path to config.yaml
            output_dir: Output directory
        """
        self.targets = targets
        self.model_types = model_types
        self.config_path = Path(config_path)
        self.output_dir = Path(output_dir)

    def run_batch(
        self,
        importance_rankings: dict[str, list[str]],
        training_data: dict[str, dict[str, Any]],
    ) -> dict[str, pd.DataFrame]:
        """Run feature selection for all target/model combinations.
        
        Args:
            importance_rankings: Mapping of target -> feature ranking list
            training_data: Mapping of target -> {X_train, X_val, X_test, y_train, y_val, y_test}
        
        Returns:
            Mapping of target -> results DataFrame
        """
        all_results = {}
        
        for target_name in self.targets:
            if target_name not in importance_rankings:
                LOGGER.warning(f"No importance ranking for {target_name}")
                continue
            
            if target_name not in training_data:
                LOGGER.warning(f"No training data for {target_name}")
                continue
            
            model_type = self.model_types.get(target_name)
            if not model_type:
                LOGGER.warning(f"No model type defined for {target_name}")
                continue
            
            LOGGER.info(f"Running feature selection for {target_name} ({model_type})")
            
            study = FeatureSelectionStudy(
                target_name=target_name,
                model_type=model_type,
                config_path=self.config_path,
                output_dir=self.output_dir,
            )
            
            data = training_data[target_name]
            ranking = importance_rankings[target_name]
            
            results_df = study.run_feature_subset_study(
                feature_ranking=ranking,
                X_train=data["X_train"],
                X_val=data["X_val"],
                X_test=data["X_test"],
                y_train=data["y_train"],
                y_val=data["y_val"],
                y_test=data["y_test"],
            )
            
            study.save_results(results_df)
            all_results[target_name] = results_df
        
        return all_results
