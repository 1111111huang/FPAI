"""Model comparison utility for analyzing MLflow runs and identifying best models."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd
import mlflow
from mlflow.entities import Run

from src.utils.logger import get_logger

LOGGER = get_logger(__name__)


class ModelComparison:
    """Analyze MLflow runs and compare model performance across splits and hyperparameters."""

    def __init__(self, experiment_name: str | None = None):
        """Initialize comparison tool with MLflow experiment context."""
        self.experiment_name = experiment_name
        self.client = mlflow.tracking.MlflowClient()

    def get_runs_by_target(self, target_name: str) -> list[Run]:
        """Fetch all MLflow runs for a specific target."""
        if self.experiment_name:
            experiment = self.client.get_experiment_by_name(self.experiment_name)
            if experiment is None:
                LOGGER.warning("Experiment '%s' not found", self.experiment_name)
                return []
            experiment_ids = [experiment.experiment_id]
        else:
            experiment_ids = [experiment.experiment_id for experiment in self.client.search_experiments()]
            if not experiment_ids:
                return []

        runs = self.client.search_runs(
            experiment_ids=experiment_ids,
            filter_string=f"tags.target = '{target_name}'"
        )
        return runs

    def extract_metrics_for_run(self, run: Run) -> dict[str, Any]:
        """Extract train/val/test metrics from a single run."""
        metrics = run.data.metrics or {}
        params = run.data.params or {}
        tags = run.data.tags or {}

        result = {
            "run_id": run.info.run_id,
            "experiment_id": run.info.experiment_id,
            "model_name": tags.get("model_name", "unknown"),
            "model_type": tags.get("model_family", tags.get("model_type", params.get("model_type", "unknown"))),
            "target": tags.get("target", "unknown"),
            "feature_schema_version": tags.get("feature_schema_version", tags.get("feature_version", "v1")),
            "sweep_stage": tags.get("sweep_stage", "unknown"),
        }

        # Extract split-specific metrics
        for split in ["train", "val", "test"]:
            for metric_type in ["log_loss", "accuracy", "mae", "rmse", "precision"]:
                suffix_key = f"{metric_type}_{split}"
                prefix_key = f"{split}_{metric_type}"
                if suffix_key in metrics:
                    result[suffix_key] = metrics[suffix_key]
                    result[prefix_key] = metrics[suffix_key]
                elif prefix_key in metrics:
                    result[suffix_key] = metrics[prefix_key]
                    result[prefix_key] = metrics[prefix_key]
                # Also check for non-split-specific metrics
                elif split == "test" and metric_type in metrics and metric_type not in result:
                    result[suffix_key] = metrics[metric_type]
                    result[prefix_key] = metrics[metric_type]

        for param_name, value in sorted(params.items()):
            if param_name not in result:
                result[f"param_{param_name}"] = value

        return result

    def compare_models_by_target(self, target_name: str) -> pd.DataFrame:
        """Generate a comparison DataFrame for all models trained on a target."""
        runs = self.get_runs_by_target(target_name)
        
        if not runs:
            LOGGER.warning(f"No runs found for target: {target_name}")
            return pd.DataFrame()

        comparison_data = []
        for run in runs:
            row = self.extract_metrics_for_run(run)
            comparison_data.append(row)

        df = pd.DataFrame(comparison_data)
        for metric in ["test_log_loss", "test_mae", "test_rmse", "test_accuracy"]:
            if metric in df.columns:
                ascending = metric != "test_accuracy"
                return df.sort_values(metric, ascending=ascending, na_position="last")
        return df

    def identify_best_model(
        self,
        target_name: str,
        metric_name: str | None = None
    ) -> dict[str, Any] | None:
        """Identify the best-performing model for a target based on a specific metric."""
        comparison_df = self.compare_models_by_target(target_name)

        if metric_name is None:
            metric_name = "test_log_loss" if "test_log_loss" in comparison_df.columns else "test_mae"
        if comparison_df.empty or metric_name not in comparison_df.columns:
            LOGGER.warning(f"Cannot identify best model: metric '{metric_name}' not found")
            return None

        # For loss/error metrics, lower is better
        if "loss" in metric_name or "mae" in metric_name or "rmse" in metric_name:
            best_idx = comparison_df[metric_name].idxmin()
        else:
            best_idx = comparison_df[metric_name].idxmax()

        best_row = comparison_df.loc[best_idx]
        return best_row.to_dict()

    def export_comparison_report(
        self,
        target_name: str,
        output_path: str | Path,
        format: str = "csv"
    ) -> Path:
        """Export comparison report to CSV, JSON, or HTML."""
        comparison_df = self.compare_models_by_target(target_name)
        
        if comparison_df.empty:
            LOGGER.warning(f"No data to export for target: {target_name}")
            return Path(output_path)

        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        if format == "csv":
            output_path = output_path.with_suffix(".csv")
            comparison_df.to_csv(output_path, index=False)
        elif format == "json":
            output_path = output_path.with_suffix(".json")
            comparison_df.to_json(output_path, orient="records", indent=2)
        elif format == "html":
            output_path = output_path.with_suffix(".html")
            comparison_df.to_html(output_path, index=False)
        else:
            raise ValueError(f"Unsupported format: {format}. Use 'csv', 'json', or 'html'.")

        LOGGER.info(f"Comparison report exported to: {output_path}")
        return output_path

    def compare_all_targets(self) -> dict[str, pd.DataFrame]:
        """Compare all targets and return DataFrames for each."""
        if self.experiment_name:
            experiment = self.client.get_experiment_by_name(self.experiment_name)
            if experiment is None:
                return {}
            experiment_ids = [experiment.experiment_id]
        else:
            experiment_ids = [experiment.experiment_id for experiment in self.client.search_experiments()]

        runs = self.client.search_runs(experiment_ids=experiment_ids)
        targets = set()
        for run in runs:
            target = run.data.tags.get("target") if run.data.tags else None
            if target:
                targets.add(target)

        results = {}
        for target in sorted(targets):
            results[target] = self.compare_models_by_target(target)
        
        return results
