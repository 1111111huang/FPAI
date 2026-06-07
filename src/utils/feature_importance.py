"""Permutation importance analysis for trained models."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.inspection import permutation_importance

from src.logic.target_registry import get_target_definition
from src.models.base_model import FPAIBaseModel, XGBoostModel, XGBoostRegressorModel
from src.utils.logger import get_logger

LOGGER = get_logger(__name__)


class PermutationImportanceAnalyzer:
    """Compute and report permutation importance for models on validation/test data."""

    def __init__(
        self,
        model_path: str | Path,
        target_name: str,
        output_dir: str | Path = "reports",
    ) -> None:
        """Initialize analyzer with a trained model artifact.
        
        Args:
            model_path: Path to joblib model file
            target_name: Name of the forecast target (e.g., 'result_3way', 'total_goals')
            output_dir: Directory to save importance reports
        """
        self.model_path = Path(model_path)
        self.target_name = target_name
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.target_definition = get_target_definition(target_name)
        
        if not self.model_path.exists():
            raise FileNotFoundError(f"Model not found: {self.model_path}")
        
        if self.target_definition.task_type == "regression":
            self.model: FPAIBaseModel = XGBoostRegressorModel.load(str(self.model_path))
        else:
            try:
                self.model = XGBoostModel.load(str(self.model_path))
            except Exception:
                self.model = joblib.load(str(self.model_path))
        LOGGER.info("Loaded model from %s", self.model_path)

    def compute_importance(
        self,
        X_eval: pd.DataFrame,
        y_eval: pd.Series,
        n_repeats: int = 10,
        random_state: int = 42,
    ) -> pd.DataFrame:
        """Compute permutation importance on evaluation data.
        
        Args:
            X_eval: Features for evaluation (validation or test split)
            y_eval: Target labels for evaluation
            n_repeats: Number of times to permute each feature
            random_state: Random seed for reproducibility
        
        Returns:
            DataFrame with columns: feature, importance_mean, importance_std, importance_pct
        """
        # Get the underlying sklearn estimator
        estimator = getattr(self.model, "model", self.model)
        if isinstance(self.model, (XGBoostModel, XGBoostRegressorModel)):
            estimator = getattr(self.model, "model", self.model)
        
        # Prepare scoring function based on task type
        if self.target_definition.task_type == "regression":
            scoring = "neg_mean_absolute_error"
            
            def score_func(model: Any, X: pd.DataFrame, y: pd.Series) -> float:
                from sklearn.metrics import mean_absolute_error
                preds = model.predict(X)
                return float(-mean_absolute_error(y, preds))
        else:
            scoring = "neg_log_loss"
            
            def score_func(model: Any, X: pd.DataFrame, y: pd.Series) -> float:
                from sklearn.metrics import log_loss
                proba = model.predict_proba(X)
                # Extract classes from model
                classes = getattr(model, "classes_", None)
                if classes is None:
                    classes = getattr(self.model, "classes_", None)
                if classes is None:
                    classes = list(self.target_definition.classes)
                return float(-log_loss(y, proba, labels=list(classes)))
        
        LOGGER.info(f"Computing permutation importance with {n_repeats} repeats...")
        
        # Compute permutation importance
        result = permutation_importance(
            estimator,
            X_eval,
            y_eval,
            n_repeats=n_repeats,
            random_state=random_state,
            n_jobs=-1,
            scoring=None,  # Will use default predict/predict_proba
        )
        
        # Create results dataframe
        feature_names = X_eval.columns.tolist()
        importance_df = pd.DataFrame({
            "feature": feature_names,
            "importance_mean": result.importances_mean,
            "importance_std": result.importances_std,
        })
        
        # Sort by importance
        importance_df = importance_df.sort_values("importance_mean", ascending=False).reset_index(drop=True)
        
        # Add percentile ranking
        max_importance = importance_df["importance_mean"].max()
        min_importance = importance_df["importance_mean"].min()
        importance_range = max_importance - min_importance
        
        if importance_range > 0:
            importance_df["importance_pct"] = (
                (importance_df["importance_mean"] - min_importance) / importance_range * 100
            ).round(2)
        else:
            importance_df["importance_pct"] = 0.0
        
        importance_df["rank"] = range(1, len(importance_df) + 1)
        
        return importance_df

    def save_report(
        self,
        importance_df: pd.DataFrame,
        top_n: int | None = None,
    ) -> Path:
        """Save importance report to CSV.
        
        Args:
            importance_df: DataFrame from compute_importance()
            top_n: If specified, only save top N features (but return full DF with marker)
        
        Returns:
            Path to saved CSV file
        """
        report_path = self.output_dir / f"permutation_importance_{self.target_name}.csv"
        
        # Save full report
        importance_df.to_csv(report_path, index=False)
        LOGGER.info(f"Saved permutation importance report: {report_path}")
        
        # Also save top-N summary if requested
        if top_n is not None and top_n > 0:
            top_df = importance_df.head(top_n).copy()
            summary_path = self.output_dir / f"permutation_importance_{self.target_name}_top{top_n}.csv"
            top_df.to_csv(summary_path, index=False)
            LOGGER.info(f"Saved top-{top_n} summary: {summary_path}")
        
        return report_path

    def get_top_features(
        self,
        importance_df: pd.DataFrame,
        n: int = 20,
    ) -> list[str]:
        """Get top N features by importance.
        
        Args:
            importance_df: DataFrame from compute_importance()
            n: Number of top features to return
        
        Returns:
            List of top feature names
        """
        return importance_df.head(n)["feature"].tolist()


class ImportanceComparison:
    """Compare permutation importance across targets and model types."""

    def __init__(self, output_dir: str | Path = "reports") -> None:
        """Initialize comparison tool.
        
        Args:
            output_dir: Directory containing importance reports
        """
        self.output_dir = Path(output_dir)

    def compare_targets(
        self,
        importance_csvs: dict[str, Path],
        top_n: int = 20,
    ) -> pd.DataFrame:
        """Compare top features across multiple targets.
        
        Args:
            importance_csvs: Mapping of target_name -> CSV path
            top_n: Number of top features to compare
        
        Returns:
            DataFrame showing top features per target
        """
        comparison_data = {}
        
        for target_name, csv_path in importance_csvs.items():
            df = pd.read_csv(csv_path)
            top_features = df.head(top_n)["feature"].tolist()
            comparison_data[target_name] = top_features
        
        # Create comparison matrix
        all_features = set()
        for features in comparison_data.values():
            all_features.update(features)
        
        comparison_df = pd.DataFrame(index=sorted(all_features))
        for target_name, top_features in comparison_data.items():
            comparison_df[target_name] = comparison_df.index.map(
                lambda f: top_features.index(f) + 1 if f in top_features else None
            )
        
        return comparison_df.sort_values(
            by=list(comparison_df.columns),
            ascending=True,
            na_position="last",
        )

    def save_comparison(
        self,
        comparison_df: pd.DataFrame,
        filename: str = "feature_importance_comparison.csv",
    ) -> Path:
        """Save comparison report to CSV.
        
        Args:
            comparison_df: DataFrame from compare_targets()
            filename: Output filename
        
        Returns:
            Path to saved file
        """
        output_path = self.output_dir / filename
        comparison_df.to_csv(output_path)
        LOGGER.info(f"Saved feature importance comparison: {output_path}")
        return output_path
