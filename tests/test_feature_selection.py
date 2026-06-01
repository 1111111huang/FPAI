"""Tests for feature selection and importance analysis."""

import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import joblib
import numpy as np
import pandas as pd
import pytest
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LogisticRegression

from src.models import ModelFactory
from src.utils.feature_importance import ImportanceComparison, PermutationImportanceAnalyzer
from src.utils.feature_selection import FeatureSelectionStudy


class TestPermutationImportanceAnalyzer:
    """Test permutation importance computation and reporting."""

    @pytest.fixture
    def sample_data(self):
        """Create sample training/validation data."""
        np.random.seed(42)
        n_samples = 200
        n_features = 10
        
        X = pd.DataFrame(
            np.random.randn(n_samples, n_features),
            columns=[f"feature_{i}" for i in range(n_features)],
        )
        y = pd.Series(np.random.randint(0, 2, n_samples))
        
        return X, y

    @pytest.fixture
    def trained_model(self, sample_data):
        """Create and train a simple model."""
        X, y = sample_data
        model = LogisticRegression(random_state=42, max_iter=1000)
        model.fit(X, y)
        return model

    @pytest.fixture
    def model_file(self, trained_model):
        """Save model to temporary file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            model_path = Path(tmpdir) / "test_model.joblib"
            joblib.dump(trained_model, model_path)
            yield model_path

    def test_analyzer_initialization(self, model_file):
        """Test analyzer initialization with valid model."""
        analyzer = PermutationImportanceAnalyzer(
            model_path=model_file,
            target_name="btts",
        )
        assert analyzer.model_path == model_file
        assert analyzer.target_name == "btts"

    def test_analyzer_missing_model(self):
        """Test analyzer raises error for missing model."""
        with pytest.raises(FileNotFoundError):
            PermutationImportanceAnalyzer(
                model_path="/nonexistent/model.joblib",
                target_name="btts",
            )

    def test_compute_importance_classifier(self, model_file, sample_data):
        """Test permutation importance computation for classifier."""
        X, y = sample_data
        analyzer = PermutationImportanceAnalyzer(
            model_path=model_file,
            target_name="btts",
        )
        
        importance_df = analyzer.compute_importance(X, y, n_repeats=2)
        
        # Check output structure
        assert "feature" in importance_df.columns
        assert "importance_mean" in importance_df.columns
        assert "importance_std" in importance_df.columns
        assert "importance_pct" in importance_df.columns
        assert "rank" in importance_df.columns
        assert len(importance_df) == len(X.columns)
        
        # Check rankings are valid
        assert list(importance_df["rank"]) == list(range(1, len(X.columns) + 1))

    def test_get_top_features(self, model_file, sample_data):
        """Test extracting top features."""
        X, y = sample_data
        analyzer = PermutationImportanceAnalyzer(
            model_path=model_file,
            target_name="btts",
        )
        
        importance_df = analyzer.compute_importance(X, y, n_repeats=2)
        top_5 = analyzer.get_top_features(importance_df, n=5)
        
        assert len(top_5) == 5
        assert all(isinstance(f, str) for f in top_5)

    def test_save_report(self, model_file, sample_data):
        """Test saving importance report to CSV."""
        X, y = sample_data
        
        with tempfile.TemporaryDirectory() as tmpdir:
            analyzer = PermutationImportanceAnalyzer(
                model_path=model_file,
                target_name="btts",
                output_dir=tmpdir,
            )
            
            importance_df = analyzer.compute_importance(X, y, n_repeats=2)
            report_path = analyzer.save_report(importance_df)
            
            assert report_path.exists()
            assert report_path.suffix == ".csv"
            
            # Verify saved data
            saved_df = pd.read_csv(report_path)
            assert len(saved_df) == len(importance_df)
            assert list(saved_df.columns) == list(importance_df.columns)

    def test_save_report_with_top_n(self, model_file, sample_data):
        """Test saving report with top-N summary."""
        X, y = sample_data
        
        with tempfile.TemporaryDirectory() as tmpdir:
            analyzer = PermutationImportanceAnalyzer(
                model_path=model_file,
                target_name="btts",
                output_dir=tmpdir,
            )
            
            importance_df = analyzer.compute_importance(X, y, n_repeats=2)
            analyzer.save_report(importance_df, top_n=3)
            
            # Check both files exist
            full_report = Path(tmpdir) / "permutation_importance_btts.csv"
            top_report = Path(tmpdir) / "permutation_importance_btts_top3.csv"
            
            assert full_report.exists()
            assert top_report.exists()
            
            # Check top report has fewer rows
            top_df = pd.read_csv(top_report)
            assert len(top_df) == 3


class TestFeatureSelectionStudy:
    """Test feature selection study functionality."""

    @pytest.fixture
    def training_splits(self):
        """Create train/val/test splits for testing."""
        np.random.seed(42)
        n_train, n_val, n_test = 100, 40, 40
        n_features = 20
        
        X_train = pd.DataFrame(
            np.random.randn(n_train, n_features),
            columns=[f"feature_{i}" for i in range(n_features)],
        )
        y_train = pd.Series(np.random.randint(0, 2, n_train))
        
        X_val = pd.DataFrame(
            np.random.randn(n_val, n_features),
            columns=[f"feature_{i}" for i in range(n_features)],
        )
        y_val = pd.Series(np.random.randint(0, 2, n_val))
        
        X_test = pd.DataFrame(
            np.random.randn(n_test, n_features),
            columns=[f"feature_{i}" for i in range(n_features)],
        )
        y_test = pd.Series(np.random.randint(0, 2, n_test))
        
        return X_train, X_val, X_test, y_train, y_val, y_test

    @pytest.fixture
    def feature_ranking(self):
        """Create ranked feature list."""
        return [f"feature_{i}" for i in range(20)]

    def test_feature_selection_study_initialization(self):
        """Test study initialization."""
        study = FeatureSelectionStudy(
            target_name="btts",
            model_type="LogisticRegression",
        )
        assert study.target_name == "btts"
        assert study.model_type == "LogisticRegression"

    def test_run_feature_subset_study(self, training_splits, feature_ranking):
        """Test running feature subset experiments."""
        X_train, X_val, X_test, y_train, y_val, y_test = training_splits
        
        study = FeatureSelectionStudy(
            target_name="btts",
            model_type="logistic_regression",
        )
        
        results_df = study.run_feature_subset_study(
            feature_ranking=feature_ranking,
            feature_subsets=[5, 10, 20],
            X_train=X_train,
            X_val=X_val,
            X_test=X_test,
            y_train=y_train,
            y_val=y_val,
            y_test=y_test,
        )
        
        assert len(results_df) == 3
        assert "n_features" in results_df.columns
        assert "log_loss_test" in results_df.columns
        assert "accuracy_test" in results_df.columns
        assert list(results_df["n_features"].values) == [5, 10, 20]

    def test_recommend_feature_set(self, training_splits, feature_ranking):
        """Test feature set recommendation."""
        X_train, X_val, X_test, y_train, y_val, y_test = training_splits
        
        study = FeatureSelectionStudy(
            target_name="btts",
            model_type="logistic_regression",
        )
        
        results_df = study.run_feature_subset_study(
            feature_ranking=feature_ranking,
            feature_subsets=[5, 10, 15, 20],
            X_train=X_train,
            X_val=X_val,
            X_test=X_test,
            y_train=y_train,
            y_val=y_val,
            y_test=y_test,
        )
        
        recommendation = study.recommend_feature_set(results_df)
        
        assert "recommendation" in recommendation
        assert "metric_baseline" in recommendation
        assert "metric_recommended" in recommendation
        assert recommendation["recommendation"] > 0

    def test_save_results(self, training_splits, feature_ranking):
        """Test saving results to CSV."""
        X_train, X_val, X_test, y_train, y_val, y_test = training_splits
        
        with tempfile.TemporaryDirectory() as tmpdir:
            study = FeatureSelectionStudy(
                target_name="btts",
                model_type="logistic_regression",
                output_dir=tmpdir,
            )
            
            results_df = study.run_feature_subset_study(
                feature_ranking=feature_ranking,
                feature_subsets=[5, 10],
                X_train=X_train,
                X_val=X_val,
                X_test=X_test,
                y_train=y_train,
                y_val=y_val,
                y_test=y_test,
            )
            
            output_path = study.save_results(results_df)
            
            assert output_path.exists()
            saved_df = pd.read_csv(output_path)
            assert len(saved_df) == len(results_df)

    def test_regression_feature_selection(self):
        """Test feature selection for regression targets."""
        np.random.seed(42)
        n_samples = 100
        n_features = 15
        
        X_train = pd.DataFrame(
            np.random.randn(n_samples, n_features),
            columns=[f"feature_{i}" for i in range(n_features)],
        )
        y_train = pd.Series(np.random.randn(n_samples))
        
        X_val = pd.DataFrame(
            np.random.randn(40, n_features),
            columns=[f"feature_{i}" for i in range(n_features)],
        )
        y_val = pd.Series(np.random.randn(40))
        
        X_test = pd.DataFrame(
            np.random.randn(40, n_features),
            columns=[f"feature_{i}" for i in range(n_features)],
        )
        y_test = pd.Series(np.random.randn(40))
        
        feature_ranking = [f"feature_{i}" for i in range(n_features)]
        
        study = FeatureSelectionStudy(
            target_name="total_goals",
            model_type="random_forest_regressor",
        )
        
        results_df = study.run_feature_subset_study(
            feature_ranking=feature_ranking,
            feature_subsets=[5, 10, 15],
            X_train=X_train,
            X_val=X_val,
            X_test=X_test,
            y_train=y_train,
            y_val=y_val,
            y_test=y_test,
        )
        
        # Check regression-specific columns
        assert "mae_test" in results_df.columns
        assert len(results_df) == 3


class TestImportanceComparison:
    """Test feature importance comparison across targets."""

    @pytest.fixture
    def importance_csvs(self):
        """Create sample importance CSV files."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create comparison for three targets
            targets = ["btts", "result_3way", "total_goals"]
            files = {}
            
            for target in targets:
                df = pd.DataFrame({
                    "feature": [f"feature_{i}" for i in range(10)],
                    "importance_mean": np.random.randn(10),
                    "importance_std": np.abs(np.random.randn(10) * 0.1),
                })
                
                path = Path(tmpdir) / f"permutation_importance_{target}.csv"
                df.to_csv(path, index=False)
                files[target] = path
            
            yield files

    def test_compare_targets(self, importance_csvs):
        """Test comparing importance across targets."""
        comparison = ImportanceComparison()
        comparison_df = comparison.compare_targets(importance_csvs, top_n=5)
        
        # Check structure
        assert len(comparison_df) <= 30  # At most 3 targets × 10 features
        assert list(comparison_df.columns) == list(importance_csvs.keys())

    def test_save_comparison(self, importance_csvs):
        """Test saving comparison report."""
        with tempfile.TemporaryDirectory() as tmpdir:
            comparison = ImportanceComparison(output_dir=tmpdir)
            comparison_df = comparison.compare_targets(importance_csvs, top_n=5)
            output_path = comparison.save_comparison(comparison_df)
            
            assert output_path.exists()
            saved_df = pd.read_csv(output_path, index_col=0)
            assert list(saved_df.columns) == list(importance_csvs.keys())
