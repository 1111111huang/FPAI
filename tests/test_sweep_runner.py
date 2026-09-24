"""Tests for SweepRunner/OptunaRunner correctly threading a non-default
league context through to ModelManager (and thus to real per-league data),
not just tagging the MLflow run with it.

Found live while validating BUG-070's xG-signal retrain candidates: a
config's context/league key was only ever used for an MLflow tag -- never
passed to ModelManager's own competition_id/context params, which default
to "E0". Every sweep-target/optuna-sweep run, regardless of config, was
silently training against E0's data no matter which league it claimed to
target."""

from __future__ import annotations

from pathlib import Path
import sys
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest
import yaml

sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.utils.sweep_runner import OptunaRunner, SweepRunner


def _write_grid_config(tmp_path: Path, context: str) -> Path:
    config = {
        "model_type": "xgboost",
        "context": context,
        "grid_search": {"n_estimators": [100]},
        "fixed_params": {"random_state": 42},
    }
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(config), encoding="utf-8")
    return path


def _write_optuna_config(tmp_path: Path, context: str) -> Path:
    config = {
        "model_type": "xgboost",
        "context": context,
        "n_trials": 1,
        "optuna_search": {"n_estimators": {"type": "int", "low": 100, "high": 100}},
        "fixed_params": {"random_state": 42},
    }
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(config), encoding="utf-8")
    return path


def _mock_manager() -> MagicMock:
    manager = MagicMock()
    empty = pd.DataFrame({"f1": [0.0, 1.0]})
    manager.prepare_training_data.return_value = (empty, empty, empty, pd.Series([0, 1]), pd.Series([0, 1]), pd.Series([0, 1]), None)
    manager._evaluate_target.return_value = ({"log_loss": 0.5}, None)
    manager.training_cutoff = "2026-01-01"
    return manager


@patch("src.utils.sweep_runner.configure_mlflow_tracking")
@patch("src.utils.sweep_runner.mlflow")
@patch("src.utils.sweep_runner.ModelManager")
@patch("src.utils.sweep_runner.ModelFactory")
def test_grid_sweep_passes_context_and_competition_id_to_model_manager(
    mock_factory, mock_manager_cls, mock_mlflow, mock_configure, tmp_path: Path
):
    mock_factory.get_model.return_value = MagicMock()
    mock_manager_cls.return_value = _mock_manager()
    mock_mlflow.start_run.return_value.__enter__ = MagicMock()
    mock_mlflow.start_run.return_value.__exit__ = MagicMock(return_value=False)

    config_path = _write_grid_config(tmp_path, context="D1")
    SweepRunner(target_name="btts", config_path=config_path).run()

    _, kwargs = mock_manager_cls.call_args
    assert kwargs["context"] == "D1"
    assert kwargs["competition_id"] == "D1"


@patch("src.utils.sweep_runner.configure_mlflow_tracking")
@patch("src.utils.sweep_runner.mlflow")
@patch("src.utils.sweep_runner.ModelManager")
@patch("src.utils.sweep_runner.ModelFactory")
def test_grid_sweep_defaults_context_to_e0_when_omitted(
    mock_factory, mock_manager_cls, mock_mlflow, mock_configure, tmp_path: Path
):
    mock_factory.get_model.return_value = MagicMock()
    mock_manager_cls.return_value = _mock_manager()
    mock_mlflow.start_run.return_value.__enter__ = MagicMock()
    mock_mlflow.start_run.return_value.__exit__ = MagicMock(return_value=False)

    config = {"model_type": "xgboost", "grid_search": {"n_estimators": [100]}}
    config_path = tmp_path / "config.yaml"
    config_path.write_text(yaml.safe_dump(config), encoding="utf-8")
    SweepRunner(target_name="btts", config_path=config_path).run()

    _, kwargs = mock_manager_cls.call_args
    assert kwargs["context"] == "E0"
    assert kwargs["competition_id"] == "E0"


@patch("src.utils.sweep_runner.optuna")
@patch("src.utils.sweep_runner.configure_mlflow_tracking")
@patch("src.utils.sweep_runner.mlflow")
@patch("src.utils.sweep_runner.ModelManager")
@patch("src.utils.sweep_runner.ModelFactory")
def test_optuna_sweep_passes_context_and_competition_id_to_model_manager(
    mock_factory, mock_manager_cls, mock_mlflow, mock_configure, mock_optuna, tmp_path: Path
):
    mock_factory.get_model.return_value = MagicMock()
    mock_manager_cls.return_value = _mock_manager()
    mock_mlflow.start_run.return_value.__enter__ = MagicMock()
    mock_mlflow.start_run.return_value.__exit__ = MagicMock(return_value=False)

    def _fake_optimize(objective, n_trials, show_progress_bar):
        trial = MagicMock()
        trial.number = 0
        trial.suggest_int.return_value = 100
        objective(trial)

    mock_optuna.create_study.return_value.optimize.side_effect = _fake_optimize

    config_path = _write_optuna_config(tmp_path, context="I1")
    OptunaRunner(target_name="btts", config_path=config_path).run()

    _, kwargs = mock_manager_cls.call_args
    assert kwargs["context"] == "I1"
    assert kwargs["competition_id"] == "I1"
