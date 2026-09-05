"""Tests for centralized MLflow tracking-URI configuration (US#185).

MLflow's implicit default (no tracking URI set at all) is the filesystem
backend -- deprecated by MLflow itself and, in this project, grown to
14GB/109k files, causing a full select-best-models scan to take ~2 hours.
Every mlflow entry point must call configure_mlflow_tracking() before any
other mlflow.* call, so they all agree on the same (DB-backed) store.
"""

from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import patch

import pytest
import yaml

sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.utils.config_loader import AppSettings
from src.utils.mlflow_config import configure_mlflow_tracking


def test_app_settings_default_mlflow_uri_is_db_backed_not_filestore() -> None:
    settings = AppSettings()
    assert settings.mlflow_tracking_uri == "sqlite:///mlflow.db"


def test_configure_mlflow_tracking_sets_uri_from_config(tmp_path: Path) -> None:
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        yaml.safe_dump({"mlflow_tracking_uri": "sqlite:///custom_test.db"}), encoding="utf-8",
    )
    with patch("src.utils.mlflow_config.mlflow.set_tracking_uri") as mock_set:
        configure_mlflow_tracking(str(config_path))
    mock_set.assert_called_once_with("sqlite:///custom_test.db")


def test_configure_mlflow_tracking_uses_default_when_config_omits_it(tmp_path: Path) -> None:
    config_path = tmp_path / "config.yaml"
    config_path.write_text(yaml.safe_dump({"project_name": "x"}), encoding="utf-8")
    with patch("src.utils.mlflow_config.mlflow.set_tracking_uri") as mock_set:
        configure_mlflow_tracking(str(config_path))
    mock_set.assert_called_once_with("sqlite:///mlflow.db")


def test_configure_mlflow_tracking_is_safe_to_call_repeatedly(tmp_path: Path) -> None:
    """No global 'already configured' guard -- calling it again (e.g. from
    a different entry point later in the same process) must not raise or
    silently no-op; setting the same URI twice is a cheap, harmless call."""
    config_path = tmp_path / "config.yaml"
    config_path.write_text(yaml.safe_dump({}), encoding="utf-8")
    with patch("src.utils.mlflow_config.mlflow.set_tracking_uri") as mock_set:
        configure_mlflow_tracking(str(config_path))
        configure_mlflow_tracking(str(config_path))
    assert mock_set.call_count == 2


def test_model_selector_calls_configure_mlflow_tracking() -> None:
    """US#185 coverage gap, found live: ModelSelector (used by both the
    select-best-models CLI command AND any direct-Python caller of
    run_select_best_models, e.g. an ad-hoc retraining script) never called
    configure_mlflow_tracking() itself -- it worked by accident through the
    CLI (main()'s own top-level call already set the URI process-wide
    before ModelSelector was ever constructed), but a script that imports
    run_select_best_models directly without going through main() got the
    implicit filesystem-backend default instead, silently re-introducing
    the exact 14GB/109k-file/~2-hour-scan problem US#185 was written to
    fix. Found live: a real multi-context select-best-models run that
    should have taken under a minute instead ran for 75+ minutes before
    being killed, traced to this exact gap via the FutureWarning
    mlflow logs when it falls back to FileStore."""
    from src.utils.model_selection import ModelSelector

    with patch("src.utils.model_selection.configure_mlflow_tracking") as mock_configure, \
         patch("src.utils.model_selection.mlflow.tracking.MlflowClient"):
        ModelSelector()

    mock_configure.assert_called_once()
