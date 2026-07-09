"""Tests for ModelSelector (BUG-012 layer 3c/3d): promotion must not write a
model_selection.yaml entry for a model whose required features can't actually
be computed by the live feature pipeline, and must backfill feature_subset
from the model's own .metadata.json when the training run didn't log one."""

from __future__ import annotations

import json
from pathlib import Path
import sys
from unittest.mock import MagicMock

import pytest
import yaml

sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.utils.model_selection import ModelSelector, missing_features


def test_missing_features_returns_only_uncomputable_names() -> None:
    assert missing_features(
        required=["A", "B", "C"], available={"A", "C"},
    ) == ["B"]


def test_missing_features_empty_when_all_available() -> None:
    assert missing_features(required=["A", "B"], available={"A", "B", "C"}) == []


def _make_run(run_id: str, model_type: str, metric_value: float, artifact_filename: str) -> MagicMock:
    run = MagicMock()
    run.info.run_id = run_id
    run.info.artifact_uri = f"file:///mlruns/1/{run_id}/artifacts"
    run.data.metrics = {"test_mae": metric_value}
    run.data.tags = {"model_family": model_type}
    run.data.params = {"artifact_filename": artifact_filename}
    return run


def test_select_for_target_context_refuses_promotion_when_features_uncomputable(tmp_path: Path) -> None:
    """BUG-012 layer 3c: a candidate whose .metadata.json requires a feature
    absent from the live-computable set must not be written to
    model_selection.yaml — refuse loudly at promotion time instead of
    shipping a model that will KeyError/degrade at live-inference time."""
    model_dir = tmp_path / "models"
    model_dir.mkdir()
    artifact_name = "home_goals_dummy_v1_20260701.joblib"
    (model_dir / artifact_name).write_bytes(b"fake")
    (model_dir / f"{artifact_name}.metadata.json").write_text(
        json.dumps({"feature_names": ["OFF_HOME_FTHG_R5", "XOC_HOME"]}),
        encoding="utf-8",
    )

    selection_path = tmp_path / "model_selection.yaml"
    selector = ModelSelector(
        config_path=selection_path,
        model_dir=model_dir,
        computable_features={"OFF_HOME_FTHG_R5", "MKT_IMPLIED_HOME"},  # no XOC_HOME
    )
    selector.client = MagicMock()
    selector.client.search_experiments.return_value = [MagicMock(experiment_id="1")]
    selector.client.search_runs.side_effect = [
        [_make_run("run1", "xgboostregressor", 0.9, artifact_name)],  # optuna
        [],  # final
    ]

    result = selector._select_for_target_context(
        target_name="home_goals", context="league", current_entry={},
        min_improvement=0.005, dry_run=False,
    )

    assert result is None, "Promotion must be refused when required features are not computable"


def test_select_for_target_context_backfills_feature_subset_from_metadata(tmp_path: Path) -> None:
    """BUG-012 layer 3d: when the MLflow run has no feature_subset param, fall
    back to the model's own .metadata.json feature_names so
    model_selection.yaml is self-documenting even for the league context."""
    model_dir = tmp_path / "models"
    model_dir.mkdir()
    artifact_name = "home_goals_dummy_v1_20260701.joblib"
    (model_dir / artifact_name).write_bytes(b"fake")
    (model_dir / f"{artifact_name}.metadata.json").write_text(
        json.dumps({"feature_names": ["OFF_HOME_FTHG_R5", "MKT_IMPLIED_HOME"]}),
        encoding="utf-8",
    )

    selection_path = tmp_path / "model_selection.yaml"
    selector = ModelSelector(
        config_path=selection_path,
        model_dir=model_dir,
        computable_features={"OFF_HOME_FTHG_R5", "MKT_IMPLIED_HOME"},
    )
    selector.client = MagicMock()
    selector.client.search_experiments.return_value = [MagicMock(experiment_id="1")]
    selector.client.search_runs.side_effect = [
        [_make_run("run1", "xgboostregressor", 0.9, artifact_name)],
        [],
    ]

    result = selector._select_for_target_context(
        target_name="home_goals", context="league", current_entry={},
        min_improvement=0.005, dry_run=False,
    )

    assert result is not None
    assert result["feature_subset"] == ["OFF_HOME_FTHG_R5", "MKT_IMPLIED_HOME"], (
        f"Expected feature_subset backfilled from .metadata.json, got {result.get('feature_subset')}"
    )
