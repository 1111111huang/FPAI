"""Regression tests for BUG-014: ModelSelector._select_for_target_context
only compared metric values between the recorded champion and the best
available MLflow run. Since select-best-models doesn't itself train
anything, re-running it against an already-recorded champion finds the
identical run/metric every time ("no improvement"), and returns None before
ever reaching the code that computes a fresh model_path -- permanently
preserving a stale/broken model_path from before the BUG-010 fix, with no
way to self-correct. Fix: promote regardless of the metric comparison when
the current champion's model_path doesn't actually resolve to a file."""

from __future__ import annotations

from pathlib import Path
import sys
from unittest.mock import MagicMock

sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.utils.model_selection import ModelSelector


def _make_run(
    run_id: str, model_type: str, metric_value: float, artifact_filename: str, metric_name: str = "test_log_loss",
) -> MagicMock:
    run = MagicMock()
    run.info.run_id = run_id
    run.info.artifact_uri = f"file:///mlruns/1/{run_id}/artifacts"
    run.data.metrics = {metric_name: metric_value}
    run.data.tags = {"model_family": model_type}
    run.data.params = {"artifact_filename": artifact_filename}
    return run


def test_promotes_despite_tied_metric_when_current_model_path_is_broken(tmp_path: Path) -> None:
    model_dir = tmp_path / "models"
    model_dir.mkdir()
    artifact_name = "btts_xgboost_v1_20260628.joblib"
    (model_dir / artifact_name).write_bytes(b"fake")

    selector = ModelSelector(config_path=tmp_path / "model_selection.yaml", model_dir=model_dir)
    selector.client = MagicMock()
    selector.client.search_experiments.return_value = [MagicMock(experiment_id="1")]
    selector.client.search_runs.side_effect = [
        [_make_run("run1", "xgboostclassifier", 0.6996, artifact_name)],  # optuna
        [],  # final
    ]

    current_entry = {
        # BUG-010-era broken path -- never resolves to a real file
        "model_path": "file:/Users/x/mlruns/1/oldrun/artifacts/model",
        "metric_value": 0.6996,  # identical metric -- would normally be "no improvement"
    }

    result = selector._select_for_target_context(
        target_name="btts", context="international", current_entry=current_entry,
        min_improvement=0.005, dry_run=False,
    )

    assert result is not None, "Must promote when the current champion's model_path is broken, even with a tied metric"
    assert result["model_path"] == f"models/{artifact_name}"


def test_still_skips_when_metric_tied_and_current_path_actually_resolves(tmp_path: Path) -> None:
    """Regression: unchanged behavior when the current champion IS loadable."""
    model_dir = tmp_path / "models"
    model_dir.mkdir()
    artifact_name = "btts_xgboost_v1_20260628.joblib"
    (model_dir / artifact_name).write_bytes(b"fake")

    selector = ModelSelector(config_path=tmp_path / "model_selection.yaml", model_dir=model_dir)
    selector.client = MagicMock()
    selector.client.search_experiments.return_value = [MagicMock(experiment_id="1")]
    selector.client.search_runs.side_effect = [
        [_make_run("run1", "xgboostclassifier", 0.6996, artifact_name)],
        [],
    ]

    current_entry = {
        "model_path": f"models/{artifact_name}",  # resolves fine relative to model_dir.parent
        "metric_value": 0.6996,
    }

    result = selector._select_for_target_context(
        target_name="btts", context="international", current_entry=current_entry,
        min_improvement=0.005, dry_run=False,
    )

    assert result is None


def test_best_run_excludes_candidates_whose_own_artifact_is_missing(tmp_path: Path) -> None:
    """BUG-014 layer 2: _best_run must not pick a candidate with a better
    metric but a nonexistent artifact over a worse-metric candidate whose
    file actually exists -- an unloadable 'best' run is not actually usable,
    so it must be excluded from consideration entirely, not just deprioritized
    against the current champion."""
    model_dir = tmp_path / "models"
    model_dir.mkdir()
    real_artifact = "away_corners_xgboostregressor_v1_20260711.joblib"
    (model_dir / real_artifact).write_bytes(b"fake")
    missing_artifact = "away_corners_xgboostregressor_v1_20260628.joblib"  # never actually saved

    selector = ModelSelector(config_path=tmp_path / "model_selection.yaml", model_dir=model_dir)
    selector.client = MagicMock()
    selector.client.search_experiments.return_value = [MagicMock(experiment_id="1")]
    selector.client.search_runs.side_effect = [
        [
            _make_run("old_run", "xgboostregressor", 2.1121, missing_artifact, metric_name="test_mae"),  # better metric, missing file
            _make_run("new_run", "xgboostregressor", 2.1125, real_artifact, metric_name="test_mae"),  # worse metric, real file
        ],
        [],
    ]

    current_entry = {
        "model_path": f"file:/mlruns/1/old_run/artifacts/model",  # broken, forces promotion regardless
        "metric_value": 2.1121,
    }

    result = selector._select_for_target_context(
        target_name="away_corners", context="international", current_entry=current_entry,
        min_improvement=0.005, dry_run=False,
    )

    assert result is not None
    assert result["model_path"] == f"models/{real_artifact}", (
        "Must select the loadable candidate even though the missing-artifact one has a better raw metric"
    )


def test_still_promotes_on_genuine_metric_improvement_regardless_of_path(tmp_path: Path) -> None:
    """Regression: the ordinary 'better metric -> promote' path is unaffected."""
    model_dir = tmp_path / "models"
    model_dir.mkdir()
    artifact_name = "btts_xgboost_v2_20260710.joblib"
    (model_dir / artifact_name).write_bytes(b"fake")

    selector = ModelSelector(config_path=tmp_path / "model_selection.yaml", model_dir=model_dir)
    selector.client = MagicMock()
    selector.client.search_experiments.return_value = [MagicMock(experiment_id="1")]
    selector.client.search_runs.side_effect = [
        [_make_run("run2", "xgboostclassifier", 0.60, artifact_name)],  # clearly better (lower log_loss)
        [],
    ]

    current_entry = {"model_path": f"models/old.joblib", "metric_value": 0.6996}

    result = selector._select_for_target_context(
        target_name="btts", context="international", current_entry=current_entry,
        min_improvement=0.005, dry_run=False,
    )

    assert result is not None
    assert result["model_path"] == f"models/{artifact_name}"
