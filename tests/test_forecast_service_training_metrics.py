"""Tests for surfacing each target's own historical training-time metrics
to the agent (A95).

Direct user request: "put the error info from each ml model into the
system prompt so the agent is aware." Every model's own `.metadata.json`
sidecar already stores a real `metrics` dict (e.g. {"log_loss": ...,
"accuracy": ...} or {"mae": ..., "rmse": ...}) computed at training time --
`_load_context_models` already reads this same file for `feature_names`
but silently drops `metrics` on the floor. This threads it through into
the metadata dict `_load_context_models` returns, so `forecast_upcoming`'s
`diagnostics.target_versions[target]["metrics"]` (and therefore the
FORECAST_PAYLOAD JSON the agent actually sees, `src/agent/pipeline.py`'s
`_format_evidence_message`) carries it with zero new plumbing beyond this.
"""

from __future__ import annotations

import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import yaml

sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.forecast.forecast_service import ForecastService
from src.models.base_model import XGBoostRegressorModel


def _write_fake_promoted_model(tmp_path: Path, metrics: dict) -> Path:
    model = XGBoostRegressorModel(n_estimators=5, max_depth=2, early_stopping_rounds=None)
    rng = np.random.default_rng(0)
    X = pd.DataFrame({"a": rng.normal(size=20)})
    y = pd.Series(rng.normal(1.5, 0.5, size=20))
    model.train(X, y)

    models_dir = tmp_path / "models"
    models_dir.mkdir()
    model_path = models_dir / "home_goals_fake.joblib"
    model.save(str(model_path))
    metadata_path = model_path.with_suffix(model_path.suffix + ".metadata.json")
    metadata_path.write_text(json.dumps({"feature_names": ["a"], "metrics": metrics}), encoding="utf-8")

    config_dir = tmp_path / "config"
    config_dir.mkdir()
    selection_path = config_dir / "model_selection.yaml"
    selection_path.write_text(
        yaml.safe_dump({
            "contexts": {
                "E0": {
                    "home_goals": {
                        "model_path": str(model_path), "model_type": "xgb_regressor",
                        "feature_subset": ["a"],
                    },
                }
            }
        }),
        encoding="utf-8",
    )
    return tmp_path / "config.yaml"


def test_load_context_models_threads_metrics_from_metadata_json(tmp_path: Path) -> None:
    real_metrics = {"mae": 0.876, "rmse": 1.1}
    config_path = _write_fake_promoted_model(tmp_path, real_metrics)

    service = ForecastService.__new__(ForecastService)
    service.config_path = config_path
    service.targets = ["home_goals"]
    service.feature_names = ["a"]

    loaded = service._load_context_models("E0")

    _, _, metadata = loaded["home_goals"]
    assert metadata["metrics"] == real_metrics


def test_load_context_models_metrics_is_none_when_metadata_json_absent(tmp_path: Path) -> None:
    """No .metadata.json sidecar at all (an older/manually-placed artifact)
    -- must not crash, metrics is just None like every other optional field
    here (calibrator, feature_subset)."""
    model = XGBoostRegressorModel(n_estimators=5, max_depth=2, early_stopping_rounds=None)
    rng = np.random.default_rng(1)
    X = pd.DataFrame({"a": rng.normal(size=20)})
    y = pd.Series(rng.normal(1.5, 0.5, size=20))
    model.train(X, y)

    models_dir = tmp_path / "models"
    models_dir.mkdir()
    model_path = models_dir / "home_goals_no_metadata.joblib"
    model.save(str(model_path))
    # deliberately no .metadata.json written

    config_dir = tmp_path / "config"
    config_dir.mkdir()
    selection_path = config_dir / "model_selection.yaml"
    selection_path.write_text(
        yaml.safe_dump({
            "contexts": {
                "E0": {
                    "home_goals": {
                        "model_path": str(model_path), "model_type": "xgb_regressor",
                        "feature_subset": ["a"],
                    },
                }
            }
        }),
        encoding="utf-8",
    )

    service = ForecastService.__new__(ForecastService)
    service.config_path = tmp_path / "config.yaml"
    service.targets = ["home_goals"]
    service.feature_names = ["a"]

    loaded = service._load_context_models("E0")

    _, _, metadata = loaded["home_goals"]
    assert metadata["metrics"] is None
