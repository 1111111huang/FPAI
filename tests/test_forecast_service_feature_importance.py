"""Tests for threading `feature_importance` through `_load_context_models`
(BUG-103). Every model's `.metadata.json` sidecar already stores a real
`feature_importance` dict, written at training time
(`src/models/model_manager.py`'s `_build_artifact_metadata`) -- but
`_load_context_models` builds its own hand-picked metadata dict and never
copies this field across, so `ForecastService._top_features()`
(`forecast_service.py`) always finds `metadata.get("feature_importance", [])`
empty, and every live recommendation's `forecast_payload.explainability.
top_features` has been `[]` since this loader was introduced -- confirmed
live, 0 of 889 real cached recommendations had any top_features. Mirrors
`test_forecast_service_training_metrics.py`'s `metrics`-threading tests
exactly, since this is the same gap in the same dict construction."""

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


def _write_fake_promoted_model(tmp_path: Path, feature_importance: list[dict]) -> Path:
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
    metadata_path.write_text(
        json.dumps({"feature_names": ["a"], "feature_importance": feature_importance}), encoding="utf-8"
    )

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


def test_load_context_models_threads_feature_importance_from_metadata_json(tmp_path: Path) -> None:
    # Real shape, confirmed against model_manager.py's _build_artifact_metadata:
    # feature_importance.head(50).to_dict(orient="records") -- a LIST of
    # {"feature": ..., "importance": ...} row-dicts, not a dict of parallel
    # lists. This is also exactly the shape _top_features() (forecast_service.py)
    # iterates via `for item in metadata.get("feature_importance", [])`.
    real_feature_importance = [{"feature": "a", "importance": 0.73}]
    config_path = _write_fake_promoted_model(tmp_path, real_feature_importance)

    service = ForecastService.__new__(ForecastService)
    service.config_path = config_path
    service.targets = ["home_goals"]
    service.feature_names = ["a"]

    loaded = service._load_context_models("E0")

    definition, _, metadata = loaded["home_goals"]
    assert metadata["feature_importance"] == real_feature_importance

    # End-to-end: confirm _top_features() itself (the actual consumer that
    # populates forecast_payload.explainability.top_features) now produces a
    # real entry from this metadata, not just that the key is present.
    row = pd.Series({"a": 2.5})
    top = ForecastService._top_features(row, {"home_goals": metadata})
    assert top == [{"name": "a", "value": 2.5, "importance": 0.73}]


def test_load_context_models_feature_importance_is_none_when_metadata_json_absent(tmp_path: Path) -> None:
    """No .metadata.json sidecar at all (an older/manually-placed artifact)
    -- must not crash, feature_importance is just None like every other
    optional field here (calibrator, metrics, feature_subset)."""
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
    assert metadata["feature_importance"] is None
