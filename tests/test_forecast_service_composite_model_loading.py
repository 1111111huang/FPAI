"""Regression test: ForecastService._load_model must reconstruct composite
models (SkellamResultModel, TwoStageResultModel) via their own .load()
classmethod, not a raw joblib.load() -- their .save() dumps a config/state
dict (not the underlying sklearn/XGBoost estimator directly, unlike
LRModel/RandomForestModel), so the generic joblib fallback returns a plain
dict with no .predict_proba() at all.

Found live (2026-09-03): promoting SkellamResultModel into
model_selection.yaml and calling ForecastService.forecast_upcoming crashed
with "'dict' object has no attribute 'predict_proba'" -- confirmed the
generic fallback never dispatches to SkellamResultModel.load() at all.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml

sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.forecast.forecast_service import ForecastService
from src.models.base_model import XGBoostRegressorModel
from src.models.skellam_result_model import SkellamResultModel
from src.models.two_stage_result_model import TwoStageResultModel


def _write_fake_goal_submodels(tmp_path: Path) -> Path:
    rng = np.random.default_rng(0)
    X = pd.DataFrame({"a": rng.normal(size=40), "b": rng.normal(size=40)})
    y_home = pd.Series(rng.poisson(1.5, size=40).astype(float))
    y_away = pd.Series(rng.poisson(1.1, size=40).astype(float))

    home_model = XGBoostRegressorModel(n_estimators=15, max_depth=2, early_stopping_rounds=None)
    home_model.train(X, y_home)
    away_model = XGBoostRegressorModel(n_estimators=15, max_depth=2, early_stopping_rounds=None)
    away_model.train(X, y_away)

    models_dir = tmp_path / "models"
    models_dir.mkdir()
    home_path = models_dir / "home_goals_fake.joblib"
    away_path = models_dir / "away_goals_fake.joblib"
    home_model.save(str(home_path))
    away_model.save(str(away_path))

    selection_path = tmp_path / "model_selection.yaml"
    selection_path.write_text(
        yaml.safe_dump({
            "contexts": {
                "E0": {
                    "home_goals": {
                        "model_path": str(home_path), "model_type": "xgb_regressor",
                        "feature_subset": ["a", "b"],
                    },
                    "away_goals": {
                        "model_path": str(away_path), "model_type": "xgb_regressor",
                        "feature_subset": ["a", "b"],
                    },
                }
            }
        }),
        encoding="utf-8",
    )
    return selection_path


def test_load_model_reconstructs_skellam_result_model_not_a_raw_dict(tmp_path: Path) -> None:
    selection_path = _write_fake_goal_submodels(tmp_path)
    model = SkellamResultModel(competition_id="E0", model_selection_path=str(selection_path))
    model.train(pd.DataFrame({"a": [0.0], "b": [0.0]}), pd.Series(["home"]))
    model_path = tmp_path / "result_3way_skellam.joblib"
    model.save(str(model_path))

    # "skellamresult" -- the actual string model_manager.py's run_pipeline()
    # writes to model_selection.yaml's model_type field
    # (self.model.__class__.__name__.lower().replace("model", "")), not the
    # bare class name -- confirmed against a real promoted entry.
    loaded = ForecastService._load_model(model_path, {"model_type": "skellamresult"})

    assert isinstance(loaded, SkellamResultModel)
    rng = np.random.default_rng(1)
    X = pd.DataFrame({"a": rng.normal(size=3), "b": rng.normal(size=3)})
    proba = loaded.predict_proba(X)
    assert proba.shape == (3, 3)
    assert np.allclose(proba.sum(axis=1), 1.0, atol=1e-6)


def test_load_model_reconstructs_two_stage_result_model_not_a_raw_dict(tmp_path: Path) -> None:
    model = TwoStageResultModel()
    rng = np.random.default_rng(2)
    X = pd.DataFrame({"strength": rng.normal(size=100)})
    y = pd.Series(np.where(X["strength"] > 0, "home", np.where(X["strength"] < -0.5, "away", "draw")))
    model.train(X, y)
    model_path = tmp_path / "result_3way_two_stage.joblib"
    model.save(str(model_path))

    loaded = ForecastService._load_model(model_path, {"model_type": "twostageresult"})

    assert isinstance(loaded, TwoStageResultModel)
    proba = loaded.predict_proba(X.iloc[:3])
    assert proba.shape == (3, 3)
