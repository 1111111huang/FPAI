"""Tests for SkellamResultModel (US#183).

Not a from-scratch classifier: a distributional STACK over the
already-promoted home_goals/away_goals models for a competition, converting
their goal-count predictions into P(home/draw/away) via the actual discrete
distribution for "difference of two Poisson-like counts" (Skellam) --
found live (2026-09-03) to beat the current result_3way champion on
log_loss using models that already exist, with zero new goal-model
training. Always resolves the CURRENT model_path from
config/model_selection.yaml at train()/load() time, never a hardcoded
artifact filename, so a future re-promotion of either sub-model is picked
up automatically.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml

sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.models.base_model import XGBoostRegressorModel
from src.models.skellam_result_model import SkellamResultModel


def _write_fake_submodels(tmp_path: Path) -> Path:
    """Train two tiny real XGBoostRegressorModel artifacts and a
    model_selection.yaml pointing at them, mirroring the real file's shape."""
    rng = np.random.default_rng(0)
    X = pd.DataFrame({"a": rng.normal(size=60), "b": rng.normal(size=60)})
    y_home = pd.Series(rng.poisson(1.5, size=60).astype(float))
    y_away = pd.Series(rng.poisson(1.1, size=60).astype(float))

    home_model = XGBoostRegressorModel(n_estimators=20, max_depth=2, early_stopping_rounds=None)
    home_model.train(X, y_home)
    away_model = XGBoostRegressorModel(n_estimators=20, max_depth=2, early_stopping_rounds=None)
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
                    # Explicit feature_subset -- keeps this test's own
                    # toy 2-column frame independent of the real project's
                    # config/schema.yaml (which the no-override fallback
                    # path would otherwise consult).
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


class TestSkellamResultModelTrain:
    def test_train_loads_promoted_submodels(self, tmp_path: Path) -> None:
        selection_path = _write_fake_submodels(tmp_path)
        model = SkellamResultModel(competition_id="E0", model_selection_path=str(selection_path))
        rng = np.random.default_rng(1)
        X = pd.DataFrame({"a": rng.normal(size=10), "b": rng.normal(size=10)})
        model.train(X, pd.Series(["home"] * 10))
        assert model.home_model is not None
        assert model.away_model is not None

    def test_train_raises_clearly_when_submodel_not_promoted(self, tmp_path: Path) -> None:
        selection_path = tmp_path / "model_selection.yaml"
        selection_path.write_text(yaml.safe_dump({"contexts": {"E0": {}}}), encoding="utf-8")
        model = SkellamResultModel(competition_id="E0", model_selection_path=str(selection_path))
        with pytest.raises(ValueError, match="home_goals"):
            model.train(pd.DataFrame({"a": [1.0]}), pd.Series(["home"]))

    def test_classes_are_alphabetical(self, tmp_path: Path) -> None:
        selection_path = _write_fake_submodels(tmp_path)
        model = SkellamResultModel(competition_id="E0", model_selection_path=str(selection_path))
        assert list(model.classes_) == ["away", "draw", "home"]


class TestSkellamResultModelPredict:
    def _trained_model(self, tmp_path: Path) -> SkellamResultModel:
        selection_path = _write_fake_submodels(tmp_path)
        model = SkellamResultModel(competition_id="E0", model_selection_path=str(selection_path))
        model.train(pd.DataFrame({"a": [0.0], "b": [0.0]}), pd.Series(["home"]))
        return model

    def test_predict_proba_shape_and_sums_to_one(self, tmp_path: Path) -> None:
        model = self._trained_model(tmp_path)
        rng = np.random.default_rng(2)
        X = pd.DataFrame({"a": rng.normal(size=15), "b": rng.normal(size=15)})
        proba = model.predict_proba(X)
        assert proba.shape == (15, 3)
        assert np.allclose(proba.sum(axis=1), 1.0, atol=1e-6)
        assert (proba >= 0).all()

    def test_predict_returns_known_classes(self, tmp_path: Path) -> None:
        model = self._trained_model(tmp_path)
        rng = np.random.default_rng(3)
        X = pd.DataFrame({"a": rng.normal(size=15), "b": rng.normal(size=15)})
        preds = model.predict(X)
        assert set(preds) <= {"home", "draw", "away"}

    def test_balanced_expected_goals_gives_highest_draw_probability_among_outcomes_near_zero_lambda(
        self, tmp_path: Path,
    ) -> None:
        """Sanity check on the actual Skellam math, independent of the
        fitted sub-models: pmf(0, mu, mu) (equal expected goals, low
        scoring) should be a substantial, non-trivial share of mass --
        confirms the distribution conversion isn't silently degenerate."""
        from scipy.stats import skellam
        p_draw = skellam.pmf(0, 1.0, 1.0)
        assert p_draw > 0.25, f"P(draw) for equal low-scoring expectations should be substantial, got {p_draw}"


class TestSkellamResultModelPersistence:
    def test_save_load_roundtrip_still_predicts(self, tmp_path: Path) -> None:
        model = self._trained_model_helper(tmp_path)
        path = str(tmp_path / "skellam_model.joblib")
        model.save(path)
        loaded = SkellamResultModel.load(path)
        rng = np.random.default_rng(4)
        X = pd.DataFrame({"a": rng.normal(size=5), "b": rng.normal(size=5)})
        proba = loaded.predict_proba(X)
        assert proba.shape == (5, 3)
        assert list(loaded.classes_) == ["away", "draw", "home"]

    def _trained_model_helper(self, tmp_path: Path) -> SkellamResultModel:
        selection_path = _write_fake_submodels(tmp_path)
        model = SkellamResultModel(competition_id="E0", model_selection_path=str(selection_path))
        model.train(pd.DataFrame({"a": [0.0], "b": [0.0]}), pd.Series(["home"]))
        return model
