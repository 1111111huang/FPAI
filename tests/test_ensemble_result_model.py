"""Tests for EnsembleResultModel (US#188) -- per direct user prioritization,
item #5 of "Do 8, 1, 3, 5, 6 in that order": ensemble the multiple
result_3way architectures now built (plain XGBoost classifier,
TwoStageResultModel, SkellamResultModel) instead of picking one champion.

Each member already conforms to FPAIBaseModel's predict_proba(X) contract
and shares the same alphabetical ["away", "draw", "home"] class order (a
LabelEncoder on string labels sorts alphabetically; TwoStageResultModel/
SkellamResultModel hardcode the same order to match) -- so this is a
straight equal-weight average of already-comparable probability vectors,
not a rescue/realignment mechanism.

Dixon-Coles is NOT included as a fourth member here: unlike the other
three, it predicts from team-name identity (predict_match(home, away)),
not a numeric feature row -- folding it into this predict_proba(X)
contract would mean piping team-identity strings through
ForecastService's shared numeric-coercion serving path for every target,
not just this one. Evaluated separately, offline, see
documents/user_stories.md US#188's completion notes for the real result.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.models.ensemble_result_model import EnsembleResultModel
from src.models.skellam_result_model import SkellamResultModel
from src.models.base_model import XGBoostRegressorModel


def _make_matches(n: int = 300, seed: int = 42, draw_rate: float = 0.25) -> tuple[pd.DataFrame, pd.Series]:
    rng = np.random.default_rng(seed)
    strength_diff = rng.normal(0, 1, size=n)
    is_draw = rng.random(n) < draw_rate
    home_wins = (strength_diff + rng.normal(0, 0.5, size=n)) > 0
    labels = np.where(is_draw, "draw", np.where(home_wins, "home", "away"))
    X = pd.DataFrame({
        "strength_diff": strength_diff,
        "noise": rng.normal(0, 1, size=n),
    })
    return X, pd.Series(labels)


def _write_fake_goal_submodels(tmp_path: Path) -> Path:
    """Mirrors test_skellam_result_model.py's fixture -- a real
    model_selection.yaml pointing at real tiny home_goals/away_goals
    artifacts, so the Skellam member can actually load something."""
    rng = np.random.default_rng(0)
    X = pd.DataFrame({"strength_diff": rng.normal(size=60), "noise": rng.normal(size=60)})
    y_home = pd.Series(rng.poisson(1.5, size=60).astype(float))
    y_away = pd.Series(rng.poisson(1.1, size=60).astype(float))

    home_model = XGBoostRegressorModel(n_estimators=20, max_depth=2, early_stopping_rounds=None)
    home_model.train(X, y_home)
    away_model = XGBoostRegressorModel(n_estimators=20, max_depth=2, early_stopping_rounds=None)
    away_model.train(X, y_away)

    home_path = tmp_path / "home_goals_fake.joblib"
    away_path = tmp_path / "away_goals_fake.joblib"
    home_model.save(str(home_path))
    away_model.save(str(away_path))

    selection = {
        "contexts": {
            "E0": {
                "home_goals": {"model_path": str(home_path), "feature_subset": ["strength_diff", "noise"]},
                "away_goals": {"model_path": str(away_path), "feature_subset": ["strength_diff", "noise"]},
            }
        }
    }
    selection_path = tmp_path / "model_selection.yaml"
    import yaml
    with selection_path.open("w", encoding="utf-8") as fh:
        yaml.safe_dump(selection, fh)
    return selection_path


class TestEnsembleResultModelTrain:
    def test_classes_are_alphabetical(self):
        model = EnsembleResultModel()
        X, y = _make_matches(200)
        model.train(X, y)
        assert list(model.classes_) == ["away", "draw", "home"]

    def test_predict_proba_shape_and_sums_to_one(self, tmp_path: Path):
        selection_path = _write_fake_goal_submodels(tmp_path)
        model = EnsembleResultModel(competition_id="E0", model_selection_path=str(selection_path))
        X, y = _make_matches(200)
        model.train(X, y)
        proba = model.predict_proba(X)
        assert proba.shape == (len(X), 3)
        assert np.allclose(proba.sum(axis=1), 1.0)

    def test_predict_returns_known_classes(self, tmp_path: Path):
        selection_path = _write_fake_goal_submodels(tmp_path)
        model = EnsembleResultModel(competition_id="E0", model_selection_path=str(selection_path))
        X, y = _make_matches(200)
        model.train(X, y)
        preds = model.predict(X)
        assert set(preds) <= {"away", "draw", "home"}

    def test_is_genuine_average_not_a_single_members_output(self, tmp_path: Path):
        """The ensemble's own prediction must differ from any single
        member's prediction in general -- otherwise it's not averaging."""
        selection_path = _write_fake_goal_submodels(tmp_path)
        model = EnsembleResultModel(competition_id="E0", model_selection_path=str(selection_path))
        X, y = _make_matches(200)
        model.train(X, y)
        ensemble_proba = model.predict_proba(X)

        xgb_only = model.members["xgboost"].predict_proba(X)
        assert not np.allclose(ensemble_proba, xgb_only)

    def test_train_with_eval_set_does_not_crash_on_3_classes(self, tmp_path: Path):
        """Regression test: found live via a real train-target run -- the
        xgboost member's default binary objective mismatches result_3way's
        3 classes only once an eval_set is actually scored (ModelManager's
        real run_pipeline() always passes one; this test's earlier fixtures
        never did, so this crash never surfaced there)."""
        selection_path = _write_fake_goal_submodels(tmp_path)
        model = EnsembleResultModel(competition_id="E0", model_selection_path=str(selection_path))
        X, y = _make_matches(200)
        X_val, y_val = _make_matches(50, seed=99)
        model.train(X, y, eval_set=[(X_val, y_val)])
        proba = model.predict_proba(X)
        assert proba.shape == (len(X), 3)

    def test_skellam_member_loads_from_model_selection_yaml(self, tmp_path: Path):
        selection_path = _write_fake_goal_submodels(tmp_path)
        model = EnsembleResultModel(competition_id="E0", model_selection_path=str(selection_path))
        X, y = _make_matches(200)
        model.train(X, y)
        assert isinstance(model.members["skellam"], SkellamResultModel)
        assert model.members["skellam"].home_model is not None


class TestEnsembleResultModelPersistence:
    def test_save_load_roundtrip_still_predicts(self, tmp_path: Path):
        selection_path = _write_fake_goal_submodels(tmp_path)
        model = EnsembleResultModel(competition_id="E0", model_selection_path=str(selection_path))
        X, y = _make_matches(200)
        model.train(X, y)
        before = model.predict_proba(X)

        save_path = tmp_path / "ensemble_result.joblib"
        model.save(str(save_path))
        loaded = EnsembleResultModel.load(str(save_path))
        after = loaded.predict_proba(X)

        assert np.allclose(before, after)
        assert list(loaded.classes_) == ["away", "draw", "home"]
