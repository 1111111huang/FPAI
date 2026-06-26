"""Tests for the Dixon-Coles baseline model (US#73)."""

from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.models.dixon_coles import DixonColesModel, _tau_matrix


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _make_matches(n: int = 200, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    teams = ["Arsenal", "Chelsea", "Liverpool", "ManCity", "Spurs", "Everton"]
    rows = []
    for _ in range(n):
        h, a = rng.choice(teams, size=2, replace=False)
        hg = int(rng.poisson(1.5))
        ag = int(rng.poisson(1.1))
        rows.append({"home_team": h, "away_team": a, "fthg": hg, "ftag": ag,
                     "hc": float(rng.poisson(5)), "ac": float(rng.poisson(4))})
    return pd.DataFrame(rows)


@pytest.fixture(scope="module")
def fitted_model() -> DixonColesModel:
    df = _make_matches(300)
    m = DixonColesModel()
    m.fit(df)
    return m


# ---------------------------------------------------------------------------
# Tau matrix
# ---------------------------------------------------------------------------

class TestTauMatrix:
    def test_shape(self):
        T = _tau_matrix(1.5, 1.1, 0.05)
        assert T.shape == (9, 9)

    def test_off_diagonal_ones(self):
        T = _tau_matrix(1.5, 1.1, 0.05)
        assert T[2, 2] == 1.0
        assert T[3, 1] == 1.0

    def test_zero_rho_is_identity(self):
        T = _tau_matrix(1.5, 1.1, 0.0)
        assert np.allclose(T, np.ones((9, 9)))


# ---------------------------------------------------------------------------
# Fitting
# ---------------------------------------------------------------------------

class TestDixonColesFit:
    def test_fit_sets_teams(self):
        df = _make_matches(100)
        m = DixonColesModel().fit(df)
        expected = sorted(set(df["home_team"]) | set(df["away_team"]))
        assert m.teams_ == expected

    def test_fit_produces_attack_defence_for_each_team(self, fitted_model):
        for team in fitted_model.teams_:
            assert team in fitted_model._attack
            assert team in fitted_model._defence

    def test_home_advantage_positive(self, fitted_model):
        # Home teams score more on average, so home_adv should be > 0
        assert fitted_model._home_adv > 0

    def test_rho_in_bounds(self, fitted_model):
        assert -1.0 < fitted_model._rho < 1.0

    def test_fitted_flag(self, fitted_model):
        assert fitted_model._fitted is True

    def test_predict_before_fit_raises(self):
        with pytest.raises(RuntimeError):
            DixonColesModel().predict_match("Arsenal", "Chelsea")


# ---------------------------------------------------------------------------
# predict_match
# ---------------------------------------------------------------------------

class TestDixonColesPredict:
    def test_result_probs_sum_to_one(self, fitted_model):
        pred = fitted_model.predict_match("Arsenal", "Chelsea")
        total = pred["result_3way_p_home"] + pred["result_3way_p_draw"] + pred["result_3way_p_away"]
        assert abs(total - 1.0) < 1e-6

    def test_btts_prob_in_unit_interval(self, fitted_model):
        pred = fitted_model.predict_match("Liverpool", "ManCity")
        assert 0.0 <= pred["btts_prob"] <= 1.0

    def test_goals_positive(self, fitted_model):
        pred = fitted_model.predict_match("Chelsea", "Spurs")
        assert pred["home_goals"] > 0
        assert pred["away_goals"] > 0
        assert abs(pred["total_goals"] - pred["home_goals"] - pred["away_goals"]) < 1e-9

    def test_result_pred_is_valid_class(self, fitted_model):
        pred = fitted_model.predict_match("Everton", "Arsenal")
        assert pred["result_3way_pred"] in {"home", "draw", "away"}

    def test_btts_pred_binary(self, fitted_model):
        pred = fitted_model.predict_match("Arsenal", "Liverpool")
        assert pred["btts_pred"] in {0, 1}

    def test_unseen_team_uses_mean_params(self, fitted_model):
        pred = fitted_model.predict_match("NewTeamFC", "Arsenal")
        assert pred["home_goals"] > 0
        assert pred["away_goals"] > 0

    def test_corner_predictions_present(self, fitted_model):
        pred = fitted_model.predict_match("Arsenal", "Chelsea")
        assert "home_corners" in pred
        assert "away_corners" in pred
        assert "total_corners" in pred

    def test_predict_batch_length(self, fitted_model):
        df = _make_matches(20)
        preds = fitted_model.predict_batch(df)
        assert len(preds) == 20
        assert "result_3way_p_home" in preds.columns


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------

class TestDixonColesPersistence:
    def test_save_load_roundtrip(self, fitted_model):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = str(Path(tmpdir) / "dc.joblib")
            fitted_model.save(path)
            loaded = DixonColesModel.load(path)
            assert loaded.teams_ == fitted_model.teams_
            assert abs(loaded._rho - fitted_model._rho) < 1e-9
            pred_orig = fitted_model.predict_match("Arsenal", "Chelsea")
            pred_load = loaded.predict_match("Arsenal", "Chelsea")
            assert abs(pred_orig["home_goals"] - pred_load["home_goals"]) < 1e-9
