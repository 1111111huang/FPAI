"""Tests for Dixon-Coles walk-forward stacking features — US#174.

`DixonColesModel` is fit and compared as a standalone baseline
(`reports/model_comparison/dixon_coles_comparison.csv`) but never fed into
XGBoost as a feature. A single whole-history fit (as that baseline uses)
would leak future results into every row's attack/defence ratings, so
`FeatureFactory._compute_dixon_coles_features` refits once per (league,
calendar month) on strictly prior matches only.
"""

from __future__ import annotations

from datetime import date, timedelta

import numpy as np
import pandas as pd
import pytest

from src.features.feature_factory import FeatureFactory

_DC_COLUMNS = [
    "DC_LAMBDA_HOME", "DC_LAMBDA_AWAY",
    "DC_ATTACK_HOME", "DC_DEFENSE_HOME",
    "DC_ATTACK_AWAY", "DC_DEFENSE_AWAY",
]

_DC_CORNER_COLUMNS = [
    "DC_CORNER_LAMBDA_HOME", "DC_CORNER_LAMBDA_AWAY",
    "DC_CORNER_ATTACK_HOME", "DC_CORNER_DEFENSE_HOME",
    "DC_CORNER_ATTACK_AWAY", "DC_CORNER_DEFENSE_AWAY",
]


def _round_robin_matches(
    start: date, n_days: int, teams: list[str], seed: int = 0, with_corners: bool = True,
) -> pd.DataFrame:
    """One synthetic match per day for n_days, deterministic pseudo-random scores."""
    rng = np.random.default_rng(seed)
    rows = []
    for i in range(n_days):
        h, a = rng.choice(teams, size=2, replace=False)
        row = {
            "match_id": f"m{i}",
            "league": "E0",
            "date": start + timedelta(days=i),
            "home_team": h,
            "away_team": a,
            "fthg": int(rng.poisson(1.4)),
            "ftag": int(rng.poisson(1.1)),
        }
        if with_corners:
            row["hc"] = int(rng.poisson(5.5))
            row["ac"] = int(rng.poisson(4.5))
        rows.append(row)
    return pd.DataFrame(rows)


def test_dixon_coles_features_nan_for_first_month_with_no_history():
    """The very first calendar month in the dataset has zero strictly-prior
    matches to fit on -> NaN, not a crash or a spuriously-fit model."""
    teams = ["Arsenal", "Chelsea", "Liverpool", "Everton"]
    raw_df = _round_robin_matches(date(2020, 1, 1), 20, teams)

    result = FeatureFactory._compute_dixon_coles_features(raw_df)
    for col in _DC_COLUMNS:
        assert col in result.columns
    first_match = result.loc[result["match_id"] == "m0"].iloc[0]
    for col in _DC_COLUMNS:
        assert pd.isna(first_match[col])


def test_dixon_coles_features_populated_once_enough_prior_history_exists():
    """A match in a later month, once 40+ strictly-prior matches exist, gets
    real (finite, non-NaN) stacking features."""
    teams = ["Arsenal", "Chelsea", "Liverpool", "Everton", "Spurs", "ManCity"]
    # January: 31 matches (not enough for the 40-match default floor).
    jan = _round_robin_matches(date(2020, 1, 1), 31, teams, seed=1)
    # February: another 28 matches -> by March 1st, 59 strictly-prior matches exist.
    feb = _round_robin_matches(date(2020, 2, 1), 28, teams, seed=2)
    march_target = pd.DataFrame([{
        "match_id": "target", "league": "E0", "date": date(2020, 3, 1),
        "home_team": "Arsenal", "away_team": "Chelsea", "fthg": 1, "ftag": 1,
    }])
    raw_df = pd.concat([jan, feb, march_target], ignore_index=True)

    result = FeatureFactory._compute_dixon_coles_features(raw_df)
    row = result.loc[result["match_id"] == "target"].iloc[0]
    for col in _DC_COLUMNS:
        assert np.isfinite(row[col]), f"{col} should be a real finite value, got {row[col]}"
    assert row["DC_LAMBDA_HOME"] > 0
    assert row["DC_LAMBDA_AWAY"] > 0


def test_dixon_coles_features_are_leakage_safe_against_future_results():
    """Changing a match's own -- and any *later* match's -- score must not
    change an EARLIER match's DC_* features. This is the core walk-forward
    guarantee: only strictly-prior months are ever used to fit."""
    teams = ["Arsenal", "Chelsea", "Liverpool", "Everton", "Spurs", "ManCity"]
    jan = _round_robin_matches(date(2020, 1, 1), 31, teams, seed=1)
    feb = _round_robin_matches(date(2020, 2, 1), 28, teams, seed=2)
    target = pd.DataFrame([{
        "match_id": "target", "league": "E0", "date": date(2020, 3, 1),
        "home_team": "Arsenal", "away_team": "Chelsea", "fthg": 1, "ftag": 1,
    }])
    # A future match, strictly after the target's date, with an extreme score.
    future_normal = pd.DataFrame([{
        "match_id": "future", "league": "E0", "date": date(2020, 3, 15),
        "home_team": "Liverpool", "away_team": "Everton", "fthg": 1, "ftag": 1,
    }])
    future_extreme = future_normal.copy()
    future_extreme.loc[0, ["fthg", "ftag"]] = [9, 0]

    base = pd.concat([jan, feb, target, future_normal], ignore_index=True)
    altered = pd.concat([jan, feb, target, future_extreme], ignore_index=True)

    result_base = FeatureFactory._compute_dixon_coles_features(base)
    result_altered = FeatureFactory._compute_dixon_coles_features(altered)

    row_base = result_base.loc[result_base["match_id"] == "target"].iloc[0]
    row_altered = result_altered.loc[result_altered["match_id"] == "target"].iloc[0]
    for col in _DC_COLUMNS:
        assert row_base[col] == pytest.approx(row_altered[col]), (
            f"{col} changed when a LATER match's score changed -- leakage"
        )


def test_dixon_coles_features_empty_input_returns_match_id_only():
    result = FeatureFactory._compute_dixon_coles_features(pd.DataFrame(
        columns=["match_id", "league", "date", "home_team", "away_team", "fthg", "ftag"]
    ))
    assert result.empty or list(result.columns) == ["match_id"]


# ---------------------------------------------------------------------------
# Corner sub-model (US#178) -- reuses the same generic Poisson machinery on
# hc/ac instead of fthg/ftag, since corners are count data just like goals.
# ---------------------------------------------------------------------------

def test_dixon_coles_corner_features_populated_with_sufficient_history():
    teams = ["Arsenal", "Chelsea", "Liverpool", "Everton", "Spurs", "ManCity"]
    jan = _round_robin_matches(date(2020, 1, 1), 31, teams, seed=1)
    feb = _round_robin_matches(date(2020, 2, 1), 28, teams, seed=2)
    march_target = pd.DataFrame([{
        "match_id": "target", "league": "E0", "date": date(2020, 3, 1),
        "home_team": "Arsenal", "away_team": "Chelsea", "fthg": 1, "ftag": 1,
        "hc": 6, "ac": 4,
    }])
    raw_df = pd.concat([jan, feb, march_target], ignore_index=True)

    result = FeatureFactory._compute_dixon_coles_features(raw_df)
    row = result.loc[result["match_id"] == "target"].iloc[0]
    for col in _DC_CORNER_COLUMNS:
        assert col in result.columns
        assert np.isfinite(row[col]), f"{col} should be a real finite value, got {row[col]}"
    assert row["DC_CORNER_LAMBDA_HOME"] > 0
    assert row["DC_CORNER_LAMBDA_AWAY"] > 0


def test_dixon_coles_corner_features_nan_when_no_corner_data():
    """A goals-only competition (Sweden stand-in: hc/ac absent entirely)
    must still compute the goals DC_* columns fine, with only the corner
    sub-model NaN -- one missing data source must not sink the other."""
    teams = ["Arsenal", "Chelsea", "Liverpool", "Everton", "Spurs", "ManCity"]
    jan = _round_robin_matches(date(2020, 1, 1), 31, teams, seed=1, with_corners=False)
    feb = _round_robin_matches(date(2020, 2, 1), 28, teams, seed=2, with_corners=False)
    march_target = pd.DataFrame([{
        "match_id": "target", "league": "E0", "date": date(2020, 3, 1),
        "home_team": "Arsenal", "away_team": "Chelsea", "fthg": 1, "ftag": 1,
    }])
    raw_df = pd.concat([jan, feb, march_target], ignore_index=True)

    result = FeatureFactory._compute_dixon_coles_features(raw_df)
    row = result.loc[result["match_id"] == "target"].iloc[0]
    for col in _DC_COLUMNS:
        assert np.isfinite(row[col]), f"goals feature {col} should be unaffected by missing corners"
    for col in _DC_CORNER_COLUMNS:
        assert pd.isna(row[col]), f"{col} should be NaN with no corner data at all"


def test_dixon_coles_corner_features_are_leakage_safe_against_future_results():
    teams = ["Arsenal", "Chelsea", "Liverpool", "Everton", "Spurs", "ManCity"]
    jan = _round_robin_matches(date(2020, 1, 1), 31, teams, seed=1)
    feb = _round_robin_matches(date(2020, 2, 1), 28, teams, seed=2)
    target = pd.DataFrame([{
        "match_id": "target", "league": "E0", "date": date(2020, 3, 1),
        "home_team": "Arsenal", "away_team": "Chelsea", "fthg": 1, "ftag": 1,
        "hc": 6, "ac": 4,
    }])
    future_normal = pd.DataFrame([{
        "match_id": "future", "league": "E0", "date": date(2020, 3, 15),
        "home_team": "Liverpool", "away_team": "Everton", "fthg": 1, "ftag": 1,
        "hc": 5, "ac": 5,
    }])
    future_extreme = future_normal.copy()
    future_extreme.loc[0, ["hc", "ac"]] = [14, 0]

    base = pd.concat([jan, feb, target, future_normal], ignore_index=True)
    altered = pd.concat([jan, feb, target, future_extreme], ignore_index=True)

    result_base = FeatureFactory._compute_dixon_coles_features(base)
    result_altered = FeatureFactory._compute_dixon_coles_features(altered)

    row_base = result_base.loc[result_base["match_id"] == "target"].iloc[0]
    row_altered = result_altered.loc[result_altered["match_id"] == "target"].iloc[0]
    for col in _DC_CORNER_COLUMNS:
        assert row_base[col] == pytest.approx(row_altered[col]), (
            f"{col} changed when a LATER match's corner score changed -- leakage"
        )
