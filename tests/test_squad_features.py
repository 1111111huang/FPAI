"""Tests for SQUAD_* rolling feature computation from raw_player_match_stats."""

from __future__ import annotations

from pathlib import Path
import sys

import pandas as pd
import pytest

sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.features.feature_factory import FeatureFactory

SQUAD_FEATURE_NAMES = [
    "SQUAD_HOME_XG_MEAN_R3", "SQUAD_HOME_XG_MEAN_R5",
    "SQUAD_HOME_XA_MEAN_R3", "SQUAD_HOME_XA_MEAN_R5",
    "SQUAD_HOME_RATING_MEAN_R3", "SQUAD_HOME_RATING_MEAN_R5",
    "SQUAD_AWAY_XG_MEAN_R3", "SQUAD_AWAY_XG_MEAN_R5",
    "SQUAD_AWAY_XA_MEAN_R3", "SQUAD_AWAY_XA_MEAN_R5",
    "SQUAD_AWAY_RATING_MEAN_R3", "SQUAD_AWAY_RATING_MEAN_R5",
]


def _raw_matches_df() -> pd.DataFrame:
    """Minimal raw_matches frame: 3 matches, Arsenal (home) vs Everton (away)."""
    return pd.DataFrame([
        {"match_id": "m1", "date": "2024-08-10", "home_team": "Arsenal", "away_team": "Everton"},
        {"match_id": "m2", "date": "2024-08-17", "home_team": "Arsenal", "away_team": "Everton"},
        {"match_id": "m3", "date": "2024-08-24", "home_team": "Arsenal", "away_team": "Everton"},
    ])


def _player_df_two_matches() -> pd.DataFrame:
    """2 players per team for matches m1 and m2."""
    rows = []
    for match_id, xg, xa, rating in [("m1", 0.5, 0.2, 7.0), ("m2", 0.8, 0.4, 7.5)]:
        for team in ["Arsenal", "Everton"]:
            for i in range(2):
                rows.append({
                    "match_id": match_id,
                    "team_name": team,
                    "xg": xg + i * 0.1,
                    "xa": xa + i * 0.05,
                    "rating": rating + i * 0.1,
                })
    return pd.DataFrame(rows)


def test_squad_rolling_returns_correct_columns() -> None:
    result = FeatureFactory._squad_rolling_from_data(_player_df_two_matches(), _raw_matches_df())

    assert "match_id" in result.columns
    for col in SQUAD_FEATURE_NAMES:
        assert col in result.columns, f"Missing column: {col}"


def test_squad_rolling_first_match_is_nan_because_no_prior_data() -> None:
    result = FeatureFactory._squad_rolling_from_data(_player_df_two_matches(), _raw_matches_df())

    # m1 is each team's first match → no prior data → all rolling features are NaN
    row_m1 = result[result["match_id"] == "m1"].iloc[0]
    assert pd.isna(row_m1["SQUAD_HOME_XG_MEAN_R3"])
    assert pd.isna(row_m1["SQUAD_AWAY_RATING_MEAN_R5"])


def test_squad_rolling_second_match_uses_first_match_stats() -> None:
    result = FeatureFactory._squad_rolling_from_data(_player_df_two_matches(), _raw_matches_df())

    # m2 R3/R5 for Arsenal home: mean of m1 Arsenal players' xG
    # m1 Arsenal players: xg=[0.5, 0.6] → mean=0.55
    row_m2 = result[result["match_id"] == "m2"].iloc[0]
    assert row_m2["SQUAD_HOME_XG_MEAN_R3"] == pytest.approx(0.55, abs=1e-3)
    assert row_m2["SQUAD_HOME_XG_MEAN_R5"] == pytest.approx(0.55, abs=1e-3)  # only 1 prior → same value


def test_squad_rolling_carries_forward_for_match_without_own_player_stats() -> None:
    """BUG-012 layer 2: m3 (in raw_matches, per _raw_matches_df) has no row of
    its own in raw_player_match_stats (per _player_df_two_matches, which only
    covers m1/m2) — this is exactly the shape of build_for_match()'s synthetic
    upcoming-match row. The rolling SQUAD_* value going into m3 must still
    reflect Arsenal/Everton's last 2 recorded performances (m1, m2), not NaN —
    an exact-match_id join silently drops any fixture lacking its own
    lineup/player-stats row, even when the team's rolling history exists."""
    result = FeatureFactory._squad_rolling_from_data(_player_df_two_matches(), _raw_matches_df())

    row_m3 = result[result["match_id"] == "m3"].iloc[0]
    # Arsenal (home) xG across m1, m2: [0.5, 0.6, 0.8, 0.9] → mean = 0.7
    assert row_m3["SQUAD_HOME_XG_MEAN_R3"] == pytest.approx(0.7, abs=1e-3), (
        f"Expected carried-forward SQUAD_HOME_XG_MEAN_R3=0.7 at m3, got {row_m3['SQUAD_HOME_XG_MEAN_R3']}"
    )
    assert not pd.isna(row_m3["SQUAD_AWAY_RATING_MEAN_R5"]), (
        "SQUAD_AWAY_RATING_MEAN_R5 at m3 must carry forward Everton's rolling "
        "rating from m1/m2, not fall back to NaN just because m3 itself has no "
        "recorded player stats"
    )


def test_squad_rolling_handles_empty_player_data() -> None:
    empty = pd.DataFrame(columns=["match_id", "team_name", "xg", "xa", "rating"])
    result = FeatureFactory._squad_rolling_from_data(empty, _raw_matches_df())

    assert result.empty or (len(result.columns) == 1 and result.columns[0] == "match_id")


def test_squad_rolling_normalises_abbreviated_team_names() -> None:
    """'Man City' in player stats must match 'Manchester City' in raw_matches."""
    raw = pd.DataFrame([
        {"match_id": "m1", "date": "2024-08-10", "home_team": "Manchester City", "away_team": "Arsenal"},
        {"match_id": "m2", "date": "2024-08-17", "home_team": "Manchester City", "away_team": "Arsenal"},
    ])
    players = pd.DataFrame([
        {"match_id": "m1", "team_name": "Man City", "xg": 0.7, "xa": 0.3, "rating": 7.2},
        {"match_id": "m1", "team_name": "Arsenal", "xg": 0.4, "xa": 0.1, "rating": 6.8},
        {"match_id": "m2", "team_name": "Man City", "xg": 0.9, "xa": 0.5, "rating": 7.8},
        {"match_id": "m2", "team_name": "Arsenal", "xg": 0.3, "xa": 0.2, "rating": 6.5},
    ])

    result = FeatureFactory._squad_rolling_from_data(players, raw)

    row_m2 = result[result["match_id"] == "m2"].iloc[0]
    # m2 home (Man City) R3 should be m1's Man City mean xg = 0.7 (not NaN / not mismatched)
    assert row_m2["SQUAD_HOME_XG_MEAN_R3"] == pytest.approx(0.7, abs=1e-3)


def test_top_features_surfaces_squad_features_when_present() -> None:
    """ForecastService._top_features must include SQUAD_* when they have importance."""
    import sys
    from pathlib import Path as _Path
    sys.path.append(str(_Path(__file__).resolve().parents[1]))
    from src.forecast.forecast_service import ForecastService
    import pandas as pd

    row = pd.Series({
        "MKT_IMPLIED_HOME": 0.45,
        "SQUAD_HOME_XG_MEAN_R5": 0.72,
        "SQUAD_AWAY_RATING_MEAN_R5": 7.1,
        "OFF_HOME_FTHG_R5": 1.8,
    })
    metadata_by_target = {
        "home_goals": {
            "feature_importance": [
                {"feature": "SQUAD_HOME_XG_MEAN_R5", "importance": 0.15},
                {"feature": "MKT_IMPLIED_HOME", "importance": 0.10},
                {"feature": "SQUAD_AWAY_RATING_MEAN_R5", "importance": 0.08},
                {"feature": "OFF_HOME_FTHG_R5", "importance": 0.05},
            ]
        }
    }

    top = ForecastService._top_features(row, metadata_by_target, limit=4)

    names = [f["name"] for f in top]
    assert "SQUAD_HOME_XG_MEAN_R5" in names
    assert "SQUAD_AWAY_RATING_MEAN_R5" in names
    squad_entry = next(f for f in top if f["name"] == "SQUAD_HOME_XG_MEAN_R5")
    assert squad_entry["value"] == pytest.approx(0.72)
    assert squad_entry["importance"] == pytest.approx(0.15, abs=1e-5)
