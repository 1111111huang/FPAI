"""Tests for LUCK_HOME/AWAY_BURNOUT_R5 team-level luck burnout features (US#106)."""

from __future__ import annotations

from pathlib import Path
import sys

import pandas as pd
import pytest

sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.features.feature_factory import FeatureFactory


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def _raw_matches_df() -> pd.DataFrame:
    """Minimal raw_matches frame: 7 matches, Arsenal (home) vs Everton (away)."""
    rows = []
    for i in range(1, 8):
        rows.append({
            "match_id": f"m{i}",
            "date": f"2024-08-{i:02d}",
            "home_team": "Arsenal",
            "away_team": "Everton",
        })
    return pd.DataFrame(rows)


def _player_df_full(n_matches: int = 6) -> pd.DataFrame:
    """
    Build a synthetic player_df where each team plays n_matches games.

    Arsenal: goals=2, assists=1, xg=1.0, xa=0.5 per match → luck = 2+1-1.0-0.5 = 1.5
    Everton: goals=1, assists=0, xg=1.2, xa=0.3 per match → luck = 1+0-1.2-0.3 = -0.5
    """
    rows = []
    for i in range(1, n_matches + 1):
        match_id = f"m{i}"
        rows.append({"match_id": match_id, "team_name": "Arsenal",
                     "goals": 2, "assists": 1, "xg": 1.0, "xa": 0.5})
        rows.append({"match_id": match_id, "team_name": "Everton",
                     "goals": 1, "assists": 0, "xg": 1.2, "xa": 0.3})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Test 1: absent table → empty DataFrame with only match_id column
# ---------------------------------------------------------------------------

def test_empty_player_stats_returns_empty() -> None:
    """When player_df is empty, _luck_burnout_from_data returns match_id-only DataFrame."""
    empty = pd.DataFrame(columns=["match_id", "team_name", "goals", "assists", "xg", "xa"])
    result = FeatureFactory._luck_burnout_from_data(empty, _raw_matches_df())

    assert result.empty or (len(result.columns) == 1 and result.columns[0] == "match_id"), (
        f"Expected empty or match_id-only frame, got columns={list(result.columns)}"
    )


# ---------------------------------------------------------------------------
# Test 2: correct rolling mean computation
# ---------------------------------------------------------------------------

def test_luck_burnout_computation() -> None:
    """For match 6, rolling R5 mean over matches 1–5 equals the constant per-match luck."""
    player_df = _player_df_full(n_matches=6)
    raw_df = _raw_matches_df()

    result = FeatureFactory._luck_burnout_from_data(player_df, raw_df)

    # Arsenal per-match luck = 2 + 1 - 1.0 - 0.5 = 1.5, constant across m1–m5
    # Everton per-match luck = 1 + 0 - 1.2 - 0.3 = -0.5, constant across m1–m5
    row_m6 = result[result["match_id"] == "m6"].iloc[0]
    assert row_m6["LUCK_HOME_BURNOUT_R5"] == pytest.approx(1.5, abs=1e-4), (
        f"Expected LUCK_HOME_BURNOUT_R5=1.5 at m6, got {row_m6['LUCK_HOME_BURNOUT_R5']}"
    )
    assert row_m6["LUCK_AWAY_BURNOUT_R5"] == pytest.approx(-0.5, abs=1e-4), (
        f"Expected LUCK_AWAY_BURNOUT_R5=-0.5 at m6, got {row_m6['LUCK_AWAY_BURNOUT_R5']}"
    )


# ---------------------------------------------------------------------------
# Test 3: shift(1) is applied — match 1 has NaN luck (no prior data)
# ---------------------------------------------------------------------------

def test_shift_is_applied() -> None:
    """Match 1 is the first appearance for each team; rolling value must be NaN (no prior match)."""
    player_df = _player_df_full(n_matches=3)
    raw_df = _raw_matches_df()

    result = FeatureFactory._luck_burnout_from_data(player_df, raw_df)

    row_m1 = result[result["match_id"] == "m1"].iloc[0]
    assert pd.isna(row_m1["LUCK_HOME_BURNOUT_R5"]), (
        f"Expected NaN for LUCK_HOME_BURNOUT_R5 at m1, got {row_m1['LUCK_HOME_BURNOUT_R5']}"
    )
    assert pd.isna(row_m1["LUCK_AWAY_BURNOUT_R5"]), (
        f"Expected NaN for LUCK_AWAY_BURNOUT_R5 at m1, got {row_m1['LUCK_AWAY_BURNOUT_R5']}"
    )


# ---------------------------------------------------------------------------
# Test 4: home/away assignment uses the correct team's luck
# ---------------------------------------------------------------------------

def test_home_away_assignment() -> None:
    """
    LUCK_HOME_BURNOUT_R5 must reflect the home team's luck and
    LUCK_AWAY_BURNOUT_R5 must reflect the away team's luck — they must differ.
    """
    player_df = _player_df_full(n_matches=6)
    raw_df = _raw_matches_df()

    result = FeatureFactory._luck_burnout_from_data(player_df, raw_df)

    # At m6 the two values differ (Arsenal luck=1.5 vs Everton luck=-0.5)
    row_m6 = result[result["match_id"] == "m6"].iloc[0]
    assert row_m6["LUCK_HOME_BURNOUT_R5"] != row_m6["LUCK_AWAY_BURNOUT_R5"], (
        "LUCK_HOME_BURNOUT_R5 and LUCK_AWAY_BURNOUT_R5 must differ when teams have different luck"
    )
    # HOME reflects Arsenal (home_team), AWAY reflects Everton (away_team)
    assert row_m6["LUCK_HOME_BURNOUT_R5"] > 0, "Arsenal overperforms xG — HOME burnout should be positive"
    assert row_m6["LUCK_AWAY_BURNOUT_R5"] < 0, "Everton underperforms xG — AWAY burnout should be negative"


# ---------------------------------------------------------------------------
# Test 4b: carry-forward for a match without its own player-stats row (BUG-012 layer 2)
# ---------------------------------------------------------------------------

def test_carries_forward_for_match_without_own_player_stats() -> None:
    """m7 exists in raw_matches but has no row of its own in player_df (only
    m1-m6 do, per _player_df_full's default) — exactly the shape of
    build_for_match()'s synthetic upcoming-match row. LUCK_*_BURNOUT_R5 going
    into m7 must still reflect Arsenal/Everton's last 5 recorded matches
    (m2-m6), not NaN."""
    player_df = _player_df_full(n_matches=6)
    raw_df = _raw_matches_df()  # m1..m7

    result = FeatureFactory._luck_burnout_from_data(player_df, raw_df)

    row_m7 = result[result["match_id"] == "m7"].iloc[0]
    assert row_m7["LUCK_HOME_BURNOUT_R5"] == pytest.approx(1.5, abs=1e-4), (
        f"Expected carried-forward LUCK_HOME_BURNOUT_R5=1.5 at m7, got {row_m7['LUCK_HOME_BURNOUT_R5']}"
    )
    assert row_m7["LUCK_AWAY_BURNOUT_R5"] == pytest.approx(-0.5, abs=1e-4), (
        f"Expected carried-forward LUCK_AWAY_BURNOUT_R5=-0.5 at m7, got {row_m7['LUCK_AWAY_BURNOUT_R5']}"
    )


# ---------------------------------------------------------------------------
# Test 5: team name normalisation (abbreviated → canonical)
# ---------------------------------------------------------------------------

def test_team_name_normalisation() -> None:
    """'Man City' in player stats must match 'Manchester City' in raw_matches."""
    raw = pd.DataFrame([
        {"match_id": "m1", "date": "2024-08-10", "home_team": "Manchester City", "away_team": "Arsenal"},
        {"match_id": "m2", "date": "2024-08-17", "home_team": "Manchester City", "away_team": "Arsenal"},
    ])
    players = pd.DataFrame([
        {"match_id": "m1", "team_name": "Man City",  "goals": 3, "assists": 2, "xg": 2.0, "xa": 1.0},
        {"match_id": "m1", "team_name": "Arsenal",   "goals": 1, "assists": 1, "xg": 1.5, "xa": 0.5},
        {"match_id": "m2", "team_name": "Man City",  "goals": 2, "assists": 1, "xg": 1.8, "xa": 0.7},
        {"match_id": "m2", "team_name": "Arsenal",   "goals": 0, "assists": 0, "xg": 0.8, "xa": 0.2},
    ])

    result = FeatureFactory._luck_burnout_from_data(players, raw)

    row_m2 = result[result["match_id"] == "m2"].iloc[0]
    # m1 Manchester City luck = 3+2-2.0-1.0 = 2.0; R5 at m2 should be 2.0 (1 prior match)
    assert row_m2["LUCK_HOME_BURNOUT_R5"] == pytest.approx(2.0, abs=1e-4), (
        f"Expected Man City luck 2.0 at m2, got {row_m2['LUCK_HOME_BURNOUT_R5']}"
    )
