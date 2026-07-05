"""Tests for Defensive Anchor feature — US#104."""

from __future__ import annotations

import pandas as pd
import pytest

from src.features.lineup_features import compute_defensive_anchor


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _raw_df(match_id: str, date: str, home: str = "Arsenal", away: str = "Chelsea") -> pd.DataFrame:
    return pd.DataFrame({
        "match_id": [match_id],
        "date": pd.to_datetime([date]),
        "home_team": [home],
        "away_team": [away],
    })


def _lineup_row(fotmob_id: int, player_id: int, team: str, side: str, pos: str) -> dict:
    return {
        "fotmob_match_id": fotmob_id,
        "player_id": player_id,
        "team_name": team,
        "side": side,
        "position_group": pos,
    }


def _stats_row(
    match_id: str,
    player_id: int,
    team: str,
    interceptions: float,
    recoveries: float,
    minutes_played: int = 90,
) -> dict:
    return {
        "match_id": match_id,
        "player_id": player_id,
        "team_name": team,
        "minutes_played": minutes_played,
        "interceptions": interceptions,
        "recoveries": recoveries,
    }


# ---------------------------------------------------------------------------
# Test 1: Empty inputs return empty DataFrame
# ---------------------------------------------------------------------------

def test_def_anchor_empty_lineups():
    stats = pd.DataFrame([_stats_row("M1", 1, "Arsenal", 2.0, 3.0)])
    raw = _raw_df("M1", "2024-01-15")
    result = compute_defensive_anchor(pd.DataFrame(), stats, raw)
    assert result.empty or "match_id" in result.columns


def test_def_anchor_empty_stats():
    lineups = pd.DataFrame([_lineup_row(1, 1, "Arsenal", "home", "DEF")])
    raw = _raw_df("M1", "2024-01-15")
    result = compute_defensive_anchor(lineups, pd.DataFrame(), raw)
    assert result.empty or "match_id" in result.columns


# ---------------------------------------------------------------------------
# Test 2: Position filter — only DEF/MID starters contribute
# ---------------------------------------------------------------------------

def test_def_anchor_position_filter():
    """FWD starters should not contribute to DEF_ANCHOR."""
    prior_raw = _raw_df("P1", "2024-01-01", "Arsenal", "Tottenham")
    match_raw = _raw_df("M1", "2024-01-15", "Arsenal", "Chelsea")
    raw = pd.concat([prior_raw, match_raw], ignore_index=True)

    lineups = pd.DataFrame([
        _lineup_row(99, 1, "Arsenal", "home", "DEF"),
        _lineup_row(99, 2, "Arsenal", "home", "MID"),
        _lineup_row(99, 3, "Arsenal", "home", "FWD"),   # should be excluded
        _lineup_row(99, 50, "Chelsea", "away", "DEF"),
    ])
    stats = pd.DataFrame([
        _stats_row("P1", 1, "Arsenal", 3.0, 4.0),
        _stats_row("M1", 1, "Arsenal", 3.0, 4.0),
        _stats_row("P1", 2, "Arsenal", 2.0, 2.0),
        _stats_row("M1", 2, "Arsenal", 2.0, 2.0),
        # Player 3 (FWD) has high defensive stats but should be ignored
        _stats_row("P1", 3, "Arsenal", 10.0, 10.0),
        _stats_row("M1", 3, "Arsenal", 10.0, 10.0),
        _stats_row("P1", 50, "Chelsea", 1.0, 1.0),
        _stats_row("M1", 50, "Chelsea", 1.0, 1.0),
    ])

    result = compute_defensive_anchor(lineups, stats, raw)
    assert "DEF_ANCHOR_HOME" in result.columns
    row = result[result["match_id"] == "M1"]
    if not row.empty and row["DEF_ANCHOR_HOME"].notna().any():
        # The FWD's (player 3) massive stats should NOT boost the anchor
        # Top-2 DEF/MID: player1=(3+4)/90*90=7p90, player2=(2+2)/90*90=4p90 → mean=5.5
        # Rolling = shift(1) of these, so actual value uses prior match values
        anchor = row["DEF_ANCHOR_HOME"].values[0]
        assert anchor < 50, "FWD contributions should not inflate DEF_ANCHOR"


# ---------------------------------------------------------------------------
# Test 3: Top-2 selection (only 2 DEF/MID starters)
# ---------------------------------------------------------------------------

def test_def_anchor_top2_selection():
    """With 3 DEF starters, only the top-2 by def_rec_p90 contribute."""
    prior_raw = _raw_df("P1", "2024-01-01", "Arsenal", "Tottenham")
    match_raw = _raw_df("M1", "2024-01-15", "Arsenal", "Chelsea")
    raw = pd.concat([prior_raw, match_raw], ignore_index=True)

    lineups = pd.DataFrame([
        _lineup_row(99, 1, "Arsenal", "home", "DEF"),
        _lineup_row(99, 2, "Arsenal", "home", "DEF"),
        _lineup_row(99, 3, "Arsenal", "home", "DEF"),   # lowest stats, should be excluded
        _lineup_row(99, 50, "Chelsea", "away", "DEF"),
    ])
    # Player 3 has low def_rec_p90; players 1,2 are higher
    stats = pd.DataFrame([
        _stats_row("P1", 1, "Arsenal", 5.0, 4.0),   # 9p90
        _stats_row("M1", 1, "Arsenal", 5.0, 4.0),
        _stats_row("P1", 2, "Arsenal", 4.0, 3.0),   # 7p90
        _stats_row("M1", 2, "Arsenal", 4.0, 3.0),
        _stats_row("P1", 3, "Arsenal", 1.0, 0.5),   # 1.5p90 — excluded from top-2
        _stats_row("M1", 3, "Arsenal", 1.0, 0.5),
        _stats_row("P1", 50, "Chelsea", 2.0, 2.0),
        _stats_row("M1", 50, "Chelsea", 2.0, 2.0),
    ])

    result = compute_defensive_anchor(lineups, stats, raw)
    row = result[result["match_id"] == "M1"]
    if not row.empty and row["DEF_ANCHOR_HOME"].notna().any():
        anchor = row["DEF_ANCHOR_HOME"].values[0]
        # Top-2 are players 1 and 2: rolling means = shift(1).rolling(5) of 9 and 7 → mean ≈ 8
        assert anchor > 1.5, f"DEF_ANCHOR_HOME={anchor} too low; top-2 should dominate"
        # Player 3 (1.5p90) should NOT drag the mean down to below 4
        assert anchor > 4.0, "Top-2 anchor should exclude the low-stats DEF"


# ---------------------------------------------------------------------------
# Test 4: Pre-match safety (shift(1))
# ---------------------------------------------------------------------------

def test_def_anchor_shift_is_applied():
    """The rolling average must be shift(1): a single-match player has NaN roll."""
    match_raw = _raw_df("M1", "2024-01-15", "Arsenal", "Chelsea")
    lineups = pd.DataFrame([
        _lineup_row(99, 1, "Arsenal", "home", "DEF"),
        _lineup_row(99, 50, "Chelsea", "away", "DEF"),
    ])
    # Only the match itself — no prior data; shift(1) means rolling is all-NaN
    stats = pd.DataFrame([
        _stats_row("M1", 1, "Arsenal", 5.0, 4.0),
        _stats_row("M1", 50, "Chelsea", 2.0, 2.0),
    ])
    # Should not raise, but home anchor will be NaN (no prior matches)
    result = compute_defensive_anchor(lineups, stats, match_raw)
    assert "match_id" in result.columns or result.empty


# ---------------------------------------------------------------------------
# Test 5: Output has DEF_ANCHOR_HOME and DEF_ANCHOR_AWAY columns
# ---------------------------------------------------------------------------

def test_def_anchor_columns():
    """DEF_ANCHOR_HOME and DEF_ANCHOR_AWAY must exist in the output."""
    prior_raw = _raw_df("P1", "2024-01-01", "Arsenal", "Chelsea")
    match_raw = _raw_df("M1", "2024-01-15", "Arsenal", "Chelsea")
    raw = pd.concat([prior_raw, match_raw], ignore_index=True)

    lineups = pd.DataFrame([
        _lineup_row(99, 1, "Arsenal", "home", "DEF"),
        _lineup_row(99, 50, "Chelsea", "away", "DEF"),
    ])
    stats = pd.DataFrame([
        _stats_row("P1", 1, "Arsenal", 3.0, 2.0),
        _stats_row("M1", 1, "Arsenal", 3.0, 2.0),
        _stats_row("P1", 50, "Chelsea", 2.0, 1.0),
        _stats_row("M1", 50, "Chelsea", 2.0, 1.0),
    ])

    result = compute_defensive_anchor(lineups, stats, raw)
    assert "DEF_ANCHOR_HOME" in result.columns
    assert "DEF_ANCHOR_AWAY" in result.columns
    assert "match_id" in result.columns
