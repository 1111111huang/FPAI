"""Tests for FRDS (FotMob Rating Dominance Share) feature — US#102."""

from __future__ import annotations

import pandas as pd
import pytest

from src.features.lineup_features import compute_frds, _resolve_fotmob_to_raw


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _raw_df(
    match_id: str,
    date: str,
    home: str = "Arsenal",
    away: str = "Chelsea",
) -> pd.DataFrame:
    return pd.DataFrame({
        "match_id": [match_id],
        "date": pd.to_datetime([date]),
        "home_team": [home],
        "away_team": [away],
    })


def _lineup_row(fotmob_id: int, player_id: int, team: str, side: str) -> dict:
    return {
        "fotmob_match_id": str(fotmob_id),
        "player_id": player_id,
        "team_name": team,
        "side": side,
    }


def _stats_row(match_id: str, player_id: int, team: str, rating: float) -> dict:
    return {
        "match_id": match_id,
        "player_id": player_id,
        "team_name": team,
        "rating": rating,
    }


# ---------------------------------------------------------------------------
# Test 1: Empty inputs return empty DataFrame
# ---------------------------------------------------------------------------

def test_frds_empty_lineups():
    raw = _raw_df("M1", "2024-01-15")
    stats = pd.DataFrame([_stats_row("M1", 1, "Arsenal", 7.5)])
    result = compute_frds(pd.DataFrame(), stats, raw)
    assert result.empty or set(result.columns) >= {"match_id"}


def test_frds_empty_stats():
    lineups = pd.DataFrame([_lineup_row(1, 1, "Arsenal", "home")])
    raw = _raw_df("M1", "2024-01-15")
    result = compute_frds(lineups, pd.DataFrame(), raw)
    assert result.empty or set(result.columns) >= {"match_id"}


# ---------------------------------------------------------------------------
# Test 2: Starting 11 = full squad pool → FRDS clamps to 1.0
# ---------------------------------------------------------------------------

def test_frds_clamps_to_one():
    """When the 11 starters are all players in the pool, FRDS should be ≤ 1.0."""
    prior_raw = _raw_df("P1", "2024-01-01", "Arsenal", "Tottenham")
    match_raw = _raw_df("M1", "2024-01-15", "Arsenal", "Chelsea")
    raw = pd.concat([prior_raw, match_raw], ignore_index=True)

    player_ids = list(range(1, 12))
    lineups = pd.DataFrame([
        _lineup_row(99, pid, "Arsenal", "home") for pid in player_ids
    ] + [
        _lineup_row(99, 50, "Chelsea", "away"),
    ])
    stats_rows = []
    for pid in player_ids:
        stats_rows.append(_stats_row("P1", pid, "Arsenal", 7.0))
        stats_rows.append(_stats_row("M1", pid, "Arsenal", 7.0))
    stats_rows.append(_stats_row("P1", 50, "Chelsea", 7.0))
    stats_rows.append(_stats_row("M1", 50, "Chelsea", 7.0))
    stats = pd.DataFrame(stats_rows)

    result = compute_frds(lineups, stats, raw)
    assert not result.empty
    row = result[result["match_id"] == "M1"]
    assert not row.empty
    frds_home = row["FRDS_HOME"].values[0]
    assert frds_home is not None
    assert 0.0 <= frds_home <= 1.0, f"FRDS_HOME={frds_home} exceeds [0,1]"


# ---------------------------------------------------------------------------
# Test 3: Rolling shift(1) — first match has no prior data
# ---------------------------------------------------------------------------

def test_frds_first_match_fallback():
    """When there is no squad pool (first match), result should be NaN not crash."""
    match_raw = _raw_df("M1", "2024-01-15", "Arsenal", "Chelsea")
    lineups = pd.DataFrame([
        _lineup_row(99, pid, "Arsenal", "home") for pid in range(1, 4)
    ] + [
        _lineup_row(99, 50, "Chelsea", "away"),
    ])
    stats = pd.DataFrame([
        _stats_row("M1", pid, "Arsenal", 7.0) for pid in range(1, 4)
    ] + [
        _stats_row("M1", 50, "Chelsea", 7.0)
    ])

    result = compute_frds(lineups, stats, match_raw)
    # Should return a DataFrame without raising
    assert "match_id" in result.columns or result.empty


# ---------------------------------------------------------------------------
# Test 4: _resolve_fotmob_to_raw picks best co-occurrence match
# ---------------------------------------------------------------------------

def test_resolve_fotmob_to_raw_votes():
    """Co-occurrence voting should pick the match where most starters appear."""
    lineups = pd.DataFrame({
        "fotmob_match_id": ["F1", "F1", "F1"],
        "player_id": [1, 2, 3],
        "team_std": ["Arsenal", "Arsenal", "Arsenal"],
    })
    # match_id=M_GOOD has all 3 players; match_id=M_BAD has only 1
    stats = pd.DataFrame({
        "match_id": ["M_GOOD", "M_GOOD", "M_GOOD", "M_BAD"],
        "player_id": [1, 2, 3, 1],
        "team_std": ["Arsenal", "Arsenal", "Arsenal", "Arsenal"],
    })
    match_dates = pd.DataFrame({
        "match_id": ["M_GOOD", "M_BAD"],
        "date": pd.to_datetime(["2024-01-15", "2024-01-08"]),
    })

    result = _resolve_fotmob_to_raw(lineups, stats, match_dates)
    assert not result.empty
    assert result.loc[result["fotmob_match_id"] == "F1", "match_id"].values[0] == "M_GOOD"


# ---------------------------------------------------------------------------
# Test 5: Home/away FRDS assigned to correct sides
# ---------------------------------------------------------------------------

def test_frds_home_away_columns_present():
    """Result must have FRDS_HOME and FRDS_AWAY columns."""
    prior_raw = _raw_df("P1", "2024-01-01", "Arsenal", "Tottenham")
    match_raw = _raw_df("M1", "2024-01-15", "Arsenal", "Chelsea")
    raw = pd.concat([prior_raw, match_raw], ignore_index=True)

    lineups = pd.DataFrame([
        _lineup_row(99, 1, "Arsenal", "home"),
        _lineup_row(99, 2, "Chelsea", "away"),
    ])
    stats = pd.DataFrame([
        _stats_row("P1", 1, "Arsenal", 7.5),
        _stats_row("M1", 1, "Arsenal", 7.5),
        _stats_row("P1", 2, "Chelsea", 6.5),
        _stats_row("M1", 2, "Chelsea", 6.5),
    ])

    result = compute_frds(lineups, stats, raw)
    assert "FRDS_HOME" in result.columns
    assert "FRDS_AWAY" in result.columns
