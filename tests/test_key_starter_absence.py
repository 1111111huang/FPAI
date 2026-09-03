"""Tests for key-attacker-absence flag — US#175.

XOC/FRDS/DEF_ANCHOR already join to each match's *actual* confirmed
lineup, so a rotated-out player already lowers those magnitudes -- but
none of them explicitly flag "the team's own identified best attacking
threat did not start this match", which is the new, non-redundant signal
this story adds.
"""

from __future__ import annotations

import pandas as pd
import pytest

from src.features.lineup_features import compute_key_starter_absence


def _raw_df(rows: list[tuple[str, str, str, str]]) -> pd.DataFrame:
    """rows: (match_id, date, home_team, away_team)."""
    return pd.DataFrame(
        [{"match_id": r[0], "date": r[1], "home_team": r[2], "away_team": r[3]} for r in rows]
    ).assign(date=lambda d: pd.to_datetime(d["date"]))


def _lineup_row(fotmob_id: int, player_id: int, team: str, side: str, pos: str = "FWD") -> dict:
    return {
        "fotmob_match_id": fotmob_id,
        "player_id": player_id,
        "team_name": team,
        "side": side,
        "position_group": pos,
    }


def _stats_row(match_id: str, player_id: int, team: str, xg: float, xa: float, minutes: int = 90) -> dict:
    return {
        "match_id": match_id,
        "player_id": player_id,
        "team_name": team,
        "minutes_played": minutes,
        "xg": xg,
        "xa": xa,
    }


def test_key_starter_absence_empty_inputs_returns_match_id_only():
    stats = pd.DataFrame([_stats_row("M1", 1, "Arsenal", 0.5, 0.1)])
    raw = _raw_df([("M1", "2024-01-15", "Arsenal", "Chelsea")])
    result = compute_key_starter_absence(pd.DataFrame(), stats, raw)
    assert result.empty or "match_id" in result.columns


def test_key_starter_absence_flags_zero_when_identified_player_starts():
    """Player 1 has the higher rolling xG+xA over the pool window and does
    start in the target match → home flag = 0 (not missing)."""
    raw = _raw_df([
        ("P1", "2024-01-01", "Arsenal", "Tottenham"),
        ("P2", "2024-01-08", "Arsenal", "Everton"),
        ("M1", "2024-01-15", "Arsenal", "Chelsea"),
    ])
    stats = pd.DataFrame([
        _stats_row("P1", 1, "Arsenal", 1.0, 0.5),   # player 1: strong rolling form
        _stats_row("P2", 1, "Arsenal", 1.0, 0.5),
        _stats_row("P1", 2, "Arsenal", 0.1, 0.0),   # player 2: weak rolling form
        _stats_row("P2", 2, "Arsenal", 0.1, 0.0),
        _stats_row("M1", 1, "Arsenal", 0.8, 0.4),   # player 1 also plays M1
        _stats_row("M1", 50, "Chelsea", 0.3, 0.1),
    ])
    lineups = pd.DataFrame([
        _lineup_row(99, 1, "Arsenal", "home"),   # player 1 (the identified key player) starts
        _lineup_row(99, 2, "Arsenal", "home"),
        _lineup_row(99, 50, "Chelsea", "away"),
    ])

    result = compute_key_starter_absence(lineups, stats, raw)
    row = result.loc[result["match_id"] == "M1"].iloc[0]
    assert row["LINEUP_HOME_KEY_ATTACKER_MISSING"] == 0.0


def test_key_starter_absence_flags_one_when_identified_player_does_not_start():
    """Player 1 has the strongest rolling form but is absent from M1's
    confirmed lineup (only player 2 starts) → home flag = 1 (missing)."""
    raw = _raw_df([
        ("P1", "2024-01-01", "Arsenal", "Tottenham"),
        ("P2", "2024-01-08", "Arsenal", "Everton"),
        ("M1", "2024-01-15", "Arsenal", "Chelsea"),
    ])
    stats = pd.DataFrame([
        _stats_row("P1", 1, "Arsenal", 1.0, 0.5),
        _stats_row("P2", 1, "Arsenal", 1.0, 0.5),
        _stats_row("P1", 2, "Arsenal", 0.1, 0.0),
        _stats_row("P2", 2, "Arsenal", 0.1, 0.0),
        # Player 1 does NOT appear in M1's own stats at all (didn't play).
        _stats_row("M1", 2, "Arsenal", 0.2, 0.1),
        _stats_row("M1", 50, "Chelsea", 0.3, 0.1),
    ])
    lineups = pd.DataFrame([
        _lineup_row(99, 2, "Arsenal", "home"),   # only player 2 starts — player 1 missing
        _lineup_row(99, 50, "Chelsea", "away"),
    ])

    result = compute_key_starter_absence(lineups, stats, raw)
    row = result.loc[result["match_id"] == "M1"].iloc[0]
    assert row["LINEUP_HOME_KEY_ATTACKER_MISSING"] == 1.0


def test_key_starter_absence_nan_when_lineup_unconfirmed():
    """No match_lineups row at all for the target match (e.g. a future
    fixture before official lineups drop) → NaN, not a crash or a silent 0."""
    raw = _raw_df([
        ("P1", "2024-01-01", "Arsenal", "Tottenham"),
        ("M1", "2024-01-15", "Arsenal", "Chelsea"),
    ])
    stats = pd.DataFrame([
        _stats_row("P1", 1, "Arsenal", 1.0, 0.5),
    ])
    # match_lineups has data for some OTHER match (P1's fotmob id), never M1.
    lineups = pd.DataFrame([
        _lineup_row(77, 1, "Arsenal", "home"),
    ])

    result = compute_key_starter_absence(lineups, stats, raw)
    row = result.loc[result["match_id"] == "M1"].iloc[0]
    assert pd.isna(row["LINEUP_HOME_KEY_ATTACKER_MISSING"])
    assert pd.isna(row["LINEUP_AWAY_KEY_ATTACKER_MISSING"])


def test_key_starter_absence_output_columns():
    raw = _raw_df([("M1", "2024-01-15", "Arsenal", "Chelsea")])
    stats = pd.DataFrame([_stats_row("M1", 1, "Arsenal", 0.5, 0.1)])
    lineups = pd.DataFrame([_lineup_row(99, 1, "Arsenal", "home")])
    result = compute_key_starter_absence(lineups, stats, raw)
    assert "LINEUP_HOME_KEY_ATTACKER_MISSING" in result.columns
    assert "LINEUP_AWAY_KEY_ATTACKER_MISSING" in result.columns
    assert "match_id" in result.columns
