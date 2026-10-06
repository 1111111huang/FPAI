"""Tests for xOC (Top-3 Offensive Concentration) feature — US#103."""

from __future__ import annotations

import pandas as pd
import pytest

from src.features.lineup_features import compute_xoc, _load_coefficient


# ---------------------------------------------------------------------------
# Helpers to build minimal DataFrames
# ---------------------------------------------------------------------------

def _make_raw_df(match_id: str = "M1", home: str = "Arsenal", away: str = "Chelsea") -> pd.DataFrame:
    return pd.DataFrame(
        {
            "match_id": [match_id],
            "date": pd.to_datetime(["2024-01-15"]),
            "home_team": [home],
            "away_team": [away],
        }
    )


def _make_lineup(
    fotmob_match_id: int,
    player_ids: list[int],
    team_name: str,
    side: str,
    position_group: str = "FWD",
) -> pd.DataFrame:
    rows = []
    for pid in player_ids:
        rows.append(
            {
                "fotmob_match_id": fotmob_match_id,
                "player_id": pid,
                "team_name": team_name,
                "side": side,
                "position_group": position_group,
            }
        )
    return pd.DataFrame(rows)


def _make_player_stats(
    match_id: str,
    player_id: int,
    team_name: str,
    xg: float,
    xa: float,
    minutes_played: int = 90,
) -> dict:
    return {
        "match_id": match_id,
        "player_id": player_id,
        "team_name": team_name,
        "minutes_played": minutes_played,
        "xg": xg,
        "xa": xa,
    }


# ---------------------------------------------------------------------------
# Test 1: Top-3 selection — 4 FWD starters
# ---------------------------------------------------------------------------

def test_xoc_top3_selection():
    """With 4 FWD starters, only the top-3 by (xG+xA)/90 contribute to xOC."""
    raw_df = _make_raw_df("M2", "Arsenal", "Chelsea")

    # 2 prior matches so rolling R5 has data
    raw_prior1 = pd.DataFrame({
        "match_id": ["P1"],
        "date": pd.to_datetime(["2024-01-01"]),
        "home_team": ["Arsenal"],
        "away_team": ["Tottenham"],
    })
    raw_prior2 = pd.DataFrame({
        "match_id": ["P2"],
        "date": pd.to_datetime(["2024-01-08"]),
        "home_team": ["Arsenal"],
        "away_team": ["Liverpool"],
    })
    raw_full = pd.concat([raw_prior1, raw_prior2, raw_df], ignore_index=True)

    # 4 FWD starters from Arsenal (home) for match M2
    lineup_home = _make_lineup(
        fotmob_match_id=99,
        player_ids=[1, 2, 3, 4],
        team_name="Arsenal",
        side="home",
        position_group="FWD",
    )
    # Chelsea away FWDs — just 1 for simplicity
    lineup_away = _make_lineup(
        fotmob_match_id=99,
        player_ids=[10],
        team_name="Chelsea",
        side="away",
        position_group="FWD",
    )
    lineups = pd.concat([lineup_home, lineup_away], ignore_index=True)

    # Player stats: prior matches (P1, P2) so rolling is populated before M2
    # Player 1: xg=0.6, xa=0.2 → xgxa_p90 = 0.8/90*90 = 0.8
    # Player 2: xg=0.4, xa=0.1 → 0.5/90*90 = 0.5
    # Player 3: xg=0.3, xa=0.0 → 0.3
    # Player 4: xg=0.1, xa=0.0 → 0.1  ← lowest; should be excluded
    stats_rows = []
    for pid, xg, xa in [(1, 0.6, 0.2), (2, 0.4, 0.1), (3, 0.3, 0.0), (4, 0.1, 0.0)]:
        stats_rows.append(_make_player_stats("P1", pid, "Arsenal", xg, xa))
        stats_rows.append(_make_player_stats("P2", pid, "Arsenal", xg, xa))
        # Also add M2 stats (won't matter for rolling since shift=1, but needed for join)
        stats_rows.append(_make_player_stats("M2", pid, "Arsenal", xg, xa))
    # Chelsea player
    stats_rows.append(_make_player_stats("P1", 10, "Chelsea", 0.2, 0.1))
    stats_rows.append(_make_player_stats("P2", 10, "Chelsea", 0.2, 0.1))
    stats_rows.append(_make_player_stats("M2", 10, "Chelsea", 0.2, 0.1))

    player_stats = pd.DataFrame(stats_rows)

    result = compute_xoc(lineups, player_stats, raw_full, league_code="E0")

    assert not result.empty, "Result should not be empty"
    row = result[result["match_id"] == "M2"]
    assert not row.empty, "M2 row missing from result"

    xoc_home = row["XOC_HOME"].values[0]
    # Top-3 rolling means: 0.8 + 0.5 + 0.3 = 1.6, coefficient=1.0
    assert xoc_home is not None, "XOC_HOME should not be None"
    assert abs(xoc_home - 1.6) < 0.05, (
        f"Expected XOC_HOME ≈ 1.6 (top-3 sum), got {xoc_home}"
    )
    # Player 4 (xgxa=0.1) must not be included
    assert xoc_home < 1.7, "Player 4 should not be in the top-3 sum"


# ---------------------------------------------------------------------------
# Test 2: Fewer than 3 FWD starters
# ---------------------------------------------------------------------------

def test_xoc_fewer_than_3_fwds():
    """With only 2 FWD starters, xOC sums 2 values without crashing."""
    raw_prior = pd.DataFrame({
        "match_id": ["Q1"],
        "date": pd.to_datetime(["2024-01-01"]),
        "home_team": ["Arsenal"],
        "away_team": ["Tottenham"],
    })
    raw_match = pd.DataFrame({
        "match_id": ["Q2"],
        "date": pd.to_datetime(["2024-01-15"]),
        "home_team": ["Arsenal"],
        "away_team": ["Chelsea"],
    })
    raw_full = pd.concat([raw_prior, raw_match], ignore_index=True)

    lineup_home = _make_lineup(
        fotmob_match_id=200,
        player_ids=[20, 21],  # Only 2 FWDs
        team_name="Arsenal",
        side="home",
        position_group="FWD",
    )
    lineup_away = _make_lineup(
        fotmob_match_id=200,
        player_ids=[30],
        team_name="Chelsea",
        side="away",
        position_group="FWD",
    )
    lineups = pd.concat([lineup_home, lineup_away], ignore_index=True)

    stats_rows = []
    for pid, xg, xa in [(20, 0.5, 0.1), (21, 0.3, 0.2)]:
        stats_rows.append(_make_player_stats("Q1", pid, "Arsenal", xg, xa))
        stats_rows.append(_make_player_stats("Q2", pid, "Arsenal", xg, xa))
    stats_rows.append(_make_player_stats("Q1", 30, "Chelsea", 0.2, 0.0))
    stats_rows.append(_make_player_stats("Q2", 30, "Chelsea", 0.2, 0.0))
    player_stats = pd.DataFrame(stats_rows)

    result = compute_xoc(lineups, player_stats, raw_full, league_code="E0")

    row = result[result["match_id"] == "Q2"]
    assert not row.empty, "Q2 row missing"
    xoc_home = row["XOC_HOME"].values[0]
    # Rolling means: 0.6 + 0.5 = 1.1, coefficient=1.0
    assert xoc_home is not None, "XOC_HOME should not be None with 2 FWDs"
    assert xoc_home > 0, "XOC_HOME should be positive"
    assert abs(xoc_home - 1.1) < 0.05, f"Expected XOC_HOME ≈ 1.1, got {xoc_home}"


# ---------------------------------------------------------------------------
# Test: a low-minute cameo must not dominate the per-90 rolling average
# ---------------------------------------------------------------------------

def test_xoc_excludes_low_minute_cameo_from_per90_extrapolation():
    """A substitute who plays only a couple of minutes but records a real
    chance (e.g. 2 min, 0.42 xG) must not have that extrapolated to an
    absurd per-90 rate (0.42/2*90 = 18.9) that then dominates the rolling
    average -- confirmed live against real data (Girona, Pablo Ibanez, 2
    minutes played, 0.42 xG + 0.24 xA -> 29.7 "xG+xA per 90", which then
    inflated XOC_HOME to 59.4 across many real matches). The cameo's per-90
    value must be excluded (NaN) from the rolling window, not included --
    so the rolling mean here must reflect only the real, full-minutes
    appearance, not be pulled toward the cameo's inflated number."""
    raw_prior1 = pd.DataFrame({
        "match_id": ["R1"],
        "date": pd.to_datetime(["2024-01-01"]),
        "home_team": ["Arsenal"],
        "away_team": ["Tottenham"],
    })
    raw_prior2 = pd.DataFrame({
        "match_id": ["R2"],
        "date": pd.to_datetime(["2024-01-08"]),
        "home_team": ["Arsenal"],
        "away_team": ["Liverpool"],
    })
    raw_match = pd.DataFrame({
        "match_id": ["R3"],
        "date": pd.to_datetime(["2024-01-15"]),
        "home_team": ["Arsenal"],
        "away_team": ["Chelsea"],
    })
    raw_full = pd.concat([raw_prior1, raw_prior2, raw_match], ignore_index=True)

    lineup_home = _make_lineup(
        fotmob_match_id=300, player_ids=[40], team_name="Arsenal", side="home", position_group="FWD",
    )
    lineup_away = _make_lineup(
        fotmob_match_id=300, player_ids=[50], team_name="Chelsea", side="away", position_group="FWD",
    )
    lineups = pd.concat([lineup_home, lineup_away], ignore_index=True)

    stats_rows = [
        # R1: a real, full-minutes appearance -- xgxa_p90 = 0.1/90*90 = 0.1
        _make_player_stats("R1", 40, "Arsenal", xg=0.1, xa=0.0, minutes_played=90),
        # R2: a 2-minute cameo with a real chance -- would extrapolate to
        # (0.42/2)*90 = 18.9 without a minimum-minutes floor.
        _make_player_stats("R2", 40, "Arsenal", xg=0.42, xa=0.0, minutes_played=2),
        _make_player_stats("R3", 40, "Arsenal", xg=0.2, xa=0.0, minutes_played=90),
        _make_player_stats("R1", 50, "Chelsea", xg=0.1, xa=0.0, minutes_played=90),
        _make_player_stats("R2", 50, "Chelsea", xg=0.1, xa=0.0, minutes_played=90),
        _make_player_stats("R3", 50, "Chelsea", xg=0.1, xa=0.0, minutes_played=90),
    ]
    player_stats = pd.DataFrame(stats_rows)

    result = compute_xoc(lineups, player_stats, raw_full, league_code="E0")

    row = result[result["match_id"] == "R3"]
    assert not row.empty
    xoc_home = row["XOC_HOME"].values[0]
    # Rolling mean over R1 (0.1) and R2 (excluded, insufficient minutes) must
    # be ~0.1 -- NOT ~9.5 (mean of 0.1 and the 18.9 cameo extrapolation).
    assert xoc_home < 1.0, (
        f"XOC_HOME ({xoc_home}) should reflect only the real 0.1 appearance, "
        f"not be inflated by the 2-minute cameo's per-90 extrapolation"
    )


# ---------------------------------------------------------------------------
# Test 3: Coefficient normalisation
# ---------------------------------------------------------------------------

def test_xoc_coefficient_normalisation(tmp_path, monkeypatch):
    """XOC_HOME should differ when coefficient != 1.0."""
    raw_prior = pd.DataFrame({
        "match_id": ["R1"],
        "date": pd.to_datetime(["2024-01-01"]),
        "home_team": ["PSG"],
        "away_team": ["Lyon"],
    })
    raw_match = pd.DataFrame({
        "match_id": ["R2"],
        "date": pd.to_datetime(["2024-01-15"]),
        "home_team": ["PSG"],
        "away_team": ["Lyon"],
    })
    raw_full = pd.concat([raw_prior, raw_match], ignore_index=True)

    lineup_home = _make_lineup(
        fotmob_match_id=300,
        player_ids=[40, 41, 42],
        team_name="PSG",
        side="home",
        position_group="FWD",
    )
    lineup_away = _make_lineup(
        fotmob_match_id=300,
        player_ids=[50],
        team_name="Lyon",
        side="away",
        position_group="FWD",
    )
    lineups = pd.concat([lineup_home, lineup_away], ignore_index=True)

    stats_rows = []
    for pid, xg, xa in [(40, 0.4, 0.2), (41, 0.3, 0.1), (42, 0.2, 0.0)]:
        stats_rows.append(_make_player_stats("R1", pid, "PSG", xg, xa))
        stats_rows.append(_make_player_stats("R2", pid, "PSG", xg, xa))
    stats_rows.append(_make_player_stats("R1", 50, "Lyon", 0.1, 0.0))
    stats_rows.append(_make_player_stats("R2", 50, "Lyon", 0.1, 0.0))
    player_stats = pd.DataFrame(stats_rows)

    # EPL coefficient = 1.0
    result_e0 = compute_xoc(lineups, player_stats, raw_full, league_code="E0")
    # Ligue 1 coefficient = 0.85
    result_f1 = compute_xoc(lineups, player_stats, raw_full, league_code="F1")

    row_e0 = result_e0[result_e0["match_id"] == "R2"]
    row_f1 = result_f1[result_f1["match_id"] == "R2"]

    assert not row_e0.empty and not row_f1.empty

    xoc_e0 = row_e0["XOC_HOME"].values[0]
    xoc_f1 = row_f1["XOC_HOME"].values[0]

    # F1 coefficient < 1.0 → XOC_F1 > XOC_E0
    assert xoc_f1 > xoc_e0, (
        f"Expected XOC with F1 coefficient ({xoc_f1:.4f}) > E0 ({xoc_e0:.4f})"
    )
    assert abs(xoc_f1 - xoc_e0 / 0.85 * 1.0) < 0.01 or abs(xoc_e0 * (1 / 0.85) - xoc_f1) < 0.01, (
        f"xoc_f1 ({xoc_f1}) should be xoc_e0 ({xoc_e0}) / 0.85"
    )


# ---------------------------------------------------------------------------
# Test 4: Home/away assignment
# ---------------------------------------------------------------------------

def test_xoc_home_away_assignment():
    """XOC_HOME and XOC_AWAY must be assigned to the correct side."""
    raw_prior = pd.DataFrame({
        "match_id": ["S1"],
        "date": pd.to_datetime(["2024-01-01"]),
        "home_team": ["Manchester City"],
        "away_team": ["Liverpool"],
    })
    raw_match = pd.DataFrame({
        "match_id": ["S2"],
        "date": pd.to_datetime(["2024-01-15"]),
        "home_team": ["Manchester City"],
        "away_team": ["Liverpool"],
    })
    raw_full = pd.concat([raw_prior, raw_match], ignore_index=True)

    # Home team has high-xG FWDs; away has low-xG FWDs
    lineup_home = _make_lineup(
        fotmob_match_id=400,
        player_ids=[60, 61, 62],
        team_name="Manchester City",
        side="home",
        position_group="FWD",
    )
    lineup_away = _make_lineup(
        fotmob_match_id=400,
        player_ids=[70, 71, 72],
        team_name="Liverpool",
        side="away",
        position_group="FWD",
    )
    lineups = pd.concat([lineup_home, lineup_away], ignore_index=True)

    stats_rows = []
    # Home FWDs: high xG
    for pid, xg, xa in [(60, 0.8, 0.3), (61, 0.6, 0.2), (62, 0.5, 0.1)]:
        stats_rows.append(_make_player_stats("S1", pid, "Manchester City", xg, xa))
        stats_rows.append(_make_player_stats("S2", pid, "Manchester City", xg, xa))
    # Away FWDs: low xG
    for pid, xg, xa in [(70, 0.2, 0.1), (71, 0.1, 0.0), (72, 0.1, 0.1)]:
        stats_rows.append(_make_player_stats("S1", pid, "Liverpool", xg, xa))
        stats_rows.append(_make_player_stats("S2", pid, "Liverpool", xg, xa))
    player_stats = pd.DataFrame(stats_rows)

    result = compute_xoc(lineups, player_stats, raw_full, league_code="E0")

    row = result[result["match_id"] == "S2"]
    assert not row.empty

    xoc_home = row["XOC_HOME"].values[0]
    xoc_away = row["XOC_AWAY"].values[0]

    assert xoc_home is not None, "XOC_HOME should not be None"
    assert xoc_away is not None, "XOC_AWAY should not be None"

    # Home team has much higher xG → XOC_HOME > XOC_AWAY
    assert xoc_home > xoc_away, (
        f"XOC_HOME ({xoc_home:.3f}) should exceed XOC_AWAY ({xoc_away:.3f})"
    )

    # Sanity: columns exist in result
    assert "XOC_HOME" in result.columns
    assert "XOC_AWAY" in result.columns
    assert "match_id" in result.columns


# ---------------------------------------------------------------------------
# Test 5: _load_coefficient helper
# ---------------------------------------------------------------------------

def test_load_coefficient_defaults(tmp_path, monkeypatch):
    """_load_coefficient returns 1.0 when file absent or code unknown."""
    monkeypatch.chdir(tmp_path)  # No league_coefficients.yaml here
    assert _load_coefficient("E0") == 1.0
    assert _load_coefficient("UNKNOWN") == 1.0
