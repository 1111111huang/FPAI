"""Tests for SQUAD_*_MKT_VALUE_MEAN_R3/R5 rolling feature computation (US#208
Phase 2) -- the same rolling-window shape _squad_rolling_from_data already
uses for SQUAD_*_RATING_MEAN_R3/R5, pointed at player_market_values.market_value_eur
(joined point-in-time via the REEP crosswalk) instead of raw_player_match_stats.rating.

Point-in-time correctness is the load-bearing requirement here (design spec):
a match's rolling window must use, for each player, the market-value snapshot
dated at-or-before THAT match's own date -- never a later one. Same
lookahead-bias class of bug W179 already had to fix for closing-line odds."""

from __future__ import annotations

from pathlib import Path
import sys

import pandas as pd
import pytest

sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.features.feature_factory import FeatureFactory

MKT_VALUE_FEATURE_NAMES = [
    "SQUAD_HOME_MKT_VALUE_MEAN_R3", "SQUAD_HOME_MKT_VALUE_MEAN_R5",
    "SQUAD_AWAY_MKT_VALUE_MEAN_R3", "SQUAD_AWAY_MKT_VALUE_MEAN_R5",
]


def _raw_matches_df() -> pd.DataFrame:
    """3 matches, Arsenal (home) vs Everton (away)."""
    return pd.DataFrame([
        {"match_id": "m1", "date": "2024-08-10", "home_team": "Arsenal", "away_team": "Everton"},
        {"match_id": "m2", "date": "2024-08-17", "home_team": "Arsenal", "away_team": "Everton"},
        {"match_id": "m3", "date": "2024-08-24", "home_team": "Arsenal", "away_team": "Everton"},
    ])


def _player_df() -> pd.DataFrame:
    """One player per team for m1/m2 (fotmob_player_id doubles as player_id)."""
    return pd.DataFrame([
        {"match_id": "m1", "team_name": "Arsenal", "player_id": 1},
        {"match_id": "m1", "team_name": "Everton", "player_id": 2},
        {"match_id": "m2", "team_name": "Arsenal", "player_id": 1},
        {"match_id": "m2", "team_name": "Everton", "player_id": 2},
    ])


def _values_df() -> pd.DataFrame:
    """Player 1 (Arsenal) has two snapshots; player 2 (Everton) has one."""
    return pd.DataFrame([
        {"fotmob_player_id": 1, "snapshot_date": "2024-07-01", "market_value_eur": 10_000_000},
        {"fotmob_player_id": 1, "snapshot_date": "2024-08-15", "market_value_eur": 20_000_000},
        {"fotmob_player_id": 2, "snapshot_date": "2024-07-01", "market_value_eur": 5_000_000},
    ])


def test_returns_correct_columns() -> None:
    result = FeatureFactory._squad_market_value_rolling_from_data(_player_df(), _values_df(), _raw_matches_df())

    assert "match_id" in result.columns
    for col in MKT_VALUE_FEATURE_NAMES:
        assert col in result.columns, f"Missing column: {col}"


def test_first_match_is_nan_because_no_prior_data() -> None:
    result = FeatureFactory._squad_market_value_rolling_from_data(_player_df(), _values_df(), _raw_matches_df())

    row_m1 = result[result["match_id"] == "m1"].iloc[0]
    assert pd.isna(row_m1["SQUAD_HOME_MKT_VALUE_MEAN_R3"])
    assert pd.isna(row_m1["SQUAD_AWAY_MKT_VALUE_MEAN_R5"])


def test_second_match_uses_first_matchs_point_in_time_snapshot() -> None:
    """m2 is 2024-08-17. Player 1's snapshot as-of m1's date (2024-08-10) is
    the 2024-07-01 one (10M) -- the 2024-08-15 snapshot (20M) postdates m1
    and must NOT be used for m2's rolling window (which reflects m1's
    value at the time m1 was played)."""
    result = FeatureFactory._squad_market_value_rolling_from_data(_player_df(), _values_df(), _raw_matches_df())

    row_m2 = result[result["match_id"] == "m2"].iloc[0]
    assert row_m2["SQUAD_HOME_MKT_VALUE_MEAN_R3"] == pytest.approx(10_000_000)
    assert row_m2["SQUAD_AWAY_MKT_VALUE_MEAN_R3"] == pytest.approx(5_000_000)


def test_a_players_snapshot_never_leaks_from_after_the_match_being_computed() -> None:
    """Direct point-in-time regression test: m3 (2024-08-24) rolls over
    m1/m2, both before the 2024-08-15 snapshot for player 1 in one case and
    after it in the other -- m2 (2024-08-17) must pick up the newer 20M
    snapshot, m1 (2024-08-10) must not."""
    result = FeatureFactory._squad_market_value_rolling_from_data(_player_df(), _values_df(), _raw_matches_df())

    row_m3 = result[result["match_id"] == "m3"].iloc[0]
    # Arsenal (home) across m1 (10M, snapshot as-of 08-10) and m2 (20M, snapshot as-of 08-17) -> mean 15M
    assert row_m3["SQUAD_HOME_MKT_VALUE_MEAN_R3"] == pytest.approx(15_000_000)


def test_handles_mismatched_datetime_dtypes_between_matches_and_snapshots() -> None:
    """Found live against the real DuckDB-backed feature_store rebuild:
    raw_matches.date comes back from DuckDB as datetime64[us], while an
    empty/freshly-loaded player_market_values.snapshot_date column can
    resolve to datetime64[ns] -- pd.merge_asof raises MergeError on a
    dtype mismatch rather than silently coercing. Must not crash the
    whole feature_store rebuild over this."""
    player_df = _player_df()
    values_df = _values_df()
    raw_df = _raw_matches_df()
    # Force the exact dtype mismatch pandas' own real-data path can produce.
    raw_df["date"] = pd.to_datetime(raw_df["date"]).astype("datetime64[us]")
    values_df["snapshot_date"] = pd.to_datetime(values_df["snapshot_date"]).astype("datetime64[ns]")

    result = FeatureFactory._squad_market_value_rolling_from_data(player_df, values_df, raw_df)

    row_m2 = result[result["match_id"] == "m2"].iloc[0]
    assert row_m2["SQUAD_HOME_MKT_VALUE_MEAN_R3"] == pytest.approx(10_000_000)


def test_handles_null_player_id_rows_without_crashing() -> None:
    """Found live against the real feature_store rebuild: some historical
    raw_player_match_stats rows have a null player_id (a real data-quality
    gap, not something to assume away) -- pd.merge_asof's by= key can't
    contain nulls at all and raises ValueError, must not crash the whole
    feature_store rebuild over a handful of unmatchable rows."""
    player_df = pd.concat([_player_df(), pd.DataFrame([{"match_id": "m1", "team_name": "Arsenal", "player_id": None}])], ignore_index=True)

    result = FeatureFactory._squad_market_value_rolling_from_data(player_df, _values_df(), _raw_matches_df())

    row_m2 = result[result["match_id"] == "m2"].iloc[0]
    assert row_m2["SQUAD_HOME_MKT_VALUE_MEAN_R3"] == pytest.approx(10_000_000)


def test_handles_a_player_row_whose_match_id_is_not_in_raw_matches() -> None:
    """Found live against the real feature_store rebuild: a small number of
    raw_player_match_stats rows reference a match_id that isn't present in
    raw_matches at all (an orphaned historical row) -- the left-join to
    match_info leaves date as NaT for that row, which merge_asof also
    rejects outright. Must drop it, not crash the whole rebuild."""
    player_df = pd.concat(
        [_player_df(), pd.DataFrame([{"match_id": "orphaned", "team_name": "Arsenal", "player_id": 1}])],
        ignore_index=True,
    )

    result = FeatureFactory._squad_market_value_rolling_from_data(player_df, _values_df(), _raw_matches_df())

    row_m2 = result[result["match_id"] == "m2"].iloc[0]
    assert row_m2["SQUAD_HOME_MKT_VALUE_MEAN_R3"] == pytest.approx(10_000_000)


def test_missing_player_market_values_table_degrades_to_empty_frame() -> None:
    """Mirrors _compute_squad_features' own degradation contract -- an
    empty player_df (e.g. the table doesn't exist yet) returns a single-
    column match_id frame, not an error."""
    result = FeatureFactory._squad_market_value_rolling_from_data(
        pd.DataFrame(columns=["match_id", "team_name", "player_id"]), _values_df(), _raw_matches_df()
    )
    assert list(result.columns) == ["match_id"]
