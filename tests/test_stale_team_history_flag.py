"""Regression tests for W198: a team can have real raw_matches rows that are
years stale (e.g. relegated out of a tracked competition long ago, then
fictionally re-listed as "current season" by a fixtures vendor). Found live
2026-09-22 -- football-data.org listed "Real Racing Club de Santander"/
"Malaga"/"La Coruna" as current SP1 fixtures despite football-data.co.uk's
own current-season CSV (raw_matches' real source) never tracking them this
season. Malaga/La Coruna both had real raw_matches rows (so US#108's own
zero-history check, tests/test_unknown_team_flag.py, never caught them) --
but every row dated to 2016-2018, years before "today." Before this fix,
they were silently treated as fully known (feature_completeness ~0.92,
unknown_team=False), and one had already reached a live direct_bet
recommendation on that stale basis."""

from __future__ import annotations

from pathlib import Path
import sys

import pandas as pd
import pytest

sys.path.append(str(Path(__file__).resolve().parents[1]))

duckdb = pytest.importorskip("duckdb")

from src.features.feature_factory import FeatureFactory


def _build_db(tmp_path: Path) -> Path:
    import yaml

    db_path = tmp_path / "test_fpai.db"
    model_dir = tmp_path / "models"
    model_dir.mkdir(exist_ok=True)
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        yaml.safe_dump({"paths": {"database_path": str(db_path), "model_dir": str(model_dir)}}),
        encoding="utf-8",
    )

    with duckdb.connect(str(db_path)) as conn:
        conn.execute(
            """
            CREATE TABLE raw_matches (
                match_id TEXT PRIMARY KEY, league TEXT, tier INTEGER, date TIMESTAMP,
                home_team TEXT, away_team TEXT, fthg INTEGER, ftag INTEGER,
                hs FLOAT, "as" FLOAT, hst FLOAT, ast FLOAT, hc FLOAT, ac FLOAT,
                hy FLOAT, ay FLOAT, hr FLOAT, ar FLOAT,
                odds_h FLOAT, odds_d FLOAT, odds_a FLOAT,
                avgh FLOAT, avgd FLOAT, avga FLOAT,
                maxch FLOAT, maxcd FLOAT, maxca FLOAT,
                avgch FLOAT, avgcd FLOAT, avgca FLOAT,
                xg_h FLOAT, xg_a FLOAT, xga_h FLOAT, xga_a FLOAT,
                over25_odds FLOAT, under25_odds FLOAT,
                ah_line FLOAT, ah_home_odds FLOAT, ah_away_odds FLOAT
            )
            """
        )
        conn.execute(
            """
            INSERT INTO raw_matches
            (match_id, league, tier, date, home_team, away_team, fthg, ftag, odds_h, odds_d, odds_a, avgh, avgd, avga)
            VALUES
            -- Everton: fresh, recent history
            ('m1', 'E0', 1, '2025-08-10 20:00:00', 'Arsenal', 'Everton', 2, 1, 1.5, 4.0, 5.0, 1.5, 4.0, 5.0),
            ('m2', 'E0', 1, '2025-08-17 20:00:00', 'Arsenal', 'Everton', 1, 1, 1.5, 4.0, 5.0, 1.5, 4.0, 5.0),
            -- Malaga: real rows, but every one is 8 years stale (2016-2018 -- its real last La Liga season)
            ('m3', 'E0', 1, '2016-08-19 20:00:00', 'Malaga', 'Arsenal', 0, 2, 3.0, 3.4, 2.2, 3.0, 3.4, 2.2),
            ('m4', 'E0', 1, '2018-05-19 20:00:00', 'Arsenal', 'Malaga', 2, 0, 1.3, 5.0, 8.0, 1.3, 5.0, 8.0)
            """
        )
    return config_path


def test_build_for_match_flags_a_stale_team_as_unknown(tmp_path: Path) -> None:
    """Malaga has real raw_matches rows, so US#108's zero-history check
    alone doesn't catch it -- every row is ~8 years old relative to the
    2025-08-24 match being forecast, well past the staleness threshold."""
    config_path = _build_db(tmp_path)
    factory = FeatureFactory(config_path=str(config_path))
    row = factory.build_for_match(
        home_team="Malaga", away_team="Everton", match_date="2025-08-24",
        league="E0", odds_h=2.5, odds_d=3.2, odds_a=2.9,
    )
    assert bool(row["_unknown_team"].iloc[0]) is True


def test_build_for_match_does_not_flag_recent_history_as_stale(tmp_path: Path) -> None:
    """Regression: Arsenal/Everton's real, recent (2025) rows must not trip
    the new staleness check -- same fixture as US#108's own known-teams test."""
    config_path = _build_db(tmp_path)
    factory = FeatureFactory(config_path=str(config_path))
    row = factory.build_for_match(
        home_team="Arsenal", away_team="Everton", match_date="2025-08-24",
        league="E0", odds_h=1.8, odds_d=3.6, odds_a=4.2,
    )
    assert bool(row["_unknown_team"].iloc[0]) is False


def test_build_for_match_only_the_stale_side_loses_its_own_history(tmp_path: Path) -> None:
    """A flag alone isn't the fix -- Malaga's own stale rows must actually be
    dropped from the rolling-feature input, not just relabeled. Black-box
    check, not coupled to exact column names: a stale team (real rows, all
    8 years old) must produce a behaviorally IDENTICAL feature row to a
    team with genuinely zero rows at all -- proving its stale history
    contributes nothing. Everton (the away side, real fresh history) is
    shared across both calls, so any difference would have to come from
    the home side's own stale-vs-zero history."""
    config_path = _build_db(tmp_path)
    factory = FeatureFactory(config_path=str(config_path))

    stale_row = factory.build_for_match(
        home_team="Malaga", away_team="Everton", match_date="2025-08-24",
        league="E0", odds_h=2.5, odds_d=3.2, odds_a=2.9,
    )
    zero_history_row = factory.build_for_match(
        home_team="Coventry City", away_team="Everton", match_date="2025-08-24",
        league="E0", odds_h=2.5, odds_d=3.2, odds_a=2.9,
    )

    numeric_cols = [
        c for c in stale_row.columns
        if c in zero_history_row.columns and pd.api.types.is_numeric_dtype(stale_row[c])
    ]
    assert numeric_cols, "expected at least one shared numeric feature column"
    pd.testing.assert_frame_equal(
        stale_row[numeric_cols].reset_index(drop=True),
        zero_history_row[numeric_cols].reset_index(drop=True),
    )
