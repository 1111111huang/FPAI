"""Regression test for W06: build_for_match's team-name resolution must log
a warning when a team name doesn't resolve via config/team_mapping.json --
"a mismatch is logged, not silently cold-started" per the acceptance
criteria. Today it uses a bespoke inline dict lookup (no logging at all) --
this replaces it with the shared, already-logging TeamNameMapper."""

from __future__ import annotations

from pathlib import Path
import sys

import pytest
import yaml

sys.path.append(str(Path(__file__).resolve().parents[1]))

duckdb = pytest.importorskip("duckdb")

from src.features.feature_factory import FeatureFactory


def _build_db(tmp_path: Path) -> Path:
    db_path = tmp_path / "test_fpai.db"
    config_path = tmp_path / "config.yaml"
    config_path.write_text(yaml.safe_dump({"paths": {"database_path": str(db_path)}}), encoding="utf-8")

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
            VALUES ('m1', 'E0', 1, '2025-08-10 20:00:00', 'Arsenal', 'Everton', 2, 1, 1.5, 4.0, 5.0, 1.5, 4.0, 5.0)
            """
        )
    return config_path


def test_build_for_match_logs_warning_for_unmapped_team_name(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    """'Arsenal Football Club' isn't in config/team_mapping.json under any
    key -- unlike 'Arsenal FC' (mapped) it must produce a visible warning,
    not silently pass through unresolved."""
    config_path = _build_db(tmp_path)
    factory = FeatureFactory(config_path=str(config_path))

    with caplog.at_level("WARNING"):
        factory.build_for_match(
            home_team="Arsenal Football Club", away_team="Everton", match_date="2025-08-24",
            league="E0", odds_h=1.8, odds_d=3.6, odds_a=4.2,
        )

    assert any("Arsenal Football Club" in record.message for record in caplog.records)


def test_build_for_match_mapped_team_names_still_resolve_correctly(tmp_path: Path) -> None:
    """Regression: a real, already-mapped variant ('Arsenal FC', added for
    W06) must still resolve to real history, not just avoid the warning."""
    config_path = _build_db(tmp_path)
    factory = FeatureFactory(config_path=str(config_path))

    row = factory.build_for_match(
        home_team="Arsenal FC", away_team="Everton", match_date="2025-08-24",
        league="E0", odds_h=1.8, odds_d=3.6, odds_a=4.2,
    )

    assert bool(row["_unknown_team"].iloc[0]) is False
