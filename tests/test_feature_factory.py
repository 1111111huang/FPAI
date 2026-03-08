from __future__ import annotations

from pathlib import Path
import sys

import pandas as pd
import pytest
import yaml

sys.path.append(str(Path(__file__).resolve().parents[1]))

duckdb = pytest.importorskip("duckdb")

from src.features.feature_factory import FeatureFactory


def test_compute_rolling_stats_and_save_features(tmp_path: Path) -> None:
    db_path = tmp_path / "test_fpai.db"
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        yaml.safe_dump({"paths": {"database_path": str(db_path)}}),
        encoding="utf-8",
    )

    with duckdb.connect(str(db_path)) as conn:
        conn.execute(
            """
            CREATE TABLE raw_matches (
                match_id TEXT PRIMARY KEY,
                league TEXT,
                date TIMESTAMP,
                home_team TEXT,
                away_team TEXT,
                fthg INTEGER,
                ftag INTEGER,
                odds_h FLOAT,
                odds_d FLOAT,
                odds_a FLOAT
            )
            """
        )
        conn.executemany(
            """
            INSERT INTO raw_matches
            (match_id, league, date, home_team, away_team, fthg, ftag, odds_h, odds_d, odds_a)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            [
                ("m1", "E0", "2025-08-15 20:00:00", "Liverpool", "Bournemouth", 4, 2, 1.3, 6.0, 8.5),
                ("m2", "E0", "2025-08-16 12:30:00", "Aston Villa", "Liverpool", 0, 1, 2.25, 3.5, 2.9),
            ],
        )

    feature_factory = FeatureFactory(config_path=str(config_path))
    features = feature_factory.compute_rolling_stats(window=5)

    assert list(features.columns) == [
        "match_id",
        "home_avg_goals_scored",
        "home_avg_goals_conceded",
        "away_avg_goals_scored",
        "away_avg_goals_conceded",
    ]
    assert len(features) == 2

    match_1 = features.loc[features["match_id"] == "m1"].iloc[0]
    assert pd.isna(match_1["home_avg_goals_scored"])
    assert pd.isna(match_1["away_avg_goals_scored"])

    match_2 = features.loc[features["match_id"] == "m2"].iloc[0]
    assert pd.isna(match_2["home_avg_goals_scored"])
    assert pd.isna(match_2["home_avg_goals_conceded"])
    assert match_2["away_avg_goals_scored"] == pytest.approx(4.0)
    assert match_2["away_avg_goals_conceded"] == pytest.approx(2.0)

    feature_factory.save_features(features)

    with duckdb.connect(str(db_path)) as conn:
        stored_count = conn.execute("SELECT COUNT(*) FROM feature_store").fetchone()[0]
        stored_row = conn.execute(
            """
            SELECT away_avg_goals_scored, away_avg_goals_conceded
            FROM feature_store
            WHERE match_id = 'm2'
            """
        ).fetchone()

    assert stored_count == 2
    assert stored_row is not None
    assert stored_row[0] == pytest.approx(4.0)
    assert stored_row[1] == pytest.approx(2.0)


def test_compute_rolling_stats_no_data_leakage_for_sixth_match(tmp_path: Path) -> None:
    db_path = tmp_path / "test_fpai.db"
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        yaml.safe_dump({"paths": {"database_path": str(db_path)}}),
        encoding="utf-8",
    )

    with duckdb.connect(str(db_path)) as conn:
        conn.execute(
            """
            CREATE TABLE raw_matches (
                match_id TEXT PRIMARY KEY,
                league TEXT,
                date TIMESTAMP,
                home_team TEXT,
                away_team TEXT,
                fthg INTEGER,
                ftag INTEGER,
                odds_h FLOAT,
                odds_d FLOAT,
                odds_a FLOAT
            )
            """
        )
        conn.executemany(
            """
            INSERT INTO raw_matches
            (match_id, league, date, home_team, away_team, fthg, ftag, odds_h, odds_d, odds_a)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            [
                ("m1", "E0", "2025-08-01 20:00:00", "Alpha FC", "Opp1", 1, 0, 1.8, 3.4, 4.2),
                ("m2", "E0", "2025-08-08 20:00:00", "Alpha FC", "Opp2", 2, 1, 1.8, 3.4, 4.2),
                ("m3", "E0", "2025-08-15 20:00:00", "Alpha FC", "Opp3", 3, 0, 1.8, 3.4, 4.2),
                ("m4", "E0", "2025-08-22 20:00:00", "Alpha FC", "Opp4", 4, 1, 1.8, 3.4, 4.2),
                ("m5", "E0", "2025-08-29 20:00:00", "Alpha FC", "Opp5", 5, 0, 1.8, 3.4, 4.2),
                ("m6", "E0", "2025-09-05 20:00:00", "Alpha FC", "Opp6", 10, 10, 1.8, 3.4, 4.2),
            ],
        )

    feature_factory = FeatureFactory(config_path=str(config_path))
    features = feature_factory.compute_rolling_stats(window=5)

    match_6 = features.loc[features["match_id"] == "m6"].iloc[0]

    expected_scored_without_leakage = (1 + 2 + 3 + 4 + 5) / 5
    expected_conceded_without_leakage = (0 + 1 + 0 + 1 + 0) / 5
    scored_with_leakage = (2 + 3 + 4 + 5 + 10) / 5

    assert match_6["home_avg_goals_scored"] == pytest.approx(expected_scored_without_leakage)
    assert match_6["home_avg_goals_conceded"] == pytest.approx(expected_conceded_without_leakage)
    assert match_6["home_avg_goals_scored"] != pytest.approx(scored_with_leakage)
