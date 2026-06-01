from __future__ import annotations

from pathlib import Path
import sys

import pandas as pd
import pytest
import yaml

sys.path.append(str(Path(__file__).resolve().parents[1]))

duckdb = pytest.importorskip("duckdb")

from src.features.feature_factory import FeatureFactory


def _create_raw_matches_table(conn) -> None:
    conn.execute(
        """
        CREATE TABLE raw_matches (
            match_id TEXT PRIMARY KEY,
            league TEXT,
            tier INTEGER,
            date TIMESTAMP,
            home_team TEXT,
            away_team TEXT,
            fthg INTEGER,
            ftag INTEGER,
            odds_h FLOAT,
            odds_d FLOAT,
            odds_a FLOAT,
            avgh FLOAT,
            avgd FLOAT,
            avga FLOAT
        )
        """
    )


def _insert_raw_matches(conn, rows: list[tuple[object, ...]]) -> None:
    conn.executemany(
        """
        INSERT INTO raw_matches
        (match_id, league, tier, date, home_team, away_team, fthg, ftag, odds_h, odds_d, odds_a, avgh, avgd, avga)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        rows,
    )


def test_compute_rolling_stats_and_save_features(tmp_path: Path) -> None:
    db_path = tmp_path / "test_fpai.db"
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        yaml.safe_dump({"paths": {"database_path": str(db_path)}}),
        encoding="utf-8",
    )

    with duckdb.connect(str(db_path)) as conn:
        _create_raw_matches_table(conn)
        _insert_raw_matches(
            conn,
            [
                ("m1", "E0", 1, "2025-08-15 20:00:00", "Liverpool", "Bournemouth", 4, 2, 1.3, 6.0, 8.5, 1.3, 6.0, 8.5),
                ("m2", "E0", 1, "2025-08-16 12:30:00", "Aston Villa", "Liverpool", 0, 1, 2.25, 3.5, 2.9, 2.25, 3.5, 2.9),
            ],
        )

    feature_factory = FeatureFactory(config_path=str(config_path))
    features = feature_factory.compute_rolling_stats(window=5)

    expected_columns = {
        "match_id",
        "OFF_HOME_FTHG_R3",
        "OFF_HOME_FTHG_R5",
        "DEF_HOME_FTAG_R3",
        "DEF_HOME_FTAG_R5",
        "OFF_AWAY_FTAG_R3",
        "OFF_AWAY_FTAG_R5",
        "DEF_AWAY_FTHG_R3",
        "DEF_AWAY_FTHG_R5",
        "CTX_HOME_REST_DAYS",
        "CTX_AWAY_REST_DAYS",
        "CTX_REST_DAYS_DIFF",
        "MKT_IMPLIED_HOME",
        "MKT_IMPLIED_DRAW",
        "MKT_IMPLIED_AWAY",
        "MKT_Home_Prob_Real",
        "MKT_Draw_Prob_Real",
        "MKT_Away_Prob_Real",
        "INTERACTION_ATTACK_GOALS_DIFF_R5",
        "INTERACTION_DEFENSE_GOALS_DIFF_R5",
        "INTERACTION_ATTACK_SOT_DIFF_R5",
        "EFFICIENCY_HOME_ATTACK_VS_AWAY_DEF_R5",
        "EFFICIENCY_AWAY_ATTACK_VS_HOME_DEF_R5",
        "EFFICIENCY_ATTACK_MATCHUP_DIFF_R5",
    }
    assert expected_columns.issubset(set(features.columns))
    assert len(features) == 2

    match_1 = features.loc[features["match_id"] == "m1"].iloc[0]
    raw_market_total = (1 / 1.3) + (1 / 6.0) + (1 / 8.5)
    assert pd.isna(match_1["OFF_HOME_FTHG_R5"])
    assert match_1["MKT_IMPLIED_HOME"] == pytest.approx((1 / 1.3) / raw_market_total)
    assert match_1["MKT_Home_Prob_Real"] == pytest.approx((1 / 1.3) / raw_market_total)

    match_2 = features.loc[features["match_id"] == "m2"].iloc[0]
    raw_market_total = (1 / 2.25) + (1 / 3.5) + (1 / 2.9)
    assert match_2["MKT_IMPLIED_HOME"] == pytest.approx((1 / 2.25) / raw_market_total)
    assert match_2["MKT_IMPLIED_DRAW"] == pytest.approx((1 / 3.5) / raw_market_total)
    assert match_2["MKT_IMPLIED_AWAY"] == pytest.approx((1 / 2.9) / raw_market_total)

    feature_factory.save_features(features)

    with duckdb.connect(str(db_path)) as conn:
        stored_count = conn.execute("SELECT COUNT(*) FROM feature_store").fetchone()[0]
        stored_row = conn.execute(
            """
            SELECT MKT_IMPLIED_HOME, MKT_IMPLIED_DRAW, MKT_IMPLIED_AWAY
            FROM feature_store
            WHERE match_id = 'm2'
            """
        ).fetchone()

    assert stored_count == 2
    assert stored_row is not None
    assert stored_row[0] == pytest.approx((1 / 2.25) / raw_market_total)
    assert stored_row[1] == pytest.approx((1 / 3.5) / raw_market_total)
    assert stored_row[2] == pytest.approx((1 / 2.9) / raw_market_total)


def test_interaction_and_efficiency_features_use_shifted_history(tmp_path: Path) -> None:
    db_path = tmp_path / "test_fpai.db"
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        yaml.safe_dump({"paths": {"database_path": str(db_path)}}),
        encoding="utf-8",
    )

    rows = []
    for index in range(5):
        rows.append(
            (
                f"h{index}",
                "E0",
                1,
                f"2025-08-{index + 1:02d} 20:00:00",
                "Home FC",
                f"Home Opp {index}",
                index + 1,
                index,
                2.0,
                3.2,
                3.8,
                2.0,
                3.2,
                3.8,
            )
        )
        rows.append(
            (
                f"a{index}",
                "E0",
                1,
                f"2025-08-{index + 1:02d} 21:00:00",
                f"Away Opp {index}",
                "Away FC",
                2,
                index + 2,
                2.0,
                3.2,
                3.8,
                2.0,
                3.2,
                3.8,
            )
        )
    rows.append(
        (
            "final",
            "E0",
            1,
            "2025-08-20 20:00:00",
            "Home FC",
            "Away FC",
            9,
            9,
            2.0,
            3.2,
            3.8,
            2.0,
            3.2,
            3.8,
        )
    )

    with duckdb.connect(str(db_path)) as conn:
        _create_raw_matches_table(conn)
        _insert_raw_matches(conn, rows)

    feature_factory = FeatureFactory(config_path=str(config_path))
    features = feature_factory.compute_rolling_stats(window=5)
    final_match = features.loc[features["match_id"] == "final"].iloc[0]

    assert final_match["INTERACTION_ATTACK_GOALS_DIFF_R5"] == pytest.approx(-1.0)
    assert final_match["INTERACTION_DEFENSE_GOALS_DIFF_R5"] == pytest.approx(0.0)
    assert final_match["EFFICIENCY_HOME_ATTACK_VS_AWAY_DEF_R5"] == pytest.approx(3.0 / 2.1)
    assert final_match["EFFICIENCY_AWAY_ATTACK_VS_HOME_DEF_R5"] == pytest.approx(4.0 / 2.1)
    assert final_match["EFFICIENCY_ATTACK_MATCHUP_DIFF_R5"] == pytest.approx(-1.0 / 2.1)


def test_compute_rolling_stats_no_data_leakage_for_sixth_match(tmp_path: Path) -> None:
    db_path = tmp_path / "test_fpai.db"
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        yaml.safe_dump({"paths": {"database_path": str(db_path)}}),
        encoding="utf-8",
    )

    with duckdb.connect(str(db_path)) as conn:
        _create_raw_matches_table(conn)
        _insert_raw_matches(
            conn,
            [
                ("m1", "E0", 1, "2025-08-01 20:00:00", "Alpha FC", "Opp1", 1, 0, 1.8, 3.4, 4.2, 1.8, 3.4, 4.2),
                ("m2", "E0", 1, "2025-08-08 20:00:00", "Alpha FC", "Opp2", 2, 1, 1.8, 3.4, 4.2, 1.8, 3.4, 4.2),
                ("m3", "E0", 1, "2025-08-15 20:00:00", "Alpha FC", "Opp3", 3, 0, 1.8, 3.4, 4.2, 1.8, 3.4, 4.2),
                ("m4", "E0", 1, "2025-08-22 20:00:00", "Alpha FC", "Opp4", 4, 1, 1.8, 3.4, 4.2, 1.8, 3.4, 4.2),
                ("m5", "E0", 1, "2025-08-29 20:00:00", "Alpha FC", "Opp5", 5, 0, 1.8, 3.4, 4.2, 1.8, 3.4, 4.2),
                ("m6", "E0", 1, "2025-09-05 20:00:00", "Alpha FC", "Opp6", 10, 10, 1.8, 3.4, 4.2, 1.8, 3.4, 4.2),
            ],
        )

    feature_factory = FeatureFactory(config_path=str(config_path))
    features = feature_factory.compute_rolling_stats(window=5)

    match_6 = features.loc[features["match_id"] == "m6"].iloc[0]

    expected_scored_without_leakage = (1 + 2 + 3 + 4 + 5) / 5
    expected_conceded_without_leakage = (0 + 1 + 0 + 1 + 0) / 5
    scored_with_leakage = (2 + 3 + 4 + 5 + 10) / 5

    assert match_6["OFF_HOME_FTHG_R5"] == pytest.approx(expected_scored_without_leakage)
    assert match_6["DEF_HOME_FTAG_R5"] == pytest.approx(expected_conceded_without_leakage)
    assert match_6["OFF_HOME_FTHG_R5"] != pytest.approx(scored_with_leakage)
    assert pd.isna(match_6["OFF_AWAY_FTAG_R5"])


def test_new_team_has_missing_rolling_history_in_expanded_schema(tmp_path: Path) -> None:
    db_path = tmp_path / "test_fpai.db"
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        yaml.safe_dump({"paths": {"database_path": str(db_path)}}),
        encoding="utf-8",
    )

    with duckdb.connect(str(db_path)) as conn:
        _create_raw_matches_table(conn)
        _insert_raw_matches(
            conn,
            [
                ("p1", "E0", 1, "2024-08-01 20:00:00", "Alpha", "Beta", 2, 0, 1.8, 3.4, 4.2, 1.8, 3.4, 4.2),
                ("p2", "E0", 1, "2024-08-08 20:00:00", "Gamma", "Delta", 1, 3, 1.8, 3.4, 4.2, 1.8, 3.4, 4.2),
                ("n1", "E0", 1, "2025-08-10 20:00:00", "Promoted FC", "Alpha", 0, 1, 1.8, 3.4, 4.2, 1.8, 3.4, 4.2),
            ],
        )

    feature_factory = FeatureFactory(config_path=str(config_path))
    features = feature_factory.compute_rolling_stats(window=5)
    new_team_match = features.loc[features["match_id"] == "n1"].iloc[0]

    assert pd.isna(new_team_match["OFF_HOME_FTHG_R5"])
    assert pd.isna(new_team_match["DEF_HOME_FTAG_R5"])


def test_rest_day_features_are_shifted_by_team_and_venue(tmp_path: Path) -> None:
    db_path = tmp_path / "test_fpai.db"
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        yaml.safe_dump({"paths": {"database_path": str(db_path)}}),
        encoding="utf-8",
    )

    with duckdb.connect(str(db_path)) as conn:
        _create_raw_matches_table(conn)
        _insert_raw_matches(
            conn,
            [
                ("r1", "E0", 1, "2025-08-01 20:00:00", "Home FC", "Away FC", 1, 0, 2.0, 3.2, 3.8, 2.0, 3.2, 3.8),
                ("r2", "E0", 1, "2025-08-08 20:00:00", "Home FC", "Away FC", 1, 1, 2.0, 3.2, 3.8, 2.0, 3.2, 3.8),
                ("r3", "E0", 1, "2025-08-20 20:00:00", "Home FC", "Away FC", 0, 1, 2.0, 3.2, 3.8, 2.0, 3.2, 3.8),
            ],
        )

    feature_factory = FeatureFactory(config_path=str(config_path))
    features = feature_factory.compute_rolling_stats(window=5)

    second_match = features.loc[features["match_id"] == "r2"].iloc[0]
    third_match = features.loc[features["match_id"] == "r3"].iloc[0]

    assert second_match["CTX_HOME_REST_DAYS"] == pytest.approx(7.0)
    assert second_match["CTX_AWAY_REST_DAYS"] == pytest.approx(7.0)
    assert second_match["CTX_REST_DAYS_DIFF"] == pytest.approx(0.0)
    assert third_match["CTX_HOME_REST_DAYS"] == pytest.approx(12.0)
    assert third_match["CTX_AWAY_REST_DAYS"] == pytest.approx(12.0)
