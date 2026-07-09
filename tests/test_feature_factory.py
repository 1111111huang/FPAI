from __future__ import annotations

from pathlib import Path
import sys

import numpy as np
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
            avga FLOAT,
            over25_odds FLOAT,
            under25_odds FLOAT,
            ah_line FLOAT,
            ah_home_odds FLOAT,
            ah_away_odds FLOAT
        )
        """
    )


_FULL_COLS = 19  # match_id + league + tier + date + home + away + fthg + ftag + odds(3) + avg(3) + new(5)


def _insert_raw_matches(conn, rows: list[tuple[object, ...]]) -> None:
    padded = [r + (None,) * (_FULL_COLS - len(r)) if len(r) < _FULL_COLS else r for r in rows]
    conn.executemany(
        """
        INSERT INTO raw_matches
        (match_id, league, tier, date, home_team, away_team, fthg, ftag, odds_h, odds_d, odds_a, avgh, avgd, avga,
         over25_odds, under25_odds, ah_line, ah_home_odds, ah_away_odds)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        padded,
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
                # (match_id, league, tier, date, home, away, fthg, ftag, odds_h, odds_d, odds_a, avgh, avgd, avga,
                #  over25_odds, under25_odds, ah_line, ah_home_odds, ah_away_odds)
                ("m1", "E0", 1, "2025-08-15 20:00:00", "Liverpool", "Bournemouth", 4, 2, 1.3, 6.0, 8.5, 1.3, 6.0, 8.5, 1.44, 2.75, -1.5, 1.85, 1.97),
                ("m2", "E0", 1, "2025-08-16 12:30:00", "Aston Villa", "Liverpool", 0, 1, 2.25, 3.5, 2.9, 2.25, 3.5, 2.9, 1.80, 2.00, 0.0, 1.92, 1.90),
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
        "MKT_IMPLIED_OVER25",
        # MKT_IMPLIED_UNDER25 removed (US#77 Tier 1 — r=0.998 with MKT_IMPLIED_OVER25)
        "MKT_AH_LINE",
        "MKT_AH_HOME_ODDS",
        "MKT_AH_AWAY_ODDS",
        # US#76: Poisson-decomposed market features
        "MKT_LAMBDA_TOTAL",
        "MKT_LAMBDA_HOME",
        "MKT_LAMBDA_AWAY",
        "MKT_POISSON_BTTS_PROB",
        "MKT_LAMBDA_AH_DIFF",
        "INTERACTION_ATTACK_GOALS_DIFF_R5",
        "INTERACTION_DEFENSE_GOALS_DIFF_R5",
        "INTERACTION_ATTACK_SOT_DIFF_R5",
        "EFFICIENCY_HOME_ATTACK_VS_AWAY_DEF_R5",
        "EFFICIENCY_AWAY_ATTACK_VS_HOME_DEF_R5",
        "EFFICIENCY_ATTACK_MATCHUP_DIFF_R5",
    }
    opp_adj_columns = {
        "OPP_ADJ_HOME_GOALS_SCORED_R3",
        "OPP_ADJ_HOME_GOALS_SCORED_R5",
        "OPP_ADJ_HOME_GOALS_CONCEDED_R3",
        "OPP_ADJ_HOME_GOALS_CONCEDED_R5",
        "OPP_ADJ_AWAY_GOALS_SCORED_R3",
        "OPP_ADJ_AWAY_GOALS_SCORED_R5",
        "OPP_ADJ_AWAY_GOALS_CONCEDED_R3",
        "OPP_ADJ_AWAY_GOALS_CONCEDED_R5",
        "OPP_ADJ_GOAL_MATCHUP_HOME_R5",
        "OPP_ADJ_GOAL_MATCHUP_AWAY_R5",
        "OPP_ADJ_CORNER_MATCHUP_HOME_R5",
        "OPP_ADJ_CORNER_MATCHUP_AWAY_R5",
    }
    assert (expected_columns | opp_adj_columns).issubset(set(features.columns))
    assert len(features) == 2

    match_1 = features.loc[features["match_id"] == "m1"].iloc[0]
    raw_market_total = (1 / 1.3) + (1 / 6.0) + (1 / 8.5)
    assert pd.isna(match_1["OFF_HOME_FTHG_R5"])
    assert match_1["MKT_IMPLIED_HOME"] == pytest.approx((1 / 1.3) / raw_market_total)
    assert match_1["MKT_Home_Prob_Real"] == pytest.approx((1 / 1.3) / raw_market_total)
    assert match_1["MKT_IMPLIED_OVER25"] == pytest.approx(1 / 1.44)
    # MKT_IMPLIED_UNDER25 removed (US#77 Tier 1)
    assert match_1["MKT_AH_LINE"] == pytest.approx(-1.5)
    assert match_1["MKT_AH_HOME_ODDS"] == pytest.approx(1.85)
    assert match_1["MKT_AH_AWAY_ODDS"] == pytest.approx(1.97)

    match_2 = features.loc[features["match_id"] == "m2"].iloc[0]
    raw_market_total = (1 / 2.25) + (1 / 3.5) + (1 / 2.9)
    assert match_2["MKT_IMPLIED_HOME"] == pytest.approx((1 / 2.25) / raw_market_total)
    assert match_2["MKT_IMPLIED_DRAW"] == pytest.approx((1 / 3.5) / raw_market_total)
    assert match_2["MKT_IMPLIED_AWAY"] == pytest.approx((1 / 2.9) / raw_market_total)
    assert match_2["MKT_IMPLIED_OVER25"] == pytest.approx(1 / 1.80)
    assert match_2["MKT_AH_LINE"] == pytest.approx(0.0)

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


def test_opp_adjusted_features_no_leakage(tmp_path: Path) -> None:
    """OPP_ADJ rolling for m6 must use only m1-m5 results, not m6's outcome."""
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

    # R5 should average m1-m5 goals scored (1+2+3+4+5)/5, not include m6's 10
    assert match_6["OPP_ADJ_HOME_GOALS_SCORED_R5"] == pytest.approx(3.0)
    # R3 should average m3-m5 goals scored (3+4+5)/3
    assert match_6["OPP_ADJ_HOME_GOALS_SCORED_R3"] == pytest.approx(4.0)
    # Goals conceded R5: (0+1+0+1+0)/5
    assert match_6["OPP_ADJ_HOME_GOALS_CONCEDED_R5"] == pytest.approx(0.4)


def test_poisson_decomposed_market_features(tmp_path: Path) -> None:
    """US#76: MKT_LAMBDA_* and MKT_POISSON_BTTS_PROB are correctly derived from O/U and AH markets."""
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
                # over25_odds=1.5 → p_over25=2/3; ah_line=-0.5 (home favourite)
                ("pm1", "E0", 1, "2025-08-15 20:00:00", "Team A", "Team B", 2, 1,
                 1.8, 3.4, 4.2, 1.8, 3.4, 4.2, 1.5, 2.5, -0.5, 1.9, 1.9),
                # over25_odds=None (missing) → all lambda features NaN
                ("pm2", "E0", 1, "2025-08-22 20:00:00", "Team A", "Team C", 1, 0,
                 1.8, 3.4, 4.2, 1.8, 3.4, 4.2, None, None, 0.0, 1.9, 1.9),
            ],
        )

    feature_factory = FeatureFactory(config_path=str(config_path))
    features = feature_factory.compute_rolling_stats(window=5)

    # --- pm1: verify lambda inversion and decomposition ---
    m = features.loc[features["match_id"] == "pm1"].iloc[0]
    lam = m["MKT_LAMBDA_TOTAL"]
    assert not pd.isna(lam), "MKT_LAMBDA_TOTAL must not be NaN when over25_odds is present"

    # P(Poisson(λ) ≥ 3) must equal the implied over25 probability (1/1.5)
    p_check = 1.0 - np.exp(-lam) * (1.0 + lam + lam**2 / 2.0)
    assert p_check == pytest.approx(1.0 / 1.5, abs=1e-4)

    # Decomposition: home = (λ + |AH|) / 2, away = (λ − |AH|) / 2
    ah_abs = 0.5
    assert m["MKT_LAMBDA_HOME"] == pytest.approx((lam + ah_abs) / 2.0, abs=1e-4)
    assert m["MKT_LAMBDA_AWAY"] == pytest.approx(max((lam - ah_abs) / 2.0, 0.0), abs=1e-4)
    assert m["MKT_LAMBDA_HOME"] + m["MKT_LAMBDA_AWAY"] == pytest.approx(lam, abs=1e-3)

    # BTTS = (1 − e^−λ_home) × (1 − e^−λ_away)
    expected_btts = (1.0 - np.exp(-m["MKT_LAMBDA_HOME"])) * (1.0 - np.exp(-m["MKT_LAMBDA_AWAY"]))
    assert m["MKT_POISSON_BTTS_PROB"] == pytest.approx(expected_btts, abs=1e-4)

    # AH_DIFF = λ_total − |AH|
    assert m["MKT_LAMBDA_AH_DIFF"] == pytest.approx(lam - ah_abs, abs=1e-4)

    # --- pm2: missing over25_odds → all Poisson features are NaN ---
    m2 = features.loc[features["match_id"] == "pm2"].iloc[0]
    for col in ["MKT_LAMBDA_TOTAL", "MKT_LAMBDA_HOME", "MKT_LAMBDA_AWAY",
                "MKT_POISSON_BTTS_PROB", "MKT_LAMBDA_AH_DIFF"]:
        assert pd.isna(m2[col]), f"{col} should be NaN when over25_odds is missing"


def test_opp_adjusted_features_combine_home_and_away_venues(tmp_path: Path) -> None:
    """OPP_ADJ rolling must aggregate a team's stats across both home and away matches."""
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
                # Alpha FC plays at home (scoring 2)
                ("h1", "E0", 1, "2025-08-01 20:00:00", "Alpha FC", "Opp1", 2, 0, 2.0, 3.2, 3.8, 2.0, 3.2, 3.8),
                # Alpha FC plays away (scoring 3)
                ("a1", "E0", 1, "2025-08-08 20:00:00", "Opp2", "Alpha FC", 1, 3, 2.0, 3.2, 3.8, 2.0, 3.2, 3.8),
                # Alpha FC plays at home again (scoring 1)
                ("h2", "E0", 1, "2025-08-15 20:00:00", "Alpha FC", "Opp3", 1, 0, 2.0, 3.2, 3.8, 2.0, 3.2, 3.8),
                # Match under test: Alpha FC at home vs Beta FC
                ("target", "E0", 1, "2025-08-22 20:00:00", "Alpha FC", "Beta FC", 9, 9, 2.0, 3.2, 3.8, 2.0, 3.2, 3.8),
            ],
        )

    feature_factory = FeatureFactory(config_path=str(config_path))
    features = feature_factory.compute_rolling_stats(window=5)
    target = features.loc[features["match_id"] == "target"].iloc[0]

    # Alpha FC's combined-venue goals scored R3: h1=2, a1=3, h2=1 → (2+3+1)/3 = 2.0
    assert target["OPP_ADJ_HOME_GOALS_SCORED_R3"] == pytest.approx(2.0)
    # Alpha FC's combined-venue goals conceded R3: h1=0, a1=1 (opponent fthg), h2=0 → (0+1+0)/3
    assert target["OPP_ADJ_HOME_GOALS_CONCEDED_R3"] == pytest.approx(1 / 3)


def test_build_for_match_includes_squad_and_luck_columns(tmp_path: Path) -> None:
    """BUG-012 layer 1: build_for_match() must compute the same SQUAD_*/LUCK_*
    columns that compute_rolling_stats() computes for feature_store, or
    ForecastService.forecast_upcoming's league path KeyErrors on them."""
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
            VALUES
            ('m1', 'E0', 1, '2025-08-10 20:00:00', 'Arsenal', 'Everton', 2, 1, 1.5, 4.0, 5.0, 1.5, 4.0, 5.0),
            ('m2', 'E0', 1, '2025-08-17 20:00:00', 'Arsenal', 'Everton', 1, 1, 1.5, 4.0, 5.0, 1.5, 4.0, 5.0)
            """
        )
        conn.execute(
            """
            CREATE TABLE raw_player_match_stats (
                match_id TEXT, team_name TEXT, xg FLOAT, xa FLOAT, rating FLOAT,
                goals INTEGER, assists INTEGER
            )
            """
        )
        conn.executemany(
            "INSERT INTO raw_player_match_stats VALUES (?, ?, ?, ?, ?, ?, ?)",
            [
                ("m1", "Arsenal", 0.6, 0.2, 7.0, 2, 1),
                ("m1", "Everton", 0.4, 0.1, 6.5, 1, 0),
            ],
        )

    feature_factory = FeatureFactory(config_path=str(config_path))
    row = feature_factory.build_for_match(
        home_team="Arsenal", away_team="Everton", match_date="2025-08-24",
        league="E0", odds_h=1.8, odds_d=3.6, odds_a=4.2,
    )

    expected_new_columns = [
        "SQUAD_HOME_XG_MEAN_R3", "SQUAD_HOME_XG_MEAN_R5",
        "SQUAD_HOME_XA_MEAN_R3", "SQUAD_HOME_XA_MEAN_R5",
        "SQUAD_HOME_RATING_MEAN_R3", "SQUAD_HOME_RATING_MEAN_R5",
        "SQUAD_AWAY_XG_MEAN_R3", "SQUAD_AWAY_XG_MEAN_R5",
        "SQUAD_AWAY_XA_MEAN_R3", "SQUAD_AWAY_XA_MEAN_R5",
        "SQUAD_AWAY_RATING_MEAN_R3", "SQUAD_AWAY_RATING_MEAN_R5",
        "LUCK_HOME_BURNOUT_R5", "LUCK_AWAY_BURNOUT_R5",
    ]
    for col in expected_new_columns:
        assert col in row.columns, f"build_for_match() missing column: {col}"
