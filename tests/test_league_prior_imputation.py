"""Regression tests for US#153: a fully-cold-start forecast (two genuinely
unseen teams, zero rows for either in raw_matches) currently falls back to a
flat 0.0 for any rolling feature it can't compute -- FeatureFactory's own
per-call cold-start mean has nothing to average from by definition. This
replaces that flat 0.0 with a real league-wide prior sourced from the
persisted feature_store table."""

from __future__ import annotations

from pathlib import Path
import sys

import pytest
import yaml

sys.path.append(str(Path(__file__).resolve().parents[1]))

duckdb = pytest.importorskip("duckdb")

from src.features.league_prior import compute_league_prior
from src.utils.db_manager import DuckDBManager


def _build_db_with_feature_store(tmp_path: Path) -> Path:
    """raw_matches with real E0/SP1 history, plus a feature_store populated
    with real per-match feature values for the same matches -- the
    league-wide table compute_league_prior reads from, distinct from
    build_for_match's own single-call frame (which has nothing to average
    from for a genuinely unseen team pair)."""
    db_path = tmp_path / "test_fpai.db"
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        yaml.safe_dump({"paths": {"database_path": str(db_path)}}), encoding="utf-8",
    )

    with duckdb.connect(str(db_path)) as conn:
        conn.execute(
            """
            CREATE TABLE raw_matches (
                match_id TEXT PRIMARY KEY, league TEXT, date TIMESTAMP,
                home_team TEXT, away_team TEXT
            )
            """
        )
        conn.execute(
            """
            INSERT INTO raw_matches (match_id, league, date, home_team, away_team) VALUES
            ('m1', 'E0', '2025-08-10 20:00:00', 'Arsenal', 'Everton'),
            ('m2', 'E0', '2025-08-17 20:00:00', 'Chelsea', 'Fulham'),
            ('m3', 'E0', '2025-08-24 20:00:00', 'Liverpool', 'Brighton'),
            ('m4', 'SP1', '2025-08-24 20:00:00', 'Sevilla', 'Valencia')
            """
        )
        conn.execute(
            """
            CREATE TABLE feature_store (
                match_id TEXT PRIMARY KEY, "OFF_HOME_FTHG_R5" FLOAT, "MKT_IMPLIED_HOME" FLOAT
            )
            """
        )
        conn.execute(
            """
            INSERT INTO feature_store (match_id, "OFF_HOME_FTHG_R5", "MKT_IMPLIED_HOME") VALUES
            ('m1', 1.0, 0.5),
            ('m2', 2.0, 0.6),
            ('m3', 3.0, NULL),
            ('m4', 100.0, 0.9)
            """
        )
    return config_path


def test_compute_league_prior_returns_the_real_league_average(tmp_path):
    config_path = _build_db_with_feature_store(tmp_path)
    db_manager = DuckDBManager(config_path=str(config_path))

    prior = compute_league_prior(db_manager, "E0", ["OFF_HOME_FTHG_R5"])

    # (1.0 + 2.0 + 3.0) / 3 = 2.0 -- E0's own 3 rows only, SP1's 100.0 excluded.
    assert prior == {"OFF_HOME_FTHG_R5": 2.0}


def test_compute_league_prior_omits_a_column_that_is_entirely_null_for_this_league(tmp_path):
    config_path = _build_db_with_feature_store(tmp_path)
    db_manager = DuckDBManager(config_path=str(config_path))

    prior = compute_league_prior(db_manager, "E0", ["MKT_IMPLIED_HOME"])

    # E0's own 3 rows: 0.5, 0.6, NULL -- AVG ignores the NULL, real mean = 0.55.
    assert prior == {"MKT_IMPLIED_HOME": pytest.approx(0.55)}


def test_compute_league_prior_omits_a_column_missing_from_every_row_of_this_league(tmp_path):
    """A competition genuinely lacking a feature family entirely (e.g. no
    corners data ingested yet) must NOT silently fall back to another
    competition's average -- the column is simply absent from the result,
    same 'stays NaN, no cross-competition contamination' contract US#134
    already established for the per-call mean."""
    config_path = _build_db_with_feature_store(tmp_path)
    db_manager = DuckDBManager(config_path=str(config_path))

    prior = compute_league_prior(db_manager, "SWE", ["OFF_HOME_FTHG_R5"])

    assert prior == {}


def test_compute_league_prior_returns_empty_dict_for_no_columns(tmp_path):
    config_path = _build_db_with_feature_store(tmp_path)
    db_manager = DuckDBManager(config_path=str(config_path))

    assert compute_league_prior(db_manager, "E0", []) == {}


def test_compute_league_prior_tolerates_a_missing_feature_store_table(tmp_path):
    """A fresh DB with raw_matches but no feature_store at all yet (e.g. the
    offline compute_rolling_stats() pipeline has never run) must return no
    prior, not raise -- forecast_upcoming must never fail just because
    feature_store hasn't been populated, same 'missing persistence = no
    signal, don't crash' contract this codebase already uses elsewhere."""
    db_path = tmp_path / "test_fpai.db"
    config_path = tmp_path / "config.yaml"
    config_path.write_text(yaml.safe_dump({"paths": {"database_path": str(db_path)}}), encoding="utf-8")
    with duckdb.connect(str(db_path)) as conn:
        conn.execute("CREATE TABLE raw_matches (match_id TEXT PRIMARY KEY, league TEXT)")
    db_manager = DuckDBManager(config_path=str(config_path))

    assert compute_league_prior(db_manager, "E0", ["OFF_HOME_FTHG_R5"]) == {}


# ---------------------------------------------------------------------------
# End-to-end: ForecastService.forecast_upcoming for a fully-cold-start pair.
# ---------------------------------------------------------------------------

import json

import joblib
from sklearn.linear_model import LinearRegression

from src.forecast.forecast_service import ForecastService


def _build_full_db(tmp_path: Path) -> Path:
    """Same raw_matches schema build_for_match's broader queries expect
    (test_unknown_team_flag.py's own _build_db), plus a feature_store table
    with real historical values for the two known E0 teams -- the league
    prior compute_league_prior reads from."""
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
            ('m1', 'E0', 1, '2025-08-10 20:00:00', 'Arsenal', 'Everton', 2, 1, 1.5, 4.0, 5.0, 1.5, 4.0, 5.0),
            ('m2', 'E0', 1, '2025-08-17 20:00:00', 'Chelsea', 'Fulham', 4, 0, 1.3, 4.5, 6.0, 1.3, 4.5, 6.0)
            """
        )
        conn.execute(
            """
            CREATE TABLE feature_store (
                match_id TEXT PRIMARY KEY, "OFF_HOME_FTHG_R5" FLOAT
            )
            """
        )
        conn.execute(
            """
            INSERT INTO feature_store (match_id, "OFF_HOME_FTHG_R5") VALUES
            ('m1', 2.0),
            ('m2', 4.0)
            """
        )
    return config_path


def _write_linear_e0_model(config_path: Path, feature_name: str) -> None:
    """A model whose prediction equals its single input feature exactly
    (coefficient 1.0, intercept 0.0) -- unlike a constant DummyRegressor,
    this makes the imputed feature value directly observable in the
    forecast's own output, which is what this story's own acceptance
    criteria asks for ('the prediction should move toward a league-typical
    range, not stay pinned near the artificially weak all-zero baseline')."""
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    model_dir = Path(config["paths"]["model_dir"])
    model_dir.mkdir(exist_ok=True)

    schema_dir = config_path.parent / "config"
    schema_dir.mkdir(exist_ok=True)
    schema_dir.joinpath("schema.yaml").write_text(
        yaml.safe_dump({"training_setup": {"selected_features": [feature_name]}}), encoding="utf-8",
    )

    model = LinearRegression().fit([[0.0], [1.0], [2.0]], [0.0, 1.0, 2.0])
    model_path = model_dir / "home_goals_linear_v1_20260701.joblib"
    joblib.dump(model, model_path)
    model_path.with_suffix(model_path.suffix + ".metadata.json").write_text(
        json.dumps({
            "target": "home_goals", "task_type": "regression", "model_type": "LinearRegressionModel",
            "artifact_name": model_path.name, "created_at": "2026-07-01T00:00:00Z",
            "feature_names": [feature_name], "feature_importance": [],
        }),
        encoding="utf-8",
    )
    schema_dir.joinpath("competitions.yaml").write_text(
        yaml.safe_dump({"competitions": {"E0": {"competition_id": "E0", "tier": "competition_specific", "league_code": "E0"}}}),
        encoding="utf-8",
    )
    schema_dir.joinpath("model_selection.yaml").write_text(
        yaml.safe_dump({"contexts": {"E0": {"home_goals": {
            "model_path": str(model_path.relative_to(config_path.parent)),
            "model_type": "LinearRegressionModel", "metric_name": "test_mae", "metric_value": 1.0,
            "selected_at": "2026-07-01T00:00:00Z", "feature_subset": [feature_name],
        }}}}),
        encoding="utf-8",
    )


def test_forecast_upcoming_fills_fully_cold_start_feature_with_league_prior_not_zero(tmp_path):
    """Two genuinely unseen teams -- Coventry City v Luton Town, neither in
    raw_matches at all. OFF_HOME_FTHG_R5's real E0 average (from
    feature_store: (2.0 + 4.0) / 2 = 3.0) should feed the model, not 0.0."""
    config_path = _build_full_db(tmp_path)
    _write_linear_e0_model(config_path, feature_name="OFF_HOME_FTHG_R5")

    service = ForecastService(config_path=str(config_path), targets=["home_goals"])
    result = service.forecast_upcoming(
        home_team="Coventry City", away_team="Luton Town", date="2025-08-24", league="E0",
        odds_h=2.0, odds_d=3.3, odds_a=3.8, match_type="league",
    )

    assert result["data_quality"]["unknown_team"] is True
    assert result["forecast"]["home_goals"]["expected"] == pytest.approx(3.0)


def test_forecast_upcoming_known_teams_unaffected_by_league_prior(tmp_path):
    """A known team pair's real feature_store row must not be touched by
    the new fallback -- it only ever applies to a value still NaN after
    everything else has already tried and failed."""
    config_path = _build_full_db(tmp_path)
    _write_linear_e0_model(config_path, feature_name="OFF_HOME_FTHG_R5")

    with duckdb.connect(str(config_path.parent / "test_fpai.db")) as conn:
        conn.execute("UPDATE feature_store SET \"OFF_HOME_FTHG_R5\" = 9.0 WHERE match_id = 'm1'")

    service = ForecastService(config_path=str(config_path), targets=["home_goals"])
    result = service.forecast_upcoming(
        home_team="Arsenal", away_team="Everton", date="2025-08-10", league="E0",
        odds_h=1.5, odds_d=4.0, odds_a=5.0, match_type="league",
    )

    assert result["data_quality"]["unknown_team"] is False
    assert result["forecast"]["home_goals"]["expected"] == pytest.approx(9.0)
