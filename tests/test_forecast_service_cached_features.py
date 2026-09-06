"""Tests for forecast_upcoming's cached-feature-store fast path (A98).

Root-caused live: refreshing a historical agent-snapshot corpus calls
forecast_upcoming() once per already-played match, which always rebuilds
every rolling/history-dependent feature from scratch via
FeatureFactory.build_for_match() -- including a full walk-forward
Dixon-Coles refit across the league's entire history (194 separate MLE
fits observed for one real E0 match, profiled at ~54% of that match's
total ~93s cost) and a from-scratch rescan of the whole lineup/player-stats
tables for FRDS (~30%). For an already-played match, this is pure waste:
`compute_rolling_stats()` (the offline pipeline that populates
`feature_store`) already computed the exact same feature row, once, using
identical logic (`BUG-012 layer 1`'s parity comment) -- every one of the
379 real E0 matches in this session's snapshot corpus already has a row
sitting in `feature_store` unused.

This fast path: before calling build_for_match, forecast_upcoming's league
branch now checks feature_store (joined on raw_matches by team/date/league,
not match_id -- the caller doesn't have to know or supply match_id) for an
existing row and uses it directly, skipping FeatureFactory entirely, when
found. Falls back to today's live computation, byte-for-byte unchanged,
whenever no cached row exists -- which is always true for a genuinely
live/upcoming match (it can't have a raw_matches row yet, so it can't have
a feature_store row either), so this is a pure no-op for real live serving.
"""

from __future__ import annotations

import json
from pathlib import Path
import sys

import joblib
import numpy as np
import pandas as pd
import pytest
import yaml
from sklearn.linear_model import LinearRegression

sys.path.append(str(Path(__file__).resolve().parents[1]))

duckdb = pytest.importorskip("duckdb")

from src.forecast.forecast_service import ForecastService


def _write_config(tmp_path: Path) -> Path:
    db_path = tmp_path / "forecast.db"
    model_dir = tmp_path / "models"
    model_dir.mkdir()
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        yaml.safe_dump({"paths": {"database_path": str(db_path), "model_dir": str(model_dir)}}),
        encoding="utf-8",
    )
    schema_dir = tmp_path / "config"
    schema_dir.mkdir()
    schema_dir.joinpath("schema.yaml").write_text(
        yaml.safe_dump({"training_setup": {"selected_features": ["OFF_HOME_FTHG_R5"]}}),
        encoding="utf-8",
    )
    schema_dir.joinpath("competitions.yaml").write_text(
        yaml.safe_dump({"competitions": {"E0": {
            "competition_id": "E0", "tier": "competition_specific", "league_code": "E0",
        }}}),
        encoding="utf-8",
    )
    return config_path


def _write_raw_matches(config_path: Path) -> None:
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    with duckdb.connect(config["paths"]["database_path"]) as conn:
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
            VALUES ('m1', 'E0', 1, '2025-08-10 20:00:00', 'Arsenal', 'Everton', 2, 1, 1.5, 4.0, 5.0, 1.5, 4.0, 5.0)
            """
        )


def _write_feature_store_row(config_path: Path, match_id: str, feature_values: dict) -> None:
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    columns_sql = ", ".join(f'"{name}" FLOAT' for name in feature_values)
    with duckdb.connect(config["paths"]["database_path"]) as conn:
        conn.execute(f"CREATE TABLE feature_store (match_id TEXT PRIMARY KEY, {columns_sql})")
        placeholders = ", ".join(["?"] * (1 + len(feature_values)))
        conn.execute(
            f"INSERT INTO feature_store (match_id, {', '.join(feature_values)}) VALUES ({placeholders})",
            [match_id, *feature_values.values()],
        )


def _write_league_model(config_path: Path, target: str, feature_names: list[str]) -> None:
    """A LinearRegression fit so predict(x) ~= x -- lets the test tell apart
    which feature source (cached vs. freshly computed) actually fed the
    model, unlike a constant-output DummyRegressor."""
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    model_dir = Path(config["paths"]["model_dir"])

    X = pd.DataFrame({name: [0.0, 1.0, 2.0] for name in feature_names})
    model = LinearRegression().fit(X, [0.0, 1.0, 2.0])
    model_path = model_dir / f"{target}_linear_v1_20260701.joblib"
    joblib.dump(model, model_path)
    model_path.with_suffix(model_path.suffix + ".metadata.json").write_text(
        json.dumps({
            "target": target, "task_type": "regression", "model_type": "LinearRegressionModel",
            "artifact_name": model_path.name, "created_at": "2026-07-01T00:00:00Z",
            "feature_names": feature_names, "feature_importance": [],
        }),
        encoding="utf-8",
    )
    selection_path = config_path.parent / "config" / "model_selection.yaml"
    selection_path.write_text(
        yaml.safe_dump({"contexts": {"E0": {target: {
            "model_path": str(model_path.relative_to(config_path.parent)),
            "model_type": "LinearRegressionModel", "metric_name": "test_mae", "metric_value": 1.0,
            "selected_at": "2026-07-01T00:00:00Z", "feature_subset": feature_names,
        }}}}),
        encoding="utf-8",
    )


def test_forecast_upcoming_uses_cached_feature_store_row_when_present(tmp_path: Path) -> None:
    """The core fast path: a feature_store row already exists for this exact
    (team, team, date, league) -- forecast_upcoming must use its value
    (42.0) directly rather than recomputing live (which would cold-start-
    impute 0.0, since this match is the team's only row in raw_matches)."""
    config_path = _write_config(tmp_path)
    _write_raw_matches(config_path)
    _write_feature_store_row(config_path, "m1", {"OFF_HOME_FTHG_R5": 42.0})
    _write_league_model(config_path, "home_goals", ["OFF_HOME_FTHG_R5"])

    service = ForecastService(config_path=str(config_path), targets=["home_goals"])
    result = service.forecast_upcoming(
        home_team="Arsenal", away_team="Everton", date="2025-08-10", league="E0",
        odds_h=1.5, odds_d=4.0, odds_a=5.0, match_type="league",
    )

    predicted = result["forecast"]["home_goals"]["expected"]
    assert predicted == pytest.approx(42.0, abs=0.05), (
        f"Expected the cached feature_store value (42.0) to reach the model, got {predicted} "
        "-- looks like build_for_match's live (cold-start) path ran instead."
    )


def test_forecast_upcoming_falls_back_to_live_computation_when_no_cached_row(tmp_path: Path) -> None:
    """No feature_store row for this match (e.g. a genuinely upcoming match
    that hasn't been played/ingested yet) -- must fall back to today's
    build_for_match path exactly as before, not error."""
    config_path = _write_config(tmp_path)
    _write_raw_matches(config_path)
    # deliberately no feature_store row written
    _write_league_model(config_path, "home_goals", ["OFF_HOME_FTHG_R5"])

    service = ForecastService(config_path=str(config_path), targets=["home_goals"])
    result = service.forecast_upcoming(
        home_team="Arsenal", away_team="Everton", date="2025-08-10", league="E0",
        odds_h=1.5, odds_d=4.0, odds_a=5.0, match_type="league",
    )

    # Cold-start (only 1 historical row, no prior match for the rolling
    # window to look back on) -- imputed to 0.0, same as unmodified behavior.
    predicted = result["forecast"]["home_goals"]["expected"]
    assert predicted == pytest.approx(0.0, abs=0.05)
    assert result["data_quality"]["prediction_basis"] == "team_history_and_market"
