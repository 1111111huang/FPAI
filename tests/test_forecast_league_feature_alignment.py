"""Regression tests for BUG-012: forecast_upcoming's league path must use each
loaded model's own feature list (from its .metadata.json sidecar), not blindly
fall back to schema.yaml's full selected_features list when model_selection.yaml
has no feature_subset override."""

from __future__ import annotations

import json
from pathlib import Path
import sys

import joblib
import pandas as pd
import pytest
import yaml
from sklearn.dummy import DummyRegressor

sys.path.append(str(Path(__file__).resolve().parents[1]))

duckdb = pytest.importorskip("duckdb")

from src.forecast.forecast_service import ForecastService


def _write_config(tmp_path: Path, schema_features: list[str]) -> Path:
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
        yaml.safe_dump({"training_setup": {"selected_features": schema_features}}),
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


def _write_league_model(
    config_path: Path, target: str, feature_names: list[str], model_selection_feature_subset: list[str] | None
) -> None:
    """Write a dummy league-context model + its .metadata.json, and register it
    in config/model_selection.yaml exactly as ModelSelector would (optionally
    without feature_subset, reproducing the BUG-012 condition)."""
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    model_dir = Path(config["paths"]["model_dir"])

    X = pd.DataFrame({name: [1.0, 2.0, 3.0] for name in feature_names})
    model = DummyRegressor(strategy="constant", constant=1.5).fit(X, [1, 2, 3])
    model_path = model_dir / f"{target}_dummy_v1_20260701.joblib"
    joblib.dump(model, model_path)
    model_path.with_suffix(model_path.suffix + ".metadata.json").write_text(
        json.dumps(
            {
                "target": target,
                "task_type": "regression",
                "model_type": "DummyRegressorModel",
                "artifact_name": model_path.name,
                "created_at": "2026-07-01T00:00:00Z",
                "feature_names": feature_names,
                "feature_importance": [],
            }
        ),
        encoding="utf-8",
    )

    entry = {
        "model_path": str(model_path.relative_to(config_path.parent)),
        "model_type": "DummyRegressorModel",
        "metric_name": "test_mae",
        "metric_value": 1.0,
        "selected_at": "2026-07-01T00:00:00Z",
    }
    if model_selection_feature_subset is not None:
        entry["feature_subset"] = model_selection_feature_subset

    selection_path = config_path.parent / "config" / "model_selection.yaml"
    selection_path.write_text(
        yaml.safe_dump({"contexts": {"league": {target: entry}}}),
        encoding="utf-8",
    )


def test_load_context_models_uses_model_own_metadata_feature_names(tmp_path: Path) -> None:
    """BUG-012 layer 3a: when model_selection.yaml has no feature_subset, the
    per-model .metadata.json feature_names must be used — not schema.yaml's
    full selected_features list."""
    config_path = _write_config(
        tmp_path, schema_features=["OFF_HOME_FTHG_R5", "MKT_IMPLIED_HOME", "SQUAD_HOME_XG_MEAN_R3"]
    )
    _write_raw_matches(config_path)
    _write_league_model(
        config_path,
        target="home_goals",
        feature_names=["OFF_HOME_FTHG_R5", "MKT_IMPLIED_HOME"],  # deliberately a subset of schema
        model_selection_feature_subset=None,  # BUG-012 condition: no override present
    )

    service = ForecastService(config_path=str(config_path), targets=["home_goals"])
    loaded = service._load_context_models("league")

    assert "home_goals" in loaded
    _, __, metadata = loaded["home_goals"]
    assert metadata["feature_names"] == ["OFF_HOME_FTHG_R5", "MKT_IMPLIED_HOME"], (
        f"Expected model's own metadata.json feature_names, got {metadata['feature_names']}"
    )


def test_forecast_upcoming_league_ignores_schema_features_model_was_not_trained_on(tmp_path: Path) -> None:
    """BUG-012 layers 1b+3a+3b end-to-end: forecast_upcoming must not KeyError
    even when schema.yaml's selected_features includes a column (XOC_HOME) that
    build_for_match cannot produce (no match_lineups table), as long as the
    loaded model itself was never trained on that column."""
    config_path = _write_config(
        tmp_path,
        schema_features=["OFF_HOME_FTHG_R5", "MKT_IMPLIED_HOME", "XOC_HOME"],  # XOC_HOME: unproducible today
    )
    _write_raw_matches(config_path)
    _write_league_model(
        config_path,
        target="home_goals",
        feature_names=["OFF_HOME_FTHG_R5", "MKT_IMPLIED_HOME"],  # model never saw XOC_HOME
        model_selection_feature_subset=None,
    )

    service = ForecastService(config_path=str(config_path), targets=["home_goals"])
    result = service.forecast_upcoming(
        home_team="Arsenal", away_team="Everton", date="2025-08-24", league="E0",
        odds_h=1.8, odds_d=3.6, odds_a=4.2, match_type="league",
    )

    assert result["data_quality"]["prediction_basis"] == "team_history_and_market"
    assert "home_goals" in result["forecast"]
