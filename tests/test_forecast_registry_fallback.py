"""Regression tests for US#107: forecast_upcoming's league branch must consult
config/competitions.yaml before committing to full team-history features. A
match_type='league' call for a competition the registry doesn't know about
(or explicitly tiers as general_purpose) must route to the market-odds-only
path automatically, instead of silently cold-starting a mislabeled
'team_history_and_market' result. A call for a real competition_specific
competition (E0) must be unaffected."""

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


def _write_competitions_registry(config_path: Path, competitions: dict) -> None:
    registry_path = config_path.parent / "config" / "competitions.yaml"
    registry_path.write_text(yaml.safe_dump({"competitions": competitions}), encoding="utf-8")


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


def _write_model(
    config_path: Path, context: str, target: str, feature_names: list[str]
) -> None:
    """Write a dummy model + its .metadata.json, and register it under the
    given model_selection.yaml context (merging with whatever is already there
    so a test can register both 'league' and 'international' contexts)."""
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    model_dir = Path(config["paths"]["model_dir"])

    X = pd.DataFrame({name: [1.0, 2.0, 3.0] for name in feature_names})
    model = DummyRegressor(strategy="constant", constant=1.5).fit(X, [1, 2, 3])
    model_path = model_dir / f"{target}_{context}_dummy_v1_20260701.joblib"
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
        "feature_subset": feature_names,
    }

    selection_path = config_path.parent / "config" / "model_selection.yaml"
    existing = {}
    if selection_path.exists():
        existing = yaml.safe_load(selection_path.read_text(encoding="utf-8")) or {}
    existing.setdefault("contexts", {}).setdefault(context, {})[target] = entry
    selection_path.write_text(yaml.safe_dump(existing), encoding="utf-8")


def test_forecast_upcoming_unregistered_league_falls_back_to_market_odds_only(tmp_path: Path) -> None:
    """A league string the registry has never heard of (e.g. 'Bundesliga') must
    not reach FeatureFactory.build_for_match at all -- it must route to the
    same market-odds-only path as an explicit match_type='international' call.

    A49: replaced "La Liga" as this test's stock unregistered-competition
    example -- La Liga (SP1) is now genuinely registered in the real
    config/competitions.yaml (US#147), so it no longer demonstrates the
    unregistered path (this test's own synthetic registry below only ever
    registers "E0" regardless, but the string choice matters for reading
    clarity/consistency across the codebase's test suite). Bundesliga/D1 is
    confirmed to still have no registry entry."""
    config_path = _write_config(tmp_path, schema_features=["OFF_HOME_FTHG_R5", "MKT_IMPLIED_HOME"])
    _write_competitions_registry(
        config_path,
        {"E0": {"competition_id": "E0", "tier": "competition_specific", "league_code": "E0"}},
    )
    _write_raw_matches(config_path)
    _write_model(config_path, context="international", target="home_goals", feature_names=["MKT_IMPLIED_HOME"])

    service = ForecastService(config_path=str(config_path), targets=["home_goals"])
    result = service.forecast_upcoming(
        home_team="Bayern Munich", away_team="Borussia Dortmund", date="2025-08-24", league="Bundesliga",
        odds_h=1.8, odds_d=3.6, odds_a=4.2, match_type="league",
    )

    assert result["data_quality"]["prediction_basis"] == "market_odds_only"
    assert result["data_quality"]["caveat"] == "Market-odds-only prediction; team history not used."
    assert "home_goals" in result["forecast"]


def test_forecast_upcoming_registered_general_purpose_tier_falls_back(tmp_path: Path) -> None:
    """A competition explicitly registered with tier='general_purpose' (not just
    an unregistered one) must also route to market-odds-only, not team history."""
    config_path = _write_config(tmp_path, schema_features=["OFF_HOME_FTHG_R5", "MKT_IMPLIED_HOME"])
    _write_competitions_registry(
        config_path,
        {
            "E0": {"competition_id": "E0", "tier": "competition_specific", "league_code": "E0"},
            "CL": {"competition_id": "CL", "tier": "general_purpose", "league_code": None},
        },
    )
    _write_raw_matches(config_path)
    _write_model(config_path, context="international", target="home_goals", feature_names=["MKT_IMPLIED_HOME"])

    service = ForecastService(config_path=str(config_path), targets=["home_goals"])
    result = service.forecast_upcoming(
        home_team="Real Madrid", away_team="Bayern Munich", date="2025-08-24", league="CL",
        odds_h=2.1, odds_d=3.4, odds_a=3.3, match_type="league",
    )

    assert result["data_quality"]["prediction_basis"] == "market_odds_only"


def test_forecast_upcoming_e0_still_uses_competition_specific_tier(tmp_path: Path) -> None:
    """Existing E0 behavior must be completely unaffected by the registry check."""
    config_path = _write_config(tmp_path, schema_features=["OFF_HOME_FTHG_R5", "MKT_IMPLIED_HOME"])
    _write_competitions_registry(
        config_path,
        {"E0": {"competition_id": "E0", "tier": "competition_specific", "league_code": "E0"}},
    )
    _write_raw_matches(config_path)
    # US#110: E0's model_selection.yaml bucket is now keyed by its own
    # competition_id ("E0"), not the flat "league" string.
    _write_model(config_path, context="E0", target="home_goals", feature_names=["OFF_HOME_FTHG_R5", "MKT_IMPLIED_HOME"])

    service = ForecastService(config_path=str(config_path), targets=["home_goals"])
    result = service.forecast_upcoming(
        home_team="Arsenal", away_team="Everton", date="2025-08-24", league="E0",
        odds_h=1.8, odds_d=3.6, odds_a=4.2, match_type="league",
    )

    assert result["data_quality"]["prediction_basis"] == "team_history_and_market"
    assert "home_goals" in result["forecast"]
