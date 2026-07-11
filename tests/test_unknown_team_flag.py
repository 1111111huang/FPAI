"""Regression tests for US#108: FeatureFactory.build_for_match() must detect
when a requested team has zero rows in raw_matches at all (not just sparse/
missing individual feature values) and surface an explicit, distinguishable
'_unknown_team' diagnostic column -- distinct from the existing feature-value-
level cold-start imputation, since 'team exists but form data is thin' and
'team has never been seen' are different failure modes worth surfacing
differently to a caller."""

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

from src.features.feature_factory import FeatureFactory
from src.forecast.forecast_service import ForecastService


def _build_db(tmp_path: Path) -> Path:
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
    return config_path


def test_build_for_match_flags_unknown_team_when_home_has_zero_history(tmp_path: Path) -> None:
    """Coventry City (promoted, never in raw_matches) as the home side must be
    flagged -- Everton having real history doesn't hide the gap."""
    config_path = _build_db(tmp_path)
    factory = FeatureFactory(config_path=str(config_path))
    row = factory.build_for_match(
        home_team="Coventry City", away_team="Everton", match_date="2025-08-24",
        league="E0", odds_h=2.5, odds_d=3.2, odds_a=2.9,
    )
    assert bool(row["_unknown_team"].iloc[0]) is True


def test_build_for_match_flags_unknown_team_when_away_has_zero_history(tmp_path: Path) -> None:
    config_path = _build_db(tmp_path)
    factory = FeatureFactory(config_path=str(config_path))
    row = factory.build_for_match(
        home_team="Arsenal", away_team="Coventry City", match_date="2025-08-24",
        league="E0", odds_h=1.4, odds_d=4.5, odds_a=7.0,
    )
    assert bool(row["_unknown_team"].iloc[0]) is True


def test_build_for_match_does_not_flag_known_teams(tmp_path: Path) -> None:
    """Regression: two teams with real history must not be flagged, even
    though their rolling features are still thin (only 2 prior matches)."""
    config_path = _build_db(tmp_path)
    factory = FeatureFactory(config_path=str(config_path))
    row = factory.build_for_match(
        home_team="Arsenal", away_team="Everton", match_date="2025-08-24",
        league="E0", odds_h=1.8, odds_d=3.6, odds_a=4.2,
    )
    assert bool(row["_unknown_team"].iloc[0]) is False


def test_build_for_match_flags_unknown_team_when_both_sides_unseen(tmp_path: Path) -> None:
    config_path = _build_db(tmp_path)
    factory = FeatureFactory(config_path=str(config_path))
    row = factory.build_for_match(
        home_team="Coventry City", away_team="Luton Town", match_date="2025-08-24",
        league="E0", odds_h=2.0, odds_d=3.3, odds_a=3.8,
    )
    assert bool(row["_unknown_team"].iloc[0]) is True


def _write_e0_league_model(config_path: Path, feature_names: list[str]) -> None:
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    model_dir = Path(config["paths"]["model_dir"])
    model_dir.mkdir(exist_ok=True)

    schema_dir = config_path.parent / "config"
    schema_dir.mkdir(exist_ok=True)
    schema_dir.joinpath("schema.yaml").write_text(
        yaml.safe_dump({"training_setup": {"selected_features": feature_names}}), encoding="utf-8",
    )

    X = pd.DataFrame({name: [1.0, 2.0, 3.0] for name in feature_names})
    model = DummyRegressor(strategy="constant", constant=1.5).fit(X, [1, 2, 3])
    model_path = model_dir / "home_goals_dummy_v1_20260701.joblib"
    joblib.dump(model, model_path)
    model_path.with_suffix(model_path.suffix + ".metadata.json").write_text(
        json.dumps({
            "target": "home_goals", "task_type": "regression", "model_type": "DummyRegressorModel",
            "artifact_name": model_path.name, "created_at": "2026-07-01T00:00:00Z",
            "feature_names": feature_names, "feature_importance": [],
        }),
        encoding="utf-8",
    )
    (config_path.parent / "config" / "competitions.yaml").write_text(
        yaml.safe_dump({"competitions": {"E0": {"competition_id": "E0", "tier": "competition_specific", "league_code": "E0"}}}),
        encoding="utf-8",
    )
    (config_path.parent / "config" / "model_selection.yaml").write_text(
        yaml.safe_dump({"contexts": {"league": {"home_goals": {
            "model_path": str(model_path.relative_to(config_path.parent)),
            "model_type": "DummyRegressorModel", "metric_name": "test_mae", "metric_value": 1.0,
            "selected_at": "2026-07-01T00:00:00Z", "feature_subset": feature_names,
        }}}}),
        encoding="utf-8",
    )


def test_forecast_upcoming_surfaces_unknown_team_in_data_quality(tmp_path: Path) -> None:
    """End-to-end: a genuinely unseen team must produce a payload where
    data_quality.unknown_team is explicitly True -- not just a low
    feature_completeness number indistinguishable from partial cold-start."""
    config_path = _build_db(tmp_path)
    _write_e0_league_model(config_path, feature_names=["OFF_HOME_FTHG_R5"])

    service = ForecastService(config_path=str(config_path), targets=["home_goals"])
    result = service.forecast_upcoming(
        home_team="Coventry City", away_team="Everton", date="2025-08-24", league="E0",
        odds_h=2.5, odds_d=3.2, odds_a=2.9, match_type="league",
    )
    assert result["data_quality"]["unknown_team"] is True


def test_forecast_upcoming_known_teams_not_flagged_unknown(tmp_path: Path) -> None:
    """Existing tests for known teams are unaffected."""
    config_path = _build_db(tmp_path)
    _write_e0_league_model(config_path, feature_names=["OFF_HOME_FTHG_R5"])

    service = ForecastService(config_path=str(config_path), targets=["home_goals"])
    result = service.forecast_upcoming(
        home_team="Arsenal", away_team="Everton", date="2025-08-24", league="E0",
        odds_h=1.8, odds_d=3.6, odds_a=4.2, match_type="league",
    )
    assert result["data_quality"]["unknown_team"] is False
