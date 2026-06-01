from __future__ import annotations

import json
from pathlib import Path
import sys

import joblib
import pandas as pd
import pytest
import yaml
from sklearn.dummy import DummyRegressor
from sklearn.linear_model import LogisticRegression

sys.path.append(str(Path(__file__).resolve().parents[1]))

duckdb = pytest.importorskip("duckdb")

from src.forecast import ForecastService, validate_forecast_payload


def _write_config(tmp_path: Path) -> Path:
    db_path = tmp_path / "forecast.db"
    model_dir = tmp_path / "models"
    model_dir.mkdir()
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        yaml.safe_dump(
            {
                "paths": {
                    "database_path": str(db_path),
                    "model_dir": str(model_dir),
                }
            }
        ),
        encoding="utf-8",
    )
    schema_dir = tmp_path / "config"
    schema_dir.mkdir()
    schema_dir.joinpath("schema.yaml").write_text(
        yaml.safe_dump(
            {
                "training_setup": {
                    "selected_features": ["OFF_HOME_FTHG_R5", "MKT_IMPLIED_HOME"]
                }
            }
        ),
        encoding="utf-8",
    )
    return config_path


def _write_database(config_path: Path) -> None:
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    with duckdb.connect(config["paths"]["database_path"]) as conn:
        conn.execute(
            """
            CREATE TABLE raw_matches (
                match_id TEXT PRIMARY KEY,
                date TIMESTAMP,
                league TEXT,
                home_team TEXT,
                away_team TEXT
            )
            """
        )
        conn.execute(
            """
            CREATE TABLE feature_store (
                match_id TEXT PRIMARY KEY,
                OFF_HOME_FTHG_R5 FLOAT,
                MKT_IMPLIED_HOME FLOAT
            )
            """
        )
        conn.execute(
            """
            INSERT INTO raw_matches
            VALUES ('m1', '2026-05-25 15:00:00', 'E0', 'Liverpool', 'Arsenal')
            """
        )
        conn.execute("INSERT INTO feature_store VALUES ('m1', 1.7, 0.48)")


def _write_artifacts(config_path: Path) -> None:
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    model_dir = Path(config["paths"]["model_dir"])
    X = pd.DataFrame(
        {
            "OFF_HOME_FTHG_R5": [0.8, 1.2, 1.8, 2.1],
            "MKT_IMPLIED_HOME": [0.3, 0.4, 0.55, 0.7],
        }
    )
    classifier = LogisticRegression().fit(X, [0, 0, 1, 1])
    classifier_path = model_dir / "btts_lr_v1_20260525.joblib"
    joblib.dump(classifier, classifier_path)
    classifier_path.with_suffix(".joblib.metadata.json").write_text(
        json.dumps(
            {
                "target": "btts",
                "task_type": "binary_classification",
                "classes": ["no", "yes"],
                "model_type": "LRModel",
                "artifact_name": classifier_path.name,
                "created_at": "2026-05-25T00:00:00Z",
                "feature_names": list(X.columns),
                "feature_importance": [
                    {"feature": "MKT_IMPLIED_HOME", "importance": 0.7},
                    {"feature": "OFF_HOME_FTHG_R5", "importance": 0.3},
                ],
            }
        ),
        encoding="utf-8",
    )

    regressor = DummyRegressor(strategy="constant", constant=2.4).fit(X, [2, 3, 1, 4])
    regressor_path = model_dir / "total_goals_randomforestregressor_v1_20260525.joblib"
    joblib.dump(regressor, regressor_path)
    regressor_path.with_suffix(".joblib.metadata.json").write_text(
        json.dumps(
            {
                "target": "total_goals",
                "task_type": "regression",
                "model_type": "RandomForestRegressorModel",
                "artifact_name": regressor_path.name,
                "created_at": "2026-05-25T00:00:00Z",
                "feature_names": list(X.columns),
                "prediction_interval": {
                    "coverage": 0.8,
                    "lower_residual": -0.9,
                    "upper_residual": 1.1,
                    "method": "validation_residual_quantile",
                },
                "feature_importance": [
                    {"feature": "OFF_HOME_FTHG_R5", "importance": 0.5},
                ],
            }
        ),
        encoding="utf-8",
    )


def test_forecast_service_emits_agent_payload(tmp_path: Path) -> None:
    config_path = _write_config(tmp_path)
    _write_database(config_path)
    _write_artifacts(config_path)

    service = ForecastService(config_path=str(config_path), targets=["btts", "total_goals"])
    payloads = service.forecast(match_ids=["m1"])

    assert len(payloads) == 1
    payload = payloads[0]
    validate_forecast_payload(payload)
    assert payload["match_id"] == "m1"
    assert set(payload["forecast"]) == {"btts", "total_goals"}
    assert set(payload["forecast"]["btts"]["probabilities"]) == {"no", "yes"}
    assert payload["forecast"]["btts"]["uncertainty"]["method"] == "entropy"
    assert payload["forecast"]["total_goals"]["expected"] == pytest.approx(2.4)
    assert payload["forecast"]["total_goals"]["prediction_interval"]["lower"] == pytest.approx(1.5)
    assert set(payload["forecast"]["total_goals"]["distribution"]) == {"0", "1", "2", "3_plus"}
    assert payload["explainability"]["top_features"][0]["name"] == "MKT_IMPLIED_HOME"
    assert payload["diagnostics"]["feature_completeness"] == pytest.approx(1.0)
    assert payload["diagnostics"]["cold_start_risk"] is False


def test_validate_forecast_payload_rejects_missing_keys() -> None:
    with pytest.raises(ValueError, match="missing required keys"):
        validate_forecast_payload({"match_id": "m1"})
