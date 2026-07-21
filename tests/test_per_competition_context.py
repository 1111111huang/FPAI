"""Regression tests for US#110: model context must be keyed off the actual
competition (competition_id), not a flat league/international binary. Before
this fix, every `competition_specific` competition shared the single
`contexts.league` bucket in config/model_selection.yaml -- registering and
training a second competition_specific competition (e.g. Sweden's SWE) would
silently collide with (overwrite or be overwritten by) E0's entries. These
tests use a *fictional* second competition ("T2") registered only in a
tmp_path registry -- Sweden itself is out of scope for this story."""

from __future__ import annotations

import json
from pathlib import Path
import sys
from unittest.mock import MagicMock

import joblib
import pandas as pd
import pytest
import yaml
from sklearn.dummy import DummyRegressor

sys.path.append(str(Path(__file__).resolve().parents[1]))

duckdb = pytest.importorskip("duckdb")

from src.forecast.forecast_service import ForecastService
from src.logic.competition_registry import list_context_keys
from src.utils.model_selection import ModelSelector


# ---------------------------------------------------------------------------
# src/logic/competition_registry.list_context_keys
# ---------------------------------------------------------------------------


def test_list_context_keys_matches_real_registry_today() -> None:
    """Regression: E0 and SWE (both competition_specific) each get their own
    bucket, plus the shared international (general_purpose) bucket -- US#128
    registered SWE alongside E0, so this list grew by one entry."""
    assert list_context_keys() == ["E0", "SWE", "international"]


def test_list_context_keys_gives_each_competition_specific_competition_its_own_bucket(tmp_path: Path) -> None:
    """A second competition_specific competition must get its own key, not be
    merged into (or collide with) E0's -- the core bug this story fixes."""
    registry_path = tmp_path / "competitions.yaml"
    registry_path.write_text(
        yaml.safe_dump(
            {
                "competitions": {
                    "E0": {"competition_id": "E0", "tier": "competition_specific", "league_code": "E0"},
                    "T2": {"competition_id": "T2", "tier": "competition_specific", "league_code": "T2"},
                    "international": {"competition_id": "international", "tier": "general_purpose", "league_code": None},
                }
            }
        ),
        encoding="utf-8",
    )
    assert list_context_keys(registry_path) == ["E0", "T2", "international"]


def test_list_context_keys_collapses_general_purpose_tier_into_one_bucket(tmp_path: Path) -> None:
    """general_purpose competitions intentionally still share a single
    'international' bucket (that tier's models are market-odds-only and
    usable for any competition) -- only competition_specific gets per-id
    buckets. Confirms this story didn't change that collapsing behavior."""
    registry_path = tmp_path / "competitions.yaml"
    registry_path.write_text(
        yaml.safe_dump(
            {
                "competitions": {
                    "E0": {"competition_id": "E0", "tier": "competition_specific", "league_code": "E0"},
                    "international": {"competition_id": "international", "tier": "general_purpose", "league_code": None},
                    "CL": {"competition_id": "CL", "tier": "general_purpose", "league_code": None},
                }
            }
        ),
        encoding="utf-8",
    )
    assert list_context_keys(registry_path) == ["E0", "international"]


# ---------------------------------------------------------------------------
# ForecastService end-to-end: two competition_specific competitions
# ---------------------------------------------------------------------------


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


def _write_model(config_path: Path, context: str, target: str, feature_names: list[str], constant: float) -> None:
    """Write a dummy model + its .metadata.json, and register it under the
    given model_selection.yaml context bucket -- merging with whatever is
    already there so a test can register more than one context."""
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    model_dir = Path(config["paths"]["model_dir"])

    X = pd.DataFrame({name: [1.0, 2.0, 3.0] for name in feature_names})
    model = DummyRegressor(strategy="constant", constant=constant).fit(X, [1, 2, 3])
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
    existing: dict = {}
    if selection_path.exists():
        existing = yaml.safe_load(selection_path.read_text(encoding="utf-8")) or {}
    existing.setdefault("contexts", {}).setdefault(context, {})[target] = entry
    selection_path.write_text(yaml.safe_dump(existing), encoding="utf-8")


def test_forecast_upcoming_second_competition_specific_competition_does_not_collide_with_e0(tmp_path: Path) -> None:
    """Registering and training a second competition_specific competition
    (T2, standing in for Sweden's future SWE) must not collide with E0's
    model_selection.yaml entry -- each forecast must load its own model."""
    config_path = _write_config(tmp_path, schema_features=["MKT_IMPLIED_HOME"])
    (config_path.parent / "config" / "competitions.yaml").write_text(
        yaml.safe_dump(
            {
                "competitions": {
                    "E0": {"competition_id": "E0", "tier": "competition_specific", "league_code": "E0"},
                    "T2": {"competition_id": "T2", "tier": "competition_specific", "league_code": "T2"},
                }
            }
        ),
        encoding="utf-8",
    )
    _write_raw_matches(config_path)
    # Distinct constants let us prove which model actually answered.
    _write_model(config_path, context="E0", target="home_goals", feature_names=["MKT_IMPLIED_HOME"], constant=1.5)
    _write_model(config_path, context="T2", target="home_goals", feature_names=["MKT_IMPLIED_HOME"], constant=9.9)

    # Sanity: both buckets landed in the same file, under separate keys.
    selection_path = config_path.parent / "config" / "model_selection.yaml"
    written = yaml.safe_load(selection_path.read_text(encoding="utf-8"))
    assert set(written["contexts"].keys()) == {"E0", "T2"}
    assert written["contexts"]["E0"]["home_goals"]["model_type"] == "DummyRegressorModel"
    assert written["contexts"]["T2"]["home_goals"]["model_type"] == "DummyRegressorModel"

    service = ForecastService(config_path=str(config_path), targets=["home_goals"])

    e0_result = service.forecast_upcoming(
        home_team="Arsenal", away_team="Everton", date="2025-08-24", league="E0",
        odds_h=1.8, odds_d=3.6, odds_a=4.2, match_type="league",
    )
    t2_result = service.forecast_upcoming(
        home_team="Malmo FF", away_team="AIK", date="2025-08-24", league="T2",
        odds_h=2.0, odds_d=3.3, odds_a=3.8, match_type="league",
    )

    assert e0_result["forecast"]["home_goals"]["expected"] == pytest.approx(1.5)
    assert t2_result["forecast"]["home_goals"]["expected"] == pytest.approx(9.9)
    # Re-fetching E0 after loading T2 must still return E0's own model's
    # output -- proves the earlier T2 load didn't clobber E0's entry.
    e0_result_again = service.forecast_upcoming(
        home_team="Arsenal", away_team="Everton", date="2025-08-24", league="E0",
        odds_h=1.8, odds_d=3.6, odds_a=4.2, match_type="league",
    )
    assert e0_result_again["forecast"]["home_goals"]["expected"] == pytest.approx(1.5)


# ---------------------------------------------------------------------------
# ModelSelector: default enumeration + deprecated 'league' alias
# ---------------------------------------------------------------------------


def _make_run(run_id: str, model_type: str, metric_value: float, artifact_filename: str) -> MagicMock:
    run = MagicMock()
    run.info.run_id = run_id
    run.info.artifact_uri = f"file:///mlruns/1/{run_id}/artifacts"
    run.data.metrics = {"test_mae": metric_value}
    run.data.tags = {"model_family": model_type}
    run.data.params = {"artifact_filename": artifact_filename}
    return run


def test_select_best_models_default_context_promotes_every_registered_competition_independently(
    tmp_path: Path,
) -> None:
    """The core select-best-models acceptance criterion: with --context
    omitted, a second competition_specific competition's champion run must be
    written to its own bucket, without disturbing E0's -- not silently
    dropped, and not merged into E0's entry."""
    registry_path = tmp_path / "competitions.yaml"
    registry_path.write_text(
        yaml.safe_dump(
            {
                "competitions": {
                    "E0": {"competition_id": "E0", "tier": "competition_specific", "league_code": "E0"},
                    "T2": {"competition_id": "T2", "tier": "competition_specific", "league_code": "T2"},
                    "international": {"competition_id": "international", "tier": "general_purpose", "league_code": None},
                }
            }
        ),
        encoding="utf-8",
    )

    model_dir = tmp_path / "models"
    model_dir.mkdir()
    e0_artifact = "home_goals_e0_dummy_v1_20260701.joblib"
    t2_artifact = "home_goals_t2_dummy_v1_20260701.joblib"
    (model_dir / e0_artifact).write_bytes(b"fake")
    (model_dir / t2_artifact).write_bytes(b"fake")

    selection_path = tmp_path / "model_selection.yaml"
    selector = ModelSelector(config_path=selection_path, model_dir=model_dir, registry_path=registry_path)
    selector.client = MagicMock()
    selector.client.search_experiments.return_value = [MagicMock(experiment_id="1")]

    def _search_runs(experiment_ids, filter_string):  # noqa: ARG001 - mock signature
        if "sweep_stage = 'final'" in filter_string:
            return []
        if "context = 'E0'" in filter_string:
            return [_make_run("run_e0", "xgboostregressor", 1.0, e0_artifact)]
        if "context = 'T2'" in filter_string:
            return [_make_run("run_t2", "xgboostregressor", 2.0, t2_artifact)]
        return []

    selector.client.search_runs.side_effect = _search_runs

    selector.run(target="home_goals", context=None, dry_run=False, min_improvement=0.005)

    config = selector.load_config()
    assert config["contexts"]["E0"]["home_goals"]["mlflow_run_id"] == "run_e0"
    assert config["contexts"]["T2"]["home_goals"]["mlflow_run_id"] == "run_t2"
    assert config["contexts"]["E0"]["home_goals"]["model_path"] == f"models/{e0_artifact}"
    assert config["contexts"]["T2"]["home_goals"]["model_path"] == f"models/{t2_artifact}"


def test_select_best_models_context_league_is_deprecated_alias_for_e0(tmp_path: Path) -> None:
    """--context league must still resolve unambiguously to E0 (the
    competition it used to mean), not be reinterpreted as 'all
    competition_specific competitions' -- that reinterpretation is the bug
    this story fixes."""
    model_dir = tmp_path / "models"
    model_dir.mkdir()
    artifact_filename = "home_goals_dummy_v1_20260701.joblib"
    (model_dir / artifact_filename).write_bytes(b"fake")

    selection_path = tmp_path / "model_selection.yaml"
    selector = ModelSelector(config_path=selection_path, model_dir=model_dir)
    selector.client = MagicMock()
    selector.client.search_experiments.return_value = [MagicMock(experiment_id="1")]
    selector.client.search_runs.side_effect = [
        [_make_run("run1", "xgboostregressor", 1.0, artifact_filename)],  # optuna
        [],  # final
    ]

    selector.run(target="home_goals", context="league", dry_run=False, min_improvement=0.005)

    config = selector.load_config()
    assert "league" not in config["contexts"]
    assert config["contexts"]["E0"]["home_goals"]["mlflow_run_id"] == "run1"
