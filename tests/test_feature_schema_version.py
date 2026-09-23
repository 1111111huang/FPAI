"""US#206: feature_schema_version must reflect a real hash of the feature
set actually used for a training run, not the static "v1" literal every
model (regardless of feature set) used to record -- and check_feature_schema_drift()
must detect when a promoted model's own recorded feature list contains a
name no longer present in the live master schema (a column renamed/removed
since training), the same class BUG-070's own dead-column gap is an
instance of."""

from __future__ import annotations

from pathlib import Path
import sys

import pandas as pd
import pytest
import yaml

sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.models.model_manager import ModelManager, check_feature_schema_drift
from src.models.base_model import RandomForestRegressorModel


def _make_manager(tmp_path: Path) -> ModelManager:
    db_path = tmp_path / "test.db"
    config_path = tmp_path / "config.yaml"
    config_path.write_text(yaml.safe_dump({"paths": {"database_path": str(db_path)}}), encoding="utf-8")
    manager = ModelManager(
        model=RandomForestRegressorModel(), config_path=str(config_path), target_config={"target": "total_goals"},
    )
    # _build_artifact_metadata calls self.model.predict(X_val) for a
    # regression target's prediction_interval -- needs a fitted model,
    # unrelated to what this file actually tests (feature_schema_version).
    manager.model.train(pd.DataFrame({"A": [1.0, 2.0], "B": [3.0, 4.0], "C": [5.0, 6.0]}), pd.Series([1.0, 2.0]))
    return manager


def test_feature_schema_version_is_not_the_old_hardcoded_literal(tmp_path: Path) -> None:
    manager = _make_manager(tmp_path)
    metadata = manager._build_artifact_metadata(
        model_path=tmp_path / "model.joblib", feature_names=["A", "B", "C"],
        X_val=pd.DataFrame({"A": [1.0], "B": [2.0], "C": [3.0]}), y_val=pd.Series([1.0]),
    )
    assert metadata["feature_schema_version"] != "v1"


def test_feature_schema_version_is_deterministic_for_the_same_feature_set(tmp_path: Path) -> None:
    manager = _make_manager(tmp_path)
    # feature_names is a plain list-of-names input to the hash, independent
    # of X_val's actual columns (which just need to match what the model
    # was trained/fitted on, for the unrelated prediction_interval step) --
    # kept fixed across both calls here.
    kwargs = dict(
        model_path=tmp_path / "model.joblib",
        X_val=pd.DataFrame({"A": [1.0], "B": [2.0], "C": [3.0]}), y_val=pd.Series([1.0]),
    )
    v1 = manager._build_artifact_metadata(feature_names=["A", "B"], **kwargs)["feature_schema_version"]
    v2 = manager._build_artifact_metadata(feature_names=["B", "A"], **kwargs)["feature_schema_version"]
    assert v1 == v2  # order-independent -- same set, same hash


def test_feature_schema_version_differs_for_a_genuinely_different_feature_set(tmp_path: Path) -> None:
    manager = _make_manager(tmp_path)
    kwargs = dict(
        model_path=tmp_path / "model.joblib",
        X_val=pd.DataFrame({"A": [1.0], "B": [2.0], "C": [3.0]}), y_val=pd.Series([1.0]),
    )
    v1 = manager._build_artifact_metadata(feature_names=["A", "B"], **kwargs)["feature_schema_version"]
    v2 = manager._build_artifact_metadata(feature_names=["A", "B", "C"], **kwargs)["feature_schema_version"]
    assert v1 != v2


def _write_schema(tmp_path: Path, selected_features: list[str]) -> str:
    path = tmp_path / "schema.yaml"
    path.write_text(yaml.safe_dump({"training_setup": {"selected_features": selected_features}}), encoding="utf-8")
    return str(path)


def test_check_feature_schema_drift_flags_a_recorded_name_no_longer_in_the_master_schema(tmp_path: Path) -> None:
    schema_path = _write_schema(tmp_path, ["OFF_HOME_FTHG_R5", "DEF_HOME_FTAG_R5"])
    drifted = check_feature_schema_drift(["OFF_HOME_FTHG_R5", "OFF_HOME_XG_R5"], schema_path=schema_path)
    assert drifted == ["OFF_HOME_XG_R5"]


def test_check_feature_schema_drift_empty_when_every_recorded_name_still_resolves(tmp_path: Path) -> None:
    schema_path = _write_schema(tmp_path, ["OFF_HOME_FTHG_R5", "DEF_HOME_FTAG_R5"])
    drifted = check_feature_schema_drift(["OFF_HOME_FTHG_R5"], schema_path=schema_path)
    assert drifted == []
