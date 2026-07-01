"""Tests for SQUAD_* feature gating in ModelManager._load_selected_features."""

from __future__ import annotations

from pathlib import Path
import sys
from unittest.mock import patch

import pytest
import yaml

sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.models.model_manager import ModelManager
from src.models.base_model import XGBoostRegressorModel


def _make_manager(competition_id: str, tmp_path: Path) -> ModelManager:
    db_path = tmp_path / "test.db"
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        yaml.safe_dump({"paths": {"database_path": str(db_path)}}), encoding="utf-8"
    )
    schema_path = tmp_path / "config" / "schema.yaml"
    schema_path.parent.mkdir(parents=True, exist_ok=True)
    schema_path.write_text(
        yaml.safe_dump({
            "training_setup": {
                "selected_features": [
                    "MKT_IMPLIED_HOME",
                    "MKT_IMPLIED_AWAY",
                    "OFF_HOME_FTHG_R5",
                    "SQUAD_HOME_XG_MEAN_R5",
                    "SQUAD_AWAY_RATING_MEAN_R3",
                ]
            }
        }),
        encoding="utf-8",
    )
    return ModelManager(
        model=XGBoostRegressorModel(),
        config_path=str(config_path),
        target_config={"target": "home_goals"},
        competition_id=competition_id,
    )


def test_competition_specific_with_squad_includes_squad_features(tmp_path: Path) -> None:
    manager = _make_manager("E0", tmp_path)
    features = manager._load_selected_features()
    squad_features = [f for f in features if f.startswith("SQUAD_")]
    assert len(squad_features) == 2  # both SQUAD_* in the schema list are included


def test_competition_without_squad_group_excludes_squad_features(tmp_path: Path) -> None:
    manager = _make_manager("international", tmp_path)
    # international has feature_subset=MKT_FEATURES (passed explicitly), so
    # _load_selected_features hits the feature_subset branch. Test the filtering
    # separately by patching feature_subset to None to exercise the gating branch.
    manager.feature_subset = None
    features = manager._load_selected_features()
    squad_features = [f for f in features if f.startswith("SQUAD_")]
    assert len(squad_features) == 0
