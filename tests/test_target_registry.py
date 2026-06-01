from __future__ import annotations

from pathlib import Path
import sys

import pytest

sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.logic.target_registry import (
    TARGET_REGISTRY,
    get_target_definition,
    list_target_definitions,
)


def test_registry_contains_forecast_targets_with_metrics() -> None:
    expected_targets = {
        "home_win",
        "result_3way",
        "btts",
        "home_goals",
        "away_goals",
        "total_goals",
        "home_corners",
        "away_corners",
        "total_corners",
    }

    assert set(TARGET_REGISTRY) == expected_targets
    assert TARGET_REGISTRY["result_3way"].classes == ("home", "draw", "away")
    assert TARGET_REGISTRY["result_3way"].primary_metric == "log_loss"
    assert TARGET_REGISTRY["total_goals"].primary_metric == "mae"
    assert TARGET_REGISTRY["total_corners"].secondary_metrics == ("rmse",)


def test_get_target_definition_normalizes_aliases() -> None:
    assert get_target_definition("both_teams_to_score").name == "btts"
    assert get_target_definition("3way").name == "result_3way"


def test_list_target_definitions_is_stable() -> None:
    names = [definition.name for definition in list_target_definitions()]

    assert names == sorted(names)


def test_get_target_definition_rejects_unknown_target() -> None:
    with pytest.raises(ValueError, match="Unsupported target"):
        get_target_definition("shots")
