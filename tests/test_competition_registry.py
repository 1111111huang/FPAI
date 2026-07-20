from __future__ import annotations

from pathlib import Path
import sys

import pytest
import yaml

sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.logic.competition_registry import (
    GENERAL_PURPOSE_FEATURES,
    get_competition_definition,
    list_competition_definitions,
    resolve_feature_subset_for_tier,
)


def test_get_competition_definition_for_e0() -> None:
    definition = get_competition_definition("E0")
    assert definition.competition_id == "E0"
    assert definition.tier == "competition_specific"
    assert definition.league_code == "E0"
    assert "MKT" in definition.enabled_feature_groups


def test_get_competition_definition_for_international() -> None:
    definition = get_competition_definition("international")
    assert definition.tier == "general_purpose"
    assert definition.league_code is None


def test_get_competition_definition_rejects_unknown_competition() -> None:
    with pytest.raises(ValueError, match="Unknown competition"):
        get_competition_definition("nonexistent")


def test_list_competition_definitions_is_stable() -> None:
    names = [definition.competition_id for definition in list_competition_definitions()]
    assert names == sorted(names)
    assert set(names) == {"E0", "international"}


def test_resolve_feature_subset_for_general_purpose_matches_legacy_mkt_features() -> None:
    # Regression guard: this must equal the old MKT_FEATURES constant that
    # used to live in main.py, so behavior is unchanged after the refactor.
    assert resolve_feature_subset_for_tier("general_purpose") == [
        "MKT_IMPLIED_HOME",
        "MKT_IMPLIED_DRAW",
        "MKT_IMPLIED_AWAY",
        "MKT_OVERROUND",
        "MKT_LAMBDA_TOTAL",
        "MKT_LAMBDA_HOME",
        "MKT_LAMBDA_AWAY",
        "MKT_POISSON_BTTS_PROB",
        "MKT_LAMBDA_AH_DIFF",
        "MKT_AH_LINE",
        "MKT_AH_HOME_ODDS",
        "MKT_AH_AWAY_ODDS",
        "MKT_IMPLIED_OVER25",
    ]


def test_resolve_feature_subset_for_competition_specific_is_none() -> None:
    # None tells ModelManager to use the full schema.yaml selected_features list.
    assert resolve_feature_subset_for_tier("competition_specific") is None


def test_resolve_feature_subset_for_unknown_tier_raises() -> None:
    with pytest.raises(ValueError, match="Unknown tier"):
        resolve_feature_subset_for_tier("not_a_tier")


def test_invalid_tier_in_yaml_raises(tmp_path: Path) -> None:
    bad_registry = tmp_path / "bad_competitions.yaml"
    bad_registry.write_text(
        yaml.dump({"competitions": {"X0": {"competition_id": "X0", "tier": "made_up_tier"}}})
    )
    with pytest.raises(ValueError, match="invalid tier"):
        get_competition_definition("X0", registry_path=bad_registry)


def test_lowercase_league_code_in_yaml_raises(tmp_path: Path) -> None:
    # US#140: exactly one canonical casing per league code must be enforced
    # here, since this registry is the shared source of truth that
    # raw_matches.league, model_selection.yaml context keys, and the CLI
    # --league flag are all expected to line up with.
    bad_registry = tmp_path / "bad_league_code.yaml"
    bad_registry.write_text(
        yaml.dump(
            {"competitions": {"SWE": {"competition_id": "SWE", "tier": "general_purpose", "league_code": "swe"}}}
        )
    )
    with pytest.raises(ValueError, match="not canonically uppercase"):
        get_competition_definition("SWE", registry_path=bad_registry)


def test_null_league_code_in_yaml_is_allowed(tmp_path: Path) -> None:
    # "international" legitimately has no single league_code -- must not be
    # rejected by the casing check.
    registry = tmp_path / "ok_competitions.yaml"
    registry.write_text(
        yaml.dump(
            {"competitions": {"international": {"competition_id": "international", "tier": "general_purpose", "league_code": None}}}
        )
    )
    definition = get_competition_definition("international", registry_path=registry)
    assert definition.league_code is None


def test_general_purpose_features_are_subset_of_full_schema() -> None:
    # US#89: enforce the feature-superset invariant. competition_specific
    # resolves to "all of config/schema.yaml's selected_features" (feature_subset=None),
    # so general_purpose's feature list must be fully contained within it.
    schema_path = Path(__file__).resolve().parents[1] / "config" / "schema.yaml"
    with schema_path.open("r", encoding="utf-8") as handle:
        schema = yaml.safe_load(handle)
    full_feature_set = set(schema["training_setup"]["selected_features"])
    assert set(GENERAL_PURPOSE_FEATURES) <= full_feature_set


def test_squad_is_in_e0_enabled_feature_groups_after_phase14c() -> None:
    """E0 must declare SQUAD once competitions.yaml is updated for Phase 14c."""
    definition = get_competition_definition("E0")
    assert "SQUAD" in definition.enabled_feature_groups
