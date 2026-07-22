from __future__ import annotations

from pathlib import Path
import sys

import pytest
import yaml

sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.logic.competition_registry import (
    GENERAL_PURPOSE_FEATURES,
    get_competition_definition,
    is_target_available,
    list_competition_definitions,
    resolve_feature_subset_for_tier,
)
from src.models.model_manager import ModelManager
from src.logic.target_registry import get_target_definition


def _selected_features_for(competition_id: str, target_name: str = "result_3way") -> list[str]:
    """Resolve a competition's real, registry-gated feature list via ModelManager,
    against the real config/competitions.yaml (not a synthetic fixture)."""
    manager = ModelManager.__new__(ModelManager)
    manager.config_path = Path("config.yaml")
    manager.feature_subset = None
    manager.target_definition = get_target_definition(target_name)
    manager.competition_id = competition_id
    return manager._load_selected_features()


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
    # US#128: SWE joined the registry alongside E0/international.
    assert set(names) == {"E0", "SWE", "international"}


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


# ---------------------------------------------------------------------------
# US#128: Sweden's real registration, verified against the real
# config/competitions.yaml -- not a synthetic/fictional fixture, unlike the
# tests in test_target_availability.py and test_feature_group_gating.py that
# proved the underlying mechanisms (US#127/US#129) before Sweden was
# actually registered.
# ---------------------------------------------------------------------------

def test_sweden_is_registered_competition_specific() -> None:
    definition = get_competition_definition("SWE")
    assert definition.tier == "competition_specific"
    assert definition.league_code == "SWE"
    assert definition.player_data_sources == ()


def test_sweden_enabled_feature_groups_match_us127_findings() -> None:
    definition = get_competition_definition("SWE")
    assert set(definition.enabled_feature_groups) == {
        "OFF_GOALS", "DEF_GOALS", "OPP_ADJ_GOALS", "STRENGTH_GOALS",
        "INTERACTION_GOALS", "CTX", "MKT", "EFFICIENCY",
    }
    # Explicitly excluded: shot/corner sub-tags, DIS, and SQUAD.
    excluded = {
        "OFF_SHOTS", "OFF_CORNERS", "DEF_SHOTS", "DEF_CORNERS",
        "OPP_ADJ_SHOTS", "OPP_ADJ_CORNERS", "STRENGTH_SHOTS",
        "INTERACTION_SHOTS", "DIS", "CTX_CORNERS", "H2H_CORNERS", "SQUAD",
    }
    assert excluded.isdisjoint(definition.enabled_feature_groups)


def test_sweden_resolved_feature_count_is_74() -> None:
    # US#127's own end-to-end test already proves this against a synthetic
    # fixture; this re-proves it against the real, now-registered entry.
    features = _selected_features_for("SWE")
    assert len(features) == 74


def test_sweden_feature_set_is_superset_of_general_purpose_features() -> None:
    # US#89/US#128: re-verify the feature-superset invariant holds for
    # Sweden's actual reduced enabled_feature_groups list, not just for the
    # full/unfiltered schema.yaml list test_general_purpose_features_are_
    # subset_of_full_schema already covers. Holds trivially today because
    # MKT_* (all 13 GENERAL_PURPOSE_FEATURES) is ungated by the split-family
    # mechanism -- resolve_feature_group_tag() returns None for every MKT_*
    # feature, so _passes_group_gate() always keeps it regardless of which
    # tags are in enabled_feature_groups -- but this test protects that
    # invariant explicitly rather than relying on it being true by accident.
    features = set(_selected_features_for("SWE"))
    assert set(GENERAL_PURPOSE_FEATURES) <= features


def test_e0_still_resolves_full_167_features_after_sweden_registration() -> None:
    # Regression: registering a second competition_specific competition must
    # not change E0's own resolved feature set.
    assert len(_selected_features_for("E0")) == 167


def test_sweden_available_targets_exclude_corners() -> None:
    definition = get_competition_definition("SWE")
    for target in ("result_3way", "btts", "home_goals", "away_goals", "total_goals"):
        assert is_target_available(definition, target) is True
    for target in ("home_corners", "away_corners", "total_corners"):
        assert is_target_available(definition, target) is False


def test_e0_available_targets_unrestricted() -> None:
    # E0 doesn't set available_targets, so every target (including corners)
    # remains available -- registering Sweden must not restrict E0.
    definition = get_competition_definition("E0")
    assert definition.available_targets is None
    for target in ("home_corners", "away_corners", "total_corners", "result_3way"):
        assert is_target_available(definition, target) is True
