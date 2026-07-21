"""Tests for goals/shots/corners feature-group gating (US#133, extended US#127).

Covers:
- `resolve_feature_group_tag` classification of the split families (OFF,
  DEF, OPP_ADJ, STRENGTH, INTERACTION), the newly-gated families added by
  US#127 (DIS, CTX_CORNERS, H2H_CORNERS), and the families still "not gated
  here" (the rest of CTX, MKT, EFFICIENCY, the rest of H2H, SQUAD-managed
  prefixes).
- The E0 regression: with the new sub-tag `enabled_feature_groups` entry,
  E0 must still resolve to the exact same 167-feature set as
  `config/schema.yaml`'s `selected_features` list (unchanged from before
  this story, when the group tags were effectively decorative).
- A hypothetical "goals-only" competition (standing in for Sweden's
  Allsvenskan, whose football-data.co.uk "New Leagues" source has goals and
  1X2 market odds only) can express "OFF minus shot-based/corner-based"
  through the registry instead of pulling in wholesale-missing columns.
- US#127: the actual Sweden `enabled_feature_groups` list (goals-only,
  no cards, no corners) correctly excludes DIS_*, CTX_*_CORNERS_STD_R5, and
  H2H_CORNERS_R5, closing the residual gap US#133 flagged.
"""

from __future__ import annotations

from pathlib import Path
import sys
from unittest.mock import patch

import pytest
import yaml

sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.logic.competition_registry import CompetitionDefinition, get_competition_definition
from src.logic.feature_groups import resolve_feature_group_tag
from src.models.base_model import XGBoostRegressorModel
from src.models.model_manager import ModelManager

SCHEMA_PATH = Path(__file__).resolve().parents[1] / "config" / "schema.yaml"


def _full_schema_features() -> list[str]:
    with SCHEMA_PATH.open("r", encoding="utf-8") as handle:
        schema = yaml.safe_load(handle)
    return list(schema["training_setup"]["selected_features"])


# --- Unit tests: resolve_feature_group_tag classification -----------------


@pytest.mark.parametrize(
    "feature_name, expected_tag",
    [
        # OFF: goals vs. shots vs. corners
        ("OFF_HOME_FTHG_R3", "OFF_GOALS"),
        ("OFF_HOME_FTHG_EMA5", "OFF_GOALS"),
        ("OFF_HOME_XG_R3", "OFF_GOALS"),
        ("OFF_HOME_LUCK_R3", "OFF_GOALS"),
        ("OFF_HOME_HS_R3", "OFF_SHOTS"),
        ("OFF_HOME_HST_R3", "OFF_SHOTS"),
        ("OFF_HOME_HST_EMA5", "OFF_SHOTS"),
        ("OFF_HOME_SHOT_ACCURACY_R3", "OFF_SHOTS"),
        ("OFF_AWAY_AST_EMA5", "OFF_SHOTS"),
        ("OFF_HOME_HC_R3", "OFF_CORNERS"),
        ("OFF_AWAY_AC_R5", "OFF_CORNERS"),
        # DEF: goals vs. shots vs. corners
        ("DEF_HOME_FTAG_R3", "DEF_GOALS"),
        ("DEF_HOME_XGA_R3", "DEF_GOALS"),
        ("DEF_HOME_AS_R3", "DEF_SHOTS"),
        ("DEF_HOME_AST_R3", "DEF_SHOTS"),
        ("DEF_HOME_SAVE_RATE_R3", "DEF_SHOTS"),
        ("DEF_HOME_AC_R3", "DEF_CORNERS"),
        ("DEF_AWAY_HC_R5", "DEF_CORNERS"),
        # DEF_ANCHOR_* is SQUAD-managed, NOT the DEF family, despite the prefix collision.
        ("DEF_ANCHOR_HOME", None),
        ("DEF_ANCHOR_AWAY", None),
        # OPP_ADJ: goals vs. shots (SOT only) vs. corners
        ("OPP_ADJ_HOME_GOALS_SCORED_R3", "OPP_ADJ_GOALS"),
        ("OPP_ADJ_AWAY_GOALS_CONCEDED_R5", "OPP_ADJ_GOALS"),
        ("OPP_ADJ_GOAL_MATCHUP_HOME_R3", "OPP_ADJ_GOALS"),
        ("OPP_ADJ_HOME_SOT_SCORED_R3", "OPP_ADJ_SHOTS"),
        ("OPP_ADJ_HOME_CORNERS_SCORED_R3", "OPP_ADJ_CORNERS"),
        ("OPP_ADJ_AWAY_CORNERS_CONCEDED_R5", "OPP_ADJ_CORNERS"),
        ("OPP_ADJ_CORNER_MATCHUP_AWAY_R3", "OPP_ADJ_CORNERS"),
        # STRENGTH: mostly goals, one SOT feature
        ("STRENGTH_Goal_Diff", "STRENGTH_GOALS"),
        ("STRENGTH_SoT_Diff", "STRENGTH_SHOTS"),
        # INTERACTION: mostly goals, one SOT feature
        ("INTERACTION_ATTACK_GOALS_DIFF_R5", "INTERACTION_GOALS"),
        ("INTERACTION_DEFENSE_GOALS_DIFF_R5", "INTERACTION_GOALS"),
        ("INTERACTION_ATTACK_SOT_DIFF_R5", "INTERACTION_SHOTS"),
        # DIS (cards): US#127 made this a real, enforced tag (previously
        # claimed as "already has its own group tag" by US#133's docstring,
        # but no code actually checked for it).
        ("DIS_HOME_HY_R3", "DIS"),
        ("DIS_HOME_DISCIPLINE_SCORE_R3", "DIS"),
        # CTX: only the two corners-dependent (hc/ac) features are gated;
        # the rest of the family (rest days, cum pts, PPG, goals/conceded
        # std, streaks) is goals/date-only and stays ungated.
        ("CTX_HOME_REST_DAYS", None),
        ("CTX_HOME_CORNERS_STD_R5", "CTX_CORNERS"),
        ("CTX_AWAY_CORNERS_STD_R5", "CTX_CORNERS"),
        ("CTX_HOME_GOALS_STD_R5", None),
        ("CTX_HOME_SCORE_STREAK", None),
        ("MKT_IMPLIED_HOME", None),
        ("EFFICIENCY_HOME_ATTACK_VS_AWAY_DEF_R5", None),
        ("EFFICIENCY_ATTACK_MATCHUP_DIFF_R5", None),
        # H2H: only the corners-dependent (hc/ac) feature is gated; the
        # goals-only H2H features stay ungated.
        ("H2H_TOTAL_GOALS_R5", None),
        ("H2H_HOME_WIN_RATE_R5", None),
        ("H2H_CORNERS_R5", "H2H_CORNERS"),
        ("SQUAD_HOME_XG_MEAN_R3", None),
        ("LUCK_HOME_BURNOUT_R5", None),
        ("XOC_HOME", None),
        ("FRDS_HOME", None),
    ],
)
def test_resolve_feature_group_tag(feature_name: str, expected_tag: str | None) -> None:
    assert resolve_feature_group_tag(feature_name) == expected_tag


# --- Regression: E0 must still resolve to the exact same 167 features -----


def test_e0_enabled_feature_groups_resolves_to_full_schema_feature_set() -> None:
    """Acceptance criterion: E0's registry entry, after the US#133 sub-tag
    split, must resolve to the identical feature set it resolved to before
    (when OFF/DEF/OPP_ADJ/STRENGTH/INTERACTION tags were effectively
    decorative and every schema feature flowed through unfiltered)."""
    definition = get_competition_definition("E0")
    full_features = _full_schema_features()

    kept = [
        f
        for f in full_features
        if (tag := resolve_feature_group_tag(f)) is None or tag in definition.enabled_feature_groups
    ]
    assert kept == full_features
    assert len(kept) == 167


def test_e0_via_model_manager_resolves_to_full_schema_feature_set() -> None:
    """Same regression, exercised end-to-end through ModelManager (the
    actual consumer of enabled_feature_groups)."""
    manager = ModelManager(
        model=XGBoostRegressorModel(),
        config_path="config.yaml",
        target_config={"target": "home_goals"},
        competition_id="E0",
    )
    features = manager._load_selected_features()
    full_features = _full_schema_features()
    assert features == full_features
    assert len(features) == 167


# --- New capability: a goals-only competition (Sweden stand-in) -----------


def _sweden_like_definition() -> CompetitionDefinition:
    return CompetitionDefinition(
        competition_id="SWE_TEST",
        tier="competition_specific",
        league_code="SWE_TEST",
        enabled_feature_groups=(
            "OFF_GOALS",
            "DEF_GOALS",
            "OPP_ADJ_GOALS",
            "STRENGTH_GOALS",
            "INTERACTION_GOALS",
            "DIS",
            "CTX",
            "MKT",
            "EFFICIENCY",
            # deliberately no *_SHOTS, *_CORNERS, or SQUAD
        ),
        player_data_sources=(),
    )


def _make_manager_with_patched_registry(tmp_path: Path, competition_id: str) -> ModelManager:
    db_path = tmp_path / "test.db"
    config_path = tmp_path / "config.yaml"
    config_path.write_text(yaml.safe_dump({"paths": {"database_path": str(db_path)}}), encoding="utf-8")
    schema_path = tmp_path / "config" / "schema.yaml"
    schema_path.parent.mkdir(parents=True, exist_ok=True)
    schema_path.write_text(
        yaml.safe_dump({"training_setup": {"selected_features": _full_schema_features()}}),
        encoding="utf-8",
    )
    return ModelManager(
        model=XGBoostRegressorModel(),
        config_path=str(config_path),
        target_config={"target": "home_goals"},
        competition_id=competition_id,
    )


def test_goals_only_competition_excludes_shot_and_corner_features(tmp_path: Path) -> None:
    with patch(
        "src.logic.competition_registry.get_competition_definition",
        return_value=_sweden_like_definition(),
    ):
        manager = _make_manager_with_patched_registry(tmp_path, "SWE_TEST")
        features = set(manager._load_selected_features())

    # Shot/SOT-dependent features (needs hs/as/hst/ast) must be excluded.
    for shot_feature in [
        "OFF_HOME_HS_R3",
        "DEF_HOME_AS_R3",
        "OFF_HOME_HST_R3",
        "DEF_HOME_AST_R3",
        "OFF_HOME_SHOT_ACCURACY_R3",
        "DEF_HOME_SAVE_RATE_R3",
        "STRENGTH_SoT_Diff",
        "INTERACTION_ATTACK_SOT_DIFF_R5",
        "OPP_ADJ_HOME_SOT_SCORED_R3",
    ]:
        assert shot_feature not in features, f"{shot_feature} should be excluded (shots-dependent)"

    # Corner-dependent features (needs hc/ac) must be excluded.
    for corner_feature in [
        "OFF_HOME_HC_R3",
        "DEF_HOME_AC_R3",
        "OPP_ADJ_HOME_CORNERS_SCORED_R3",
        "OPP_ADJ_CORNER_MATCHUP_HOME_R3",
    ]:
        assert corner_feature not in features, f"{corner_feature} should be excluded (corners-dependent)"

    # Goals-only features must remain.
    for goals_feature in [
        "OFF_HOME_FTHG_R3",
        "DEF_HOME_FTAG_R3",
        "CTX_HOME_REST_DAYS",
        "MKT_IMPLIED_HOME",
        "STRENGTH_Goal_Diff",
        "INTERACTION_ATTACK_GOALS_DIFF_R5",
        "OPP_ADJ_HOME_GOALS_SCORED_R3",
        "EFFICIENCY_HOME_ATTACK_VS_AWAY_DEF_R5",
    ]:
        assert goals_feature in features, f"{goals_feature} should be included (goals-only)"

    # SQUAD_* (and friends) still excluded via the pre-existing US#97 mechanism.
    assert not any(f.startswith("SQUAD_") for f in features)

    # DIS (cards) is enabled in this fixture -- passes through since "DIS" is present.
    assert "DIS_HOME_HY_R3" in features

    # This fixture enables "CTX"/"MKT" but not "CTX_CORNERS"/"H2H_CORNERS" --
    # US#127: the two corners-dependent CTX features and H2H_CORNERS_R5 must
    # now be excluded (previously they passed through ungated regardless).
    assert "CTX_HOME_CORNERS_STD_R5" not in features
    assert "CTX_AWAY_CORNERS_STD_R5" not in features
    assert "H2H_CORNERS_R5" not in features
    # The rest of CTX/H2H (goals/date-only) still passes through unchanged.
    assert "CTX_HOME_GOALS_STD_R5" in features
    assert "CTX_HOME_SCORE_STREAK" in features
    assert "H2H_TOTAL_GOALS_R5" in features
    assert "H2H_HOME_WIN_RATE_R5" in features


# --- US#127: the actual Sweden `enabled_feature_groups` list -----------


def _sweden_definition() -> CompetitionDefinition:
    """The final, concrete `enabled_feature_groups` list US#127 recommends
    for Sweden's real `config/competitions.yaml` entry (registered by
    US#128, the next story). Unlike `_sweden_like_definition()` above (a
    generic "goals-only" mechanism stand-in from US#133), this fixture is
    Sweden's actual proposed list: goals-only sub-tags, no cards (DIS), no
    corners (CTX_CORNERS/H2H_CORNERS), no SQUAD (no FotMob player data
    source), but CTX/MKT/EFFICIENCY still enabled since most of their
    features are goals/date/1X2-odds-only and computable from Sweden's
    football-data.co.uk "New Leagues" source.
    """
    return CompetitionDefinition(
        competition_id="SWE",
        tier="competition_specific",
        league_code="SWE",
        enabled_feature_groups=(
            "OFF_GOALS",
            "DEF_GOALS",
            "OPP_ADJ_GOALS",
            "STRENGTH_GOALS",
            "INTERACTION_GOALS",
            "CTX",
            "MKT",
            "EFFICIENCY",
            # deliberately excluded: OFF_SHOTS/DEF_SHOTS/OPP_ADJ_SHOTS/
            # STRENGTH_SHOTS/INTERACTION_SHOTS/OFF_CORNERS/DEF_CORNERS/
            # OPP_ADJ_CORNERS (no shots/SOT/corners data), DIS (no cards
            # data), CTX_CORNERS/H2H_CORNERS (no corners data), SQUAD (no
            # FotMob player data source for Sweden).
        ),
        player_data_sources=(),
    )


def test_sweden_enabled_feature_groups_excludes_all_uncomputable_features(tmp_path: Path) -> None:
    """US#127 acceptance: Sweden's real proposed list excludes every feature
    family Sweden's goals + closing-1X2-odds-only source can't populate --
    shots/SOT, corners (including the CTX/H2H corners gap US#133 flagged),
    cards, and squad-level data -- while keeping every goals/odds-based
    feature intact."""
    with patch(
        "src.logic.competition_registry.get_competition_definition",
        return_value=_sweden_definition(),
    ):
        manager = _make_manager_with_patched_registry(tmp_path, "SWE")
        features = set(manager._load_selected_features())

    uncomputable = [
        # Shots/SOT
        "OFF_HOME_HS_R3", "OFF_HOME_HST_R3", "DEF_HOME_AS_R3", "DEF_HOME_AST_R3",
        "OFF_HOME_SHOT_ACCURACY_R3", "DEF_HOME_SAVE_RATE_R3",
        "STRENGTH_SoT_Diff", "INTERACTION_ATTACK_SOT_DIFF_R5",
        "OPP_ADJ_HOME_SOT_SCORED_R3",
        # Corners
        "OFF_HOME_HC_R3", "DEF_HOME_AC_R3",
        "OPP_ADJ_HOME_CORNERS_SCORED_R3", "OPP_ADJ_CORNER_MATCHUP_HOME_R3",
        "CTX_HOME_CORNERS_STD_R5", "CTX_AWAY_CORNERS_STD_R5", "H2H_CORNERS_R5",
        # Cards
        "DIS_HOME_HY_R3", "DIS_AWAY_DISCIPLINE_SCORE_R5",
        # Squad (FotMob)
        "SQUAD_HOME_XG_MEAN_R3", "LUCK_HOME_BURNOUT_R5", "XOC_HOME", "FRDS_HOME",
        "DEF_ANCHOR_HOME",
    ]
    for feature in uncomputable:
        assert feature not in features, f"{feature} should be excluded for Sweden"

    computable = [
        "OFF_HOME_FTHG_R3", "DEF_HOME_FTAG_R3", "OFF_HOME_XG_R3", "OFF_HOME_LUCK_R3",
        "OPP_ADJ_HOME_GOALS_SCORED_R3", "OPP_ADJ_GOAL_MATCHUP_HOME_R3",
        "STRENGTH_Goal_Diff", "INTERACTION_ATTACK_GOALS_DIFF_R5",
        "EFFICIENCY_HOME_ATTACK_VS_AWAY_DEF_R5", "EFFICIENCY_ATTACK_MATCHUP_DIFF_R5",
        "CTX_HOME_REST_DAYS", "CTX_HOME_CUM_PTS", "CTX_HOME_PPG_L10",
        "CTX_HOME_GOALS_STD_R5", "CTX_HOME_SCORE_STREAK",
        "H2H_TOTAL_GOALS_R5", "H2H_HOME_WIN_RATE_R5",
        "MKT_IMPLIED_HOME", "MKT_OVERROUND",
        # Known accepted gap (documented, not excluded): these MKT_* features
        # depend on over25/AH odds Sweden's source never populates, so they
        # will always be NaN for Sweden -- but MKT_* is excluded from
        # cold-start mean-imputation by design (NaN = genuine missing), so
        # this is not a misleading-value bug like the CTX/H2H corners gap
        # was, and is intentionally left unsplit (see feature_groups.py).
        "MKT_AH_LINE", "MKT_IMPLIED_OVER25", "MKT_LAMBDA_TOTAL",
    ]
    for feature in computable:
        assert feature in features, f"{feature} should be included for Sweden"
