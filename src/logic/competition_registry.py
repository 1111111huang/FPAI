"""Central registry for competitions and their model-tier configuration.

Two tiers exist today: "general_purpose" (market-odds-only, usable for any
competition) and "competition_specific" (full team-form feature set). A
competition_specific feature list must always be a superset of
general_purpose's (see Phase 14, FRAI_TECHSPEC.md Section 27.2). If a future
tier needs an architecture where a literal feature superset doesn't apply,
the design reserves room for a competition_specific model to instead consume
the general_purpose model's own prediction as an input feature (stacking).
That stacking path is not implemented here (US#90) — this module only
resolves which feature subset a tier uses today.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import yaml

GENERAL_PURPOSE_FEATURES: list[str] = [
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

VALID_TIERS = ("general_purpose", "competition_specific")

DEFAULT_REGISTRY_PATH = Path("config/competitions.yaml")


@dataclass(frozen=True)
class CompetitionDefinition:
    """Definition of a registered competition's model tier and feature scope."""

    competition_id: str
    tier: str
    league_code: str | None
    enabled_feature_groups: tuple[str, ...]
    player_data_sources: tuple[str, ...] = ()


def _load_registry(registry_path: str | Path = DEFAULT_REGISTRY_PATH) -> dict[str, CompetitionDefinition]:
    path = Path(registry_path)
    if not path.exists():
        raise FileNotFoundError(f"Missing competition registry: {path}")
    with path.open("r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle) or {}
    competitions = raw.get("competitions", {})
    if not competitions:
        raise ValueError(f"No competitions defined in {path}")

    registry: dict[str, CompetitionDefinition] = {}
    for competition_id, entry in competitions.items():
        tier = entry.get("tier")
        if tier not in VALID_TIERS:
            raise ValueError(
                f"Competition '{competition_id}' has invalid tier '{tier}'. Must be one of {VALID_TIERS}."
            )
        registry[competition_id] = CompetitionDefinition(
            competition_id=competition_id,
            tier=tier,
            league_code=entry.get("league_code"),
            enabled_feature_groups=tuple(entry.get("enabled_feature_groups") or ()),
            player_data_sources=tuple(entry.get("player_data_sources") or ()),
        )
    return registry


def get_competition_definition(
    competition_id: str, registry_path: str | Path = DEFAULT_REGISTRY_PATH
) -> CompetitionDefinition:
    """Return the competition definition or raise a helpful error."""
    registry = _load_registry(registry_path)
    try:
        return registry[competition_id]
    except KeyError as exc:
        valid = ", ".join(sorted(registry))
        raise ValueError(f"Unknown competition '{competition_id}'. Registered competitions: {valid}") from exc


def list_competition_definitions(
    registry_path: str | Path = DEFAULT_REGISTRY_PATH,
) -> list[CompetitionDefinition]:
    """Return registered competition definitions in stable competition_id order."""
    registry = _load_registry(registry_path)
    return [registry[name] for name in sorted(registry)]


def resolve_feature_subset_for_tier(tier: str) -> list[str] | None:
    """Return the feature subset for a tier, or None to use the full schema.yaml list."""
    if tier == "general_purpose":
        return list(GENERAL_PURPOSE_FEATURES)
    if tier == "competition_specific":
        return None
    raise ValueError(f"Unknown tier '{tier}'. Must be one of {VALID_TIERS}.")


def list_context_keys(registry_path: str | Path = DEFAULT_REGISTRY_PATH) -> list[str]:
    """Return the model_selection.yaml `contexts` bucket keys implied by the registry (US#110).

    Every `competition_specific` competition gets its own bucket keyed by its
    `competition_id` (e.g. "E0", "SWE"), so two competition_specific competitions
    never collide over the same model_selection.yaml entry. All `general_purpose`
    competitions still share a single "international" bucket, since that tier's
    models are market-odds-only and usable for any competition — that collapsing
    is intentional and predates this story (see forecast_upcoming's
    effective_context resolution in src/forecast/forecast_service.py).
    """
    definitions = list_competition_definitions(registry_path)
    competition_specific_ids = sorted(
        definition.competition_id for definition in definitions if definition.tier == "competition_specific"
    )
    return [*competition_specific_ids, "international"]
