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

from src.logic.target_registry import TARGET_REGISTRY, normalize_target_name

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
    # US#129: explicit allow-list of forecast targets this competition's data
    # source can actually train (keyed off TARGET_REGISTRY names). None means
    # "no restriction" -- every registered target is assumed trainable, which
    # preserves today's behavior for competitions (e.g. E0) that don't set
    # this key. Set it when a competition's source is missing raw columns a
    # target's labels depend on (e.g. Sweden's football-data.co.uk "New
    # Leagues" CSV has no hc/ac corners columns -- see US#125's
    # _NULL_COLUMNS -- so home_corners/away_corners/total_corners must be
    # excluded or every training row gets dropna()'d out, the same failure
    # shape as BUG-001).
    available_targets: tuple[str, ...] | None = None
    # display_enabled: whether this competition is currently surfaced to
    # users (fixtures endpoint, pregeneration, nightly EOD/T-30 batch) and
    # spent compute/API quota on. Defaults to True (unset key = shown),
    # matching available_targets' "no key = no restriction" convention.
    # Training/backtesting/CLI tooling is unaffected -- this only gates the
    # live app's "which competitions are we actively running for users"
    # switch, e.g. flipping SWE off once the big-5 leagues' season starts.
    display_enabled: bool = True


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
        league_code = entry.get("league_code")
        # US#140: exactly one canonical casing per league code (e.g. always
        # "E0", always "SWE") must hold across raw_matches.league, this
        # registry's league_code, model_selection.yaml context keys, and the
        # CLI --league flag -- this is the one place all of those trace back
        # to, so it's the natural spot to catch a mismatch early rather than
        # letting it silently reach match_id hashing or forecast lookups.
        if league_code is not None and league_code != league_code.upper():
            raise ValueError(
                f"Competition '{competition_id}' has league_code '{league_code}' "
                "that is not canonically uppercase. Use one consistent casing "
                "(e.g. 'E0', 'SWE') everywhere: raw_matches.league, "
                "competitions.yaml's league_code, model_selection.yaml context "
                "keys, and the CLI --league flag."
            )
        raw_available_targets = entry.get("available_targets")
        available_targets: tuple[str, ...] | None = None
        if raw_available_targets is not None:
            normalized_targets = tuple(normalize_target_name(name) for name in raw_available_targets)
            unknown = sorted(set(normalized_targets) - set(TARGET_REGISTRY))
            if unknown:
                valid = ", ".join(sorted(TARGET_REGISTRY))
                raise ValueError(
                    f"Competition '{competition_id}' has unknown available_targets {unknown}. "
                    f"Valid targets: {valid}."
                )
            available_targets = normalized_targets

        registry[competition_id] = CompetitionDefinition(
            competition_id=competition_id,
            tier=tier,
            league_code=league_code,
            enabled_feature_groups=tuple(entry.get("enabled_feature_groups") or ()),
            player_data_sources=tuple(entry.get("player_data_sources") or ()),
            available_targets=available_targets,
            display_enabled=bool(entry.get("display_enabled", True)),
        )
    return registry


# A50: a few registered competitions have a well-known free-text name that
# differs from their own registry code (e.g. "La Liga" -> "SP1") --
# resolve_competition's own docstring tells the calling LLM "La Liga" is a
# valid example input, but a raw dict-key lookup never matched it, even
# after SP1 was fully registered and trained (the gap this phase's scoping
# note found). Tried only as a case-insensitive fallback after the exact
# competition_id lookup misses, so passing the real code always wins and
# this can never mask a genuine unregistered/typo'd code. Placed at this one
# choke point (not duplicated per caller) since every real caller --
# resolve_competition (src/agent/tools.py) and
# ForecastService.forecast_upcoming (src/forecast/forecast_service.py) --
# already routes its own tier/competition lookup through this function.
COMPETITION_NAME_ALIASES: dict[str, str] = {
    "la liga": "SP1",
    "premier league": "E0",
    "english premier league": "E0",
    "epl": "E0",
    "allsvenskan": "SWE",
}


def get_competition_definition(
    competition_id: str, registry_path: str | Path = DEFAULT_REGISTRY_PATH
) -> CompetitionDefinition:
    """Return the competition definition or raise a helpful error."""
    registry = _load_registry(registry_path)
    if competition_id in registry:
        return registry[competition_id]
    alias = COMPETITION_NAME_ALIASES.get(competition_id.strip().lower())
    if alias is not None and alias in registry:
        return registry[alias]
    valid = ", ".join(sorted(registry))
    raise ValueError(f"Unknown competition '{competition_id}'. Registered competitions: {valid}")


def list_competition_definitions(
    registry_path: str | Path = DEFAULT_REGISTRY_PATH,
) -> list[CompetitionDefinition]:
    """Return registered competition definitions in stable competition_id order."""
    registry = _load_registry(registry_path)
    return [registry[name] for name in sorted(registry)]


def list_display_enabled_competition_ids(
    registry_path: str | Path = DEFAULT_REGISTRY_PATH,
) -> list[str]:
    """Return competition_ids with display_enabled=True (the default), in
    stable order. Single source of truth for "which competitions are we
    currently showing/running for users" -- the live app's fixtures
    endpoint and nightly EOD/T-30 scheduler both filter through this
    instead of each hardcoding its own on/off list, so toggling a
    competition in config/competitions.yaml is the one edit needed."""
    return [d.competition_id for d in list_competition_definitions(registry_path) if d.display_enabled]


def resolve_feature_subset_for_tier(tier: str) -> list[str] | None:
    """Return the feature subset for a tier, or None to use the full schema.yaml list."""
    if tier == "general_purpose":
        return list(GENERAL_PURPOSE_FEATURES)
    if tier == "competition_specific":
        return None
    raise ValueError(f"Unknown tier '{tier}'. Must be one of {VALID_TIERS}.")


def is_target_available(competition_def: CompetitionDefinition, target_name: str) -> bool:
    """Return whether `target_name` is trainable for this competition (US#129).

    `available_targets` unset (None) means "no restriction" -- every target
    in TARGET_REGISTRY is assumed available, which is today's behavior for
    every competition that doesn't declare the key (e.g. E0). When it is
    set, only targets explicitly listed are available.
    """
    if competition_def.available_targets is None:
        return True
    return normalize_target_name(target_name) in competition_def.available_targets


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
