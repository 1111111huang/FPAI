"""W03: the app decides the 'league' value passed to run_agent() itself,
rather than trusting the agent/engine to route correctly.

Kept as a double-safety layer even after the root-cause fixes land --
US#107 (registry-driven fallback in ForecastService.forecast_upcoming())
and A27 (agent-side resolve_competition tool) address this at the engine
and agent layers respectively, but the app should not assume either has
shipped or stays correct forever. Deliberately a hardcoded allowlist, not a
live read of config/competitions.yaml -- the whole point is that this
judgment is independent of the engine's own registry, so a registry
misconfiguration there doesn't silently cascade into the app's routing too.
"""

from __future__ import annotations

# Sourced from config/competitions.yaml's competition_specific entries as of
# 2026-08-15 (W135) -- update by hand if that registry changes, not automatically.
COMPETITION_ALLOWLIST: frozenset[str] = frozenset({"E0", "SWE", "SP1", "I1", "D1", "F1"})


def gate_league(competition_id: str | None) -> str | None:
    """Return competition_id if it's an allowlisted domestic league the
    engine has real model coverage for, else None -- matching the existing
    CLI convention ('omit --league for international fixtures')."""
    if competition_id in COMPETITION_ALLOWLIST:
        return competition_id
    return None
