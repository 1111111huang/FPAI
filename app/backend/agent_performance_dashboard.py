"""W171: top/bottom staked-bet examples for the agent performance
dashboard, enriched with match date/competition/team names. The only piece
of this feature needing RecommendationCache (DB I/O) -- recommendation_stats.py
stays pure aggregation on purpose, mirroring bet_stats.py's own separation
from bet_tracker.py."""

from __future__ import annotations

from typing import Any

from app.backend.recommendation_cache import RecommendationCache
from app.backend.recommendation_outcomes import RecommendationOutcome
from app.backend.recommendation_stats import compute_recommendation_stats
from src.agent.schema import reported_teams


def _enrich_bet(
    bet: dict[str, Any],
    outcomes_by_match: dict[str, RecommendationOutcome],
    cache: RecommendationCache,
) -> dict[str, Any]:
    """Attaches date/competition (from the resolved outcome itself, always
    available) and home_team/away_team (from the cached recommendation's
    own self-reported match field, via reported_teams() -- the same
    helper already shared for this exact home/home_team key-spelling
    ambiguity elsewhere in this codebase, BUG-023/024). A cache miss (the
    recommendation was purged, or a genuine race) degrades team names to
    None rather than failing the whole dashboard -- same "never let one
    bad row break the page" discipline validate_and_degrade already uses."""
    outcome = outcomes_by_match.get(bet["match_id"])
    date = outcome.date if outcome else None
    competition = outcome.competition if outcome else None
    home_team: str | None = None
    away_team: str | None = None
    if date is not None:
        entry = cache.get_latest_any_config(bet["match_id"], date)
        if entry is not None:
            teams = reported_teams(entry.recommendation.get("match") or {})
            if teams is not None:
                home_team, away_team = teams
    return {**bet, "date": date, "competition": competition, "home_team": home_team, "away_team": away_team}


def compute_agent_performance_dashboard(
    outcomes: list[RecommendationOutcome], cache: RecommendationCache, top_n: int = 5
) -> dict[str, Any]:
    """Everything compute_recommendation_stats already returns, plus
    top_winners/top_losers: the top_n highest- and lowest-payout entries
    from staked_bets, enriched with match context. Only these <=2*top_n
    rows get the (relatively) expensive cache lookup -- not every staked
    bet, however many there are."""
    stats = compute_recommendation_stats(outcomes)
    staked_bets = stats["staked_bets"]
    outcomes_by_match = {o.match_id: o for o in outcomes}

    winners = sorted((b for b in staked_bets if b["payout"] > 0), key=lambda b: b["payout"], reverse=True)[:top_n]
    losers = sorted((b for b in staked_bets if b["payout"] < 0), key=lambda b: b["payout"])[:top_n]

    return {
        **stats,
        "top_winners": [_enrich_bet(b, outcomes_by_match, cache) for b in winners],
        "top_losers": [_enrich_bet(b, outcomes_by_match, cache) for b in losers],
    }
