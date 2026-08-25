"""Shared market-outcome resolution logic (W13). Used by both
src/agent/backtest.py (historical backtest scoring, DataFrame-row-sourced)
and the app's live settlement job (app/backend/settlement.py, W05-API-result-
sourced) -- extracted so the two never drift out of sync on which markets
can be programmatically resolved, or how.
"""

from __future__ import annotations

from typing import Any

# Markets whose correctness can be programmatically resolved. home_corners/
# away_corners are excluded: MatchRecommendation has no numeric line field
# for them (only current_odds/min_odds), so we cannot tell what threshold
# the agent's "selection" refers to. Accepted, ongoing limitation, not a v1
# gap -- see documents/app_user_stories.md Integration Gaps.
RESOLVABLE_MARKETS = {"result_3way", "btts", "total_goals"}


def market_correct(market_rec: dict[str, Any], actual: dict[str, Any]) -> bool | None:
    """Resolve whether a market recommendation/bet matches the actual outcome.

    Returns True/False for resolvable markets (result_3way, btts, total_goals).
    Returns None -- not False -- for markets with no programmatic resolution
    (e.g. home_corners/away_corners). Callers MUST treat None as "unknown,
    skip" and never coerce it to a loss.
    """
    market = market_rec.get("market")
    if market not in RESOLVABLE_MARKETS:
        return None
    selection = market_rec.get("selection")
    if market == "result_3way":
        return selection == actual["result"]
    if market == "btts":
        return selection == actual["btts"]
    return selection == actual["total_goals_side"]  # market == "total_goals"


def build_actual_outcome(home_goals: int, away_goals: int) -> dict[str, Any]:
    """Build the resolvable-outcome dict shape from plain home/away goal
    counts -- usable from any live-result source (not just a raw_matches
    DataFrame row, which src/agent/backtest.py's load_outcome() sources this
    same shape from)."""
    home_goals, away_goals = int(home_goals), int(away_goals)
    if home_goals > away_goals:
        result = "home"
    elif home_goals < away_goals:
        result = "away"
    else:
        result = "draw"
    total_goals = home_goals + away_goals
    return {
        "fthg": home_goals,
        "ftag": away_goals,
        "result": result,
        "btts": "yes" if (home_goals > 0 and away_goals > 0) else "no",
        "total_goals": total_goals,
        "total_goals_side": "over_2.5" if total_goals > 2 else "under_2.5",
    }


def pick_recommended_market(markets: list[dict[str, Any]]) -> dict[str, Any] | None:
    """Which single market a recommendation actually picked -- ports
    MatchUI.tsx's bestMarket() into Python (A81) so the app's outcome
    resolver (app/backend/recommendation_outcomes.py, W167) can determine
    server-side the same market a completed card's Hit/Not-Hit badge
    already reflects client-side. Prefers a non-'no_bet' market; falls back
    to ranking among all markets (including no_bet) only when nothing is
    actionable at all. Ties broken by value_edge, highest first -- Python's
    max() returns the first maximal element on ties, matching a stable
    descending sort's own tie-break order."""
    if not markets:
        return None
    actionable = [m for m in markets if m.get("recommendation_type") != "no_bet"]
    pool = actionable if actionable else markets
    return max(pool, key=lambda m: m.get("value_edge") or 0.0)
