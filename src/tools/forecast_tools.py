"""Forecast tools for agent/MCP consumption (US#82)."""

from __future__ import annotations

from typing import Any

from src.forecast import ForecastService


def forecast_matches(
    league: str | None = None,
    match_ids: list[str] | None = None,
    targets: list[str] | None = None,
    limit: int | None = None,
) -> list[dict[str, Any]]:
    """Return forecast payloads for historical feature-store matches.

    Args:
        league: Optional league code filter (e.g. 'E0').
        match_ids: Optional list of specific match IDs to forecast.
        targets: Optional forecast target subset.
        limit: Optional maximum number of matches to return.

    Returns:
        List of forecast payload dicts. Empty list if no matches found.
    """
    service = ForecastService(targets=targets)
    return service.forecast(match_ids=match_ids, league=league, limit=limit)


def forecast_upcoming(
    home_team: str,
    away_team: str,
    date: str,
    league: str | None = None,
    odds_h: float | None = None,
    odds_d: float | None = None,
    odds_a: float | None = None,
    match_type: str = "league",
    over25_odds: float | None = None,
    ah_line: float | None = None,
    ah_home_odds: float | None = None,
    ah_away_odds: float | None = None,
    targets: list[str] | None = None,
) -> dict[str, Any]:
    """Produce an on-demand forecast for a named upcoming match.

    match_type='league': Full rolling feature computation via team history.
        Requires --league and odds.
    match_type='international': MKT_* features only from odds; team name lookup skipped.
        --league is optional.

    Returns:
        Forecast payload dict including 'data_quality' field describing the prediction basis.
    """
    if odds_h is None or odds_d is None or odds_a is None:
        raise ValueError("odds_h, odds_d, and odds_a are required for spot inference.")
    if match_type == "league" and not league:
        raise ValueError("league is required for match_type='league'.")

    service = ForecastService(targets=targets)
    return service.forecast_upcoming(
        home_team=home_team,
        away_team=away_team,
        date=date,
        league=league or "",
        odds_h=float(odds_h),
        odds_d=float(odds_d),
        odds_a=float(odds_a),
        match_type=match_type,
        over25_odds=over25_odds,
        ah_line=ah_line,
        ah_home_odds=ah_home_odds,
        ah_away_odds=ah_away_odds,
    )
