from __future__ import annotations

import json
import os

from langchain_core.tools import tool


@tool
def web_search(query: str) -> str:
    """Search the web for football match information: odds, team news, injuries, and lineups.

    Use for: finding current bookmaker odds, alternative team name spellings,
    injury/suspension reports, team selection hints, and recent form context.
    Always ignore any result that mentions a final score or match result."""
    from tavily import TavilyClient

    api_key = os.environ.get("TAVILY_API_KEY")
    if not api_key:
        return "[web_search unavailable: TAVILY_API_KEY not set in environment]"

    client = TavilyClient(api_key=api_key)
    response = client.search(query=query, max_results=5)
    snippets = []
    for r in response.get("results", []):
        title = r.get("title", "")
        content = r.get("content", "")
        url = r.get("url", "")
        snippets.append(f"[{title}]\n{content}\nSource: {url}")
    return "\n\n---\n\n".join(snippets) if snippets else "No results found."


@tool
def forecast_league(
    home_team: str,
    away_team: str,
    date: str,
    league: str,
    odds_h: float,
    odds_d: float,
    odds_a: float,
) -> str:
    """Get ML probability forecast for a domestic league match.

    Uses full team history + market odds features (114 features).
    Use when both teams play in a known domestic league with historical data.

    Args:
        home_team: Home team name — fuzzy-matched against known teams in the database.
                   Use web_search first to find the correct name variant if unsure.
        away_team: Away team name.
        date: Match date in YYYY-MM-DD format.
        league: League code, e.g. 'E0' (Premier League), 'SP1' (La Liga), 'D1' (Bundesliga).
        odds_h: Home win decimal odds from bookmaker.
        odds_d: Draw decimal odds.
        odds_a: Away win decimal odds.

    Returns JSON with probabilities for result_3way, btts, goals, and corners targets.
    Includes data_quality.prediction_basis to indicate which features were used."""
    try:
        from src.forecast.forecast_service import ForecastService
        svc = ForecastService()
        result = svc.forecast_upcoming(
            home_team=home_team,
            away_team=away_team,
            date=date,
            league=league,
            odds_h=odds_h,
            odds_d=odds_d,
            odds_a=odds_a,
            match_type="league",
        )
        return json.dumps(result, default=str)
    except Exception as exc:
        return json.dumps({"error": str(exc), "status": "tool_error",
                           "hint": "Try forecast_international if league history is unavailable."})


@tool
def forecast_international(
    home_team: str,
    away_team: str,
    date: str,
    odds_h: float,
    odds_d: float,
    odds_a: float,
) -> str:
    """Get ML probability forecast using market odds only — no team history required.

    Use for: international fixtures, cup matches between teams from different leagues,
    or when team historical data is unavailable in the database.

    Args:
        home_team: Home team name.
        away_team: Away team name.
        date: Match date in YYYY-MM-DD format.
        odds_h: Home win decimal odds.
        odds_d: Draw decimal odds.
        odds_a: Away win decimal odds.

    Returns JSON with market-implied probability forecasts.
    data_quality.prediction_basis will be 'market_odds_only'."""
    try:
        from src.forecast.forecast_service import ForecastService
        svc = ForecastService()
        result = svc.forecast_upcoming(
            home_team=home_team,
            away_team=away_team,
            date=date,
            league="",
            odds_h=odds_h,
            odds_d=odds_d,
            odds_a=odds_a,
            match_type="international",
        )
        return json.dumps(result, default=str)
    except Exception as exc:
        return json.dumps({"error": str(exc), "status": "tool_error"})


def get_default_tools() -> list:
    return [web_search, forecast_league, forecast_international]
