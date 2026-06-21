from __future__ import annotations

import json
import os

from langchain_core.tools import tool

from src.agent.snapshot_store import SnapshotStore, SnapshotMode
from src.utils.logger import get_logger

_LOG = get_logger(__name__)

_snapshot_store = SnapshotStore()


def configure_snapshot_store(mode: SnapshotMode, match_id: str | None = None, match_date: str | None = None) -> None:
    """Configure the module-level SnapshotStore shared by all tool functions.

    Call this before run_agent() to switch between live/record/replay. In record
    and replay mode, match_id is required (raises ValueError otherwise, from
    SnapshotStore._path). match_date, if given, is appended to web_search queries
    as 'before:<match_date>' to reduce post-match result leakage (A10).
    """
    _snapshot_store.set_mode(mode)
    if match_id is not None:
        _snapshot_store.set_match(match_id, match_date)


def get_snapshot_store() -> SnapshotStore:
    return _snapshot_store


def _web_search_impl(query: str) -> str:
    from tavily import TavilyClient

    api_key = os.environ.get("TAVILY_API_KEY")
    if not api_key:
        return (
            "TOOL_PERMANENTLY_UNAVAILABLE: web_search has no API key configured. "
            "Do NOT call web_search again — it will always return this message. "
            "Output your final JSON recommendation now using only the forecast data already retrieved."
        )

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
def web_search(query: str) -> str:
    """Search the web for football match information: odds, team news, injuries, and lineups.

    Use for: finding current bookmaker odds, alternative team name spellings,
    injury/suspension reports, team selection hints, and recent form context.
    Always ignore any result that mentions a final score or match result."""
    effective_query = query
    if _snapshot_store.mode in ("record", "replay") and _snapshot_store.match_date:
        effective_query = f"{query} before:{_snapshot_store.match_date}"
    return _snapshot_store.wrap("web_search", _web_search_impl)(query=effective_query)


def _forecast_league_impl(
    home_team: str,
    away_team: str,
    date: str,
    league: str,
    odds_h: float,
    odds_d: float,
    odds_a: float,
) -> str:
    try:
        from src.forecast.forecast_service import ForecastService
        svc = ForecastService()
        try:
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
        except FileNotFoundError:
            # League-context models not yet trained — use international (market-odds-only) path
            _LOG.info("forecast_league | league_models_absent | falling_back_to_international | home=%s away=%s", home_team, away_team)
            result = svc.forecast_upcoming(
                home_team=home_team,
                away_team=away_team,
                date=date,
                league=league,
                odds_h=odds_h,
                odds_d=odds_d,
                odds_a=odds_a,
                match_type="international",
            )
            result.setdefault("data_quality", {})["prediction_basis"] = "market_odds_only_league_fallback"
            return json.dumps(result, default=str)
    except Exception as exc:
        return json.dumps({"error": str(exc), "status": "tool_error"})


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
    return _snapshot_store.wrap("forecast_league", _forecast_league_impl)(
        home_team=home_team, away_team=away_team, date=date, league=league,
        odds_h=odds_h, odds_d=odds_d, odds_a=odds_a,
    )


def _forecast_international_impl(
    home_team: str,
    away_team: str,
    date: str,
    odds_h: float,
    odds_d: float,
    odds_a: float,
) -> str:
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
    return _snapshot_store.wrap("forecast_international", _forecast_international_impl)(
        home_team=home_team, away_team=away_team, date=date,
        odds_h=odds_h, odds_d=odds_d, odds_a=odds_a,
    )


def get_default_tools() -> list:
    return [web_search, forecast_league, forecast_international]
