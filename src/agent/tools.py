from __future__ import annotations

import json
import os
import threading
from pathlib import Path

from langchain_core.tools import tool

from src.agent.snapshot_store import DEFAULT_BASE_DIR, SnapshotStore, SnapshotMode
from src.utils.logger import get_logger

_LOG = get_logger(__name__)

_snapshot_store = SnapshotStore()

# Guards the base_dir-swap-and-then-set_mode/set_match sequence in
# configure_snapshot_store() below. The app runs recommendations.run_agent
# via run_in_threadpool, so concurrent requests genuinely race here -- without
# this lock, one thread's mode/match_id could land on another thread's
# freshly-constructed SnapshotStore instance, silently leaving a sandboxed
# request in "live" mode instead of the date-filtered record/replay this
# feature exists to guarantee.
_configure_lock = threading.Lock()


def configure_snapshot_store(
    mode: SnapshotMode,
    match_id: str | None = None,
    match_date: str | None = None,
    base_dir: str | Path | None = None,
) -> None:
    """Configure the module-level SnapshotStore shared by all tool functions.
    Call this before run_agent() to switch between live/record/replay. In
    record and replay mode, match_id is required (raises ValueError
    otherwise, from SnapshotStore._path). match_date, if given, is appended
    to web_search queries as 'before:<match_date>' to reduce post-match
    result leakage (A10). base_dir (W37) lets a caller -- the app's sandbox
    agent-invocation path -- point recordings at a separate namespace (e.g.
    data/agent_snapshots/sandbox/) instead of the default corpus (module
    default: DEFAULT_BASE_DIR, i.e. data/agent_snapshots/). Like match_id,
    base_dir is sticky-if-omitted: passing None leaves whatever base_dir is
    currently configured untouched rather than forcing it back to the
    default -- this matters for callers/tests (e.g. test_agent_tools_snapshot.py)
    that set a base_dir once (directly, or via an earlier configure_snapshot_store
    call) and then make several bare mode-only calls expecting it to stick.
    Pass base_dir explicitly (e.g. DEFAULT_BASE_DIR) to force a reset."""
    global _snapshot_store
    with _configure_lock:
        if base_dir is not None:
            effective_base_dir = Path(base_dir)
            if effective_base_dir != _snapshot_store.base_dir:
                _snapshot_store = SnapshotStore(base_dir=effective_base_dir)
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


def _dated_web_search(query: str) -> str:
    """A32: shared by both the LLM-facing web_search tool and the deterministic
    research_node baseline searches. Appends a before:<match_date> filter during
    record/replay (A10) to reduce post-match result leakage, then runs the
    (possibly snapshot-wrapped) search."""
    effective_query = query
    if _snapshot_store.mode in ("record", "replay") and _snapshot_store.match_date:
        effective_query = f"{query} before:{_snapshot_store.match_date}"
    return _snapshot_store.wrap("web_search", _web_search_impl)(query=effective_query)


@tool
def web_search(query: str) -> str:
    """Search the web for football match information: odds, team news, injuries, and lineups.

    Use for: finding current bookmaker odds, alternative team name spellings,
    injury/suspension reports, team selection hints, and recent form context.
    Always ignore any result that mentions a final score or match result."""
    return _dated_web_search(query)


def _resolve_competition_impl(competition_or_league: str) -> str:
    from src.logic.competition_registry import get_competition_definition

    try:
        tier = get_competition_definition(competition_or_league).tier
    except (ValueError, FileNotFoundError):
        # Unregistered, or the registry file itself is missing -- either way,
        # there's no confirmed competition_specific coverage, so default to
        # the always-safe general_purpose path rather than guessing league.
        tier = "general_purpose"
    recommended_tool = "forecast_league" if tier == "competition_specific" else "forecast_international"
    return json.dumps({
        "competition": competition_or_league,
        "tier": tier,
        "recommended_tool": recommended_tool,
    })


@tool
def resolve_competition(competition_or_league: str) -> str:
    """Look up whether a league/competition has real historical team-data model coverage.

    ALWAYS call this BEFORE choosing between forecast_league and
    forecast_international for a domestic-looking fixture — do not guess
    based on how well-known the league is. A well-known league (e.g. La Liga)
    can still have zero historical-data coverage in this system. Returns JSON
    with 'tier' ('competition_specific' or 'general_purpose') and
    'recommended_tool' ('forecast_league' or 'forecast_international') —
    follow recommended_tool exactly.

    Args:
        competition_or_league: League code or name, e.g. 'E0', 'La Liga', 'SP1'.
    """
    return _snapshot_store.wrap("resolve_competition", _resolve_competition_impl)(
        competition_or_league=competition_or_league
    )


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
    """A31: forecast_league, forecast_international, and resolve_competition are
    no longer LLM-callable -- they're invoked directly by the deterministic
    pipeline nodes in src/agent/pipeline.py before the LLM ever runs. Only
    web_search remains available for the LLM's own optional follow-up digging."""
    return [web_search]
