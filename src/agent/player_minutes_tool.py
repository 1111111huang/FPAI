"""get_player_minutes (A131): a get_player_rating-sibling tool letting the
LLM check whether a web_search-flagged absence is actually meaningful,
rather than taking the search text's own framing at face value -- the
Al-Dakhil/Puado/Gittens class of error (a genuinely absent-or-irrelevant
player still cited as if it mattered), distinct from A130's grounding fix
(is this player actually on the pitch).

Mirrors player_rating_tool.py's exact shape: built fresh per match by
build_player_minutes_tool(), closed over that match's own two rosters, same
match_player_name fuzzy layer, same fail-closed matched:false discipline.
No baked-in verdict returned -- like get_player_rating's raw market_value_eur,
this hands back raw numbers and leaves "was this absence meaningful" as a
prompt-side judgment call, not a hardcoded rule.

Returns team_matches_available alongside the player's own minutes/
appearances (direct user catch, 2026-10-13): a flat trailing-window minutes
count conflates "this player is fringe" with "the team barely played during
this window at all" (an international break, a postponed fixture, early
season) -- team_matches_available is the player's OWN team's match count
over the identical window, so involvement reads as a share of matches
available, not an absolute number vulnerable to a quiet calendar stretch.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from langchain_core.tools import tool as tool_decorator

from src.agent.player_matching import match_player_name
from src.agent.player_rating_tool import get_team_roster
from src.utils.logger import get_logger

if TYPE_CHECKING:
    from src.utils.db_manager import DuckDBManager

LOGGER = get_logger(__name__)


def _team_matches_available(db_manager: "DuckDBManager", team_name: str, as_of_date: str) -> int:
    """Distinct matches team_name played within ROSTER_POOL_DAYS before
    as_of_date -- same date-window shape as get_team_roster's own query, so
    both tools agree on what "recent" means."""
    with db_manager.connection(read_only=True) as conn:
        row = conn.execute(
            """
            SELECT COUNT(DISTINCT s.match_id)
            FROM raw_player_match_stats s
            JOIN raw_matches m ON s.match_id = m.match_id
            WHERE s.team_name = ?
              AND m.date >= CAST(? AS TIMESTAMP) - INTERVAL '90 days'
              AND m.date < CAST(? AS TIMESTAMP)
            """,
            [team_name, as_of_date, as_of_date],
        ).fetchone()
    return row[0] if row else 0


def _lookup_minutes(
    db_manager: "DuckDBManager", player_name: str, home_roster: list[str], away_roster: list[str],
    home_team: str, away_team: str, as_of_date: str,
) -> dict:
    roster = home_roster + away_roster
    matched_name = match_player_name(player_name, roster)
    if matched_name is None:
        return {"matched": False}

    team_name = home_team if matched_name in home_roster else away_team

    with db_manager.connection(read_only=True) as conn:
        row = conn.execute(
            """
            SELECT COALESCE(SUM(s.minutes_played), 0), COUNT(DISTINCT s.match_id)
            FROM raw_player_match_stats s
            JOIN raw_matches m ON s.match_id = m.match_id
            JOIN player_dim pd ON s.player_id = pd.player_id
            WHERE pd.player_name = ?
              AND m.date >= CAST(? AS TIMESTAMP) - INTERVAL '90 days'
              AND m.date < CAST(? AS TIMESTAMP)
            """,
            [matched_name, as_of_date, as_of_date],
        ).fetchone()
    minutes, appearances = (row[0], row[1]) if row else (0, 0)

    return {
        "matched": True,
        "minutes_last_90_days": int(minutes),
        "appearances_last_90_days": int(appearances),
        "team_matches_available": _team_matches_available(db_manager, team_name, as_of_date),
    }


def build_player_minutes_tool(db_manager: "DuckDBManager", home_team: str, away_team: str, as_of_date: str):
    """Returns a fresh get_player_minutes tool closed over this match's own
    roster pool (both teams, kept separate so a matched player's own team
    -- needed for team_matches_available -- is known) -- call once per
    match, in graph.py's run_agent(), before build_graph()/bind_tools()."""
    home_roster = get_team_roster(db_manager, home_team, as_of_date)
    away_roster = get_team_roster(db_manager, away_team, as_of_date)

    @tool_decorator
    def get_player_minutes(player_name: str) -> dict:
        """Look up a player's real recent involvement -- minutes played and
        appearances over the trailing ~90 days, plus their own team's total
        match count over the same window -- to judge whether a web_search-
        flagged absence is actually meaningful. Judge involvement as a share
        of team_matches_available (e.g. 9/10 appearances is a clear
        regular; 1/10, or matched: false, is not meaningful signal), not an
        absolute minutes number, since the window can span a break. Only
        resolves names on this match's own two rosters. Returns
        {"matched": false} for any name not confidently found -- never
        guesses."""
        return _lookup_minutes(
            db_manager, player_name, home_roster, away_roster, home_team, away_team, as_of_date,
        )

    return get_player_minutes
