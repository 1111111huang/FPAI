"""get_player_rating: the agent tool that replaces A121's web_search-based
FIFA-rating workaround. Scoped automatically to the current match's two
rosters (~40-50 names) -- narrowing the candidate pool this way is what
makes reliable name matching possible at all (design spec, Phase 1
section). Built fresh per match by build_player_rating_tool(), closed over
that match's roster, rather than reading match context from any global
state -- see graph.py's run_agent() for the call site.

Underlying signal is Transfermarkt market_value_eur (not an EA FC-style
ability rating, see the design spec's 2026-09-23 revision note) -- a
market-perception-of-quality/depth proxy, same underlying purpose as the
originally-planned SoFIFA rating.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from langchain_core.tools import tool as tool_decorator

from src.agent.player_matching import match_player_name
from src.utils.logger import get_logger

if TYPE_CHECKING:
    from src.utils.db_manager import DuckDBManager

LOGGER = get_logger(__name__)

# Design spec: rolls over each team's past matchday squads, same trailing
# pool window SQUAD_*/lineup_features.py's compute_frds already uses --
# reused here as the "current roster" definition, not reinvented.
ROSTER_POOL_DAYS = 90


def get_team_roster(db_manager: "DuckDBManager", team_name: str, as_of_date: str) -> list[str]:
    """Distinct player names who've played for team_name -- matched
    directly against raw_player_match_stats.team_name (FotMob's own team
    string for that row, set at ingestion time by fotmob/merge.py) -- within
    ROSTER_POOL_DAYS before as_of_date. Known limitation: if the caller's
    team_name spelling (from match_info, sourced from football-data.org's
    fixtures API) differs from FotMob's own spelling for the same club, this
    returns an empty roster rather than a wrong one -- get_player_rating
    then safely degrades to matched: false, same non-guessing discipline as
    match_player_name itself. Not yet using TeamNameMapper's fuzzy layer
    here; revisit if this proves a real gap in practice."""
    with db_manager.connection(read_only=True) as conn:
        rows = conn.execute(
            """
            SELECT DISTINCT pd.player_name
            FROM raw_player_match_stats s
            JOIN raw_matches m ON s.match_id = m.match_id
            JOIN player_dim pd ON s.player_id = pd.player_id
            WHERE s.team_name = ?
              AND m.date >= CAST(? AS TIMESTAMP) - INTERVAL '90 days'
              AND m.date < CAST(? AS TIMESTAMP)
            """,
            [team_name, as_of_date, as_of_date],
        ).fetchall()
    return [row[0] for row in rows]


def _lookup_market_value(db_manager: "DuckDBManager", player_name: str, roster: list[str], as_of_date: str) -> dict:
    matched_name = match_player_name(player_name, roster)
    if matched_name is None:
        return {"matched": False}

    with db_manager.connection(read_only=True) as conn:
        row = conn.execute(
            """
            SELECT v.market_value_eur
            FROM player_market_values v
            JOIN player_dim pd ON v.fotmob_player_id = pd.player_id
            WHERE pd.player_name = ? AND v.snapshot_date <= ?
            ORDER BY v.snapshot_date DESC
            LIMIT 1
            """,
            [matched_name, as_of_date],
        ).fetchone()
    if row is None:
        return {"matched": False}
    return {"matched": True, "market_value_eur": row[0]}


def build_player_rating_tool(db_manager: "DuckDBManager", home_team: str, away_team: str, as_of_date: str):
    """Returns a fresh get_player_rating tool closed over this match's own
    roster pool (both teams combined) -- call once per match, in
    graph.py's run_agent(), before build_graph()/bind_tools()."""
    roster = get_team_roster(db_manager, home_team, as_of_date) + get_team_roster(db_manager, away_team, as_of_date)

    @tool_decorator
    def get_player_rating(player_name: str) -> dict:
        """Look up a player's Transfermarkt market value (market_value_eur)
        -- a quick, structured quality-gap proxy when weighing whether an
        absent player's replacement is a real downgrade. Only resolves
        names on this match's own two rosters. Returns {"matched": false}
        for any name not confidently found, or with no listed value --
        never guesses."""
        return _lookup_market_value(db_manager, player_name, roster, as_of_date)

    return get_player_rating
