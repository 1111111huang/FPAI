"""W28: sandbox odds source -- real historical 1X2 odds, not synthetic. The
Odds API (W07) is a live-current-odds-only service with no historical
replay -- a past sandbox date needs a different real source for odds.
raw_matches already carries real odds_h/odds_d/odds_a (football-data.co.uk),
covering 2016-08-13 through the table's last refresh.

Implements the exact same get_odds() -> list[NormalizedOdds] | None shape
OddsAPIClient does, so eod_batch.py/t30_refresh.py need zero changes to
consume it -- same interface, different backing data. Team names come
through in raw_matches's own canonical form, so BUG-015's existing
TeamNameMapper-based matching in odds_lookup()/match_odds() resolves them
unmodified. Deliberately excludes odds movement -- a single closing-line
snapshot per match, not a time series (see W32)."""

from __future__ import annotations

from app.backend.eod_batch import LEAGUE_CODE
from app.backend.odds_api_client import NormalizedOdds
from app.backend.odds_sport_keys import ODDS_SPORT_KEY_BY_COMPETITION
from src.utils.db_manager import DuckDBManager

# BUG-034: sport_key -> league reverse lookup, so this client actually scopes
# its query to the requested competition instead of always querying E0
# regardless of what sport_key was passed in.
_LEAGUE_BY_SPORT_KEY: dict[str, str] = {v: k for k, v in ODDS_SPORT_KEY_BY_COMPETITION.items()}


class HistoricalOddsClient:
    """Sandbox-mode replacement for OddsAPIClient: serves real historical
    1X2 odds for a given sandbox_date from raw_matches instead of calling
    the live Odds API. get_odds() returns None when no odds are recorded
    for that date (e.g. no fixtures, or the date predates raw_matches's
    coverage)."""

    def __init__(self, sandbox_date: str, db_manager: DuckDBManager | None = None) -> None:
        self._sandbox_date = sandbox_date
        self._db_manager = db_manager or DuckDBManager()

    def get_odds(self, sport_key: str = "soccer_epl", date: str | None = None) -> list[NormalizedOdds] | None:
        # BUG-034: sport_key was previously accepted but silently ignored --
        # this always queried league=E0 regardless of what was requested,
        # a latent bug since Sweden's addition (never caught because SWE's
        # *live* odds fetching goes through OddsAPIClient, not this
        # sandbox-only client -- only a sandbox-mode SWE/SP1 request ever
        # exercised this path) and newly, concretely wrong for La Liga
        # (confirmed live: a SANDBOX_MODE=1 request for a real SP1 fixture's
        # odds returned a mismatched E0 match instead). Falls back to
        # LEAGUE_CODE ("E0") for an unrecognized sport_key, preserving the
        # exact prior default behavior for the default/no-sport_key case.
        #
        # W54: `date`, when given, overrides the instance's own sandbox_date
        # for this call only -- lets one client serve odds for several
        # different fixture dates (a sandbox fallback-window batch), not
        # just the single date it was constructed with.
        league = _LEAGUE_BY_SPORT_KEY.get(sport_key, LEAGUE_CODE)
        query_date = date or self._sandbox_date
        with self._db_manager.connection(read_only=True) as conn:
            # odds_h/odds_d/odds_a are FLOAT (32-bit) in raw_matches, so a
            # value stored as 1.80 round-trips as 1.7999999523162842 without
            # rounding -- football-data.co.uk odds are always quoted to 2dp,
            # so rounding here is lossless and keeps output matching the
            # source values exactly.
            rows = conn.execute(
                """
                SELECT home_team, away_team, date,
                       ROUND(CAST(odds_h AS DOUBLE), 2),
                       ROUND(CAST(odds_d AS DOUBLE), 2),
                       ROUND(CAST(odds_a AS DOUBLE), 2)
                FROM raw_matches
                WHERE league = ? AND CAST(date AS DATE) = CAST(? AS DATE)
                """,
                (league, query_date),
            ).fetchall()

        if not rows:
            return None

        return [
            NormalizedOdds(
                home_team=home_team,
                away_team=away_team,
                commence_time=match_date.isoformat() if hasattr(match_date, "isoformat") else str(match_date),
                home_odds=odds_h,
                draw_odds=odds_d,
                away_odds=odds_a,
            )
            for home_team, away_team, match_date, odds_h, odds_d, odds_a in rows
        ]
