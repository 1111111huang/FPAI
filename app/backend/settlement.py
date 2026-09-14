"""W13: auto-settlement job (D3a). Sourced from an on-demand W05 API call
(FootballDataClient.get_results), never from the DuckDB raw_matches table --
raw_matches is a batch-refreshed table (stale since 2026-05-24) and
structurally unsuitable for near-real-time settlement of a match that just
finished. Only src.agent.market_resolution's pure resolution logic is
reused -- the 'actual' outcome dict is built directly from the live API
result (NormalizedMatch), not a DataFrame row sourced from DuckDB.

W213: results are fetched for *every* football-data.org-covered league
(FOOTBALL_DATA_CODE_BY_LEAGUE -- E0/SP1/I1/D1/F1), not just EPL. Found live:
this had been hardcoded to "PL" since W13, silently predating La
Liga/Serie A/Bundesliga/Ligue 1 (W76/W134) -- a bet logged against any of
those four leagues (fully loggable since W211 widened fixture search) could
never auto-settle, cache or no cache, since results_by_id would simply never
contain its match_id. No `Bet.league` column is needed to know which
competition code a given bet's match_id belongs to: mirrors
recommendation_outcomes.py's resolve_pending_recommendations, which already
loops over this same mapping for the identical reason. One competition's
transient failure (RequestException) doesn't block the others for the same
date, same fault-isolation precedent as that function.

Requests are grouped by date to respect the client's ~10-requests/minute
budget: one get_results() call per distinct bet date per competition, not
one per bet.

W57: a second, optional `sweden_client` (SwedenFixturesClient, The Odds API
-- W55 found football-data.org has no Allsvenskan coverage at all) is
consulted for the same dates. Results from every source are simply merged
into one results_by_id dict per date and a bet resolves against whichever
source actually has its match_id -- match_ids across providers occupy
disjoint id spaces in practice (small numeric strings vs. 32-char hex).
"""

from __future__ import annotations

import requests

from app.backend.bet_tracker import Bet, BetTracker
from app.backend.football_data_client import FootballDataClient, NormalizedMatch
from app.backend.football_data_competition_codes import FOOTBALL_DATA_CODE_BY_LEAGUE
from src.agent.market_resolution import RESOLVABLE_MARKETS, build_actual_outcome, market_correct
from src.utils.logger import get_logger

LOGGER = get_logger(__name__)


def settle_open_bets(
    tracker: BetTracker,
    client: FootballDataClient,
    sweden_client: object | None = None,
    user_id: int | None = None,
) -> list[Bet]:
    """Attempt to settle every open, scorable-market bet against live
    results. Returns the bets actually settled (won/lost) this call --
    corners bets and not-yet-finished matches are left open, untouched.
    `sweden_client`, when supplied, must expose `get_results(date_from,
    date_to) -> list[NormalizedMatch]` (SwedenFixturesClient satisfies this).
    `user_id` (W210), when supplied, scopes settlement to just that user's
    open bets -- omitted (None) settles every open bet regardless of owner,
    unchanged pre-W210 behavior."""
    resolvable_bets = [b for b in tracker.list_open_bets(user_id=user_id) if b.market in RESOLVABLE_MARKETS]

    bets_by_date: dict[str, list[Bet]] = {}
    for bet in resolvable_bets:
        bets_by_date.setdefault(bet.date, []).append(bet)

    settled: list[Bet] = []
    for date, bets_on_date in bets_by_date.items():
        results: list[NormalizedMatch] = []
        for competition_code in FOOTBALL_DATA_CODE_BY_LEAGUE.values():
            try:
                results += client.get_results(competition_code=competition_code, date_from=date, date_to=date)
            except requests.exceptions.RequestException:
                LOGGER.warning(
                    "settle_open_bets: get_results failed for competition_code=%s date=%s -- "
                    "skipping, other competitions/dates unaffected.", competition_code, date, exc_info=True,
                )
                continue
        if sweden_client is not None:
            results += sweden_client.get_results(date_from=date, date_to=date)
        results_by_id = {match.match_id: match for match in results}
        for bet in bets_on_date:
            match = results_by_id.get(bet.match_id)
            if match is None or match.home_goals is None or match.away_goals is None:
                continue
            actual = build_actual_outcome(match.home_goals, match.away_goals)
            correct = market_correct({"market": bet.market, "selection": bet.selection}, actual)
            if correct is None:
                continue
            settled.append(tracker.settle_bet(bet.id, outcome="won" if correct else "lost"))
    return settled
