"""W13: auto-settlement job (D3a). Sourced from an on-demand W05 API call
(FootballDataClient.get_results), never from the DuckDB raw_matches table --
raw_matches is a batch-refreshed table (stale since 2026-05-24) and
structurally unsuitable for near-real-time settlement of a match that just
finished. Only src.agent.market_resolution's pure resolution logic is
reused -- the 'actual' outcome dict is built directly from the live API
result (NormalizedMatch), not a DataFrame row sourced from DuckDB.

Every bet the app tracks is for the E0/"PL" competition (see
match_info.COMPETITION_ALLOWLIST) -- results are always fetched for that
competition code. Requests are grouped by date to respect the client's
~10-requests/minute budget: one get_results() call per distinct bet date,
not one per bet.
"""

from __future__ import annotations

from app.backend.bet_tracker import Bet, BetTracker
from app.backend.football_data_client import FootballDataClient
from src.agent.market_resolution import RESOLVABLE_MARKETS, build_actual_outcome, market_correct

COMPETITION_CODE = "PL"


def settle_open_bets(tracker: BetTracker, client: FootballDataClient) -> list[Bet]:
    """Attempt to settle every open, scorable-market bet against live
    results. Returns the bets actually settled (won/lost) this call --
    corners bets and not-yet-finished matches are left open, untouched."""
    resolvable_bets = [b for b in tracker.list_open_bets() if b.market in RESOLVABLE_MARKETS]

    bets_by_date: dict[str, list[Bet]] = {}
    for bet in resolvable_bets:
        bets_by_date.setdefault(bet.date, []).append(bet)

    settled: list[Bet] = []
    for date, bets_on_date in bets_by_date.items():
        results = client.get_results(competition_code=COMPETITION_CODE, date_from=date, date_to=date)
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
