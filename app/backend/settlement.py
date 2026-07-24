"""W13: auto-settlement job (D3a). Sourced from an on-demand W05 API call
(FootballDataClient.get_results), never from the DuckDB raw_matches table --
raw_matches is a batch-refreshed table (stale since 2026-05-24) and
structurally unsuitable for near-real-time settlement of a match that just
finished. Only src.agent.market_resolution's pure resolution logic is
reused -- the 'actual' outcome dict is built directly from the live API
result (NormalizedMatch), not a DataFrame row sourced from DuckDB.

Every EPL bet the app tracks is for the E0/"PL" competition (see
match_info.COMPETITION_ALLOWLIST) -- results are always fetched for that
competition code. Requests are grouped by date to respect the client's
~10-requests/minute budget: one get_results() call per distinct bet date,
not one per bet.

W57: a second, optional `sweden_client` (SwedenFixturesClient, The Odds API
-- W55 found football-data.org has no Allsvenskan coverage at all) is
consulted for the same dates. No `Bet.league` column is needed to know
which client a given bet's match_id belongs to -- football-data.org's
match_ids (small numeric strings) and The Odds API's event ids (32-char hex
strings) occupy disjoint id spaces in practice, so results from both
sources are simply merged into one results_by_id dict per date and a bet
resolves against whichever source actually has its match_id.
"""

from __future__ import annotations

from app.backend.bet_tracker import Bet, BetTracker
from app.backend.football_data_client import FootballDataClient, NormalizedMatch
from src.agent.market_resolution import RESOLVABLE_MARKETS, build_actual_outcome, market_correct

COMPETITION_CODE = "PL"


def settle_open_bets(
    tracker: BetTracker,
    client: FootballDataClient,
    sweden_client: object | None = None,
) -> list[Bet]:
    """Attempt to settle every open, scorable-market bet against live
    results. Returns the bets actually settled (won/lost) this call --
    corners bets and not-yet-finished matches are left open, untouched.
    `sweden_client`, when supplied, must expose `get_results(date_from,
    date_to) -> list[NormalizedMatch]` (SwedenFixturesClient satisfies this)."""
    resolvable_bets = [b for b in tracker.list_open_bets() if b.market in RESOLVABLE_MARKETS]

    bets_by_date: dict[str, list[Bet]] = {}
    for bet in resolvable_bets:
        bets_by_date.setdefault(bet.date, []).append(bet)

    settled: list[Bet] = []
    for date, bets_on_date in bets_by_date.items():
        results: list[NormalizedMatch] = list(
            client.get_results(competition_code=COMPETITION_CODE, date_from=date, date_to=date)
        )
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
