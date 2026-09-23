"""Live-vs-backtest ROI comparison (A123).

No such comparison existed before this: agent_performance_dashboard.py only
ever showed live stats on their own, with no backtest baseline alongside
them, so "live ROI is bad" was a qualitative impression, not a measured
number checked against what the same config was expected to do.

Resolves realized live picks from GET /api/fixtures (date-ranged, for real
finished scores) joined against GET /api/recommendations/{match_id} (the
cached pick for that match), scored with the exact same
src.agent.market_resolution logic src/agent/backtest.py uses for its own
market_results -- not the recommendation_outcomes store, which was found
live (2026-09-22, documents/agent_user_stories.md's A123 entry) to be stuck
on a settlement backlog (BUG-058): every resolved row there was dated
2026-09-16 or earlier despite querying days later. /api/fixtures + cached
recommendations has no dependency on that backlog ever being cleared.

Usage:
    python scripts/live_vs_backtest_roi.py \\
        --league E0 --date-from 2026-09-17 --date-to 2026-09-20 \\
        --backtest-report reports/agent_backtest/20260722T101500Z_a1b2c3d4.json

The backtest report is whatever `agent-backtest --league E0 ...` already
saved (config/model version is implied by which report you point at -- this
tool doesn't guess one for you, see its own module docstring on why: no
report currently records its own league).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import httpx

from src.agent.market_resolution import build_actual_outcome, market_correct, resolve_recommendation_pick

DEFAULT_BASE_URL = "https://fpai-production.up.railway.app"
# Arbitrary flat unit -- ROI (profit/staked) is invariant to the stake size
# as long as every bet uses the same one, matching simulate_flat_stake's own
# convention (src/agent/staking.py) so live and backtest ROI are computed by
# the identical formula, not just eyeballed as "similar-looking" numbers.
FLAT_STAKE = 1.0


def fetch_live_bets(base_url: str, league: str, date_from: str, date_to: str) -> list[dict[str, Any]]:
    """One resolved bet dict per finished match with a direct_bet pick and a
    resolvable market outcome. Mirrors src/agent/backtest.py's own
    market_results construction (resolve_recommendation_pick + market_correct)
    exactly, just sourced from the live API instead of a DataFrame row."""
    with httpx.Client(base_url=base_url, timeout=30.0) as client:
        resp = client.get("/api/fixtures", params={"date_from": date_from, "date_to": date_to})
        resp.raise_for_status()
        fixtures = resp.json()

        bets: list[dict[str, Any]] = []
        for match in fixtures:
            if match.get("competition") != league:
                continue
            if match.get("status") != "FINISHED":
                continue
            home_goals, away_goals = match.get("home_goals"), match.get("away_goals")
            if home_goals is None or away_goals is None:
                continue

            match_id = match["match_id"]
            match_date = str(match["utc_date"])[:10]
            rec_resp = client.get(f"/api/recommendations/{match_id}", params={"date": match_date})
            if rec_resp.status_code == 404:
                continue  # nothing was ever generated/cached for this match
            rec_resp.raise_for_status()
            recommendation = rec_resp.json()

            picked = resolve_recommendation_pick(
                recommendation.get("candidates") or [], recommendation.get("recommendation_pick")
            )
            if not picked or picked.get("recommendation_type") != "direct_bet":
                continue
            if picked.get("current_odds") is None:
                continue  # agent marked direct_bet with no odds found -- cannot stake

            actual = build_actual_outcome(home_goals, away_goals)
            correct = market_correct(picked, actual)
            if correct is None:
                continue  # unresolvable market (e.g. home_corners/away_corners) -- skip, never coerce to a loss

            odds = float(picked["current_odds"])
            payout = FLAT_STAKE * (odds - 1) if correct else -FLAT_STAKE
            bets.append({
                "match_id": match_id, "date": match_date,
                "home_team": match.get("home_team"), "away_team": match.get("away_team"),
                "market": picked["market"], "selection": picked["selection"],
                "odds": odds, "stake": FLAT_STAKE, "won": bool(correct), "payout": payout,
            })
        return bets


def summarize(bets: list[dict[str, Any]]) -> dict[str, Any]:
    staked = sum(b["stake"] for b in bets)
    profit = sum(b["payout"] for b in bets)
    wins = sum(1 for b in bets if b["won"])
    return {
        "picks": len(bets),
        "wins": wins,
        "hit_rate": round(wins / len(bets), 4) if bets else None,
        "roi": round(profit / staked, 4) if staked > 0 else None,
    }


def by_market(bets: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    groups: dict[str, list[dict[str, Any]]] = {}
    for bet in bets:
        groups.setdefault(f"{bet['market']}/{bet['selection']}", []).append(bet)
    return {key: summarize(group) for key, group in sorted(groups.items())}


def load_backtest_report(path: str) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def compare(
    live_bets: list[dict[str, Any]], backtest_report: dict[str, Any], market_filter: str | None = None,
) -> dict[str, Any]:
    live_overall = summarize(live_bets)
    live_by_market = by_market(live_bets)
    backtest_overall = {
        "picks": backtest_report.get("bets_placed"),
        "wins": backtest_report.get("bets_won"),
        "hit_rate": backtest_report.get("hit_rate"),
        "roi": backtest_report.get("roi"),
    }
    backtest_by_market = backtest_report.get("market_breakdown", {})

    if market_filter:
        live_by_market = {k: v for k, v in live_by_market.items() if k.startswith(market_filter)}
        backtest_by_market = {k: v for k, v in backtest_by_market.items() if k.startswith(market_filter)}

    return {
        "live": {"overall": live_overall, "by_market": live_by_market},
        "backtest": {"overall": backtest_overall, "by_market": backtest_by_market},
    }


def print_comparison(result: dict[str, Any], league: str, date_from: str, date_to: str) -> None:
    def _row(label: str, live: dict[str, Any], bt: dict[str, Any]) -> str:
        live_roi = "n/a" if live.get("roi") is None else f"{live['roi']:+.1%}"
        bt_roi = "n/a" if bt.get("roi") is None else f"{bt['roi']:+.1%}"
        return (
            f"  {label:<26} live: {live_roi:>8} (n={live.get('picks', 0):<3}, "
            f"hit={live.get('hit_rate')})   backtest: {bt_roi:>8} (n={bt.get('picks', 0):<3}, "
            f"hit={bt.get('hit_rate')})"
        )

    print(f"\nLive vs backtest ROI: {league} {date_from}..{date_to}")
    print("=" * 100)
    print(_row("OVERALL", result["live"]["overall"], result["backtest"]["overall"]))
    print("-" * 100)
    markets = sorted(set(result["live"]["by_market"]) | set(result["backtest"]["by_market"]))
    for market in markets:
        print(_row(
            market,
            result["live"]["by_market"].get(market, {}),
            result["backtest"]["by_market"].get(market, {}),
        ))
    print("=" * 100)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--league", required=True, help="Competition id as /api/fixtures tags it, e.g. E0, SP1.")
    parser.add_argument("--date-from", required=True)
    parser.add_argument("--date-to", required=True)
    parser.add_argument("--backtest-report", required=True, help="Path to a JSON report saved by `agent-backtest` for this league/config version.")
    parser.add_argument("--market", default=None, help="Filter the per-market breakdown to markets starting with this prefix, e.g. 'total_goals'.")
    parser.add_argument("--base-url", default=DEFAULT_BASE_URL)
    parser.add_argument("--json", action="store_true", help="Print the raw comparison dict instead of a table.")
    args = parser.parse_args()

    live_bets = fetch_live_bets(args.base_url, args.league, args.date_from, args.date_to)
    backtest_report = load_backtest_report(args.backtest_report)
    result = compare(live_bets, backtest_report, args.market)

    if args.json:
        print(json.dumps(result, indent=2))
    else:
        print_comparison(result, args.league, args.date_from, args.date_to)


if __name__ == "__main__":
    main()
