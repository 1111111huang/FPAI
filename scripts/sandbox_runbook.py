#!/usr/bin/env python3
"""W31: sandbox scenario runbook -- repeatable for any historical date.
Boots the app in sandbox mode for a given date and drives the full real
user journey against it: Dashboard fixtures, a real recommendation
generation (real agent/LLM call, real ML model prediction from real
point-in-time features), logging one bet from that recommendation and one
manually, then settling both against the real historical result.

Non-determinism in the agent's own predicted values is expected and
accepted -- this validates the app's plumbing works end-to-end for an
arbitrary day, not that the agent's predicted values are "correct" (a
distinct, separate concern this script does not test).

Usage:
    SANDBOX_MODE=1 SANDBOX_DATE=2026-05-24 python scripts/sandbox_runbook.py

(FOOTBALL_DATA_API_KEY/ODDS_API_KEY/TAVILY_API_KEY are loaded from .env, same
as app/backend/main.py does -- no need to pass them on the command line.)
"""

from __future__ import annotations

import os
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).resolve().parent.parent))

from dotenv import load_dotenv

load_dotenv()

from app.backend import bets, recommendations
from app.backend.agent_config_hash import compute_agent_config_hash
from app.backend.football_data_client import FootballDataClient
from app.backend.historical_odds_client import HistoricalOddsClient
from app.backend.recommendations import RecommendationRequest, validate_and_degrade
from app.backend.sandbox_clock import is_sandbox_mode, sandbox_date
from app.backend.settlement import settle_open_bets
from src.agent.agent_config import AgentConfig


def main() -> None:
    if not is_sandbox_mode() or sandbox_date() is None:
        print("SANDBOX_MODE=1 and SANDBOX_DATE=<real historical date> must both be set.")
        sys.exit(1)

    date_str = sandbox_date().isoformat()
    print(f"=== Sandbox runbook for {date_str} ===")

    # 1. Dashboard: real fixtures for the sandbox date
    fixtures_client = FootballDataClient(api_key=os.environ["FOOTBALL_DATA_API_KEY"])
    fixtures = fixtures_client.get_results(competition_code="PL", date_from=date_str, date_to=date_str)
    if not fixtures:
        print(f"No completed E0 fixtures found for {date_str} -- pick a different date.")
        sys.exit(1)
    fixture = fixtures[0]
    print(f"Fixture: {fixture.home_team} vs {fixture.away_team} ({fixture.home_goals}-{fixture.away_goals})")

    # 2. Real historical odds for that date (same league, from raw_matches)
    odds_client = HistoricalOddsClient(sandbox_date=date_str)
    odds_events = odds_client.get_odds()
    assert odds_events is not None, "HistoricalOddsClient returned no odds for this date"
    print(f"Odds events found: {len(odds_events)}")

    # 3. Generate a real recommendation (real agent/LLM call). league="E0" is
    # passed explicitly, same as eod_batch.py does for real E0 fixtures --
    # gate_league() lets it straight through since E0 is allowlisted.
    # match_id is fixture.match_id (football-data.org's real numeric id), the
    # same value eod_batch.py/t30_refresh.py always key the cache with -- NOT
    # RecommendationRequest.effective_match_id()'s synthetic team-name
    # fallback, which settle_open_bets() below could never match against a
    # real NormalizedMatch.match_id.
    config = AgentConfig.default()
    request = RecommendationRequest(
        home_team=fixture.home_team, away_team=fixture.away_team, date=date_str,
        league="E0", match_id=fixture.match_id,
    )
    raw = recommendations.run_agent(request.to_match_info())
    result = validate_and_degrade(raw)
    print(f"Recommendation overall={result.overall} markets={len(result.markets)}")

    cache = recommendations.get_cache()
    cache.record_generation(
        match_id=request.effective_match_id(), date=date_str,
        agent_config_hash=compute_agent_config_hash(config),
        odds={}, recommendation=result.model_dump(), triggered_by="manual_regenerate",
    )
    print(f"Recorded to sandbox cache: {cache._db_path}")

    # 4. Log one bet from the recommendation (if any market was generated)
    tracker = bets.get_bet_tracker()
    if result.markets:
        market = result.markets[0]
        bet_from_rec = tracker.create_bet(
            match_id=request.effective_match_id(), date=date_str,
            home_team=fixture.home_team, away_team=fixture.away_team,
            market=market.market, selection=market.selection,
            odds=market.current_odds or 2.0, stake=10.0,
            source="from_recommendation", recommendation_snapshot=result.model_dump(),
        )
        print(f"Logged bet from recommendation: id={bet_from_rec.id}")
    else:
        print("No markets in the recommendation -- skipping the from-recommendation bet.")

    # 5. Log one bet manually
    bet_manual = tracker.create_bet(
        match_id=request.effective_match_id(), date=date_str,
        home_team=fixture.home_team, away_team=fixture.away_team,
        market="result_3way", selection="home", odds=2.0, stake=5.0,
        source="manual", recommendation_snapshot=None,
    )
    print(f"Logged manual bet: id={bet_manual.id}")

    # 6. Settle both against the real historical result
    settled = settle_open_bets(tracker, fixtures_client)
    print(f"Settled {len(settled)} bet(s):")
    for bet in settled:
        print(f"  id={bet.id} outcome={bet.outcome} profit_loss={bet.profit_loss}")

    print("\nSandbox-scoped resources used -- real app/data/*.db untouched:")
    print(f"  cache: {cache._db_path}")
    print(f"  bets:  {tracker._db_path}")


if __name__ == "__main__":
    main()
