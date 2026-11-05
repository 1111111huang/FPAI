"""One-off spike (domestic cup pre-flight, 2026-11-04): for each domestic
cup tournament ID discovered in Task 1, fetch real fixtures and check
whether real 1X2 odds exist. Reuses the already-confirmed-live market ID
(MATCH_WINNER_MARKET_ID = "101", outcomes 101/102/103 for home/draw/away --
discovered for the UCL/UEL backfill, not re-guessed here) and the
already-confirmed statusId gotcha (statusId=0 = Pre-Game; statusId=1 always
returns zero rows, per W200/W236).

Spends real credits: one /v4/odds call per cup with at least one upcoming
fixture.
"""
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

import requests
from dotenv import load_dotenv

load_dotenv()

API_KEY = os.environ.get("ODDSPAPI_API_KEY")
if not API_KEY:
    sys.exit("No API key: set ODDSPAPI_API_KEY in .env.")
BASE_URL = "https://api.oddspapi.io"
MATCH_WINNER_MARKET_ID = "101"
MATCH_WINNER_OUTCOME_HOME = "101"
MATCH_WINNER_OUTCOME_DRAW = "102"
MATCH_WINNER_OUTCOME_AWAY = "103"

# Confirmed live, Task 1 (2026-11-04).
TOURNAMENT_IDS: dict[str, int] = {
    "FA_CUP": 19,
    "CARABAO_CUP": 21,
    "COPA_DEL_REY": 329,
    "DFB_POKAL": 217,
    "COUPE_DE_FRANCE": 335,
    "COPPA_ITALIA": 328,
}


def _get(path: str, **params) -> tuple[int, dict | list]:
    params["apiKey"] = API_KEY
    resp = requests.get(f"{BASE_URL}{path}", params=params, timeout=30)
    try:
        body = resp.json() if resp.content else {}
    except ValueError:
        body = {}
    return resp.status_code, body


def _last_price(outcomes: dict, outcome_id: str) -> float | None:
    outcome = outcomes.get(outcome_id)
    if not outcome:
        return None
    # BUG found live (2026-11-04): "players"->"0" is a single price dict in
    # the real payload (confirmed: {"active": true, ..., "price": 9.86}),
    # not a list of historical "ticks" as the existing production
    # _last_price (app/backend/oddspapi_client.py) assumes. That assumption
    # never crashed in production because the wrong "bookmakers" key above
    # always made outcomes={} first, so this line was never reached with
    # real data -- two bugs masking each other.
    player = outcome.get("players", {}).get("0", {})
    return player.get("price")


def main() -> None:
    findings = {}
    for cup, tournament_id in TOURNAMENT_IDS.items():
        status, fixtures = _get("/v4/fixtures", tournamentId=tournament_id, statusId=0)
        if status == 404:
            fixtures = []
        elif not isinstance(fixtures, list):
            print(f"{cup}: unexpected fixtures response (status={status}): {fixtures}")
            fixtures = []

        print(f"\n{cup}: {len(fixtures)} upcoming (statusId=0) fixtures")
        if not fixtures:
            findings[cup] = {"fixture_count": 0, "has_1x2_odds": None}
            continue

        # BUG found live (2026-11-04): /v4/fixtures does not return results
        # sorted by date -- fixtures[0] picked an arbitrary far-future
        # fixture with no odds posted yet, which looked like "no coverage"
        # but was really just "too early for this specific match." Sort by
        # startTime and use the nearest one instead.
        fixtures_sorted = sorted(fixtures, key=lambda f: f.get("startTime", "9999"))
        sample = fixtures_sorted[0]
        fixture_id = str(sample["fixtureId"])
        print(f"  nearest fixture start_time={sample.get('startTime')}")
        print(f"  ALERT: spending 1 real OddsPapi credit -- /v4/odds for fixture {fixture_id}")
        time.sleep(5.0)
        odds_status, odds_payload = _get("/v4/odds", fixtureId=fixture_id, bookmakers="pinnacle")
        outcomes = {}
        if odds_status == 200 and isinstance(odds_payload, dict):
            # BUG found live (2026-11-04): the real top-level key is
            # "bookmakerOdds", not "bookmakers" -- confirmed against a real
            # payload where market 101 (match-winner) had real prices
            # sitting under "bookmakerOdds" and nothing under "bookmakers".
            # This same wrong key also appears in production code
            # (app/backend/oddspapi_client.py's _parse_corners_odds) and two
            # other scripts -- flagged separately, out of scope to fix here.
            outcomes = (
                odds_payload.get("bookmakerOdds", {}).get("pinnacle", {})
                .get("markets", {}).get(MATCH_WINNER_MARKET_ID, {}).get("outcomes", {})
            )
        home = _last_price(outcomes, MATCH_WINNER_OUTCOME_HOME)
        draw = _last_price(outcomes, MATCH_WINNER_OUTCOME_DRAW)
        away = _last_price(outcomes, MATCH_WINNER_OUTCOME_AWAY)
        has_odds = all(v is not None for v in (home, draw, away))
        print(f"  sample fixture {fixture_id}: home={home} draw={draw} away={away} -> has_1x2_odds={has_odds}")
        findings[cup] = {"fixture_count": len(fixtures), "has_1x2_odds": has_odds}

    out_path = Path(__file__).parent.parent / "data" / "domestic_cup_odds_coverage.json"
    out_path.write_text(json.dumps(findings, indent=2))
    print(f"\nSaved findings to {out_path}")


if __name__ == "__main__":
    main()
