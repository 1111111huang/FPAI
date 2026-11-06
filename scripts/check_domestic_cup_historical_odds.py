"""One-off spike (domestic cup pre-flight, 2026-11-04): check whether
OddsPapi's historical-odds endpoint (/v4/historical-odds) -- the real
source for *training* odds, not /v4/odds which is live-only -- actually
covers the 6 domestic cups, and from what date. Mirrors
scripts/pull_oddspapi_btts_corners.py's already-confirmed-live call shape
and the already-confirmed match-winner market ID (101).
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
    # Confirmed live, 2026-11-04: /v4/historical-odds's real shape is
    # "players"->"0" as a LIST of price ticks over time (createdAt/price
    # pairs) -- genuinely different from /v4/odds (Task 2), which is a
    # single dict. Take the last (most recent/closing) tick.
    ticks = outcome.get("players", {}).get("0", [])
    return ticks[-1].get("price") if ticks else None


def _extract_1x2(payload: dict) -> tuple[float | None, float | None, float | None]:
    # Confirmed live, 2026-11-04: /v4/historical-odds's real top-level key
    # is "bookmakers" (not "bookmakerOdds" -- that's the LIVE /v4/odds
    # endpoint's key, Task 2). This matches what production's
    # _parse_corners_odds already assumes -- correct for THIS endpoint,
    # just applied to the wrong one in app/backend/oddspapi_client.py's
    # get_corners_odds(), which calls /v4/odds, not /v4/historical-odds.
    outcomes = (
        payload.get("bookmakers", {}).get("pinnacle", {})
        .get("markets", {}).get(MATCH_WINNER_MARKET_ID, {}).get("outcomes", {})
    )
    return (
        _last_price(outcomes, MATCH_WINNER_OUTCOME_HOME),
        _last_price(outcomes, MATCH_WINNER_OUTCOME_DRAW),
        _last_price(outcomes, MATCH_WINNER_OUTCOME_AWAY),
    )


def main() -> None:
    findings = {}
    for cup, tournament_id in TOURNAMENT_IDS.items():
        # statusId=2 = finished, same convention pull_oddspapi_btts_corners.py
        # already confirmed live for the 5 modeled leagues.
        status, fixtures = _get("/v4/fixtures", tournamentId=tournament_id, statusId=2)
        if not isinstance(fixtures, list):
            print(f"{cup}: unexpected fixtures response (status={status}): {fixtures}")
            findings[cup] = {"finished_fixture_count": 0, "checked_samples": []}
            continue

        fixtures_sorted = sorted(fixtures, key=lambda f: f.get("startTime", ""))
        print(f"\n{cup}: {len(fixtures_sorted)} finished fixtures total")
        if not fixtures_sorted:
            findings[cup] = {"finished_fixture_count": 0, "checked_samples": []}
            continue

        checked = []
        for fixture in (fixtures_sorted[0], fixtures_sorted[-1]):
            fixture_id = str(fixture["fixtureId"])
            start_time = fixture.get("startTime")
            print(f"  ALERT: spending 1 real OddsPapi credit -- /v4/historical-odds for fixture {fixture_id} ({start_time})")
            time.sleep(5.0)
            odds_status, payload = _get("/v4/historical-odds", fixtureId=fixture_id, bookmakers="pinnacle")
            home = draw = away = None
            if odds_status == 200 and isinstance(payload, dict):
                home, draw, away = _extract_1x2(payload)
            has_odds = all(v is not None for v in (home, draw, away))
            print(f"    fixture {fixture_id} ({start_time}): status={odds_status} home={home} draw={draw} away={away} -> has_1x2_odds={has_odds}")
            checked.append({"start_time": start_time, "has_odds": has_odds})

        findings[cup] = {"finished_fixture_count": len(fixtures_sorted), "checked_samples": checked}

    out_path = Path(__file__).parent.parent / "data" / "domestic_cup_historical_odds_coverage.json"
    out_path.write_text(json.dumps(findings, indent=2))
    print(f"\nSaved findings to {out_path}")


if __name__ == "__main__":
    main()
