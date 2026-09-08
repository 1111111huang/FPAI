"""One-off script (A100 investigation): pull real historical BTTS/corners odds
from OddsPapi for the current-season backtest corpus, most-recent-match-first,
until the free-tier quota (250 req/month) is exhausted.

Confirmed live before writing this:
- Auth: query param apiKey (not a header).
- Historical odds ONLY exist from 2026-01-01 onward -- an Aug 2025 fixture
  returned "No historical odds found" (404); a May 2026 fixture returned
  real data. So only matches >= 2026-01-01 are attempted at all -- calling
  for anything older is a guaranteed-wasted call against a 250/month budget.
- One /v4/historical-odds call returns EVERY market for that fixture (BTTS
  id=104, many corners over/under lines, etc.) -- not per-market billing.
- Fixture discovery is (almost) free: tournamentId+statusId=2 with no date
  range returns the WHOLE finished-fixture history for that league in one
  call, rather than needing one call per 48h window.

Saves one raw JSON file per match to data/oddspapi_snapshots/<league>/<fixtureId>.json
(SnapshotStore's own hashing convention isn't reused here -- this is a
different, adjacent artifact, not a tool-call recording) plus a manifest.json
index. Stops the moment the API signals quota exhaustion and reports a
summary rather than guessing when to stop.
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

# A100 follow-up: real keys never belong in source -- read from .env
# (ODDSPAPI_API_KEY / _2 / _3, see .env.example) instead of a hardcoded
# default. A positional CLI arg still overrides, for a one-off ad-hoc key.
API_KEY = sys.argv[1] if len(sys.argv) > 1 else os.environ.get("ODDSPAPI_API_KEY")
if not API_KEY:
    sys.exit("No API key: set ODDSPAPI_API_KEY in .env or pass one as a CLI argument.")
BASE_URL = "https://api.oddspapi.io"
OUT_DIR = Path(__file__).parent.parent / "data" / "oddspapi_snapshots"
COOLDOWN_SECONDS = 5.0
CUTOFF_DATE = "2026-01-01"  # confirmed: nothing before this returns odds

# Confirmed live via /v4/tournaments?sportId=10
LEAGUE_TOURNAMENT_IDS = {
    "E0": 17,   # Premier League, England
    "SP1": 8,   # LaLiga, Spain
    "I1": 23,   # Serie A, Italy
    "F1": 34,   # Ligue 1, France
    "D1": 35,   # Bundesliga, Germany
}


def _get(path: str, **params) -> dict | list:
    params["apiKey"] = API_KEY
    resp = requests.get(f"{BASE_URL}{path}", params=params, timeout=30)
    return resp.status_code, resp.json() if resp.content else {}


def fetch_finished_fixtures(league: str, tournament_id: int) -> list[dict]:
    status, data = _get("/v4/fixtures", tournamentId=tournament_id, statusId=2)
    if status != 200 or not isinstance(data, list):
        print(f"  WARNING: fixture list for {league} failed (status={status}): {data}", file=sys.stderr)
        return []
    fixtures = [f for f in data if f.get("startTime", "") >= CUTOFF_DATE]
    print(f"  {league}: {len(data)} finished fixtures total, {len(fixtures)} >= {CUTOFF_DATE}")
    return [{**f, "_league": league} for f in fixtures]


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    manifest_path = OUT_DIR / "manifest.json"
    manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else {}

    print("=== Discovering finished fixtures per league (>= 2026-01-01 only) ===")
    all_fixtures: list[dict] = []
    for league, tid in LEAGUE_TOURNAMENT_IDS.items():
        all_fixtures.extend(fetch_finished_fixtures(league, tid))
        time.sleep(COOLDOWN_SECONDS)

    # Most recent first, across all leagues combined.
    all_fixtures.sort(key=lambda f: f["startTime"], reverse=True)
    print(f"\nTotal candidate matches (>= {CUTOFF_DATE}, all 5 leagues): {len(all_fixtures)}")

    todo = [f for f in all_fixtures if f["fixtureId"] not in manifest]
    print(f"Already pulled in a prior run: {len(all_fixtures) - len(todo)}")
    print(f"Remaining to attempt this run: {len(todo)}\n")

    pulled = 0
    empty = 0
    errors = 0
    for i, fixture in enumerate(todo, 1):
        fid = fixture["fixtureId"]
        league = fixture["_league"]
        label = f"{fixture.get('participant1Name')} vs {fixture.get('participant2Name')} ({fixture['startTime'][:10]}, {league})"

        status, data = _get("/v4/historical-odds", fixtureId=fid, bookmakers="pinnacle")

        if status == 429 or (isinstance(data, dict) and data.get("error", {}).get("code") in ("RATE_LIMIT_EXCEEDED", "QUOTA_EXCEEDED")):
            print(f"[{i}/{len(todo)}] QUOTA EXHAUSTED at {label}: {data}")
            break
        if status == 404 or (isinstance(data, dict) and "error" in data):
            empty += 1
            manifest[fid] = {"league": league, "date": fixture["startTime"][:10], "status": "no_odds", "detail": data}
            print(f"[{i}/{len(todo)}] no odds  | {label}")
        elif status == 200:
            league_dir = OUT_DIR / league
            league_dir.mkdir(exist_ok=True)
            (league_dir / f"{fid}.json").write_text(json.dumps(data))
            markets = list(data.get("bookmakers", {}).get("pinnacle", {}).get("markets", {}).keys())
            has_btts = "104" in markets
            manifest[fid] = {
                "league": league, "date": fixture["startTime"][:10], "status": "ok",
                "market_count": len(markets), "has_btts": has_btts,
            }
            pulled += 1
            print(f"[{i}/{len(todo)}] OK ({len(markets)} markets, btts={has_btts}) | {label}")
        else:
            errors += 1
            print(f"[{i}/{len(todo)}] ERROR status={status} | {label}: {data}", file=sys.stderr)

        manifest_path.write_text(json.dumps(manifest, indent=2))
        time.sleep(COOLDOWN_SECONDS)
    else:
        print("\n=== Finished attempting every candidate match (quota not exhausted) ===")

    print(f"\nDone this run. Pulled: {pulled} | No odds: {empty} | Errors: {errors} | "
          f"Total candidates: {len(all_fixtures)} | Covered so far (all runs): {len(manifest)}")


if __name__ == "__main__":
    main()
