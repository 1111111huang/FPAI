"""llm_only tier (Task 15): pull real historical 1X2 odds from OddsPapi for
UEFA Champions League (UCL) and Europa League (UEL), most-recent-match-first,
until the free-tier quota (250 req/month) is exhausted.

Deliberately a SEPARATE script from scripts/pull_oddspapi_btts_corners.py
(which already covers E0/SP1/I1/F1/D1 plus, incidentally, UCL/UEL in its own
LEAGUE_TOURNAMENT_IDS dict) rather than just running that script: reusing it
would sort UCL/UEL fixtures together with the 5 domestic leagues' own
most-recent-first backfill into ONE shared manifest/quota budget, making
UCL/UEL's consumption unpredictable and entangled with unrelated progress.
This script keeps its own manifest.json and its own 250/month quota usage
fully scoped to just UCL+UEL.

Same confirmed facts as the existing script (live-verified 2026-10-06):
- Historical odds only exist from 2026-01-01 onward.
- One /v4/historical-odds call returns every market for that fixture.
- Fixture discovery (tournamentId+statusId=2, no date range) is one call
  per league for the whole finished-fixture history.

Unlike the existing script (which only checks for BTTS market presence via
a boolean flag), this one extracts real 1X2 odds via extract_1x2_odds() and
stores the {"home", "draw", "away"} prices directly in the manifest entry --
so a later task (17, join against fotmob results by team name + date) can
read the manifest alone without re-parsing every raw JSON snapshot. It also
records participant1Name/participant2Name/startTime per entry for that join,
since fixtureId alone won't line up with fotmob's own identifiers.

Saves one raw JSON file per match to
data/oddspapi_ucl_uel_snapshots/<league>/<fixtureId>.json plus a manifest.json
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

sys.path.append(str(Path(__file__).resolve().parent.parent))
from app.backend.oddspapi_client import extract_1x2_odds  # noqa: E402

load_dotenv()

API_KEY = sys.argv[1] if len(sys.argv) > 1 else os.environ.get("ODDSPAPI_API_KEY")
if not API_KEY:
    sys.exit("No API key: set ODDSPAPI_API_KEY in .env or pass one as a CLI argument.")
BASE_URL = "https://api.oddspapi.io"
OUT_DIR = Path(__file__).parent.parent / "data" / "oddspapi_ucl_uel_snapshots"
COOLDOWN_SECONDS = 5.0
CUTOFF_DATE = "2026-01-01"  # confirmed: nothing before this returns odds

# Confirmed live via /v4/tournaments?sportId=10, 2026-10-06 (same ids already
# registered in app/backend/oddspapi_client.py and pull_oddspapi_btts_corners.py)
LEAGUE_TOURNAMENT_IDS = {
    "UCL": 7,    # UEFA Champions League
    "UEL": 679,  # UEFA Europa League
}


def _get(path: str, **params) -> tuple[int, dict | list]:
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

    print("=== Discovering finished UCL/UEL fixtures (>= 2026-01-01 only) ===")
    all_fixtures: list[dict] = []
    for league, tid in LEAGUE_TOURNAMENT_IDS.items():
        all_fixtures.extend(fetch_finished_fixtures(league, tid))
        time.sleep(COOLDOWN_SECONDS)

    # Most recent first, across UCL+UEL combined.
    all_fixtures.sort(key=lambda f: f["startTime"], reverse=True)
    print(f"\nTotal candidate matches (>= {CUTOFF_DATE}, UCL+UEL): {len(all_fixtures)}")

    todo = [f for f in all_fixtures if str(f["fixtureId"]) not in manifest]
    print(f"Already pulled in a prior run: {len(all_fixtures) - len(todo)}")
    print(f"Remaining to attempt this run: {len(todo)}\n")

    pulled = 0
    no_odds = 0
    errors = 0
    for i, fixture in enumerate(todo, 1):
        fid = str(fixture["fixtureId"])
        league = fixture["_league"]
        label = f"{fixture.get('participant1Name')} vs {fixture.get('participant2Name')} ({fixture['startTime'][:10]}, {league})"

        status, data = _get("/v4/historical-odds", fixtureId=fid, bookmakers="pinnacle")

        if status == 429 or (isinstance(data, dict) and data.get("error", {}).get("code") in ("RATE_LIMIT_EXCEEDED", "QUOTA_EXCEEDED")):
            print(f"[{i}/{len(todo)}] QUOTA EXHAUSTED at {label}: {data}")
            break
        if status == 404 or (isinstance(data, dict) and "error" in data):
            no_odds += 1
            manifest[fid] = {
                "league": league, "date": fixture["startTime"][:10], "status": "no_odds",
                "participant1Name": fixture.get("participant1Name"),
                "participant2Name": fixture.get("participant2Name"),
                "startTime": fixture.get("startTime"),
                "detail": data,
            }
            print(f"[{i}/{len(todo)}] no odds  | {label}")
        elif status == 200:
            league_dir = OUT_DIR / league
            league_dir.mkdir(exist_ok=True)
            (league_dir / f"{fid}.json").write_text(json.dumps(data))
            markets = list(data.get("bookmakers", {}).get("pinnacle", {}).get("markets", {}).keys())
            odds_1x2 = extract_1x2_odds(data)  # None is a real, expected case (not every market exists)
            manifest[fid] = {
                "league": league, "date": fixture["startTime"][:10], "status": "ok",
                "market_count": len(markets), "odds_1x2": odds_1x2,
                "participant1Name": fixture.get("participant1Name"),
                "participant2Name": fixture.get("participant2Name"),
                "startTime": fixture.get("startTime"),
            }
            pulled += 1
            print(f"[{i}/{len(todo)}] OK ({len(markets)} markets, odds_1x2={odds_1x2}) | {label}")
        else:
            errors += 1
            print(f"[{i}/{len(todo)}] ERROR status={status} | {label}: {data}", file=sys.stderr)

        manifest_path.write_text(json.dumps(manifest, indent=2))
        time.sleep(COOLDOWN_SECONDS)
    else:
        print("\n=== Finished attempting every candidate match (quota not exhausted) ===")

    print(f"\nDone this run. Pulled: {pulled} | No odds: {no_odds} | Errors: {errors} | "
          f"Total candidates: {len(all_fixtures)} | Covered so far (all runs): {len(manifest)}")


if __name__ == "__main__":
    main()
