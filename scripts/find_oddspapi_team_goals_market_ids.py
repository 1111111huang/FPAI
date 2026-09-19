"""One-off investigation script (W199): pin which of OddsPapi's team-goals
market IDs (10224-10236 and 10240-10250, already confirmed present in
data/oddspapi_snapshots/ -- 98.3% of the 1,005 already-pulled matches carry
at least one) map to home vs. away and to the 1.5 line specifically. Mirrors
how data/oddspapi_snapshots/corners_line_map.json was built for corners'
own 9.5 line.

Usage: python scripts/find_oddspapi_team_goals_market_ids.py
Requires ODDSPAPI_API_KEY in .env. /v4/markets is a metadata/reference
endpoint -- confirmed in the W199 investigation notes (documents/
app_user_stories.md) not to count against the 250/month historical-odds
quota."""
from __future__ import annotations

import json
import os
import sys

import requests
from dotenv import load_dotenv

load_dotenv()

API_KEY = os.environ.get("ODDSPAPI_API_KEY")
if not API_KEY:
    sys.exit("No API key: set ODDSPAPI_API_KEY in .env")

BASE_URL = "https://api.oddspapi.io"
_CANDIDATE_IDS = list(range(10224, 10237, 2)) + list(range(10240, 10251, 2))


def main() -> None:
    resp = requests.get(f"{BASE_URL}/v4/markets", params={"apiKey": API_KEY, "sportId": 10}, timeout=30)
    resp.raise_for_status()
    markets = resp.json()

    by_id = {str(m.get("marketId")): m for m in markets if isinstance(m, dict)}

    print(f"Fetched {len(markets)} total market definitions.\n")
    print("=== Candidate team-goals market IDs (from already-pulled snapshot data) ===")
    for market_id in _CANDIDATE_IDS:
        entry = by_id.get(str(market_id))
        if entry is None:
            print(f"{market_id}: NOT FOUND in /v4/markets response")
            continue
        print(f"{market_id}: {json.dumps(entry)}")


if __name__ == "__main__":
    main()
