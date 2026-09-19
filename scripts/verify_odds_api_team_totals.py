"""One-off investigation script (W199): capture a real team_totals response
from The Odds API before writing a parser against it -- this codebase's
established practice (see app/backend/odds_api_client.py's W164 docstring)
is to never ship a new odds-JSON parser against an assumed shape.

Usage: python scripts/verify_odds_api_team_totals.py
Requires ODDS_API_KEY in .env (this makes one real, billed API call plus
one bulk-odds call to find a live event_id -- ~4-7 credits total against
the free-tier 500/month budget)."""
from __future__ import annotations

import json
import os
import sys

import requests
from dotenv import load_dotenv

load_dotenv()

API_KEY = os.environ.get("ODDS_API_KEY")
if not API_KEY:
    sys.exit("No API key: set ODDS_API_KEY in .env")

BASE_URL = "https://api.the-odds-api.com/v4"
SPORT_KEY = "soccer_epl"


def main() -> None:
    bulk_resp = requests.get(
        f"{BASE_URL}/sports/{SPORT_KEY}/odds",
        params={"apiKey": API_KEY, "regions": "uk", "markets": "h2h", "oddsFormat": "decimal"},
        timeout=10,
    )
    bulk_resp.raise_for_status()
    events = bulk_resp.json()
    if not events:
        sys.exit("No upcoming EPL events found -- try again closer to a matchday.")

    event = events[0]
    print(f"Using event: {event['home_team']} v {event['away_team']} ({event['id']})")

    event_resp = requests.get(
        f"{BASE_URL}/sports/{SPORT_KEY}/events/{event['id']}/odds",
        params={"apiKey": API_KEY, "regions": "uk,us,us2", "markets": "team_totals", "oddsFormat": "decimal"},
        timeout=10,
    )
    event_resp.raise_for_status()
    payload = event_resp.json()

    print("\n=== Full response ===")
    print(json.dumps(payload, indent=2))

    print("\n=== team_totals markets found, by bookmaker ===")
    for bookmaker in payload.get("bookmakers", []):
        for market in bookmaker.get("markets", []):
            if market.get("key") == "team_totals":
                print(f"\nBookmaker: {bookmaker['key']}")
                for outcome in market.get("outcomes", []):
                    print(f"  {outcome}")


if __name__ == "__main__":
    main()
