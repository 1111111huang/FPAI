"""One-off spike (domestic cup pre-flight, 2026-11-04): discover The Odds
API sport_keys for the 5 second divisions in scope, the same way every
existing entry in app/backend/odds_sport_keys.py was confirmed (live
/v4/sports/?all=true listing, matched by real title, not a name-pattern
guess)."""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import requests
from dotenv import load_dotenv

load_dotenv()

API_KEY = os.environ.get("ODDS_API_KEY")
if not API_KEY:
    sys.exit("No API key: set ODDS_API_KEY in .env.")
BASE_URL = "https://api.the-odds-api.com"

# Name fragments, one per second division in scope. Confirm against the real
# listing below -- The Odds API's own key naming isn't known in advance.
DIVISION_NAME_FRAGMENTS = {
    "CHAMPIONSHIP": ["championship"],
    # Confirmed live, 2026-11-04: the real title is "La Liga 2 - Spain", not
    # "Segunda" -- matching only on "segunda" (a reasonable-looking guess)
    # found nothing; the real key (soccer_spain_segunda_division) does
    # contain "segunda", but the *title* field this script matches on does
    # not, so both fragments are needed.
    "SEGUNDA": ["segunda", "la liga 2"],
    "SERIE_B": ["serie b"],
    "BUNDESLIGA_2": ["bundesliga"],  # must disambiguate from top-flight D1 + Austria, see Step 2
    "LIGUE_2": ["ligue 2"],
}


def main() -> None:
    resp = requests.get(f"{BASE_URL}/v4/sports/", params={"apiKey": API_KEY, "all": "true"}, timeout=30)
    resp.raise_for_status()
    sports = resp.json()
    print(f"Fetched {len(sports)} total sports/leagues.")

    matches: dict[str, list[dict]] = {d: [] for d in DIVISION_NAME_FRAGMENTS}
    for s in sports:
        if not str(s.get("key", "")).startswith("soccer_"):
            continue
        title_lower = str(s.get("title", "")).lower()
        for division, fragments in DIVISION_NAME_FRAGMENTS.items():
            if any(frag in title_lower for frag in fragments):
                matches[division].append(
                    {"key": s.get("key"), "title": s.get("title"), "active": s.get("active")}
                )

    for division, found in matches.items():
        print(f"\n{division}:")
        if not found:
            print("  NO MATCH FOUND")
        for f in found:
            print(f"  key={f['key']}  title={f['title']!r}  active={f['active']}")

    out_path = Path(__file__).parent.parent / "data" / "second_division_sport_key_discovery.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(matches, indent=2))
    print(f"\nSaved raw matches to {out_path}")


if __name__ == "__main__":
    main()
