"""One-off spike (domestic cup pre-flight, 2026-11-04): discover OddsPapi
tournament IDs for the 6 domestic cups in scope. Follows the same
confirmed-live pattern already used for the 5 modeled leagues and UCL/UEL
(scripts/pull_oddspapi_btts_corners.py's LEAGUE_TOURNAMENT_IDS comment):
/v4/tournaments?sportId=10 lists every soccer tournament OddsPapi knows
about; fixture/tournament discovery calls are (almost) free, unlike
/v4/odds, so this script is safe to re-run.
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import requests
from dotenv import load_dotenv

load_dotenv()

API_KEY = os.environ.get("ODDSPAPI_API_KEY")
if not API_KEY:
    sys.exit("No API key: set ODDSPAPI_API_KEY in .env.")
BASE_URL = "https://api.oddspapi.io"

# Name fragments + category, one per cup in scope. OddsPapi reuses generic
# names like "FA Cup"/"League Cup" across dozens of countries (confirmed
# live, 2026-11-04: 14 "FA Cup"-named entries alone, covering England,
# China, South Africa, Ghana, Thailand, Malaysia, Fiji, Hong Kong, plus
# women's/qualification variants) -- name alone is not enough,
# "categoryName" (the real per-tournament country/region field) is required
# to disambiguate.
CUP_NAME_FRAGMENTS = {
    "FA_CUP": (["fa cup"], "England"),
    "CARABAO_CUP": (["efl cup"], "England"),
    "COPA_DEL_REY": (["copa del rey"], "Spain"),
    "DFB_POKAL": (["dfb pokal"], "Germany"),
    "COUPE_DE_FRANCE": (["coupe de france"], "France"),
    "COPPA_ITALIA": (["coppa italia"], "Italy"),
}


def main() -> None:
    resp = requests.get(
        f"{BASE_URL}/v4/tournaments", params={"apiKey": API_KEY, "sportId": 10}, timeout=30,
    )
    resp.raise_for_status()
    tournaments = resp.json()
    print(f"Fetched {len(tournaments)} total soccer tournaments.")

    # Real field names confirmed live (not "id"/"name" as the inline
    # LEAGUE_TOURNAMENT_IDS comments elsewhere implied): tournamentId,
    # tournamentName, categoryName.
    matches: dict[str, list[dict]] = {cup: [] for cup in CUP_NAME_FRAGMENTS}
    for t in tournaments:
        name_lower = str(t.get("tournamentName", "")).lower()
        category = str(t.get("categoryName", ""))
        for cup, (fragments, expected_category) in CUP_NAME_FRAGMENTS.items():
            if any(frag in name_lower for frag in fragments) and category == expected_category:
                matches[cup].append({
                    "id": t.get("tournamentId"), "name": t.get("tournamentName"), "category": category,
                })

    for cup, found in matches.items():
        print(f"\n{cup}:")
        if not found:
            print("  NO MATCH FOUND")
        for f in found:
            print(f"  id={f['id']}  name={f['name']!r}  category={f['category']!r}")

    out_path = Path(__file__).parent.parent / "data" / "domestic_cup_tournament_discovery.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(matches, indent=2))
    print(f"\nSaved raw matches to {out_path}")


if __name__ == "__main__":
    main()
