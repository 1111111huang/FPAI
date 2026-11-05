"""One-off spike (domestic cup pre-flight, 2026-11-04): confirm The Odds API
actually returns real odds rows (not just a listed-but-dormant sport) for
each second division discovered in Task 4. Reuses the existing production
OddsAPIClient instead of re-implementing the HTTP call."""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

from dotenv import load_dotenv

sys.path.insert(0, str(Path(__file__).parent.parent))
from app.backend.odds_api_client import CreditCounter, OddsAPIClient  # noqa: E402

load_dotenv()

# ODDS_API_KEY and _2 confirmed exhausted live, 2026-11-04 (401
# OUT_OF_USAGE_CREDITS, x-requests-remaining=0) -- unrelated to domestic
# cups, found while testing this task. _3 confirmed to have real remaining
# capacity (169/500), used here instead so this check can actually run.
API_KEY = os.environ.get("ODDS_API_KEY_3")
if not API_KEY:
    sys.exit("No API key: set ODDS_API_KEY_3 in .env.")

# Confirmed live, Task 4 (2026-11-04).
SPORT_KEYS: dict[str, str] = {
    "CHAMPIONSHIP": "soccer_efl_champ",
    "SEGUNDA": "soccer_spain_segunda_division",
    "SERIE_B": "soccer_italy_serie_b",
    "BUNDESLIGA_2": "soccer_germany_bundesliga2",
    "LIGUE_2": "soccer_france_ligue_two",
}


def main() -> None:
    # A fresh, zeroed CreditCounter -- this spike's own handful of calls
    # (cost = len(markets) * len(regions) = 1 each with the client's
    # defaults) shouldn't be tallied against the real production counter
    # persisted at data/odds_api_credit_counter.json, and doesn't need to be
    # -- reconciled manually afterward instead, same as the OddsPapi calls.
    client = OddsAPIClient(api_key=API_KEY, credit_counter=CreditCounter())
    findings = {}
    for division, sport_key in SPORT_KEYS.items():
        print(f"ALERT: spending 1 real Odds API credit -- get_odds({sport_key!r})")
        odds = client.get_odds(sport_key=sport_key)
        count = len(odds) if odds else 0
        print(f"{division} ({sport_key}): {count} real odds rows")
        findings[division] = {"sport_key": sport_key, "odds_row_count": count}

    out_path = Path(__file__).parent.parent / "data" / "second_division_odds_coverage.json"
    out_path.write_text(json.dumps(findings, indent=2))
    print(f"\nSaved findings to {out_path}")


if __name__ == "__main__":
    main()
