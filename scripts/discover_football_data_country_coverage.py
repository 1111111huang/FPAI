"""One-off spike (UCL/UEL pre-flight, 2026-11-04): check which additional
countries football-data.co.uk covers beyond the 6 modeled leagues, by
testing real <country>m.php pages (same pattern main.py's
_SCRAPE_SOURCE_OVERRIDE_BY_LEAGUE already uses) -- a candidate list drawn
from UCL/UEL's actual participant countries (Task 1), not guessed in the
abstract."""
from __future__ import annotations

import json
from pathlib import Path

import requests

# Candidate countries, chosen because UCL/UEL clubs from them appear in
# Task 1's real club list -- not an abstract "every UEFA country" guess.
CANDIDATES = [
    "netherlands", "portugal", "belgium", "turkey", "scotland", "greece",
    "austria", "switzerland", "denmark", "norway", "poland", "russia",
    "romania", "ireland", "finland", "ukraine", "croatia", "serbia",
    "czechrepublic", "israel", "cyprus",
]


def main() -> None:
    found = []
    not_found = []
    for country in CANDIDATES:
        url = f"https://www.football-data.co.uk/{country}m.php"
        resp = requests.get(url, timeout=15)
        if resp.status_code == 200:
            found.append(country)
            print(f"{country}: FOUND ({url})")
        else:
            not_found.append(country)
            print(f"{country}: not found (status={resp.status_code})")

    out_path = Path(__file__).parent.parent / "data" / "football_data_wider_country_coverage.json"
    out_path.write_text(json.dumps({"found": found, "not_found": not_found}, indent=2))
    print(f"\nSaved to {out_path}")


if __name__ == "__main__":
    main()
