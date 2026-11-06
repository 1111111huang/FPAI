"""One-off spike (domestic cup pre-flight, 2026-11-04): check whether each
country's football-data.co.uk page (the same pages main.py's
_SCRAPE_SOURCE_OVERRIDE_BY_LEAGUE already points at for the 6 leagues) also
lists a second-division CSV, and what season range it covers. Read-only --
does not run the actual scraper, just inspects the page for a CSV link."""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import requests

# Same pages main.py already uses for the top-flight leagues (englandm.php
# has no override entry because E0 is football-data.co.uk's own default --
# see main.py's _SCRAPE_SOURCE_OVERRIDE_BY_LEAGUE comment).
COUNTRY_PAGES = {
    "CHAMPIONSHIP": "https://www.football-data.co.uk/englandm.php",
    "SEGUNDA": "https://www.football-data.co.uk/spainm.php",
    "SERIE_B": "https://www.football-data.co.uk/italym.php",
    "BUNDESLIGA_2": "https://www.football-data.co.uk/germanym.php",
    "LIGUE_2": "https://www.football-data.co.uk/francem.php",
}

# Division codes to look for in each page's CSV links -- football-data.co.uk's
# own convention (E0=Premier League, E1=Championship, SP1/SP2, I1/I2, D1/D2,
# F1/F2 per the existing adding-a-league pattern). Confirm these appear as
# real links below, don't assume the convention holds for every country.
EXPECTED_CSV_CODE = {
    "CHAMPIONSHIP": "E1",
    "SEGUNDA": "SP2",
    "SERIE_B": "I2",
    "BUNDESLIGA_2": "D2",
    "LIGUE_2": "F2",
}


def main() -> None:
    findings = {}
    for division, page_url in COUNTRY_PAGES.items():
        resp = requests.get(page_url, timeout=30)
        resp.raise_for_status()
        html = resp.text
        code = EXPECTED_CSV_CODE[division]
        # football-data.co.uk links each division's current-season CSV as
        # mmz4281/<season>/<CODE>.csv -- look for the exact code as its own
        # path segment, not a loose substring (avoids e.g. "E1" matching
        # inside an unrelated "E10" or similar).
        csv_links = sorted(set(re.findall(rf"mmz4281/\d+/{code}\.csv", html)))
        print(f"{division} ({code}): {len(csv_links)} CSV link(s) found on {page_url}")
        for link in csv_links:
            print(f"  {link}")
        findings[division] = {"page_url": page_url, "csv_code": code, "csv_links_found": csv_links}

    out_path = Path(__file__).parent.parent / "data" / "second_division_historical_coverage.json"
    out_path.write_text(json.dumps(findings, indent=2))
    print(f"\nSaved findings to {out_path}")


if __name__ == "__main__":
    main()
