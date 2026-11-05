"""One-off spike (UCL/UEL pre-flight, 2026-11-04): for clubs Task 3 could
not resolve to a football-data.co.uk country, check fotmob's unfiltered
/api/data/matches payload for any recent match involving them -- the
lighter-weight signal the Dixon-Coles fallback would actually have to rely
on for this long tail."""
from __future__ import annotations

import json
import sys
from datetime import date, timedelta
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from src.ingestion.fotmob.fetcher import _fetch_matches_payload  # noqa: E402

BUCKETS_PATH = Path(__file__).parent.parent / "data" / "ucl_uel_club_buckets.json"


def main() -> None:
    with open(BUCKETS_PATH) as f:
        unresolved = json.load(f)["unresolved"]

    today = date.today()
    found: dict[str, bool] = {club: False for club in unresolved}

    for offset in range(14):
        day = today - timedelta(days=offset)
        payload = _fetch_matches_payload(day, delay=1.0)
        if payload is None:
            continue
        for league_entry in payload.get("leagues", []):
            for match in league_entry.get("matches", []):
                home = str(match.get("home", {}).get("name", ""))
                away = str(match.get("away", {}).get("name", ""))
                for club in unresolved:
                    if found[club]:
                        continue
                    club_key = club.split()[0].lower()
                    if club_key in home.lower() or club_key in away.lower():
                        found[club] = True

    resolved_count = sum(found.values())
    print(f"{resolved_count}/{len(unresolved)} previously-unresolved clubs found in fotmob's payload (last 14 days)")
    for club, was_found in sorted(found.items()):
        print(f"  {club}: {'FOUND' if was_found else 'not found'}")

    out_path = Path(__file__).parent.parent / "data" / "ucl_uel_fotmob_long_tail_check.json"
    out_path.write_text(json.dumps(found, indent=2))
    print(f"\nSaved to {out_path}")


if __name__ == "__main__":
    main()
