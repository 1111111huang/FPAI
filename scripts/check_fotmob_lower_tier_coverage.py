"""One-off spike (domestic cup pre-flight, 2026-11-04): check whether
fotmob's /api/data/matches payload (the same unfiltered endpoint
fetch_all_matches() already uses for UCL/UEL, per US#212's docstring) also
carries the 5 second divisions and 6 domestic cups in scope -- reading the
raw 'leagues' list directly, since fetch_all_matches()/_parse_finished_matches()
already discard league identity before this check could run."""
from __future__ import annotations

import json
import sys
from datetime import date, timedelta
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from src.ingestion.fotmob.fetcher import _fetch_matches_payload  # noqa: E402

NAME_FRAGMENTS = {
    "CHAMPIONSHIP": ["championship"],
    "SEGUNDA": ["segunda", "laliga 2", "la liga 2"],
    "SERIE_B": ["serie b"],
    "BUNDESLIGA_2": ["2. bundesliga", "bundesliga 2"],
    "LIGUE_2": ["ligue 2"],
    "FA_CUP": ["fa cup"],
    "CARABAO_CUP": ["carabao", "efl cup", "league cup"],
    "COPA_DEL_REY": ["copa del rey"],
    "DFB_POKAL": ["dfb", "pokal"],
    "COUPE_DE_FRANCE": ["coupe de france"],
    "COPPA_ITALIA": ["coppa italia"],
}


def main() -> None:
    today = date.today()
    found: dict[str, set[str]] = {k: set() for k in NAME_FRAGMENTS}

    # Check the last 7 days -- enough to catch at least one matchday for
    # most of these, without a long multi-week backfill-style loop.
    for offset in range(7):
        day = today - timedelta(days=offset)
        payload = _fetch_matches_payload(day, delay=1.0)
        if payload is None:
            continue
        for league_entry in payload.get("leagues", []):
            name = str(league_entry.get("name", ""))
            name_lower = name.lower()
            for key, fragments in NAME_FRAGMENTS.items():
                if any(frag in name_lower for frag in fragments):
                    found[key].add(name)

    for key, names in found.items():
        if names:
            print(f"{key}: FOUND -- real league names seen: {sorted(names)}")
        else:
            print(f"{key}: NOT FOUND in the last 7 days")

    out_path = Path(__file__).parent.parent / "data" / "fotmob_lower_tier_coverage.json"
    out_path.write_text(json.dumps({k: sorted(v) for k, v in found.items()}, indent=2))
    print(f"\nSaved findings to {out_path}")


if __name__ == "__main__":
    main()
