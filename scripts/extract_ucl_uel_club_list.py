"""One-off spike (UCL/UEL pre-flight, 2026-11-04): extract the real set of
unique clubs that appeared in UCL/UEL since 2026-01-01, from the
already-fetched historical manifest (zero new API cost -- this data was
paid for once, during the abandoned branch's original backfill)."""
from __future__ import annotations

import json
from pathlib import Path

MANIFEST_PATH = "/tmp/ucl_uel_manifest.json"


def main() -> None:
    with open(MANIFEST_PATH) as f:
        manifest = json.load(f)

    clubs: set[str] = set()
    by_league: dict[str, set[str]] = {"UCL": set(), "UEL": set()}
    for fixture in manifest.values():
        p1, p2 = fixture["participant1Name"], fixture["participant2Name"]
        clubs.add(p1)
        clubs.add(p2)
        league = fixture.get("league", "UNKNOWN")
        by_league.setdefault(league, set()).update([p1, p2])

    print(f"{len(manifest)} fixtures, {len(clubs)} unique clubs")
    for league, names in by_league.items():
        print(f"  {league}: {len(names)} unique clubs")

    out_path = Path(__file__).parent.parent / "data" / "ucl_uel_club_list.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps({
        "all_clubs": sorted(clubs),
        "by_league": {k: sorted(v) for k, v in by_league.items()},
    }, indent=2))
    print(f"\nSaved to {out_path}")


if __name__ == "__main__":
    main()
