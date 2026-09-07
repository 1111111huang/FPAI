"""A100: extract clean, single-value BTTS/corners odds from the raw OddsPapi
historical-odds snapshots (scripts/pull_oddspapi_btts_corners.py's output)
into a small lookup keyed by our own real match_id, for _build_match_info()
(src/agent/backtest.py) to thread into the agent -- mirroring A69's own
total_goals_odds precedent exactly.

Each raw snapshot file is a full price-movement time series (dozens of
timestamped ticks per market/outcome), not a single number. Takes the LAST
tick (closest to kickoff) as the "current odds" value, matching this
codebase's existing convention (over25_odds/under25_odds are themselves a
market's closing/average line, not an opening one).

Corners line fixed at 9.5 (99.9% real-tick coverage across all 861 resolved
matches, confirmed by direct inspection -- the highest of any line and the
natural analogue of total_goals' own fixed 2.5 line).
"""
from __future__ import annotations

import json
from pathlib import Path

BASE_DIR = Path(__file__).parent.parent / "data" / "oddspapi_snapshots"
OUT_PATH = Path(__file__).parent.parent / "data" / "oddspapi_btts_corners_odds.json"

BTTS_MARKET_ID = "104"
BTTS_OUTCOME_YES = "104"
BTTS_OUTCOME_NO = "105"
CORNERS_LINE = 9.5
CORNERS_MARKET_ID = "10803"  # confirmed via corners_line_map.json: handicap 9.5
CORNERS_OUTCOME_OVER = "10803"
CORNERS_OUTCOME_UNDER = "10804"


def _last_price(outcomes: dict, outcome_id: str) -> float | None:
    outcome = outcomes.get(outcome_id)
    if not outcome:
        return None
    ticks = outcome.get("players", {}).get("0", [])
    if not ticks:
        return None
    return ticks[-1].get("price")


def main() -> None:
    manifest = json.loads((BASE_DIR / "manifest.json").read_text())
    fixture_to_match_id = json.loads((BASE_DIR / "fixture_to_match_id.json").read_text())

    lookup: dict[str, dict] = {}
    skipped_no_file = 0
    for fixture_id, match_id in fixture_to_match_id.items():
        league = manifest[fixture_id]["league"]
        path = BASE_DIR / league / f"{fixture_id}.json"
        if not path.exists():
            skipped_no_file += 1
            continue
        data = json.loads(path.read_text())
        outcomes_by_market = data.get("bookmakers", {}).get("pinnacle", {}).get("markets", {})

        btts_outcomes = outcomes_by_market.get(BTTS_MARKET_ID, {}).get("outcomes", {})
        btts_yes = _last_price(btts_outcomes, BTTS_OUTCOME_YES)
        btts_no = _last_price(btts_outcomes, BTTS_OUTCOME_NO)

        corners_outcomes = outcomes_by_market.get(CORNERS_MARKET_ID, {}).get("outcomes", {})
        corners_over = _last_price(corners_outcomes, CORNERS_OUTCOME_OVER)
        corners_under = _last_price(corners_outcomes, CORNERS_OUTCOME_UNDER)

        entry: dict = {}
        if btts_yes is not None and btts_no is not None:
            entry["btts_odds"] = {"yes": btts_yes, "no": btts_no}
        if corners_over is not None and corners_under is not None:
            entry[f"corners_{CORNERS_LINE}_odds"] = {"over": corners_over, "under": corners_under}
        if entry:
            lookup[match_id] = entry

    OUT_PATH.write_text(json.dumps(lookup, indent=2))
    with_btts = sum(1 for v in lookup.values() if "btts_odds" in v)
    with_corners = sum(1 for v in lookup.values() if f"corners_{CORNERS_LINE}_odds" in v)
    print(f"Matches with any odds: {len(lookup)}")
    print(f"  with btts_odds: {with_btts}")
    print(f"  with corners_{CORNERS_LINE}_odds: {with_corners}")
    print(f"Skipped (no saved file): {skipped_no_file}")
    print(f"Wrote {OUT_PATH}")


if __name__ == "__main__":
    main()
