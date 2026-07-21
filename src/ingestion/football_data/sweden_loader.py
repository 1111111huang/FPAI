"""Normalize the football-data.co.uk Sweden CSV and upsert into raw_matches (US#125).

`fetch_sweden_csv()` (`sweden_fetcher.py`, US#124) is fetch-and-parse only --
it returns the Sweden "New Leagues" CSV's own 25 columns, unrenamed. This
module is the mapping step: it takes that DataFrame and writes rows into the
same `raw_matches` table EPL data lives in (`loader.py::process_v1_csv` is
the primary pattern reference this follows), tagging every row `league='SWE'`.

Column mapping
--------------
Home/Away/Date  -> home_team/away_team/date directly. `Time` is read
    defensively (may be blank/missing for historical rows) but deliberately
    NOT combined into `date` -- `loader.py`'s EPL path ignores the `Time`
    column entirely even though modern E0 CSVs carry one (it's never in the
    rename map, the required-columns set, or the final column selection), so
    dropping it here matches existing behavior rather than introducing a
    Sweden-only sub-day precision difference in a column (`raw_matches.date`,
    TIMESTAMP) shared with EPL rows.
HG/AG           -> fthg/ftag.
AvgCH/AvgCD/AvgCA (closing average across bookmakers) -> avgh/avgd/avga,
    falling back to B365CH/B365CD/B365CA *per field* when AvgC* is blank --
    the same fallback *pattern* BUG-009/US#56 established for pre-2020 EPL
    rows (there, `raw_df["avgh"] = raw_df["avgh"].fillna(raw_df["odds_h"])`
    at feature-computation time in `feature_factory.py`, since pre-2020 EPL
    CSVs don't ship AvgH/AvgD/AvgA at all). Sweden has no separate "required"
    odds column distinct from the closing average the way EPL has B365H/D/A
    as its always-present base, so the same AvgC*-with-B365C*-fallback value
    is written to BOTH `avgh/avgd/avga` *and* `odds_h/odds_d/odds_a` -- it's
    the best (only) 1X2 market signal this source provides, and populating
    `odds_h/d/a` keeps Sweden rows usable by the odds-driven strategy/backtest
    modules (`strategy_engine.py`, `backtester.py`) that key off `odds_h`
    directly, the same way EPL rows are. This goes slightly beyond the
    story's literal text (which only spells out the avgh/avgd/avga mapping)
    -- a deliberate judgment call, not an oversight.

Everything else this source doesn't provide -- hs/as/hst/ast/hc/ac/hy/ay/hr/
ar, xg_h/xg_a/xga_h/xga_a, over25_odds/under25_odds, ah_line/ah_home_odds/
ah_away_odds -- is left NULL. Confirmed by reading `loader.py`'s
`_create_raw_matches_table` DDL: none of those columns carry a NOT NULL
constraint, and pre-2020 EPL rows already ship with `avgh/avgd/avga` NULL via
the exact same "column/value absent -> None" path -- no `raw_matches`
migration needed for Sweden.

`tier` (an EPL-division-specific 1-4 concept from `match_schema.py`, used
nowhere downstream of ingestion -- not read anywhere in `feature_factory.py`)
is left NULL for Sweden rather than defaulting through
`map_league_code_to_tier`'s unknown-code fallback (which would silently
store `4`, a real English "League Two" tier value that has no meaning for
Sweden).

Season: Sweden's `Season` is a single calendar year (e.g. 2026) and is not
stored in `raw_matches` at all (there is no `season` column) -- so the EPL
`year1year2` convention was never actually load-bearing here; nothing reads
a season value off `raw_matches`.

Team names: inserted as the CSV's own spelling (e.g. "Malmo FF"), run only
through `standardize_team_name` (whitespace normalization + a small
EPL-specific alias map that is a no-op for Swedish club names) -- the exact,
and only, normalization `loader.py`'s EPL path applies to `home_team`/
`away_team`. `TeamNameMapper`/`config/team_mapping.json` fuzzy-matching is
deliberately NOT invoked here -- that's US#126.

Idempotency: matches `loader.py::process_v1_csv`'s convention exactly --
`INSERT OR IGNORE` by default keyed on `match_id` (the primary key), so
calling this repeatedly (e.g. a weekly `refresh-data` run re-fetching the
whole Sweden history file) never duplicates rows or errors on rows already
present. `overwrite=True` switches to `INSERT OR REPLACE` for a forced
re-ingestion, the same escape hatch `process_v1_csv` offers.
"""

from __future__ import annotations

import pandas as pd

from src.ingestion.football_data.loader import CSVLoader
from src.utils.db_manager import DuckDBManager
from src.utils.helpers import generate_match_id, standardize_team_name
from src.utils.logger import get_logger

LOGGER = get_logger(__name__)

LEAGUE_CODE = "SWE"

# Columns this source has no data for at all -- always NULL for Sweden rows.
_NULL_COLUMNS: list[str] = [
    "hs", "as", "hst", "ast", "hc", "ac", "hy", "ay", "hr", "ar",
    "xg_h", "xg_a", "xga_h", "xga_a",
    "over25_odds", "under25_odds", "ah_line", "ah_home_odds", "ah_away_odds",
]

_INSERT_COLUMNS: list[str] = [
    "match_id", "league", "tier", "date", "home_team", "away_team",
    "fthg", "ftag", "odds_h", "odds_d", "odds_a",
    "hs", "as", "hst", "ast", "hc", "ac", "hy", "ay", "hr", "ar",
    "avgh", "avgd", "avga", "xg_h", "xg_a", "xga_h", "xga_a",
    "over25_odds", "under25_odds", "ah_line", "ah_home_odds", "ah_away_odds",
]


def upsert_sweden_matches(
    df: pd.DataFrame,
    db_manager: DuckDBManager,
    overwrite: bool = False,
) -> dict[str, int]:
    """Normalize a Sweden CSV DataFrame (`fetch_sweden_csv`'s output shape)
    and upsert it into `raw_matches`, tagged `league='SWE'`.

    Rows missing a parseable date, home/away team name, or final score
    (HG/AG -- e.g. a not-yet-played fixture row) are skipped, not errored on,
    the same "skip rather than crash" posture `process_v1_csv` takes for
    rows with missing critical odds.

    Returns a dict with 'rows_in' (rows received), 'skipped' (rows dropped
    for missing required fields), and 'upserted' (net new rows actually
    written -- mirrors `process_v1_csv`'s `added_matches` return convention,
    computed from the before/after `raw_matches` row count).
    """
    if df.empty:
        return {"rows_in": 0, "skipped": 0, "upserted": 0}

    working = df.copy()
    rows_in = len(working)

    required_columns = ["Date", "Home", "Away", "HG", "AG"]
    missing_required = [c for c in required_columns if c not in working.columns]
    if missing_required:
        raise ValueError(f"Sweden dataframe missing required columns: {missing_required}")

    for optional_col in ["AvgCH", "AvgCD", "AvgCA", "B365CH", "B365CD", "B365CA"]:
        if optional_col not in working.columns:
            working[optional_col] = None

    working["_date"] = pd.to_datetime(working["Date"], dayfirst=True, errors="coerce")
    working["_home_team"] = working["Home"].astype(str).map(standardize_team_name)
    working["_away_team"] = working["Away"].astype(str).map(standardize_team_name)
    working["_fthg"] = pd.to_numeric(working["HG"], errors="coerce")
    working["_ftag"] = pd.to_numeric(working["AG"], errors="coerce")

    # Row-level fallback (BUG-009/US#56 pattern): AvgC* preferred, B365C* per-field fallback.
    working["_avgh"] = pd.to_numeric(working["AvgCH"], errors="coerce").fillna(
        pd.to_numeric(working["B365CH"], errors="coerce")
    )
    working["_avgd"] = pd.to_numeric(working["AvgCD"], errors="coerce").fillna(
        pd.to_numeric(working["B365CD"], errors="coerce")
    )
    working["_avga"] = pd.to_numeric(working["AvgCA"], errors="coerce").fillna(
        pd.to_numeric(working["B365CA"], errors="coerce")
    )

    home_present = working["Home"].notna() & working["Home"].astype(str).str.strip().ne("")
    away_present = working["Away"].notna() & working["Away"].astype(str).str.strip().ne("")
    valid_mask = (
        working["_date"].notna()
        & home_present
        & away_present
        & working["_fthg"].notna()
        & working["_ftag"].notna()
    )
    skipped = int((~valid_mask).sum())
    if skipped:
        LOGGER.warning(
            "Skipping %d Sweden rows missing a parseable date/team/final score "
            "(e.g. unplayed fixtures).",
            skipped,
        )
    working = working.loc[valid_mask].copy()

    def _f(value: object) -> float | None:
        return float(value) if pd.notna(value) else None

    records: list[tuple] = []
    for row in working.to_dict(orient="records"):
        match_date = row["_date"].to_pydatetime()
        home_team = row["_home_team"]
        away_team = row["_away_team"]
        match_id = generate_match_id(
            date=match_date.date().isoformat(),
            home_team=home_team,
            away_team=away_team,
            league=LEAGUE_CODE,
        )
        avgh, avgd, avga = _f(row["_avgh"]), _f(row["_avgd"]), _f(row["_avga"])

        record: dict[str, object] = {
            "match_id": match_id,
            "league": LEAGUE_CODE,
            "tier": None,
            "date": match_date,
            "home_team": home_team,
            "away_team": away_team,
            "fthg": int(row["_fthg"]),
            "ftag": int(row["_ftag"]),
            "odds_h": avgh,
            "odds_d": avgd,
            "odds_a": avga,
            "avgh": avgh,
            "avgd": avgd,
            "avga": avga,
        }
        for null_col in _NULL_COLUMNS:
            record[null_col] = None
        records.append(tuple(record[col] for col in _INSERT_COLUMNS))

    upserted = 0
    with db_manager.connection() as conn:
        CSVLoader._create_raw_matches_table(conn)
        before_row = conn.execute("SELECT COUNT(*) FROM raw_matches").fetchone()
        before_count = int(before_row[0]) if before_row is not None else 0

        if records:
            insert_clause = "INSERT OR REPLACE" if overwrite else "INSERT OR IGNORE"
            columns_sql = ", ".join(f'"{col}"' if col == "as" else col for col in _INSERT_COLUMNS)
            placeholders = ", ".join(["?"] * len(_INSERT_COLUMNS))
            conn.executemany(
                f"{insert_clause} INTO raw_matches ({columns_sql}) VALUES ({placeholders})",
                records,
            )

        after_row = conn.execute("SELECT COUNT(*) FROM raw_matches").fetchone()
        after_count = int(after_row[0]) if after_row is not None else before_count
        upserted = max(0, after_count - before_count)

    LOGGER.info(
        "Sweden ingestion complete | rows_in=%d | skipped=%d | new_rows=%d",
        rows_in, skipped, upserted,
    )
    return {"rows_in": rows_in, "skipped": skipped, "upserted": upserted}
