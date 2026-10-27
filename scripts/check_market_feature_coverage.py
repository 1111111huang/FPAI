"""BUG-096: per-league coverage check for the goal-market inputs.

Found 2026-10-26: every live E0 and SP1 forecast scored with
MKT_LAMBDA_HOME/MKT_LAMBDA_AWAY/MKT_POISSON_BTTS_PROB/MKT_LAMBDA_AH_DIFF
equal to exactly 0.0, while D1/I1/F1 got a league-average value. The
forecast service fills a missing market feature from the league's own
feature_store mean (src/features/league_prior.py) and falls back to a flat
0.0 when that mean is NULL -- so a 0.0 in E0/SP1 means production's
feature_store has no non-null MKT_LAMBDA_* rows for those two leagues,
most likely because raw_matches has no ah_line there or feature_store was
never regenerated after it did (BUG-062: production's feature_store is a
separate file on Railway's volume, never touched by a code deploy).

This script answers that directly, on whichever DB config.yaml points at.
Run it against production's DB:

    python scripts/check_market_feature_coverage.py
    python scripts/check_market_feature_coverage.py --since 2024-07-01

Reading the output:
- raw_matches ah_line/over25 coverage low for a league -> those rows were
  stored without the columns. Re-running refresh-data does NOT fix this:
  CSVLoader skips already-processed files and inserts with INSERT OR
  IGNORE, so existing rows are never updated. Use
  `python scripts/backfill_goal_market_odds.py --leagues XX` (dry run
  first, then --apply), then rebuild feature_store.
- raw_matches coverage fine but feature_store MKT_LAMBDA_HOME coverage low
  -> feature_store is stale; regenerate it with `python main.py ingest`
  (same step BUG-075's fix used).
- Never use `python main.py ingest --force` in production for either case:
  it deletes all of raw_matches and rebuilds only from CSVs on disk.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.utils.db_manager import DuckDBManager  # noqa: E402

_RAW_COLUMNS = ("over25_odds", "ah_line", "ah_home_odds", "ah_away_odds")
_FEATURE_COLUMNS = ("MKT_IMPLIED_OVER25", "MKT_AH_LINE", "MKT_LAMBDA_HOME", "MKT_LAMBDA_AWAY")

# Below this share of non-null rows, a league is flagged.
_WARN_COVERAGE = 0.8


def _coverage_query(table_sql: str, columns: tuple[str, ...], since: str | None) -> tuple[str, list]:
    selects = ", ".join(f'AVG(CASE WHEN "{c}" IS NULL THEN 0.0 ELSE 1.0 END) AS "{c}"' for c in columns)
    where = "WHERE rm.date >= ?" if since else ""
    params = [since] if since else []
    sql = f"""
        SELECT rm.league, COUNT(*) AS n_rows, {selects}
        FROM {table_sql}
        {where}
        GROUP BY rm.league
        ORDER BY rm.league
    """
    return sql, params


def coverage_by_league(db_manager: DuckDBManager, since: str | None = None) -> dict[str, dict]:
    """{league: {"raw": {...}, "features": {...}}}, each column's value the
    share of rows (0..1) where it is non-null. "features" is missing for a
    league with no feature_store rows at all."""
    result: dict[str, dict] = {}
    with db_manager.connection(read_only=True) as conn:
        sql, params = _coverage_query("raw_matches rm", _RAW_COLUMNS, since)
        for row in conn.execute(sql, params).fetchall():
            league, n_rows, *shares = row
            result.setdefault(league, {})["raw"] = {"n_rows": n_rows, **dict(zip(_RAW_COLUMNS, shares))}

        # Only columns this feature_store actually has -- an older schema
        # may lack some, which is itself worth reporting.
        present = {r[0] for r in conn.execute("SELECT column_name FROM information_schema.columns WHERE table_name = 'feature_store'").fetchall()}
        feature_cols = tuple(c for c in _FEATURE_COLUMNS if c in present)
        missing_cols = [c for c in _FEATURE_COLUMNS if c not in present]
        if feature_cols:
            sql, params = _coverage_query(
                "feature_store fs JOIN raw_matches rm ON fs.match_id = rm.match_id", feature_cols, since,
            )
            # The CASE expressions above reference bare column names, which
            # resolve to feature_store's since raw_matches has no MKT_* columns.
            for row in conn.execute(sql, params).fetchall():
                league, n_rows, *shares = row
                result.setdefault(league, {})["features"] = {"n_rows": n_rows, **dict(zip(feature_cols, shares))}
        for league in result:
            result[league]["missing_feature_columns"] = missing_cols
    return result


def _flag(share: float | None) -> str:
    if share is None:
        return "  n/a"
    marker = "  " if share >= _WARN_COVERAGE else " !"
    return f"{share:5.0%}{marker}"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--since", help="Only count matches on/after this date (YYYY-MM-DD).")
    parser.add_argument("--config", default="config.yaml", help="Config file that points at the DB.")
    args = parser.parse_args(argv)

    coverage = coverage_by_league(DuckDBManager(config_path=args.config), since=args.since)
    if not coverage:
        print("No raw_matches rows found.")
        return 1

    header = "league  rows   " + "  ".join(f"{c:>12}" for c in _RAW_COLUMNS) + "   |  " + "  ".join(f"{c:>18}" for c in _FEATURE_COLUMNS)
    print(header)
    flagged = []
    for league, data in sorted(coverage.items()):
        raw = data.get("raw", {})
        feats = data.get("features", {})
        raw_cells = "  ".join(f"{_flag(raw.get(c)):>12}" for c in _RAW_COLUMNS)
        feat_cells = "  ".join(f"{_flag(feats.get(c)):>18}" for c in _FEATURE_COLUMNS)
        print(f"{league:<6}  {raw.get('n_rows', 0):>5}  {raw_cells}   |  {feat_cells}")
        if (raw.get("ah_line") or 0) < _WARN_COVERAGE:
            flagged.append(
                f"{league}: raw_matches.ah_line coverage {raw.get('ah_line', 0):.0%} -- fill it with "
                f"`python scripts/backfill_goal_market_odds.py --leagues {league}` (refresh-data won't update existing rows)."
            )
        elif (feats.get("MKT_LAMBDA_HOME") or 0) < _WARN_COVERAGE:
            flagged.append(
                f"{league}: raw ah_line is fine but feature_store MKT_LAMBDA_HOME coverage is "
                f"{feats.get('MKT_LAMBDA_HOME', 0):.0%} -- regenerate feature_store (`python main.py ingest`)."
            )
    missing = next(iter(coverage.values())).get("missing_feature_columns") or []
    if missing:
        flagged.append(f"feature_store has no column(s) {missing} -- regenerate feature_store.")

    print()
    if flagged:
        print("Problems (flag '!' = below {:.0%} coverage):".format(_WARN_COVERAGE))
        for line in flagged:
            print(f"  - {line}")
        return 2
    print("All leagues have goal-market coverage above {:.0%}.".format(_WARN_COVERAGE))
    return 0


if __name__ == "__main__":
    sys.exit(main())
