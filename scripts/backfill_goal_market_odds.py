"""BUG-096 follow-up: fill missing goal-market odds on EXISTING raw_matches rows.

Why this is needed: CSVLoader inserts with INSERT OR IGNORE and skips any
CSV whose hash it has already processed. A raw_matches row stored once
without over/under 2.5 or Asian handicap odds therefore never gets them,
even after the CSV on disk has them. Nothing else in the codebase updates
a raw_matches row in place. In production, every E0/SP1 row appears to
lack ah_line (see scripts/check_market_feature_coverage.py), which is what
zeroed MKT_LAMBDA_* for those leagues.

What it does, and does NOT do:
- Parses the CSVs already on disk with the real CSVLoader.process_v1_csv,
  into a throwaway staging DuckDB file, so column mapping and match_id
  generation are exactly what normal ingest produces.
- For rows whose match_id exists in both, fills ONLY columns that are NULL
  in the real table (COALESCE). It never overwrites a value, never inserts
  new rows and never deletes anything.
- Dry run by default: prints what it would fill. Pass --apply to write.

Do NOT use `python main.py ingest --force` for this in production: that
deletes all of raw_matches and feature_store and rebuilds only from the
CSVs currently on disk, which in production covers only recent seasons.

After --apply, rebuild feature_store so MKT_* picks up the filled columns:
    python main.py ingest
(W252: a full feature rebuild has been OOM-killed on Railway before --
watch that run.)

Usage:
    python scripts/backfill_goal_market_odds.py                    # dry run, all 5 leagues
    python scripts/backfill_goal_market_odds.py --leagues E0 SP1
    python scripts/backfill_goal_market_odds.py --leagues E0 SP1 --apply
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys
import tempfile

import yaml

sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.ingestion.football_data.loader import CSVLoader  # noqa: E402
from src.utils.db_manager import DuckDBManager  # noqa: E402

DEFAULT_LEAGUES = ("E0", "SP1", "I1", "D1", "F1")
BACKFILL_COLUMNS = ("over25_odds", "under25_odds", "ah_line", "ah_home_odds", "ah_away_odds")


def _write_staging_config(config_path: Path, staging_dir: Path) -> Path:
    """A copy of config.yaml whose database_path points at a throwaway
    staging file. raw_data_dir is made absolute so the staging loader reads
    the same CSVs the real one does."""
    settings = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    paths = settings.setdefault("paths", {})
    raw_dir = paths.get("raw_data_dir")
    if raw_dir:
        paths["raw_data_dir"] = str(Path(raw_dir).resolve())
    paths["database_path"] = str(staging_dir / "staging.duckdb")
    staging_config = staging_dir / "config.yaml"
    staging_config.write_text(yaml.safe_dump(settings), encoding="utf-8")
    return staging_config


def stage_csvs(config_path: Path, staging_dir: Path, leagues: tuple[str, ...]) -> tuple[Path, int]:
    """Parse every on-disk CSV for `leagues` into a staging DB with the real
    loader. Returns (staging DB path, number of CSV files parsed)."""
    loader = CSVLoader(config_path=str(_write_staging_config(config_path, staging_dir)))
    files = 0
    for league in leagues:
        for csv_path in sorted(loader.raw_data_dir.glob(f"{league}_*.csv")):
            loader.process_v1_csv(file_path=str(csv_path), league_code=league, overwrite=False)
            files += 1
    return loader.db_manager.db_path, files


def _null_fill_counts(conn, leagues: tuple[str, ...]) -> dict[str, dict[str, int]]:
    """{league: {column: rows NULL in the real table but present in staging}}."""
    placeholders = ", ".join("?" for _ in leagues)
    selects = ", ".join(
        f'SUM(CASE WHEN r."{c}" IS NULL AND s."{c}" IS NOT NULL THEN 1 ELSE 0 END) AS "{c}"'
        for c in BACKFILL_COLUMNS
    )
    rows = conn.execute(
        f"""
        SELECT r.league, COUNT(*) AS matched, {selects}
        FROM raw_matches r
        JOIN staging.raw_matches s ON r.match_id = s.match_id
        WHERE r.league IN ({placeholders})
        GROUP BY r.league
        ORDER BY r.league
        """,
        list(leagues),
    ).fetchall()
    return {
        league: {"matched": int(matched), **{c: int(v or 0) for c, v in zip(BACKFILL_COLUMNS, counts)}}
        for league, matched, *counts in rows
    }


def backfill(db_manager: DuckDBManager, staging_db: Path, leagues: tuple[str, ...], apply: bool) -> dict:
    """Returns the fill counts (computed before any write). Writes only when
    `apply` is True, and only into columns that are NULL."""
    with db_manager.connection() as conn:
        conn.execute(f"ATTACH '{staging_db}' AS staging (READ_ONLY)")
        try:
            counts = _null_fill_counts(conn, leagues)
            if apply and any(sum(v for k, v in c.items() if k != "matched") for c in counts.values()):
                placeholders = ", ".join("?" for _ in leagues)
                assignments = ", ".join(f'"{c}" = COALESCE(raw_matches."{c}", s."{c}")' for c in BACKFILL_COLUMNS)
                any_null = " OR ".join(f'raw_matches."{c}" IS NULL' for c in BACKFILL_COLUMNS)
                conn.execute(
                    f"""
                    UPDATE raw_matches SET {assignments}
                    FROM staging.raw_matches s
                    WHERE raw_matches.match_id = s.match_id
                      AND raw_matches.league IN ({placeholders})
                      AND ({any_null})
                    """,
                    list(leagues),
                )
        finally:
            conn.execute("DETACH staging")
    return counts


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--leagues", nargs="+", default=list(DEFAULT_LEAGUES))
    parser.add_argument("--apply", action="store_true", help="Write the fills. Without it, dry run only.")
    parser.add_argument("--config", default="config.yaml")
    args = parser.parse_args(argv)
    leagues = tuple(league.upper() for league in args.leagues)

    with tempfile.TemporaryDirectory(prefix="bug096_backfill_") as tmp:
        staging_db, files = stage_csvs(Path(args.config), Path(tmp), leagues)
        if files == 0:
            print(f"No CSV files on disk for {', '.join(leagues)} -- nothing to backfill from.")
            return 1
        counts = backfill(DuckDBManager(config_path=args.config), staging_db, leagues, apply=args.apply)

    print(f"Parsed {files} CSV file(s). Rows matched by match_id, and NULLs that CSV data can fill:")
    print("league  matched  " + "  ".join(f"{c:>13}" for c in BACKFILL_COLUMNS))
    for league in leagues:
        c = counts.get(league, {"matched": 0})
        print(f"{league:<6}  {c['matched']:>7}  " + "  ".join(f"{c.get(col, 0):>13}" for col in BACKFILL_COLUMNS))

    total = sum(v for c in counts.values() for k, v in c.items() if k != "matched")
    if total == 0:
        print("\nNothing to fill.")
    elif args.apply:
        print(f"\nFilled {total} NULL value(s). Now rebuild feature_store: python main.py ingest")
    else:
        print(f"\nDry run: {total} NULL value(s) would be filled. Re-run with --apply to write them.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
