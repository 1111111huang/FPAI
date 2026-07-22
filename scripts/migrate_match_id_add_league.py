#!/usr/bin/env python3
"""US#140: one-time migration to recompute match_id under the new
league-inclusive scheme.

``generate_match_id`` used to hash only date|home|away, so two different
competitions with a match on the same date between similarly-named teams
could collide onto the same match_id. It now also hashes league, which means
*every* previously-ingested match_id changes value -- this script remaps the
old ids to the new ones in place.

Why an in-place remap instead of a full re-ingest (`python main.py ingest
--force`)? A full re-ingest looked simpler at first (there's direct
precedent for it -- see FRAI_TECHSPEC.md Section 27.3's migration note: CSV
re-ingestion is idempotent because raw_matches inserts are keyed by
match_id). But `ingest --force` only rebuilds raw_matches and feature_store
from the CSVs on disk; it does NOT repopulate raw_player_match_stats, which
also has a match_id column and is 100% joinable against raw_matches today
(FotMob squad-stat rows, populated by a separate `fetch-fotmob` pipeline
that is slow/rate-limited and reflects point-in-time historical rosters that
aren't trivially reproducible on demand). A full re-ingest would silently
orphan every one of those rows. Remapping match_id in place across all three
tables preserves everything.

match_lineups is NOT touched here: it keys off `fotmob_match_id`, FotMob's
own id, not our generate_match_id output, so it is unaffected.

Usage:
    python scripts/migrate_match_id_add_league.py --dry-run
    python scripts/migrate_match_id_add_league.py
    python scripts/migrate_match_id_add_league.py --db-path /path/to/other.db
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import duckdb

sys.path.append(str(Path(__file__).resolve().parent.parent))

from src.utils.helpers import generate_match_id

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_DB_PATH = ROOT / "data" / "fpai_core.db"

# Tables (besides raw_matches itself) that store match_id values and must be
# remapped in lockstep so nothing is left pointing at a stale id.
DEPENDENT_TABLES = ("feature_store", "raw_player_match_stats")


def _existing_tables(conn: duckdb.DuckDBPyConnection) -> set[str]:
    return {row[0] for row in conn.execute("SHOW TABLES").fetchall()}


def build_id_mapping(conn: duckdb.DuckDBPyConnection) -> dict[str, str]:
    """Compute old_match_id -> new_match_id for every row in raw_matches."""
    rows = conn.execute(
        "SELECT match_id, date, home_team, away_team, league FROM raw_matches"
    ).fetchall()
    mapping: dict[str, str] = {}
    for old_id, match_date, home_team, away_team, league in rows:
        date_str = match_date.date().isoformat() if hasattr(match_date, "date") else str(match_date)
        new_id = generate_match_id(
            date=date_str, home_team=home_team, away_team=away_team, league=league
        )
        mapping[old_id] = new_id
    return mapping


def migrate(conn: duckdb.DuckDBPyConnection, dry_run: bool = False) -> dict:
    """Remap match_id in raw_matches and dependent tables to the new,
    league-inclusive scheme. Returns a summary dict suitable for logging/
    verification (row counts before/after, and an orphan check)."""
    existing_tables = _existing_tables(conn)
    if "raw_matches" not in existing_tables:
        return {"raw_matches_before": 0, "raw_matches_after": 0, "changed": 0, "unchanged": 0}

    before_counts = {"raw_matches": conn.execute("SELECT COUNT(*) FROM raw_matches").fetchone()[0]}
    for table in DEPENDENT_TABLES:
        if table in existing_tables:
            before_counts[table] = conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]

    mapping = build_id_mapping(conn)

    new_ids = list(mapping.values())
    if len(set(new_ids)) != len(new_ids):
        raise RuntimeError(
            "Migration aborted: recomputed match_ids collide with each other -- "
            "refusing to write anything. Investigate before re-running."
        )

    changed = {old: new for old, new in mapping.items() if old != new}

    if dry_run:
        return {
            "raw_matches_before": before_counts["raw_matches"],
            "to_change": len(changed),
            "already_current": len(mapping) - len(changed),
            "dependent_row_counts": {t: before_counts.get(t, 0) for t in DEPENDENT_TABLES},
        }

    conn.execute("BEGIN TRANSACTION")
    try:
        conn.execute("CREATE TEMP TABLE _match_id_map (old_id TEXT, new_id TEXT)")
        if changed:
            conn.executemany("INSERT INTO _match_id_map VALUES (?, ?)", list(changed.items()))
            conn.execute(
                """
                UPDATE raw_matches
                SET match_id = m.new_id
                FROM _match_id_map m
                WHERE raw_matches.match_id = m.old_id
                """
            )
            for table in DEPENDENT_TABLES:
                if table in existing_tables:
                    conn.execute(
                        f"""
                        UPDATE {table}
                        SET match_id = m.new_id
                        FROM _match_id_map m
                        WHERE {table}.match_id = m.old_id
                        """
                    )
        conn.execute("DROP TABLE _match_id_map")
        conn.execute("COMMIT")
    except Exception:
        conn.execute("ROLLBACK")
        raise

    after_counts = {"raw_matches": conn.execute("SELECT COUNT(*) FROM raw_matches").fetchone()[0]}
    orphans = {}
    for table in DEPENDENT_TABLES:
        if table in existing_tables:
            after_counts[table] = conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
            orphans[table] = conn.execute(
                f"""
                SELECT COUNT(*) FROM {table} t
                LEFT JOIN raw_matches r ON t.match_id = r.match_id
                WHERE r.match_id IS NULL
                """
            ).fetchone()[0]

    return {
        "raw_matches_before": before_counts["raw_matches"],
        "raw_matches_after": after_counts["raw_matches"],
        "changed": len(changed),
        "already_current": len(mapping) - len(changed),
        "dependent_row_counts_before": {t: before_counts.get(t, 0) for t in DEPENDENT_TABLES},
        "dependent_row_counts_after": {t: after_counts.get(t, 0) for t in DEPENDENT_TABLES},
        "orphaned_dependent_rows": orphans,
    }


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--db-path", type=Path, default=DEFAULT_DB_PATH, help="DuckDB file to migrate in place.")
    parser.add_argument("--dry-run", action="store_true", help="Report what would change without writing.")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    if not args.db_path.exists():
        print(f"No database found at {args.db_path} -- nothing to migrate.")
        return

    conn = duckdb.connect(str(args.db_path))
    try:
        summary = migrate(conn, dry_run=args.dry_run)
    finally:
        conn.close()

    label = "=== Dry run: match_id migration (US#140) ===" if args.dry_run else "=== match_id migration complete (US#140) ==="
    print(label)
    for key, value in summary.items():
        print(f"  {key}: {value}")


if __name__ == "__main__":
    main()
