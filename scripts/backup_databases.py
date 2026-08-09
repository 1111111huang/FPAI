#!/usr/bin/env python3
"""US#157: on-demand backup/restore for FPAI's production databases.

Direct gap found while validating scheduled-refresh production-readiness
(Phase 23/24): `.gitignore` already anticipated this (`data/*.db.*_backup`,
`data/*.db.bak`) but no script ever wrote one -- `data/fpai_core.db` (the
ML-engine's full match/feature history, 168MB+) and the app's
`app/data/recommendation_cache.db` / `app/data/user_bets.db` (recommendation
history and a user's actual bet-tracking records) each exist as exactly one
file, with no backup anywhere. Losing or corrupting that file loses the data
permanently -- no restore path exists today.

Two different DB engines, two different safe-copy mechanisms:
  - DuckDB (`fpai_core.db`): enforces a real exclusive per-process file lock
    (confirmed empirically -- W93/W95, `src/utils/db_manager.py`). Opening
    our own read-only connection via `DuckDBManager.connection()` (which
    already retries if a writer briefly holds the lock, W95) guarantees no
    concurrent writer can be mid-write while we copy the file -- any writer
    that tries will retry-and-wait behind us instead of corrupting anything.
  - SQLite (`recommendation_cache.db`, `user_bets.db`, `job_runs.db`):
    stdlib's `sqlite3.Connection.backup()` is the correct online-backup API
    -- safe to call while another connection is writing, unlike a naive
    file copy which could copy a torn write mid-transaction.

Backup filenames follow the `.gitignore`-anticipated convention:
`<original_name>.<UTC timestamp>_backup`.

# ponytail: no retention/pruning -- backups accumulate forever. Add a
# --keep-last-N prune step if disk usage on the backup destination ever
# becomes a real problem; not speculative work today.

Usage:
    python scripts/backup_databases.py                    # back up all known DBs
    python scripts/backup_databases.py --dest /some/dir    # custom backup dir
    python scripts/backup_databases.py --restore <backup_file> <target_path>
    python scripts/backup_databases.py --restore <backup_file> <target_path> --force
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
from pathlib import Path
import shutil
import sqlite3
import sys

import duckdb

sys.path.append(str(Path(__file__).resolve().parent.parent))

from src.utils.db_manager import DuckDBManager
from src.utils.logger import get_logger

LOGGER = get_logger(__name__)

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_SQLITE_DBS = (
    ROOT / "app" / "data" / "recommendation_cache.db",
    ROOT / "app" / "data" / "user_bets.db",
    ROOT / "app" / "data" / "job_runs.db",
)


def _timestamped_backup_path(db_path: Path, dest_dir: Path | None = None) -> Path:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    name = f"{db_path.name}.{stamp}_backup"
    return (dest_dir / name) if dest_dir else (db_path.parent / name)


def backup_duckdb(config_path: str = "config.yaml", dest_dir: Path | None = None) -> Path:
    """Safely copy the DuckDB file, holding a read-only connection (with
    W95's built-in retry) for the duration so no concurrent writer can be
    mid-write during the copy."""
    manager = DuckDBManager(config_path=config_path)
    dest = _timestamped_backup_path(manager.db_path, dest_dir)
    dest.parent.mkdir(parents=True, exist_ok=True)
    with manager.connection(read_only=True):
        shutil.copy2(manager.db_path, dest)
    LOGGER.info("Backed up DuckDB %s -> %s", manager.db_path, dest)
    return dest


def backup_sqlite(db_path: Path, dest_dir: Path | None = None) -> Path:
    """Online-backup a SQLite db via the stdlib backup API -- safe even if
    another connection is concurrently writing."""
    dest = _timestamped_backup_path(db_path, dest_dir)
    dest.parent.mkdir(parents=True, exist_ok=True)
    source = sqlite3.connect(db_path)
    try:
        target = sqlite3.connect(dest)
        try:
            source.backup(target)
        finally:
            target.close()
    finally:
        source.close()
    LOGGER.info("Backed up SQLite %s -> %s", db_path, dest)
    return dest


def restore(backup_path: Path, target_path: Path, force: bool = False) -> Path:
    """Copy a backup file back to its original (or a chosen) location.
    Refuses to clobber an existing file unless --force, since restoring is
    itself a destructive operation on whatever's currently there."""
    if target_path.exists() and not force:
        raise FileExistsError(
            f"{target_path} already exists -- pass --force to overwrite it with the backup"
        )
    target_path.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(backup_path, target_path)
    LOGGER.info("Restored %s -> %s", backup_path, target_path)
    return target_path


def _run_backup_all(config_path: str, dest: Path | None) -> list[Path]:
    written: list[Path] = []
    try:
        written.append(backup_duckdb(config_path=config_path, dest_dir=dest))
    except (duckdb.IOException, FileNotFoundError) as exc:
        LOGGER.warning("Skipped DuckDB backup: %s", exc)
    for sqlite_path in DEFAULT_SQLITE_DBS:
        if not sqlite_path.exists():
            LOGGER.warning("Skipped SQLite backup, file not found: %s", sqlite_path)
            continue
        written.append(backup_sqlite(sqlite_path, dest_dir=dest))
    return written


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="config.yaml", help="Path to the ML-engine config.yaml")
    parser.add_argument("--dest", type=Path, default=None, help="Backup destination dir (default: alongside each source file)")
    parser.add_argument("--restore", nargs=2, metavar=("BACKUP_FILE", "TARGET_PATH"))
    parser.add_argument("--force", action="store_true", help="With --restore, overwrite an existing target")
    args = parser.parse_args()

    if args.restore:
        backup_file, target_path = args.restore
        restore(Path(backup_file), Path(target_path), force=args.force)
        return

    written = _run_backup_all(args.config, args.dest)
    if not written:
        print("No databases found to back up.")
        return
    print(f"Backed up {len(written)} database(s):")
    for path in written:
        print(f"  {path}")


if __name__ == "__main__":
    main()
