"""W210: one-time migration -- every user_bets row created before multi-user
existed has user_id=NULL. Assigns them all to a single owner account.
Idempotent: only touches rows still NULL, so running it twice (or against a
DB that's already been migrated) is a safe no-op on the second run."""

from __future__ import annotations

import argparse
import sqlite3
from pathlib import Path

from app.backend.bet_tracker import DEFAULT_DB_PATH as DEFAULT_BETS_DB_PATH
from app.backend.users import DEFAULT_DB_PATH as DEFAULT_USERS_DB_PATH
from app.backend.users import UserStore


def migrate_bets_to_owner(
    owner_email: str,
    bets_db_path: str | Path = DEFAULT_BETS_DB_PATH,
    users_db_path: str | Path = DEFAULT_USERS_DB_PATH,
) -> int:
    """Returns the number of rows migrated."""
    owner = UserStore(db_path=users_db_path).get_or_create(owner_email)
    conn = sqlite3.connect(bets_db_path)
    try:
        cursor = conn.execute(
            "UPDATE user_bets SET user_id = ? WHERE user_id IS NULL", (owner.id,)
        )
        conn.commit()
        return cursor.rowcount
    finally:
        conn.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--owner-email", required=True, help="Email to assign all existing (user_id=NULL) bets to")
    args = parser.parse_args()
    count = migrate_bets_to_owner(owner_email=args.owner_email)
    print(f"Migrated {count} bet(s) to {args.owner_email}")
