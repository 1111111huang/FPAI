"""W210: per-user accounts for multi-user bet tracking. A user is created
lazily on first sign-in (get_or_create) -- Auth.js's own signIn allowlist
callback (frontend) is what actually restricts who can reach this point at
all; by the time FastAPI ever sees an email here, Google + the allowlist
have already vouched for it. Same SQLite-per-concern pattern as
bet_tracker.py/recommendation_cache.py (W163: repo-root data/, not
app/data/)."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
import sqlite3

DEFAULT_DB_PATH = Path(__file__).parent.parent.parent / "data" / "users.db"


@dataclass(frozen=True)
class User:
    id: int
    email: str
    created_at: str


class UserStore:
    def __init__(self, db_path: str | Path = DEFAULT_DB_PATH) -> None:
        self._db_path = Path(db_path)
        self._db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_schema()

    def _connect(self) -> sqlite3.Connection:
        return sqlite3.connect(self._db_path)

    def _init_schema(self) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS users (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    email TEXT NOT NULL UNIQUE,
                    created_at TEXT NOT NULL
                )
                """
            )

    @staticmethod
    def _find_by_email(conn: sqlite3.Connection, normalized: str) -> User | None:
        row = conn.execute(
            "SELECT id, email, created_at FROM users WHERE email = ?", (normalized,)
        ).fetchone()
        return User(id=row[0], email=row[1], created_at=row[2]) if row else None

    def get_or_create(self, email: str) -> User:
        normalized = email.strip().lower()
        created_at = datetime.now(timezone.utc).replace(microsecond=0).isoformat()
        with self._connect() as conn:
            # INSERT OR IGNORE + re-SELECT (rather than SELECT-then-INSERT) so a
            # concurrent insert of the same brand-new email can't raise
            # IntegrityError on the UNIQUE constraint -- whichever insert wins,
            # the SELECT reads back the row either way.
            conn.execute(
                "INSERT OR IGNORE INTO users (email, created_at) VALUES (?, ?)",
                (normalized, created_at),
            )
            return self._find_by_email(conn, normalized)  # type: ignore[return-value]

    def get_by_email(self, email: str) -> User | None:
        normalized = email.strip().lower()
        with self._connect() as conn:
            return self._find_by_email(conn, normalized)
