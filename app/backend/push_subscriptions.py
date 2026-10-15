"""Web Push subscriptions (one row per installed PWA instance a user has
granted notification permission to) -- same SQLite-per-concern pattern as
users.py/bet_tracker.py. A user may have multiple rows (phone + desktop);
`endpoint` is the natural unique key Web Push already assigns per device."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
import sqlite3

DEFAULT_DB_PATH = Path(__file__).parent.parent.parent / "data" / "push_subscriptions.db"


@dataclass(frozen=True)
class PushSubscription:
    id: int
    user_id: int
    endpoint: str
    p256dh: str
    auth: str
    created_at: str


class PushSubscriptionStore:
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
                CREATE TABLE IF NOT EXISTS push_subscriptions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    user_id INTEGER NOT NULL,
                    endpoint TEXT NOT NULL UNIQUE,
                    p256dh TEXT NOT NULL,
                    auth TEXT NOT NULL,
                    created_at TEXT NOT NULL
                )
                """
            )

    def add(self, user_id: int, endpoint: str, p256dh: str, auth: str) -> PushSubscription:
        created_at = datetime.now(timezone.utc).replace(microsecond=0).isoformat()
        with self._connect() as conn:
            # ON CONFLICT ... DO UPDATE: re-subscribing the same endpoint
            # (the browser renews it) refreshes its keys/owner instead of
            # erroring on the UNIQUE constraint.
            conn.execute(
                """
                INSERT INTO push_subscriptions (user_id, endpoint, p256dh, auth, created_at)
                VALUES (?, ?, ?, ?, ?)
                ON CONFLICT(endpoint) DO UPDATE SET
                    user_id = excluded.user_id, p256dh = excluded.p256dh,
                    auth = excluded.auth, created_at = excluded.created_at
                """,
                (user_id, endpoint, p256dh, auth, created_at),
            )
            row = conn.execute(
                "SELECT id, user_id, endpoint, p256dh, auth, created_at FROM push_subscriptions WHERE endpoint = ?",
                (endpoint,),
            ).fetchone()
            return PushSubscription(*row)

    def get_by_endpoint(self, endpoint: str) -> PushSubscription | None:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT id, user_id, endpoint, p256dh, auth, created_at FROM push_subscriptions WHERE endpoint = ?",
                (endpoint,),
            ).fetchone()
            return PushSubscription(*row) if row else None

    def remove(self, endpoint: str) -> None:
        with self._connect() as conn:
            conn.execute("DELETE FROM push_subscriptions WHERE endpoint = ?", (endpoint,))

    def list_for_user(self, user_id: int) -> list[PushSubscription]:
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT id, user_id, endpoint, p256dh, auth, created_at FROM push_subscriptions WHERE user_id = ?",
                (user_id,),
            ).fetchall()
            return [PushSubscription(*row) for row in rows]

    def list_subscribed_user_ids(self) -> list[int]:
        with self._connect() as conn:
            rows = conn.execute("SELECT DISTINCT user_id FROM push_subscriptions").fetchall()
            return [row[0] for row in rows]
