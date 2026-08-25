"""W11: recommendation caching layer, in app/backend. Not a reuse of
SnapshotStore (src/agent/snapshot_store.py) -- that component's record/replay
semantics are purpose-built for backtest determinism, keyed by tool-call
SHA-256, gitignored; repurposing it for live caching would conflate two
different concerns. This is a new, SQLite-backed store keyed by
(match_id, date, agent_config_hash).

Append-only: every generation is kept as a row, not just the latest -- a
lightweight generation history (timestamp + odds snapshot per generation)
lets a future consumer (W10) cheaply detect "no new data" before deciding
whether to regenerate, and doubles as an audit trail. "The cache" for a key
is simply its most recent row (get_latest); get_history returns the rest.

W163: DEFAULT_DB_PATH resolves to repo-root data/ (three .parent hops from
this file, under app/backend/), not app/data/ (two hops) -- found live via a
direct user report: Railway's single mounted volume sits at /app/data
(container root == repo root, confirmed from the service's own start
command, no Root Directory override), covering the ML engine's own
data/fpai_core.db (config_loader.py's bare "data/fpai_core.db", resolved
against cwd) but never this file's two-hop path, which lands one directory
deeper at /app/app/data -- silently unpersisted across every redeploy.
Railway services get exactly one volume, so the fix is aligning every
app/backend/ data path onto the repo-root data/ directory the volume
already covers (this file, bet_tracker.py, scheduler.py's JobRunLog,
sandbox_clock.py's sandbox_scoped_path(), scheduler_wiring.py's
CREDIT_COUNTER_PATH[_2]) -- not requesting a second volume that isn't
available, and not moving the volume itself (which would orphan
fpai_core.db instead). recommendations.py's own
_SANDBOX_SNAPSHOT_BASE_DIR/_CORPUS_BASE_DIR already used this same
three-hop convention -- this bug was this file (and its siblings) not
matching a pattern that already existed correctly elsewhere in this exact
codebase.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import json
from pathlib import Path
import sqlite3
from typing import Literal

DEFAULT_DB_PATH = Path(__file__).parent.parent.parent / "data" / "recommendation_cache.db"

TriggeredBy = Literal["scheduled", "manual_regenerate"]


@dataclass(frozen=True)
class CacheEntry:
    match_id: str
    date: str
    agent_config_hash: str
    odds: dict
    recommendation: dict
    generated_at: str
    triggered_by: TriggeredBy


class RecommendationCache:
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
                CREATE TABLE IF NOT EXISTS recommendation_generations (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    match_id TEXT NOT NULL,
                    date TEXT NOT NULL,
                    agent_config_hash TEXT NOT NULL,
                    odds_json TEXT NOT NULL,
                    recommendation_json TEXT NOT NULL,
                    generated_at TEXT NOT NULL,
                    triggered_by TEXT NOT NULL
                )
                """
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_recgen_key "
                "ON recommendation_generations (match_id, date, agent_config_hash)"
            )

    def record_generation(
        self,
        match_id: str,
        date: str,
        agent_config_hash: str,
        odds: dict,
        recommendation: dict,
        triggered_by: TriggeredBy,
        generated_at: str | None = None,
    ) -> None:
        generated_at = generated_at or datetime.now(timezone.utc).replace(microsecond=0).isoformat()
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO recommendation_generations
                (match_id, date, agent_config_hash, odds_json, recommendation_json, generated_at, triggered_by)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (match_id, date, agent_config_hash, json.dumps(odds), json.dumps(recommendation), generated_at, triggered_by),
            )

    def get_latest(self, match_id: str, date: str, agent_config_hash: str) -> CacheEntry | None:
        with self._connect() as conn:
            row = conn.execute(
                """
                SELECT match_id, date, agent_config_hash, odds_json, recommendation_json, generated_at, triggered_by
                FROM recommendation_generations
                WHERE match_id = ? AND date = ? AND agent_config_hash = ?
                ORDER BY id DESC LIMIT 1
                """,
                (match_id, date, agent_config_hash),
            ).fetchone()
        return self._row_to_entry(row) if row else None

    def get_latest_any_config(self, match_id: str, date: str) -> CacheEntry | None:
        """A65/A66 follow-up: a config change (a tunable threshold, a model
        swap) bumps agent_config_hash for every match at once, making every
        prior generation briefly unreachable via get_latest()'s exact-hash
        lookup until each match is regenerated under the new config. If that
        regeneration also fails for an unrelated reason (confirmed live: a
        DeepSeek billing outage failed every match in the same batch
        identically), get_latest() alone leaves nothing to serve at all,
        even though a perfectly good prior recommendation still physically
        exists in this same table under an older hash. Used only as an
        explicit fallback (main.py's GET /api/recommendations/{match_id}) --
        not a replacement for get_latest()'s own exact-hash semantics, which
        eod_batch.py's already_fresh() still needs unchanged (a config
        change should still trigger fresh regeneration, not be masked by
        this)."""
        with self._connect() as conn:
            row = conn.execute(
                """
                SELECT match_id, date, agent_config_hash, odds_json, recommendation_json, generated_at, triggered_by
                FROM recommendation_generations
                WHERE match_id = ? AND date = ?
                ORDER BY id DESC LIMIT 1
                """,
                (match_id, date),
            ).fetchone()
        return self._row_to_entry(row) if row else None

    def get_history(self, match_id: str, date: str, agent_config_hash: str) -> list[CacheEntry]:
        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT match_id, date, agent_config_hash, odds_json, recommendation_json, generated_at, triggered_by
                FROM recommendation_generations
                WHERE match_id = ? AND date = ? AND agent_config_hash = ?
                ORDER BY id ASC
                """,
                (match_id, date, agent_config_hash),
            ).fetchall()
        return [self._row_to_entry(row) for row in rows]

    @staticmethod
    def _row_to_entry(row: tuple) -> CacheEntry:
        return CacheEntry(
            match_id=row[0],
            date=row[1],
            agent_config_hash=row[2],
            odds=json.loads(row[3]),
            recommendation=json.loads(row[4]),
            generated_at=row[5],
            triggered_by=row[6],
        )
