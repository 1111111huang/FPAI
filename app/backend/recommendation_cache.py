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
    # A107: the LLM's own reasoning/tool-call trace (src.agent.graph.
    # serialize_agent_messages) and the raw forecast tool payload for this
    # generation -- the same tracing agent-train/agent-backtest already
    # persist to agent_telemetry, now captured for live generations too.
    # Both nullable/additive: a pre-A107 row simply has neither.
    reasoning_trace: list[dict] | None = None
    forecast_payload: dict | None = None


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
            # A107: additive migration for a table that may already exist
            # (real rows) from before these columns existed -- same
            # idempotent PRAGMA-guarded ALTER TABLE discipline
            # recommendation_outcomes.py's own W175/W176 columns already
            # established (SQLite's ADD COLUMN has no IF NOT EXISTS).
            existing_columns = {row[1] for row in conn.execute("PRAGMA table_info(recommendation_generations)")}
            for column in ("reasoning_trace_json", "forecast_payload_json"):
                if column not in existing_columns:
                    conn.execute(f"ALTER TABLE recommendation_generations ADD COLUMN {column} TEXT")

    def record_generation(
        self,
        match_id: str,
        date: str,
        agent_config_hash: str,
        odds: dict,
        recommendation: dict,
        triggered_by: TriggeredBy,
        generated_at: str | None = None,
        reasoning_trace: list[dict] | None = None,
        forecast_payload: dict | None = None,
    ) -> None:
        generated_at = generated_at or datetime.now(timezone.utc).replace(microsecond=0).isoformat()
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO recommendation_generations
                (match_id, date, agent_config_hash, odds_json, recommendation_json, generated_at, triggered_by,
                 reasoning_trace_json, forecast_payload_json)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    match_id, date, agent_config_hash, json.dumps(odds), json.dumps(recommendation), generated_at, triggered_by,
                    json.dumps(reasoning_trace), json.dumps(forecast_payload),
                ),
            )

    def get_latest(self, match_id: str, date: str, agent_config_hash: str) -> CacheEntry | None:
        with self._connect() as conn:
            row = conn.execute(
                """
                SELECT match_id, date, agent_config_hash, odds_json, recommendation_json, generated_at, triggered_by,
                       reasoning_trace_json, forecast_payload_json
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
                SELECT match_id, date, agent_config_hash, odds_json, recommendation_json, generated_at, triggered_by,
                       reasoning_trace_json, forecast_payload_json
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
                SELECT match_id, date, agent_config_hash, odds_json, recommendation_json, generated_at, triggered_by,
                       reasoning_trace_json, forecast_payload_json
                FROM recommendation_generations
                WHERE match_id = ? AND date = ? AND agent_config_hash = ?
                ORDER BY id ASC
                """,
                (match_id, date, agent_config_hash),
            ).fetchall()
        return [self._row_to_entry(row) for row in rows]

    def list_latest_per_match(self) -> list[CacheEntry]:
        """One row per distinct (match_id, date) -- the single most recent
        generation across any agent_config_hash, mirroring
        get_latest_any_config()'s own "ignore the hash" fallback semantics
        but for every cached match at once. Used by
        recommendation_outcomes.py's resolution job (W167) to find every
        match with a live-generated recommendation, not just ones a caller
        already knows the key for."""
        with self._connect() as conn:
            # ponytail: correlated subquery over the whole table -- fine at today's
            # scale (an on-demand diagnostics job, not a hot path); if
            # recommendation_generations grows into the 100K+ row range, revisit with
            # an index on (match_id, date, id) or a GROUP BY + IN pattern.
            rows = conn.execute(
                """
                SELECT match_id, date, agent_config_hash, odds_json, recommendation_json, generated_at, triggered_by,
                       reasoning_trace_json, forecast_payload_json
                FROM recommendation_generations rg
                WHERE id = (
                    SELECT MAX(id) FROM recommendation_generations rg2
                    WHERE rg2.match_id = rg.match_id AND rg2.date = rg.date
                )
                """
            ).fetchall()
        return [self._row_to_entry(row) for row in rows]

    def most_recent_generated_at(self) -> str | None:
        """Latest `generated_at` across every row, any match/config -- used
        by main.py's boot-time pregenerate cooldown to tell "the cache is
        already fresh from a recent EOD/T-30/pregenerate pass" from "this
        container has been cold for a while", without caring which job
        produced the freshness."""
        with self._connect() as conn:
            row = conn.execute("SELECT MAX(generated_at) FROM recommendation_generations").fetchone()
        return row[0] if row and row[0] else None

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
            # A107: a genuinely pre-migration row has SQL NULL here (ALTER
            # TABLE ADD COLUMN backfills existing rows with NULL, not the
            # JSON string "null" record_generation's own default writes for
            # a NEW row with no trace) -- guard against json.loads(None).
            reasoning_trace=json.loads(row[7]) if row[7] is not None else None,
            forecast_payload=json.loads(row[8]) if row[8] is not None else None,
        )
