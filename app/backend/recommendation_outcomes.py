"""W167: durable outcome tracking for live-generated recommendations,
independent of whether the user ever placed a bet on them -- unlike
bet_tracker.py (which only records what the user chose to log), this
resolves every actionable recommendation the agent produced against real
results, for the user's own diagnostics (GET /api/recommendations/stats,
recommendation_stats.py). One row per (match_id, date): the agent's actual
pick (A81's pick_recommended_market) from that match's latest cached
recommendation, resolved won/lost via src.agent.market_resolution.

Own db file (data/recommendation_outcomes.db), not recommendation_cache.db --
matches this codebase's established one-concern-one-db-file convention
(recommendation_cache.db, user_bets.db, job_runs.db all already do this)."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
import sqlite3

import requests

from app.backend.football_data_client import FootballDataClient
from app.backend.football_data_competition_codes import FOOTBALL_DATA_CODE_BY_LEAGUE
from app.backend.recommendation_cache import RecommendationCache
from app.backend.sandbox_clock import is_sandbox_mode, sandbox_scoped_path
from src.agent.market_resolution import RESOLVABLE_MARKETS, build_actual_outcome, market_correct, pick_recommended_market
from src.utils.logger import get_logger

LOGGER = get_logger(__name__)

DEFAULT_DB_PATH = Path(__file__).parent.parent.parent / "data" / "recommendation_outcomes.db"


@dataclass(frozen=True)
class RecommendationOutcome:
    id: int
    match_id: str
    date: str
    competition: str | None
    market: str
    selection: str
    recommendation_type: str
    confidence: str | None
    odds: float | None
    value_edge: float | None
    correct: bool
    generated_at: str
    resolved_at: str
    # W175: the verified football-data.org-routed code (E0/SP1/SWE/...),
    # distinct from `competition` above (the LLM's own unverified
    # match.league string) -- and the raw final score, so live_lessons.py
    # (W177) can rebuild the exact actual-outcome dict without a second
    # results fetch. All three nullable/additive -- a pre-migration row
    # simply carries None, see resolve_pending_recommendations() below.
    competition_id: str | None = None
    home_goals: int | None = None
    away_goals: int | None = None
    # W176: idempotency marker for the daily lesson-batching job (Task 3/4)
    # -- an ISO timestamp once this outcome has been folded into an
    # agent_lessons candidate, NULL until then. Mirrors this store's own
    # resolved_keys() idempotency pattern, one level further downstream.
    lesson_batched_at: str | None = None


class RecommendationOutcomeStore:
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
                CREATE TABLE IF NOT EXISTS recommendation_outcomes (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    match_id TEXT NOT NULL,
                    date TEXT NOT NULL,
                    competition TEXT,
                    market TEXT NOT NULL,
                    selection TEXT NOT NULL,
                    recommendation_type TEXT NOT NULL,
                    confidence TEXT,
                    odds REAL,
                    value_edge REAL,
                    correct INTEGER NOT NULL,
                    generated_at TEXT NOT NULL,
                    resolved_at TEXT NOT NULL,
                    UNIQUE(match_id, date)
                )
                """
            )
            # W175/W176: additive migration for a table that may already
            # exist (and have real rows) from before these columns existed.
            # The CREATE TABLE literal above is intentionally left as the
            # original pre-W175 DDL, not updated to include the columns
            # added since -- they're only ever added below, via ALTER
            # TABLE, for every table, fresh or not. SQLite's ALTER TABLE ADD
            # COLUMN has no IF NOT EXISTS clause (that's only valid on
            # CREATE TABLE/INDEX), so idempotency is done by hand via
            # PRAGMA table_info: every open runs this check; a brand-new
            # table has none of these columns yet (CREATE TABLE never added
            # them), so every ALTER below fires for it; a table that
            # already went through this migration already has them all, so
            # the PRAGMA guard turns every ALTER into a no-op. Same
            # idempotent-migration discipline as lessons.py's own DuckDB
            # ALTER TABLE for rule_text (A44). Add future columns the same
            # way -- append to the tuple below, no other change needed. No
            # backfill for rows resolved before this ships -- they simply
            # never get batched into a lesson (see live_lessons.py's own
            # NULL-competition_id skip, W177), an accepted small one-time
            # gap rather than a migration script.
            existing_columns = {row[1] for row in conn.execute("PRAGMA table_info(recommendation_outcomes)")}
            for column, coltype in (
                ("competition_id", "TEXT"),
                ("home_goals", "INTEGER"),
                ("away_goals", "INTEGER"),
                # W176: same additive-migration discipline as the three
                # W175 columns above -- nullable, backfilled to NULL for
                # every pre-existing row, guarded by the same PRAGMA check.
                ("lesson_batched_at", "TEXT"),
            ):
                if column not in existing_columns:
                    conn.execute(f"ALTER TABLE recommendation_outcomes ADD COLUMN {column} {coltype}")

    def resolved_keys(self) -> set[tuple[str, str]]:
        with self._connect() as conn:
            rows = conn.execute("SELECT match_id, date FROM recommendation_outcomes").fetchall()
        return {(row[0], row[1]) for row in rows}

    def insert(
        self,
        match_id: str,
        date: str,
        competition: str | None,
        market: str,
        selection: str,
        recommendation_type: str,
        confidence: str | None,
        odds: float | None,
        value_edge: float | None,
        correct: bool,
        generated_at: str,
        competition_id: str | None = None,
        home_goals: int | None = None,
        away_goals: int | None = None,
    ) -> RecommendationOutcome:
        resolved_at = datetime.now(timezone.utc).replace(microsecond=0).isoformat()
        with self._connect() as conn:
            cursor = conn.execute(
                """
                INSERT INTO recommendation_outcomes
                (match_id, date, competition, market, selection, recommendation_type,
                 confidence, odds, value_edge, correct, generated_at, resolved_at,
                 competition_id, home_goals, away_goals)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (match_id, date, competition, market, selection, recommendation_type,
                 confidence, odds, value_edge, int(correct), generated_at, resolved_at,
                 competition_id, home_goals, away_goals),
            )
            row_id = cursor.lastrowid
        return RecommendationOutcome(
            id=row_id, match_id=match_id, date=date, competition=competition, market=market,
            selection=selection, recommendation_type=recommendation_type, confidence=confidence,
            odds=odds, value_edge=value_edge, correct=correct, generated_at=generated_at, resolved_at=resolved_at,
            competition_id=competition_id, home_goals=home_goals, away_goals=away_goals,
        )

    def list_all(self, since: str | None = None) -> list[RecommendationOutcome]:
        query = (
            "SELECT id, match_id, date, competition, market, selection, recommendation_type, "
            "confidence, odds, value_edge, correct, generated_at, resolved_at, "
            "competition_id, home_goals, away_goals, lesson_batched_at FROM recommendation_outcomes"
        )
        params: tuple = ()
        if since is not None:
            query += " WHERE date >= ?"
            params = (since,)
        query += " ORDER BY date ASC"
        with self._connect() as conn:
            rows = conn.execute(query, params).fetchall()
        return [self._row_to_outcome(row) for row in rows]

    @staticmethod
    def _row_to_outcome(row: tuple) -> RecommendationOutcome:
        return RecommendationOutcome(
            id=row[0], match_id=row[1], date=row[2], competition=row[3], market=row[4], selection=row[5],
            recommendation_type=row[6], confidence=row[7], odds=row[8], value_edge=row[9],
            correct=bool(row[10]), generated_at=row[11], resolved_at=row[12],
            competition_id=row[13], home_goals=row[14], away_goals=row[15], lesson_batched_at=row[16],
        )

    def list_unbatched_for_lessons(self) -> list[RecommendationOutcome]:
        """W176: every outcome not yet folded into a lesson-generation
        batch. Deliberately unfiltered by date (unlike list_all(since=...))
        -- a prior run could have resolved an outcome it never got to batch
        (e.g. a crash between the resolve and batch steps), and that must
        still surface here regardless of age."""
        query = (
            "SELECT id, match_id, date, competition, market, selection, recommendation_type, "
            "confidence, odds, value_edge, correct, generated_at, resolved_at, "
            "competition_id, home_goals, away_goals, lesson_batched_at "
            "FROM recommendation_outcomes WHERE lesson_batched_at IS NULL ORDER BY date ASC"
        )
        with self._connect() as conn:
            rows = conn.execute(query).fetchall()
        return [self._row_to_outcome(row) for row in rows]

    def mark_lesson_batched(self, outcome_ids: list[int]) -> None:
        """Call only after the corresponding agent_lessons INSERT has
        actually succeeded -- marking first and inserting second would
        silently lose these outcomes from ever being reconsidered if the
        DuckDB write then failed."""
        if not outcome_ids:
            return
        batched_at = datetime.now(timezone.utc).replace(microsecond=0).isoformat()
        placeholders = ",".join("?" for _ in outcome_ids)
        with self._connect() as conn:
            conn.execute(
                f"UPDATE recommendation_outcomes SET lesson_batched_at = ? WHERE id IN ({placeholders})",
                (batched_at, *outcome_ids),
            )


_store_singleton: RecommendationOutcomeStore | None = None
_SANDBOX_STORE_DB_PATH = sandbox_scoped_path("recommendation_outcomes.db")


def get_recommendation_outcome_store() -> RecommendationOutcomeStore:
    """FastAPI dependency -- overridden in tests via app.dependency_overrides.
    Sandbox mode (W29) points this at a scratch db path, same convention as
    recommendations.get_cache()/bets.get_bet_tracker()."""
    global _store_singleton
    if _store_singleton is None:
        _store_singleton = (
            RecommendationOutcomeStore(db_path=_SANDBOX_STORE_DB_PATH) if is_sandbox_mode() else RecommendationOutcomeStore()
        )
    return _store_singleton


def resolve_pending_recommendations(
    cache: RecommendationCache,
    store: RecommendationOutcomeStore,
    client: FootballDataClient,
    sweden_client: object | None = None,
) -> list[RecommendationOutcome]:
    """W167: resolves every match's latest cached recommendation's actual
    pick against real results, mirroring settlement.py's settle_open_bets()
    structure (per-date result batching to respect the ~10 req/min
    football-data.org budget) but generalized across every domestic league
    the agent covers (FOOTBALL_DATA_CODE_BY_LEAGUE), not just PL.

    Deliberately does NOT trust the recommendation's own self-reported
    match.league for routing -- that field is LLM-authored and unverified,
    the same trust level as home_team/away_team. Instead merges results
    from every football-data.org competition code plus sweden_client for
    each date, keyed by match_id -- the same "disjoint id space" reasoning
    settlement.py already relies on for its own EPL+Sweden merge, just
    extended from one competition code to all of them."""
    already_resolved = store.resolved_keys()
    candidates: list[tuple] = []
    for entry in cache.list_latest_per_match():
        if (entry.match_id, entry.date) in already_resolved:
            continue
        rec = entry.recommendation
        if rec.get("overall") not in ("direct_bet", "conditional"):
            continue
        picked = pick_recommended_market(rec.get("markets") or [])
        if picked is None or picked.get("market") not in RESOLVABLE_MARKETS:
            continue
        candidates.append((entry, rec, picked))

    by_date: dict[str, list[tuple]] = {}
    for candidate in candidates:
        by_date.setdefault(candidate[0].date, []).append(candidate)

    newly_resolved: list[RecommendationOutcome] = []
    for date, group in by_date.items():
        results_by_id = {}
        # W175: track which of our own competition ids (E0/SP1/I1/D1/F1,
        # or SWE below) actually produced each result -- iterate .items()
        # instead of the old .values()-only loop specifically so this is
        # recoverable. The merge-everything-then-look-up-by-match_id shape
        # (results_by_id) is unchanged; this just runs a second dict
        # alongside it.
        competition_id_by_match: dict[str, str] = {}
        # W178: each call isolated with its own try/except -- found live
        # (2026-08-27) that a single competition's transient connection
        # error (RemoteDisconnected) propagated straight up through this
        # function, aborting outcome resolution *and* the daily_live_lessons
        # job that calls it (prepare_lesson_batches -> here) for every other
        # league and every other date in the same run, not just the one
        # flaky call. A skipped competition/date just leaves its candidates
        # unresolved for this pass -- already_resolved's dedup means the
        # next run (tomorrow, or a manual retry) picks them up again for
        # free, same as any other "no result yet" case this loop already
        # handles via the `match is None` check below.
        for internal_id, competition_code in FOOTBALL_DATA_CODE_BY_LEAGUE.items():
            try:
                results = client.get_results(competition_code=competition_code, date_from=date, date_to=date)
            except requests.exceptions.RequestException as exc:
                LOGGER.warning(
                    "resolve_pending_recommendations: get_results failed for competition_code=%s date=%s "
                    "(%s) -- skipping, other leagues/dates unaffected.",
                    competition_code, date, exc,
                )
                continue
            for match in results:
                results_by_id[match.match_id] = match
                competition_id_by_match[match.match_id] = internal_id
        if sweden_client is not None:
            try:
                sweden_results = sweden_client.get_results(date_from=date, date_to=date)
            except requests.exceptions.RequestException as exc:
                LOGGER.warning(
                    "resolve_pending_recommendations: sweden_client.get_results failed for date=%s (%s) "
                    "-- skipping, other leagues/dates unaffected.",
                    date, exc,
                )
                sweden_results = []
            for match in sweden_results:
                results_by_id[match.match_id] = match
                competition_id_by_match[match.match_id] = "SWE"

        for entry, rec, picked in group:
            match = results_by_id.get(entry.match_id)
            if match is None or match.home_goals is None or match.away_goals is None:
                continue
            actual = build_actual_outcome(match.home_goals, match.away_goals)
            correct = market_correct(picked, actual)
            if correct is None:
                continue
            outcome = store.insert(
                match_id=entry.match_id,
                date=entry.date,
                competition=(rec.get("match") or {}).get("league"),
                market=picked["market"],
                selection=picked["selection"],
                recommendation_type=picked["recommendation_type"],
                confidence=rec.get("confidence"),
                odds=picked.get("current_odds"),
                value_edge=picked.get("value_edge"),
                correct=correct,
                generated_at=entry.generated_at,
                competition_id=competition_id_by_match.get(entry.match_id),
                home_goals=match.home_goals,
                away_goals=match.away_goals,
            )
            newly_resolved.append(outcome)
    return newly_resolved
