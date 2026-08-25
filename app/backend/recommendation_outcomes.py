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

from app.backend.football_data_client import FootballDataClient
from app.backend.football_data_competition_codes import FOOTBALL_DATA_CODE_BY_LEAGUE
from app.backend.recommendation_cache import RecommendationCache
from app.backend.sandbox_clock import is_sandbox_mode, sandbox_scoped_path
from src.agent.market_resolution import RESOLVABLE_MARKETS, build_actual_outcome, market_correct, pick_recommended_market

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
    ) -> RecommendationOutcome:
        resolved_at = datetime.now(timezone.utc).replace(microsecond=0).isoformat()
        with self._connect() as conn:
            cursor = conn.execute(
                """
                INSERT INTO recommendation_outcomes
                (match_id, date, competition, market, selection, recommendation_type,
                 confidence, odds, value_edge, correct, generated_at, resolved_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (match_id, date, competition, market, selection, recommendation_type,
                 confidence, odds, value_edge, int(correct), generated_at, resolved_at),
            )
            row_id = cursor.lastrowid
        return RecommendationOutcome(
            id=row_id, match_id=match_id, date=date, competition=competition, market=market,
            selection=selection, recommendation_type=recommendation_type, confidence=confidence,
            odds=odds, value_edge=value_edge, correct=correct, generated_at=generated_at, resolved_at=resolved_at,
        )

    def list_all(self, since: str | None = None) -> list[RecommendationOutcome]:
        query = (
            "SELECT id, match_id, date, competition, market, selection, recommendation_type, "
            "confidence, odds, value_edge, correct, generated_at, resolved_at FROM recommendation_outcomes"
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
        for competition_code in FOOTBALL_DATA_CODE_BY_LEAGUE.values():
            for match in client.get_results(competition_code=competition_code, date_from=date, date_to=date):
                results_by_id[match.match_id] = match
        if sweden_client is not None:
            for match in sweden_client.get_results(date_from=date, date_to=date):
                results_by_id[match.match_id] = match

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
            )
            newly_resolved.append(outcome)
    return newly_resolved
