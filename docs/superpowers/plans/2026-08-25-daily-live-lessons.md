# Daily Live-Recommendation Lessons Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Turn the app's own daily finished-match recommendations into reviewable lesson candidates in the existing `agent_lessons` table — the same human-approval pipeline `agent-train` already feeds, now also fed daily from live results instead of only from backtest corpora.

**Architecture:** Fix two real gaps in `recommendation_outcomes` (an unverified competition string instead of a real code; the raw final score discarded after use), add an idempotency marker, then a new `app/backend/live_lessons.py` module that adapts `RecommendationOutcome` rows into the exact `BacktestRecord` shape `src/agent/lessons.py`'s existing, unmodified functions already consume — wired into the backend scheduler as a new daily job that resolves outcomes and generates lessons as one unattended pipeline.

**Tech Stack:** Python, SQLite (`recommendation_outcomes.db`), DuckDB (`agent_lessons` in `data/fpai_core.db`), APScheduler (via the existing `RecoverableScheduler`).

Design doc: `docs/superpowers/specs/2026-08-25-daily-live-lessons-design.md`

---

## File Structure

- **Modify:** `app/backend/recommendation_outcomes.py` — new `competition_id`/`home_goals`/`away_goals`/`lesson_batched_at` columns on `recommendation_outcomes`; `resolve_pending_recommendations()` fixed to persist the real competition code and raw score it already computes; new `list_unbatched_for_lessons()`/`mark_lesson_batched()` methods.
- **Create:** `app/backend/live_lessons.py` — `_to_lesson_record()` adapter, `generate_daily_lessons()` pipeline. The only new module; everything in `src/agent/lessons.py` is reused unmodified.
- **Modify:** `app/backend/scheduler_wiring.py` — new `register_lessons_job()`, mirroring `register_eod_job()`'s existing shape, plus its own small `_build_lessons_llm_invoke()` helper.
- **Modify:** `app/backend/main.py` — one new call in `lifespan()`, registering the lessons job alongside the existing EOD job registration, same `ENABLE_SCHEDULER` gate.
- **Test:** `app/backend/tests/test_recommendation_outcomes.py`, new `app/backend/tests/test_live_lessons.py`, `app/backend/tests/test_scheduler_wiring.py`.

---

### Task 1: Persist the real competition code and raw score on `recommendation_outcomes` (W175)

**Files:**
- Modify: `app/backend/recommendation_outcomes.py`
- Test: `app/backend/tests/test_recommendation_outcomes.py`

**Context for this task:** `recommendation_outcomes.competition` is filled from the LLM's own self-reported `match.league` string (e.g. `"Premier League"`), never validated — `resolve_pending_recommendations()` deliberately doesn't trust it for routing *result lookups* (see its own docstring), but still writes it into the row unchanged. Meanwhile the function's per-date loop already knows, for every match it actually finds a result for, exactly which `FOOTBALL_DATA_CODE_BY_LEAGUE` key (`E0`/`SP1`/`I1`/`D1`/`F1`) or `"SWE"` (Sweden) produced that result — this is thrown away today. This task persists that real code as a new `competition_id` column (the existing `competition` column is untouched, still whatever free text the dashboard already displays), plus the raw `home_goals`/`away_goals` from the same already-fetched result (also currently discarded after computing `correct`) — both needed by Task 3 so lesson-generation never has to re-fetch results a second time.

- [ ] **Step 1: Write the failing tests**

In `app/backend/tests/test_recommendation_outcomes.py`, add these tests after the existing `test_skips_unresolvable_markets`:

```python
def test_persists_the_verified_competition_id_not_the_self_reported_league(tmp_path: Path) -> None:
    """W175: `competition` stays the LLM's self-reported string (existing,
    unchanged behavior) -- the new `competition_id` is the real code the
    results lookup actually matched against. Deliberately different values
    here to prove these are two independent columns, not aliases."""
    cache = RecommendationCache(db_path=tmp_path / "cache.db")
    store = RecommendationOutcomeStore(db_path=tmp_path / "outcomes.db")
    cache.record_generation(
        "m1", "2026-08-22", "hash1", {},
        _rec("direct_bet", "result_3way", "home", "direct_bet", 2.0, league="a made-up league name"),
        "scheduled",
    )
    client = MagicMock()

    def fake_get_results(competition_code, date_from, date_to):
        return [_match("m1", 2, 1)] if competition_code == "PD" else []

    client.get_results.side_effect = fake_get_results

    resolved = resolve_pending_recommendations(cache, store, client)

    assert resolved[0].competition == "a made-up league name"
    assert resolved[0].competition_id == "SP1"


def test_persists_the_raw_final_score(tmp_path: Path) -> None:
    cache = RecommendationCache(db_path=tmp_path / "cache.db")
    store = RecommendationOutcomeStore(db_path=tmp_path / "outcomes.db")
    cache.record_generation("m1", "2026-08-22", "hash1", {}, _rec("direct_bet", "result_3way", "home", "direct_bet", 2.0), "scheduled")
    client = MagicMock()
    client.get_results.return_value = [_match("m1", 2, 1)]

    resolved = resolve_pending_recommendations(cache, store, client)

    assert resolved[0].home_goals == 2
    assert resolved[0].away_goals == 1


def test_persists_sweden_competition_id_via_the_sweden_client(tmp_path: Path) -> None:
    cache = RecommendationCache(db_path=tmp_path / "cache.db")
    store = RecommendationOutcomeStore(db_path=tmp_path / "outcomes.db")
    cache.record_generation("m1", "2026-08-22", "hash1", {}, _rec("direct_bet", "result_3way", "home", "direct_bet", 2.0), "scheduled")
    client = MagicMock()
    client.get_results.return_value = []
    sweden_client = MagicMock()
    sweden_client.get_results.return_value = [_match("m1", 2, 1)]

    resolved = resolve_pending_recommendations(cache, store, client, sweden_client)

    assert resolved[0].competition_id == "SWE"


def test_reopening_store_after_schema_migration_is_idempotent(tmp_path: Path) -> None:
    db_path = tmp_path / "outcomes.db"
    RecommendationOutcomeStore(db_path=db_path)
    RecommendationOutcomeStore(db_path=db_path)  # second open must not raise
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest app/backend/tests/test_recommendation_outcomes.py -v`
Expected: the three new behavioral tests FAIL with `AttributeError: 'RecommendationOutcome' object has no attribute 'competition_id'` (or similar for `home_goals`/`away_goals`); the idempotency test passes already (nothing to migrate yet) — that's fine, it becomes a real regression guard once Step 3 lands.

- [ ] **Step 3: Add the columns, dataclass fields, and population fix**

In `app/backend/recommendation_outcomes.py`, replace the `RecommendationOutcome` dataclass:

```python
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
```

Then `_init_schema`, adding the migration lines after the existing `CREATE TABLE IF NOT EXISTS`:

```python
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
            # W175: additive migration for a table that may already exist
            # (and have real rows) from before this column existed --
            # ADD COLUMN IF NOT EXISTS is safe against both a fresh table
            # (no-op) and a populated one (existing rows get NULL), same
            # idempotent-migration discipline as lessons.py's own DuckDB
            # ALTER TABLE for rule_text (A44). No backfill for rows
            # resolved before this ships -- they simply never get batched
            # into a lesson (see live_lessons.py's own NULL-competition_id
            # skip, W177), an accepted small one-time gap rather than a
            # migration script.
            conn.execute("ALTER TABLE recommendation_outcomes ADD COLUMN IF NOT EXISTS competition_id TEXT")
            conn.execute("ALTER TABLE recommendation_outcomes ADD COLUMN IF NOT EXISTS home_goals INTEGER")
            conn.execute("ALTER TABLE recommendation_outcomes ADD COLUMN IF NOT EXISTS away_goals INTEGER")
```

Then `insert()` — add the three new parameters and thread them through both the INSERT and the returned object:

```python
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
```

Then `list_all()` and `_row_to_outcome()`:

```python
    def list_all(self, since: str | None = None) -> list[RecommendationOutcome]:
        query = (
            "SELECT id, match_id, date, competition, market, selection, recommendation_type, "
            "confidence, odds, value_edge, correct, generated_at, resolved_at, "
            "competition_id, home_goals, away_goals FROM recommendation_outcomes"
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
            competition_id=row[13], home_goals=row[14], away_goals=row[15],
        )
```

Finally, `resolve_pending_recommendations()` — replace the per-date results loop and the `store.insert(...)` call:

```python
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
        for internal_id, competition_code in FOOTBALL_DATA_CODE_BY_LEAGUE.items():
            for match in client.get_results(competition_code=competition_code, date_from=date, date_to=date):
                results_by_id[match.match_id] = match
                competition_id_by_match[match.match_id] = internal_id
        if sweden_client is not None:
            for match in sweden_client.get_results(date_from=date, date_to=date):
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
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest app/backend/tests/test_recommendation_outcomes.py -v`
Expected: PASS, all tests including the 4 new ones.

Then run the wider suite to confirm nothing that constructs/reads `RecommendationOutcome` broke (the new fields are keyword-only-in-practice with defaults, so existing callers are unaffected — this just proves it):

Run: `pytest app/backend/tests/test_recommendation_outcomes.py app/backend/tests/test_recommendation_outcomes_endpoints.py app/backend/tests/test_recommendation_stats.py app/backend/tests/test_agent_performance_dashboard.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add app/backend/recommendation_outcomes.py app/backend/tests/test_recommendation_outcomes.py
git commit -m "feat(app): persist verified competition_id + raw score on recommendation_outcomes (W175)

Co-Authored-By: Claude Sonnet 5 <noreply@anthropic.com>"
```

## Context

Task 1 of a 5-task plan for daily live-recommendation lessons. Design: `docs/superpowers/specs/2026-08-25-daily-live-lessons-design.md`. This is a self-contained fix to `app/backend/recommendation_outcomes.py` (W167's existing module) — no new module yet, that's Task 3.

## Before You Begin

If anything above is unclear, ask now.

## Your Job

Implement exactly what's specified, run the tests, verify they pass, commit, self-review, report back.

Work from: /Users/tianqihuang/Documents/GitHub/FPAI

## Report Format

- **Status:** DONE | DONE_WITH_CONCERNS | BLOCKED | NEEDS_CONTEXT
- What you implemented, test results (paste output), files changed, self-review findings, commit SHA.

---

### Task 2: Idempotency marker for lesson batching (W176)

**Files:**
- Modify: `app/backend/recommendation_outcomes.py`
- Test: `app/backend/tests/test_recommendation_outcomes.py`

**Context for this task:** Nothing today marks "this outcome has already been folded into a lesson batch." Without a marker, a rerun of the daily job (Task 3/4) would re-lesson the same finished matches every day forever. This adds a `lesson_batched_at` column plus two store methods, mirroring the store's own existing `resolved_keys()` idempotency pattern.

- [ ] **Step 1: Write the failing tests**

In `app/backend/tests/test_recommendation_outcomes.py`, add:

```python
def test_list_unbatched_for_lessons_excludes_already_batched(tmp_path: Path) -> None:
    store = RecommendationOutcomeStore(db_path=tmp_path / "outcomes.db")
    o1 = store.insert(
        match_id="m1", date="2026-08-22", competition="E0", market="result_3way", selection="home",
        recommendation_type="direct_bet", confidence="medium", odds=2.0, value_edge=0.1, correct=True,
        generated_at="2026-08-22T10:00:00+00:00", competition_id="E0", home_goals=2, away_goals=1,
    )
    store.insert(
        match_id="m2", date="2026-08-22", competition="E0", market="result_3way", selection="away",
        recommendation_type="direct_bet", confidence="medium", odds=2.0, value_edge=0.1, correct=False,
        generated_at="2026-08-22T10:00:00+00:00", competition_id="E0", home_goals=0, away_goals=1,
    )

    store.mark_lesson_batched([o1.id])

    unbatched = store.list_unbatched_for_lessons()
    assert [o.match_id for o in unbatched] == ["m2"]


def test_mark_lesson_batched_is_a_noop_for_an_empty_list(tmp_path: Path) -> None:
    store = RecommendationOutcomeStore(db_path=tmp_path / "outcomes.db")
    store.mark_lesson_batched([])  # must not raise


def test_list_unbatched_for_lessons_has_no_date_filter(tmp_path: Path) -> None:
    """Unlike list_all(since=...) -- a prior run could have resolved an
    outcome it never got to batch (e.g. a crash between steps), and that
    must still surface here no matter how old it is."""
    store = RecommendationOutcomeStore(db_path=tmp_path / "outcomes.db")
    store.insert(
        match_id="old", date="2020-01-01", competition="E0", market="result_3way", selection="home",
        recommendation_type="direct_bet", confidence="medium", odds=2.0, value_edge=0.1, correct=True,
        generated_at="2020-01-01T10:00:00+00:00", competition_id="E0", home_goals=1, away_goals=0,
    )
    assert len(store.list_unbatched_for_lessons()) == 1
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest app/backend/tests/test_recommendation_outcomes.py -v`
Expected: FAIL with `AttributeError: 'RecommendationOutcomeStore' object has no attribute 'list_unbatched_for_lessons'`.

- [ ] **Step 3: Add the column and the two methods**

In `app/backend/recommendation_outcomes.py`, add one more migration line to `_init_schema` (right after the three added in Task 1):

```python
            conn.execute("ALTER TABLE recommendation_outcomes ADD COLUMN IF NOT EXISTS lesson_batched_at TEXT")
```

Add `lesson_batched_at: str | None = None` as a fourth new field at the end of the `RecommendationOutcome` dataclass (after `away_goals`).

Update `list_all()`'s SELECT and `_row_to_outcome()` to also carry it through:

```python
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
```

Then add the two new methods (place after `resolved_keys()`):

```python
    def list_unbatched_for_lessons(self) -> list[RecommendationOutcome]:
        """W176: every outcome not yet folded into a lesson-generation
        batch. Deliberately unfiltered by date (unlike list_all(since=...))
        -- see the module-level docstring on why an old unbatched row must
        still surface here."""
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
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest app/backend/tests/test_recommendation_outcomes.py -v`
Expected: PASS, all tests.

- [ ] **Step 5: Commit**

```bash
git add app/backend/recommendation_outcomes.py app/backend/tests/test_recommendation_outcomes.py
git commit -m "feat(app): idempotency marker for lesson batching on recommendation_outcomes (W176)

Co-Authored-By: Claude Sonnet 5 <noreply@anthropic.com>"
```

## Context

Task 2 of a 5-task plan. Depends on Task 1's schema/dataclass changes already being in place (this task extends the same file/dataclass further).

## Before You Begin

If anything above is unclear, ask now.

## Your Job

Implement exactly what's specified, run the tests, verify they pass, commit, self-review, report back.

Work from: /Users/tianqihuang/Documents/GitHub/FPAI

## Report Format

- **Status:** DONE | DONE_WITH_CONCERNS | BLOCKED | NEEDS_CONTEXT
- What you implemented, test results (paste output), files changed, self-review findings, commit SHA.

---

### Task 3: `app/backend/live_lessons.py` — adapter + daily batch pipeline (W177)

**Files:**
- Create: `app/backend/live_lessons.py`
- Test: `app/backend/tests/test_live_lessons.py`

**Context for this task:** This is the one new module. It reuses `src/agent/lessons.py`'s `generate_batch_lesson_text`/`generate_batch_reflection`/`insert_lesson_candidate` completely unmodified — the only new code is (a) an adapter turning a `RecommendationOutcome` (+ a `RecommendationCache` join) into the exact `BacktestRecord` shape those functions already expect, and (b) the batch/grouping/idempotency loop around it. `src/agent/backtest.py`'s `BacktestRecord` has fields `match_id, home_team, away_team, date, league, recommendation, actual, market_results` (`full_state` is optional, leave it unset). Every live-sourced `BacktestRecord`'s `market_results` will have exactly one entry (the market actually picked) — unlike a training-sourced one, which scores every market the agent evaluated; this is an accepted, deliberate scope narrowing (see the design doc), not a bug.

- [ ] **Step 1: Write the failing tests**

Create `app/backend/tests/test_live_lessons.py`:

```python
"""W177: live_lessons.py -- adapts RecommendationOutcome rows into
BacktestRecord-shaped objects and batches them into agent_lessons
candidates via src/agent/lessons.py's existing, unmodified functions."""

from __future__ import annotations

from pathlib import Path
import sys
from unittest.mock import MagicMock

import duckdb

sys.path.append(str(Path(__file__).resolve().parents[3]))

from app.backend.football_data_client import NormalizedMatch
from app.backend.live_lessons import _to_lesson_record, generate_daily_lessons
from app.backend.recommendation_cache import RecommendationCache
from app.backend.recommendation_outcomes import RecommendationOutcome, RecommendationOutcomeStore
from src.agent.lessons import create_lessons_tables


def _outcome(**overrides) -> RecommendationOutcome:
    defaults = dict(
        id=1, match_id="m1", date="2026-08-22", competition="Premier League",
        market="result_3way", selection="home", recommendation_type="direct_bet",
        confidence="medium", odds=2.0, value_edge=0.1, correct=True,
        generated_at="2026-08-22T10:00:00+00:00", resolved_at="2026-08-23T00:00:00+00:00",
        competition_id="E0", home_goals=2, away_goals=1, lesson_batched_at=None,
    )
    defaults.update(overrides)
    return RecommendationOutcome(**defaults)


def _rec(league: str = "E0") -> dict:
    return {
        "match": {"home": "Arsenal", "away": "Everton", "date": "2026-08-22", "league": league},
        "overall": "direct_bet",
        "markets": [{
            "market": "result_3way", "selection": "home", "recommendation_type": "direct_bet",
            "current_odds": 2.0, "value_edge": 0.1,
        }],
        "confidence": "medium", "explanation": ["good value"], "limitations": [],
        "prediction_basis": "team_history_and_market",
    }


def _duckdb_conn() -> duckdb.DuckDBPyConnection:
    conn = duckdb.connect(":memory:")
    create_lessons_tables(conn)
    return conn


def test_to_lesson_record_enriches_from_a_real_cache_hit(tmp_path: Path) -> None:
    cache = RecommendationCache(db_path=tmp_path / "cache.db")
    cache.record_generation("m1", "2026-08-22", "hash1", {}, _rec(), "scheduled")

    record = _to_lesson_record(_outcome(), cache)

    assert record.home_team == "Arsenal"
    assert record.away_team == "Everton"
    assert record.league == "E0"
    assert record.actual["result"] == "home"
    assert record.market_results == [{"market": "result_3way", "selection": "home", "correct": True}]


def test_to_lesson_record_degrades_gracefully_on_cache_miss(tmp_path: Path) -> None:
    cache = RecommendationCache(db_path=tmp_path / "cache.db")  # nothing recorded

    record = _to_lesson_record(_outcome(), cache)

    assert record.home_team == ""
    assert record.away_team == ""
    assert record.recommendation == {}


def test_to_lesson_record_degrades_gracefully_on_a_pre_migration_missing_score(tmp_path: Path) -> None:
    cache = RecommendationCache(db_path=tmp_path / "cache.db")
    cache.record_generation("m1", "2026-08-22", "hash1", {}, _rec(), "scheduled")

    record = _to_lesson_record(_outcome(home_goals=None, away_goals=None), cache)

    assert record.actual == {}


def test_generate_daily_lessons_groups_by_competition_and_date(tmp_path: Path) -> None:
    cache = RecommendationCache(db_path=tmp_path / "cache.db")
    store = RecommendationOutcomeStore(db_path=tmp_path / "outcomes.db")
    store.insert(
        match_id="m1", date="2026-08-22", competition="Premier League", market="result_3way",
        selection="home", recommendation_type="direct_bet", confidence="medium", odds=2.0,
        value_edge=0.1, correct=True, generated_at="2026-08-22T10:00:00+00:00",
        competition_id="E0", home_goals=2, away_goals=1,
    )
    store.insert(
        match_id="m2", date="2026-08-22", competition="La Liga", market="result_3way",
        selection="away", recommendation_type="direct_bet", confidence="medium", odds=2.0,
        value_edge=0.1, correct=False, generated_at="2026-08-22T10:00:00+00:00",
        competition_id="SP1", home_goals=0, away_goals=1,
    )
    client = MagicMock()
    client.get_results.return_value = []  # nothing new to resolve this run
    conn = _duckdb_conn()

    lesson_ids = generate_daily_lessons(cache, store, client, conn, llm_invoke=None)

    assert len(lesson_ids) == 2  # E0 batch and SP1 batch, never merged
    rows = conn.execute(
        "SELECT competition_id, source_match_id FROM agent_lessons ORDER BY competition_id"
    ).fetchall()
    assert rows == [("E0", "m1"), ("SP1", "m2")]


def test_generate_daily_lessons_skips_outcomes_with_no_competition_id(tmp_path: Path) -> None:
    cache = RecommendationCache(db_path=tmp_path / "cache.db")
    store = RecommendationOutcomeStore(db_path=tmp_path / "outcomes.db")
    store.insert(
        match_id="m1", date="2026-08-22", competition="Premier League", market="result_3way",
        selection="home", recommendation_type="direct_bet", confidence="medium", odds=2.0,
        value_edge=0.1, correct=True, generated_at="2026-08-22T10:00:00+00:00",
    )  # competition_id deliberately omitted -- simulates a pre-W175 row
    client = MagicMock()
    client.get_results.return_value = []
    conn = _duckdb_conn()

    lesson_ids = generate_daily_lessons(cache, store, client, conn, llm_invoke=None)

    assert lesson_ids == []
    assert len(store.list_unbatched_for_lessons()) == 1  # still unbatched, not silently marked done


def test_generate_daily_lessons_marks_outcomes_batched(tmp_path: Path) -> None:
    cache = RecommendationCache(db_path=tmp_path / "cache.db")
    store = RecommendationOutcomeStore(db_path=tmp_path / "outcomes.db")
    store.insert(
        match_id="m1", date="2026-08-22", competition="Premier League", market="result_3way",
        selection="home", recommendation_type="direct_bet", confidence="medium", odds=2.0,
        value_edge=0.1, correct=True, generated_at="2026-08-22T10:00:00+00:00",
        competition_id="E0", home_goals=2, away_goals=1,
    )
    client = MagicMock()
    client.get_results.return_value = []
    conn = _duckdb_conn()

    generate_daily_lessons(cache, store, client, conn, llm_invoke=None)

    assert store.list_unbatched_for_lessons() == []


def test_generate_daily_lessons_prepends_the_live_source_note_and_skips_reflection_without_an_llm(tmp_path: Path) -> None:
    cache = RecommendationCache(db_path=tmp_path / "cache.db")
    store = RecommendationOutcomeStore(db_path=tmp_path / "outcomes.db")
    store.insert(
        match_id="m1", date="2026-08-22", competition="Premier League", market="result_3way",
        selection="home", recommendation_type="direct_bet", confidence="medium", odds=2.0,
        value_edge=0.1, correct=True, generated_at="2026-08-22T10:00:00+00:00",
        competition_id="E0", home_goals=2, away_goals=1,
    )
    client = MagicMock()
    client.get_results.return_value = []
    conn = _duckdb_conn()

    generate_daily_lessons(cache, store, client, conn, llm_invoke=None)

    lesson_text = conn.execute("SELECT lesson_text FROM agent_lessons").fetchone()[0]
    assert lesson_text.startswith("Live-sourced batch:")
    assert "Reflection:" not in lesson_text


def test_generate_daily_lessons_appends_reflection_when_llm_invoke_given(tmp_path: Path) -> None:
    cache = RecommendationCache(db_path=tmp_path / "cache.db")
    store = RecommendationOutcomeStore(db_path=tmp_path / "outcomes.db")
    store.insert(
        match_id="m1", date="2026-08-22", competition="Premier League", market="result_3way",
        selection="home", recommendation_type="direct_bet", confidence="medium", odds=2.0,
        value_edge=0.1, correct=True, generated_at="2026-08-22T10:00:00+00:00",
        competition_id="E0", home_goals=2, away_goals=1,
    )
    client = MagicMock()
    client.get_results.return_value = []
    conn = _duckdb_conn()

    generate_daily_lessons(cache, store, client, conn, llm_invoke=lambda prompt: "a real reflection")

    lesson_text = conn.execute("SELECT lesson_text FROM agent_lessons").fetchone()[0]
    assert "Reflection: a real reflection" in lesson_text


def test_generate_daily_lessons_resolves_pending_recommendations_first(tmp_path: Path) -> None:
    """End-to-end: a brand-new, not-yet-resolved recommendation gets
    resolved and then batched in the same call -- proves the two steps
    run as one pipeline, not something a caller must sequence itself."""
    cache = RecommendationCache(db_path=tmp_path / "cache.db")
    store = RecommendationOutcomeStore(db_path=tmp_path / "outcomes.db")
    cache.record_generation("m1", "2026-08-22", "hash1", {}, _rec(), "scheduled")
    client = MagicMock()

    def fake_get_results(competition_code, date_from, date_to):
        if competition_code == "PL":
            return [NormalizedMatch(
                match_id="m1", utc_date="2026-08-22T15:00:00Z", status="FINISHED",
                home_team="Arsenal", away_team="Everton", home_goals=2, away_goals=1,
            )]
        return []

    client.get_results.side_effect = fake_get_results
    conn = _duckdb_conn()

    lesson_ids = generate_daily_lessons(cache, store, client, conn, llm_invoke=None)

    assert len(lesson_ids) == 1
    assert store.list_all()[0].correct is True
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest app/backend/tests/test_live_lessons.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'app.backend.live_lessons'`.

- [ ] **Step 3: Create the module**

Create `app/backend/live_lessons.py`:

```python
"""W177: turns the app's own daily finished-match recommendations into
lesson candidates for src/agent/lessons.py's existing human-review pipeline
(A33/A39-A47) -- the same agent_lessons table agent-train writes to, sourced
here from live recommendation_outcomes (W167) instead of a backtest corpus.

Internal use only: every candidate this writes lands as status='pending',
exactly like a training-sourced one -- it only reaches live serving once a
human runs `agent-lessons approve <id> --scope ...` (main.py), unchanged.

Kept out of recommendation_stats.py (needs real DB I/O beyond pure
aggregation -- same separation agent_performance_dashboard.py already
established for its own DB-touching enrichment)."""

from __future__ import annotations

from collections import defaultdict
from typing import Callable

import duckdb

from app.backend.football_data_client import FootballDataClient
from app.backend.recommendation_cache import RecommendationCache
from app.backend.recommendation_outcomes import (
    RecommendationOutcome,
    RecommendationOutcomeStore,
    resolve_pending_recommendations,
)
from src.agent.backtest import BacktestRecord
from src.agent.lessons import generate_batch_lesson_text, generate_batch_reflection, insert_lesson_candidate
from src.agent.market_resolution import build_actual_outcome
from src.agent.schema import reported_teams
from src.logic.competition_registry import get_competition_definition
from src.utils.logger import get_logger

LOGGER = get_logger(__name__)

LIVE_SOURCE_NOTE = (
    "Live-sourced batch: reflects only the market actually recommended per "
    "match, not every market the agent evaluated."
)


def _to_lesson_record(outcome: RecommendationOutcome, cache: RecommendationCache) -> BacktestRecord:
    """Enrichment-complete adapter -- unlike recommendation_stats.py's own
    minimal _to_backtest_records (which only needs market_results for the
    Kelly simulation), generate_batch_lesson_text/generate_batch_reflection
    also read home_team/away_team/recommendation.{overall,confidence,
    explanation,limitations}/actual.result. A cache miss or a pre-migration
    outcome (competition_id/home_goals/away_goals all NULL, resolved before
    W175) degrades to blank fields rather than raising -- the record still
    joins its batch, just with less color, matching the dashboard's own
    degrade-one-row discipline (agent_performance_dashboard.py's
    _enrich_bet)."""
    entry = cache.get_latest_any_config(outcome.match_id, outcome.date)
    recommendation = entry.recommendation if entry is not None else {}
    teams = reported_teams(recommendation.get("match") or {}) if entry is not None else None
    home_team, away_team = teams if teams is not None else ("", "")
    actual = (
        build_actual_outcome(outcome.home_goals, outcome.away_goals)
        if outcome.home_goals is not None and outcome.away_goals is not None
        else {}
    )
    return BacktestRecord(
        match_id=outcome.match_id,
        home_team=home_team,
        away_team=away_team,
        date=outcome.date,
        league=outcome.competition_id or "",
        recommendation=recommendation,
        actual=actual,
        market_results=[{
            "market": outcome.market,
            "selection": outcome.selection,
            "correct": outcome.correct,
        }],
    )


def generate_daily_lessons(
    cache: RecommendationCache,
    store: RecommendationOutcomeStore,
    client: FootballDataClient,
    duckdb_conn: duckdb.DuckDBPyConnection,
    sweden_client: object | None = None,
    llm_invoke: Callable[[str], str] | None = None,
) -> list[int]:
    """The daily job body (wired by scheduler_wiring.py's
    register_lessons_job). Resolves pending outcomes first (W167) so this
    runs as one unattended pipeline, then batches whatever hasn't yet been
    folded into a lesson by (competition_id, date) -- one candidate per
    league per day, per direct user decision.

    llm_invoke=None skips generate_batch_reflection entirely (a stats-only
    candidate) -- used by callers that can't or don't want to pay for the
    LLM call (e.g. a fast unit test), not a distinct product mode."""
    resolve_pending_recommendations(cache, store, client, sweden_client)

    pending = store.list_unbatched_for_lessons()
    groups: dict[tuple[str, str], list[RecommendationOutcome]] = defaultdict(list)
    for outcome in pending:
        if outcome.competition_id is None:
            LOGGER.warning(
                "live_lessons: skipping outcome match_id=%s (date=%s) -- no verified "
                "competition_id (likely resolved before W175's migration).",
                outcome.match_id, outcome.date,
            )
            continue
        groups[(outcome.competition_id, outcome.date)].append(outcome)

    lesson_ids: list[int] = []
    for (competition_id, _date), group in groups.items():
        try:
            tier = get_competition_definition(competition_id).tier
        except ValueError:
            LOGGER.warning("live_lessons: skipping batch for unrecognized competition_id=%s.", competition_id)
            continue

        records = [_to_lesson_record(outcome, cache) for outcome in group]
        stats_text = generate_batch_lesson_text(records)
        lesson_text = f"{LIVE_SOURCE_NOTE} {stats_text}"
        if llm_invoke is not None:
            reflection = generate_batch_reflection(records, stats_text, llm_invoke)
            if reflection:
                lesson_text = f"{lesson_text}\n\nReflection: {reflection}"

        match_ids = ",".join(outcome.match_id for outcome in group)
        lesson_id = insert_lesson_candidate(duckdb_conn, lesson_text, competition_id, tier, match_ids)
        store.mark_lesson_batched([outcome.id for outcome in group])
        lesson_ids.append(lesson_id)

    return lesson_ids
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest app/backend/tests/test_live_lessons.py -v`
Expected: PASS, all 8 tests.

Then run the wider backend suite to confirm nothing else regressed:

Run: `pytest app/backend/tests/ tests/ -q`
Expected: PASS (same pre-existing unrelated failures as before this plan, if any — no new ones).

- [ ] **Step 5: Commit**

```bash
git add app/backend/live_lessons.py app/backend/tests/test_live_lessons.py
git commit -m "feat(app): live_lessons.py -- daily batch pipeline into agent_lessons (W177)

Co-Authored-By: Claude Sonnet 5 <noreply@anthropic.com>"
```

## Context

Task 3 of a 5-task plan. Depends on Tasks 1 and 2 (the `competition_id`/`home_goals`/`away_goals`/`lesson_batched_at` fields and `list_unbatched_for_lessons()`/`mark_lesson_batched()` methods this task calls). `src/agent/lessons.py` and `src/agent/backtest.py` are pre-existing, unmodified — read them if anything about `BacktestRecord`, `generate_batch_lesson_text`, or `generate_batch_reflection` is unclear rather than guessing their shape.

## Before You Begin

If anything above is unclear, ask now.

## Your Job

Implement exactly what's specified, run the tests, verify they pass, commit, self-review, report back.

Work from: /Users/tianqihuang/Documents/GitHub/FPAI

## Report Format

- **Status:** DONE | DONE_WITH_CONCERNS | BLOCKED | NEEDS_CONTEXT
- What you implemented, test results (paste output), files changed, self-review findings, commit SHA.

---

### Task 4: Scheduler wiring — `register_lessons_job()` + `main.py` registration (W178)

**Files:**
- Modify: `app/backend/scheduler_wiring.py`
- Modify: `app/backend/main.py`
- Test: `app/backend/tests/test_scheduler_wiring.py`

**Context for this task:** Registers the new daily job on the same `RecoverableScheduler` the EOD job already uses, at a distinct hour (`06:00` ET, vs. EOD's `23:00`), gated behind the same `ENABLE_SCHEDULER` check. The job needs a write-mode `DuckDBManager` (distinct from `lessons_node`'s own `read_only=True` live-serving connection) and an `llm_invoke` built from the same live default `AgentConfig` the app already loads at startup. `app/backend/` has never imported from the root `main.py` CLI script (confirmed: neither file imports the other today) — rather than make this job the first thing to cross that boundary, `_build_lessons_llm_invoke()` is a small, deliberate duplicate of `main.py`'s existing `_build_llm_invoke()` (6 lines, both just wrap `src.agent.graph._build_llm`/`_extract_text`).

- [ ] **Step 1: Write the failing tests**

In `app/backend/tests/test_scheduler_wiring.py`, add `EOD_HOUR`, `LESSONS_HOUR`, `LESSONS_JOB_ID`, `register_lessons_job` to the existing `from app.backend.scheduler_wiring import (...)` block, and add these two imports:

```python
from app.backend.recommendation_outcomes import RecommendationOutcomeStore
from src.utils.db_manager import DuckDBManager
```

Then add these tests at the end of the file:

```python
def test_register_lessons_job_runs_at_a_different_hour_than_eod() -> None:
    assert LESSONS_HOUR != EOD_HOUR


def test_register_lessons_job_generates_a_candidate_and_marks_the_scheduler_run(tmp_path: Path) -> None:
    cache = RecommendationCache(db_path=tmp_path / "cache.db")
    store = RecommendationOutcomeStore(db_path=tmp_path / "outcomes.db")
    store.insert(
        match_id="m1", date="2026-08-22", competition="Premier League", market="result_3way",
        selection="home", recommendation_type="direct_bet", confidence="medium", odds=2.0,
        value_edge=0.1, correct=True, generated_at="2026-08-22T10:00:00+00:00",
        competition_id="E0", home_goals=2, away_goals=1,
    )
    client = MagicMock()
    client.get_results.return_value = []
    duckdb_manager = DuckDBManager()
    duckdb_manager.db_path = tmp_path / "fpai_core.db"
    config = AgentConfig.default()
    run_log = JobRunLog(db_path=tmp_path / "job_runs.db")

    now = datetime(_FUTURE_DAY.year, _FUTURE_DAY.month, _FUTURE_DAY.day, 23, 30, tzinfo=NY_TZ)
    scheduler = RecoverableScheduler(run_log=run_log, now_fn=lambda: now)

    with patch("app.backend.scheduler_wiring._build_lessons_llm_invoke", return_value=None):
        register_lessons_job(
            scheduler, cache=cache, store=store, client=client,
            duckdb_manager=duckdb_manager, config=config,
        )
        assert _wait_until(lambda: run_log.has_run(LESSONS_JOB_ID, now.date().isoformat()))

    with duckdb_manager.connection(read_only=True) as conn:
        assert conn.execute("SELECT COUNT(*) FROM agent_lessons").fetchone()[0] == 1
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest app/backend/tests/test_scheduler_wiring.py -v -k lessons_job`
Expected: FAIL with `ImportError: cannot import name 'register_lessons_job'`.

- [ ] **Step 3: Add the wiring**

In `app/backend/scheduler_wiring.py`, add these imports near the top (alongside the existing ones):

```python
from app.backend.live_lessons import generate_daily_lessons
from app.backend.recommendation_outcomes import RecommendationOutcomeStore
from src.agent.lessons import create_lessons_tables
from src.utils.db_manager import DuckDBManager
```

Add the new constants next to `EOD_JOB_ID`/`EOD_HOUR`/`EOD_MINUTE`:

```python
LESSONS_JOB_ID = "daily_live_lessons"
LESSONS_HOUR = 6
LESSONS_MINUTE = 0
```

Add these two functions after `register_eod_job` (and its helpers) in the file:

```python
def register_lessons_job(
    scheduler: RecoverableScheduler,
    cache: RecommendationCache,
    store: RecommendationOutcomeStore,
    client: FootballDataClient,
    duckdb_manager: DuckDBManager,
    config: AgentConfig,
    sweden_client: object | None = None,
) -> None:
    """Registers the daily live-lessons job (W175-W178): resolves pending
    recommendation_outcomes (W167) then batches whatever's newly unbatched
    into agent_lessons candidates via live_lessons.generate_daily_lessons.
    Runs at LESSONS_HOUR (06:00 ET, distinct from EOD_HOUR's 23:00, and
    after football-data.org has typically posted the prior day's results)
    -- same schedule_daily restart/catch-up guarantee as the EOD job.

    duckdb_manager: a write-mode DuckDBManager (matches main.py's own
    `agent-lessons approve` CLI pattern) -- distinct from lessons_node's
    own read_only=True live-serving connection, since this job writes new
    pending rows."""

    def _lessons_job() -> None:
        llm_invoke = _build_lessons_llm_invoke(config)
        with duckdb_manager.connection() as conn:
            create_lessons_tables(conn)
            lesson_ids = generate_daily_lessons(cache, store, client, conn, sweden_client, llm_invoke)
        LOGGER.info("Daily live lessons: %d candidate(s) generated.", len(lesson_ids))

    scheduler.schedule_daily(LESSONS_JOB_ID, _lessons_job, hour=LESSONS_HOUR, minute=LESSONS_MINUTE)


def _build_lessons_llm_invoke(config: AgentConfig) -> Callable[[str], str]:
    """Deliberate small duplication of main.py's own _build_llm_invoke --
    app/backend/ has never imported from the root main.py CLI script (nor
    vice versa); a 6-line copy is a smaller, safer diff than making this
    job the first thing to cross that boundary. Keep in sync with
    main.py's _build_llm_invoke if its shape ever changes."""
    from src.agent.graph import _build_llm, _extract_text

    llm = _build_llm(config)

    def _invoke(prompt: str) -> str:
        response = llm.invoke(prompt)
        return _extract_text(response.content)

    return _invoke
```

In `app/backend/main.py`, update the scheduler_wiring import:

```python
from app.backend.scheduler_wiring import build_odds_client, build_schedule_t30, register_eod_job, register_lessons_job
```

And in `lifespan()`, right after the existing `register_eod_job(...)` call (still inside the `if os.environ.get("ENABLE_SCHEDULER", ...)` block, before `scheduler.start()`):

```python
        register_lessons_job(
            scheduler,
            cache=recommendations.get_cache(),
            store=get_recommendation_outcome_store(),
            client=get_fixtures_client(),
            duckdb_manager=DuckDBManager(),
            config=config,
            sweden_client=get_sweden_fixtures_client(),
        )
```

This needs `from src.utils.db_manager import DuckDBManager` added to `main.py`'s imports (`get_recommendation_outcome_store`/`get_fixtures_client`/`get_sweden_fixtures_client`/`recommendations.get_cache` are all already imported/used there for the existing EOD registration).

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest app/backend/tests/test_scheduler_wiring.py -v`
Expected: PASS, all tests including the 2 new ones.

Then run the full backend suite:

Run: `pytest app/backend/tests/ tests/ -q`
Expected: PASS (same pre-existing unrelated failures as before this plan, if any — no new ones).

- [ ] **Step 5: Commit**

```bash
git add app/backend/scheduler_wiring.py app/backend/main.py app/backend/tests/test_scheduler_wiring.py
git commit -m "feat(app): wire daily live-lessons job into the scheduler (W178)

Co-Authored-By: Claude Sonnet 5 <noreply@anthropic.com>"
```

## Context

Task 4 of a 5-task plan. Depends on Task 3's `app/backend/live_lessons.py`. This is the last implementation task — Task 5 is verification and documentation only.

## Before You Begin

If anything about the `main.py` lifespan wiring location is unclear, read the existing `register_eod_job(...)` call in `lifespan()` first — this task's addition sits immediately after it, same indentation, same `if` block. Ask if still unclear.

## Your Job

Implement exactly what's specified, run the tests, verify they pass, commit, self-review, report back.

Work from: /Users/tianqihuang/Documents/GitHub/FPAI

## Report Format

- **Status:** DONE | DONE_WITH_CONCERNS | BLOCKED | NEEDS_CONTEXT
- What you implemented, test results (paste output), files changed, self-review findings, commit SHA.

---

### Task 5: Full verification + mark stories completed

**Files:**
- Modify: `documents/app_user_stories.md`

This task is done by the plan's controller (not a fresh subagent) after a final whole-feature code review, mirroring the previous feature's own Task 8.

- [ ] **Step 1:** Dispatch a final code-quality reviewer subagent over the whole diff (all 4 tasks' commits) against the design doc, looking specifically for: (a) any place `src/agent/lessons.py` ended up modified after all (it should not be), (b) the `competition_id`/`home_goals`/`away_goals`/`lesson_batched_at` migration being genuinely idempotent against a real pre-existing `recommendation_outcomes.db`, (c) `mark_lesson_batched` never being called before its corresponding `insert_lesson_candidate` succeeds.

- [ ] **Step 2:** Run the full test suite:

Run: `pytest app/backend/tests/ tests/ -q`
Expected: PASS (same pre-existing unrelated failures noted in this session's earlier features — missing `data/fpai_core.db` in a worktree — no new failures).

- [ ] **Step 3:** Manual sanity check: with a real (or worktree-local) `data/fpai_core.db` present, construct a `RecommendationOutcomeStore` with a couple of `direct_bet` outcomes (one per league) and call `generate_daily_lessons(...)` directly against a real `DuckDBManager().connection()` — confirm real rows land in `agent_lessons` with `status='pending'` and readable `lesson_text`, and that `python main.py agent-lessons approve <id> --scope competition` (the pre-existing, unmodified CLI) can approve one of them without error.

- [ ] **Step 4:** Add a new `## PHASE 39: Daily Live-Recommendation Lessons` section to `documents/app_user_stories.md`, following the exact style of the immediately preceding phase (a short intro paragraph naming the direct user request and linking the design doc, then a table). Rows: `W175` (competition_id/home_goals/away_goals persistence fix), `W176` (idempotency marker), `W177` (`live_lessons.py`), `W178` (scheduler wiring) — each `completed`, with real test counts and any deviations found during implementation (fill in from each task's actual `DONE`/review reports, not invented).

- [ ] **Step 5:** Commit:

```bash
git add documents/app_user_stories.md
git commit -m "docs: mark W175-W178 completed with verification results

Co-Authored-By: Claude Sonnet 5 <noreply@anthropic.com>"
```

- [ ] **Step 6:** Proceed to `superpowers:finishing-a-development-branch`.

## Context

Final task of a 5-task plan. By this point all 4 implementation tasks are committed and individually reviewed; this task is the whole-feature gate before merge, matching the previous feature's own final-task pattern in this same session.

## Before You Begin

N/A — this task is executed by the plan's controller directly, not a fresh implementer subagent.

## Your Job

Run the verification steps, write the story rows accurately from the real implementation history, commit, then hand off to `finishing-a-development-branch`.

## Report Format

N/A (controller task).
