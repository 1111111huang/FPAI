# Autonomous Live-Lesson Judging (Phase 1) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** The agent autonomously approves/rejects each day's freshly-generated, live-deployment-sourced lesson candidates — never touching `agent-train`'s human-reviewed candidates.

**Architecture:** Two additive columns on `agent_lessons` (`source`, `auto_decision_reasoning`), a new conservative LLM-driven judge function reusing every existing distillation/conflict-check function unmodified, and a new three-phase step (`auto_judge_live_lessons`) appended to the daily job — mirroring the exact prepare/LLM/commit discipline the live-lessons feature's own Task 4 fix already established, so this never reintroduces "DuckDB lock held across a network/LLM call."

**Tech Stack:** Python, DuckDB (`agent_lessons` in `data/fpai_core.db`), the same `llm_invoke: str -> str` convention `src/agent/lessons.py` already uses throughout.

Design doc: `docs/superpowers/specs/2026-08-26-autonomous-live-lesson-judging-design.md`

---

## File Structure

- **Modify:** `src/agent/lessons.py` — `create_lessons_tables()` gains 2 more `ALTER TABLE ADD COLUMN IF NOT EXISTS` lines; `insert_lesson_candidate()` gains a `source: str = "train"` parameter; new `list_pending_by_source()`, `judge_lesson_candidate()`, `LessonDecision`, and a small private `_parse_judge_json()` helper.
- **Modify:** `app/backend/live_lessons.py` — `commit_lesson_batches()` passes `source="live"`; new `auto_judge_live_lessons()`.
- **Modify:** `app/backend/scheduler_wiring.py` — `_lessons_job()` calls `auto_judge_live_lessons()` after `commit_lesson_batches()`.
- **Test:** `tests/test_agent_lessons.py`, `app/backend/tests/test_live_lessons.py`, `app/backend/tests/test_scheduler_wiring.py`.

---

### Task 1: Schema + source plumbing (`src/agent/lessons.py`)

**Files:**
- Modify: `src/agent/lessons.py`
- Test: `tests/test_agent_lessons.py`

**Context for this task:** Purely mechanical: two new nullable columns on `agent_lessons`, one new parameter on `insert_lesson_candidate` (default preserves every existing caller unchanged), and one new query function. No LLM logic in this task — that's Task 2.

- [ ] **Step 1: Write the failing tests**

In `tests/test_agent_lessons.py`, add these tests. Find `test_create_lessons_tables_creates_both_tables` and update its `lesson_cols` assertion set to include the two new columns:

```python
def test_create_lessons_tables_creates_both_tables():
    conn = _conn()
    lesson_cols = {row[1] for row in conn.execute("PRAGMA table_info('agent_lessons')").fetchall()}
    telemetry_cols = {row[1] for row in conn.execute("PRAGMA table_info('agent_telemetry')").fetchall()}
    assert lesson_cols == {
        "id", "lesson_text", "status", "competition_id", "tier", "scope",
        "source_match_id", "created_at", "reviewed_at", "reviewer", "rule_text",
        "source", "auto_decision_reasoning",
    }
    assert telemetry_cols == {
        "match_id", "run_id", "competition_resolution", "research_evidence",
        "forecast_payload", "recommendation", "created_at",
    }
```

Then add these new tests after `test_insert_lesson_candidate_allows_null_competition_id`:

```python
def test_insert_lesson_candidate_defaults_source_to_train():
    conn = _conn()
    lesson_id = insert_lesson_candidate(conn, "text", "E0", "competition_specific", "m1")
    source = conn.execute("SELECT source FROM agent_lessons WHERE id = ?", [lesson_id]).fetchone()[0]
    assert source == "train"


def test_insert_lesson_candidate_accepts_explicit_live_source():
    conn = _conn()
    lesson_id = insert_lesson_candidate(conn, "text", "E0", "competition_specific", "m1", source="live")
    source = conn.execute("SELECT source FROM agent_lessons WHERE id = ?", [lesson_id]).fetchone()[0]
    assert source == "live"


def test_list_pending_by_source_excludes_other_sources_and_null():
    conn = _conn()
    live_id = insert_lesson_candidate(conn, "live text", "E0", "competition_specific", "m1", source="live")
    insert_lesson_candidate(conn, "train text", "E0", "competition_specific", "m2", source="train")
    # A pre-migration row -- source column exists but was never populated for it.
    conn.execute(
        "INSERT INTO agent_lessons (lesson_text, status, competition_id, tier, source_match_id, created_at) "
        "VALUES ('legacy text', 'pending', 'E0', 'competition_specific', 'm3', ?)",
        [datetime.now(timezone.utc)],
    )

    pending = list_pending_by_source(conn, source="live")

    assert [p["id"] for p in pending] == [live_id]
    assert pending[0]["lesson_text"] == "live text"
    assert pending[0]["competition_id"] == "E0"
    assert pending[0]["tier"] == "competition_specific"


def test_list_pending_by_source_excludes_already_reviewed_rows():
    conn = _conn()
    live_id = insert_lesson_candidate(conn, "live text", "E0", "competition_specific", "m1", source="live")
    reject_lesson(conn, live_id, reviewer="test")

    assert list_pending_by_source(conn, source="live") == []
```

`datetime`/`timezone` need importing at the top of the test file if not already present — check first (`from datetime import datetime, timezone`).

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_agent_lessons.py -v`
Expected: the updated `test_create_lessons_tables_creates_both_tables` fails on the set comparison; the 4 new tests fail with `TypeError: insert_lesson_candidate() got an unexpected keyword argument 'source'` or `ImportError: cannot import name 'list_pending_by_source'`.

- [ ] **Step 3: Add the columns, parameter, and query function**

In `src/agent/lessons.py`, add two lines to `create_lessons_tables()`, right after the existing `rule_text` migration line:

```python
    conn.execute("ALTER TABLE agent_lessons ADD COLUMN IF NOT EXISTS rule_text TEXT")
    # 2026-08-26 (autonomous live-lesson judging, Phase 1): source
    # distinguishes an agent-train-sourced candidate ('train', the default
    # below and the only source that ever existed before this) from a
    # live-deployment-sourced one ('live', live_lessons.py's
    # commit_lesson_batches). auto_decision_reasoning is the audit trail
    # for an autonomous approve/reject decision -- a human reviewer's own
    # judgment call is visible in the CLI transcript; this is the
    # equivalent for a decision nobody watched happen. Both nullable/
    # additive -- a pre-migration row simply carries NULL for both, and
    # NULL is never matched by `source = 'live'` (SQL semantics), so
    # existing rows are structurally excluded from live-lesson-only
    # queries like list_pending_by_source() below, not just conventionally.
    conn.execute("ALTER TABLE agent_lessons ADD COLUMN IF NOT EXISTS source TEXT")
    conn.execute("ALTER TABLE agent_lessons ADD COLUMN IF NOT EXISTS auto_decision_reasoning TEXT")
```

Then replace `insert_lesson_candidate`:

```python
def insert_lesson_candidate(
    conn: duckdb.DuckDBPyConnection,
    lesson_text: str,
    competition_id: str | None,
    tier: str,
    source_match_id: str,
    source: str = "train",
) -> int:
    """Insert a pending, unscoped lesson candidate. Returns its id.

    source: 'train' (default, preserves every pre-existing caller
    unchanged -- agent-train's own CLI path) or 'live' (live_lessons.py's
    commit_lesson_batches, the only caller that passes this explicitly)."""
    row = conn.execute(
        """
        INSERT INTO agent_lessons (lesson_text, status, competition_id, tier, source_match_id, created_at, source)
        VALUES (?, 'pending', ?, ?, ?, ?, ?)
        RETURNING id
        """,
        [lesson_text, competition_id, tier, source_match_id, datetime.now(timezone.utc), source],
    ).fetchone()
    return int(row[0])
```

Then add `list_pending_by_source` after `load_approved_lessons`:

```python
def list_pending_by_source(conn: duckdb.DuckDBPyConnection, source: str) -> list[dict[str, Any]]:
    """Every pending lesson candidate from one source ('train'/'live') --
    used by live_lessons.py's auto_judge_live_lessons() to find only its
    own population. WHERE source = ? naturally excludes both the other
    source and any pre-migration row (source IS NULL, since SQL's
    `NULL = 'live'` is never true) -- agent-train's human-reviewed queue is
    structurally unreachable from here, not just conventionally avoided."""
    rows = conn.execute(
        "SELECT id, lesson_text, competition_id, tier FROM agent_lessons WHERE status = 'pending' AND source = ?",
        [source],
    ).fetchall()
    return [{"id": row[0], "lesson_text": row[1], "competition_id": row[2], "tier": row[3]} for row in rows]
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_agent_lessons.py -v`
Expected: PASS, all tests including the 4 new ones and the updated column-set assertion.

Then run the wider suite to confirm no existing `insert_lesson_candidate` caller broke:

Run: `pytest tests/test_agent_lessons.py tests/test_main_agent_lessons.py app/backend/tests/test_live_lessons.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/agent/lessons.py tests/test_agent_lessons.py
git commit -m "feat(agent): source/auto_decision_reasoning columns + list_pending_by_source on agent_lessons

Co-Authored-By: Claude Sonnet 5 <noreply@anthropic.com>"
```

## Context

Task 1 of a 5-task plan. Depends on nothing new (extends A33/A44's existing `agent_lessons` table and `src/agent/lessons.py` module, both pre-existing and untouched in shape). This task is purely additive plumbing — no LLM logic here, that's Task 2.

## Before You Begin

If anything above is unclear, ask now.

## Your Job

Implement exactly what's specified, run the tests, verify they pass, commit, self-review, report back.

Work from: /Users/tianqihuang/Documents/GitHub/FPAI

## Report Format

- **Status:** DONE | DONE_WITH_CONCERNS | BLOCKED | NEEDS_CONTEXT
- What you implemented, test results (paste output), files changed, self-review findings, commit SHA.

---

### Task 2: The judge (`judge_lesson_candidate` in `src/agent/lessons.py`)

**Files:**
- Modify: `src/agent/lessons.py`
- Test: `tests/test_agent_lessons.py`

**Context for this task:** This is the one genuinely new piece of LLM logic. Given `lesson_text`, `competition_id`, and `tier`, it decides approve/reject and (if approving) scope — the same decision a human currently makes via `agent-lessons approve --scope`. This is the FIRST JSON-structured LLM call in this module (`generate_batch_reflection`/`generate_rule_from_lesson`/`find_conflicting_rule` all just return/parse plain prose) — the LLM might still wrap its JSON in a markdown code fence despite instructions not to, so parsing needs a small defensive unwrap, not a bare `json.loads`.

**Depends on:** Task 1 (this task doesn't touch the schema, but it's most naturally reviewed after the plumbing it will eventually be wired to already exists — no hard code dependency either way, safe to implement immediately after Task 1 lands).

- [ ] **Step 1: Write the failing tests**

In `tests/test_agent_lessons.py`, add `from dataclasses import` is NOT needed in the test file (only in `lessons.py` itself) — just add `LessonDecision`, `judge_lesson_candidate` to the existing `from src.agent.lessons import (...)` block, then add these tests after the `find_conflicting_rule` tests:

```python
def test_judge_lesson_candidate_parses_a_plain_json_approval():
    captured = {}

    def fake_invoke(prompt: str) -> str:
        captured["prompt"] = prompt
        return '{"approve": true, "scope": "competition", "reasoning": "Clear systematic pattern."}'

    decision = judge_lesson_candidate("WHEN evaluating a batch...", "E0", "competition_specific", fake_invoke)

    assert decision.approve is True
    assert decision.scope == "competition"
    assert decision.reasoning == "Clear systematic pattern."
    assert "E0" in captured["prompt"]
    assert "competition_specific" in captured["prompt"]


def test_judge_lesson_candidate_parses_json_wrapped_in_a_markdown_fence():
    def fake_invoke(prompt: str) -> str:
        return '```json\n{"approve": false, "scope": null, "reasoning": "Sample too thin."}\n```'

    decision = judge_lesson_candidate("WHEN evaluating a batch...", "E0", "competition_specific", fake_invoke)

    assert decision.approve is False
    assert decision.scope is None
    assert decision.reasoning == "Sample too thin."


def test_judge_lesson_candidate_rejects_on_invalid_scope_value():
    def fake_invoke(prompt: str) -> str:
        return '{"approve": true, "scope": "everywhere", "reasoning": "Broad pattern."}'

    decision = judge_lesson_candidate("some lesson", "E0", "competition_specific", fake_invoke)

    assert decision.approve is False
    assert decision.scope is None
    assert "everywhere" in decision.reasoning


def test_judge_lesson_candidate_rejects_on_malformed_json():
    def fake_invoke(prompt: str) -> str:
        return "not json at all"

    decision = judge_lesson_candidate("some lesson", "E0", "competition_specific", fake_invoke)

    assert decision.approve is False
    assert decision.scope is None


def test_judge_lesson_candidate_rejects_on_llm_exception():
    def failing_invoke(prompt: str) -> str:
        raise RuntimeError("API down")

    decision = judge_lesson_candidate("some lesson", "E0", "competition_specific", failing_invoke)

    assert decision.approve is False
    assert "API down" in decision.reasoning
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_agent_lessons.py -v -k judge_lesson_candidate`
Expected: FAIL with `ImportError: cannot import name 'judge_lesson_candidate'`.

- [ ] **Step 3: Add `LessonDecision`, `_parse_judge_json`, and `judge_lesson_candidate`**

In `src/agent/lessons.py`, add `from dataclasses import dataclass` to the imports at the top (alongside the existing `import json`/`from collections import Counter`/etc.).

Add these three pieces after `find_conflicting_rule`:

```python
@dataclass
class LessonDecision:
    approve: bool
    scope: str | None  # "competition" | "tier", only set when approve=True
    reasoning: str      # always set -- the audit trail (agent_lessons.auto_decision_reasoning)


def _parse_judge_json(raw: str) -> dict[str, Any]:
    """Defensive unwrap for judge_lesson_candidate()'s response -- the
    first JSON-structured LLM call in this module (every other function
    here parses/returns plain prose). An LLM can still wrap valid JSON in
    a markdown code fence despite an explicit "output nothing else"
    instruction; strip one if present, then parse normally. Raises
    json.JSONDecodeError (uncaught) on genuinely malformed input --
    judge_lesson_candidate is responsible for catching that, matching this
    module's existing per-function (not shared) error-handling convention."""
    text = raw.strip()
    if text.startswith("```"):
        text = text.strip("`")
        if text.lower().startswith("json"):
            text = text[4:]
        text = text.strip()
    return json.loads(text)


def judge_lesson_candidate(
    lesson_text: str, competition_id: str | None, tier: str, llm_invoke: Callable[[str], str],
) -> LessonDecision:
    """2026-08-26 (autonomous live-lesson judging, Phase 1): decides
    whether a live-deployment-sourced lesson candidate (see
    live_lessons.py) is worth turning into a durable rule, and if so
    whether it should apply narrowly (this competition) or broadly (every
    competition sharing this tier) -- the same decision a human reviewer
    makes via `agent-lessons approve --scope`. Never called for a
    train-sourced candidate -- see live_lessons.py's auto_judge_live_lessons
    for the source='live'-only query this feeds from.

    Deliberately conservative: live-deployment batches are small (W177
    batches one candidate per league per day), so the prompt is explicitly
    instructed to default to reject on a thin sample or a pattern that
    isn't clearly systematic -- a bad rule silently baked into the live
    prompt is worse than one more day with no new rule. Mirrors the live
    agent's own already-investigated conservative posture on value edges
    (A71's DeepSeek decline-bias findings) -- conservative-by-default is
    this system's existing house style, not a new invention.

    Returns approve=False with a reasoning string on ANY failure (a raised
    exception, malformed JSON, or an invalid scope value) -- fail-closed,
    matching every other LLM-driven function in this module's posture on
    the safe side of a coin flip."""
    prompt = (
        f"You are deciding whether to promote a batch of live betting-recommendation results into a "
        f"standing rule for an automated agent's future recommendations in this competition "
        f"(competition_id={competition_id!r}, tier={tier!r}).\n\n"
        f"{lesson_text}\n\n"
        "Only approve if the pattern is clearly systematic, not noise from a small sample -- when in "
        "doubt, reject. If you approve, also decide scope: \"competition\" if the pattern is specific to "
        "this one competition, \"tier\" if it reflects something general enough to apply to every "
        "competition of this tier.\n\n"
        "Respond with exactly one JSON object, nothing else: "
        '{"approve": true|false, "scope": "competition"|"tier"|null, "reasoning": "one or two sentences"}'
    )
    try:
        parsed = _parse_judge_json(llm_invoke(prompt))
        approve = bool(parsed["approve"])
        scope = parsed.get("scope") if approve else None
        reasoning = str(parsed.get("reasoning") or "").strip() or "(no reasoning given)"
        if approve and scope not in _VALID_SCOPES:
            return LessonDecision(approve=False, scope=None, reasoning=f"invalid scope {scope!r} returned -- defaulting to reject")
        return LessonDecision(approve=approve, scope=scope, reasoning=reasoning)
    except Exception as exc:
        return LessonDecision(approve=False, scope=None, reasoning=f"judge call failed ({exc!r}) -- defaulting to reject")
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_agent_lessons.py -v`
Expected: PASS, all tests including the 5 new ones.

- [ ] **Step 5: Commit**

```bash
git add src/agent/lessons.py tests/test_agent_lessons.py
git commit -m "feat(agent): judge_lesson_candidate -- conservative autonomous approve/reject/scope decision

Co-Authored-By: Claude Sonnet 5 <noreply@anthropic.com>"
```

## Context

Task 2 of a 5-task plan. Builds on Task 1's schema but has no hard code dependency on it -- this task is pure function logic, testable in isolation with a fake `llm_invoke`.

## Before You Begin

If anything above is unclear, ask now.

## Your Job

Implement exactly what's specified, run the tests, verify they pass, commit, self-review, report back.

Work from: /Users/tianqihuang/Documents/GitHub/FPAI

## Report Format

- **Status:** DONE | DONE_WITH_CONCERNS | BLOCKED | NEEDS_CONTEXT
- What you implemented, test results (paste output), files changed, self-review findings, commit SHA.

---

### Task 3: `auto_judge_live_lessons()` in `app/backend/live_lessons.py`

**Files:**
- Modify: `app/backend/live_lessons.py`
- Test: `app/backend/tests/test_live_lessons.py`

**Context for this task:** Wires Tasks 1+2's building blocks into the actual daily-job step. Three phases, matching `prepare_lesson_batches`/`commit_lesson_batches`'s own discipline exactly: a brief read (fetch pending `source='live'` candidates), all LLM work (judge, and for an approval, distill + conflict-check) with zero DuckDB connections open, then a brief write (apply every decision). `commit_lesson_batches()` also needs one small change first: it must pass `source="live"` to `insert_lesson_candidate`, so this task's own new function has something to find.

- [ ] **Step 1: Write the failing tests**

In `app/backend/tests/test_live_lessons.py`, add `auto_judge_live_lessons` to the existing `from app.backend.live_lessons import (...)` import line, and add these two new imports:

```python
from src.utils.db_manager import DuckDBManager
```

(`duckdb` is already imported in this file.)

Then add these tests at the end of the file:

```python
def test_commit_lesson_batches_writes_source_live(tmp_path: Path) -> None:
    store = RecommendationOutcomeStore(db_path=tmp_path / "outcomes.db")
    conn = _duckdb_conn()
    batch = PreparedLessonBatch(
        competition_id="E0", tier="competition_specific", lesson_text="Live-sourced batch: test.",
        match_ids="m1", outcome_ids=[],
    )

    commit_lesson_batches(conn, store, [batch])

    source = conn.execute("SELECT source FROM agent_lessons").fetchone()[0]
    assert source == "live"


def _dm(tmp_path: Path) -> DuckDBManager:
    dm = DuckDBManager()
    dm.db_path = tmp_path / "fpai_core.db"
    return dm


def test_auto_judge_live_lessons_approves_a_good_candidate(tmp_path: Path) -> None:
    dm = _dm(tmp_path)
    store = RecommendationOutcomeStore(db_path=tmp_path / "outcomes.db")
    with dm.connection() as conn:
        create_lessons_tables(conn)
        insert_lesson_candidate(conn, "Live-sourced batch: strong pattern.", "E0", "competition_specific", "m1", source="live")

    def fake_invoke(prompt: str) -> str:
        if "deciding whether to promote" in prompt:
            return '{"approve": true, "scope": "competition", "reasoning": "Clear pattern."}'
        if "Extract ONLY the single most programmatic" in prompt:
            return "NEVER bet result_3way on a thin sample."
        if "checking a new proposed rule" in prompt:
            return "NONE"
        raise AssertionError(f"unexpected prompt: {prompt}")

    outcomes = auto_judge_live_lessons(dm, fake_invoke)

    assert len(outcomes) == 1
    with dm.connection(read_only=True) as conn:
        row = conn.execute(
            "SELECT status, scope, rule_text, reviewer, auto_decision_reasoning FROM agent_lessons"
        ).fetchone()
    assert row[0] == "approved"
    assert row[1] == "competition"
    assert row[2] == "NEVER bet result_3way on a thin sample."
    assert row[3] == "agent-auto"
    assert row[4] == "Clear pattern."


def test_auto_judge_live_lessons_rejects_a_weak_candidate(tmp_path: Path) -> None:
    dm = _dm(tmp_path)
    with dm.connection() as conn:
        create_lessons_tables(conn)
        insert_lesson_candidate(conn, "Live-sourced batch: n=1.", "E0", "competition_specific", "m1", source="live")

    def fake_invoke(prompt: str) -> str:
        return '{"approve": false, "scope": null, "reasoning": "Sample too thin."}'

    auto_judge_live_lessons(dm, fake_invoke)

    with dm.connection(read_only=True) as conn:
        row = conn.execute("SELECT status, reviewer, auto_decision_reasoning FROM agent_lessons").fetchone()
    assert row[0] == "rejected"
    assert row[1] == "agent-auto"
    assert row[2] == "Sample too thin."


def test_auto_judge_live_lessons_leaves_a_conflict_pending(tmp_path: Path) -> None:
    dm = _dm(tmp_path)
    with dm.connection() as conn:
        create_lessons_tables(conn)
        # An already-approved rule in the same scope to conflict with.
        existing_id = insert_lesson_candidate(conn, "existing text", "E0", "competition_specific", "m0", source="train")
        approve_lesson(conn, existing_id, scope="competition", reviewer="test", rule_text="ALWAYS bet result_3way when confident.")
        insert_lesson_candidate(conn, "Live-sourced batch: new pattern.", "E0", "competition_specific", "m1", source="live")

    def fake_invoke(prompt: str) -> str:
        if "deciding whether to promote" in prompt:
            return '{"approve": true, "scope": "competition", "reasoning": "Looks solid."}'
        if "Extract ONLY the single most programmatic" in prompt:
            return "NEVER bet result_3way when confident."
        if "checking a new proposed rule" in prompt:
            return "Conflicts with rule 1: it recommends the exact opposite action."
        raise AssertionError(f"unexpected prompt: {prompt}")

    auto_judge_live_lessons(dm, fake_invoke)

    with dm.connection(read_only=True) as conn:
        rows = conn.execute(
            "SELECT status, auto_decision_reasoning FROM agent_lessons WHERE source = 'live'"
        ).fetchall()
    assert rows[0][0] == "pending"
    assert "Conflicts with rule 1" in rows[0][1]


def test_auto_judge_live_lessons_leaves_pending_on_distillation_failure(tmp_path: Path) -> None:
    dm = _dm(tmp_path)
    with dm.connection() as conn:
        create_lessons_tables(conn)
        insert_lesson_candidate(conn, "Live-sourced batch: pattern.", "E0", "competition_specific", "m1", source="live")

    def fake_invoke(prompt: str) -> str:
        if "deciding whether to promote" in prompt:
            return '{"approve": true, "scope": "competition", "reasoning": "Looks solid."}'
        if "Extract ONLY the single most programmatic" in prompt:
            return "   "  # blank -- generate_rule_from_lesson returns None for this
        raise AssertionError(f"unexpected prompt: {prompt}")

    auto_judge_live_lessons(dm, fake_invoke)

    with dm.connection(read_only=True) as conn:
        row = conn.execute("SELECT status, auto_decision_reasoning FROM agent_lessons").fetchone()
    assert row[0] == "pending"
    assert "distillation failed" in row[1]


def test_auto_judge_live_lessons_never_touches_train_sourced_rows(tmp_path: Path) -> None:
    dm = _dm(tmp_path)
    with dm.connection() as conn:
        create_lessons_tables(conn)
        train_id = insert_lesson_candidate(conn, "train text", "E0", "competition_specific", "m1", source="train")

    def fake_invoke(prompt: str) -> str:
        raise AssertionError("should never be called -- no live-sourced pending rows exist")

    outcomes = auto_judge_live_lessons(dm, fake_invoke)

    assert outcomes == []
    with dm.connection(read_only=True) as conn:
        status = conn.execute("SELECT status FROM agent_lessons WHERE id = ?", [train_id]).fetchone()[0]
    assert status == "pending"


def test_auto_judge_live_lessons_is_a_noop_when_llm_invoke_is_none(tmp_path: Path) -> None:
    dm = _dm(tmp_path)
    with dm.connection() as conn:
        create_lessons_tables(conn)
        insert_lesson_candidate(conn, "text", "E0", "competition_specific", "m1", source="live")

    outcomes = auto_judge_live_lessons(dm, None)

    assert outcomes == []
    with dm.connection(read_only=True) as conn:
        status = conn.execute("SELECT status FROM agent_lessons").fetchone()[0]
    assert status == "pending"
```

Also add `create_lessons_tables`, `insert_lesson_candidate`, `approve_lesson` to the test file's existing `from src.agent.lessons import (...)` line (currently just `create_lessons_tables`).

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest app/backend/tests/test_live_lessons.py -v -k "auto_judge or source_live"`
Expected: FAIL with `ImportError: cannot import name 'auto_judge_live_lessons'`.

- [ ] **Step 3: Update `commit_lesson_batches` and add `auto_judge_live_lessons`**

In `app/backend/live_lessons.py`, update the imports:

```python
from src.agent.lessons import (
    find_conflicting_rule,
    generate_batch_lesson_text,
    generate_batch_reflection,
    generate_rule_from_lesson,
    insert_lesson_candidate,
    judge_lesson_candidate,
    list_pending_by_source,
    load_approved_lessons,
)
```

(replaces the current single-line `from src.agent.lessons import generate_batch_lesson_text, generate_batch_reflection, insert_lesson_candidate`)

Also add:

```python
from src.utils.db_manager import DuckDBManager
```

In `commit_lesson_batches`, change the `insert_lesson_candidate` call to pass `source="live"`:

```python
        lesson_id = insert_lesson_candidate(
            duckdb_conn, batch.lesson_text, batch.competition_id, batch.tier, batch.match_ids, source="live",
        )
```

Then add `auto_judge_live_lessons` at the end of the file:

```python
def auto_judge_live_lessons(
    duckdb_manager: DuckDBManager,
    llm_invoke: Callable[[str], str] | None,
) -> list[dict[str, Any]]:
    """2026-08-26 (autonomous live-lesson judging, Phase 1): approves/
    rejects each pending source='live' candidate right after it's created,
    mirroring what a human currently decides via `agent-lessons
    approve/reject` -- but never touches source='train' (or pre-migration
    source IS NULL) rows, which stay 100% human-reviewed (list_pending_by_source
    structurally excludes them, not just conventionally).

    llm_invoke=None means the daily job's own LLM client failed to build
    (see scheduler_wiring.py's register_lessons_job try/except) -- there's
    no judging without an LLM, so this is a no-op; every pending candidate
    is simply left for the next day's run to judge once the LLM client is
    available again.

    Three phases, same discipline as prepare_lesson_batches/
    commit_lesson_batches: a brief read, all LLM work with no DuckDB
    connection open, then a brief write. Returns a list of dicts (one per
    candidate processed) for logging -- {id, action, reasoning}."""
    if llm_invoke is None:
        return []

    with duckdb_manager.connection(read_only=True) as conn:
        pending = list_pending_by_source(conn, source="live")

    results: list[dict[str, Any]] = []
    for candidate in pending:
        decision = judge_lesson_candidate(
            candidate["lesson_text"], candidate["competition_id"], candidate["tier"], llm_invoke,
        )
        action = "reject" if not decision.approve else "approve"
        scope = decision.scope
        rule_text: str | None = None
        reasoning = decision.reasoning

        if decision.approve:
            rule_text = generate_rule_from_lesson(candidate["lesson_text"], llm_invoke)
            if rule_text is None:
                action = "defer"
                reasoning = f"{decision.reasoning} (rule distillation failed -- left pending for retry)"
            else:
                with duckdb_manager.connection(read_only=True) as conn:
                    existing_rules = load_approved_lessons(conn, candidate["competition_id"], candidate["tier"])
                conflict = find_conflicting_rule(rule_text, existing_rules, llm_invoke)
                if conflict is not None:
                    action = "defer"
                    reasoning = f"Would approve, but a conflict was found: {conflict}"

        results.append({
            "id": candidate["id"], "action": action, "scope": scope,
            "rule_text": rule_text, "reasoning": reasoning,
        })

    with duckdb_manager.connection() as conn:
        for result in results:
            if result["action"] == "approve":
                approve_lesson(conn, result["id"], result["scope"], reviewer="agent-auto", rule_text=result["rule_text"])
            elif result["action"] == "reject":
                reject_lesson(conn, result["id"], reviewer="agent-auto")
            # "defer" -- leave status as-is, just record the reasoning below.
            conn.execute(
                "UPDATE agent_lessons SET auto_decision_reasoning = ? WHERE id = ?",
                [result["reasoning"], result["id"]],
            )

    return results
```

This needs `approve_lesson`/`reject_lesson` added to the `from src.agent.lessons import (...)` block above too (they weren't in the list given above — the full final import line is `find_conflicting_rule, generate_batch_lesson_text, generate_batch_reflection, generate_rule_from_lesson, insert_lesson_candidate, judge_lesson_candidate, list_pending_by_source, load_approved_lessons, approve_lesson, reject_lesson`, alphabetized however this file's existing convention sorts them — check the file's own existing import-sorting style, e.g. via `isort`/`ruff` config, and match it rather than guessing).

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest app/backend/tests/test_live_lessons.py -v`
Expected: PASS, all tests including the 8 new ones (2 for `commit_lesson_batches`'s `source="live"` + 6 for `auto_judge_live_lessons`... actually count exactly from what's above: `test_commit_lesson_batches_writes_source_live` + 6 `auto_judge_live_lessons` tests = 7 new tests).

Then run the wider backend suite:

Run: `pytest app/backend/tests/ tests/ -q`
Expected: PASS (the same 2 pre-existing unrelated failures noted in the live-lessons feature's own tasks, if run in a worktree missing `data/fpai_core.db` -- no new ones).

- [ ] **Step 5: Commit**

```bash
git add app/backend/live_lessons.py app/backend/tests/test_live_lessons.py
git commit -m "feat(app): auto_judge_live_lessons -- three-phase autonomous approve/reject for live-sourced candidates

Co-Authored-By: Claude Sonnet 5 <noreply@anthropic.com>"
```

## Context

Task 3 of a 5-task plan. Depends on Task 1 (`list_pending_by_source`, `source` param) and Task 2 (`judge_lesson_candidate`, `LessonDecision`) both already landed.

## Before You Begin

If anything above is unclear, ask now. In particular: verify the exact final shape of the `from src.agent.lessons import (...)` block against this file's existing formatting convention (check whether the project's `ruff`/`isort` config sorts these alphabetically or by some other rule) rather than copying the illustrative list above verbatim if it doesn't match.

## Your Job

Implement exactly what's specified, run the tests, verify they pass, commit, self-review, report back.

Work from: /Users/tianqihuang/Documents/GitHub/FPAI

## Report Format

- **Status:** DONE | DONE_WITH_CONCERNS | BLOCKED | NEEDS_CONTEXT
- What you implemented, test results (paste output), files changed, self-review findings, commit SHA.

---

### Task 4: Wire into the daily job (`app/backend/scheduler_wiring.py`)

**Files:**
- Modify: `app/backend/scheduler_wiring.py`
- Test: `app/backend/tests/test_scheduler_wiring.py`

**Context for this task:** The smallest task in this plan -- one new import, one new call, appended after the existing `commit_lesson_batches` block (outside its own `with duckdb_manager.connection()` block, so `auto_judge_live_lessons`'s own three-phase connection handling starts fresh rather than nesting inside an already-open one).

- [ ] **Step 1: Write the failing tests**

In `app/backend/tests/test_scheduler_wiring.py`, add `insert_lesson_candidate` to whatever `src.agent.lessons` import already exists in this file (check first -- `create_lessons_tables` is likely already imported for the existing `test_register_lessons_job_generates_a_candidate_and_marks_the_scheduler_run` test; add `insert_lesson_candidate` alongside it).

Add this test at the end of the file:

```python
def test_register_lessons_job_auto_judges_generated_candidates(tmp_path: Path) -> None:
    """End-to-end: a finished match generates a candidate, and that same
    job run judges it too -- proves auto_judge_live_lessons is actually
    reachable from the real daily job, not just unit-tested in isolation."""
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

    def fake_llm_invoke(prompt: str) -> str:
        return '{"approve": false, "scope": null, "reasoning": "Single-match sample, too thin to judge."}'

    with patch("app.backend.scheduler_wiring._build_lessons_llm_invoke", return_value=fake_llm_invoke):
        register_lessons_job(
            scheduler, cache=cache, store=store, client=client,
            duckdb_manager=duckdb_manager, config=config,
        )
        assert _wait_until(lambda: run_log.has_run(LESSONS_JOB_ID, now.date().isoformat()))

    with duckdb_manager.connection(read_only=True) as conn:
        row = conn.execute("SELECT status, source, auto_decision_reasoning FROM agent_lessons").fetchone()
    assert row[0] == "rejected"
    assert row[1] == "live"
    assert row[2] == "Single-match sample, too thin to judge."
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest app/backend/tests/test_scheduler_wiring.py -v -k auto_judges`
Expected: FAIL — the auto-judge step doesn't run yet, so the row stays `status='pending'` with `auto_decision_reasoning IS NULL`, failing the assertions.

- [ ] **Step 3: Wire it in**

In `app/backend/scheduler_wiring.py`, update the import:

```python
from app.backend.live_lessons import auto_judge_live_lessons, commit_lesson_batches, prepare_lesson_batches
```

(replaces the current `from app.backend.live_lessons import commit_lesson_batches, prepare_lesson_batches`)

In `_lessons_job()`, add the call right after the existing `LOGGER.info("Daily live lessons: %d candidate(s) generated.", len(lesson_ids))` line:

```python
        LOGGER.info("Daily live lessons: %d candidate(s) generated.", len(lesson_ids))

        judged = auto_judge_live_lessons(duckdb_manager, llm_invoke)
        LOGGER.info("Daily live lessons: %d candidate(s) auto-judged.", len(judged))
```

Note this is called with the SAME `llm_invoke` already built earlier in `_lessons_job()` (or `None`, if the try/except above caught a build failure) -- `auto_judge_live_lessons` already handles `llm_invoke=None` as a clean no-op per Task 3, so no additional guard is needed here.

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest app/backend/tests/test_scheduler_wiring.py -v`
Expected: PASS, all tests including the new one.

Then run the full backend suite:

Run: `pytest app/backend/tests/ tests/ -q`
Expected: PASS (same pre-existing unrelated failures as always, no new ones).

Also confirm the app boots cleanly: `python -c "from app.backend.main import app"` should exit 0.

- [ ] **Step 5: Commit**

```bash
git add app/backend/scheduler_wiring.py app/backend/tests/test_scheduler_wiring.py
git commit -m "feat(app): wire auto_judge_live_lessons into the daily lessons job

Co-Authored-By: Claude Sonnet 5 <noreply@anthropic.com>"
```

## Context

Task 4 of a 5-task plan. Depends on Task 3's `auto_judge_live_lessons` already landing. This is the last implementation task -- Task 5 is verification and documentation only.

## Before You Begin

If anything above is unclear, ask now.

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

This task is done by the plan's controller (not a fresh subagent) after a final whole-feature code review, mirroring the previous two features' own final task in this session.

- [ ] **Step 1:** Dispatch a final code-quality reviewer subagent over the whole diff (all 4 tasks' commits) against the design doc, looking specifically for: (a) `source='train'` rows genuinely unreachable from every new code path (not just from `list_pending_by_source`, but confirm `auto_judge_live_lessons` never queries `agent_lessons` any other way); (b) the conflict-blocks-approval invariant actually holds in every code path, not just the happy path; (c) whether the new judge step reintroduces any DuckDB-lock-across-LLM-call risk (re-check with fresh eyes, since this is exactly the class of bug Task 4 of the prior feature found).
- [ ] **Step 2:** Run the full test suite: `pytest app/backend/tests/ tests/ -q`. Expected: PASS, same pre-existing unrelated failures, no new ones.
- [ ] **Step 3:** Manual sanity check: with a real (or worktree-local) `data/fpai_core.db`, insert a `source='live'` pending candidate directly, call `auto_judge_live_lessons(DuckDBManager(), real_llm_invoke)` against a real LLM client (reuse `_build_lessons_llm_invoke(AgentConfig.default())`), and confirm a real decision lands with `auto_decision_reasoning` populated. Clean up the test row afterward (same real-vs-test-data discipline this codebase already follows elsewhere -- verify by exact `id` before deleting).
- [ ] **Step 4:** Add a new `## PHASE 40: Autonomous Live-Lesson Judging (Phase 1)` section to `documents/app_user_stories.md`, following the exact style of the immediately preceding phase. Rows for the 4 tasks (IDs continuing from W178: `W179`-`W182`), each `completed`, with real test counts and any deviations found during implementation.
- [ ] **Step 5:** Commit:

```bash
git add documents/app_user_stories.md
git commit -m "docs: mark W179-W182 completed with verification results

Co-Authored-By: Claude Sonnet 5 <noreply@anthropic.com>"
```

- [ ] **Step 6:** Proceed to `superpowers:finishing-a-development-branch`.

## Context

Final task of a 5-task plan. By this point all 4 implementation tasks are committed and individually reviewed; this task is the whole-feature gate before merge.

## Before You Begin

N/A -- this task is executed by the plan's controller directly, not a fresh implementer subagent.

## Your Job

Run the verification steps, write the story rows accurately from the real implementation history, commit, then hand off to `finishing-a-development-branch`.

## Report Format

N/A (controller task).
