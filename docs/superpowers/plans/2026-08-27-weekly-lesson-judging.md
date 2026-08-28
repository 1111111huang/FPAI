# Weekly (Grouped) Live-Lesson Judging Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Move live-lesson auto-judging from "once a day, one candidate at a time" to "once a week, every still-pending candidate for a `(competition_id, tier)` judged together in one combined call" — so a recurring pattern across several days finally gives `judge_lesson_candidate` more than n=1 to work with.

**Architecture:** Daily candidate generation is untouched. A new `RecoverableScheduler.schedule_weekly` primitive drives a new weekly job that groups all still-`pending`, `source='live'` `agent_lessons` rows by `(competition_id, tier)`, joins each group's `lesson_text`s into one combined string, and runs that combined text through the existing, unmodified `judge_lesson_candidate` → `generate_rule_from_lesson` → `find_conflicting_rule` chain exactly once per group — applying the one resulting decision to every row in the group.

**Tech Stack:** Python, DuckDB (`agent_lessons`), APScheduler (`CronTrigger`), pytest.

**Spec:** `docs/superpowers/specs/2026-08-27-weekly-lesson-judging-design.md`

---

### Task 1: `RecoverableScheduler.schedule_weekly`

**Files:**
- Modify: `app/backend/scheduler.py:126-140` (insert a new method between `schedule_once` and `_run_and_mark`)
- Test: `app/backend/tests/test_scheduler.py`

- [x] **Step 1: Write the failing tests**

Add these four tests to `app/backend/tests/test_scheduler.py`, right after `test_schedule_once_not_triggered_early` (currently ending at line 127) and before `test_restart_mid_day_detects_and_runs_a_missed_job_only_once`:

```python
def test_weekly_job_catches_up_when_trigger_day_and_time_already_passed(tmp_path: Path) -> None:
    """2026-08-23 is a Sunday. 'now' is past that Sunday's 09:00 trigger and
    the job hasn't run yet this week -- must run immediately, not wait for
    next Sunday's cron fire."""
    run_log = JobRunLog(db_path=tmp_path / "job_runs.db")
    calls = []
    now = datetime(2026, 8, 23, 9, 30, tzinfo=NY_TZ)

    scheduler = RecoverableScheduler(run_log=run_log, now_fn=lambda: now)
    scheduler.schedule_weekly("weekly_review", lambda: calls.append("ran"), day_of_week=6, hour=9, minute=0)

    assert _wait_until(lambda: run_log.has_run("weekly_review", "2026-08-23"))
    assert calls == ["ran"]


def test_weekly_job_does_not_catch_up_on_a_different_weekday_even_past_the_trigger_time(tmp_path: Path) -> None:
    """2026-08-24 is a Monday -- 09:30 is past 09:00, but this isn't the
    target weekday (Sunday=6), so schedule_daily()'s own 'hour:minute
    already passed today' catch-up logic must NOT fire here."""
    run_log = JobRunLog(db_path=tmp_path / "job_runs.db")
    calls = []
    now = datetime(2026, 8, 24, 9, 30, tzinfo=NY_TZ)

    RecoverableScheduler(run_log=run_log, now_fn=lambda: now).schedule_weekly(
        "weekly_review", lambda: calls.append("ran"), day_of_week=6, hour=9, minute=0
    )

    assert calls == []


def test_weekly_job_not_triggered_early_on_the_target_weekday(tmp_path: Path) -> None:
    run_log = JobRunLog(db_path=tmp_path / "job_runs.db")
    calls = []
    now = datetime(2026, 8, 23, 3, 0, tzinfo=NY_TZ)  # Sunday, well before 09:00

    RecoverableScheduler(run_log=run_log, now_fn=lambda: now).schedule_weekly(
        "weekly_review", lambda: calls.append("ran"), day_of_week=6, hour=9, minute=0
    )

    assert calls == []


def test_weekly_job_does_not_rerun_once_already_run_this_week(tmp_path: Path) -> None:
    run_log = JobRunLog(db_path=tmp_path / "job_runs.db")
    run_log.mark_ran("weekly_review", "2026-08-23")
    calls = []
    now = datetime(2026, 8, 23, 9, 30, tzinfo=NY_TZ)

    RecoverableScheduler(run_log=run_log, now_fn=lambda: now).schedule_weekly(
        "weekly_review", lambda: calls.append("ran"), day_of_week=6, hour=9, minute=0
    )

    assert calls == []
```

- [x] **Step 2: Run the new tests to verify they fail**

Run: `/Users/tianqihuang/Documents/GitHub/FPAI/venv/bin/python -m pytest app/backend/tests/test_scheduler.py -k weekly -v` (run from inside this worktree directory)
Expected: 4 FAILs, each with `AttributeError: 'RecoverableScheduler' object has no attribute 'schedule_weekly'`

- [x] **Step 3: Implement `schedule_weekly`**

In `app/backend/scheduler.py`, insert this new method immediately after `schedule_once` (i.e. right after the line `self._run_and_mark(job_id, fn, run_key, wait=False)` that ends `schedule_once`, and before the `def _run_and_mark(...)` line):

```python
    def schedule_weekly(self, job_id: str, fn: Callable[[], None], day_of_week: int, hour: int, minute: int) -> None:
        """Same restart-safe catch-up contract as schedule_daily(), scoped to
        one weekday instead of every day. day_of_week: 0=Monday..6=Sunday --
        Python's own datetime.weekday() convention, which APScheduler's
        CronTrigger day_of_week integer form also uses, so the catch-up
        check below can compare them directly with no translation.

        Without the day_of_week condition in the catch-up check, a restart
        on any day past `hour:minute` would incorrectly look identical to
        schedule_daily()'s own "missed today's run" case and fire
        immediately regardless of which weekday it actually is."""
        self._scheduler.add_job(
            lambda: self._run_and_mark(job_id, fn, run_key=self._now_fn().date().isoformat()),
            trigger=CronTrigger(day_of_week=day_of_week, hour=hour, minute=minute, timezone=self.timezone),
            id=job_id,
            replace_existing=True,
        )
        now = self._now_fn()
        trigger_today = now.replace(hour=hour, minute=minute, second=0, microsecond=0)
        run_key = now.date().isoformat()
        if now.weekday() == day_of_week and now >= trigger_today and not self.run_log.has_run(job_id, run_key):
            self._run_and_mark(job_id, fn, run_key, wait=False)
```

- [x] **Step 4: Run the tests to verify they pass**

Run: `/Users/tianqihuang/Documents/GitHub/FPAI/venv/bin/python -m pytest app/backend/tests/test_scheduler.py -v` (run from inside this worktree directory)
Expected: all tests PASS (the 4 new ones plus every pre-existing one in this file, unaffected).

- [x] **Step 5: Commit**

```bash
git add app/backend/scheduler.py app/backend/tests/test_scheduler.py
git commit -m "feat(app): W183 -- add RecoverableScheduler.schedule_weekly

Same restart-safe catch-up guarantee as schedule_daily, scoped to one
weekday via APScheduler's CronTrigger(day_of_week=...). Needed for the
new weekly live-lesson review job.

Co-Authored-By: Claude Sonnet 5 <noreply@anthropic.com>"
```

**Task 1 status: DONE. Implemented as commit `2bf3605`. Spec-compliance review: ✅ compliant. Code-quality review: ✅ approved, no Critical/Important issues.**

---

### Task 2: `list_pending_by_source` gains `created_at`/`source_match_id`

**Files:**
- Modify: `src/agent/lessons.py:238-249`
- Test: `tests/test_agent_lessons.py`

- [ ] **Step 1: Write the failing test**

Add this test to `tests/test_agent_lessons.py`, right after `test_list_pending_by_source_excludes_other_sources_and_null` (currently ending around line 109):

```python
def test_list_pending_by_source_includes_created_at_and_source_match_id():
    """New fields needed by app/backend/live_lessons.py's weekly grouped
    judge to label each candidate's section when joining several days'
    lesson_texts into one combined document (W184)."""
    conn = _conn()
    insert_lesson_candidate(conn, "live text", "E0", "competition_specific", "m1,m2", source="live")

    pending = list_pending_by_source(conn, source="live")

    assert pending[0]["source_match_id"] == "m1,m2"
    assert isinstance(pending[0]["created_at"], datetime)
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `/Users/tianqihuang/Documents/GitHub/FPAI/venv/bin/python -m pytest tests/test_agent_lessons.py::test_list_pending_by_source_includes_created_at_and_source_match_id -v` (run from inside this worktree directory)
Expected: FAIL with `KeyError: 'source_match_id'`

- [ ] **Step 3: Extend `list_pending_by_source`**

In `src/agent/lessons.py`, replace the existing function body:

```python
def list_pending_by_source(conn: duckdb.DuckDBPyConnection, source: str) -> list[dict[str, Any]]:
    """Every pending lesson candidate from one source ('train'/'live') --
    used by app/backend/live_lessons.py's auto_judge_live_lessons() to find
    only its own population. WHERE source = ? naturally excludes both the
    other source and any pre-migration row
    (source IS NULL, since SQL's `NULL = 'live'` is never true) --
    agent-train's human-reviewed queue is structurally unreachable from
    here, not just conventionally avoided.

    created_at/source_match_id (W184) let a caller group several rows
    together and label each one when combining them (auto_judge_live_lessons'
    weekly grouped judge) -- unused before that caller existed."""
    rows = conn.execute(
        "SELECT id, lesson_text, competition_id, tier, created_at, source_match_id FROM agent_lessons "
        "WHERE status = 'pending' AND source = ? ORDER BY created_at",
        [source],
    ).fetchall()
    return [
        {
            "id": row[0], "lesson_text": row[1], "competition_id": row[2], "tier": row[3],
            "created_at": row[4], "source_match_id": row[5],
        }
        for row in rows
    ]
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `/Users/tianqihuang/Documents/GitHub/FPAI/venv/bin/python -m pytest tests/test_agent_lessons.py -v` (run from inside this worktree directory)
Expected: all tests PASS (the new one, plus every existing test in this file -- the two other tests calling `list_pending_by_source` only assert `p["id"]`/`p["lesson_text"]`/`p["competition_id"]`/`p["tier"]` by key, never the whole dict by `==`, so the two new keys are additive and don't break them).

- [ ] **Step 5: Commit**

```bash
git add src/agent/lessons.py tests/test_agent_lessons.py
git commit -m "feat(app): W184 -- list_pending_by_source returns created_at/source_match_id

Additive fields (existing callers use dict-key access, not whole-dict
equality, so nothing breaks) needed by the new weekly grouped judge to
label each candidate when joining several days' lesson_text into one
combined document.

Co-Authored-By: Claude Sonnet 5 <noreply@anthropic.com>"
```

---

### Task 3: Group-then-judge in `auto_judge_live_lessons`

**Files:**
- Modify: `app/backend/live_lessons.py:209-323` (the whole `auto_judge_live_lessons` function, plus one new helper above it)
- Test: `app/backend/tests/test_live_lessons.py`

- [ ] **Step 1: Write the failing tests**

Add `_format_group_lesson_text` to the import list in `app/backend/tests/test_live_lessons.py` (currently lines 16-22):

```python
from app.backend.live_lessons import (
    PreparedLessonBatch,
    _format_group_lesson_text,
    _to_lesson_record,
    auto_judge_live_lessons,
    commit_lesson_batches,
    generate_daily_lessons,
)
```

Also add `from datetime import datetime, timezone` near the top of the file's import block (currently just `from pathlib import Path`, `import sys`, `from unittest.mock import MagicMock, patch`, `import duckdb`):

```python
from datetime import datetime, timezone
```

Add these two new tests right after `test_commit_lesson_batches_writes_source_live` (currently ending around line 265, right before `test_auto_judge_live_lessons_approves_a_good_candidate`):

```python
def test_format_group_lesson_text_orders_by_date_and_labels_each_section() -> None:
    candidates = [
        {"created_at": datetime(2026, 8, 26, tzinfo=timezone.utc), "source_match_id": "m2", "lesson_text": "second day's text"},
        {"created_at": datetime(2026, 8, 24, tzinfo=timezone.utc), "source_match_id": "m1", "lesson_text": "first day's text"},
    ]

    combined = _format_group_lesson_text(candidates)

    assert combined.index("first day's text") < combined.index("second day's text")
    assert "2026-08-24" in combined
    assert "2026-08-26" in combined
    assert "match_ids: m1" in combined
    assert "match_ids: m2" in combined


def test_auto_judge_live_lessons_groups_same_competition_and_tier_candidates_into_one_judge_call(tmp_path: Path) -> None:
    """W185: two daily candidates for the same (competition_id, tier) --
    simulating two days' worth of accumulated pending rows now that daily
    auto-judging is removed -- must be judged together with ONE combined
    LLM call, not two independent ones, so the judge finally sees real
    sample size instead of n=1 every time."""
    dm = _dm(tmp_path)
    with dm.connection() as conn:
        create_lessons_tables(conn)
        insert_lesson_candidate(conn, "Live-sourced batch: day one, 0/1 correct.", "SP1", "competition_specific", "m1", source="live")
        insert_lesson_candidate(conn, "Live-sourced batch: day two, 0/1 correct.", "SP1", "competition_specific", "m2", source="live")

    judge_prompts = []

    def fake_invoke(prompt: str) -> str:
        if "deciding whether to promote" in prompt:
            judge_prompts.append(prompt)
            return '{"approve": true, "scope": "competition", "reasoning": "Recurs across both days."}'
        if "Extract ONLY the single most programmatic" in prompt:
            return "NEVER bet the draw as the only positive-edge market against a strong favorite."
        if "checking a new proposed rule" in prompt:
            return "NONE"
        raise AssertionError(f"unexpected prompt: {prompt}")

    results = auto_judge_live_lessons(dm, fake_invoke)

    assert len(judge_prompts) == 1  # one combined call, not one per candidate
    assert "day one" in judge_prompts[0]
    assert "day two" in judge_prompts[0]
    assert len(results) == 2  # the one decision applied to both rows
    with dm.connection(read_only=True) as conn:
        rows = conn.execute("SELECT status, rule_text FROM agent_lessons WHERE source = 'live'").fetchall()
    assert len(rows) == 2
    assert all(row[0] == "approved" for row in rows)
    assert all(row[1] == "NEVER bet the draw as the only positive-edge market against a strong favorite." for row in rows)
```

Now update `test_auto_judge_live_lessons_isolates_a_conflict_check_failure_to_its_own_candidate` (currently lines 402-441) -- give its two candidates *different* `competition_id`s so they land in separate groups, since the whole point of this test is proving one group's conflict-check failure doesn't discard another group's already-computed decision (with the old per-row judging, "group" and "row" were the same thing; now they aren't). Replace the test's body with:

```python
def test_auto_judge_live_lessons_isolates_a_conflict_check_failure_to_its_own_group(tmp_path: Path) -> None:
    """A raised exception from find_conflicting_rule (fail-closed by
    design -- see its own docstring) must defer only the group it was
    checking, not silently discard a sibling group's already-computed
    decision in the same run. Two different competition_ids so pattern A
    and pattern B land in separate (competition_id, tier) groups -- with
    grouped judging, two candidates sharing the same competition_id/tier
    would be joined into one group and judged together instead."""
    dm = _dm(tmp_path)
    with dm.connection() as conn:
        create_lessons_tables(conn)
        existing_id = insert_lesson_candidate(conn, "existing text", "E0", "competition_specific", "m0", source="train")
        approve_lesson(conn, existing_id, scope="competition", reviewer="test", rule_text="ALWAYS bet result_3way when confident.")
        insert_lesson_candidate(conn, "Live-sourced batch: pattern A.", "E0", "competition_specific", "m1", source="live")
        insert_lesson_candidate(conn, "Live-sourced batch: pattern B.", "SP1", "competition_specific", "m2", source="live")

    def fake_invoke(prompt: str) -> str:
        if "deciding whether to promote" in prompt:
            reasoning = "A looks solid." if "pattern A" in prompt else "B looks solid."
            return f'{{"approve": true, "scope": "competition", "reasoning": "{reasoning}"}}'
        if "Extract ONLY the single most programmatic" in prompt:
            return "NEVER bet result_3way on pattern A." if "pattern A" in prompt else "NEVER bet result_3way on pattern B."
        if "checking a new proposed rule" in prompt:
            if "pattern A" in prompt:
                raise RuntimeError("conflict-check API down")
            return "NONE"
        raise AssertionError(f"unexpected prompt: {prompt}")

    results = auto_judge_live_lessons(dm, fake_invoke)

    assert len(results) == 2  # both groups processed -- neither lost
    with dm.connection(read_only=True) as conn:
        rows = {
            row[0]: row for row in conn.execute(
                "SELECT lesson_text, status, rule_text, auto_decision_reasoning FROM agent_lessons WHERE source = 'live'"
            ).fetchall()
        }
    a_row = rows["Live-sourced batch: pattern A."]
    b_row = rows["Live-sourced batch: pattern B."]
    assert a_row[1] == "pending"
    assert "conflict check failed" in a_row[3]
    assert b_row[1] == "approved"
    assert b_row[2] == "NEVER bet result_3way on pattern B."
```

- [ ] **Step 2: Run the new/modified tests to verify they fail**

Run: `/Users/tianqihuang/Documents/GitHub/FPAI/venv/bin/python -m pytest app/backend/tests/test_live_lessons.py -v` (run from inside this worktree directory)
Expected: `test_format_group_lesson_text_orders_by_date_and_labels_each_section` FAILs with `ImportError: cannot import name '_format_group_lesson_text'`; `test_auto_judge_live_lessons_groups_same_competition_and_tier_candidates_into_one_judge_call` FAILs (two candidates currently judged independently, so `len(judge_prompts) == 1` fails -- actually 2); `test_auto_judge_live_lessons_isolates_a_conflict_check_failure_to_its_own_group` currently PASSes already (it's testing the same underlying per-item isolation the old code already has, just renamed/re-shaped) -- that's fine, it'll still pass after Step 3's rewrite too, just via the new grouped code path instead of the old per-row one.

- [ ] **Step 3: Rewrite `auto_judge_live_lessons`, add `_format_group_lesson_text`**

In `app/backend/live_lessons.py`, insert this new helper function immediately before `def auto_judge_live_lessons(`:

```python
def _format_group_lesson_text(candidates: list[dict[str, Any]]) -> str:
    """Joins one (competition_id, tier) group's individual daily
    candidates into a single combined lesson_text, in date order, each
    section labeled with its own date and source match ids -- so a week's
    worth of daily reports reads as one document instead of a single day's,
    giving judge_lesson_candidate real sample size to apply its existing
    "reject if noise, approve if clearly systematic" test to, rather than
    the n=1 it always got when judged one candidate at a time."""
    ordered = sorted(candidates, key=lambda c: c["created_at"])
    sections = [
        f"--- {c['created_at'].date()} (match_ids: {c['source_match_id']}) ---\n{c['lesson_text']}"
        for c in ordered
    ]
    return "\n\n".join(sections)
```

Then replace the entire body of `auto_judge_live_lessons` (from its `def auto_judge_live_lessons(` line through its final `return results` line) with:

```python
def auto_judge_live_lessons(
    duckdb_manager: DuckDBManager,
    llm_invoke: Callable[[str], str] | None,
) -> list[dict[str, Any]]:
    """2026-08-27 (W183-W185, superseding the 2026-08-26 per-candidate
    version): judges every still-pending source='live' candidate for a
    (competition_id, tier) *together*, once a week, instead of one
    candidate at a time right after it's created. A single day's batch is
    typically n=1 match -- judge_lesson_candidate's own "reject on a thin
    sample" prompt could never clear that bar even when the exact same
    failure mode recurred for weeks, since it never saw more than one
    day's evidence. Grouping several days' already-computed lesson_text
    into one combined document (_format_group_lesson_text) gives it real
    sample size instead, with zero changes to judge_lesson_candidate,
    generate_rule_from_lesson, or find_conflicting_rule themselves.

    Never touches source='train' (or pre-migration, source IS NULL) rows --
    list_pending_by_source(source='live') structurally excludes both.

    llm_invoke=None means the weekly job's own LLM client failed to build --
    a no-op; every pending candidate simply waits for the following week.

    Three phases, same discipline as prepare_lesson_batches/
    commit_lesson_batches: a brief read, all LLM work with no DuckDB
    connection open, then a brief write. Both the per-group conflict check
    and the per-row write are isolated in their own try/except -- a
    failure in either only defers/skips what it was working on, never
    discarding another group's (or another row's) already-computed
    decision. Returns a list of dicts, one per underlying row (a group of
    N candidates contributes N entries, all sharing the same decision) --
    {id, action, reasoning}."""
    if llm_invoke is None:
        return []

    with duckdb_manager.connection(read_only=True) as conn:
        pending = list_pending_by_source(conn, source="live")

    groups: dict[tuple[str | None, str], list[dict[str, Any]]] = defaultdict(list)
    for candidate in pending:
        groups[(candidate["competition_id"], candidate["tier"])].append(candidate)

    results: list[dict[str, Any]] = []
    for (competition_id, tier), candidates in groups.items():
        combined_text = _format_group_lesson_text(candidates)
        row_ids = [candidate["id"] for candidate in candidates]

        decision = judge_lesson_candidate(combined_text, competition_id, tier, llm_invoke)
        action = "reject" if not decision.approve else "approve"
        scope = decision.scope
        rule_text: str | None = None
        reasoning = decision.reasoning

        if decision.approve:
            rule_text = generate_rule_from_lesson(combined_text, llm_invoke)
            if rule_text is None:
                action = "defer"
                reasoning = f"{decision.reasoning} (rule distillation failed -- left pending for retry)"
            else:
                try:
                    with duckdb_manager.connection(read_only=True) as conn:
                        existing_rules = load_approved_lessons(conn, competition_id, tier)
                    conflict = find_conflicting_rule(rule_text, existing_rules, llm_invoke)
                except Exception as exc:
                    action = "defer"
                    reasoning = f"{decision.reasoning} (conflict check failed: {exc!r} -- left pending for retry)"
                    LOGGER.warning(
                        "live_lessons: conflict check failed for lesson group competition_id=%s tier=%s.",
                        competition_id, tier, exc_info=True,
                    )
                else:
                    if conflict is not None:
                        action = "defer"
                        reasoning = f"Would approve, but a conflict was found: {conflict}"

        for row_id in row_ids:
            results.append({
                "id": row_id, "action": action, "scope": scope,
                "rule_text": rule_text, "reasoning": reasoning,
            })

    with duckdb_manager.connection() as conn:
        for result in results:
            try:
                # Re-check status right before writing -- a human can run
                # `agent-lessons approve/reject <id>` on this exact row at
                # any point during this function's judge/distill/
                # conflict-check phase above, which holds no DuckDB
                # connection open and can run for a while (LLM calls).
                # Without this, our write here would silently clobber that
                # human decision with a stale one computed before it
                # happened. Applied uniformly to approve/reject/defer.
                current_status = conn.execute(
                    "SELECT status FROM agent_lessons WHERE id = ?", [result["id"]]
                ).fetchone()
                if current_status is None or current_status[0] != "pending":
                    LOGGER.warning(
                        "live_lessons: skipping auto-judge write for lesson id=%s -- "
                        "already reviewed (status=%s) since this run started.",
                        result["id"], current_status[0] if current_status else "missing",
                    )
                    continue
                if result["action"] == "approve":
                    approve_lesson(conn, result["id"], result["scope"], reviewer="agent-auto", rule_text=result["rule_text"])
                elif result["action"] == "reject":
                    reject_lesson(conn, result["id"], reviewer="agent-auto")
                # "defer" -- leave status as-is, just record the reasoning below.
                conn.execute(
                    "UPDATE agent_lessons SET auto_decision_reasoning = ? WHERE id = ?",
                    [result["reasoning"], result["id"]],
                )
            except Exception:
                LOGGER.warning(
                    "live_lessons: failed to write auto-judge decision for lesson id=%s -- left as-is for retry.",
                    result["id"], exc_info=True,
                )

    return results
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `/Users/tianqihuang/Documents/GitHub/FPAI/venv/bin/python -m pytest app/backend/tests/test_live_lessons.py -v` (run from inside this worktree directory)
Expected: all tests PASS, including every pre-existing `test_auto_judge_live_lessons_*` test (each uses exactly one live-sourced candidate per test except the isolation one just rewritten, so each forms a group of size 1 and behaves identically to the old per-candidate path -- the substring-based `fake_invoke` routing in those tests still matches, since the raw `lesson_text` remains a substring of the new joined/labeled text).

- [ ] **Step 5: Commit**

```bash
git add app/backend/live_lessons.py app/backend/tests/test_live_lessons.py
git commit -m "feat(app): W185 -- group pending live candidates by (competition_id, tier) before judging

auto_judge_live_lessons now judges every still-pending source='live'
candidate for a (competition_id, tier) together, in one combined
judge_lesson_candidate call, instead of one candidate at a time.
judge_lesson_candidate/generate_rule_from_lesson/find_conflicting_rule
are all reused completely unmodified -- only the text they're given
(now a multi-day joined document via new _format_group_lesson_text)
and the number of rows one decision gets applied to changed.

Co-Authored-By: Claude Sonnet 5 <noreply@anthropic.com>"
```

---

### Task 4: Wire the weekly job into the scheduler, remove daily auto-judge

**Files:**
- Modify: `app/backend/scheduler_wiring.py:43-45` (constants), `:311-366` (`register_lessons_job`)
- Test: `app/backend/tests/test_scheduler_wiring.py`

- [ ] **Step 1: Write the failing test, delete the obsolete one**

In `app/backend/tests/test_scheduler_wiring.py`, add the new constants to the existing import block (currently lines 25-36):

```python
from app.backend.scheduler_wiring import (
    EOD_HOUR,
    EOD_JOB_ID,
    FallbackOddsClient,
    LESSONS_HOUR,
    LESSONS_JOB_ID,
    LESSONS_WEEKLY_DAY_OF_WEEK,
    LESSONS_WEEKLY_HOUR,
    LESSONS_WEEKLY_JOB_ID,
    LESSONS_WEEKLY_MINUTE,
    PersistingOddsClient,
    next_day_date_str,
    register_eod_job,
    register_lessons_job,
    t30_run_at,
)
```

Delete `test_register_lessons_job_auto_judges_generated_candidates` entirely (currently lines 530-566, from its `def` line through the blank line before `def test_register_lessons_job_degrades_to_stats_only_when_llm_build_fails`) -- it specifically tests the daily job auto-judging, which no longer happens.

In its place, add this new end-to-end test for the weekly job:

```python
def test_register_lessons_job_weekly_review_judges_accumulated_candidates(tmp_path: Path) -> None:
    """End-to-end: a candidate already sitting pending (as it would after
    a week of daily-only generation, since the daily job no longer judges)
    gets judged once the weekly job's own trigger day/time arrives --
    proves the weekly review is actually reachable from register_lessons_job,
    not just unit-tested in isolation. Pre-seeding the pending row directly
    (rather than relying on the daily job's own same-run catch-up to create
    it) avoids a real race between two independently-threaded catch-up
    fires that can happen when 'now' is past both jobs' trigger times at
    once, exactly as it is here."""
    cache = RecommendationCache(db_path=tmp_path / "cache.db")
    store = RecommendationOutcomeStore(db_path=tmp_path / "outcomes.db")
    client = MagicMock()
    client.get_results.return_value = []
    duckdb_manager = DuckDBManager()
    duckdb_manager.db_path = tmp_path / "fpai_core.db"
    config = AgentConfig.default()
    run_log = JobRunLog(db_path=tmp_path / "job_runs.db")

    from src.agent.lessons import create_lessons_tables, insert_lesson_candidate
    with duckdb_manager.connection() as conn:
        create_lessons_tables(conn)
        insert_lesson_candidate(conn, "Live-sourced batch: pattern.", "E0", "competition_specific", "m1", source="live")

    assert LESSONS_WEEKLY_DAY_OF_WEEK == 6  # Sunday -- 2026-08-23 below must match
    now = datetime(2026, 8, 23, LESSONS_WEEKLY_HOUR, LESSONS_WEEKLY_MINUTE + 5, tzinfo=NY_TZ)
    scheduler = RecoverableScheduler(run_log=run_log, now_fn=lambda: now)

    def fake_llm_invoke(prompt: str) -> str:
        return '{"approve": false, "scope": null, "reasoning": "Single-match sample, too thin to judge."}'

    with patch("app.backend.scheduler_wiring._build_lessons_llm_invoke", return_value=fake_llm_invoke):
        register_lessons_job(
            scheduler, cache=cache, store=store, client=client,
            duckdb_manager=duckdb_manager, config=config,
        )
        assert _wait_until(lambda: run_log.has_run(LESSONS_WEEKLY_JOB_ID, "2026-08-23"))

    with duckdb_manager.connection(read_only=True) as conn:
        row = conn.execute("SELECT status, source, auto_decision_reasoning FROM agent_lessons").fetchone()
    assert row[0] == "rejected"
    assert row[1] == "live"
    assert row[2] == "Single-match sample, too thin to judge."
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `/Users/tianqihuang/Documents/GitHub/FPAI/venv/bin/python -m pytest app/backend/tests/test_scheduler_wiring.py -k weekly_review -v` (run from inside this worktree directory)
Expected: FAIL with `ImportError: cannot import name 'LESSONS_WEEKLY_DAY_OF_WEEK'`

- [ ] **Step 3: Wire the weekly job**

In `app/backend/scheduler_wiring.py`, replace the existing constants block:

```python
LESSONS_JOB_ID = "daily_live_lessons"
LESSONS_HOUR = 6
LESSONS_MINUTE = 0
```

with:

```python
LESSONS_JOB_ID = "daily_live_lessons"
LESSONS_HOUR = 6
LESSONS_MINUTE = 0
# W183-W185: judging moved off the daily job onto its own weekly one (see
# docs/superpowers/specs/2026-08-27-weekly-lesson-judging-design.md) --
# 10 minutes after the daily job's own slot, on the daily job's own
# schedule_daily trigger day, so a Sunday's own freshly-generated candidate
# is always included in that same week's review rather than deferred to
# the following week, and the weekly job's read never races the daily
# job's write for the same morning.
LESSONS_WEEKLY_JOB_ID = "weekly_live_lesson_review"
LESSONS_WEEKLY_DAY_OF_WEEK = 6  # Sunday (0=Monday..6=Sunday)
LESSONS_WEEKLY_HOUR = 6
LESSONS_WEEKLY_MINUTE = 10
```

Then replace `register_lessons_job`'s entire body (from its docstring's closing `"""` through the final `scheduler.schedule_daily(LESSONS_JOB_ID, _lessons_job, hour=LESSONS_HOUR, minute=LESSONS_MINUTE)` line) with:

```python
    def _lessons_job() -> None:
        try:
            llm_invoke = _build_lessons_llm_invoke(config)
        except Exception:
            LOGGER.warning(
                "live_lessons: could not build an LLM client -- generating stats-only "
                "lessons for today instead of failing the whole run.", exc_info=True,
            )
            llm_invoke = None

        batches = prepare_lesson_batches(cache, store, client, sweden_client, llm_invoke)

        with duckdb_manager.connection() as conn:
            create_lessons_tables(conn)
            lesson_ids = commit_lesson_batches(conn, store, batches)
        LOGGER.info("Daily live lessons: %d candidate(s) generated.", len(lesson_ids))

    def _weekly_review_job() -> None:
        try:
            llm_invoke = _build_lessons_llm_invoke(config)
        except Exception:
            LOGGER.warning(
                "live_lessons: could not build an LLM client -- skipping this week's "
                "auto-judge review (every pending candidate waits for next week's run).",
                exc_info=True,
            )
            llm_invoke = None

        judged = auto_judge_live_lessons(duckdb_manager, llm_invoke)
        action_counts = Counter(j["action"] for j in judged)
        LOGGER.info(
            "Weekly live-lesson review: %d candidate(s) auto-judged (approved=%d, rejected=%d, deferred=%d).",
            len(judged), action_counts["approve"], action_counts["reject"], action_counts["defer"],
        )

    scheduler.schedule_daily(LESSONS_JOB_ID, _lessons_job, hour=LESSONS_HOUR, minute=LESSONS_MINUTE)
    scheduler.schedule_weekly(
        LESSONS_WEEKLY_JOB_ID, _weekly_review_job,
        day_of_week=LESSONS_WEEKLY_DAY_OF_WEEK, hour=LESSONS_WEEKLY_HOUR, minute=LESSONS_WEEKLY_MINUTE,
    )
```

And update `register_lessons_job`'s own docstring (the block currently starting `"""Registers the daily live-lessons job (W175-W178): ...` through `...never held across either (Task 4 code-quality review finding)."""`) to:

```python
    """Registers two jobs (W175-W185): the daily live-lessons job resolves
    pending recommendation_outcomes (W167) and batches whatever's newly
    unbatched into agent_lessons candidates via live_lessons.py's
    prepare_lesson_batches/commit_lesson_batches, at LESSONS_HOUR (06:00 ET,
    distinct from EOD_HOUR's 23:00, and after football-data.org has
    typically posted the prior day's results) -- same schedule_daily
    restart/catch-up guarantee as the EOD job. It no longer judges anything
    itself (W183-W185): a separate weekly job (LESSONS_WEEKLY_*) judges
    every candidate still pending, grouped by (competition_id, tier), via
    auto_judge_live_lessons -- see docs/superpowers/specs/
    2026-08-27-weekly-lesson-judging-design.md for why judging moved off
    the daily cadence.

    duckdb_manager: a write-mode DuckDBManager (matches main.py's own
    `agent-lessons approve` CLI pattern) -- distinct from lessons_node's
    own read_only=True live-serving connection, since both jobs write.
    Deliberately not sandbox-routed, unlike its sibling dependencies here --
    agent_lessons is one persistent human-review queue regardless of
    SANDBOX_MODE, matching lessons_node's own always-real-path read
    behavior.

    Each job's own DuckDB connection is opened only around its brief write
    step -- prepare_lesson_batches/auto_judge_live_lessons both do all
    their network-bound (football-data.org results lookups, possibly
    rate-limited) and LLM-bound (reflection, judging) work first, with no
    DuckDB connection open at all, so data/fpai_core.db's exclusive file
    lock is never held across either (Task 4 code-quality review finding,
    unchanged by this split)."""
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `/Users/tianqihuang/Documents/GitHub/FPAI/venv/bin/python -m pytest app/backend/tests/test_scheduler_wiring.py -v` (run from inside this worktree directory)
Expected: all tests PASS.

- [ ] **Step 5: Commit**

```bash
git add app/backend/scheduler_wiring.py app/backend/tests/test_scheduler_wiring.py
git commit -m "feat(app): W185 -- wire the weekly live-lesson review job, remove daily auto-judge

register_lessons_job now registers two jobs instead of one:
_lessons_job (daily, generation only -- the auto_judge_live_lessons
call is removed) and a new _weekly_review_job (Sunday 06:10 ET, 10 min
after the daily slot so it never races that morning's own write),
which calls the now-grouped auto_judge_live_lessons.

Co-Authored-By: Claude Sonnet 5 <noreply@anthropic.com>"
```

---

### Task 5: Record the user story

**Files:**
- Modify: `documents/app_user_stories.md` (append after PHASE 40, currently ending at the line containing `**Manually verified (2026-08-26):** ran auto_judge_live_lessons end-to-end...`)

- [ ] **Step 1: Append the new phase**

Append this new section at the end of `documents/app_user_stories.md`:

```markdown

## PHASE 41: Weekly (Grouped) Live-Lesson Judging

> Direct user request (2026-08-27), following review of a real rejected candidate (lesson #163, SP1): PHASE 40's per-day auto-judge only ever sees one day's evidence (typically n=1 match), so its own conservative "reject on a thin sample" prompt could never clear even if the identical failure mode recurred for weeks straight -- there's no memory connecting separate days' candidates into a trend. Design: `docs/superpowers/specs/2026-08-27-weekly-lesson-judging-design.md`. Daily candidate *generation* (PHASE 39) is unchanged; only when/how candidates get judged moves. Depends on PHASE 40's `agent_lessons`/`judge_lesson_candidate`/`auto_judge_live_lessons`.

| ID | Status | Description | Comments |
|---|---|---|---|
| W183 | active | **`RecoverableScheduler.schedule_weekly(job_id, fn, day_of_week, hour, minute)`** (`app/backend/scheduler.py`) -- same restart-safe catch-up guarantee as `schedule_daily`, scoped to one weekday via APScheduler's `CronTrigger(day_of_week=...)` (0=Monday..6=Sunday, matching Python's own `datetime.weekday()`). | Size XS · Depends on: none (extends PHASE 8's `RecoverableScheduler`). |
| W184 | active | **`list_pending_by_source` returns `created_at`/`source_match_id`** alongside its existing `id`/`lesson_text`/`competition_id`/`tier` fields (`src/agent/lessons.py`) -- additive, existing callers use dict-key access not whole-dict equality. Needed so the weekly judge can label and date-order each candidate when joining several days' `lesson_text` into one combined document. | Size XS · Depends on: none (extends PHASE 40's `list_pending_by_source`). |
| W185 | active | **`auto_judge_live_lessons` groups pending candidates by `(competition_id, tier)` before judging** (`app/backend/live_lessons.py`), joining each group's `lesson_text`s into one combined document (new `_format_group_lesson_text`) and running it through the existing, completely unmodified `judge_lesson_candidate` → `generate_rule_from_lesson` → `find_conflicting_rule` chain once per group -- the one resulting decision applies to every row in the group. Wired into the scheduler as a new weekly job (`register_lessons_job`, `app/backend/scheduler_wiring.py`, Sunday 06:10 ET) instead of the removed daily auto-judge call. | Size M · Depends on: W183, W184. |
```

- [ ] **Step 2: Commit**

```bash
git add documents/app_user_stories.md
git commit -m "docs: append PHASE 41 (W183-W185) user stories for weekly lesson judging

Co-Authored-By: Claude Sonnet 5 <noreply@anthropic.com>"
```

---

### Task 6: Full suite, mark the phase complete

**Files:**
- Modify: `documents/app_user_stories.md` (the three rows just added)

- [ ] **Step 1: Run the full backend and root test suites**

Run: `/Users/tianqihuang/Documents/GitHub/FPAI/venv/bin/python -m pytest app/backend/tests/ tests/test_agent_lessons.py -q` (run from inside this worktree directory)
Expected: all PASS, zero regressions (one pre-existing, unrelated failure -- `test_fixtures_endpoint_sources_historical_swe_results_from_raw_matches_not_the_odds_api`, missing gitignored `data/fpai_core.db` in this fresh worktree -- is expected and not a regression from this plan). Note the exact pass count for the completion notes below.

- [ ] **Step 2: Manually verify end-to-end against a real temp DuckDB**

Run this from inside this worktree directory (adjust nothing -- it's self-contained and uses a temp file):

```bash
/Users/tianqihuang/Documents/GitHub/FPAI/venv/bin/python -c "
import tempfile, os
from pathlib import Path
from src.agent.lessons import create_lessons_tables, insert_lesson_candidate
from src.utils.db_manager import DuckDBManager
from app.backend.live_lessons import auto_judge_live_lessons

with tempfile.TemporaryDirectory() as tmp:
    dm = DuckDBManager()
    dm.db_path = Path(tmp) / 'fpai_core.db'
    with dm.connection() as conn:
        create_lessons_tables(conn)
        insert_lesson_candidate(conn, 'Live-sourced batch: day one.', 'SP1', 'competition_specific', 'm1', source='live')
        insert_lesson_candidate(conn, 'Live-sourced batch: day two.', 'SP1', 'competition_specific', 'm2', source='live')

    calls = []
    def fake_invoke(prompt: str) -> str:
        calls.append(prompt)
        if 'deciding whether to promote' in prompt:
            return '{\"approve\": false, \"scope\": null, \"reasoning\": \"Still just two days.\"}'
        raise AssertionError('unexpected prompt')

    results = auto_judge_live_lessons(dm, fake_invoke)
    assert len(calls) == 1, f'expected 1 combined judge call, got {len(calls)}'
    assert len(results) == 2, f'expected the one decision applied to both rows, got {len(results)}'
    with dm.connection(read_only=True) as conn:
        rows = conn.execute(\"SELECT status FROM agent_lessons WHERE source='live'\").fetchall()
    assert all(r[0] == 'rejected' for r in rows)
    print('OK: 2 daily candidates for the same (competition_id, tier) judged together in 1 call, both rejected together.')
"
```

Expected output: `OK: 2 daily candidates for the same (competition_id, tier) judged together in 1 call, both rejected together.`

- [ ] **Step 3: Mark W183-W185 completed**

In `documents/app_user_stories.md`, change each of the three rows added in Task 5 from `active` to `completed`, and replace their `Comments` cell content with completion notes following this codebase's established format (fill in the actual pass count from Step 1's real output, and any real deviation/finding from implementation -- do not invent findings that didn't happen):

```
| W183 | completed | **`RecoverableScheduler.schedule_weekly(job_id, fn, day_of_week, hour, minute)`** (`app/backend/scheduler.py`) -- same restart-safe catch-up guarantee as `schedule_daily`, scoped to one weekday via APScheduler's `CronTrigger(day_of_week=...)` (0=Monday..6=Sunday, matching Python's own `datetime.weekday()`). | Size XS · Depends on: none (extends PHASE 8's `RecoverableScheduler`). **Completion notes (2026-08-27):** [fill in from actual implementation -- test count, any real finding]. |
```

(and similarly for W184, W185, each describing what actually happened during that task's implementation -- a clean pass with no deviations is a valid, expected outcome and should be stated plainly, not padded with invented findings.)

- [ ] **Step 4: Commit**

```bash
git add documents/app_user_stories.md
git commit -m "docs: mark PHASE 41 (W183-W185) completed

Co-Authored-By: Claude Sonnet 5 <noreply@anthropic.com>"
```

---

## Plan Self-Review Notes

- **Spec coverage:** `schedule_weekly` (Task 1) ✓, `list_pending_by_source` additive fields (Task 2) ✓, grouped-judging + combined-text format (Task 3) ✓, scheduler wiring / removal of the daily auto-judge call (Task 4) ✓, error handling for a group's judge/conflict-check failure and a row's write failure (Task 3, preserved from the existing per-candidate code, now scoped per-group/per-row) ✓, testing section (schedule_weekly catch-up variants, grouping behavior, isolation, daily job no longer judges) ✓, user-story bookkeeping per `CLAUDE.md` (Tasks 5-6) ✓. Explicitly-out-of-scope items (daily generation changes, retroactive re-evaluation, configurable cadence, prompt changes) are untouched by every task above -- confirmed no task modifies `prepare_lesson_batches`, `commit_lesson_batches`, `judge_lesson_candidate`'s prompt, `generate_rule_from_lesson`, or `find_conflicting_rule`.
- **Type consistency:** `list_pending_by_source`'s dict keys (`id`, `lesson_text`, `competition_id`, `tier`, `created_at`, `source_match_id` -- Task 2) match exactly what `_format_group_lesson_text` (Task 3) reads (`c["created_at"]`, `c["source_match_id"]`, `c["lesson_text"]`) and what the grouping loop reads (`candidate["competition_id"]`, `candidate["tier"]`, `candidate["id"]`). `schedule_weekly`'s signature (Task 1: `job_id, fn, day_of_week, hour, minute`) matches its call site in Task 4 (`LESSONS_WEEKLY_JOB_ID, _weekly_review_job, day_of_week=..., hour=..., minute=...`). `LESSONS_WEEKLY_JOB_ID`/`LESSONS_WEEKLY_DAY_OF_WEEK`/`LESSONS_WEEKLY_HOUR`/`LESSONS_WEEKLY_MINUTE` (defined Task 4 Step 3) match the names imported in Task 4 Step 1's test and used in that same test's `now`/`assert` construction.
- **Placeholder scan:** no TBD/TODO; Task 6 Step 3's "[fill in from actual implementation]" is deliberate -- completion notes can only be written after the work happens, matching every prior phase's own pattern in this same file (the notes describe what a review actually found, not what's predicted in advance).

## Execution Note

Executed via `superpowers:subagent-driven-development` in an isolated worktree (`.claude/worktrees/weekly-lesson-judging`, branch `worktree-weekly-lesson-judging`). This plan file itself was written to the main checkout before the worktree was created (a `fresh` worktree branches from `origin/<default-branch>`, not the main checkout's uncommitted working tree) and had to be re-added here as its own commit once that gap was noticed during Task 1's code-quality review -- a process note for next time: commit the plan doc before creating the worktree, not after.
