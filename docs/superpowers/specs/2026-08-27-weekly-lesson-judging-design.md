# Weekly (Grouped) Live-Lesson Judging — Design

**Date:** 2026-08-27
**Status:** approved, pending implementation plan
**Origin:** direct user request, following a live incident review. `docs/superpowers/specs/2026-08-26-autonomous-live-lesson-judging-design.md` ("Phase 1") judges each day's freshly-generated live candidate immediately, in isolation. Walking through an actual rejected candidate (lesson #163, SP1, 2026-08-27) surfaced a structural blind spot: `judge_lesson_candidate` only ever sees *that one day's* batch (typically n=1 match), so its own prompt's conservative bar ("reject if noise, not clearly systematic") can never clear — even if the identical failure mode recurred every single day for weeks, each day's candidate is judged with no memory of the others. The user confirmed they want the fix to be **when/how candidates get judged**, not a change to daily *generation*.

Not to be confused with that prior design's own "Phase 2" (deferred there): that was about re-evaluating already-**approved** lessons against their live track record and retiring ones that aren't helping — attribution-tracking territory, still not built, still out of scope here. This design is about the **first, initial** judgment of still-`pending` candidates — just made less near-sighted.

## The one thing this changes

Judging happens **once a week, over every still-pending candidate accumulated that week, grouped and judged together per `(competition_id, tier)`** — instead of once a day, one candidate at a time.

Daily candidate *generation* (`prepare_lesson_batches`/`commit_lesson_batches`, one `agent_lessons` row per `(competition_id, date)`) is completely unchanged — still runs every morning at `LESSONS_HOUR`, still one row per match-day, still the permanent per-day audit trail. Only the auto-judge call is removed from that daily job.

## Why grouping-then-judging, not a new similarity/clustering step

The obvious-looking alternative — write new logic to detect "do these N already-rejected daily candidates describe the same recurring theme" — would mean comparing loosely-worded LLM-generated English paragraphs for similarity, a fuzzier and more novel piece of code than this needs. Instead: join that week's `pending` candidates' `lesson_text`s (already contain real computed stats + an LLM reflection each) into one combined string and hand it to the **existing, completely unmodified** `judge_lesson_candidate(lesson_text, competition_id, tier, llm_invoke)`. Its prompt already says exactly the right thing — *"Only approve if the pattern is clearly systematic, not noise from a small sample"* — it simply never had more than one day's evidence to apply that test to before. A joined week's worth of text (e.g. 5 daily reports, 4 of them naming the same failure mode) is precisely the input that prompt was written for. No prompt change, no new judging function, no new LLM call shape.

The same reasoning carries through the rest of the decision: `generate_rule_from_lesson` and `find_conflicting_rule` are also reused completely unmodified, given the same joined text.

## Grouping and combined-text shape

For each `(competition_id, tier)` with 1 or more `status='pending', source='live'` rows at the time the weekly job runs:

```
--- 2026-08-24 (match_ids: 559702) ---
<lesson_text of that day's row>

--- 2026-08-26 (match_ids: 564631) ---
<lesson_text of that day's row>
```

(`date` and `source_match_id` are already stored per row — this just renders them as section headers when joining, so the combined text stays legible to both the LLM and anyone reading `lesson_text` back later. Rows are joined in `date` order.)

A group of exactly one row still goes through the same path unchanged — the LLM sees one day's worth of evidence, same as every daily judgment does today, and will typically still reject it. No special-casing for small groups; this is just today's real behavior for a slow week, preserved rather than reproduced.

## Applying the decision

`judge_lesson_candidate` (and, on approval, `generate_rule_from_lesson` + `find_conflicting_rule`) is called **once per group**, producing one `LessonDecision`. That single decision (approve/reject/defer, with its `scope`, `rule_text`, and `reasoning`) is then applied to **every individual pending row in the group** — looping `approve_lesson`/`reject_lesson` per row, each with the group's shared `rule_text`/`scope`/`reasoning`, `reviewer="agent-auto"` (unchanged — no new reviewer label; nothing downstream currently branches on that string, so there's no reason to fork it).

The existing per-row safety check carries over unchanged and matters *more* here, since the LLM work for a whole group can now take longer than a single-candidate judgment did: **re-check `status = 'pending'` immediately before each row's write**. A human can run `agent-lessons approve/reject <id>` on any individual row in the group at any point during the group's judge/distill/conflict-check phase (which holds no DB connection open) — that row is silently skipped (logged) rather than clobbered by the group's decision, exactly as today's per-row logic already does.

## Scheduling — a new `schedule_weekly` primitive

`RecoverableScheduler` (`app/backend/scheduler.py`) currently has `schedule_daily` (fires every day) and `schedule_once` (fires at one specific datetime) — nothing recurring-but-not-daily. Adding:

```python
def schedule_weekly(self, job_id: str, fn: Callable[[], None], day_of_week: int, hour: int, minute: int) -> None:
```

Mirrors `schedule_daily` exactly:
- Recurring trigger: `CronTrigger(day_of_week=day_of_week, hour=hour, minute=minute, timezone=self.timezone)`. APScheduler's `day_of_week` integer convention (0=Mon..6=Sun) matches Python's own `datetime.weekday()`, so the catch-up check below can compare them directly with no translation.
- Catch-up-on-registration check: same as `schedule_daily`'s, with one added condition — `now.weekday() == day_of_week and now >= trigger_today and not self.run_log.has_run(job_id, run_key)`. Without the weekday condition, every restart on a non-target day would incorrectly treat "hour:minute already passed today" as a missed weekly run and fire immediately.
- `run_key = now.date().isoformat()` — same as `schedule_daily`, still unique per calendar day, so only the actual target weekday's occurrence is ever marked run, and a mid-week restart correctly does not re-trigger the job.

Registered as **Sunday 06:10 ET** — 10 minutes after the daily job's own 06:00 slot, so it never races the daily job's `commit_lesson_batches` write for the same morning, and always runs after that day's own candidate has already landed (a Sunday's own match results are included in the same week's judgment, not deferred to the following week).

## Pipeline integration

`register_lessons_job` (`app/backend/scheduler_wiring.py`) registers both jobs:

- `_lessons_job()` (daily, unchanged except deletion): resolve outcomes, generate/commit that day's candidates. The `auto_judge_live_lessons(...)` call and its logging are removed from here — nothing else in this function changes.
- `_weekly_review_job()` (new): builds an LLM client the same way `_lessons_job` already does (existing `_build_lessons_llm_invoke`, same try/except-degrades-to-None-and-logs posture), then calls the rewritten `auto_judge_live_lessons(duckdb_manager, llm_invoke)`.

`auto_judge_live_lessons` (`app/backend/live_lessons.py`) keeps its existing three-phase shape (brief read → all LLM/network work with no DB connection held → brief write) — Task 4's original "don't hold DuckDB's exclusive lock across a network or LLM call" discipline still applies, now scaled to a week's worth of groups instead of a day's worth of individual rows:

1. **Read**: `list_pending_by_source(conn, source="live")` (unchanged), grouped in Python by `(competition_id, tier)`.
2. **LLM work, no DB connection open**: per group — join texts, `judge_lesson_candidate`, and on approval `generate_rule_from_lesson` then a brief read of `load_approved_lessons` then `find_conflicting_rule` (this inner read/LLM interleaving is exactly what today's per-candidate version already does; only the outer loop is now per-group instead of per-row).
3. **Write**: per group's decision, looped over every row in that group, with the existing re-check-status-before-write guard.

## Error handling

- **A group's judge/distill/conflict-check LLM work raises an exception**: caught per-group (matching today's per-candidate isolation, just moved one level up) — logged, that group's rows are left `pending` exactly as they were, next week's run re-gathers and re-judges them (now alongside whatever new candidates arrived that week too). Never aborts other groups' decisions in the same run.
- **A row's write fails** (e.g. a concurrent human edit racing the status-recheck): logged, skipped, left for the next run — unchanged from today's per-row behavior.
- **Zero pending live-sourced candidates for a given week**: no-op, same as every other empty-input case in this pipeline.
- **LLM client fails to build** (`_build_lessons_llm_invoke` raises): `llm_invoke=None`, `auto_judge_live_lessons` no-ops entirely (existing contract, unchanged) — every pending candidate simply waits for the following week.

## Testing

- `schedule_weekly`: fires on the correct weekday only; restart on a non-target weekday does not catch-up-fire; restart on the target weekday after the trigger time, not yet run this week, does catch-up-fire; already-run-this-week does not double-fire.
- `auto_judge_live_lessons`: candidates for the same `(competition_id, tier)` across multiple dates are grouped and judged with one combined `judge_lesson_candidate` call, not N separate calls; the resulting decision is applied to every row in the group; a human-modified row (status no longer `pending` by write time) is skipped without disturbing the rest of the group; a group of size 1 behaves identically to today's single-candidate path; a raised exception during one group's processing doesn't prevent other groups from being judged and written.
- `_lessons_job`: no longer calls `auto_judge_live_lessons` — existing tests asserting judge-count logging from the daily job are removed/updated accordingly.

## Explicitly out of scope

- Any change to daily candidate *generation* — still one row per `(competition_id, date)`, every morning, unchanged.
- Re-evaluating already-`approved`/`rejected` rows retroactively (still the prior design's deferred "Phase 2" — attribution-tracking territory, unaffected by this change).
- A configurable review cadence/window (e.g. "every N days" instead of weekly, or a rolling window instead of calendar-week) — Sunday-anchored weekly is the one cadence this design implements; revisit only if real usage shows weekly doesn't fit.
- Any change to `judge_lesson_candidate`'s prompt, `generate_rule_from_lesson`, or `find_conflicting_rule` — all three are reused completely unmodified.
