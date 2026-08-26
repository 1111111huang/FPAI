# Daily Live-Recommendation Lessons — Design

**Date:** 2026-08-25
**Status:** approved, pending implementation plan
**Origin:** direct user request — reuse the existing training-side lessons pipeline (A33/A39–A47), but source lesson candidates from the app's own live, finished recommendations (`recommendation_outcomes`, W167) instead of only from `agent-train`'s backtest corpora. Internal use only: candidates land as `status='pending'` in the same `agent_lessons` table used today and require the same human `agent-lessons approve/reject` step before they can ever affect a live recommendation — nothing here is shown to end users or auto-injected.

## Problem

`agent-train` generates reviewable lesson candidates, but only over historical backtest corpora the user explicitly collects and replays. The live app now durably tracks every finished recommendation's actual outcome (`recommendation_outcomes`, W167) — but nothing turns that daily trickle of real results into lesson candidates. The agent can't learn from its own live mistakes without someone manually re-running training tooling against manually-exported live data.

## Scope decisions (from user)

- **Batch grouping:** one lesson candidate per `(competition_id, date)` — whatever finished that day for that league. No cross-day accumulation, no `--batch-size N` porting.
- **Trigger:** a new automatic daily scheduled job (mirrors `_eod_job` in `scheduler_wiring.py`), not a manual CLI or admin endpoint.
- **Resolution + lessons run together:** the new job calls the existing `resolve_pending_recommendations()` (W167) first, then generates lessons from whatever is newly resolved — one unattended daily pipeline, not two things the user has to remember to sequence.
- **LLM reflection included:** each batch gets both the deterministic stats (`generate_batch_lesson_text`, zero LLM calls) and an LLM-written narrative (`generate_batch_reflection`, one call per batch) — reuses the live agent's own default provider (`_build_llm_invoke` against the same `agent_config.yaml` default currently serving live recommendations, e.g. `deepseek-v4-pro`), not a separate hardcoded one.
- **Human approval gate unchanged:** new rows are `status='pending'`, indistinguishable in kind from a training-sourced candidate. They only reach live serving once reviewed via the existing `agent-lessons approve <id> --scope ...` CLI (A44/A45's conflict-check and distillation flow applies unchanged).
- **Accepted methodology narrowing, stated explicitly:** `RecommendationOutcome` only ever records the *one* market the agent actually picked (via `pick_recommended_market`), not every market it evaluated — unlike a training `BacktestRecord`, whose `market_results` scores every market in the full recommendation. `generate_batch_lesson_text`'s own template is population-agnostic (it never claims "every market" either way), so this doesn't need a wording change there — but a future reviewer shouldn't assume a live-sourced batch's stats are comparable to a training-sourced one. `live_lessons.py` prepends one fixed sentence to the stats text before insertion — *"Live-sourced batch: reflects only the market actually recommended per match, not every market the agent evaluated."* — confined entirely to the new module, `src/agent/lessons.py` itself is untouched.

## Two real gaps this fixes along the way (not new features — corrections)

1. **`recommendation_outcomes.competition` is unverified, LLM-self-reported text**, not a canonical registry code. `resolve_pending_recommendations()` already knows the real football-data.org competition code per match (the key its own results query used) but currently discards it after the results merge. New nullable `competition_id TEXT` column on `recommendation_outcomes`, populated with the real code at resolution time — `agent_lessons.competition_id` must route on the real code, since that's what `lessons_node` matches against for live injection; routing on free text would silently misfile every lesson. The existing `competition` column is untouched (still whatever the dashboard/stats breakdowns display today).
2. **Tier has no live-side source of truth today.** Solved by lookup, not tracking: `src.logic.competition_registry.get_competition_definition(competition_id).tier` is a static per-competition config value already used to route model selection — no new table, no new field beyond the `competition_id` fix above.

## Architecture

### New: `app/backend/live_lessons.py`

The one new module, following `agent_performance_dashboard.py`'s established precedent (the piece that needs real DB I/O — here, both the SQLite `recommendation_outcomes`/`recommendation_cache` and the DuckDB `agent_lessons` — kept out of `recommendation_stats.py`, which stays pure aggregation).

```
generate_daily_lessons(
    cache: RecommendationCache,
    store: RecommendationOutcomeStore,
    client: FootballDataClient,
    sweden_client: object | None,
    duckdb_conn: duckdb.DuckDBPyConnection,
    llm_invoke: Callable[[str], str] | None,
) -> list[int]  # ids of inserted lesson candidates
```

Steps:
1. `resolved = resolve_pending_recommendations(cache, store, client, sweden_client)` — existing W167 function, unmodified.
2. `pending = store.list_unbatched_for_lessons()` — new store method, same idempotency shape as the existing `resolved_keys()`: selects outcomes where `lesson_batched_at IS NULL`, so a rerun never double-counts a match. (Not scoped to just `resolved` from step 1 — a prior run could have resolved outcomes it never got to batch, e.g. a crash between steps; picking up *all* unbatched rows, not only today's newly-resolved ones, is the correct exclude-set semantics here.)
3. Group `pending` by `(competition_id, date)`. Rows with `competition_id IS NULL` (a pre-migration row, or a genuine resolution edge case) are skipped, not defaulted to "unknown" — an unroutable lesson candidate is worse than no candidate; logged, not silently dropped.
4. For each group: build `BacktestRecord`-shaped objects (see next section), call `generate_batch_lesson_text(records)`, prepend the fixed live-sourced-scope sentence (see Scope decisions above), optionally append `generate_batch_reflection(records, stats_text, llm_invoke)` (`llm_invoke=None` skips it — the job's own resilience path, see Error Handling), `tier = get_competition_definition(competition_id).tier`, `insert_lesson_candidate(duckdb_conn, lesson_text, competition_id, tier, source_match_id=",".join(match_ids))`.
5. Mark every batched outcome's `lesson_batched_at` via a new `store.mark_lesson_batched(ids)` call — only after the DuckDB insert for its group actually succeeds, so a failed insert leaves those rows eligible for the next run instead of silently vanishing.

### `_to_lesson_record(outcome, cache) -> BacktestRecord`

A new adapter (private to `live_lessons.py`), analogous to `recommendation_stats.py`'s existing `_to_backtest_records` but enrichment-complete rather than minimal, since `generate_batch_lesson_text`/`generate_batch_reflection`/`_describe_record` (unlike the Kelly-sim path) actually read `home_team`/`away_team`/`recommendation.{overall,confidence,explanation,limitations}`/`actual.result`:

- `home_team`/`away_team`: `cache.get_latest_any_config(outcome.match_id, outcome.date)` → `reported_teams(entry.recommendation.get("match") or {})`. `None` on a cache miss (purged entry) — degrades that one record to empty team-name strings, same "never let one bad row break the batch" discipline as the dashboard's top/bottom-bet enrichment; the record is still included, just less readable.
- `recommendation`: the same cache entry's full `.recommendation` dict, verbatim — this is exactly what a training `BacktestRecord.recommendation` holds, so `generate_batch_lesson_text` needs no changes at all.
- `actual`: recomputed via `build_actual_outcome(home_goals, away_goals)` from the real result already fetched during this run's `resolve_pending_recommendations()` call (its `results_by_id` map — threaded through as a parameter, not re-fetched) — no duplicate football-data.org call.
- `market_results`: single-entry list, `[{"market": outcome.market, "selection": outcome.selection, "correct": outcome.correct}]` — the accepted scope-narrowing above.
- `league`: `outcome.competition_id` (the real code, not the free-text display field) — consistent with what `generate_batch_lesson_text`'s own league-label formatting expects.

No changes to `src/agent/lessons.py` — every function it exposes (`generate_batch_lesson_text`, `generate_batch_reflection`, `insert_lesson_candidate`, `create_lessons_tables`) is called exactly as `main.py`'s `agent-train` already calls it.

### Scheduler wiring (`app/backend/scheduler_wiring.py`)

New `LESSONS_HOUR = 6`, `LESSONS_MINUTE = 0` (ET) — distinct from `EOD_HOUR = 23`, chosen to run well after most matches from "yesterday" have finished and football-data.org has posted results, and after EOD's own late-night batch. `scheduler.schedule_daily(LESSONS_JOB_ID, _lessons_job, hour=LESSONS_HOUR, minute=LESSONS_MINUTE)`, registered in the same `configure_scheduler()`/`ENABLE_SCHEDULER` gate the EOD/T-30 jobs already share — no new environment flag.

`_lessons_job()` builds its own `DuckDBManager()` connection (write-mode, matching `main.py`'s `agent-lessons approve` CLI pattern — not the `read_only=True` mode `lessons_node`'s live-serving read path uses) and its own `llm_invoke` via `_build_llm_invoke(load_config())` (reusing the same config loader the live agent path already uses for its default config) — wrapped in a broad `try/except` so a reflection-LLM outage degrades to stats-only for that run (see Error Handling), never crashes the scheduler thread.

## Data flow

```
06:00 ET daily
  → resolve_pending_recommendations(cache, store, client, sweden_client)
      [W167, unmodified — now also feeds this job, not just diagnostics]
  → store.list_unbatched_for_lessons()
  → group by (competition_id, date)
  → per group:
      records = [_to_lesson_record(o, cache, results_by_id) for o in group]
      stats_text = generate_batch_lesson_text(records)
      reflection = generate_batch_reflection(records, stats_text, llm_invoke)  [best-effort]
      tier = get_competition_definition(competition_id).tier
      insert_lesson_candidate(duckdb_conn, stats_text + reflection, competition_id, tier, match_ids)
      store.mark_lesson_batched(outcome_ids)
  → candidates sit as status='pending' in agent_lessons
      [unchanged from today] agent-lessons approve <id> --scope ... reviews them
```

## Error handling

- **Reflection LLM call fails** (network/provider error): `generate_batch_reflection` already returns `None` on any exception (existing contract, unchanged) — the job falls back to stats-only for that batch, not a failed run. Matches the existing training-side resilience.
- **DuckDB write contention**: the live-serving `lessons_node` read path already tolerates a `duckdb.IOException` during a concurrent write (documented existing tradeoff, Section 20.3 of `agent_techspec.md`, accepted for `agent-train`'s low-frequency writes) — this job is equally low-frequency (once daily, brief), so the same acceptance applies unchanged. No new locking/retry logic.
- **A group with `competition_id IS NULL`**: skipped and logged (not defaulted to "unknown"), left with `lesson_batched_at` still `NULL` so it's picked up automatically once that gap is fixed rather than needing a backfill script.
- **A record's cache entry is missing** (purged/evicted): that one record gets blank team names, stays in the batch — matches the dashboard's existing degrade-one-row discipline, doesn't drop the whole batch.
- **Zero unbatched outcomes on a given day**: job is a no-op, same as `resolve_pending_recommendations` already handling zero pending recommendations cleanly.

## Testing

- `app/backend/tests/test_recommendation_outcomes.py`: new tests for the `competition_id` column (populated correctly from the real result-lookup code, not the self-reported string), `list_unbatched_for_lessons()`, `mark_lesson_batched()`.
- New `app/backend/tests/test_live_lessons.py`: `_to_lesson_record` enrichment (real cache hit, degraded cache miss), grouping by `(competition_id, date)`, `competition_id IS NULL` rows skipped not defaulted, `generate_daily_lessons` end-to-end against a fake DuckDB connection + fake `llm_invoke`, reflection failure degrades to stats-only without failing the batch, idempotency (`mark_lesson_batched` only called after a successful insert; a rerun with nothing new is a no-op).
- `app/backend/tests/test_scheduler_wiring.py`: new job registered at the right hour/minute, distinct from `EOD_HOUR`, gated by the same `ENABLE_SCHEDULER` flag.
- No frontend changes — nothing here has a UI surface.

## Explicitly out of scope

- Any frontend display of lessons, live or otherwise — this is 100% for the human reviewer's own CLI workflow, unchanged from today's `agent-lessons approve/reject`.
- Backfilling `competition_id` for outcome rows already resolved before this ships — those rows simply never get batched (their `competition_id` stays `NULL`), which is an acceptable, small, one-time gap rather than a migration script.
- Any change to `find_conflicting_rule`/`generate_rule_from_lesson`/the approval CLI itself — this feature only adds a second *source* of pending candidates, the review/approval mechanics are untouched.
- Retrying/backfilling a batch whose reflection failed — stats-only is an acceptable permanent state for that batch, not something a later run re-attempts.
