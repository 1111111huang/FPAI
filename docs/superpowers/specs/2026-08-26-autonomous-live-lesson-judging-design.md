# Autonomous Live-Lesson Judging (Phase 1) — Design

**Date:** 2026-08-26
**Status:** approved, pending implementation plan
**Origin:** direct user request — the agent should "self pick and choose lessons after each day," reflecting on which are worth keeping. Narrowed through discussion into two explicit halves: lessons from `agent-train` (backtest corpora) stay 100% human-reviewed exactly as today — the user does not want those touched. Lessons sourced from the daily live-deployment job (`docs/superpowers/specs/2026-08-25-daily-live-lessons-design.md`, shipped 2026-08-25) get an autonomous approve/reject decision instead of sitting in a human's queue.

## Scope

Split into two projects; this design covers only the first.

- **Phase 1 (this design):** the agent judges each day's freshly-generated, live-deployment-sourced lesson candidates — approve/reject, and if approving, which scope and what rule — right when they're created. No reconsideration of anything already decided.
- **Phase 2 (explicitly deferred, not designed here):** periodically re-evaluate already-approved live-sourced lessons against their real track record (hit-rate/ROI on recommendations made while each was active) and retire ones that aren't helping. Deferred because it needs attribution data (which approved lessons actually applied to which live recommendation) that doesn't exist yet, and because there's essentially zero approved-lesson history today to validate a measurement methodology against — the source feature shipped yesterday. Phase 1 does **not** add speculative attribution-tracking plumbing for this either: it would mean touching `src/agent/pipeline.py`'s `lessons_node` (a deterministic, leakage-guard-sensitive piece of the graph) for a consumer that doesn't exist yet and whose real shape won't be known until Phase 2 is actually scoped. Building it now risks guessing the wrong shape; Phase 2 can add it once it knows what it needs.

## Two real, load-bearing invariants this design must not weaken

1. **`agent-train`'s human-review flow is completely untouched.** Nothing in Phase 1 reads, judges, or writes a `source='train'` (or pre-migration, `source IS NULL`) pending row. `agent-lessons approve/reject` behaves identically to today for that population.
2. **A genuine rule conflict still always defers to a human**, exactly as A45 established for the manual flow — Phase 1 does not get a `--force`-equivalent auto-override. Detecting a conflict during auto-judging means the candidate stays `pending`, not approved and not rejected, with the conflict explained in a new audit field.

## Schema — two additive columns on `agent_lessons`

Both via the same `ALTER TABLE agent_lessons ADD COLUMN IF NOT EXISTS ...` pattern this table already uses for `rule_text` (DuckDB supports `IF NOT EXISTS` on `ADD COLUMN` directly, unlike SQLite — no workaround needed here):

- `source TEXT` — `'train'` or `'live'`. Existing/pre-migration rows stay `NULL`; every query that cares treats `NULL` as `'train'` (the only source that ever existed before yesterday), so no backfill statement is needed. `live_lessons.py`'s write path sets `source='live'` explicitly.
- `auto_decision_reasoning TEXT` — the judge's stated reasoning for its decision (approve, reject, or "would approve but a conflict was found"). The audit trail that replaces a human reviewer's own judgment call being visible in the CLI transcript.

`insert_lesson_candidate()` gains one new parameter: `source: str = "train"` (default preserves every existing `agent-train`/CLI caller unchanged; `live_lessons.py`'s `commit_lesson_batches()` is the only caller that passes `source="live"`).

## The judge

New `judge_lesson_candidate(lesson_text, competition_id, tier, llm_invoke) -> LessonDecision` in `src/agent/lessons.py`, alongside the module's other LLM-driven functions (`generate_rule_from_lesson`, `find_conflicting_rule`) — same decoupled `str -> str` `llm_invoke` convention, same "the human-facing pieces already established the pattern, this just adds one more" reasoning for keeping it in this module rather than `live_lessons.py`.

```python
@dataclass
class LessonDecision:
    approve: bool
    scope: str | None  # "competition" | "tier", only set when approve=True
    reasoning: str      # always set — the audit trail
```

Given a live-deployment batch is small (W177 batches one candidate per league per day, typically a handful of matches), the prompt explicitly instructs conservatism: default to reject on a thin sample or a pattern that isn't clearly systematic. A bad rule silently baked into the live prompt is worse than one more day with no new rule. This mirrors the live agent's own already-investigated conservative posture on value edges (A71) — conservative-by-default is this system's existing house style, not a new invention. The prompt is given `competition_id`/`tier` as context specifically so it can reason about whether a pattern looks competition-specific or genuinely tier-general, the same distinction a human reviewer makes when choosing `--scope`.

On `approve=True`: reuse the existing, **unmodified** `generate_rule_from_lesson()` for distillation. If distillation fails (returns `None` — an existing, already-handled failure mode, e.g. a transient API error), the candidate is left exactly as-is (`status='pending'`, no `reviewed_at`) with `auto_decision_reasoning` explaining why — the next day's run will pick it up again automatically via the same `WHERE status='pending'` query, no special retry logic needed. If distillation succeeds, reuse the existing, unmodified `find_conflicting_rule()` against currently-approved rules in that scope (fetched via the existing `load_approved_lessons()`). A conflict blocks approval per the invariant above; no conflict proceeds to `approve_lesson(conn, lesson_id, scope, reviewer="agent-auto", rule_text)`.

On `approve=False`: `reject_lesson(conn, lesson_id, reviewer="agent-auto")`, with `auto_decision_reasoning` set from the judge's own stated reasoning.

## Pipeline integration — three phases in the same daily job, not two new connections held open

Task 4 of the live-lessons feature already fixed one instance of "DuckDB's exclusive file lock held across a network/LLM call" (the prepare/commit split in `generate_daily_lessons`). This judge step reintroduces the same shape of risk if built carelessly, so it gets the identical treatment. New `auto_judge_live_lessons(duckdb_manager, llm_invoke) -> list[...]` in `live_lessons.py`, called from `scheduler_wiring.py`'s `_lessons_job()` right after `commit_lesson_batches()`, itself internally three phases:

1. **Brief read**: open a connection, call a new `list_pending_by_source(conn, source="live")` (new, since nothing today lists pending candidates at all — this codebase's only existing lesson-review entry points are `approve`/`reject` by a known id), close the connection.
2. **LLM work, zero DB connections open**: for each candidate, `judge_lesson_candidate()`; on approve, `generate_rule_from_lesson()` then (if that succeeded) a second brief read to fetch `load_approved_lessons()` for conflict context, then `find_conflicting_rule()`. All of this happens with no DuckDB connection held.
3. **Brief write**: open one connection, apply every decision (`approve_lesson`/`reject_lesson`/leave-pending-with-reasoning), close it.

`WHERE source = 'live'` naturally excludes both `source = 'train'` rows and pre-migration `source IS NULL` rows (SQL's `NULL = 'live'` is never true) — the human-review population for `agent-train` is structurally unreachable from this code path, not just conventionally avoided.

## Error handling

- **Judge or distillation LLM call fails outright** (not just "declines" — an actual exception): same posture as the existing `generate_batch_reflection`/`generate_rule_from_lesson` contract — caught, logged, candidate left `pending` for the next run to retry. Nothing here should ever crash the daily job over one bad LLM call (`_lessons_job()`'s own already-established resilience, extended to this step too).
- **Conflict found**: left `pending`, `auto_decision_reasoning` names the conflicting rule and why — visible to you the next time you look, but never blocking the rest of that day's batch or forcing a decision. Since the row stays `pending`, the next day's run re-judges it from scratch too — if the same conflicting rule is still approved, expect the same conflict to be found again, indefinitely, until you resolve it via the normal CLI (approve/reject it yourself, or address the conflicting rule). That's the safety valve working as intended, not a bug to fix later.
- **Zero pending live-sourced candidates**: the whole step is a no-op, same as every other empty-input case in this pipeline.

## Explicitly out of scope

- Anything touching `source='train'` rows, ever.
- Reconsidering an already-approved or already-rejected `source='live'` lesson (Phase 2).
- Attribution tracking (which lessons applied to which recommendation) — deliberately not built now, see Scope above.
- A `--force`-equivalent override for auto-judging a conflict — conflicts always defer to a human, no autonomous override path exists.
