# Lesson Staleness / Versioning — Design

**Date:** 2026-09-22
**Status:** approved, pending implementation plan
**Origin:** direct user observation — the `agent_lessons` system doesn't account for model/agent iterations at all; conceptually, a lesson approved under one model or prompt version may no longer hold once that version changes, and nothing today notices.

## Problem

Confirmed via code research, not assumed:

- `agent_lessons` (`src/agent/lessons.py:35-46`) has no model-version, prompt-version, or config-hash column of any kind.
- `load_approved_lessons()` (`lessons.py:224-267`), the sole read path, injected into every future prompt via `pipeline.py:283-307`, pulls every approved, scope-matching rule with no filter on age, model version, or config version. A lesson approved months ago under a since-replaced model is injected exactly as confidently as one approved yesterday.
- No staleness/expiry/supersession mechanism exists — only `pending → approved`/`rejected` transitions. The only "cleanup" that has ever happened was a manual one-off data wipe (`agent_techspec.md:1275-1277`), not a designed process.
- Current scale: 4 approved lessons (2026-07-28 to 2026-09-07), 453 pending (all `source='train'`), none ever re-reviewed since approval.

Two lesson sources already exist (`agent_lessons.source`), and they need different treatment:
- **`train`** — a discrete `agent-train` batch run, evaluated against a specific historical snapshot corpus at a specific model/config version. All lessons from one run share the same evaluation context.
- **`live`** — trickles in daily (`app/backend/live_lessons.py`, `documents/app_user_stories.md` Phase 39-41) under whatever model/config was live that specific day.

Not every lesson is equally version-sensitive: a reasoning/prompt-behavior lesson (e.g. `A120`'s "bench ≠ injured") would likely survive a pure ML-model swap fine; a lesson correcting a specific model-calibration quirk would not.

## Decisions from clarifying questions

- **Sensitivity classification** (does this lesson survive an ML model change): set by the LLM during `judge_lesson_candidate`'s existing per-lesson judgment pass, not the human reviewer and not "treat everything as model-sensitive." Defaults to `false` (model-sensitive) whenever the LLM's response doesn't clearly say otherwise — never silently defaults to "safe."
- **Trigger scope**: whole-competition, not per-market/target. Lessons aren't tagged by which market they concern today (no such column exists), so the fingerprint covers that competition's *entire* `model_selection.yaml` entry (every target) plus the agent config. Coarser than a per-target trigger, but requires no new market-tagging schema/classification on top of the sensitivity tag already decided, and a false-positive re-review is cheap at this lesson volume.
- **Mechanism**: a lazy check at read time inside `load_approved_lessons`, not a scheduled re-scan job and not an event hook on promotion. Rejected alternatives: a scheduled job adds new infrastructure and leaves a window where a stale lesson could still be injected before it next runs; an event hook on model promotion doesn't reliably work in this codebase — this session's own history shows several promotions were hand-edited directly into `config/model_selection.yaml` with no single CLI code path to hook into.

## Schema changes

New columns on `agent_lessons`:
- `run_id` — the `agent-train` run's already-generated UUID (`main.py:1767`, currently reaches only `agent_telemetry` and stops there), threaded through `insert_lesson_candidate` for `source='train'` rows. Groups one run's lessons for reviewer convenience — see "Run-level batching" below. `NULL` for `source='live'` rows (they already have a natural per-day grouping via `live_lessons.py`'s existing daily batching).
- `model_fingerprint` — a hash of the competition's full `model_selection.yaml` entry (every target) as of generation time.
- `agent_config_fingerprint` — reuses `compute_agent_config_hash()` (`app/backend/agent_config_hash.py:13-30`) directly; this value already exists and already changes independently of `model_fingerprint`.
- `survives_model_change` (boolean, default `false`) — the LLM's classification from `judge_lesson_candidate`.

## The read-time check

Inside `load_approved_lessons`, for each approved lesson under consideration:

1. Compute the competition's *current* `model_fingerprint` and `agent_config_fingerprint`.
2. Compare against the lesson's stored values:
   - `agent_config_fingerprint` mismatch → always disqualifies the lesson from injection this call.
   - `model_fingerprint` mismatch → disqualifies unless `survives_model_change=True`.
   - A `NULL` stored fingerprint (pre-migration lessons) is treated as an automatic mismatch.
3. On disqualification: exclude the lesson from the injected set for this call (fail-safe — never inject a stale lesson into a live prompt), and flip its `status` to `needs_review`, recording which fingerprint(s) mismatched in the existing `auto_decision_reasoning` column (already exists for exactly this "why did an automated step decide this" purpose — no new column for the reason text).

**Run-level batching is a consequence, not new logic.** All lessons from one `agent-train` run share identical fingerprint values (generated in the same run, against the same active versions at that moment), so the per-lesson check above naturally produces the same pass/fail result across an entire run without any batch-specific code. `run_id` exists so a reviewer can see "these 40 flagged lessons all came from the same now-stale run" and act on them together, not because the check itself needs to treat a run as a unit.

## Surfacing to the reviewer

No new review UI. `needs_review` becomes one more listed/filterable status on the existing `GET /api/admin/lessons` endpoint and the existing `agent-lessons approve/reject` CLI flow, with `auto_decision_reasoning` shown (the mismatch reason) and `train`-sourced rows groupable/sortable by `run_id`.

## Migration

The 4 existing approved lessons predate this feature and have `NULL` fingerprints. Per the read-time check's own rule (a `NULL` fingerprint is an automatic mismatch), all 4 flip to `needs_review` the first time `load_approved_lessons` runs after this ships — a cheap, one-time forced human pass over exactly the lessons that exist today. No separate backfill script needed at this volume.

## Testing

- `load_approved_lessons` excludes a lesson whose `model_fingerprint` no longer matches current `model_selection.yaml` for its competition, and flips its status to `needs_review` with a reason recorded.
- A lesson with `survives_model_change=True` is not excluded when only `model_fingerprint` changes, but is excluded when `agent_config_fingerprint` changes.
- `insert_lesson_candidate` correctly threads `run_id` for `source='train'` rows; `source='live'` rows get correct per-day fingerprints via `live_lessons.py`.
- Pre-migration lessons (`NULL` fingerprints) flip to `needs_review` on first post-deploy read.
- `judge_lesson_candidate`'s sensitivity classification defaults to `false` on an ambiguous or missing LLM response.
- Admin listing surfaces `needs_review` lessons with their mismatch reason, grouped by `run_id` where applicable.
