# Lesson Staleness/Versioning (A127) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Fingerprint every approved `agent_lessons` row (model version + agent config version at approval time) and disqualify/flag stale ones as `needs_review` at the exact point they'd otherwise be injected into a live prompt, so a lesson approved under a since-replaced model or config stops being trusted silently.

**Architecture:** Two new pure fingerprint helpers (one relocated for layering, one new); four additive columns on `agent_lessons`; a read-time compare-and-flip inside `load_approved_lessons` that never blocks on a read-only connection; fingerprints captured at *approval* time (not insert time — see Deviations below) via `approve_lesson`'s three real call sites; a new lightweight LLM classification call for the train/human-approval path, folded into the existing per-lesson judgment call for the live/autonomous path.

**Tech Stack:** Python, DuckDB, pytest. No new dependencies.

**Design doc:** `docs/superpowers/specs/2026-09-22-lesson-staleness-versioning-design.md` (approved). Two deliberate deviations from its literal wording, both confirmed with the user before this plan was written:

1. **Fingerprints captured at approval time, not insert/generation time.** The design doc says "as of generation time." A train-sourced lesson can sit pending for weeks before a human reviews it; fingerprinting at insert time means it could already read as stale the instant it's approved. Fingerprinting at approval time means "valid as of the model/config active right now — re-check once that changes," which is both more useful and lets the classification call (below) live in one place.
2. **A new `classify_lesson_sensitivity()` function for the train/human-approval path.** The design doc says the sensitivity classification is "set by the LLM during `judge_lesson_candidate`'s existing per-lesson judgment pass" — but `judge_lesson_candidate` is only ever called for `source='live'` lessons (`auto_judge_live_lessons`). All 453 currently-pending lessons are `source='train'`, approved via a human CLI command that calls no LLM judgment today. The live path still folds the classification into `judge_lesson_candidate`'s existing single JSON response (zero extra LLM round-trips); the train path gets one new, small, separate function, called once per human approval action (not per pending lesson — bounded cost).

---

## File Structure

| File | Change |
|---|---|
| `src/agent/agent_config_hash.py` | **New** — relocated canonical `compute_agent_config_hash()` (was `app/backend/agent_config_hash.py`). Fixes a layering violation: `src/agent/lessons.py` needs this value and must not import from `app.backend`. |
| `app/backend/agent_config_hash.py` | **Modify** — becomes a one-line re-export of the relocated function. Every existing caller's import path is unchanged. |
| `src/agent/lesson_fingerprint.py` | **New** — `compute_model_fingerprint(competition_id)`, hashing one competition's full `model_selection.yaml` entry. |
| `src/agent/lessons.py` | **Modify** — schema migration (4 new columns), `run_id` threaded into `insert_lesson_candidate`, `LessonDecision`/`judge_lesson_candidate` extended, new `classify_lesson_sensitivity()`, `approve_lesson()` extended, `load_approved_lessons()` rewritten with the staleness check. |
| `src/agent/pipeline.py` | **Modify** — `lessons_node` takes an optional `config` param and computes/passes current fingerprints. |
| `src/agent/graph.py` | **Modify** — `build_graph` binds `config` into `lessons_node` via a closure, mirroring the existing `forecast_node_with_config` pattern. |
| `main.py` | **Modify** — `run_agent_lessons_approve` computes fingerprints + calls `classify_lesson_sensitivity`; `_write_train_artifacts` threads `run_id` into both `insert_lesson_candidate` call sites. |
| `app/backend/live_lessons.py` | **Modify** — `auto_judge_live_lessons` takes a `config` param, threads fingerprints/`survives_model_change` into its `approve_lesson` call. |
| `app/backend/scheduler_wiring.py` | **Modify** — one-line: pass the already-in-scope `config` into `auto_judge_live_lessons`. |
| `app/backend/main.py` | **Modify** — `list_lessons`'s `status` Literal gains `"needs_review"`; `run_id` added to the SELECT/response columns. |
| Tests | New/modified across `tests/test_agent_lessons.py`, `tests/test_agent_pipeline.py`, `tests/test_main_agent_lessons.py`, `app/backend/tests/test_live_lessons.py`, `app/backend/tests/test_list_lessons_endpoint.py` (exact names confirmed in Task 0). |

---

## Task 0: Confirm exact existing test file names

**Files:** none modified — read-only recon so later tasks reference real paths.

- [ ] **Step 1: List the real test files this plan will touch**

Run: `ls tests/ | grep -i lesson && ls app/backend/tests/ | grep -i lesson`

Expected: a list including (names may differ slightly from the plan's guesses above) files covering `src/agent/lessons.py`, `src/agent/pipeline.py`'s `lessons_node`, `main.py`'s `run_agent_lessons_approve`/`run_agent_lessons_reject`, and `app/backend/live_lessons.py`. Note the real names and substitute them into every later task's `Run:` lines before executing that task.

---

## Task 1: Relocate `compute_agent_config_hash` to fix the layering violation

**Files:**
- Create: `src/agent/agent_config_hash.py`
- Modify: `app/backend/agent_config_hash.py`
- Test: `app/backend/tests/test_recommendation_cache.py` (existing — must still pass unchanged; it's the closest thing to a direct test of this function today)

- [ ] **Step 1: Create the relocated module**

```python
# src/agent/agent_config_hash.py
"""Stable hash of an AgentConfig's tunable fields. Canonical location as of
A127 -- src/agent (the agent engine) must not depend on app.backend (the web
app depends on the engine, never the reverse; every other module in this
directory already follows that direction). Originally lived at
app/backend/agent_config_hash.py (W11, used as part of the recommendation
cache key); that module now re-exports this one so every existing caller's
import path stays unchanged. A127 needs this same value (agent_config_fingerprint)
computed from inside src/agent/lessons.py, which cannot reach into app.backend."""

from __future__ import annotations

import hashlib
import json

from src.agent.agent_config import AgentConfig


def compute_agent_config_hash(config: AgentConfig) -> str:
    payload = json.dumps(
        {
            "model": config.model,
            "provider": config.provider,
            "temperature": config.temperature,
            "max_tool_calls": config.max_tool_calls,
            "min_odds_threshold": config.min_odds_threshold,
            "max_odds_threshold": config.max_odds_threshold,
            "min_conditional_odds_threshold": config.min_conditional_odds_threshold,
            "max_conditional_odds_threshold": config.max_conditional_odds_threshold,
            "min_value_edge": config.min_value_edge,
            "min_value_edge_result_3way_draw": config.min_value_edge_result_3way_draw,
            "markets": config.markets,
            "system_prompt_version": config.system_prompt_version,
        },
        sort_keys=True,
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]
```

- [ ] **Step 2: Replace the old module with a re-export**

```python
# app/backend/agent_config_hash.py
"""Relocated to src/agent/agent_config_hash.py (A127) -- src/agent needed this
value and must not be depended on by app.backend in reverse. Re-exported here
so every existing caller (eod_batch.py, recommendation_cache.py, main.py,
t30_refresh.py, ...) keeps working with no import-path change."""

from __future__ import annotations

from src.agent.agent_config_hash import compute_agent_config_hash

__all__ = ["compute_agent_config_hash"]
```

- [ ] **Step 3: Run the existing test suite to confirm zero regressions**

Run: `python -m pytest app/backend/tests/test_recommendation_cache.py app/backend/tests/test_eod_batch.py app/backend/tests/test_t30_refresh.py -q`
Expected: PASS, same count as before this change.

- [ ] **Step 4: Commit**

```bash
git add src/agent/agent_config_hash.py app/backend/agent_config_hash.py
git commit -m "refactor(agent): relocate compute_agent_config_hash into src/agent (A127 prep)"
```

---

## Task 2: `compute_model_fingerprint`

**Files:**
- Create: `src/agent/lesson_fingerprint.py`
- Test: `tests/test_lesson_fingerprint.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_lesson_fingerprint.py
from __future__ import annotations

import yaml

from src.agent.lesson_fingerprint import compute_model_fingerprint


def _write_selection_yaml(tmp_path, contexts: dict) -> str:
    path = tmp_path / "model_selection.yaml"
    path.write_text(yaml.dump({"contexts": contexts}))
    return str(path)


def test_returns_none_for_a_competition_with_no_contexts_entry(tmp_path):
    path = _write_selection_yaml(tmp_path, {"E0": {"result_3way": {"model_path": "a.joblib"}}})
    assert compute_model_fingerprint("SWE", selection_path=path) is None


def test_same_entry_hashes_identically(tmp_path):
    contexts = {"E0": {"result_3way": {"model_path": "a.joblib"}, "btts": {"model_path": "b.joblib"}}}
    path = _write_selection_yaml(tmp_path, contexts)
    assert compute_model_fingerprint("E0", selection_path=path) == compute_model_fingerprint("E0", selection_path=path)


def test_changing_any_target_field_changes_the_fingerprint(tmp_path):
    path_a = _write_selection_yaml(tmp_path, {"E0": {"result_3way": {"model_path": "a.joblib"}}})
    path_b = _write_selection_yaml(tmp_path, {"E0": {"result_3way": {"model_path": "a-retrained.joblib"}}})
    assert compute_model_fingerprint("E0", selection_path=path_a) != compute_model_fingerprint("E0", selection_path=path_b)


def test_a_different_competitions_own_change_does_not_affect_this_ones_fingerprint(tmp_path):
    path_a = _write_selection_yaml(tmp_path, {
        "E0": {"result_3way": {"model_path": "a.joblib"}},
        "SP1": {"result_3way": {"model_path": "sp1-old.joblib"}},
    })
    path_b = _write_selection_yaml(tmp_path, {
        "E0": {"result_3way": {"model_path": "a.joblib"}},
        "SP1": {"result_3way": {"model_path": "sp1-retrained.joblib"}},
    })
    assert compute_model_fingerprint("E0", selection_path=path_a) == compute_model_fingerprint("E0", selection_path=path_b)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_lesson_fingerprint.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'src.agent.lesson_fingerprint'`

- [ ] **Step 3: Write the implementation**

```python
# src/agent/lesson_fingerprint.py
"""A127: model-version fingerprinting for lesson staleness. Whole-competition
scope (not per-market/target) -- agent_lessons rows aren't tagged by which
market they concern, so the fingerprint covers a competition's entire
model_selection.yaml entry (every target). Coarser than a per-target trigger,
deliberately -- see the design doc's "Trigger scope" decision."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import yaml

_DEFAULT_SELECTION_PATH = Path("config/model_selection.yaml")


def compute_model_fingerprint(
    competition_id: str | None, selection_path: str | Path = _DEFAULT_SELECTION_PATH,
) -> str | None:
    """Hash of every target's full model_selection.yaml entry for one
    competition -- changes whenever ANY target's feature_subset/model_path/
    model_type/metric/selected_at changes for this competition, whether via
    ModelSelector.run() or a direct hand-edit (both are real, confirmed
    promotion paths in this codebase). None when the competition has no
    contexts entry at all (e.g. an unrecognized/leagueless competition_id)
    -- callers treat None the same as any other mismatch, never as "nothing
    to check"."""
    with open(selection_path) as f:
        config = yaml.safe_load(f) or {}
    entry = (config.get("contexts") or {}).get(competition_id)
    if entry is None:
        return None
    payload = json.dumps(entry, sort_keys=True)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_lesson_fingerprint.py -v`
Expected: 4 passed

- [ ] **Step 5: Commit**

```bash
git add src/agent/lesson_fingerprint.py tests/test_lesson_fingerprint.py
git commit -m "feat(agent): add compute_model_fingerprint (A127)"
```

---

## Task 3: Schema migration + `run_id` threaded into `insert_lesson_candidate`

**Files:**
- Modify: `src/agent/lessons.py` (`create_lessons_tables`, `insert_lesson_candidate`)
- Test: `tests/test_agent_lessons.py` (substitute the real filename found in Task 0 if different)

- [ ] **Step 1: Write the failing tests**

```python
# add to tests/test_agent_lessons.py

def test_create_lessons_tables_adds_a127_columns(tmp_path):
    import duckdb
    from src.agent.lessons import create_lessons_tables

    conn = duckdb.connect(":memory:")
    create_lessons_tables(conn)
    columns = {row[0] for row in conn.execute("DESCRIBE agent_lessons").fetchall()}
    assert {"run_id", "model_fingerprint", "agent_config_fingerprint", "survives_model_change"} <= columns


def test_insert_lesson_candidate_stores_run_id_for_train_rows():
    import duckdb
    from src.agent.lessons import create_lessons_tables, insert_lesson_candidate

    conn = duckdb.connect(":memory:")
    create_lessons_tables(conn)
    lesson_id = insert_lesson_candidate(conn, "lesson text", "E0", "competition_specific", "m1", run_id="run-abc")
    stored = conn.execute("SELECT run_id FROM agent_lessons WHERE id = ?", [lesson_id]).fetchone()[0]
    assert stored == "run-abc"


def test_insert_lesson_candidate_defaults_run_id_to_null():
    import duckdb
    from src.agent.lessons import create_lessons_tables, insert_lesson_candidate

    conn = duckdb.connect(":memory:")
    create_lessons_tables(conn)
    lesson_id = insert_lesson_candidate(conn, "lesson text", "E0", "competition_specific", "m1", source="live")
    stored = conn.execute("SELECT run_id FROM agent_lessons WHERE id = ?", [lesson_id]).fetchone()[0]
    assert stored is None
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_agent_lessons.py -k "a127_columns or stores_run_id or defaults_run_id" -v`
Expected: FAIL — `create_lessons_tables` produces no such columns yet; `insert_lesson_candidate()` raises `TypeError: unexpected keyword argument 'run_id'`.

- [ ] **Step 3: Migrate the schema and thread `run_id` through**

In `src/agent/lessons.py`, inside `create_lessons_tables`, right after the existing `auto_decision_reasoning` migration line:

```python
    conn.execute("ALTER TABLE agent_lessons ADD COLUMN IF NOT EXISTS auto_decision_reasoning TEXT")
    # A127: lesson staleness/versioning. run_id groups one agent-train run's
    # lessons for reviewer convenience (NULL for source='live' rows -- they
    # already have a natural per-day grouping via live_lessons.py's own
    # batching). model_fingerprint/agent_config_fingerprint/survives_model_change
    # are populated at APPROVAL time, not insert time (see this story's plan
    # doc, "Deviations" -- fingerprinting at insert time would let a
    # slow-reviewed lesson go stale before it's ever approved). NULL on
    # every pre-migration approved row -- load_approved_lessons() treats a
    # NULL stored fingerprint as an automatic mismatch, so the 4 lessons
    # that existed before this shipped all flip to needs_review the first
    # time they're read after this migration runs; no separate backfill
    # script needed at this volume.
    conn.execute("ALTER TABLE agent_lessons ADD COLUMN IF NOT EXISTS run_id TEXT")
    conn.execute("ALTER TABLE agent_lessons ADD COLUMN IF NOT EXISTS model_fingerprint TEXT")
    conn.execute("ALTER TABLE agent_lessons ADD COLUMN IF NOT EXISTS agent_config_fingerprint TEXT")
    conn.execute("ALTER TABLE agent_lessons ADD COLUMN IF NOT EXISTS survives_model_change BOOLEAN DEFAULT false")
```

Then update `insert_lesson_candidate`:

```python
def insert_lesson_candidate(
    conn: duckdb.DuckDBPyConnection,
    lesson_text: str,
    competition_id: str | None,
    tier: str,
    source_match_id: str,
    source: str = "train",
    run_id: str | None = None,
) -> int:
    """Insert a pending, unscoped lesson candidate. Returns its id.

    source: 'train' (default, preserves every pre-existing caller
    unchanged -- agent-train's own CLI path) or 'live'
    (app/backend/live_lessons.py's commit_lesson_batches, the only caller
    that passes this explicitly).

    run_id (A127): the agent-train run's UUID, for source='train' rows only
    -- every other caller omits it and gets NULL, unchanged."""
    row = conn.execute(
        """
        INSERT INTO agent_lessons (lesson_text, status, competition_id, tier, source_match_id, created_at, source, run_id)
        VALUES (?, 'pending', ?, ?, ?, ?, ?, ?)
        RETURNING id
        """,
        [lesson_text, competition_id, tier, source_match_id, datetime.now(timezone.utc), source, run_id],
    ).fetchone()
    return int(row[0])
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_agent_lessons.py -k "a127_columns or stores_run_id or defaults_run_id" -v`
Expected: 3 passed

- [ ] **Step 5: Run the full existing lessons test file to confirm zero regressions**

Run: `python -m pytest tests/test_agent_lessons.py -v`
Expected: every pre-existing test still PASS (insert_lesson_candidate's new param is keyword-only-by-default-value, every old call site unaffected).

- [ ] **Step 6: Commit**

```bash
git add src/agent/lessons.py tests/test_agent_lessons.py
git commit -m "feat(agent): A127 schema migration -- run_id/model_fingerprint/agent_config_fingerprint/survives_model_change columns"
```

---

## Task 4: Extend `judge_lesson_candidate` with `survives_model_change` (live path)

**Files:**
- Modify: `src/agent/lessons.py` (`LessonDecision`, `judge_lesson_candidate`)
- Test: `tests/test_agent_lessons.py`

- [ ] **Step 1: Write the failing tests**

```python
# add to tests/test_agent_lessons.py

def test_judge_lesson_candidate_parses_survives_model_change_true():
    from src.agent.lessons import judge_lesson_candidate

    def fake_invoke(prompt):
        return '{"approve": true, "scope": "tier", "reasoning": "clear pattern", "survives_model_change": true}'

    decision = judge_lesson_candidate("some lesson text", "E0", "competition_specific", fake_invoke)
    assert decision.survives_model_change is True


def test_judge_lesson_candidate_defaults_survives_model_change_false_when_absent():
    from src.agent.lessons import judge_lesson_candidate

    def fake_invoke(prompt):
        return '{"approve": true, "scope": "tier", "reasoning": "clear pattern"}'

    decision = judge_lesson_candidate("some lesson text", "E0", "competition_specific", fake_invoke)
    assert decision.survives_model_change is False


def test_judge_lesson_candidate_defaults_survives_model_change_false_on_non_boolean():
    from src.agent.lessons import judge_lesson_candidate

    def fake_invoke(prompt):
        return '{"approve": true, "scope": "tier", "reasoning": "clear pattern", "survives_model_change": "yes"}'

    decision = judge_lesson_candidate("some lesson text", "E0", "competition_specific", fake_invoke)
    assert decision.survives_model_change is False


def test_judge_lesson_candidate_defaults_survives_model_change_false_on_exception():
    from src.agent.lessons import judge_lesson_candidate

    def fake_invoke(prompt):
        raise RuntimeError("boom")

    decision = judge_lesson_candidate("some lesson text", "E0", "competition_specific", fake_invoke)
    assert decision.survives_model_change is False
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_agent_lessons.py -k survives_model_change -v`
Expected: FAIL — `AttributeError: 'LessonDecision' object has no attribute 'survives_model_change'`

- [ ] **Step 3: Extend the dataclass and the prompt/parse logic**

```python
@dataclass
class LessonDecision:
    approve: bool
    scope: str | None  # "competition" | "tier", only set when approve=True
    reasoning: str      # always set -- the audit trail (agent_lessons.auto_decision_reasoning)
    # A127: does this lesson generalize across an ML-model swap (True) or is
    # it tied to this specific model's current calibration quirk (False,
    # the default)? Only ever meaningful when approve=True -- a rejected
    # lesson never gets fingerprinted at all. Defaults False on any
    # ambiguous/missing/non-boolean response or a raised exception -- same
    # fail-closed posture as `approve` itself.
    survives_model_change: bool = False
```

Update `judge_lesson_candidate`'s prompt (inside the existing function body) to also ask for the new field:

```python
    prompt = (
        f"You are deciding whether to promote a batch of live betting-recommendation results into a "
        f"standing rule for an automated agent's future recommendations in this competition "
        f"(competition_id={competition_id!r}, tier={tier!r}).\n\n"
        f"{lesson_text}\n\n"
        "Only approve if the pattern is clearly systematic, not noise from a small sample -- when in "
        "doubt, reject. If you approve, also decide scope: \"competition\" if the pattern is specific to "
        "this one competition, \"tier\" if it reflects something general enough to apply to every "
        "competition of this tier. Also decide (A127): would this rule still hold even if the "
        "underlying ML forecasting model were retrained or replaced (a general reasoning/prompt-behavior "
        "insight), or is it tied to this specific model's current calibration/output quirk and likely to "
        "stop holding once that model changes?\n\n"
        "Respond with exactly one JSON object, nothing else, with \"approve\" and \"survives_model_change\" "
        "as JSON boolean literals (not strings): "
        '{"approve": true|false, "scope": "competition"|"tier"|null, "reasoning": "one or two sentences", '
        '"survives_model_change": true|false}'
    )
    try:
        parsed = _parse_judge_json(llm_invoke(prompt))
        # ponytail: `is True` (not bool(...)) -- a stringified "false", a
        # dict/list, or a nonzero int must all fail closed to reject, not
        # silently coerce to approve. See the coordinator's exploit report.
        approve = parsed["approve"] is True
        scope = parsed.get("scope") if approve else None
        reasoning = str(parsed.get("reasoning") or "").strip() or "(no reasoning given)"
        survives_model_change = parsed.get("survives_model_change") is True
        if approve and scope not in _VALID_SCOPES:
            return LessonDecision(
                approve=False, scope=None,
                reasoning=f"invalid scope {scope!r} returned -- defaulting to reject",
                survives_model_change=False,
            )
        return LessonDecision(approve=approve, scope=scope, reasoning=reasoning, survives_model_change=survives_model_change)
    except Exception as exc:
        return LessonDecision(
            approve=False, scope=None, reasoning=f"judge call failed ({exc!r}) -- defaulting to reject",
            survives_model_change=False,
        )
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_agent_lessons.py -k "judge_lesson_candidate or survives_model_change" -v`
Expected: all passed, including every pre-existing `judge_lesson_candidate` test (the new field is additive with a safe default).

- [ ] **Step 5: Commit**

```bash
git add src/agent/lessons.py tests/test_agent_lessons.py
git commit -m "feat(agent): judge_lesson_candidate classifies survives_model_change (A127 live path)"
```

---

## Task 5: `classify_lesson_sensitivity` (train/human-approval path)

**Files:**
- Modify: `src/agent/lessons.py`
- Test: `tests/test_agent_lessons.py`

- [ ] **Step 1: Write the failing tests**

```python
# add to tests/test_agent_lessons.py

def test_classify_lesson_sensitivity_true():
    from src.agent.lessons import classify_lesson_sensitivity

    def fake_invoke(prompt):
        return '{"survives_model_change": true, "reasoning": "pure reasoning insight"}'

    assert classify_lesson_sensitivity("bench != injured", "E0", "competition_specific", fake_invoke) is True


def test_classify_lesson_sensitivity_defaults_false_when_absent():
    from src.agent.lessons import classify_lesson_sensitivity

    def fake_invoke(prompt):
        return '{"reasoning": "no clear field"}'

    assert classify_lesson_sensitivity("some lesson", "E0", "competition_specific", fake_invoke) is False


def test_classify_lesson_sensitivity_defaults_false_on_exception():
    from src.agent.lessons import classify_lesson_sensitivity

    def fake_invoke(prompt):
        raise RuntimeError("boom")

    assert classify_lesson_sensitivity("some lesson", "E0", "competition_specific", fake_invoke) is False


def test_classify_lesson_sensitivity_defaults_false_on_non_boolean():
    from src.agent.lessons import classify_lesson_sensitivity

    def fake_invoke(prompt):
        return '{"survives_model_change": "true"}'

    assert classify_lesson_sensitivity("some lesson", "E0", "competition_specific", fake_invoke) is False
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_agent_lessons.py -k classify_lesson_sensitivity -v`
Expected: FAIL — `ImportError: cannot import name 'classify_lesson_sensitivity'`

- [ ] **Step 3: Write the implementation**

Add to `src/agent/lessons.py`, right after `judge_lesson_candidate`:

```python
def classify_lesson_sensitivity(
    lesson_text: str, competition_id: str | None, tier: str, llm_invoke: Callable[[str], str],
) -> bool:
    """A127: does this APPROVED lesson generalize across an ML-model swap (a
    reasoning/prompt-behavior insight, e.g. "a bench listing does not mean
    injured") or is it tied to this specific model's current calibration
    quirk (unlikely to still hold once that model is retrained/replaced)?

    Called once per human `agent-lessons approve` action on a train-sourced
    lesson (main.py's run_agent_lessons_approve) -- NOT called for
    live-sourced lessons, which get the equivalent question folded into
    judge_lesson_candidate's existing single JSON response instead, since
    that call already runs before every live-sourced approval and a second
    round-trip there would be pure waste.

    Defaults to False (model-sensitive -- the conservative, re-review-me
    default) on ANY failure, malformed JSON, or a non-boolean response --
    same fail-closed posture as judge_lesson_candidate; never silently
    assumes a lesson survives a model change it might not."""
    prompt = (
        f"A lesson was just approved for an automated betting agent evaluating {tier} matches "
        f"(competition_id={competition_id!r}):\n\n{lesson_text}\n\n"
        "Does this lesson reflect a general reasoning/prompt-behavior insight that would still hold "
        "even if the underlying ML forecasting model were retrained or replaced (e.g. \"a bench "
        "listing does not mean injured\")? Or is it tied to this specific model's current "
        "calibration/output quirk, and likely to stop holding once that model changes?\n\n"
        "Respond with exactly one JSON object, nothing else, with \"survives_model_change\" as a JSON "
        "boolean literal (not a string): "
        '{"survives_model_change": true|false, "reasoning": "one sentence"}'
    )
    try:
        parsed = _parse_judge_json(llm_invoke(prompt))
        return parsed.get("survives_model_change") is True
    except Exception:
        return False
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_agent_lessons.py -k classify_lesson_sensitivity -v`
Expected: 4 passed

- [ ] **Step 5: Commit**

```bash
git add src/agent/lessons.py tests/test_agent_lessons.py
git commit -m "feat(agent): classify_lesson_sensitivity for train-sourced lesson approvals (A127)"
```

---

## Task 6: Extend `approve_lesson` with fingerprints + `survives_model_change`

**Files:**
- Modify: `src/agent/lessons.py`
- Test: `tests/test_agent_lessons.py`

- [ ] **Step 1: Write the failing tests**

```python
# add to tests/test_agent_lessons.py

def test_approve_lesson_stores_fingerprints_and_survives_flag():
    import duckdb
    from src.agent.lessons import approve_lesson, create_lessons_tables, insert_lesson_candidate

    conn = duckdb.connect(":memory:")
    create_lessons_tables(conn)
    lesson_id = insert_lesson_candidate(conn, "lesson text", "E0", "competition_specific", "m1")
    approve_lesson(
        conn, lesson_id, "competition", "reviewer1", "NEVER do X.",
        model_fingerprint="mfp123", agent_config_fingerprint="cfp456", survives_model_change=True,
    )
    row = conn.execute(
        "SELECT model_fingerprint, agent_config_fingerprint, survives_model_change FROM agent_lessons WHERE id = ?",
        [lesson_id],
    ).fetchone()
    assert row == ("mfp123", "cfp456", True)


def test_approve_lesson_defaults_fingerprints_to_null_and_survives_to_false():
    import duckdb
    from src.agent.lessons import approve_lesson, create_lessons_tables, insert_lesson_candidate

    conn = duckdb.connect(":memory:")
    create_lessons_tables(conn)
    lesson_id = insert_lesson_candidate(conn, "lesson text", "E0", "competition_specific", "m1")
    approve_lesson(conn, lesson_id, "competition", "reviewer1", "NEVER do X.")
    row = conn.execute(
        "SELECT model_fingerprint, agent_config_fingerprint, survives_model_change FROM agent_lessons WHERE id = ?",
        [lesson_id],
    ).fetchone()
    assert row == (None, None, False)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_agent_lessons.py -k "approve_lesson_stores_fingerprints or approve_lesson_defaults_fingerprints" -v`
Expected: FAIL — `TypeError: approve_lesson() got an unexpected keyword argument 'model_fingerprint'`

- [ ] **Step 3: Extend the implementation**

```python
def approve_lesson(
    conn: duckdb.DuckDBPyConnection, lesson_id: int, scope: str, reviewer: str, rule_text: str,
    model_fingerprint: str | None = None,
    agent_config_fingerprint: str | None = None,
    survives_model_change: bool = False,
) -> None:
    """Approve a lesson, requiring the reviewer to pick a scope explicitly.

    scope='competition' pins the lesson to its recorded competition_id;
    scope='tier' widens it to every match resolving to its recorded tier.

    rule_text (A44) is required, not optional -- an approved lesson with no
    rule_text would silently vanish from live use (load_approved_lessons
    only reads rule_text), which is a worse failure mode than forcing every
    approval to supply one. Callers (main.py's run_agent_lessons_approve)
    either take it from --rule or auto-distill via generate_rule_from_lesson
    before calling this.

    model_fingerprint/agent_config_fingerprint/survives_model_change (A127):
    captured HERE, at approval time -- not at insert/generation time. A
    train-sourced lesson can sit pending for weeks; fingerprinting at insert
    time would let it read as already-stale the instant it's approved.
    Fingerprinting at approval time means "valid as of the model/config
    active right now" -- exactly the guarantee a reviewer is actually
    making. All three default to the pre-A127 behavior (NULL/False) for
    app/backend/main.py's sync_lessons, which reproduces an
    already-approved lesson from a different database and deliberately
    does NOT carry over its origin's fingerprints (the target deployment
    may run a different model) -- it lands with NULL fingerprints, which
    load_approved_lessons() treats as an automatic mismatch, forcing a
    fresh, correct re-approval on the target deployment."""
    if scope not in _VALID_SCOPES:
        raise ValueError(f"scope must be one of {_VALID_SCOPES}, got {scope!r}")
    if not rule_text or not rule_text.strip():
        raise ValueError("rule_text must be a non-empty string")
    _require_lesson_exists(conn, lesson_id)
    conn.execute(
        """
        UPDATE agent_lessons
        SET status = 'approved', scope = ?, rule_text = ?, reviewed_at = ?, reviewer = ?,
            model_fingerprint = ?, agent_config_fingerprint = ?, survives_model_change = ?
        WHERE id = ?
        """,
        [
            scope, rule_text.strip(), datetime.now(timezone.utc), reviewer,
            model_fingerprint, agent_config_fingerprint, survives_model_change, lesson_id,
        ],
    )
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_agent_lessons.py -v`
Expected: all passed, including every pre-existing `approve_lesson` test (new params all default to the exact prior behavior).

- [ ] **Step 5: Commit**

```bash
git add src/agent/lessons.py tests/test_agent_lessons.py
git commit -m "feat(agent): approve_lesson captures fingerprints at approval time (A127)"
```

---

## Task 7: Rewrite `load_approved_lessons` with the staleness check

**Files:**
- Modify: `src/agent/lessons.py`
- Test: `tests/test_agent_lessons.py`

This is the core read-time check. `current_model_fingerprint`/`current_agent_config_fingerprint` become required params (no silent-skip default) — every real caller is updated in Tasks 8-10.

- [ ] **Step 1: Write the failing tests**

```python
# add to tests/test_agent_lessons.py

def _approve(conn, competition_id="E0", tier="competition_specific", model_fp="mfp1", config_fp="cfp1", survives=False):
    from src.agent.lessons import approve_lesson, insert_lesson_candidate
    lesson_id = insert_lesson_candidate(conn, "lesson text", competition_id, tier, "m1")
    approve_lesson(
        conn, lesson_id, "competition", "reviewer1", "NEVER do X.",
        model_fingerprint=model_fp, agent_config_fingerprint=config_fp, survives_model_change=survives,
    )
    return lesson_id


def test_load_approved_lessons_returns_the_rule_when_fingerprints_match():
    import duckdb
    from src.agent.lessons import create_lessons_tables, load_approved_lessons

    conn = duckdb.connect(":memory:")
    create_lessons_tables(conn)
    _approve(conn)
    result = load_approved_lessons(conn, "E0", "competition_specific", "mfp1", "cfp1")
    assert result == ["NEVER do X."]


def test_load_approved_lessons_excludes_and_flips_on_model_fingerprint_mismatch():
    import duckdb
    from src.agent.lessons import create_lessons_tables, load_approved_lessons

    conn = duckdb.connect(":memory:")
    create_lessons_tables(conn)
    lesson_id = _approve(conn, model_fp="mfp-old", config_fp="cfp1", survives=False)

    result = load_approved_lessons(conn, "E0", "competition_specific", "mfp-new", "cfp1")

    assert result == []
    row = conn.execute("SELECT status, auto_decision_reasoning FROM agent_lessons WHERE id = ?", [lesson_id]).fetchone()
    assert row[0] == "needs_review"
    assert "model_fingerprint" in row[1]


def test_load_approved_lessons_keeps_a_lesson_that_survives_model_change():
    import duckdb
    from src.agent.lessons import create_lessons_tables, load_approved_lessons

    conn = duckdb.connect(":memory:")
    create_lessons_tables(conn)
    _approve(conn, model_fp="mfp-old", config_fp="cfp1", survives=True)

    result = load_approved_lessons(conn, "E0", "competition_specific", "mfp-new", "cfp1")

    assert result == ["NEVER do X."]


def test_load_approved_lessons_excludes_even_a_survivor_on_agent_config_mismatch():
    import duckdb
    from src.agent.lessons import create_lessons_tables, load_approved_lessons

    conn = duckdb.connect(":memory:")
    create_lessons_tables(conn)
    lesson_id = _approve(conn, model_fp="mfp1", config_fp="cfp-old", survives=True)

    result = load_approved_lessons(conn, "E0", "competition_specific", "mfp1", "cfp-new")

    assert result == []
    status = conn.execute("SELECT status FROM agent_lessons WHERE id = ?", [lesson_id]).fetchone()[0]
    assert status == "needs_review"


def test_load_approved_lessons_treats_a_null_stored_fingerprint_as_a_mismatch():
    """Migration case: the 4 pre-A127 approved lessons have NULL fingerprints."""
    import duckdb
    from src.agent.lessons import approve_lesson, create_lessons_tables, insert_lesson_candidate, load_approved_lessons

    conn = duckdb.connect(":memory:")
    create_lessons_tables(conn)
    lesson_id = insert_lesson_candidate(conn, "lesson text", "E0", "competition_specific", "m1")
    approve_lesson(conn, lesson_id, "competition", "reviewer1", "NEVER do X.")  # no fingerprints -- pre-migration shape

    result = load_approved_lessons(conn, "E0", "competition_specific", "mfp-anything", "cfp-anything")

    assert result == []
    status = conn.execute("SELECT status FROM agent_lessons WHERE id = ?", [lesson_id]).fetchone()[0]
    assert status == "needs_review"


def test_load_approved_lessons_does_not_raise_on_a_read_only_connection(tmp_path):
    """lessons_node's own live-serving connection is read_only=True -- the
    exclusion must still apply, and the flip attempt must not raise, only
    be skipped."""
    import duckdb
    from src.agent.lessons import approve_lesson, create_lessons_tables, insert_lesson_candidate, load_approved_lessons

    db_path = str(tmp_path / "test.db")
    conn = duckdb.connect(db_path)
    create_lessons_tables(conn)
    lesson_id = insert_lesson_candidate(conn, "lesson text", "E0", "competition_specific", "m1")
    approve_lesson(
        conn, lesson_id, "competition", "reviewer1", "NEVER do X.",
        model_fingerprint="mfp-old", agent_config_fingerprint="cfp1", survives_model_change=False,
    )
    conn.close()

    ro_conn = duckdb.connect(db_path, read_only=True)
    result = load_approved_lessons(ro_conn, "E0", "competition_specific", "mfp-new", "cfp1")
    ro_conn.close()

    assert result == []
    # status flip was skipped (read-only), not raised -- reopen writable to confirm it's still 'approved'
    rw_conn = duckdb.connect(db_path)
    status = rw_conn.execute("SELECT status FROM agent_lessons WHERE id = ?", [lesson_id]).fetchone()[0]
    assert status == "approved"


def test_load_approved_lessons_still_dedupes_identical_rule_text():
    import duckdb
    from src.agent.lessons import approve_lesson, create_lessons_tables, insert_lesson_candidate, load_approved_lessons

    conn = duckdb.connect(":memory:")
    create_lessons_tables(conn)
    for _ in range(2):
        lesson_id = insert_lesson_candidate(conn, "lesson text", "E0", "competition_specific", "m1")
        approve_lesson(
            conn, lesson_id, "competition", "reviewer1", "NEVER do X.",
            model_fingerprint="mfp1", agent_config_fingerprint="cfp1",
        )
    result = load_approved_lessons(conn, "E0", "competition_specific", "mfp1", "cfp1")
    assert result == ["NEVER do X."]
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_agent_lessons.py -k load_approved_lessons -v`
Expected: FAIL — `TypeError: load_approved_lessons() missing 2 required positional arguments`

- [ ] **Step 3: Rewrite the implementation**

```python
def load_approved_lessons(
    conn: duckdb.DuckDBPyConnection,
    competition_id: str | None,
    tier: str,
    current_model_fingerprint: str | None,
    current_agent_config_fingerprint: str | None,
) -> list[str]:
    """Approved, distilled rule text (A44: rule_text, never the raw
    lesson_text) for one match's competition_id/tier. Excludes any approved
    row with a NULL rule_text (old rows approved before A44 shipped).
    Tolerates a missing agent_lessons table by returning no lessons rather
    than raising -- live recommendation runs must never fail just because
    train mode hasn't produced anything yet.

    A127 staleness check, applied to every otherwise-matching row before
    it's returned:
    - agent_config_fingerprint mismatch (stored value, including a NULL
      pre-migration one, != current_agent_config_fingerprint) ALWAYS
      disqualifies -- a config change (prompt version, thresholds, ...)
      can change what's safe to inject regardless of the ML model.
    - model_fingerprint mismatch disqualifies UNLESS the row's own
      survives_model_change=True.
    - A disqualified row is excluded from the returned set immediately
      (fail-safe: a stale lesson is never injected, regardless of what
      happens next) and, best-effort, flipped to 'needs_review' with the
      mismatch reason(s) recorded in auto_decision_reasoning. The flip is
      skipped (never raised) when `conn` is opened read_only=True --
      pipeline.py's lessons_node (live serving) and
      app/backend/live_lessons.py's auto_judge_live_lessons (its own
      conflict-check lookup) both use a read-only connection here by
      design; the exclusion above already gives the fail-safe guarantee
      either way, so the persisted flip (a nice-to-have audit signal) is
      simply deferred to whichever caller next holds a writable connection
      for this same row.

    current_model_fingerprint may be None (an unrecognized competition_id
    has no model_selection.yaml entry at all, see compute_model_fingerprint)
    -- that still correctly mismatches any real stored value via plain
    Python inequality, and also correctly matches another NULL/None stored
    value, which is what you want: two "no model registered" states are
    the same state, not a mismatch.

    W185 code-quality follow-up: dedup happens in Python (not SQL DISTINCT)
    since DISTINCT's own ORDER BY rule requires every ORDER BY column to
    also appear in the SELECT list, which created_at doesn't."""
    try:
        rows = conn.execute(
            """
            SELECT id, rule_text, model_fingerprint, agent_config_fingerprint, survives_model_change
            FROM agent_lessons
            WHERE status = 'approved'
              AND rule_text IS NOT NULL
              AND ((scope = 'competition' AND competition_id = ?)
                OR (scope = 'tier' AND tier = ?))
            ORDER BY created_at
            """,
            [competition_id, tier],
        ).fetchall()
    except duckdb.CatalogException:
        return []

    seen: set[str] = set()
    deduped: list[str] = []
    for lesson_id, rule_text, stored_model_fp, stored_config_fp, survives in rows:
        reasons = []
        if stored_config_fp != current_agent_config_fingerprint:
            reasons.append("agent_config_fingerprint changed since approval")
        if stored_model_fp != current_model_fingerprint and not survives:
            reasons.append("model_fingerprint changed since approval")
        if reasons:
            try:
                conn.execute(
                    "UPDATE agent_lessons SET status = 'needs_review', auto_decision_reasoning = ? WHERE id = ?",
                    ["; ".join(reasons), lesson_id],
                )
            except duckdb.InvalidInputException:
                pass  # read-only connection -- exclusion below still applies; flip deferred to a writable caller
            continue
        if rule_text not in seen:
            seen.add(rule_text)
            deduped.append(rule_text)
    return deduped
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_agent_lessons.py -v`
Expected: all passed. (This will also break every OTHER existing test that calls `load_approved_lessons` with the old 3-arg signature -- that's expected and fixed in Tasks 8-10, which are the only other call sites. Do not attempt to fix those callers' tests here.)

- [ ] **Step 5: Commit**

```bash
git add src/agent/lessons.py tests/test_agent_lessons.py
git commit -m "feat(agent): load_approved_lessons excludes and flips stale approved rows (A127)"
```

---

## Task 8: Wire the live-serving path (`pipeline.py` + `graph.py`)

**Files:**
- Modify: `src/agent/pipeline.py` (`lessons_node`)
- Modify: `src/agent/graph.py` (`build_graph`)
- Test: `tests/test_agent_pipeline.py`

- [ ] **Step 1: Add one new test for fingerprint threading -- confirmed the existing 8 `test_lessons_node_*` tests need NO changes**

Read via Task 0's recon: every existing `test_lessons_node_*` test in `tests/test_agent_pipeline.py` calls `lessons_node(state_dict)` with no second arg, and none of them assert on `load_approved_lessons`'s call args past index 2 (competition_id/tier) except one that stops at index 2 already. Since `config` will default to `None`, and `lessons_node`'s body (Step 3) only calls `compute_agent_config_hash` when `config is not None`, every existing test keeps passing unmodified -- `compute_model_fingerprint` runs for real (reads the actual `config/model_selection.yaml`, harmless) but its return value doesn't affect any of these tests' assertions since `load_approved_lessons` itself stays mocked. Add one new test instead, for the real fingerprint-threading behavior:

```python
def test_lessons_node_threads_config_fingerprints_into_load_approved_lessons():
    from src.agent.agent_config import AgentConfig
    from src.agent.pipeline import lessons_node
    from src.agent import tools as agent_tools

    agent_tools._snapshot_store.set_mode("live")
    config = AgentConfig.default()
    with patch("src.agent.lessons.load_approved_lessons", return_value=[]) as mock_load, \
         patch("src.utils.db_manager.DuckDBManager") as MockDB, \
         patch("src.agent.lesson_fingerprint.compute_model_fingerprint", return_value="mfp1"), \
         patch("src.agent.agent_config_hash.compute_agent_config_hash", return_value="cfp1"):
        MockDB.return_value.connection.return_value.__enter__.return_value = MagicMock()
        lessons_node({"competition_resolution": {"competition": "E0", "tier": "competition_specific"}}, config=config)

    assert mock_load.call_args.args[1:] == ("E0", "competition_specific", "mfp1", "cfp1")
```

- [ ] **Step 2: Run to verify the new test fails**

Run: `python -m pytest tests/test_agent_pipeline.py -k threads_config_fingerprints -v`
Expected: FAIL — `lessons_node()` doesn't accept `config` yet.

- [ ] **Step 3: Update `lessons_node`**

```python
def lessons_node(state: dict, config: "AgentConfig | None" = None) -> dict:
    """A33: inject reviewer-approved lessons scoped to this match's
    competition/tier as a HumanMessage before the LLM's turn -- same
    injection pattern forecast_node uses for evidence.

    ... (existing docstring paragraphs unchanged) ...

    config (A127): the AgentConfig this run is using -- needed to compute
    the CURRENT agent_config_fingerprint for load_approved_lessons' own
    staleness check. Optional/None-defaulted so every pre-A127 direct call
    of this function (existing unit tests that don't care about staleness)
    keeps working -- when None, this node computes NO fingerprints and
    calls load_approved_lessons with model_fingerprint=None and
    agent_config_fingerprint=None (both compare as a mismatch against ANY
    real stored, non-NULL fingerprint, which is the same fail-safe posture
    as "we don't know, so don't trust it" -- it does NOT silently skip the
    check). The real graph (graph.py's build_graph) always passes it via a
    closure, mirroring forecast_node_with_config."""
    from src.agent.tools import get_snapshot_store

    store = get_snapshot_store()
    if not (store.mode == "live" or (store.mode == "replay" and store.allow_lessons_in_replay)):
        return {}

    from src.agent.lesson_fingerprint import compute_model_fingerprint
    from src.agent.agent_config_hash import compute_agent_config_hash
    from src.agent.lessons import extract_competition_scope, load_approved_lessons
    from src.utils.db_manager import DuckDBManager

    competition_id, tier = extract_competition_scope(state)
    current_model_fingerprint = compute_model_fingerprint(competition_id)
    current_agent_config_fingerprint = compute_agent_config_hash(config) if config is not None else None
    try:
        with DuckDBManager().connection(read_only=True) as conn:
            lessons = load_approved_lessons(
                conn, competition_id, tier, current_model_fingerprint, current_agent_config_fingerprint,
            )
    except (duckdb.IOException, duckdb.ConnectionException):
        return {}
    if not lessons:
        return {}
    lessons_text = "Lessons from past evaluated matches:\n" + "\n".join(f"- {lesson}" for lesson in lessons)
    return {"messages": [HumanMessage(content=lessons_text)]}
```

- [ ] **Step 4: Wire `graph.py`'s `build_graph` to bind config into `lessons_node`**

```python
    def forecast_node_with_config(state: AgentState) -> dict:
        return forecast_node(state, suppress_uncertainty=config.suppress_forecast_uncertainty)

    def lessons_node_with_config(state: AgentState) -> dict:
        return lessons_node(state, config=config)

    graph = StateGraph(AgentState)
    graph.add_node("resolve_competition", resolve_competition_node)
    graph.add_node("research", research_node)
    graph.add_node("forecast", forecast_node_with_config)
    graph.add_node("lessons", lessons_node_with_config)
```

(Only the `graph.add_node("lessons", ...)` line changes, from `lessons_node` to the new `lessons_node_with_config`; the new closure function is added right next to the existing `forecast_node_with_config` one.)

- [ ] **Step 5: Run tests to verify they pass**

Run: `python -m pytest tests/test_agent_pipeline.py -v`
Expected: all passed.

- [ ] **Step 6: Run the full existing test suite's agent-graph coverage to confirm no wiring regression**

Run: `python -m pytest tests/test_agent_graph.py tests/test_main_agent_snapshot.py -q` (substitute real graph-level test filenames if different from Task 0's recon)
Expected: PASS, unchanged counts.

- [ ] **Step 7: Commit**

```bash
git add src/agent/pipeline.py src/agent/graph.py tests/test_agent_pipeline.py
git commit -m "feat(agent): thread AgentConfig into lessons_node for A127 staleness check"
```

---

## Task 9: Wire the train/human-approval path (`main.py`)

**Files:**
- Modify: `main.py` (`run_agent_lessons_approve`, `_write_train_artifacts`)
- Test: `tests/test_main_agent_lessons.py`, `tests/test_main_agent_train.py`

- [ ] **Step 1: Write/update the failing tests**

```python
# add to tests/test_main_agent_lessons.py

def test_run_agent_lessons_approve_stores_fingerprints_and_classification():
    import duckdb
    from unittest.mock import patch
    from main import run_agent_lessons_approve
    from src.agent.lessons import create_lessons_tables, insert_lesson_candidate

    conn = duckdb.connect(":memory:")
    create_lessons_tables(conn)
    lesson_id = insert_lesson_candidate(conn, "lesson text", "E0", "competition_specific", "m1")

    with patch("main.DuckDBManager") as MockManager, \
         patch("main._build_llm_invoke", return_value=lambda p: "n/a"), \
         patch("src.agent.lessons.generate_rule_from_lesson", return_value="NEVER do X."), \
         patch("src.agent.lessons.find_conflicting_rule", return_value=None), \
         patch("src.agent.lessons.classify_lesson_sensitivity", return_value=True) as mock_classify, \
         patch("src.agent.lesson_fingerprint.compute_model_fingerprint", return_value="mfp1"), \
         patch("src.agent.agent_config_hash.compute_agent_config_hash", return_value="cfp1"):
        MockManager.return_value.connection.return_value.__enter__.return_value = conn
        run_agent_lessons_approve(lesson_id=lesson_id, scope="competition", reviewer="tester")

    row = conn.execute(
        "SELECT model_fingerprint, agent_config_fingerprint, survives_model_change FROM agent_lessons WHERE id = ?",
        [lesson_id],
    ).fetchone()
    assert row == ("mfp1", "cfp1", True)
    mock_classify.assert_called_once()
```

```python
# add to tests/test_main_agent_train.py (or the real filename covering _write_train_artifacts)

def test_write_train_artifacts_threads_run_id_into_lesson_candidates():
    import duckdb
    from main import _write_train_artifacts

    conn = duckdb.connect(":memory:")
    record = _record(full_state={
        "competition_resolution": {"competition": "E0", "tier": "competition_specific"},
        "research_evidence": {"availability": "ok"},
        "forecast_payload": {"result_3way": {}},
    })

    _write_train_artifacts(conn, [record], run_id="run-xyz")

    stored_run_id = conn.execute("SELECT run_id FROM agent_lessons").fetchone()[0]
    assert stored_run_id == "run-xyz"
```

(Reuses the existing `_record` helper already defined at the top of that test file.)

- [ ] **Step 2: Run to verify they fail**

Run: `python -m pytest tests/test_main_agent_lessons.py tests/test_main_agent_train.py -k "run_id or fingerprints_and_classification" -v`
Expected: FAIL — `run_agent_lessons_approve` doesn't call `classify_lesson_sensitivity`/fingerprint functions yet; `_write_train_artifacts`'s `insert_lesson_candidate` calls don't pass `run_id` yet.

- [ ] **Step 3: Update `_write_train_artifacts`'s two `insert_lesson_candidate` call sites**

In the `batch_size <= 1` branch:

```python
        for record, competition_id, tier in scoped:
            reasoning_trace = serialize_agent_messages(record.full_state.get("messages", []))
            lesson_text = generate_match_reflection(record, reasoning_trace, llm_invoke, record.match_stats)
            insert_lesson_candidate(conn, lesson_text, competition_id, tier, record.match_id, run_id=run_id)
            lessons_written += 1
```

In the batch-size>1 `_flush()` closure:

```python
        match_ids = ",".join(r.match_id for r in current_batch)
        insert_lesson_candidate(conn, lesson_text, competition_id, tier, match_ids, run_id=run_id)
        lessons_written += 1
```

(`run_id` is already the enclosing function's own parameter -- both closures already capture it via the existing Python scoping, no new parameter threading needed.)

- [ ] **Step 4: Update `run_agent_lessons_approve`**

```python
def run_agent_lessons_approve(
    lesson_id: int, scope: str, reviewer: str | None, rule: str | None = None,
    config_path: str | None = None, force: bool = False,
) -> None:
    """... (existing docstring paragraphs unchanged) ...

    A127: after distillation/conflict-checking succeeds, this also runs
    classify_lesson_sensitivity (one more LLM call, reusing the llm_invoke
    already built above for distillation/conflict-checking -- no new
    client construction) and computes this lesson's CURRENT
    model_fingerprint/agent_config_fingerprint, both stored on approval.
    See src/agent/lessons.py::approve_lesson's own docstring for why
    fingerprinting happens here, at approval time, not at insert time."""
    import getpass

    from src.agent.agent_config import AgentConfig
    from src.agent.agent_config_hash import compute_agent_config_hash
    from src.agent.lesson_fingerprint import compute_model_fingerprint
    from src.agent.lessons import (
        approve_lesson, classify_lesson_sensitivity, create_lessons_tables,
        find_conflicting_rule, generate_rule_from_lesson, load_approved_lessons,
    )
    from src.utils.db_manager import DuckDBManager

    db = DuckDBManager()
    with db.connection() as conn:
        create_lessons_tables(conn)
        row = conn.execute(
            "SELECT lesson_text, competition_id, tier FROM agent_lessons WHERE id = ?", [lesson_id]
        ).fetchone()
        if row is None:
            raise ValueError(f"No lesson with id={lesson_id}")
        lesson_text, competition_id, tier = row

        cfg = AgentConfig.from_yaml(config_path) if config_path else AgentConfig.default()
        llm_invoke = _build_llm_invoke(cfg)

        if rule is not None:
            rule_text = rule
        else:
            rule_text = generate_rule_from_lesson(lesson_text, llm_invoke)
            if not rule_text:
                raise ValueError(
                    f"Could not auto-distill a rule for lesson {lesson_id} (LLM call failed or returned "
                    'empty). Re-run with --rule "..." to supply one manually.'
                )
            print(f"Auto-distilled rule: {rule_text}")

        existing_rules = load_approved_lessons(
            conn, competition_id, tier,
            compute_model_fingerprint(competition_id), compute_agent_config_hash(cfg),
        )
        try:
            conflict = find_conflicting_rule(rule_text, existing_rules, llm_invoke)
        except Exception as exc:
            print(f"  warning: conflict check failed ({exc}) -- proceeding without it")
            conflict = None
        if conflict:
            if not force:
                raise ValueError(
                    f"Proposed rule conflicts with an existing approved rule: {conflict} "
                    "Re-run with --force to approve anyway, or reword the rule."
                )
            print(f"  warning: approving despite detected conflict: {conflict}")

        survives_model_change = classify_lesson_sensitivity(lesson_text, competition_id, tier, llm_invoke)
        model_fingerprint = compute_model_fingerprint(competition_id)
        agent_config_fingerprint = compute_agent_config_hash(cfg)

        approve_lesson(
            conn, lesson_id, scope, reviewer or getpass.getuser(), rule_text,
            model_fingerprint=model_fingerprint,
            agent_config_fingerprint=agent_config_fingerprint,
            survives_model_change=survives_model_change,
        )
    print(f"Approved lesson {lesson_id} (scope={scope}, survives_model_change={survives_model_change})")
```

Note: `load_approved_lessons`'s call here now also needs real fingerprints (it's a required-param function as of Task 7) -- passing THIS approval's own about-to-be-stored fingerprints as the "current" ones for the conflict-check lookup is correct: we want to check the new rule against whatever's ALREADY validly approved under today's model/config, which is exactly what those two freshly-computed values represent.

- [ ] **Step 5: Run tests to verify they pass**

Run: `python -m pytest tests/test_main_agent_lessons.py tests/test_main_agent_train.py -v`
Expected: all passed.

- [ ] **Step 6: Commit**

```bash
git add main.py tests/test_main_agent_lessons.py tests/test_main_agent_train.py
git commit -m "feat(agent): wire A127 fingerprinting/classification into agent-train and agent-lessons approve"
```

---

## Task 10: Wire the live/autonomous path (`live_lessons.py` + `scheduler_wiring.py`)

**Files:**
- Modify: `app/backend/live_lessons.py` (`auto_judge_live_lessons`)
- Modify: `app/backend/scheduler_wiring.py` (`register_lessons_job`, one line)
- Test: `app/backend/tests/test_live_lessons.py` (substitute real filename from Task 0)

**`config` is optional (default `None`), not required** -- unlike the design's literal ask, this avoids forcing an update to the ~10 pre-existing `auto_judge_live_lessons(dm, fake_invoke)` call sites already in this test file (none of which test fingerprinting; they test approve/reject/defer/conflict logic). `config=None` degrades exactly like `lessons_node`'s own pattern: skip `agent_config_fingerprint` computation, `approve_lesson` gets called without those kwargs and falls back to its own `None`/`False` defaults. The real production caller (`scheduler_wiring.py`) always passes a real config.

- [ ] **Step 1: Write the new failing test**

```python
# add to app/backend/tests/test_live_lessons.py

def test_auto_judge_live_lessons_stores_fingerprints_and_survives_flag():
    import duckdb
    from unittest.mock import patch
    from app.backend.live_lessons import auto_judge_live_lessons
    from src.agent.agent_config import AgentConfig
    from src.agent.lessons import create_lessons_tables, insert_lesson_candidate
    from src.utils.db_manager import DuckDBManager

    db_manager = DuckDBManager(config_path="config.yaml")
    # Reuse whatever in-memory/tmp-file DuckDBManager construction pattern
    # this test file's OTHER auto_judge_live_lessons tests already use --
    # substitute that exact setup here instead of DuckDBManager(config_path=...)
    # if the real pattern differs (check Task 0's file before writing this).

    with db_manager.connection() as conn:
        create_lessons_tables(conn)
        lesson_id = insert_lesson_candidate(conn, "lesson text", "E0", "competition_specific", "m1", source="live")

    def fake_invoke(prompt):
        return (
            '{"approve": true, "scope": "competition", "reasoning": "clear pattern", '
            '"survives_model_change": true}'
        )

    with patch("app.backend.live_lessons.generate_rule_from_lesson", return_value="NEVER do X."), \
         patch("app.backend.live_lessons.find_conflicting_rule", return_value=None), \
         patch("src.agent.lesson_fingerprint.compute_model_fingerprint", return_value="mfp1"), \
         patch("src.agent.agent_config_hash.compute_agent_config_hash", return_value="cfp1"):
        auto_judge_live_lessons(db_manager, fake_invoke, AgentConfig.default())

    with db_manager.connection(read_only=True) as conn:
        row = conn.execute(
            "SELECT status, model_fingerprint, agent_config_fingerprint, survives_model_change "
            "FROM agent_lessons WHERE id = ?", [lesson_id],
        ).fetchone()
    assert row == ("approved", "mfp1", "cfp1", True)
```

- [ ] **Step 2: Run to verify it fails**

Run: `python -m pytest app/backend/tests/test_live_lessons.py -k stores_fingerprints_and_survives_flag -v`
Expected: FAIL — `auto_judge_live_lessons()` doesn't accept a third `config` positional arg yet.

- [ ] **Step 3: Update `auto_judge_live_lessons`**

```python
def auto_judge_live_lessons(
    duckdb_manager: DuckDBManager,
    llm_invoke: Callable[[str], str] | None,
    config: "AgentConfig | None" = None,
) -> list[dict[str, Any]]:
    """... (existing docstring paragraphs unchanged) ...

    config (A127, optional): needed to compute this call's current
    agent_config_fingerprint, stored on every row this function approves.
    Always passed by scheduler_wiring.py's register_lessons_job in real
    production use. Defaults to None so every pre-existing test call site
    in this file (none of which test fingerprinting) keeps working
    unmodified -- None skips agent_config_fingerprint computation entirely
    and approve_lesson falls back to its own pre-A127 None/False defaults,
    same degrade-gracefully contract lessons_node uses for its own optional
    config param."""
    if llm_invoke is None:
        return []

    from src.agent.agent_config_hash import compute_agent_config_hash
    from src.agent.lesson_fingerprint import compute_model_fingerprint

    with duckdb_manager.connection(read_only=True) as conn:
        pending = list_pending_by_source(conn, source="live")

    groups: dict[tuple[str | None, str], list[dict[str, Any]]] = defaultdict(list)
    for candidate in pending:
        groups[(candidate["competition_id"], candidate["tier"])].append(candidate)

    current_agent_config_fingerprint = compute_agent_config_hash(config) if config is not None else None

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
                    current_model_fingerprint = compute_model_fingerprint(competition_id)
                    with duckdb_manager.connection(read_only=True) as conn:
                        existing_rules = load_approved_lessons(
                            conn, competition_id, tier, current_model_fingerprint, current_agent_config_fingerprint,
                        )
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
                "competition_id": competition_id,
                "survives_model_change": decision.survives_model_change,
            })

    with duckdb_manager.connection() as conn:
        for result in results:
            try:
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
                    approve_lesson(
                        conn, result["id"], result["scope"], reviewer="agent-auto", rule_text=result["rule_text"],
                        model_fingerprint=compute_model_fingerprint(result["competition_id"]),
                        agent_config_fingerprint=current_agent_config_fingerprint,
                        survives_model_change=result["survives_model_change"],
                    )
                elif result["action"] == "reject":
                    reject_lesson(conn, result["id"], reviewer="agent-auto")
                conn.execute(
                    "UPDATE agent_lessons SET auto_decision_reasoning = ? WHERE id = ?",
                    [result["reasoning"], result["id"]],
                )
            except Exception:
                LOGGER.warning(
                    "live_lessons: failed to write auto-judge decision for lesson id=%s.",
                    result["id"], exc_info=True,
                )
    return results
```

(Only the additions shown above change; the rest of the function -- the final `except Exception:` block's exact original tail -- stays as-is. Read the real current tail before pasting, since this excerpt reconstructs it from the earlier `Read` in this plan's research and may not be byte-for-byte identical past the point already changed.)

- [ ] **Step 4: Update the one call site in `scheduler_wiring.py`**

```python
        judged = auto_judge_live_lessons(duckdb_manager, llm_invoke, config)
```

(`config` is already `register_lessons_job`'s own parameter, already in scope inside `_weekly_review_job`'s closure -- no new plumbing.)

- [ ] **Step 5: Run tests to verify they pass**

Run: `python -m pytest app/backend/tests/test_live_lessons.py app/backend/tests/test_scheduler_integration.py -v`
Expected: all passed, including every pre-existing `auto_judge_live_lessons` test updated to pass a third `config` argument (add `AgentConfig.default()` to each existing call site found in this test file that doesn't already appear in Step 1's new test).

- [ ] **Step 6: Commit**

```bash
git add app/backend/live_lessons.py app/backend/scheduler_wiring.py app/backend/tests/test_live_lessons.py
git commit -m "feat(agent): wire A127 fingerprinting into the live auto-judge path"
```

---

## Task 11: Surface `needs_review` in the admin endpoint

**Files:**
- Modify: `app/backend/main.py` (`list_lessons`)
- Test: `app/backend/tests/test_list_lessons_endpoint.py` (real filename confirmed in Task 0)

- [ ] **Step 1: Write the failing test**

Reuses this file's own existing `_db_manager_for`/`_seed_lesson` helpers and `patch("app.backend.main.DuckDBManager", return_value=manager)` + `TestClient(app)` convention (see its other tests) rather than a generic `client` fixture:

```python
# add to app/backend/tests/test_list_lessons_endpoint.py

def test_list_lessons_filters_by_needs_review_status_and_returns_run_id(tmp_path):
    manager = _db_manager_for(tmp_path)
    lesson_id = _seed_lesson(manager, source="train", status="needs_review")
    with manager.connection() as conn:
        conn.execute(
            "UPDATE agent_lessons SET run_id = ?, auto_decision_reasoning = ? WHERE id = ?",
            ["run-1", "model changed", lesson_id],
        )

    with patch("app.backend.main.DuckDBManager", return_value=manager):
        with TestClient(app) as client:
            response = client.get("/api/admin/lessons", params={"status": "needs_review"})

    assert response.status_code == 200
    lessons = response.json()["lessons"]
    assert any(
        l["id"] == lesson_id and l["run_id"] == "run-1" and l["auto_decision_reasoning"] == "model changed"
        for l in lessons
    )
```

(Use whatever `client`/DB-setup fixture this test file's OTHER endpoint tests already use -- check the real file from Task 0 before writing this; the `DuckDBManager().connection()` shown here is illustrative of the shape, not necessarily the exact fixture pattern already established.)

- [ ] **Step 2: Run to verify it fails**

Run: `python -m pytest app/backend/tests/test_list_lessons_endpoint.py -k needs_review -v`
Expected: FAIL — `422 Unprocessable Entity` (status Literal rejects `"needs_review"`), or a `KeyError`/missing `run_id` in the response.

- [ ] **Step 3: Update the endpoint**

```python
@app.get("/api/admin/lessons")
def list_lessons(
    status: Literal["pending", "approved", "rejected", "needs_review"] | None = None,
    source: Literal["train", "live"] | None = None,
    limit: int = Query(50, ge=1, le=500),
) -> dict:
    """... (existing docstring, extend the closing sentence:) ... status/source
    filter with plain SQL equality; omitting either returns every value
    including legacy NULL source rows. 'needs_review' (A127) is a real,
    filterable status alongside the original three -- a lesson the A127
    staleness check disqualified at read time, awaiting re-review."""
    from src.agent.lessons import create_lessons_tables

    db = DuckDBManager()
    with db.connection() as conn:
        create_lessons_tables(conn)
        clauses, params = [], []
        if status is not None:
            clauses.append("status = ?")
            params.append(status)
        if source is not None:
            clauses.append("source = ?")
            params.append(source)
        where = f"WHERE {' AND '.join(clauses)}" if clauses else ""
        params.append(limit)
        rows = conn.execute(
            f"""
            SELECT id, lesson_text, rule_text, status, competition_id, tier, scope,
                   source_match_id, source, created_at, reviewed_at, reviewer,
                   auto_decision_reasoning, run_id
            FROM agent_lessons {where} ORDER BY created_at DESC LIMIT ?
            """,
            params,
        ).fetchall()
    columns = (
        "id", "lesson_text", "rule_text", "status", "competition_id", "tier", "scope",
        "source_match_id", "source", "created_at", "reviewed_at", "reviewer",
        "auto_decision_reasoning", "run_id",
    )
    return {"lessons": [dict(zip(columns, row)) for row in rows]}
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest app/backend/tests/test_list_lessons_endpoint.py -v`
Expected: all passed.

- [ ] **Step 5: Commit**

```bash
git add app/backend/main.py app/backend/tests/test_list_lessons_endpoint.py
git commit -m "feat(app): surface needs_review status and run_id on GET /api/admin/lessons (A127)"
```

---

## Task 12: Full-suite verification

**Files:** none modified.

- [ ] **Step 1: Run the full root test suite**

Run: `python -m pytest tests/ -q`
Expected: same pass count as the pre-A127 baseline plus every new test added across Tasks 1-11, zero unexplained failures. (Baseline going into this plan: 1220 passed / 1 skipped.)

- [ ] **Step 2: Run the full app/backend test suite**

Run: `python -m pytest app/backend/tests/ -q`
Expected: the 8 pre-existing failures already confirmed unrelated to this codebase area (fixtures/dashboard/recommendation-outcomes endpoints -- confirmed via `git stash` during A18's work earlier this session) are still the ONLY failures. Any NEW failure here is a real regression from this plan and must be fixed before considering A127 done.

- [ ] **Step 3: Mark A127 completed in `documents/agent_user_stories.md`**

Find the `A127` row and change its status from whatever it currently is to `completed`, appending a `**Completion notes (<today's date>):**` paragraph summarizing what was built, explicitly naming the two documented deviations from the design doc (fingerprint timing, train-path classification) and the exact test/pass counts from Steps 1-2 -- following this file's own established completion-note convention (see A18's or A68's entries for the expected level of detail).

- [ ] **Step 4: Final commit**

```bash
git add documents/agent_user_stories.md
git commit -m "docs: mark A127 completed -- lesson staleness/fingerprint versioning shipped"
```
