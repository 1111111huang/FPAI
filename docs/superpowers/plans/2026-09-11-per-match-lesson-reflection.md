# Per-Match Lesson Reflection Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the deterministic single-match lesson template with an LLM reflection on that match's own recorded reasoning trace and outcome, in both `agent-train`'s default (`--batch-size 1`) path and live's daily per-day lesson groups — falling back to the existing template whenever there's no trace or no LLM to reflect with.

**Architecture:** One new function, `generate_match_reflection()` (`src/agent/lessons.py`), duck-typed like its siblings, called from two existing sites (`main.py:_write_train_artifacts`, `app/backend/live_lessons.py:prepare_lesson_batches`) in place of `generate_lesson_text()`/`generate_batch_lesson_text()`+`generate_batch_reflection()` respectively. A new `BacktestRecord.match_stats` field (train only) threads already-ingested box-score columns (shots, cards) through as extra grounding — zero new API calls, zero new infra.

**Tech Stack:** Python, DuckDB, pandas (`raw_matches` row access), existing `llm_invoke: Callable[[str], str]` decoupling pattern (no langchain import in `lessons.py`).

**Spec:** `docs/superpowers/specs/2026-09-11-per-match-lesson-reflection-design.md`

---

### Task 1: `load_match_stats()` + `BacktestRecord.match_stats`

**Files:**
- Modify: `src/agent/backtest.py:79-104` (dataclass + `load_outcome`), `:261-271` (`process_match_row`'s return)
- Test: `tests/test_backtest.py`

- [ ] **Step 1: Write the failing tests**

Add to `tests/test_backtest.py` (near the existing `load_outcome` tests, after line 70):

```python
def test_load_match_stats_includes_shots_and_cards_when_present():
    row = _row(hs=14.0, **{"as": 8.0}, hst=6.0, ast=3.0, hy=2.0, ay=1.0, hr=0.0, ar=1.0)
    stats = load_match_stats(row)
    assert stats == {
        "home_shots": 14, "away_shots": 8,
        "home_shots_on_target": 6, "away_shots_on_target": 3,
        "home_yellow_cards": 2, "away_yellow_cards": 1,
        "home_red_cards": 0, "away_red_cards": 1,
    }


def test_load_match_stats_omits_pairs_with_nan_and_returns_none_if_all_absent():
    row = _row(hs=float("nan"), **{"as": float("nan")})  # no hst/ast/hy/ay/hr/ar at all
    assert load_match_stats(row) is None


def test_load_match_stats_includes_only_whichever_pairs_are_present():
    row = _row(hs=10.0, **{"as": 5.0})  # shots present, cards columns absent entirely
    stats = load_match_stats(row)
    assert stats == {"home_shots": 10, "away_shots": 5}
```

Add `load_match_stats` to the existing import block at the top of `tests/test_backtest.py`:

```python
from src.agent.backtest import (
    LEAKAGE_GUARD_INSTRUCTIONS,
    BacktestHarness,
    BacktestRecord,
    _build_match_info,
    load_match_stats,
    load_outcome,
    match_in_test_split,
    process_match_row,
)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_backtest.py -k load_match_stats -v`
Expected: FAIL with `ImportError: cannot import name 'load_match_stats'`

- [ ] **Step 3: Implement `load_match_stats` and the `match_stats` field**

In `src/agent/backtest.py`, add after `load_outcome` (after line 104):

```python
# A109: shots/shots-on-target/cards columns raw_matches already carries for
# most sources (not Sweden -- see src/ingestion/football_data/sweden_loader.py),
# standard football-data.co.uk schema. Used only as extra grounding for
# generate_match_reflection()'s reflective lesson -- never for market
# resolution (that's load_outcome/build_actual_outcome's job; shots and
# cards aren't a tradeable market here), so kept as a wholly separate dict
# rather than threaded into build_actual_outcome.
_MATCH_STAT_PAIRS = {
    "shots": ("hs", "as"),
    "shots_on_target": ("hst", "ast"),
    "yellow_cards": ("hy", "ay"),
    "red_cards": ("hr", "ar"),
}


def load_match_stats(row: pd.Series) -> dict[str, Any] | None:
    """Box-score stats for a historical match, whichever of shots/shots-on-
    target/cards are actually present on this row -- same pd.notna() pair-
    presence check load_outcome already uses for hc/ac. None if the source
    has none of these columns at all (e.g. Sweden)."""
    stats: dict[str, Any] = {}
    for stat, (home_col, away_col) in _MATCH_STAT_PAIRS.items():
        home_val, away_val = row.get(home_col), row.get(away_col)
        if pd.notna(home_val) and pd.notna(away_val):
            stats[f"home_{stat}"] = int(home_val)
            stats[f"away_{stat}"] = int(away_val)
    return stats or None
```

Add the field to `BacktestRecord` (line 89, right after `full_state`):

```python
    full_state: dict[str, Any] | None = None
    match_stats: dict[str, Any] | None = None
```

Wire it into `process_match_row`'s return (around line 268, alongside the existing `actual = load_outcome(row)` line):

```python
    actual = load_outcome(row)
    match_stats = load_match_stats(row)
```

and add `match_stats=match_stats,` to the `BacktestRecord(...)` construction a few lines below it (alongside `full_state=full_state,`).

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_backtest.py -v`
Expected: all pass, including the 3 new ones. (Confirms the existing `process_match_row`/`load_outcome` tests are unaffected by the new field.)

- [ ] **Step 5: Commit**

```bash
git add src/agent/backtest.py tests/test_backtest.py
git commit -m "$(cat <<'EOF'
feat(agent): A109 -- load_match_stats, BacktestRecord.match_stats

Box-score stats (shots, shots-on-target, cards) already sit in
raw_matches unused. New load_match_stats(row) reads them with the
same pd.notna() pair-presence pattern load_outcome already uses for
hc/ac; None when a source has none of these columns (Sweden). Kept
separate from build_actual_outcome's resolvable-market-outcome dict
on purpose -- shots/cards aren't a tradeable market here, this is
grounding for the next task's generate_match_reflection() only.

Co-Authored-By: Claude Sonnet 5 <noreply@anthropic.com>
EOF
)"
```

---

### Task 2: `generate_match_reflection()` in `src/agent/lessons.py`

**Files:**
- Modify: `src/agent/lessons.py:424-448` (extract shared helper, add new function)
- Test: `tests/test_agent_lessons.py`

- [ ] **Step 1: Write the failing tests**

Add to `tests/test_agent_lessons.py`, right after `test_generate_lesson_text_handles_no_markets_and_limitations` (after line 372) — reuses the existing `_FakeRecord` fixture defined at line 328:

```python
def test_generate_match_reflection_falls_back_to_template_when_trace_missing():
    record = _FakeRecord(
        league="E0",
        recommendation={"overall": "direct_bet", "confidence": "high", "prediction_basis": "x", "limitations": []},
        market_results=[{"market": "result_3way", "selection": "home", "correct": True}],
        actual={"result": "home"},
    )
    text = generate_match_reflection(record, reasoning_trace=None, llm_invoke=lambda p: "unused")
    assert text == generate_lesson_text(record)


def test_generate_match_reflection_falls_back_to_template_when_llm_invoke_missing():
    record = _FakeRecord(
        league="E0", recommendation={"overall": "direct_bet", "confidence": "high", "prediction_basis": "x", "limitations": []},
        market_results=[], actual={"result": "home"},
    )
    trace = [{"role": "ai", "content": "Home side has won 4 of last 5."}]
    text = generate_match_reflection(record, reasoning_trace=trace, llm_invoke=None)
    assert text == generate_lesson_text(record)


def test_generate_match_reflection_falls_back_to_template_when_llm_invoke_raises():
    record = _FakeRecord(
        league="E0", recommendation={"overall": "direct_bet", "confidence": "high", "prediction_basis": "x", "limitations": []},
        market_results=[], actual={"result": "away"},
    )
    trace = [{"role": "ai", "content": "Home side has won 4 of last 5."}]

    def _raise(prompt: str) -> str:
        raise RuntimeError("provider down")

    text = generate_match_reflection(record, reasoning_trace=trace, llm_invoke=_raise)
    assert text == generate_lesson_text(record)


def test_generate_match_reflection_returns_the_llm_narrative_on_the_happy_path():
    record = _FakeRecord(
        league="E0", recommendation={"overall": "direct_bet", "confidence": "high", "prediction_basis": "x", "limitations": []},
        market_results=[{"market": "result_3way", "selection": "home", "correct": False}],
        actual={"result": "away"},
    )
    trace = [{"role": "ai", "content": "Searched recent form, saw 4 home wins, picked home."}]
    seen_prompts = []

    def _invoke(prompt: str) -> str:
        seen_prompts.append(prompt)
        return "  The agent over-weighted stale form data and missed the away side's injury return.  "

    text = generate_match_reflection(record, reasoning_trace=trace, llm_invoke=_invoke)

    assert text == "The agent over-weighted stale form data and missed the away side's injury return."
    assert "Searched recent form" in seen_prompts[0]
    assert "result_3way=home (incorrect)" in seen_prompts[0]
    assert "Post-match stats" not in seen_prompts[0]


def test_generate_match_reflection_includes_match_stats_in_the_prompt_when_given():
    record = _FakeRecord(
        league="E0", recommendation={"overall": "no_bet", "confidence": "low", "prediction_basis": "x", "limitations": []},
        market_results=[], actual={"result": "draw"},
    )
    trace = [{"role": "ai", "content": "No clear edge found."}]
    seen_prompts = []

    def _invoke(prompt: str) -> str:
        seen_prompts.append(prompt)
        return "ok"

    generate_match_reflection(
        record, reasoning_trace=trace, llm_invoke=_invoke,
        match_stats={"home_shots": 14, "away_shots": 8},
    )

    assert "home_shots=14" in seen_prompts[0]
    assert "away_shots=8" in seen_prompts[0]
```

In `tests/test_agent_lessons.py`'s existing `from src.agent.lessons import (...)` block, change:

```python
    generate_lesson_text,
    generate_rule_from_lesson,
```

to:

```python
    generate_lesson_text,
    generate_match_reflection,
    generate_rule_from_lesson,
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_agent_lessons.py -k generate_match_reflection -v`
Expected: FAIL with `ImportError: cannot import name 'generate_match_reflection'`

- [ ] **Step 3: Refactor `generate_lesson_text` and add `generate_match_reflection`**

Replace `src/agent/lessons.py:424-448` (the whole current `generate_lesson_text` function) with:

```python
def _market_and_limitations_summary(record: Any) -> tuple[str, str]:
    """Shared grounding-fact formatting -- generate_lesson_text's template
    and generate_match_reflection's prompt both need the same 'market=
    selection (outcome)' / limitations-joined summary, computed once rather
    than duplicated."""
    market_lines = []
    for market in record.market_results:
        correct = market.get("correct")
        outcome = "correct" if correct is True else "incorrect" if correct is False else "unresolved"
        market_lines.append(f"{market.get('market')}={market.get('selection')} ({outcome})")
    markets_summary = "; ".join(market_lines) if market_lines else "no markets recommended"
    limitations = record.recommendation.get("limitations") or []
    limitations_summary = "; ".join(limitations) if limitations else "none noted"
    return markets_summary, limitations_summary


def generate_lesson_text(record: Any) -> str:
    """Deterministic lesson-candidate template from a BacktestRecord-shaped
    object (duck-typed: .league, .recommendation, .market_results, .actual --
    see src/agent/backtest.py). Not an attempt at insightful NLG -- the
    reviewer judges usefulness at approval time; this just surfaces a
    structured summary of what happened for them to judge. Also
    generate_match_reflection's fallback whenever there's no reasoning_trace
    or no LLM to reflect with."""
    context_label = record.league or "an unlabeled competition"
    overall = record.recommendation.get("overall", "unknown")
    confidence = record.recommendation.get("confidence", "unknown")
    basis = record.recommendation.get("prediction_basis", "unknown")
    markets_summary, limitations_summary = _market_and_limitations_summary(record)

    return (
        f"WHEN evaluating {context_label} matches: a recommendation of '{overall}' "
        f"(confidence={confidence}, basis={basis}) had actual result={record.actual.get('result')}. "
        f"Markets: {markets_summary}. Limitations noted at the time: {limitations_summary}."
    )


def generate_match_reflection(
    record: Any,
    reasoning_trace: list[dict[str, Any]] | None,
    llm_invoke: Callable[[str], str] | None,
    match_stats: dict[str, Any] | None = None,
) -> str:
    """A109: reflects an LLM on one match's own recorded reasoning trace
    (A106's reasoning_trace -- the agent's actual tool-call/investigation
    trail, not just its final explanation) plus its outcome, replacing
    generate_lesson_text's deterministic template -- the reviewer gets the
    agent's own investigation and judgment, not a fill-in-the-blank
    summary. match_stats (A109, train-only -- see src/agent/backtest.py's
    load_match_stats) is extra grounding when available, never required.

    Falls back to generate_lesson_text(record) whenever there's nothing to
    reflect on (no trace) or no LLM to do it with, and again if the LLM
    call itself raises -- same 'never lose the lesson' contract
    generate_batch_reflection already uses for its own LLM call, just
    covering the input-missing case too, not only a provider failure."""
    if not reasoning_trace or llm_invoke is None:
        return generate_lesson_text(record)

    markets_summary, _ = _market_and_limitations_summary(record)
    trace_text = "\n".join(f"[{m.get('role')}] {m.get('content')}" for m in reasoning_trace)
    stats_line = (
        f" Post-match stats: {', '.join(f'{k}={v}' for k, v in match_stats.items())}."
        if match_stats else ""
    )

    prompt = (
        f"You are reviewing a betting recommendation an automated agent made for a "
        f"{record.league or 'an unlabeled competition'} match, now that the actual result is known.\n\n"
        f"Recommendation: '{record.recommendation.get('overall', 'unknown')}' "
        f"(confidence={record.recommendation.get('confidence', 'unknown')}). "
        f"Markets: {markets_summary}. Actual result: {record.actual.get('result')}.{stats_line}\n\n"
        f"The agent's own reasoning and tool calls at the time:\n{trace_text}\n\n"
        "Write a short reflective lesson (3-5 sentences) covering: (a) whether the agent's own "
        "investigation actually surfaced the information that would have led to the right call, "
        "(b) if it missed, whether that's a reasoning gap or an evidence gap, (c) one concrete, "
        "actionable adjustment for future recommendations in this competition. Do not invent facts "
        "not present above, and do not use generic hedging language like 'more data would help'."
    )
    try:
        reflection = llm_invoke(prompt)
    except Exception:
        return generate_lesson_text(record)
    return reflection.strip() or generate_lesson_text(record)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_agent_lessons.py -v`
Expected: all pass (confirms the refactor didn't change `generate_lesson_text`'s existing behavior, and the 5 new tests pass).

- [ ] **Step 5: Commit**

```bash
git add src/agent/lessons.py tests/test_agent_lessons.py
git commit -m "$(cat <<'EOF'
feat(agent): A109 -- generate_match_reflection

New LLM reflection over a match's own reasoning_trace (A106) + outcome
(+ optional match_stats grounding), replacing generate_lesson_text's
deterministic template at the two call sites the next two tasks wire
up. Falls back to the unchanged template when there's no trace, no
llm_invoke, or the LLM call raises -- same best-effort contract
generate_batch_reflection already uses. generate_lesson_text itself is
a pure refactor here (shared _market_and_limitations_summary
extracted), no behavior change -- its own existing tests pass
unmodified.

Co-Authored-By: Claude Sonnet 5 <noreply@anthropic.com>
EOF
)"
```

---

### Task 3: Wire into `agent-train`'s `--batch-size 1` path

**Files:**
- Modify: `main.py:1602-1636` (`_write_train_artifacts`)
- Test: `tests/test_main_agent_train.py`

- [ ] **Step 1: Write the failing test**

Add to `tests/test_main_agent_train.py`, after `test_write_train_artifacts_persists_reasoning_trace_from_full_state_messages` (after line 92):

```python
def test_write_train_artifacts_uses_llm_reflection_for_batch_size_1_when_config_given():
    """A109: batch_size<=1 now reflects via generate_match_reflection when a
    config (and therefore an llm_invoke) is given -- previously config was
    accepted but silently unused on this path."""
    from src.agent.agent_config import AgentConfig

    conn = duckdb.connect(":memory:")
    record = _record(full_state={
        "competition_resolution": {"competition": "E0", "tier": "competition_specific"},
        "research_evidence": {"availability": "ok"},
        "forecast_payload": {"result_3way": {}},
        "messages": [AIMessage(content="Picked no_bet, no clear edge.")],
    })
    config = AgentConfig(
        model="stub-model", provider="ollama", temperature=0.0, max_tool_calls=5,
        min_odds_threshold=1.2, max_odds_threshold=11.0, min_conditional_odds_threshold=1.5,
        min_value_edge=0.05, markets=["result_3way"], system_prompt_version="v1",
    )

    with patch("main._build_llm_invoke", return_value=lambda p: "The agent correctly found no edge and stood aside."):
        _write_train_artifacts(conn, [record], run_id="run-7", batch_size=1, config=config)

    lesson_text = conn.execute("SELECT lesson_text FROM agent_lessons").fetchone()[0]
    assert lesson_text == "The agent correctly found no edge and stood aside."


def test_write_train_artifacts_batch_size_1_falls_back_without_config():
    """Unchanged-behavior guard: every pre-existing caller of this path
    passes no config (the default), so it must keep producing exactly
    generate_lesson_text's template, not attempt any LLM call."""
    from src.agent.lessons import generate_lesson_text

    conn = duckdb.connect(":memory:")
    record = _record(full_state={
        "competition_resolution": {"competition": "E0", "tier": "competition_specific"},
        "research_evidence": {"availability": "ok"},
        "forecast_payload": {"result_3way": {}},
        "messages": [AIMessage(content="Picked no_bet, no clear edge.")],
    })

    _write_train_artifacts(conn, [record], run_id="run-8", batch_size=1)

    lesson_text = conn.execute("SELECT lesson_text FROM agent_lessons").fetchone()[0]
    assert lesson_text == generate_lesson_text(record)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_main_agent_train.py -k "batch_size_1" -v`
Expected: `test_write_train_artifacts_uses_llm_reflection_for_batch_size_1_when_config_given` FAILs (lesson_text is the old template, not the patched LLM's string). `test_write_train_artifacts_batch_size_1_falls_back_without_config` passes already (no behavior change needed for it) — confirms it's a true regression guard, not a new-behavior test.

- [ ] **Step 3: Wire `generate_match_reflection` into `_write_train_artifacts`**

In `main.py`, change the import block and the `batch_size <= 1` branch (lines 1602-1616):

```python
    from src.agent.lessons import (
        create_lessons_tables,
        generate_batch_lesson_text,
        generate_batch_reflection,
        generate_match_reflection,
        insert_lesson_candidate,
    )

    create_lessons_tables(conn)
    scoped = _write_telemetry_rows(conn, records, run_id)
    telemetry_written = len(scoped)

    if batch_size <= 1:
        from src.agent.graph import serialize_agent_messages

        llm_invoke = _build_llm_invoke(config) if config is not None else None
        lessons_written = 0
        for record, competition_id, tier in scoped:
            reasoning_trace = serialize_agent_messages(record.full_state.get("messages", []))
            lesson_text = generate_match_reflection(record, reasoning_trace, llm_invoke, record.match_stats)
            insert_lesson_candidate(conn, lesson_text, competition_id, tier, record.match_id)
            lessons_written += 1
        return lessons_written, telemetry_written
```

(This removes `generate_lesson_text` from the import list — it's no longer called directly here, only internally by `generate_match_reflection`'s own fallback inside `lessons.py`.)

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_main_agent_train.py -v`
Expected: all pass, including the two new ones.

- [ ] **Step 5: Commit**

```bash
git add main.py tests/test_main_agent_train.py
git commit -m "$(cat <<'EOF'
feat(app): A109 -- agent-train batch-size-1 uses LLM reflection

_write_train_artifacts' default (--batch-size 1) path now builds an
llm_invoke from the run's own config (same _build_llm_invoke already
used by the batch_size>1 branch) and calls generate_match_reflection
per record instead of generate_lesson_text. config=None (every
pre-existing caller) keeps producing exactly the old template --
confirmed by a new regression-guard test.

Co-Authored-By: Claude Sonnet 5 <noreply@anthropic.com>
EOF
)"
```

---

### Task 4: Wire into live's daily per-day lesson groups

**Files:**
- Modify: `app/backend/live_lessons.py:29-40` (imports), `:140-168` (`prepare_lesson_batches`)
- Test: `app/backend/tests/test_live_lessons.py`

- [ ] **Step 1: Update the import block**

In `app/backend/live_lessons.py`, change:

```python
from src.agent.lessons import (
    approve_lesson,
    find_conflicting_rule,
    generate_batch_lesson_text,
    generate_batch_reflection,
    generate_rule_from_lesson,
    insert_lesson_candidate,
    judge_lesson_candidate,
    list_pending_by_source,
    load_approved_lessons,
    reject_lesson,
)
```

to:

```python
from src.agent.lessons import (
    approve_lesson,
    find_conflicting_rule,
    generate_match_reflection,
    generate_rule_from_lesson,
    insert_lesson_candidate,
    judge_lesson_candidate,
    list_pending_by_source,
    load_approved_lessons,
    reject_lesson,
)
```

- [ ] **Step 2: Write the failing test**

Update `test_generate_daily_lessons_appends_reflection_when_llm_invoke_given` (`app/backend/tests/test_live_lessons.py:212-228`) — replace it entirely (this test's old assertion relied on `generate_batch_reflection`'s "Reflection: " prefix, which no longer exists):

```python
def test_generate_daily_lessons_reflects_per_match_when_llm_invoke_and_trace_are_available(tmp_path: Path) -> None:
    """A109: per-match reflection needs a recorded reasoning_trace (from the
    cache entry, same shape train uses) to actually invoke the LLM -- a
    cache hit with no reasoning_trace still falls back to the template
    (covered by the unmodified test right above this one)."""
    cache = RecommendationCache(db_path=tmp_path / "cache.db")
    cache.record_generation(
        "m1", "2026-08-22", "hash1", {}, _rec(), "scheduled",
        reasoning_trace=[{"role": "ai", "content": "Home side has won 4 of last 5, picked home."}],
    )
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
    assert lesson_text == "Live-sourced batch: reflects only the market actually recommended per match, not every market the agent evaluated.\n\na real reflection"
```

Also add `generate_match_reflection` to this test file's own direct-import assertions if any exist (check: none do — `generate_match_reflection` is only exercised indirectly through `generate_daily_lessons` here, not imported directly in the test file).

- [ ] **Step 3: Run the test to verify it fails**

Run: `python -m pytest app/backend/tests/test_live_lessons.py -k reflects_per_match -v`
Expected: FAIL — current code still runs `generate_batch_lesson_text`/`generate_batch_reflection`, producing a stats-aggregate `lesson_text` with a `"\n\nReflection: a real reflection"` suffix, not the exact joined string asserted above.

- [ ] **Step 4: Rewrite `prepare_lesson_batches`'s per-group body**

In `app/backend/live_lessons.py`, replace lines 154-160 (inside the `for (competition_id, _date), group in groups.items():` loop, right after the `tier = ...` / `except` block):

```python
        records = [_to_lesson_record(outcome, cache) for outcome in group]
        reflections = []
        for outcome, record in zip(group, records):
            entry = cache.get_latest_any_config(outcome.match_id, outcome.date)
            reasoning_trace = entry.reasoning_trace if entry is not None else None
            reflections.append(generate_match_reflection(record, reasoning_trace, llm_invoke))
        lesson_text = f"{LIVE_SOURCE_NOTE}\n\n" + "\n\n".join(reflections)
```

(This removes the old `stats_text = generate_batch_lesson_text(records)` / `if llm_invoke is not None: reflection = generate_batch_reflection(...)` block entirely — `match_stats` is omitted here, always `None` for live, per the design.)

Also update `prepare_lesson_batches`'s own docstring (the `llm_invoke=None skips generate_batch_reflection entirely` paragraph, lines 123-125) to:

```python
    llm_invoke=None makes every per-match reflection fall back to the
    deterministic template (generate_match_reflection's own contract) --
    used by callers that can't or don't want to pay for the LLM call (e.g.
    a fast unit test), not a distinct product mode."""
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `python -m pytest app/backend/tests/test_live_lessons.py -v`
Expected: all pass. `test_generate_daily_lessons_prepends_the_live_source_note_and_skips_reflection_without_an_llm` should pass unmodified (llm_invoke=None still triggers the template fallback regardless of trace, and the joined text still starts with `LIVE_SOURCE_NOTE` and contains no "Reflection:" text).

- [ ] **Step 6: Run the full backend test suite to check for regressions**

Run: `python -m pytest app/backend/tests/ -v`
Expected: same pass/fail counts as before this task started, aside from the tests touched above (confirm no other module imported `generate_batch_lesson_text`/`generate_batch_reflection` from `live_lessons.py` specifically — they remain exported from `src/agent/lessons.py` itself, unaffected, still used by `main.py`'s `batch_size > 1` path).

- [ ] **Step 7: Commit**

```bash
git add app/backend/live_lessons.py app/backend/tests/test_live_lessons.py
git commit -m "$(cat <<'EOF'
feat(app): A109 -- live daily lessons use per-match LLM reflection

prepare_lesson_batches() now calls generate_match_reflection() once
per match inside each (competition_id, date) group (threading that
match's own cache-entry reasoning_trace through), joining the results
into the group's lesson_text -- replacing generate_batch_lesson_text's
stats aggregate + generate_batch_reflection's appended narrative at
this call site. Both functions remain used by agent-train's
--batch-size N>1 path, untouched. match_stats is always None here --
live's outcome resolution has no box-score source, per the design
spec's explicit rejection of adding one.

Co-Authored-By: Claude Sonnet 5 <noreply@anthropic.com>
EOF
)"
```

---

### Task 5: Full suite, user story, final docs

**Files:**
- Modify: `documents/agent_user_stories.md` (append new phase + row)

- [ ] **Step 1: Run the full test suite**

Run: `python -m pytest tests/ app/backend/tests/ -q`
Expected: same failure set as the pre-existing, documented, unrelated `test_fixtures_endpoint.py` failures (see A106/A107/A108's completion notes for the baseline count) — zero new failures.

- [ ] **Step 2: Append the user story**

`documents/agent_user_stories.md` stopped growing a parallel `agent_techspec.md` section per story around A93 — every recent story (A100-A108) documents itself fully in its own completion notes here, so this follows that same established precedent; no `agent_techspec.md` section is added.

Append a new phase section at the end of the file (after A108's row / the existing table). Before pasting this in, replace the final sentence's test-count clause with the exact output Step 1 actually printed (e.g. "Full suite: 1029 passed / 1 skipped (tests/), 499 passed / 5 failed (app/backend/tests/, same pre-existing unrelated test_fixtures_endpoint.py failures)") — copy the real numbers from your own terminal, don't leave this file's wording as-is:

```markdown
## PHASE 37: Per-Match Lesson Reflection (A109)

Direct user request, following a review of how lessons are currently generated: single-match lessons (`agent-train --batch-size 1`'s default path, and live's per-day groups) came from a deterministic template or stats aggregate, never from the agent's own recorded reasoning. Full design discussion and two rejected alternatives (a post-match Tavily search with SnapshotStore caching; snapshotting the box-score stats themselves) in `docs/superpowers/specs/2026-09-11-per-match-lesson-reflection-design.md`.

| ID | Status | Description | Comments |
|---|---|---|---|
| A109 | completed | **Generate single-match lessons by having an LLM reflect on that match's own reasoning_trace (A106) and outcome, replacing the deterministic fill-in-the-blank template, in both agent-train's --batch-size 1 default and live's daily per-day lesson groups.** | Size M · Depends on: A106 (reasoning_trace capture), A108 (live reasoning_trace capture). **Completion notes (2026-09-11):** New `generate_match_reflection()` (`src/agent/lessons.py`) reflects on a match's reasoning_trace + outcome (+ optional box-score `match_stats`, train-only); falls back to the unchanged `generate_lesson_text()` template whenever trace/llm_invoke is missing or the LLM call itself raises. Two alternatives considered and explicitly rejected during design, not pursued: a post-match Tavily web search (would have needed a new SnapshotStore tool name + record/replay wiring to avoid re-paying the cost on every agent-train rerun) and snapshotting the box-score stats themselves (unnecessary -- they're a plain, already-deterministic local `raw_matches` read, nothing to cache). New `load_match_stats()`/`BacktestRecord.match_stats` (`src/agent/backtest.py`) threads already-ingested `hs`/`as`/`hst`/`ast`/`hy`/`ay`/`hr`/`ar` columns through as extra reflection grounding -- zero new API dependency. `main.py`'s `_write_train_artifacts` (`batch_size <= 1` branch) now builds an `llm_invoke` from the run's config (previously only the `batch_size > 1` branch did) and calls `generate_match_reflection` per record; `config=None` (every pre-existing caller) is unaffected, confirmed by a regression-guard test. `app/backend/live_lessons.py`'s `prepare_lesson_batches()` now calls `generate_match_reflection` once per match inside each `(competition_id, date)` group (threading that match's own cache-entry `reasoning_trace`), joining the results into the group's `lesson_text` -- replaces `generate_batch_lesson_text`/`generate_batch_reflection` at this call site only; both remain in use, unchanged, by `agent-train --batch-size N>1`. New tests: `tests/test_backtest.py` (+3, `load_match_stats`), `tests/test_agent_lessons.py` (+5, `generate_match_reflection`: fallback on missing trace/llm/LLM-exception, happy path, match_stats-in-prompt), `tests/test_main_agent_train.py` (+2, reflection-used-when-config-given, unchanged-fallback-without-config), `app/backend/tests/test_live_lessons.py` (1 rewritten: the old "Reflection: " appended-narrative assertion no longer applies since the reflection *is* the lesson text now, not an addendum). Full suite: `<fill in actual pass/fail counts from Step 1>`. |
```

- [ ] **Step 3: Commit**

```bash
git add documents/agent_user_stories.md
git commit -m "$(cat <<'EOF'
docs(agent): A109 -- per-match lesson reflection user story

Co-Authored-By: Claude Sonnet 5 <noreply@anthropic.com>
EOF
)"
```
