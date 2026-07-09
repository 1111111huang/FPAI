# A23–A26: Full-Season Backtest Expansion Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement stories A23–A26 from `documents/agent_user_stories.md` (Phase 8): fix snapshot-recording determinism, refresh raw season data through end-of-season, collect a full 2025/26 E0 snapshot corpus (~380 matches), and run full-season flat/kelly backtests — expanding the 24-match A20/A21 pilot into a statistically meaningful baseline.

**Architecture:** A23 is a small config + plumbing fix (temperature `0.1` → `0.0`) validated by unit tests plus a bounded operational re-run of the existing 24-match pilot corpus. A24 reuses the existing `refresh-data` CLI (`scrape` → `ingest` → `fetch-understat` → `fetch-fotmob`) to backfill `raw_matches`/`feature_store` through the end of the season. A25 reuses the existing sequential `agent-snapshot` CLI over the full season date range. A26 reuses the existing `agent-backtest` CLI (sample sanity check, then full flat/kelly runs) and appends a new section to `documents/agent_techspec.md` with the results. **No new modules are created** — A09–A16 already built all the infrastructure this phase needs; this phase is almost entirely disciplined execution of it, gated by one small determinism bugfix.

**Tech Stack:** existing FPAI stack — DuckDB, pandas, LangGraph, Ollama (`llama3.1:8b`), Tavily, pytest.

**Known scope decisions (read before implementing):**

1. **Live-model determinism can't be unit tested.** True Ollama sampling determinism at `temperature=0` can only be verified by actually invoking the model twice, not by a mock. Task 1's unit tests lock in the two things that *are* testable without a live model — the config default, and that `temperature` actually reaches the `ChatOllama`/`ChatAnthropic` constructor — while the acceptance criterion's "zero skips on replay" claim is verified operationally in Task 2 against the real 24-match pilot corpus, not by a mocked test. `_build_llm` currently has zero direct test coverage (grep confirms every existing test patches it away), so the new test in Task 1 is a genuine coverage gain, not just a formality.
2. **Clearing `data/agent_snapshots/` before Task 2.** As of this writing the directory contains exactly the 24 matches from the A20 pilot (verify with `ls data/agent_snapshots | wc -l` before deleting — don't blind-delete if that count is unexpectedly different). `agent-snapshot` has no `--force` flag, only a skip-if-`_complete.json`-exists check, so pre-fix (temperature=0.1) snapshots must be removed first, or A23's re-verification will silently reuse stale recordings and prove nothing.
3. **A25's live collection is long-running.** `agent-snapshot` is sequential — no concurrency flag (`agent_techspec.md` Section 10). ~370+ matches, each a live Ollama tool-calling loop plus several live Tavily searches, is a multi-hour operation. Start it deliberately with `run_in_background: true` (or `nohup`), not as a blocking foreground step. It is resumable: a match with an existing `_complete.json` is skipped, so an interrupted run can just be re-invoked with the same date range.
4. **A26's `--sample 30` quick path exists to catch a broken corpus in minutes, not after a multi-hour full run.** Always run it (Task 5) before the full flat/kelly passes (Tasks 6–7).
5. **Per `CLAUDE.md`, `documents/agent_user_stories.md` gets marked complete with real numbers**, not placeholders — Task 8 does this last, after Tasks 2–7 have produced actual output to report, matching the prose style already used for A19–A21's completion notes.
6. **If any verification step fails its stated threshold** (pilot skips > 0 in Task 2, corpus errors > 0 in Task 4, sample skips > 2 in Task 5, full-run skip rate > 5% in Tasks 6–7), stop and diagnose via `superpowers:systematic-debugging` before proceeding to the next task or writing up results — do not average it away or note it as a caveat and continue.

---

## Task 1: A23 — `temperature=0` default + regression tests

**Files:**
- Modify: `config/agent_config.yaml`
- Modify: `tests/test_agent_config.py`
- Modify: `tests/test_agent_graph.py`
- Modify: `tests/test_snapshot_store.py`

- [ ] **Step 1: Write the failing config-default test**

Add to `tests/test_agent_config.py`:

```python
def test_default_config_temperature_is_zero_for_deterministic_replay():
    """A23: agent-snapshot relies on the LLM regenerating byte-identical
    tool-call args on replay so SnapshotStore.key_for() (a pure hash of
    those args) produces a cache hit. At temperature=0.1 (the pre-A23
    default) sampling noise let the LLM vary its own tool-call arguments
    between runs, causing SnapshotMissingError skips on replay even
    though the recorded corpus was complete (agent_techspec.md Section
    18.6). temperature=0 selects greedy decoding for both providers,
    removing that source of nondeterminism."""
    cfg = AgentConfig.default()
    assert cfg.temperature == 0.0
```

- [ ] **Step 2: Run it to verify it fails**

Run: `python -m pytest tests/test_agent_config.py::test_default_config_temperature_is_zero_for_deterministic_replay -v`
Expected: FAIL — `assert 0.1 == 0.0`

- [ ] **Step 3: Fix the config**

Edit `config/agent_config.yaml` line 6, replacing:

```yaml
temperature: 0.1                # lower = more deterministic; use 0.0 for backtesting
```

with:

```yaml
temperature: 0.0                # greedy decode — required for deterministic snapshot replay (A23)
```

- [ ] **Step 4: Run it to verify it passes**

Run: `python -m pytest tests/test_agent_config.py::test_default_config_temperature_is_zero_for_deterministic_replay -v`
Expected: PASS

- [ ] **Step 5: Add the `_build_llm` plumbing regression test**

Add to `tests/test_agent_graph.py` (uses the existing `_make_config` helper already defined at the top of this file):

```python
def test_build_llm_passes_configured_temperature_to_ollama():
    """A23: locks in that AgentConfig.temperature actually reaches the
    ChatOllama constructor — the config default alone (previous test)
    is useless if this plumbing silently breaks. No prior test in this
    file exercises _build_llm directly; every other test patches it away."""
    from unittest.mock import patch
    from src.agent.graph import _build_llm

    cfg = _make_config(provider="ollama", temperature=0.0, model="llama3.1:8b")
    with patch("langchain_ollama.ChatOllama") as MockChatOllama:
        _build_llm(cfg)
    MockChatOllama.assert_called_once_with(model="llama3.1:8b", temperature=0.0)


def test_build_llm_passes_configured_temperature_to_anthropic():
    from unittest.mock import patch
    from src.agent.graph import _build_llm

    cfg = _make_config(provider="anthropic", temperature=0.0, model="claude-haiku-4-5")
    with patch("langchain_anthropic.ChatAnthropic") as MockChatAnthropic:
        _build_llm(cfg)
    MockChatAnthropic.assert_called_once_with(model="claude-haiku-4-5", temperature=0.0)
```

- [ ] **Step 6: Run to verify both pass**

Run: `python -m pytest tests/test_agent_graph.py::test_build_llm_passes_configured_temperature_to_ollama tests/test_agent_graph.py::test_build_llm_passes_configured_temperature_to_anthropic -v`
Expected: PASS (both) — `_build_llm` already forwards `config.temperature` correctly, so this is a coverage-adding regression test, not a code fix.

- [ ] **Step 7: Add the snapshot-key stability test**

Add to `tests/test_snapshot_store.py`:

```python
def test_repeated_record_calls_with_identical_args_produce_identical_key(tmp_path):
    """A23: with temperature=0, the LLM is expected to regenerate byte-identical
    tool-call args on every agent-snapshot run over the same match. This test
    locks in the mechanism that guarantee depends on: SnapshotStore.key_for()
    must map identical args to the identical file every time a record-mode
    call is made, so a genuinely deterministic LLM produces zero
    SnapshotMissingError skips on replay (see agent_techspec.md Section 19)."""
    store = SnapshotStore(base_dir=tmp_path)
    store.set_mode("record")
    store.set_match("match-123")
    args = {"home_team": "Man City", "away_team": "Arsenal", "date": "2026-03-01"}

    store.wrap("forecast_league", lambda **kw: "response-1")(**args)
    key_first = store.key_for(args)

    store.wrap("forecast_league", lambda **kw: "response-2-would-mean-a-second-key")(**args)
    key_second = store.key_for(args)

    assert key_first == key_second
    files = list((tmp_path / "match-123").glob("forecast_league_*.json"))
    assert len(files) == 1
    assert json.loads(files[0].read_text())["response"] == "response-2-would-mean-a-second-key"
```

- [ ] **Step 8: Run to verify it passes**

Run: `python -m pytest tests/test_snapshot_store.py::test_repeated_record_calls_with_identical_args_produce_identical_key -v`
Expected: PASS

- [ ] **Step 9: Run the full agent test suite to check for regressions**

Run: `python -m pytest tests/test_agent_config.py tests/test_agent_graph.py tests/test_snapshot_store.py -v`
Expected: all PASS

- [ ] **Step 10: Commit**

```bash
git add config/agent_config.yaml tests/test_agent_config.py tests/test_agent_graph.py tests/test_snapshot_store.py
git commit -m "fix: default agent temperature to 0 for deterministic snapshot replay (A23)"
```

---

## Task 2: A23 — operational re-verification against the 24-match pilot

No source files change in this task — it exercises Task 1's fix against real (recorded) data to satisfy A23's operational acceptance criterion. Record the exact output of every step; it feeds Task 8's writeup.

- [ ] **Step 1: Confirm the snapshot directory holds only the known pilot corpus**

Run: `ls data/agent_snapshots | wc -l`
Expected: `24`. If it's not 24, stop and inspect `data/agent_snapshots/` before deleting anything — someone may have collected additional snapshots since this plan was written.

- [ ] **Step 2: Clear the pilot corpus so it gets re-recorded under temperature=0**

Run: `rm -rf data/agent_snapshots/*`

- [ ] **Step 3: Dry-run confirms the same 24 fixtures**

Run: `python main.py agent-snapshot --from-date 2026-03-01 --to-date 2026-03-16 --league E0 --dry-run`
Expected: `Matches in range: 24 | already complete: 0 | to process: 24`

- [ ] **Step 4: Re-record the pilot corpus for real**

Run: `python main.py agent-snapshot --from-date 2026-03-01 --to-date 2026-03-16 --league E0`
Expected: final line `Done. Processed: 24 | Errors: 0 | Skipped: 0`. If `Errors` is nonzero, stop and diagnose before continuing (per scope decision 6).

- [ ] **Step 5: Flat backtest over the freshly-recorded pilot**

Run: `python main.py agent-backtest --from-date 2026-03-01 --to-date 2026-03-16 --league E0 --stake-mode flat`
Expected: no `SKIP <match_id>: SnapshotMissingError` lines on stderr; printed report's `matches_evaluated` field equals `24`.

- [ ] **Step 6: Kelly backtest over the same range**

Run: `python main.py agent-backtest --from-date 2026-03-01 --to-date 2026-03-16 --league E0 --stake-mode kelly`
Expected: printed report's `matches_evaluated` field equals `24`.

- [ ] **Step 7: Confirm A23's acceptance criterion is met**

Both Step 5 and Step 6 must show `matches_evaluated: 24` (zero `SnapshotMissingError` skips on first replay, for both stake modes). Note the two reports' `roi`/`hit_rate`/`bet_frequency` values — Task 8 will note whether they materially differ from the original A21 baseline (23/24 flat, 20/24 kelly) now that every match is evaluated instead of a run-dependent subset.

---

## Task 3: A24 — refresh raw season data through end-of-season

- [ ] **Step 1: Run the composite refresh pipeline**

Run: `python main.py refresh-data --league E0 --force`

This runs `scrape` (re-downloads `E0_2526.csv` with `--force`, so it isn't skipped as already-present) → `ingest` (reloads `raw_matches`, rebuilds `feature_store` via `FeatureFactory.compute_rolling_stats`) → `fetch-understat` → `fetch-fotmob`, per `main.py:run_refresh_data` (lines 425–432).

- [ ] **Step 2: Verify the CSV now covers the full season**

Run: `wc -l data/raw/football_data/E0_2526.csv`
Expected: ≥ 381 lines (380 matches + 1 header). Before this task it was 302 lines (301 matches, ending 2026-03-16).

- [ ] **Step 3: Verify the date range extends through the season finale**

Run: `tail -3 data/raw/football_data/E0_2526.csv`
Expected: dates in May 2026, not March 2026.

- [ ] **Step 4: Verify `forecast_league` no longer needs a fallback for a late-season match**

Run:

```bash
python - <<'PY'
import json
from src.utils import DuckDBManager
from src.agent.tools import forecast_league

db = DuckDBManager()
with db.connection() as conn:
    row = conn.execute(
        "SELECT home_team, away_team, date, league, odds_h, odds_d, odds_a "
        "FROM raw_matches WHERE league = 'E0' AND date >= '2026-04-01' "
        "AND odds_h IS NOT NULL ORDER BY date LIMIT 1"
    ).fetchone()

assert row is not None, "No April/May 2026 E0 match with odds found — refresh-data may not have backfilled odds"
home, away, date, league, oh, od, oa = row
result = json.loads(forecast_league.invoke({
    "home_team": home, "away_team": away, "date": str(date), "league": league,
    "odds_h": oh, "odds_d": od, "odds_a": oa,
}))
basis = result["data_quality"]["prediction_basis"]
print(f"{home} vs {away} on {date}: prediction_basis={basis}")
assert basis == "team_history_and_market", result["data_quality"]
PY
```

Expected: prints `prediction_basis=team_history_and_market` and the assertion passes (not `market_odds_only_league_fallback`).

- [ ] **Step 5: Record the actual row count and date range** for Task 8's writeup.

---

## Task 4: A25 — full-season snapshot collection

- [ ] **Step 1: Dry-run to confirm fixture count**

Run: `python main.py agent-snapshot --from-date 2025-08-15 --to-date 2026-05-25 --league E0 --dry-run`
Expected: `Matches in range: N | ...` with N ≥ 370.

- [ ] **Step 2: Start the real collection run in the background**

This is a multi-hour, sequential, live-Ollama-and-Tavily operation (scope decision 3). Launch it with the Bash tool's `run_in_background: true`:

Run (background): `python main.py agent-snapshot --from-date 2025-08-15 --to-date 2026-05-25 --league E0`

Do not block synchronously waiting on this. It is safe to interrupt and re-run with the identical command — already-`_complete.json`-marked matches are skipped on the next invocation.

- [ ] **Step 3: Periodically check progress without blocking**

Run: `find data/agent_snapshots -maxdepth 1 -name "_complete.json" 2>/dev/null | wc -l` — actually `_complete.json` lives one level inside each match directory, so use:

Run: `find data/agent_snapshots -mindepth 2 -maxdepth 2 -name "_complete.json" | wc -l`

Compare against the dry-run's fixture count from Step 1 to gauge progress. Use `ScheduleWakeup`/`Monitor` to check back periodically rather than polling continuously.

- [ ] **Step 4: Verify completion**

Once the background command finishes, check its final output.
Expected: `Done. Processed: X | Errors: 0 | Skipped: 0` where `X` matches the dry-run count from Step 1. If `Errors` is nonzero, investigate and resolve per A25's acceptance criterion before proceeding — do not carry errors forward into Task 5.

- [ ] **Step 5: Verify every match directory has at least one `forecast_league_*.json` file**

Run:

```bash
python - <<'PY'
from pathlib import Path
base = Path("data/agent_snapshots")
match_dirs = [d for d in base.iterdir() if d.is_dir()]
missing = [d.name for d in match_dirs if not list(d.glob("forecast_league_*.json"))]
print(f"match dirs: {len(match_dirs)} | missing forecast_league snapshot: {len(missing)}")
assert not missing, f"BUG-011-style regression — zero tool-call files in: {missing[:5]}"
PY
```

Expected: `missing forecast_league snapshot: 0`. A nonzero count here is exactly the BUG-011 symptom (`agent_techspec.md` Section 18.4) — directories with only `_complete.json` and no real tool-response files — and must be treated as a blocking regression, not a caveat.

- [ ] **Step 6: Verify web_search snapshots were recorded too**

Run: `find data/agent_snapshots -name 'web_search_*.json' | wc -l`
Expected: a nonzero count.

- [ ] **Step 7: Record the final processed/error/skip counts and total match-dir count** for Task 8's writeup.

---

## Task 5: A26 — quick sample sanity check

- [ ] **Step 1: Run a 30-match stratified sample before committing to the full run**

Run: `python main.py agent-backtest --from-date 2025-08-15 --to-date 2026-05-25 --league E0 --sample 30 --stake-mode flat`

- [ ] **Step 2: Verify the skip count is within tolerance**

From the printed report's `matches_evaluated` field: `30 - matches_evaluated` must be ≤ 2 (A26's quick-test acceptance threshold). If it's higher, stop — treat it as a signal the full-season corpus has a systemic problem (e.g. a widespread key-mismatch pattern per Section 18.6) and diagnose via `superpowers:systematic-debugging` before running Tasks 6–7.

---

## Task 6: A26 — full flat-stake backtest

- [ ] **Step 1: Run the full flat backtest**

Run: `python main.py agent-backtest --from-date 2025-08-15 --to-date 2026-05-25 --league E0 --stake-mode flat`

Note: this replays recorded tool responses (no live Tavily/forecast calls), but the LLM call itself is still live per match, at the default `--concurrency 5`. Expect this to take meaningfully longer than a few minutes but far less than the multi-hour Task 4 recording run.

- [ ] **Step 2: Verify the skip rate is within tolerance**

`(total_matches - matches_evaluated) / total_matches` must be ≤ 5% (A26's full-run acceptance threshold), where `total_matches` is the dry-run fixture count from Task 4 Step 1.

- [ ] **Step 3: Confirm the report was saved**

Run: `ls -t reports/agent_backtest/ | head -1`
Expected: a new `<timestamp>_<config_hash>.json` file, newer than any pre-existing report in that directory.

- [ ] **Step 4: Record the full report's `roi`/`hit_rate`/`bet_frequency`/`max_drawdown`/`ending_bankroll`/`matches_evaluated`/`bets_placed` values** for Task 8's writeup.

---

## Task 7: A26 — full Kelly-stake backtest

- [ ] **Step 1: Run the full Kelly backtest**

Run: `python main.py agent-backtest --from-date 2025-08-15 --to-date 2026-05-25 --league E0 --stake-mode kelly`

- [ ] **Step 2: Verify the skip rate is within tolerance**

Same ≤ 5% threshold as Task 6 Step 2.

- [ ] **Step 3: Confirm the report was saved**

Run: `ls -t reports/agent_backtest/ | head -2` — expect both this run's report and Task 6's report present.

- [ ] **Step 4: Record the full report's metrics** for Task 8's writeup.

---

## Task 8: Document findings and mark stories complete

**Files:**
- Modify: `documents/agent_techspec.md`
- Modify: `documents/agent_user_stories.md`

- [ ] **Step 1: Append a new Section 19 to `documents/agent_techspec.md`**, immediately after the existing Section 18 (which ends at line ~689 with "18.8 Recommended next step"). Use the real values recorded in Tasks 2–7 — do not fabricate numbers ahead of running them. Structure to mirror Section 18's style:

```markdown
## 19. Full-Season Backtest Expansion (<actual date>, A23–A26)

Expands the 24-match A20/A21 pilot (2026-03-01 – 2026-03-16) to the full
2025/26 E0 season, gated by fixing a snapshot-determinism bug surfaced in
Section 18.6.

### 19.1 A23: temperature=0 and pilot re-verification

`config/agent_config.yaml`'s `temperature` changed from `0.1` to `0.0`
(greedy decode). Re-recorded the 24-match pilot corpus from scratch under
the new default and re-ran both stake modes:

| Stake mode | Matches evaluated (before A23) | Matches evaluated (after A23) |
|---|---|---|
| flat | 23/24 | <actual>/24 |
| kelly | 20/24 | <actual>/24 |

<One sentence on whether ROI/hit rate moved materially now that every
match is evaluated instead of a run-dependent subset, or note if a
skip still occurred and how it was resolved.>

### 19.2 A24: raw data refresh

`data/raw/football_data/E0_2526.csv` grew from 301 to <actual> matches,
now covering <actual start> through <actual end>. `forecast_league` for
a late-season match returns `prediction_basis: "team_history_and_market"`
with no fallback tag.

### 19.3 A25: full-season snapshot corpus

<actual> E0 matches, 2025-08-15 – 2026-05-25. `agent-snapshot` completed
with <actual> errors. Every match directory contains at least one
`forecast_league_*.json`; <actual count> `web_search_*.json` files
recorded in total.

### 19.4 A26: full-season backtest results

| Stake mode | Matches evaluated | Bets placed | Bets won | Hit rate | ROI | Max drawdown | Ending bankroll |
|---|---|---|---|---|---|---|---|
| flat | <actual> | <actual> | <actual> | <actual> | <actual> | <actual> | <actual> |
| kelly | <actual> | <actual> | <actual> | <actual> | <actual> | <actual> | <actual> |

**Findings:**
- <Note sample size relative to the 20-23 match pilot — is this now a
  large enough sample to say something about edge, or still not?>
- The forecast models' training cutoff (2023-04-27, per Section 22) predates
  the entire 2025/26 season, so this remains a genuinely out-of-sample
  evaluation.
- <Note whether the Section 18.7 result-leakage pattern recurred anywhere
  in the full corpus, or wasn't specifically re-audited at this scale.>
- <Any other qualitative pattern worth flagging for A22-style config
  comparison work, now that a full-season corpus exists to run it against.>
```

- [ ] **Step 2: Update `documents/agent_user_stories.md`** — change each of A23, A24, A25, A26's `Status` column from `future` to `completed`, and append a `**Completion notes (<actual date>):**` sentence or two to each row's Comments cell, in the same prose style already used for A19–A21 (i.e., state what actually happened/was measured, including any snag hit and how it was resolved — not just "done").

- [ ] **Step 3: Commit**

```bash
git add documents/agent_techspec.md documents/agent_user_stories.md
git commit -m "docs: record full-season backtest expansion results (A23-A26)"
```

---

## Self-Review Notes

- **Spec coverage:** A23's three acceptance bullets map to Task 1 (config) + Task 1 Steps 5–8 (plumbing/hash tests) + Task 2 (operational re-run). A24's three bullets map to Task 3 Steps 1–4. A25's four bullets map to Task 4 Steps 1, 4, 5, 6. A26's bullets map to Tasks 5–7 plus the techspec writeup in Task 8, including the explicit out-of-sample training-cutoff note.
- **CLAUDE.md compliance:** Task 8 satisfies both "document new functionality... in the technical document" and "mark the user story as completed."
- **No fabricated numbers:** every `<actual>` placeholder in Task 8's templates is filled in only after the corresponding operational task produces real output — this is intentional (a plan can't know a live model's ROI before it runs), not a violation of the "no placeholders" rule, which concerns *implementation* steps, not report templates awaiting real data.
