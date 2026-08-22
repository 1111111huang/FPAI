# Agent Risk Posture Presets Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add three selectable, independently-readable agent configs (conservative/balanced/aggressive) that control how often the agent proposes a `direct_bet`/`conditional` recommendation, calibrated against a real, observed (direct_bet + conditional) recommendations-per-matchday rate for SP1.

**Architecture:** Reuses the existing `AgentConfig.system_prompt_version` + `--config` CLI mechanism (already used by `config/agent_config_deepseek.yaml`) — zero new code for the selection mechanism itself. Three new prompt files change the LLM's own willingness to act on a numerically-qualifying `value_edge` (confirmed the dominant lever); three new config files pair each prompt with a secondary `min_value_edge` adjustment. Calibration is empirical against small, real backtest samples — cost-conscious by design (10-match samples, not full-season runs, until a posture is confirmed correct).

**Tech Stack:** Python, existing `agent-backtest` CLI, DeepSeek (`config/agent_config.yaml`'s current default provider).

**Spec:** `docs/superpowers/specs/2026-08-21-agent-risk-posture-design.md`

**Cost note for whoever executes this:** every calibration step below is a real, billed DeepSeek call per match. Sample sizes are deliberately kept to 10 matches per test (not the full 299-match season) — do not enlarge a sample "just to be sure" without checking with the user first, per their own stated cost sensitivity earlier in this investigation.

---

### Task 1: Create the three posture prompt files

**Files:**
- Create: `config/prompts/agent_v1_conservative.txt`
- Create: `config/prompts/agent_v1_balanced.txt`
- Create: `config/prompts/agent_v1_aggressive.txt`

Each file is `config/prompts/agent_v1.txt` verbatim, except lines 36-40 (the four `value_edge`/odds-bounds/conditional-eligibility bullets under `## Value Calculation`) are replaced with the posture-specific block below. Read `config/prompts/agent_v1.txt` yourself first to get its exact current byte-for-byte content — do not reconstruct it from memory or from this plan's own quoting of it, in case it has changed since this plan was written.

- [ ] **Step 1: Create `agent_v1_conservative.txt`**

Copy `config/prompts/agent_v1.txt` to `config/prompts/agent_v1_conservative.txt`, then replace this line:

```
- Never recommend direct_bet unless this specific market's own value_edge is actually >= 0.05 — recommendation_type='direct_bet' with a value_edge below that (including negative) is incoherent and will be downgraded to "no_bet" automatically. If the numbers don't support a direct bet, call it "no_bet", don't call it "direct_bet" anyway.
```

with these two lines (keep the two lines after it — odds-bounds and conditional-eligibility — unchanged):

```
- Never recommend direct_bet unless this specific market's own value_edge is actually >= 0.05 — recommendation_type='direct_bet' with a value_edge below that (including negative) is incoherent and will be downgraded to "no_bet" automatically. If the numbers don't support a direct bet, call it "no_bet", don't call it "direct_bet" anyway.
- A qualifying value_edge is necessary but not sufficient. Before recommending, check the forecast's own uncertainty (high entropy/cold_start_risk) and the available team news for anything that meaningfully contradicts the model's assumption. Decline if either raises real doubt, even without a single decisive red flag — err toward no_bet when genuinely unsure.
```

- [ ] **Step 2: Create `agent_v1_balanced.txt`**

Copy `config/prompts/agent_v1.txt` to `config/prompts/agent_v1_balanced.txt`, then replace the same line as Step 1 with:

```
- Never recommend direct_bet unless this specific market's own value_edge is actually >= 0.05 — recommendation_type='direct_bet' with a value_edge below that (including negative) is incoherent and will be downgraded to "no_bet" automatically. If the numbers don't support a direct bet, call it "no_bet", don't call it "direct_bet" anyway.
- If value_edge clears the threshold and current_odds is a real, current price in the allowed range, recommend it. Only decline a qualifying edge when you have a SPECIFIC, concrete reason tied to this exact match (e.g. a confirmed absence of a key starter, verified conflicting recorded evidence, or a stated data-quality problem with the forecast itself) — not generic caution like "the model may lack full context" or "uncertainty is high" on its own. A qualifying number is the default signal to act on, not a suggestion to second-guess.
```

- [ ] **Step 3: Create `agent_v1_aggressive.txt`**

Copy `config/prompts/agent_v1.txt` to `config/prompts/agent_v1_aggressive.txt`, then replace the same line with:

```
- Never recommend direct_bet unless this specific market's own value_edge is actually >= 0.05 — recommendation_type='direct_bet' with a value_edge below that (including negative) is incoherent and will be downgraded to "no_bet" automatically. If the numbers don't support a direct bet, call it "no_bet", don't call it "direct_bet" anyway.
- Trust a qualifying value_edge by default. Decline only if there is a specific, named, current fact that directly contradicts the recommendation (e.g. a confirmed missing key player, or evidence the match itself may not occur as scheduled) — general model uncertainty, entropy, or "the model might be wrong" reasoning alone is not sufficient grounds to decline a market that already clears the threshold.
```

- [ ] **Step 4: Verify all three files are otherwise identical to the base prompt**

Run: `diff config/prompts/agent_v1.txt config/prompts/agent_v1_conservative.txt`, `diff config/prompts/agent_v1.txt config/prompts/agent_v1_balanced.txt`, `diff config/prompts/agent_v1.txt config/prompts/agent_v1_aggressive.txt`.
Expected: each diff shows exactly one changed region (the line replaced by two lines in Steps 1-3), nothing else differs.

- [ ] **Step 5: Commit**

```bash
git add config/prompts/agent_v1_conservative.txt config/prompts/agent_v1_balanced.txt config/prompts/agent_v1_aggressive.txt
git commit -m "feat(agent): add conservative/balanced/aggressive prompt variants (risk posture)

Co-Authored-By: Claude Sonnet 5 <noreply@anthropic.com>"
```

---

### Task 2: Create the three posture config files

**Files:**
- Create: `config/agent_config_conservative.yaml`
- Create: `config/agent_config_balanced.yaml`
- Create: `config/agent_config_aggressive.yaml`
- Test: `tests/test_agent_config.py`

- [ ] **Step 1: Write the failing tests**

Read `tests/test_agent_config.py` first to match its existing style/imports. Add:

```python
def test_conservative_posture_config_loads_with_own_prompt_and_edge():
    cfg = AgentConfig.from_yaml("config/agent_config_conservative.yaml")
    assert cfg.system_prompt_version == "v1_conservative"
    assert cfg.min_value_edge == 0.06


def test_balanced_posture_config_loads_with_own_prompt_and_edge():
    cfg = AgentConfig.from_yaml("config/agent_config_balanced.yaml")
    assert cfg.system_prompt_version == "v1_balanced"
    assert cfg.min_value_edge == 0.05


def test_aggressive_posture_config_loads_with_own_prompt_and_edge():
    cfg = AgentConfig.from_yaml("config/agent_config_aggressive.yaml")
    assert cfg.system_prompt_version == "v1_aggressive"
    assert cfg.min_value_edge == 0.04


def test_all_three_posture_configs_keep_every_other_field_identical_to_default():
    default = AgentConfig.default()
    for posture in ("conservative", "balanced", "aggressive"):
        cfg = AgentConfig.from_yaml(f"config/agent_config_{posture}.yaml")
        assert cfg.model == default.model
        assert cfg.provider == default.provider
        assert cfg.temperature == default.temperature
        assert cfg.max_tool_calls == default.max_tool_calls
        assert cfg.min_odds_threshold == default.min_odds_threshold
        assert cfg.max_odds_threshold == default.max_odds_threshold
        assert cfg.min_conditional_odds_threshold == default.min_conditional_odds_threshold
        assert cfg.markets == default.markets
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `./venv/bin/python -m pytest tests/test_agent_config.py -k posture -v` (adjust the venv path if not using the repo's own `./venv/bin/python`).
Expected: FAIL — `FileNotFoundError: Agent config not found: config/agent_config_conservative.yaml` (files don't exist yet).

- [ ] **Step 3: Read `config/agent_config.yaml` to get its exact current content**

Read the file yourself — do not reconstruct it from memory or from any other document's quotation of it, in case it has changed since this plan was written.

- [ ] **Step 4: Create the three config files**

Copy `config/agent_config.yaml` to each of the three new filenames, then in each, change only `system_prompt_version` and `min_value_edge`:

`config/agent_config_conservative.yaml`: `system_prompt_version: "v1_conservative"`, `min_value_edge: 0.06`
`config/agent_config_balanced.yaml`: `system_prompt_version: "v1_balanced"`, `min_value_edge: 0.05`
`config/agent_config_aggressive.yaml`: `system_prompt_version: "v1_aggressive"`, `min_value_edge: 0.04`

Add a one-line comment above each changed field noting it's part of the risk-posture experiment (2026-08-21) and pointing at `docs/superpowers/specs/2026-08-21-agent-risk-posture-design.md`, matching this repo's existing convention of dating/explaining config changes inline.

- [ ] **Step 5: Run tests to verify they pass**

Run: `./venv/bin/python -m pytest tests/test_agent_config.py -v`
Expected: all PASS, zero regressions on the pre-existing tests in this file.

- [ ] **Step 6: Commit**

```bash
git add config/agent_config_conservative.yaml config/agent_config_balanced.yaml config/agent_config_aggressive.yaml tests/test_agent_config.py
git commit -m "feat(agent): add conservative/balanced/aggressive config presets (risk posture)

Co-Authored-By: Claude Sonnet 5 <noreply@anthropic.com>"
```

---

### Task 3: Calibrate the conservative posture

**Files:** none changed by this task unless wording needs adjusting (see Step 3)

**Cost:** 10 real DeepSeek calls per attempt (one small sample), plus up to 10 more if a re-test is needed. Do not use a larger sample without checking with the user first.

**Why `agent-train`, not `agent-backtest`:** confirmed directly in this project's own investigation (see design spec's Motivation) that plain `agent-backtest` never writes per-match detail to `agent_telemetry` at all — only `agent-train` does, and its own report/stdout summary doesn't break out `conditional` counts either. `agent-train` is used here purely as a read mechanism for per-match recommendation JSON; the lesson candidates it writes as a side effect are calibration noise, not wanted output, and get deleted in Step 3 below.

- [ ] **Step 1: Run the primary calibration sample**

```bash
./venv/bin/python main.py agent-train --from-date 2025-08-29 --to-date 2025-08-31 --league SP1 --split all --config config/agent_config_conservative.yaml
```

(`--split all` here, not `train`/`test` — this is a fixed 10-match calibration window used identically across all three postures for a fair comparison, not part of the train/test corpus split used for the real SP1 lesson-generation work done earlier.) Note the `run_id` printed in the CLI's own final summary line (`Wrote N lesson candidates and M telemetry rows (run_id=...)`).

- [ ] **Step 2: Compute the actual rate from telemetry**

```bash
./venv/bin/python -c "
import duckdb, json, collections
conn = duckdb.connect('data/fpai_core.db', read_only=True)
rows = conn.execute('SELECT recommendation FROM agent_telemetry WHERE run_id=?', ['<RUN_ID_FROM_STEP_1>']).fetchall()
counts = collections.Counter(json.loads(r[0])['overall'] for r in rows)
n = len(rows)
actionable = counts.get('direct_bet', 0) + counts.get('conditional', 0)
print('n=', n, dict(counts), 'actionable=', actionable)
"
```

Target: `actionable < 3` across this ~10-match sample (i.e. roughly < 3/matchday, since this window is one matchday-sized sample).

- [ ] **Step 3: Clean up this calibration run's lesson candidates**

```bash
./venv/bin/python3 -c "
import duckdb
conn = duckdb.connect('data/fpai_core.db')
conn.execute(\"DELETE FROM agent_lessons WHERE source_match_id IN (SELECT DISTINCT match_id FROM agent_telemetry WHERE run_id=?) AND status='pending'\", ['<RUN_ID_FROM_STEP_1>'])
conn.close()
"
```

Confirm first that this only targets pending, unreviewed rows (it does, by construction — `status='pending'`) and that it's scoped to this specific run's match_ids only, not a blanket delete. Do NOT touch the real SP1 train-split lesson candidates already generated and reviewed as part of the earlier lesson-generation work in this project — those are a completely disjoint set of `run_id`s from this calibration testing.

- [ ] **Step 4: If the rate misses the target, adjust wording and re-test once on the second window**

If `actionable >= 3`: the prompt is still too permissive for "conservative" — strengthen `config/prompts/agent_v1_conservative.txt`'s second bullet (from Task 1) with more specific caution language, then repeat Steps 1-3 against the **second** calibration window instead of the same one (avoids overfitting wording to one specific sample):

```bash
./venv/bin/python main.py agent-train --from-date 2025-11-07 --to-date 2025-11-09 --league SP1 --split all --config config/agent_config_conservative.yaml
```

If `actionable == 0`: the prompt (or `min_value_edge=0.06`) may now be too strict in the other direction — loosen slightly and re-test the same way. Iterate at most twice before escalating to the user rather than continuing to guess at wording indefinitely.

- [ ] **Step 5: Record the result**

Note the final prompt wording used (if changed from Task 1's draft) and the actual measured rate — this gets written into the completion notes in Task 7, not a separate file.

---

### Task 4: Calibrate the balanced posture

Same procedure as Task 3 (including the Step 3 lesson-candidate cleanup — never skip it), using `config/agent_config_balanced.yaml`, target **~5** (direct_bet + conditional) recommendations across the 10-match sample. Same two calibration windows, same iterate-at-most-twice rule, same cost note.

---

### Task 5: Calibrate the aggressive posture

Same procedure as Task 3 (including the Step 3 lesson-candidate cleanup — never skip it), using `config/agent_config_aggressive.yaml`, target **> 7** (direct_bet + conditional) recommendations across the 10-match sample. Same two calibration windows, same iterate-at-most-twice rule, same cost note.

---

### Task 6: One larger validation pass per posture

**Cost:** three runs, each covering a modest (not full-season) sample — keep this deliberately smaller than a full 299-match validation unless the user asks for that level of confidence; a full-season run is an easy follow-up they can run themselves later with the exact same command, one `--to-date` change.

- [ ] **Step 1: Run each posture against a larger (but still modest) sample to confirm the rate holds at scale**

For each posture, run against a ~40-match window (roughly 4 matchdays) not used in Tasks 3-5's calibration:

```bash
./venv/bin/python main.py agent-train --from-date 2025-12-01 --to-date 2025-12-31 --league SP1 --split all --config config/agent_config_<posture>.yaml
```

Note each run's `run_id`.

- [ ] **Step 2: Compute and record each posture's rate over this larger sample, plus its ROI/hit-rate**

Reuse Step 2's query from Task 3 (per-run `actionable` count) to get the rate, normalized to per-matchday (matches / ~10). Also pull `roi`/`hit_rate` from the same CLI run's printed evaluation report (this is a train-split-shaped report the same way earlier `agent-train` runs in this project produced one) — not to declare a winner (a ~40-match sample is still small, same standing caveat this project attaches to every small-sample backtest number), just as an honest data point alongside the rate.

- [ ] **Step 3: Clean up these three runs' lesson candidates**

Same deletion pattern as Task 3 Step 3, once per posture's `run_id` from Step 1.

---

### Task 7: Document and mark complete

**Files:**
- Modify: `documents/agent_user_stories.md`
- Modify: `documents/agent_techspec.md`

- [ ] **Step 1: Append a new completed story to `documents/agent_user_stories.md`**

Follow the exact row format of the existing entries (see A69 for the most recent example of this project's own style: real numbers, honest reporting, no overclaiming). Content: the original finding (SP1's 08-07 218-bet run vs. today's ~1-bet run, root-caused to prompt-language hardening not model/thresholds), the three posture presets built, each posture's final calibrated prompt wording delta and `min_value_edge`, the actual measured rates from Tasks 3-6 (small-sample and larger-sample), and the ROI/hit-rate data points from Task 6 with the small-sample caveat stated plainly.

- [ ] **Step 2: Add a short technical-doc note**

In `documents/agent_techspec.md`, add a new section (check the file's current highest `## N.` section number first and use the next one) describing: the mechanism (reuses `system_prompt_version`/`--config`, zero new runtime code), the three files per posture, why prompt language rather than thresholds is the primary lever (the evidence from this investigation), and the calibration numbers from Tasks 3-6.

- [ ] **Step 3: Commit**

```bash
git add documents/agent_user_stories.md documents/agent_techspec.md
git commit -m "docs: agent risk-posture presets — calibration results and story completion

Co-Authored-By: Claude Sonnet 5 <noreply@anthropic.com>"
```
