---
name: agent-backtest-run
description: Use when running a cost-aware, trustworthy agent-backtest for a league or leagues — checks DeepSeek balance before every chunk, validates each chunk's results before spending on the next, stops on repeated invalid results, and records which agent code version and ML model versions produced the report. Triggers on requests like "run a backtest for <league>", "get a real ROI number for <league>", "run the full-season backtest".
---

# Agent Backtest Run

## Overview

A single unmoderated `agent-backtest` invocation over a large date range has already
produced two real, silent failures in this project's history, both of which completed
with exit code 0 and looked like a normal run from the outside:

1. **BUG-072** — the snapshot corpus went stale (a new query added by A115 wasn't in
   older recordings) and the run silently evaluated a degraded match subset.
2. **A56** — a mid-run DeepSeek balance lapse truncated a run at 61/82 matches with no
   warning; the partial run's own +26.1% ROI was later cited as if it were a real,
   complete-sample number.

This skill wraps a plain `agent-backtest` invocation with four things aimed squarely at
those two failure modes, plus a real audit trail:

1. **Pre-chunk DeepSeek balance check** — a single call cannot be paused mid-flight to
   re-check balance (no such hook exists in the CLI), so the run is split into chunks
   sized to the *current* balance, and balance is re-checked before every chunk, not
   just once at the start.
2. **Per-chunk validation** — skip rate, the model-staleness line `agent-backtest`
   already prints (fixed in BUG-073 — see Gotchas), and result plausibility, checked
   after every chunk, not only at the end.
3. **A hard stop on repeated invalid results** — mirrors `agent-train-experiment`'s own
   Step 7 gate, reusing this project's own established A26 skip-rate thresholds.
4. **A traceability manifest** — the git commit hash (the actual *code* version —
   `config_hash`/`agent_config_hash` only ever hash `AgentConfig`'s YAML fields, never
   the pipeline/prompt code) plus `config/model_selection.yaml`'s contents at run time
   (the actual *model* version), written alongside the combined report.

**Accepted vs. rejected variance:** BUG-074 confirmed real LLM synthesis-call
non-determinism at temperature=0 (1/9 matches flipped recommendation across two
identical back-to-back runs). This skill does not try to eliminate that — it's treated
as noise averaged out by test-set size. What this skill *does* guard is the
deterministic-pipeline side: which snapshot gets matched on replay must stay reliable
(canonicalization, A23, already fixed) and every chunk must be checked against the
currently-promoted models before its numbers are trusted. Don't report a single small
chunk's ROI as a headline number — pool bets across the whole run first (Step 6).

## Definitions

- **Chunk**: one bounded `agent-backtest` invocation (`--sample N` or a narrow
  `--from-date`/`--to-date` slice), sized so its estimated cost sits comfortably inside
  the *current* remaining DeepSeek balance — never "the whole date range in one command."
- **Real per-match cost**: **~$0.02 USD/match**, reconciled from the account's own
  Cost(CNY) dashboard (`documents/agent_user_stories.md` A72/A109) — not balance-delta
  arithmetic, which this project has already found unreliable twice (a UTC/CST bucketing
  error, and an unexplained ~3.5x gap against a token×rate-card estimate). Use this
  figure only to size a chunk conservatively; the Cost(CNY) dashboard is the authoritative
  record of what a run actually cost, checked after the fact.
- **Model staleness**: whether a match's recorded forecast reflects the currently
  promoted model (`config/model_selection.yaml`) or an older, since-replaced one.
  `agent-backtest` already prints this automatically (`check_model_staleness` /
  `print_staleness_summary`, `src/agent/backtest.py`) — no extra command needed, just
  read the printed line (see Step 4).
- **Agent version**: the git commit hash of the working tree at run time
  (`git rev-parse HEAD`) — confirmed via direct code read to be the only thing that
  captures a pipeline/prompt/schema code change; both existing hash mechanisms
  (`config_hash` in `src/agent/evaluation.py`, `agent_config_hash`) only ever hash
  `AgentConfig`'s YAML-sourced fields.

## Steps

1. **Confirm the working tree is clean before spending anything.**
   ```bash
   git status --short
   ```
   A report traced to "whatever the working tree happened to contain" isn't traceable —
   don't proceed on a dirty tree unless the user explicitly says otherwise.

2. **Check DeepSeek balance and size the first chunk.**
   ```bash
   python3 -c "
   import os, requests
   from dotenv import load_dotenv
   load_dotenv()
   key = os.environ['DEEPSEEK_API_KEY']
   resp = requests.get('https://api.deepseek.com/user/balance',
                        headers={'Authorization': f'Bearer {key}'}, timeout=15)
   print(resp.json())
   "
   ```
   Read `balance_infos` for the funding currency's `total_balance` (this account funds
   in CNY). Convert to USD at the *current* exchange rate — don't reuse an old cached
   rate. Size the chunk so its estimated cost (`match_count × ~$0.02`) is well under half
   the remaining balance; never plan a chunk that would exhaust the account if the
   estimate runs 2x high (real precedent: A77's estimate was off by ~3.5x once).

3. **Run one chunk — full output to a real file, never piped through `tail` directly.**
   A `  SKIP <match_id>: <exc>` line is printed to stderr (`main.py`'s
   `_run_backtest_concurrent`), never logged, and unrecoverable once scrolled past
   (`agent-train-experiment`'s own hard-learned Gotcha — the identical risk applies
   here verbatim):
   ```bash
   ./venv/bin/python -m main agent-backtest --from-date <chunk_start> --to-date <chunk_end> \
     --league <LEAGUE> --split <all|train|test> --stake-mode <flat|kelly> \
     [--sample N] --config <CONFIG> \
     > reports/agent_backtest/<UTC timestamp>_<league>_chunk.raw.log 2>&1
   tail -60 reports/agent_backtest/<...>.raw.log
   ```

4. **Validate this chunk before spending on the next one.** All three checks, every
   chunk, not just at the end:
   - **Skip rate**: `grep -c "^  SKIP " <raw log>` against the chunk's requested match
     count. Reuse this project's own A26 acceptance bar: **≤2 skips** on a small
     (~30-match) chunk, **≤5% of the chunk** on a full-size one. Exceeding either: stop,
     read the actual `SKIP <match_id>: <exc>` lines (the exception text says exactly
     what broke) — don't just note the count and continue.
   - **Model staleness**: read the `Model staleness check: N/M matches...` line the CLI
     prints automatically. Any staleness at all: stop, run
     `agent-snapshot --refresh-model` for the affected league (BUG-073's own
     remediation), then re-run this chunk — don't mix a stale-model chunk in with fresh
     ones. **No line printed at all** means `matches_checked` was 0 (no captured
     diagnostics), not "confirmed fresh" — treat silence as "unable to verify," not as a
     pass.
   - **Result plausibility**: from the printed report, `bet_frequency`/`hit_rate` should
     be in `[0, 1]`, `roi` not `NaN` or a suspicious flat value (e.g. every bet resolving
     identically usually means a market-resolution bug, not real signal), and
     `insufficient_data_rate` not near 1.0 (near-total forecast failure, not a real
     betting-signal read).
   - **The hard stop**: if any single chunk fails the skip-rate bar, or two consecutive
     chunks show unexplained staleness or implausible results, stop the whole run and
     investigate before spending on any further chunk.

5. **Re-check balance before every subsequent chunk** (repeat Step 2) — not just once at
   the start. This is the direct fix for A56's own incident: a chunk affordable at the
   start of a long session may not be by the fifth chunk.

6. **Record the traceability manifest** alongside the combined report, one per run:
   ```bash
   python3 -c "
   import subprocess, json, yaml, datetime
   commit = subprocess.check_output(['git', 'rev-parse', 'HEAD']).decode().strip()
   dirty = bool(subprocess.check_output(['git', 'status', '--short']).decode().strip())
   with open('config/model_selection.yaml') as f:
       model_selection = yaml.safe_load(f)
   manifest = {
       'generated_at': datetime.datetime.now(datetime.timezone.utc).isoformat(),
       'agent_git_commit': commit,
       'working_tree_dirty': dirty,
       'model_selection_contexts': model_selection.get('contexts', {}),
   }
   print(json.dumps(manifest, indent=2))
   " > reports/agent_backtest/<UTC timestamp>_<league>_manifest.json
   ```
   This is what actually answers "which agent" and "which ML models" produced a report
   later — the report filename's `config_hash` alone cannot (Gotchas).

7. **Report back**: per-chunk stats, the pooled ROI/hit-rate across all chunks (bet-count
   weighted, not chunk-count weighted; label separately if any chunk differs materially
   in character — different date range, different split), the manifest file path, real
   spend reconciled against the DeepSeek Cost(CNY) dashboard, and a one-line reminder
   that a single small-sample ROI is one noisy draw (BUG-074), not ground truth, unless
   pooled over a large-enough sample (A111's own ~3.3pp-standard-error, n≈229-bet
   precedent for what "large enough" has looked like on this project).

## Gotchas

- **A completed CLI exit code proves nothing.** Both BUG-072 and A56's balance lapse
  completed with exit code 0 and looked like clean runs from the outside.
- **`config_hash`/`agent_config_hash` are not a substitute for the git commit.**
  Confirmed via direct code read: both only ever hash `AgentConfig`'s YAML-sourced
  fields, never the graph/schema/prompt code. Two reports with an identical
  `config_hash` are not guaranteed to reflect the same agent behavior.
- **`check_model_staleness()` was silently broken until BUG-073 (fixed 2026-09-26/27)**
  — both real call sites defaulted to `config_path="config.yaml"` (no `contexts` key),
  so it reported "0 stale" unconditionally regardless of truth. A report predating that
  fix's commit has a meaningless staleness line; check which commit produced it (Step 6)
  before trusting it.
- **Balance-delta arithmetic is not reliable for cost tracking** — this project has
  already been burned twice (a UTC/CST bucketing error, an unexplained ~3.5x estimate
  gap). Use the balance API purely as a stop/go gate before spending; use the account's
  own Cost(CNY) dashboard as the authoritative record of what a run actually cost.
- **Never pipe a chunk's live run through `tail` directly** — identical risk to
  `agent-train-experiment`'s own hard-learned Gotcha: a `SKIP` line is stderr-only,
  never logged, and unrecoverable once scrolled past. Always redirect to a file (Step 3).
- **home_corners/away_corners never resolve in any backtest** (`RESOLVABLE_MARKETS`
  structurally excludes them) — if a chunk placed such a bet, it's real spend that
  contributes nothing to the reported ROI/hit-rate. Not a bug to chase here.
- **A single chunk's ROI is not the final number.** Pool bets across every chunk
  (weighted by bets placed) before reporting a headline ROI — a small chunk's ROI
  swinging wildly between two otherwise-identical runs is BUG-074's confirmed effect,
  not a chunk-sizing mistake.
