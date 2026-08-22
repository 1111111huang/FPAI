---
name: agent-train-experiment
description: Use when running a cost-aware agent-train experiment for a league — stratified sampling, full per-match logging (recommendation, lesson, reasoning), and a summarized-lessons readout. Triggers on requests like "run a training experiment for <league>", "sample training for <league>", "run an N-sample training and summarize lessons".
---

# Agent Train Experiment

## Overview

`agent-train` already writes results to DuckDB (`agent_telemetry`, `agent_lessons`), but
neither table alone gives a reviewable per-match record of what happened in one run, and
there's no built-in "what did we learn" summary. This skill wraps a single `agent-train`
invocation with three things on top:

1. **Cost-aware defaults** — `--sample 100` (stratified by actual result, seeded, see
   `src/agent/backtest.py::_stratified_sample`) instead of a full season, `--batch-size 1`
   (one lesson per match, the CLI's own default). At the measured Gemini rate (~$0.015
   CAD/match, `documents/agent_user_stories.md` A71/A72), a 100-match run costs ~$1.50 CAD.
2. **A combined per-match experiment log** — one JSON file with league, sample size, batch
   size, and for every match: the full recommendation, the lesson text (if one was written
   for that match), and the "thought process" (see Definitions below).
3. **A summarized "popular lessons" readout** — read every `lesson_text` this run produced
   and group them by recurring theme, most-frequent first. This is a reading/judgment step,
   not a deterministic script — do it directly, don't try to automate it with keyword
   matching.

**This is real, wanted data, not a throwaway calibration test.** Unlike the ad hoc
prompt-wording tests earlier in this project's history, do NOT delete or clean up the
`agent_lessons`/`agent_telemetry` rows this produces afterward. If the sampled matches
overlap an existing pending batch for the same league (very likely once a league already
has a large real batch — SP1's ~380-match season and a 100-match sample will overlap
heavily), that's fine and expected — note it in the summary, don't delete anything. The
new rows reflect whatever config/provider is live *now*, which may differ from an older
batch's provider (e.g. SP1's original 299-lesson batch predates the Gemini switch, A72) —
that's a genuine, useful comparison point, not duplication to clean up.

## Definitions

- **Sample**: `--sample N`, stratified across home/draw/away outcomes, `random_state=42` —
  reproducible across repeated runs with the same N and date range.
- **Batch size** ("matches per lesson"): `--batch-size B`. `B=1` (default) writes one
  lesson candidate per match. `B>1` aggregates up to `B` same-competition/tier matches into
  one shared lesson candidate instead (A39) — only use this if the request explicitly asks
  for aggregated lessons.
- **Thought process**: this pipeline doesn't expose real chain-of-thought (Gemini's API
  returns an opaque `thoughtSignature`, not readable reasoning text). The closest genuine
  analog already in the schema is the recommendation's own `explanation` array — the
  model's stated bullet-point reasoning for its call. Log that field under this label, and
  say so explicitly when reporting results — don't imply it's raw model reasoning if asked.

## Steps

1. **Resolve parameters.** Required: `league`. Defaults unless the request says otherwise:
   `sample=100`, `batch_size=1`, `split=all`, `config=config/agent_config.yaml` (the live
   default — currently Gemini, A72). Resolve `--from-date`/`--to-date` to the league's full
   season span (required flags even with `--sample`):
   ```
   ./venv/bin/python3 -c "
   import duckdb
   con = duckdb.connect('data/fpai_core.db', read_only=True)
   print(con.execute(\"SELECT MIN(date), MAX(date), COUNT(*) FROM raw_matches WHERE league='<LEAGUE>' AND date >= '2025-08-01'\").fetchone())
   "
   ```
2. **State the cost estimate before running** (sample_size × ~$0.015 CAD) so spend stays
   visible against whatever budget is in play.
3. **Run it**:
   ```
   ./venv/bin/python -m main agent-train --league <LEAGUE> --split all \
     --from-date <SEASON_START> --to-date <SEASON_END> \
     --sample <N> --batch-size <B> --config <CONFIG> 2>&1 | tail -40
   ```
   Capture the printed `run_id=...` and the report path from stdout.
4. **Pull this run's rows.** `agent_telemetry` has `run_id` directly; `agent_lessons` does
   not (see the hard-learned discipline in `agent_user_stories.md`'s A71/calibration
   notes) — join by `source_match_id` against this run's telemetry match_ids, scoped by
   `created_at` at/after the run's start time:
   ```python
   import duckdb, json
   con = duckdb.connect("data/fpai_core.db", read_only=True)
   rows = con.execute(
       "SELECT match_id, recommendation FROM agent_telemetry WHERE run_id = ?", [run_id]
   ).fetchall()
   match_ids = [r[0] for r in rows]
   lessons = con.execute(
       f"SELECT source_match_id, lesson_text, created_at FROM agent_lessons "
       f"WHERE source_match_id IN ({','.join(['?']*len(match_ids))}) AND created_at >= ?",
       match_ids + [run_start_ts],
   ).fetchall()
   ```
5. **Write the combined log** to `reports/agent_experiments/<UTC timestamp>_<league>_sample<N>_batch<B>.json`:
   ```json
   {
     "league": "...", "sample_size": 100, "batch_size": 1, "split": "all",
     "config_path": "config/agent_config.yaml", "model": "...", "provider": "...",
     "run_id": "...", "report_summary": { /* the agent-train report JSON verbatim */ },
     "matches": [
       {
         "match_id": "...", "home": "...", "away": "...", "date": "...",
         "recommendation": { /* full recommendation JSON */ },
         "lesson_text": "..." ,
         "thought_process": ["..."]
       }
     ]
   }
   ```
6. **Summarize popular lessons.** Read every `lesson_text` in the log, group by recurring
   theme (e.g. "stale/dated news evidence", "missing secondary-market odds", "high-entropy
   forecast flagged", "conflicting recent-form signals") and report the top themes ranked
   by frequency, with 1-2 representative quotes each — as prose in the response, not just
   left in the JSON file.
7. **Report back**: the report_summary stats (matches_evaluated, bets_placed, roi,
   hit_rate), the log file path, the popular-lessons summary, and actual spend (sample ×
   measured per-match rate) against budget.

## Gotchas

- `--from-date`/`--to-date` are required by the CLI even when `--sample` does the real
  narrowing — always resolve the full season span first, don't guess it.
- `agent_lessons` has no `run_id` column — never scope a query (or, worse, a DELETE) to it
  by date/text guesswork alone; always join through `agent_telemetry`'s real `run_id` →
  `match_id` first, exactly as step 4 does.
- Don't clean up this run's rows afterward (see Overview) — that discipline exists for
  *calibration test* runs specifically, not for real experiment/lesson-generation runs like
  this one.
