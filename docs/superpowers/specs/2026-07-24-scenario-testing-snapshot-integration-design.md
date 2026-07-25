# Scenario Testing: Integrating the Agent-Snapshot Corpus into the Webapp Sandbox

**Date:** 2026-07-24
**Status:** Approved, pre-implementation
**Covers:** new `W##` stories to be appended to `documents/app_user_stories.md`

## Motivation

A new `agent-snapshot` corpus is being actively recorded (Anthropic API-based, per `documents/agent_user_stories.md`'s A34 rebaseline story) intended to cover Aug 2025–May 2026 for both E0 and SWE. The user asked whether the redesigned webapp will "work with" these new snapshots for scenario testing. Investigation found it will not, as things stand — two structurally separate systems exist today:

- **`agent-snapshot`/`agent-backtest`** (`src/agent/snapshot_store.py`, `main.py`) — a standalone evaluation tool, writing to `data/agent_snapshots/<LEAGUE>/<match_id>/`, keyed by the real `raw_matches.match_id`.
- **The webapp's sandbox mode** (`app/backend/recommendations.py`) — writes/reads its own `data/agent_snapshots/sandbox/<home>__<away>__<date>/`, keyed by a composite string, entirely separate from the corpus above.

Two further gaps were found while investigating:

1. The webapp's sandbox mode doesn't even reuse its **own** prior recordings across a backend restart — record-vs-replay is decided by an in-memory Python `set()` that starts empty every process start, so a fresh process re-records (and silently overwrites) rather than replaying.
2. SWE fixture discovery for arbitrary historical dates doesn't work at all — `SwedenFixturesClient` is backed by The Odds API, which only ever exposes the last few real days, not an arbitrary date in the past (unlike football-data.org, which W45 already wired up for E0).

## Goals

- Any fixture in the `E0`/`SWE` agent-snapshot corpus, once recorded, can be replayed by the webapp's sandbox mode without a live LLM/API call.
- The webapp's own sandbox recordings survive a backend restart (fixes the in-memory persistence bug as a natural side effect of the same code path).
- SWE fixtures resolve correctly for arbitrary historical dates in `/api/fixtures`, matching E0's existing behavior (W45).
- `scripts/launch_sandbox.py --precompute` covers both E0 and SWE.
- A repeatable, documented way exists to verify scenario coverage across the full Aug 2025–May 2026 range for both leagues, in this project's existing runbook style (W31/W44) — not a new CI suite, since this project has none.

## Non-goals

- No changes to `agent-snapshot`/`agent-backtest`/`SnapshotStore`'s own recording format or CLI — this is purely about the webapp *consuming* what already exists.
- No guarantee that every single date in the range has a corpus recording — the corpus is whatever A34's rebaseline job actually produces. This work makes the webapp correctly use whatever exists; it doesn't expand corpus coverage itself.
- No change to which LLM provider the live (non-snapshot) path uses — `config/agent_config.yaml` is out of scope here.

## Architecture

### 1. Corpus-aware sandbox replay (`app/backend/recommendations.py`)

Replace `_sandbox_recorded_matches: set[str]` and its three call sites with a function that decides mode by checking disk, in priority order:

```
def _resolve_sandbox_snapshot_source(home_team, away_team, date, league) -> (mode, match_id, base_dir):
    1. sandbox_match_id = _composite_match_key(home_team, away_team, date)
       if a complete recording exists at data/agent_snapshots/sandbox/<sandbox_match_id>/:
           return ("replay", sandbox_match_id, SANDBOX_BASE_DIR)
    2. corpus_match_id = _lookup_corpus_match_id(home_team, away_team, date, league)  # via TeamNameMapper + raw_matches
       if corpus_match_id and a complete recording exists at data/agent_snapshots/<league>/<corpus_match_id>/:
           return ("replay", corpus_match_id, CORPUS_BASE_DIR / league)
    3. return ("record", sandbox_match_id, SANDBOX_BASE_DIR)
```

"Complete" means the same marker `agent-snapshot`'s own CLI already uses (`_complete.json` present) — reusing the existing convention rather than inventing a new one.

`_lookup_corpus_match_id` resolves `home_team`/`away_team` through the existing `TeamNameMapper` (`src/ingestion/common/team_mapping.py`, the same tool W06 already uses for the identical class of problem — a fixtures-API team name needing to match the ML engine's canonical name) before querying `raw_matches` for the matching `match_id`. A mapping miss degrades to "no corpus match, fall through to record" — never an error, consistent with this codebase's existing "an unmapped team logs a warning, doesn't crash" precedent (`FeatureFactory.build_for_match`).

The existing W43 fallback (a replay-mode `SnapshotMissingError` retries once in record mode, since LLM-influenced follow-up `web_search` calls aren't perfectly reproducible) is preserved unchanged and now also covers a corpus-replay miss, not just a sandbox-partition miss.

**Why this also fixes the persistence bug:** step 1 alone — checking disk instead of an in-memory set — means the sandbox's own recordings now survive a restart, with no separate fix needed.

### 2. SWE historical fixture source (`app/backend/main.py` / new small module)

Add a `raw_matches`-backed historical source for SWE, used for the past-date branch of `/api/fixtures` (mirroring the existing `results`/`results_swe` split, `main.py:308-330`, but sourcing SWE's past-date branch from the DB instead of `SwedenFixturesClient`, since the Odds API has no arbitrary-historical-date endpoint at all). Converts `raw_matches` rows to the existing `NormalizedMatch` shape (`status="FINISHED"`, goals from the row), tagged `competition="SWE"` (reusing W64's tagging convention). `SwedenFixturesClient` continues to serve the future/fixtures-side branch unchanged — this only replaces the past/results-side branch, which the Odds API was never a correct source for anyway (W57 chose it only because football-data.org has zero Allsvenskan coverage on any plan; `raw_matches` was always the real store of Allsvenskan history — it's the ML engine's own ingestion target, per `src/ingestion/football_data/sweden_fetcher.py`).

### 3. Multi-competition precompute (`scripts/launch_sandbox.py`)

Extend `precompute_recommendations()`/`fetch_sandbox_fixtures()` to loop over `("E0", "SWE")`, mirroring the loop `scheduler_wiring.py` already has for the live EOD batch (W62) — same shape, applied to this one remaining single-competition call site. SWE's fixture discovery uses the new source from item 2 above (past dates only — a sandbox date is always in the past by construction, per the existing docstring's own reasoning for why E0's branch is unconditionally `get_results()`).

### 4. Scenario-testing runbook (new script)

A new `scripts/scenario_runbook.py` — distinct from `scripts/sandbox_runbook.py` (W31), which documents one specific single-date scenario walkthrough; this one walks a *range* of dates, a different responsibility. Following the W31/W44 precedent: takes a date range, walks a representative sample (not necessarily every single day — matching W20's existing "size for wall-clock feasibility" reasoning), and for each sampled date: launches the sandbox, runs `--precompute` for both leagues, and verifies (a) fixtures were found, (b) recommendations were generated via replay — not a live call, checked by asserting on `SnapshotStore`'s own mode/outcome rather than just "something rendered" — and (c) the Dashboard/Match Explorer/Match Analysis pages (this session's redesigned `AppShell`-based UI) render them correctly. Produces a written report in this project's existing runbook-documentation style.

### Data flow summary

No changes to `agent-snapshot`/`SnapshotStore`'s recording format. Two small backend additions (the corpus-lookup helper, the SWE historical source) plus a CLI-script extension and a new verification script — no new user-facing API surface.

### Testing

Each story gets its own unit tests per this project's established TDD convention (mocked `raw_matches`/corpus-directory fixtures for the lookup logic; the existing scheduler/EOD-batch test patterns for the precompute loop extension). The runbook (item 4) is the integration-level verification, run manually and documented, matching W23/W31/W44 rather than an automated CI suite (this project has none).
