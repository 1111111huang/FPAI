# FPAI Systematic Practices Catalog

This document enumerates recurring operational practices this project has learned the hard way — real incidents, not generic ML-ops advice. It's the input to a later pass that turns each into a `.claude/skills/` skill (mirroring the existing `adding-a-league` and `agent-train-experiment` skills), so these steps stop depending on someone remembering them.

Two skills already exist and are not re-described here: `adding-a-league` (competitions/leagues) and `agent-train-experiment` (sampling/training-experiment runs).

Compiled 2026-09-22 via a full sweep of `documents/bugs.md`, `documents/user_stories.md`, `documents/agent_user_stories.md`, `documents/app_user_stories.md`, `documents/experiment_log.md`, and `documents/agent_techspec.md`. Every practice below cites the real incident that revealed it.

---

## A. ML model & feature lifecycle

### A1. Real-edge validation before promoting a classifier
- **Trigger:** promoting `result_3way`, `btts`, or `total_goals` — any target with a real market-odds column to check against.
- **Why:** `US#172` shipped a `result_3way` retrain that cleanly passed the offline log_loss/recall gate, then measurably worsened live draw predictions (`US#173`) — nothing in the pipeline caught it pre-promotion. Recurred for `btts:no` (`US#192`) and `total_goals` (`US#193`).
- **Steps:** run `scripts/validate_real_edge.py` (qualifying bet = `predicted_prob - implied_prob >= 0.05`) using a genuinely held-out fit (`ModelManager.train()` directly — **not** a `refit_on_full_data` artifact, which leaked once already, `US#203`'s corrected numbers). Sweep a candidate parameter on one half of real-odds data, confirm on the untouched other half. A negative/mixed result is a legitimate documented outcome (`US#200`/`US#202`), not something to hide.
- **Status quo:** exists only as a manual script; not wired into `select-best-models` (`US#205`, logged this session, still `future`).

### A2. New feature families need both the batch pipeline AND the live inference builder updated
- **Trigger:** adding engineered feature families to the batch `feature_store` builder.
- **Why:** `BUG-012` — 20 new columns were added to the batch pipeline but not `FeatureFactory.build_for_match()` (live path), so `forecast_league` failed 100% of the time, silently masked by a fallback to `forecast_international`.
- **Steps:** update both builders together; verify `feature_subset` is set on promotion (now auto-backfilled from `.metadata.json`); verify via a real `build_for_match()` call, not just unit tests, before promoting.
- **Status quo:** fix + rationale live only in `BUG-012`'s writeup.

### A3. New feature groups must be gated per-competition — verify the gate actually enforces
- **Trigger:** adding a feature family gated by per-competition data availability (player stats, cards, shots).
- **Why:** repeatedly found "decorative" gates — `OPP_ADJ_*`/`H2H_*` flowed through completely ungated pre-`US#133`; `DIS` (cards) was listed in `enabled_feature_groups` with no code path actually checking it. Cold-start imputation's per-competition column mean means an ungated NaN family can silently cross-contaminate from another competition's mean.
- **Steps:** add a real branch to `resolve_feature_group_tag()`, not just a config entry; verify end-to-end via `ModelManager._load_selected_features()` against a real competition; check whether cold-start imputation would cross-contaminate before assuming a bare tag is sufficient.
- **Status quo:** partially covered by `adding-a-league`'s `enabled_feature_groups` copy-and-prune step; the broader "does this gate actually enforce" verification is not.

### A4. MLflow runs need the full required-tag set, `context` included
- **Trigger:** any change to the sweep/training runner or a new experiment config template.
- **Why:** the June 2026 `OptunaRunner` bug — no `context` tag meant `select-best-models`' `tags.context='league'` filter made genuinely-better runs invisible; worse runs got promoted for an unknown period (fixed commit `c365089`).
- **Steps:** every run-producing code path sets `target`, `task_type`, `model_family`, `feature_schema_version`, `split_policy`, `league`, `sweep_stage`, `experiment_version`, `context`. When a "better" run doesn't get promoted, check tags before assuming the comparison logic is broken.
- **Status quo:** documented as a standing note in `experiment_log.md` + `FRAI_TECHSPEC.md`, not enforced anywhere in code/CI.

### A5. Hand-edited `model_selection.yaml` promotions bypass CLI safety checks
- **Trigger:** promoting a composite/novel model type, or overriding the automated gate on real-edge evidence.
- **Why:** hand-edits (used repeatedly — `US#183`, `US#199`, `US#203`) skip `feature_subset` backfill; `US#203` had to manually re-sync `feature_subset` for 4 leagues afterward because they still carried a stale 147-feature list.
- **Steps:** after any hand-edit, manually verify/sync `feature_subset` from the new artifact's own `.metadata.json`, and verify live end-to-end via `ForecastService.forecast_upcoming` — every documented hand-promotion in this project did this rather than trusting the YAML edit alone.
- **Status quo:** no checklist; scattered across completion notes.

### A6. Recency/time-decay changes need the leak-free real-edge check, and logloss-vs-real-edge disagreement is a per-league decision, not a bug
- **Trigger:** setting/sweeping `time_decay_half_life_days`.
- **Why:** `US#203` found the automated logloss gate and real-edge check disagreeing for 4/5 leagues — recency weighting traded old-history fit for current-season fit, worsening logloss while improving real edge. Also watch `BUG-068`: `refit_on_full_data` + `time_decay_half_life_days` together crash on a date-vector shape mismatch unless `self.full_data_dates` is threaded through.
- **Steps:** compute both the standard promotion gate and leak-free real edge; treat disagreement as an explicit per-league policy decision, not something to paper over.
- **Status quo:** described only in `US#203`'s completion note and `BUG-068`.

### A7. Isotonic calibrators on small samples silently collapse to plateaus — the in-sample self-check can't catch it
- **Trigger:** any classifier's calibrator being fit/refit.
- **Why:** `BUG-066` — calibrators for `result_3way`/`btts` across multiple leagues mapped whole probability bands to one constant; the in-sample `ll_before/ll_after` gate showed a false "5% better" on a calibrator later proven 13% worse out-of-sample. Real cost: `btts:no` hit 0/11 in a live window with the LLM citing match-specific "evidence" for a probability with zero relation to the match.
- **Steps:** isotonic only above `_MIN_ISOTONIC_SAMPLES = 1000`, else sigmoid/Platt fallback; `_fit_and_save_calibrator` must be verified against genuinely held-out `X_test`/`y_test`, never the in-sample check alone; disable a broken calibrator by renaming its sidecar (`.broken_bug066`), not deleting, so the loader cleanly no-ops.
- **Status quo:** narrated only in `BUG-066`; a full re-fit of the currently-disabled calibrators is still open.

### A8. Model artifact filenames need a competition-id component
- **Trigger:** training the same target/model-type for two competitions on the same calendar day.
- **Why:** `BUG-017` — identical filenames for same-day `international` and `SWE` training silently clobbered Sweden's just-written artifacts; the existing stale-path guard didn't catch it since the YAML still pointed at "a real file."
- **Steps:** use `build_artifact_filename(target_name, competition_id, model_prefix, date_tag)`.
- **Status quo:** fixed as a helper function; no separate checklist.

---

## B. Agent & prompt lifecycle

### B1. Numeric thresholds in prompts must be templated from `AgentConfig`, never hardcoded prose
- **Trigger:** any prompt edit that references a numeric config value (odds bounds, edge thresholds).
- **Why:** `BUG-054` — all 4 posture files stayed in sync *with each other* but drifted from `agent_config.yaml`'s real enforced thresholds (a second, stale copy hardcoded into all 4 simultaneously) — the agent argued for a bet its own code then refused.
- **Steps:** template numeric values (`{{MIN_VALUE_EDGE}}`-style placeholders substituted from `AgentConfig` at load time) rather than writing the number into prose; prose-only edits still go to all 4 files identically. `tests/test_agent_prompt_thresholds.py` is the existing pattern (asserts no unsubstituted placeholder in any of the 4 files).
- **Status quo:** enforced by the templating mechanism for the numeric subset; no general doc for the "always all 4 files" convention itself, though no incident has ever shown a file being outright forgotten (3-of-4).

### B2. What a new agent tool means for the snapshot corpus — corrected understanding
- **Trigger:** adding a new agent tool (e.g. `A126`'s `get_player_rating`).
- **Why:** `agent-snapshot` genuinely never calls the LLM (`A97`, confirmed live with every provider's API key set invalid — zero errors across 10 matches) — it only freezes deterministic tool-node output (`resolve_competition`, `research_node`/`web_search`, `forecast_node`). The LLM (and whatever tools it's allowed to call) only runs live, fresh, every time `agent-train`/`agent-backtest` replay a snapshot. In `SnapshotStore`'s replay mode, a tool call with no matching frozen response raises `SnapshotMissingError` immediately (no silent fallback) — caught per-match, printed as a skip, never silently averaged away.
- **Steps:** a brand-new tool doesn't retroactively break old snapshots (it just can't be exercised by them — replay either skips those matches loudly or the old corpus simply never tests the new tool). Only re-record if you specifically want old matches to exercise the new tool's behavior.
- **This corrects your original framing** — of your three examples (model / prompt / market), a **prompt-only change does not require re-recording at all**: replaying old, unchanged snapshots against a new prompt is the intended way to validate a prompt change (same inputs, varying prompt). The two things that genuinely do require re-recording are:
  - **An ML model promotion** — `forecast_league`/`forecast_international`'s frozen output *is* the old model's predictions; an old snapshot keeps replaying stale probabilities forever unless re-recorded. Currently nothing tracks or reminds you of this for the training/backtest corpus specifically (`BUG-036`'s fingerprint fix only covers the *live-serving sandbox replay cache*, not this corpus — this exact gap is `A124`, already logged this session).
  - **A new market or new tool**, when you want old matches to actually exercise it — old snapshots simply never fetched that market's odds or called that tool at all.
- **Status quo:** not documented anywhere as a decision table; `BUG-036` flags the model-swap case as "worth a dedicated future story if this recurs," still open.

### B3. New agent tool checklist (reconstructed from precedent, never written down)
- **Trigger:** adding any new callable tool to the agent.
- **Steps traced from `A04`/`A126`:** (1) tool function + clear description; (2) route through `SnapshotStore.wrap` if it should be replay-frozen; (3) prompt instructions in all 4 posture files (`B1`); (4) if it changes candidate shape, review `schema.py`'s downgrade-rule chain (the established "downgrade-only, never auto-promote" convention, `A88`-`A91`); (5) tests.
- **Status quo:** no checklist doc exists; this is inferred from precedent, which is itself the risk `A126` will be the next live test of.

### B4. New market checklist (the project's most-repeated, most-often-incomplete practice)
- **Trigger:** adding a new recommendable market (proven twice — `total_corners`/`A101`, then `home_goals`/`away_goals`/`W199`).
- **Why it matters — the same bug class recurred:** `A101` (total_corners) found `app/backend/recommendations.py`'s *separate, duplicated* `Literal` types (mirroring `schema.py` but not importing it) had been missed in the first pass, requiring an explicit "align live too" follow-up the same day. **Twelve days later, the identical gap recurred for `home_goals`/`away_goals` (`W239`)** — silently dropping every candidate as "malformed data" for a full day before being caught live. No checklist existed to prevent the repeat.
- **Full touch-list, reconstructed from the `home_goals`/`away_goals` rollout (17 commits):**
  1. `src/agent/schema.py` — `market`/`selection` `Literal` types, `_CONDITIONAL_ELIGIBLE_MARKETS`.
  2. `app/backend/recommendations.py` — its own **duplicated** `Literal` mirror (the recurring miss above) + `_SHAP_POSITIVE_SELECTION`.
  3. `app/backend/eod_batch.py`'s `add_secondary_odds()` — its own cache-reuse gate, keyed on value *truthiness* not key presence (`W240`: a `None` cached before a book posted a line replayed forever, same bug class already present twice before this market and fixed for all three at once).
  4. `app/backend/scheduler_wiring.py`'s wrapper classes (`PersistingOddsClient`/`FallbackOddsClient`) — must forward new params through (same bug class as `W99`, recurred once inside this same story before shipping).
  5. `src/agent/market_resolution.py` — an explicit new resolver branch per market, plus `RESOLVABLE_MARKETS` and `build_actual_outcome()` param threading (not automatic).
  6. `app/frontend/components/MatchUI.tsx` — a third, hand-ported TypeScript mirror of the same resolution logic (`app/frontend/lib/types.ts` itself is safe — `market` is a plain `string` there, not a `Literal`).
  7. Prompt files, all 4 postures — the market isn't recommendable at all until the LLM is told it exists (JSON template enum line, conditional-eligibility sentence, and often a new evidence-priority paragraph).
  8. `config/agent_config*.yaml`, all 6 variants (including backtest).
  9. `src/agent/backtest.py`'s `_build_match_info()` — a new historical-odds source (OddsPapi market-ID lookup) if the market isn't in `raw_matches` already. `A73` found populating `match_info` alone insufficient once already — `BacktestHarness.load_matches()`'s own SQL SELECT also needs the new columns, or a 99-match run silently runs on incomplete data despite passing unit tests.
- **Status quo:** no `.claude/skills/adding-a-market/SKILL.md` exists (confirmed distinct from `adding-a-league`, which is genuinely league/competition-specific with no overlap). This is the strongest single candidate for a new skill — the exact bug class has now shipped to production twice.

### B5. `agent_config_hash` must be updated the same commit as any new `AgentConfig` field
- **Trigger:** adding a new tunable field to `AgentConfig`.
- **Why:** the hash only covers tunable config fields, never pipeline/graph code — an admitted, standing gap. Real incidents: 33/136 stale cache rows after a restructure (`A31`); `max_odds_threshold` omitted from the hash for a period, letting two differing configs collide on the same cache filename.
- **Steps:** add the new field into `compute_agent_config_hash()` in the same commit it's added to `AgentConfig` (done correctly for `min/max_conditional_odds_threshold`).
- **Status quo:** a fix pattern followed since `A66`/`A84`, not an enforced test/checklist.

---

## C. Deployment & production operations

### C1. A deployed fix needs `force=true` regeneration, not just a redeploy
- **Trigger:** a model/calibrator/config fix lands on `main` and deploys.
- **Why:** `run_eod_batch()`'s `already_fresh()` gate skips regeneration whenever cached odds haven't moved — a deployed fix can silently do nothing until odds move or someone forces it (`W204`).
- **Steps:** `POST /api/admin/pregenerate-recommendations?force=true` (scope with `date_from`/`date_to`/`league`). `force=true` skips the odds-freshness check but never bypasses `has_kicked_off()` — live/finished matches are never touched regardless.
- **Status quo:** scattered across `W204`, `BUG-067`, `agent_techspec.md` §27.3/27.5; already captured in this session's own memory note (`fpai-prod-regen-trigger`) — a strong skill candidate to formalize.

### C2. Boot-time background work needs reduced concurrency, not the nightly job's default
- **Trigger:** every process boot with `ENABLE_SCHEDULER=1` (persists across redeploys).
- **Why:** `BUG-045` — boot-time pregenerate stacking the nightly job's default `concurrency=5` directly on fresh, unsettled import-time memory caused an OOM kill Railway's own periodic memory graph never showed (only the crash-notification email caught it); because pregenerate re-fires on every boot, one OOM became an infinite crash loop.
- **Steps:** boot-time work uses a separate, lower concurrency constant (`_PREGENERATE_DEFAULT_CONCURRENCY = 2`, vs. the nightly job's untouched default of 5); scope the boot-time lookahead window to only what the frontend actually needs (`W165` cut 5→3 days, cutting call volume ~79%).
- **Status quo:** `BUG-045`'s writeup + regression test only; not generalized to "any new boot-time work" anywhere.

### C3. Don't manually re-trigger a job while its own rate-limited background pass might still be running
- **Trigger:** manually invoking `settle-open`, `/api/fixtures`, or similar diagnostic triggers while investigating an issue.
- **Why:** `BUG-058`'s own notes admit the investigating session's repeated manual triggers "very likely compounded" the football-data.org 429s it was diagnosing. Two other independent rate limits exist with their own fixes: The Odds API (`BUG-056`, key fallback) and OddsPapi (`W243`, a global `_throttle_oddspapi_request()` before every call, since concurrent per-fixture calls fired with zero delay).
- **Steps:** before manually re-triggering a data-fetching endpoint during debugging, check whether a scheduled job touching the same rate-limited vendor might already be mid-run.
- **Status quo:** **no project-wide written convention exists** — each incident was fixed independently at the client layer. This is a real gap and a good skill candidate (a short "before you manually re-trigger X, check Y" reference).

### C4. Back up the DB file before a risky migration/cleanup
- **Trigger:** any manual bulk-delete or migration against a live DuckDB/SQLite file.
- **Why:** the only instance of this in the project's history is a good one — before deleting 1,150 stale `agent_lessons` rows, the file was copied to `data/fpai_core.db.pre-lessons-cleanup-backup-20260901` first.
- **Steps:** copy the file with a `.pre-<change>-backup-<date>` suffix before the operation; `POST /api/admin/restore-database` exists for restoring `fpai_core`/`recommendation_cache` from a URL if needed, but doesn't cover `agent_lessons` cleanly (a full-file restore would clobber production's fresher `raw_matches`/`feature_store` in the same file).
- **Status quo:** a one-off filename convention, not a script or checklist.

### C5. No recurring post-deploy smoke test exists
- **Trigger:** every ordinary deploy (not just pre-launch).
- **Why:** `documents/prelaunch_smoke_test_checklist.md` exists but is a one-time, pre-launch manual E2E checklist last run 2026-07-12 — not a recurring ritual. `sandbox_testing_runbook.md` is dev/sandbox-focused, not production verification.
- **Status quo:** genuine gap, no practice to extract yet — flagged here as a candidate to *design*, not one to merely formalize.

---

## Candidate skills to build next

Roughly in order of how many real incidents each would have prevented:

1. **`adding-a-market`** — B4's full touch-list. Two production incidents from the identical gap (app-backend `Literal` mirror forgotten twice).
2. **`promoting-a-model`** — A1, A5, A6, A8 combined: real-edge validation, `feature_subset` sync, logloss-vs-real-edge policy call, filename collision avoidance.
3. **`prod-regen-and-rate-limits`** — C1 + C3: the force-regen trigger, plus the "don't stack manual triggers on a running rate-limited job" convention.
4. **`adding-an-agent-tool`** — B2 (corrected snapshot-recording rule) + B3's reconstructed checklist.
5. **`boot-time-background-work`** — C2, narrower but has caused a real production outage (`BUG-045`).

Everything else in this catalog (A2/A3/A4/A7, B1/B5, C4) is either a narrower one-off gotcha better folded as a "watch out for" note inside one of the above, or (C5) not yet a practice to formalize at all — a gap to design first.
