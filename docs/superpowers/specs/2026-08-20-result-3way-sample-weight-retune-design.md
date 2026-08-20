# `result_3way` Sample-Weight Retune: Fixing Draw Overcorrection

**Date:** 2026-08-20
**Status:** Approved, pre-implementation
**Covers:** new story to be appended to `documents/user_stories.md` (ML engine)

## Motivation

Investigated a user report that "draw" dominates the deployed recommendation page (Daily Edges). Root-caused to `result_3way`'s sample-weight rebalancing fix (commit `093919c`, 2026-08-14, `_compute_sample_weight()` in `src/models/model_manager.py`): sklearn's `compute_sample_weight("balanced", y)` fixed a real, severe pre-existing bug (draw recall 0.7% on E0, 2.1% on SP1 — the model almost never predicted draw), but overshot. Verified live against the current production models (`ForecastService.forecast_upcoming`) on three real, non-cold-start matchups: predicted P(draw) came out at 38–40% in all three — the single highest of the three outcomes each time, well above football's real ~24–27% draw base rate. Since bookmakers correctly price draw down in lopsided matches while the model's draw estimate barely moves, `value_edge = ml_probability - implied_probability` comes out strongly positive on draw for a large share of fixtures.

Also checked and ruled out: this is not the BUG-016-style integer-class-label-mismatch — verified the home/away/draw probability mapping is correct (a genuine big favorite came back with a correctly-high home-win probability). And checked whether the isotonic calibrator already fitted alongside these models (`models/result_3way_xgboost_v1_20260814.joblib.calibration.pkl` etc., produced by the same 08-14 commit but never actually loaded/applied by `ForecastService` — a separate, out-of-scope dead-code gap) would fix this on its own: applied it directly to the Nottingham v Leeds probabilities and draw stayed dominant (39.6% raw → 38.5% calibrated). Calibration can't invent discriminative signal the model doesn't have, so a real retrain is needed.

## Goals

- `_compute_sample_weight()` gains a dampening knob so class balancing can be partial instead of only fully-on (today, "balanced") or fully-off (pre-08-14, unweighted).
- Find a dampening value, per league, that lands `result_3way` draw recall in a healthy band (~20–30%) **and** keeps draw precision and mean predicted draw probability sane (close to each league's true draw base rate) — not just recall, which is how the 08-14 fix overshot unnoticed.
- Retrain and re-promote `result_3way` for all 5 leagues (E0, SP1, D1, I1, F1) under the new default, so the mechanism and its validation are consistent everywhere going forward, not patched only for the two currently-broken leagues.
- No regression to any other classification target (`btts`, `home_win`) or to `alpha=1.0` behavior for anything not re-promoted by this work.

## Non-goals

- Not wiring the existing (currently-unused) calibration sidecar into `ForecastService`'s serving path — confirmed separately that it wouldn't fix this issue anyway; flagged as its own follow-up, not fixed silently as a side effect here.
- Not adding a permanent CLI flag for the dampening exponent. This is a one-time tuning constant; once chosen it becomes the new hardcoded default so ordinary `train-forecast-suite` runs need no new flags.
- Not running a full agent-level `agent-backtest` (ROI/hit-rate) as a promotion gate this pass — the validation gate is pure ML metrics (recall/precision/mean predicted probability) on the existing held-out test split. (Note: this is a different concern from A57's leakage bug, which is about the LLM agent's web-search evidence corpus, not the ML model's own train/test data — A57 does not block this work.)
- Not touching `documents/agent_user_stories.md` A56/A57 (E0's skipped agent-backtest validation, E0's evidence-corpus leakage) — those remain open, separate stories.

## Architecture

### 1. Dampening knob (`src/models/model_manager.py`)

```python
def _compute_sample_weight(y, task_type, alpha: float = 1.0):
    if task_type == "regression":
        return None
    weights = compute_sample_weight("balanced", y)
    if alpha != 1.0:
        weights = weights ** alpha
    return weights
```

`alpha=1.0` (default) is byte-identical to current behavior. `alpha=0.0` recovers pre-08-14 unweighted training. Threaded as an optional kwarg through `ModelManager.train()`'s two call sites (currently lines ~603/642).

### 2. Tuning script (standalone, not a permanent CLI command)

For each of the 5 leagues' `result_3way` target, sweeps candidate `alpha` values (e.g. 0.0 / 0.3 / 0.5 / 0.7 / 1.0), retrains, and reports on the held-out test split:
- `log_loss` (existing primary selection metric, for continuity)
- per-class recall and precision via `sklearn.metrics.recall_score`/`precision_score` (precision_score already imported in `model_manager.py`; recall_score is a new import, same library)
- mean predicted P(draw) across the test set

No committed script from the original 08-14 fix exists to extend (that was itself a one-off, matching this pass's `alpha=0` baseline case) — this is new, throwaway tooling.

### 3. Promotion

Once `alpha` is chosen per the validation gate (one shared value across leagues unless the sweep clearly shows otherwise), retrain final `result_3way` models for all 5 leagues and update `config/model_selection.yaml` the same way the 08-14 promotion did (`model_path`, `metric_value`, `mlflow_run_id`, `selected_at`).

### 4. Testing

Extend `tests/test_model_manager_sample_weight.py`:
- `alpha=1.0` → output identical to today's `compute_sample_weight("balanced", y)` (regression safety)
- `alpha=0.0` → uniform (all-ones-equivalent) weights
- `alpha=0.5` → strictly between the two elementwise

No test pins a "correct" alpha value — that's an empirical result from the tuning script, not a unit-testable code property.

### 5. Docs

Per `CLAUDE.md`: append a new story to `documents/user_stories.md` (mirrors `US#167`'s style — real baseline numbers, real verification, not assumed) and mark it completed once done; note the finding + fix in the relevant technical doc.

## Validation gate (promotion criteria)

For each league's retrained `result_3way`:
- Draw recall in ~20–30%
- Draw precision not collapsed (catches the failure mode recall alone missed in 08-14)
- Mean predicted P(draw) across the held-out test set close to that league's true draw base rate (~24–27%, confirm actual per-league rate from data rather than assuming one number for all 5)
- `log_loss` reported for continuity, not used alone as a promotion gate (same mistake as 08-14)
