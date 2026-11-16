# Goal-Market Lambda Fallback — Design

## 1. Problem

Investigated 2026-11-12/13 via two generated pick-review artifacts comparing the
week of Nov 1 vs. the week of Nov 7: both weeks showed a strong "over" skew in
goal markets (`home_goals`/`away_goals`/`total_goals`), worse in Nov 7 (16/17
picks "over", 43.8% hit rate vs. Nov 1's 12/14, 66.7%).

Root-caused to `MKT_LAMBDA_HOME`/`MKT_LAMBDA_AWAY` going `NaN` whenever the
Asian Handicap line is missing at generation time
(`src/features/feature_factory.py` lines 840-852) — a mathematically necessary
`NaN` (there is no way to split `MKT_LAMBDA_TOTAL` between home/away without a
real handicap line), but one that then flows raw into XGBoost-family models per
[[BUG-096]]'s own fix. That fix's notes flag the residual risk directly: *"MKT_*
are ~100% populated in training, so the NaN fallback is not learned behavior."*

Confirmed empirically: when `MKT_LAMBDA_HOME`/`AWAY` is null, mean SHAP
contribution toward "over" is strongly positive (+0.31 to +0.48 across the two
sampled weeks) vs. near-zero when the feature is real — an artifact of
untrained missing-value routing, not real signal, inflating how many "over"
picks clear the value-edge bar. A guardrail that would have downgraded these
picks (`f33ab0f`) shipped, then was explicitly removed the next day by direct
user request, on the principle that a missing live price should only block the
market it prices, not every bet whose features happen to be placeholders.

Direction from the user: fix this with a better fallback *value* for the
feature, not by cancelling or downgrading the prediction.

## 2. Investigation findings that shaped this design

Checked whether `team_totals` odds (`home_goals_odds`/`away_goals_odds`,
already live-fetched per W199/[[BUG-104]]) could feed `MKT_LAMBDA_HOME`/`AWAY`
directly instead of a fallback — removing the AH dependency altogether rather
than papering over it.

Found: `feature_factory.py`'s training-time computation sources
`ah_line`/`over25_odds` purely from football-data.co.uk CSV columns
(`AHh`/`AvgAHH`/`AvgAHA`, or legacy `BbAHh`/`BbAvAHH`/`BbAvAHA`), which carry
**no team-goals column at all, ever** — that market was simply never priced by
this historical source. The only historical source for a `team_totals`
equivalent is OddsPapi via `backtest.py`'s `_build_match_info`, scoped to
**2026-01-01 onward only**, feeding agent-backtest/agent-train LLM replay — not
the ML `feature_store` the actual regressors are fit on.

**Conclusion**: using `team_totals` odds for this feature would be a
train/serve mismatch (a feature computed by a method the model never saw
during training) and needs a real backfill + retrain + promotion cycle to do
safely. Scoped **out** of this design as a separate, larger follow-up.

`MKT_IMPLIED_HOME`/`MKT_IMPLIED_AWAY` (from 1X2 odds), by contrast, were
confirmed non-null in every SHAP dump pulled during the investigation — even on
rows where `MKT_LAMBDA_HOME`/`AWAY` was null — and have equivalent coverage in
training (football-data.co.uk always carries 1X2 prices). This is the basis
for this design's fallback.

## 3. Goals

- When `MKT_AH_LINE` is NaN but `MKT_LAMBDA_TOTAL` is present, compute a real
  (non-NaN) `MKT_LAMBDA_HOME`/`MKT_LAMBDA_AWAY` split instead of propagating
  NaN.
- The fallback must be more informative than a blind 50/50 split — calibrated
  against the match's own 1X2-implied-probability gap, per league.
- No retrain of any existing classifier/regressor required.
- No change to live betting behavior beyond the feature value itself: no new
  downgrade, no new blocking, no change to value-edge thresholds or confidence
  levels.
- `MKT_POISSON_BTTS_PROB` (downstream of `lam_home`/`lam_away`) should stop
  going NaN as a natural consequence of the fix, not a separately-implemented
  change.
- The fallback's use should be observable after the fact via a diagnostic
  field, without feeding back into the model.

## 4. Non-goals

- Not replacing `MKT_LAMBDA_HOME`/`AWAY`'s primary AH-derived computation when
  AH *is* available — this only engages on the NaN branch.
- Not sourcing `team_totals` odds into this feature (see §2) — logged as a
  separate follow-up, not bundled here.
- Not adding a trained "this value is imputed" model feature — would require a
  retrain cycle; deferred as future work alongside the `team_totals` backfill.
- Not changing `MKT_LAMBDA_AH_DIFF` or `MKT_AH_LINE`'s own NaN behavior —
  neither has a sensible fallback without a real handicap line (the former
  needs the actual margin; the latter has no synthesizable proxy).
- Not touching training-time/replay feature computation — live-serving only,
  matching BUG-096's existing "live mode only, replay snapshot keys stay
  byte-identical" discipline.

## 5. The fallback computation

Trigger: `MKT_LAMBDA_TOTAL` is non-NaN **and** `MKT_AH_LINE` is NaN.

```
home_share = clip(
    intercept[league] + slope[league] * (MKT_IMPLIED_HOME - MKT_IMPLIED_AWAY),
    0.30, 0.75,
)
MKT_LAMBDA_HOME = MKT_LAMBDA_TOTAL * home_share
MKT_LAMBDA_AWAY = MKT_LAMBDA_TOTAL * (1 - home_share)
```

Fallback-of-fallback: if `MKT_IMPLIED_HOME`/`AWAY` are *also* NaN (1X2 missing
too — rare), use a flat per-league historical average `home_share` instead
(same clip bounds), rather than leaving NaN.

`MKT_POISSON_BTTS_PROB`'s existing formula is unchanged — it already
recomputes from `lam_home`/`lam_away`, so it stops going NaN automatically
once they do.

## 6. Calibration (per-league slope/intercept)

Offline, one-time (re-run only if deliberately refreshed): over historical
`feature_store` rows where `MKT_AH_LINE` is non-null, compute
`real_home_share = MKT_LAMBDA_HOME / MKT_LAMBDA_TOTAL` for each row, and fit a
per-league linear regression against
`(MKT_IMPLIED_HOME - MKT_IMPLIED_AWAY)`. One slope/intercept pair per league —
home-advantage strength varies meaningfully by league, so a single global fit
would discard real signal the model otherwise has access to.

New script (e.g. `scripts/calibrate_goal_market_fallback.py`, mirroring the
existing `scripts/check_market_feature_coverage.py` pattern), run against
production `feature_store`, output committed as a small static table — not
recomputed per-request, not auto-refreshed. Clip bounds `[0.30, 0.75]` applied
uniformly regardless of league, guarding against a pathological extrapolation
at extreme implied-probability gaps.

Where the coefficient table lives (a small YAML alongside
`config/competitions.yaml` vs. a dict literal in `feature_factory.py`) is an
implementation-time call, not a blocking design decision.

## 7. Observability

New `forecast_payload.data_quality.lambda_home_away_imputed: bool`, alongside
the existing `goal_market_features_available` — set `True` whenever this
fallback path fired for a given match. Purely informational: never read by any
scoring/staking/edge logic, only exposed for future analysis (e.g., re-running
this week's SHAP-null investigation directly instead of inferring it from
null-valued SHAP entries).

## 8. Testing

- Unit test on the fallback function: given a known `lambda_total`/implied-prob
  gap/league, output sums to `lambda_total` and is monotonically increasing in
  the implied-prob gap.
- Calibration-quality check: on held-out historical AH-present rows, the
  fallback's predicted split has lower mean absolute error vs. the real split
  than a flat 50/50 baseline, per league.
- Regression test: an input row with NaN `ah_line` but real
  `lambda_total`/`implied_home`/`implied_away` no longer produces NaN
  `MKT_LAMBDA_HOME`/`AWAY`/`POISSON_BTTS_PROB`.
- Post-ship validation: re-run the per-league real-edge check (US#193-style)
  restricted to picks where `lambda_home_away_imputed=true`, confirming hit
  rate moves closer to the calibrated split's implied probability than today's
  observed ~42-50% coin-flip-with-over-skew.

## 9. What's explicitly reused vs. new

| Piece | Status |
|---|---|
| `MKT_LAMBDA_TOTAL` computation (`_solve_lambda`) | Reused as-is |
| `MKT_IMPLIED_HOME`/`AWAY` | Reused as-is (new consumer) |
| `MKT_POISSON_BTTS_PROB` formula | Reused as-is (unblocked as a side effect) |
| `data_quality.goal_market_features_available` | Reused by precedent (new sibling field) |
| `feature_factory.py`'s `_compute_mkt_features_from_odds` | Modified — NaN branch replaced |
| Per-league slope/intercept calibration | New, offline, one-time |
| `scripts/calibrate_goal_market_fallback.py` | New |
| `data_quality.lambda_home_away_imputed` | New |

## 10. Self-review notes

- **Placeholder scan**: no TBD left in except the explicitly-deferred "where
  the coefficient table lives" call, flagged as implementation-time, not a
  missing requirement.
- **Consistency**: the trigger condition in §5 matches the non-goal in §4 (the
  AH-available path is untouched); the observability field name matches
  between §7 and §9's table.
- **Scope**: a single cohesive change (one feature-factory function, one
  calibration script, one diagnostic field), appropriately sized for one
  implementation plan — not decomposed further.
- **Follow-ups logged, not bundled**: (a) a `team_totals`-based `feature_store`
  backfill + retrain + promotion review, (b) a trained "imputed" indicator
  feature — both explicitly out of scope here (§4), left as future work.
