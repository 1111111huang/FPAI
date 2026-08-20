# result_3way retune — before/after verification against original symptom

Task 8 of the plan (docs/superpowers/plans/2026-08-20-result-3way-sample-weight-retune.md).
Re-ran the exact three reproduction cases from the original investigation, in a fresh
`./venv/bin/python -c "..."` process, against the models promoted in Task 7
(`config/model_selection.yaml`, commit `3520a25`): E0 alpha=0.90, SP1 alpha=0.85,
D1 alpha=0.7 (not exercised by this reproduction set), I1 alpha=1.0/unchanged (not
exercised either — see design spec for why).

Confirmed via `diagnostics.target_versions.result_3way.artifact` on a live call that the
new artifact (`result_3way_xgboost_v1_20260820.joblib`) is what actually served these
predictions, not a stale cached model.

## Results

| Match | League | Odds (H/D/A) | Market-implied (H/D/A) | BEFORE (H/D/A) | AFTER (H/D/A) | Draw still highest? |
|---|---|---|---|---|---|---|
| Nottingham v Leeds United | E0 | 2.60 / 3.30 / 2.75 | 36.6% / 28.8% / 34.6% | 26.8% / **39.6%** / 33.6% | 27.7% / **41.4%** / 30.9% | Yes — and higher than before |
| Real Betis v Real Sociedad | SP1 | 2.30 / 3.30 / 3.10 | 41.0% / 28.6% / 30.4% | 31.3% / **40.2%** / 28.5% | 30.0% / **41.4%** / 28.6% | Yes — and higher than before |
| Ipswich Town v Sunderland | E0 | 2.70 / 3.30 / 2.60 | 35.0% / 28.6% / 36.4% | 30.8% / **38.5%** / 30.6% | 30.8% / **36.7%** / 32.4% | Yes — modestly lower than before, still highest |

Raw AFTER output (probabilities sum to 1.0 in all three, no exceptions):

```
Nottingham v Leeds United -> {'away': 0.309736, 'draw': 0.41359, 'home': 0.276675}
Real Betis v Real Sociedad -> {'away': 0.285765, 'draw': 0.414135, 'home': 0.3001}
Ipswich Town v Sunderland -> {'away': 0.323871, 'draw': 0.367651, 'home': 0.308479}
```

## Assessment

The fix did **not** resolve the originally-diagnosed symptom for these three live cases.
Draw is still the single highest-probability outcome in all three matchups after the
retune — the same as before. In two of the three (Nottingham v Leeds, Real Betis v Real
Sociedad) draw's probability is actually *higher* than it was before the retune (39.6% →
41.4%, and 40.2% → 41.4% respectively). Only Ipswich v Sunderland shows a modest
improvement (38.5% → 36.7%), and even there draw remains dominant and well above the
market-implied draw probability (~28.6%) and the ~24-27% real-world base rate.

Home/away plausibility relative to market odds is also not fixed:

- **Nottingham v Leeds** (home is the market's modest favorite, 36.6% implied vs 34.6%
  away): the model's home probability (27.7%) is still *below* both draw (41.4%) and
  barely above where it started (26.8%). Home does not sit above draw, unlike what the
  task's expected-outcome language anticipated.
- **Real Betis v Real Sociedad** (home is the clear market favorite at 41.0% implied):
  model home probability actually dropped slightly (31.3% → 30.0%) and remains far
  below draw (41.4%).
- **Ipswich v Sunderland** (away is the market's modest favorite, 36.4% implied vs 35.0%
  home): model's away probability rose above home (32.4% vs 30.8%), which is directionally
  more consistent with the market than before (before: 30.6% vs 30.8%, roughly tied) — but
  draw is still highest at 36.7%.

**Bottom line: the E0/SP1 alpha retune (0.90 / 0.85) did not fix the live over-prediction
of draw for these three specific matchups.** It made two of the three worse and only
marginally improved the third. This does not necessarily mean the retune was worthless in
aggregate (Task 7's promotion was based on held-out test-set metrics, not these three live
spot-checks), but on the exact symptom this plan set out to fix, the symptom persists.
