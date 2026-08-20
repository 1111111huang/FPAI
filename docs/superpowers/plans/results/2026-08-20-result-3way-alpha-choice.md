# result_3way sample_weight alpha choice (2026-08-20)

Full sweep data: `reports/result_3way_sample_weight_sweep.csv`

**Update:** the initial 5-point grid (`[0.0, 0.3, 0.5, 0.7, 1.0]`) left E0 and SP1
failing the gate at every point, missing only on `mean_predicted_draw_proba`
tolerance at alpha=1.0 (the only point where `draw_recall` cleared 0.20) while
alpha=0.7's `draw_recall` fell just short of the floor -- suggesting the right
value sat between 0.7 and 1.0. Per the user's approval, a finer grid
(`[0.75, 0.80, 0.85, 0.90, 0.95]`) was run for E0 and SP1 only (F1 and the
already-passing D1/I1 were left untouched). Data:
`reports/result_3way_sample_weight_sweep_fine_e0_sp1.csv`. **Result: the finer
grid still does not produce a passing alpha for either league** -- see updated
rows and notes below. This is reported as-is per the plan's instruction not to
force a choice or expand the grid further unilaterally.

Gate applied per league (smallest alpha satisfying all three):
- `draw_recall` in `[0.20, 0.30]`
- `draw_precision` not collapsed relative to that league's own `alpha=1.0` value (no more than ~15 percentage points below it)
- `mean_predicted_draw_proba` within roughly ±0.05 of `true_draw_rate`

| League | Chosen alpha | draw_recall | draw_precision | mean_predicted_draw_proba | true_draw_rate | Notes |
|---|---|---|---|---|---|---|
| E0  | **0.90 (accepted as a close-enough pass)** | 0.1931 | 0.2569 | 0.3054 | 0.2544 | Coarse and finer grids never land exactly inside the gate box (see full history below) -- best point is alpha=0.90: draw_recall 0.1931 (need >=0.20, short by 0.0069) and mean_predicted_draw_proba gap 0.0510 (need <=0.05, over by 0.0010). Both misses are under 1 percentage point -- smaller than what sampling noise from a single 80/20 split would plausibly explain. **User decision 2026-08-20: accepted as close enough**, since it's also a real, large improvement over today's live alpha=1.0 (proba gap 0.0575 -> 0.0510). |
| SP1 | **0.85 (accepted as a close-enough pass)** | 0.2429 | 0.3208 | 0.3045 | 0.2456 | Coarse grid only reached the recall floor at alpha=1.0, missing proba tolerance by 0.0170. Finer grid: draw_recall clears 0.20 comfortably at alpha=0.85 (0.2429) with draw_precision 0.3208 (not collapsed vs. the alpha=1.0 baseline 0.3417); mean_predicted_draw_proba gap is 0.0589 (need <=0.05, over by 0.0089). **User decision 2026-08-20: accepted as close enough** -- recall/precision are both comfortably healthy, and the proba-gap miss is under 1 percentage point. |
| D1  | **0.7** | 0.2066 | 0.3472 | 0.3023 | 0.2636 | draw_recall 0.2066 is in [0.20, 0.30]. draw_precision 0.3472 is *higher* than alpha=1.0's own 0.3265 — no collapse. mean_predicted_draw_proba gap is \|0.3023-0.2636\|=0.0387, within ±0.05. This is the smallest (and only) swept alpha that clears all three gates for D1. |
| I1  | **1.0** | 0.24 | 0.3103 | 0.3126 | 0.2632 | Only alpha=1.0 reaches draw_recall in range (0.24, alpha=0.7 gives only 0.1467). Precision is the alpha=1.0 baseline itself, so trivially not collapsed. mean_predicted_draw_proba gap is \|0.3126-0.2632\|=0.0494 — inside ±0.05 by a thin margin (0.0006). Marginal pass; flagged for a possible finer sweep between 0.7 and 1.0 if more headroom is wanted, since nothing in that interval was tested. |
| F1  | **None -- explicitly deferred, not retrained/promoted this pass** | — | — | — | 0.2203 | draw_recall never reaches the 0.20 floor anywhere in the coarse sweep — the max across all 5 alphas is 0.1826 at alpha=1.0 (short by 0.0174). Even full balancing (alpha=1.0) under-recalls draws for F1, so this isn't a dampening problem for this league, and a finer alpha sweep alone would not fix it. Per the user's explicit decision, F1 is out of scope for this pass: left untouched (no model/training changes), flagged for a separate follow-up story to investigate its draw-signal/feature issue. |

League with no alpha satisfying the gate exactly, deferred rather than force-fit: **F1 only.** F1 fails purely on `draw_recall`, which never reaches 0.20 even at full balancing (alpha=1.0 tops out at 0.1826). This points to a different underlying issue for F1 (e.g. insufficient draw signal in its features, or a class-imbalance/threshold problem beyond sample-weight dampening) rather than something a finer alpha sweep alone would fix. **F1 is explicitly deferred: left untouched this pass, no model/training changes, flagged for a separate follow-up story.**

D1 (alpha=0.7) and I1 (alpha=1.0, marginal) had a clean per-criteria decision from the coarse sweep. E0 (alpha=0.90) and SP1 (alpha=0.85) needed a finer sweep and a final user-approved close-enough-pass call (see below) -- both still get retrained/promoted in Task 6/7.

**Final per-league alphas going into Task 6:** E0=0.90, SP1=0.85, D1=0.7, I1=1.0, F1=not retrained (deferred).

**Finer E0/SP1 sweep (alpha in [0.75, 0.80, 0.85, 0.90, 0.95], run 2026-08-20 after user approval):**
No alpha lands exactly inside the gate box for either league (see updated
rows above and `reports/result_3way_sample_weight_sweep_fine_e0_sp1.csv` for
the full data). E0's best point (alpha=0.90) misses the recall floor by
0.0069 and the proba tolerance by 0.0010; SP1's best point (alpha=0.85)
clears recall/precision comfortably but misses proba tolerance by 0.0089.

**Final decision (2026-08-20, user-approved):** both leagues' near-misses are
under 1 percentage point -- smaller than what sampling noise from a single
80/20 split would plausibly explain -- and both represent a real, large
improvement over today's live alpha=1.0 behavior. Accepted as close-enough
passes rather than running a further, even-finer sweep: **E0 -> alpha=0.90,
SP1 -> alpha=0.85.** These are the values used in Task 6's retrain.
