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
| E0  | **None (no swept alpha satisfies the gate, coarse or fine)** | — | — | — | 0.2544 | Coarse grid: only alpha=1.0 reaches draw_recall in range (0.2345), but mean_predicted_draw_proba 0.3119 misses true_draw_rate by 0.0575 (over tolerance by 0.0075). **Finer grid (0.75-0.95, see `reports/result_3way_sample_weight_sweep_fine_e0_sp1.csv`) re-run 2026-08-20:** draw_recall never reaches the 0.20 floor at any of the 5 finer points -- 0.1655, 0.1655, 0.1793, 0.1931, 0.1931 for alpha=0.75/0.80/0.85/0.90/0.95 respectively. Closest is alpha=0.90/0.95 at 0.1931, still short of 0.20 by 0.0069. draw_recall then jumps to 0.2345 at alpha=1.0 (untested interval 0.95-1.0), where the proba-tolerance miss from the coarse grid recurs. No alpha in [0.75, 1.0) clears the recall floor; alpha=1.0 clears recall but misses proba tolerance. **Gate still not satisfied anywhere on the tested grid.** |
| SP1 | **None (no swept alpha satisfies the gate, coarse or fine)** | — | — | — | 0.2456 | Coarse grid: only alpha=1.0 reaches draw_recall in range (0.2929), but mean_predicted_draw_proba 0.3126 misses true_draw_rate by 0.0670 (over tolerance by 0.0170). **Finer grid (0.75-0.95) re-run 2026-08-20:** draw_recall clears 0.20 at alpha=0.85 (0.2429), 0.90 (0.2500), and 0.95 (0.2571) -- draw_precision at those points (0.3208, 0.3333, 0.3214) is not collapsed relative to the alpha=1.0 baseline (0.3417). But at every one of those three points, mean_predicted_draw_proba (0.3045, 0.3046, 0.3063) exceeds true_draw_rate + 0.05 (ceiling 0.2956) by 0.0089-0.0107. alpha=0.75/0.80 fail the recall floor instead (0.1571 each). **No alpha in the finer grid clears all three gates simultaneously -- SP1 has a structural tension where the recall floor and the proba ceiling can't both be satisfied on this grid.** |
| D1  | **0.7** | 0.2066 | 0.3472 | 0.3023 | 0.2636 | draw_recall 0.2066 is in [0.20, 0.30]. draw_precision 0.3472 is *higher* than alpha=1.0's own 0.3265 — no collapse. mean_predicted_draw_proba gap is \|0.3023-0.2636\|=0.0387, within ±0.05. This is the smallest (and only) swept alpha that clears all three gates for D1. |
| I1  | **1.0** | 0.24 | 0.3103 | 0.3126 | 0.2632 | Only alpha=1.0 reaches draw_recall in range (0.24, alpha=0.7 gives only 0.1467). Precision is the alpha=1.0 baseline itself, so trivially not collapsed. mean_predicted_draw_proba gap is \|0.3126-0.2632\|=0.0494 — inside ±0.05 by a thin margin (0.0006). Marginal pass; flagged for a possible finer sweep between 0.7 and 1.0 if more headroom is wanted, since nothing in that interval was tested. |
| F1  | **None -- explicitly deferred, not retrained/promoted this pass** | — | — | — | 0.2203 | draw_recall never reaches the 0.20 floor anywhere in the coarse sweep — the max across all 5 alphas is 0.1826 at alpha=1.0 (short by 0.0174). Even full balancing (alpha=1.0) under-recalls draws for F1, so this isn't a dampening problem for this league, and a finer alpha sweep alone would not fix it. Per the user's explicit decision, F1 is out of scope for this pass: left untouched (no model/training changes), flagged for a separate follow-up story to investigate its draw-signal/feature issue. |

Any league with no alpha satisfying the gate: **E0, SP1, F1**. None of the 5 coarse-grid alpha values (0.0/0.3/0.5/0.7/1.0) satisfy all three criteria simultaneously for these three leagues:
- E0 and SP1 both failed only on the `mean_predicted_draw_proba` vs `true_draw_rate` tolerance at the one alpha (1.0) where `draw_recall` landed in range — off by 0.0075 and 0.0170 respectively, with alpha=0.7's `draw_recall` falling just short of the 0.20 floor for both. This is what motivated the finer sweep below.
- F1 fails purely on `draw_recall`, which never reaches 0.20 even at full balancing (alpha=1.0 tops out at 0.1826). This points to a different underlying issue for F1 (e.g. insufficient draw signal in its features, or a class-imbalance/threshold problem beyond sample-weight dampening) rather than something a finer alpha sweep alone would fix. **F1 is explicitly deferred: left untouched this pass, no model/training changes, flagged for a separate follow-up story.**

D1 (alpha=0.7) and I1 (alpha=1.0, marginal) are the two leagues with a clean per-criteria decision from the coarse sweep.

**Finer E0/SP1 sweep (alpha in [0.75, 0.80, 0.85, 0.90, 0.95], run 2026-08-20 after user approval):**
Still no passing alpha for either league (see updated rows above and
`reports/result_3way_sample_weight_sweep_fine_e0_sp1.csv` for the full data).
E0's `draw_recall` stays below the 0.20 floor across the entire finer grid
(max 0.1931 at alpha=0.90/0.95) and only clears it at alpha=1.0, where the
original proba-tolerance miss recurs -- there is no tested alpha where both
gates hold at once. SP1's `draw_recall` clears 0.20 at alpha=0.85-0.95, but
at those same points `mean_predicted_draw_proba` overshoots the ±0.05
tolerance band (by 0.0089-0.0107) -- a structural tension on this grid
between the recall floor and the proba ceiling. Per the plan, this is
reported honestly rather than forcing a choice or expanding the grid further
unilaterally; both leagues need a controller decision on next steps (e.g. an
even finer sweep between 0.95 and 1.0 for E0, or revisiting the proba
tolerance/gate definition for SP1).
