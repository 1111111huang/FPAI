# result_3way sample_weight alpha choice (2026-08-20)

Full sweep data: `reports/result_3way_sample_weight_sweep.csv`

Gate applied per league (smallest alpha satisfying all three):
- `draw_recall` in `[0.20, 0.30]`
- `draw_precision` not collapsed relative to that league's own `alpha=1.0` value (no more than ~15 percentage points below it)
- `mean_predicted_draw_proba` within roughly ±0.05 of `true_draw_rate`

| League | Chosen alpha | draw_recall | draw_precision | mean_predicted_draw_proba | true_draw_rate | Notes |
|---|---|---|---|---|---|---|
| E0  | **None (no swept alpha satisfies the gate)** | — | — | — | 0.2544 | Only alpha=1.0 reaches draw_recall in range (0.2345); draw_precision there is the baseline itself (0.2742, no collapse). But mean_predicted_draw_proba 0.3119 misses true_draw_rate 0.2544 by 0.0575 — over the ±0.05 tolerance by 0.0075. No lower alpha even reaches the recall floor (alpha=0.7 → 0.1379). |
| SP1 | **None (no swept alpha satisfies the gate)** | — | — | — | 0.2456 | Only alpha=1.0 reaches draw_recall in range (0.2929, precision 0.3417 baseline). mean_predicted_draw_proba 0.3126 misses true_draw_rate 0.2456 by 0.0670 — over tolerance by 0.0170. alpha=0.7's draw_recall is 0.1786, just under the 0.20 floor. |
| D1  | **0.7** | 0.2066 | 0.3472 | 0.3023 | 0.2636 | draw_recall 0.2066 is in [0.20, 0.30]. draw_precision 0.3472 is *higher* than alpha=1.0's own 0.3265 — no collapse. mean_predicted_draw_proba gap is \|0.3023-0.2636\|=0.0387, within ±0.05. This is the smallest (and only) swept alpha that clears all three gates for D1. |
| I1  | **1.0** | 0.24 | 0.3103 | 0.3126 | 0.2632 | Only alpha=1.0 reaches draw_recall in range (0.24, alpha=0.7 gives only 0.1467). Precision is the alpha=1.0 baseline itself, so trivially not collapsed. mean_predicted_draw_proba gap is \|0.3126-0.2632\|=0.0494 — inside ±0.05 by a thin margin (0.0006). Marginal pass; flagged for a possible finer sweep between 0.7 and 1.0 if more headroom is wanted, since nothing in that interval was tested. |
| F1  | **None (no swept alpha satisfies the gate)** | — | — | — | 0.2203 | draw_recall never reaches the 0.20 floor anywhere in the sweep — the max across all 5 alphas is 0.1826 at alpha=1.0 (short by 0.0174). Even full balancing (alpha=1.0) under-recalls draws for F1, so this isn't a dampening problem for this league. |

Any league with no alpha satisfying the gate: **E0, SP1, F1**. None of the 5 swept alpha values (0.0/0.3/0.5/0.7/1.0) satisfy all three criteria simultaneously for these three leagues:
- E0 and SP1 both fail only on the `mean_predicted_draw_proba` vs `true_draw_rate` tolerance at the one alpha (1.0) where `draw_recall` lands in range — off by 0.0075 and 0.0170 respectively. A finer sweep near 1.0, or accepting a slightly wider proba tolerance for these two leagues specifically, are the two most plausible follow-ups.
- F1 fails purely on `draw_recall`, which never reaches 0.20 even at full balancing (alpha=1.0 tops out at 0.1826). This points to a different underlying issue for F1 (e.g. insufficient draw signal in its features, or a class-imbalance/threshold problem beyond sample-weight dampening) rather than something a finer alpha sweep alone would fix.

D1 (alpha=0.7) and I1 (alpha=1.0, marginal) are the two leagues with a clean per-criteria decision from this sweep.
