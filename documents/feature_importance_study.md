# Feature Importance Study — Phase 7 Models

**Date**: 2026-06-07  
**Models**: Phase 7 broad-sweep XGBoost (288 combos × 8 targets, 133 features)  
**Source notebook**: `exploration/feature_importance_analysis.ipynb`

---

## 1. Overview

This study analyses what each of the eight FPAI prediction models actually relies on when making predictions, using SHAP (SHapley Additive exPlanations) computed on the held-out test split (2024-04-03 → present, n ≈ 760 matches).

SHAP is used instead of XGBoost's native gain importance because gain is biased toward high-cardinality features and toward features used early in trees. Mean |SHAP value| across the test set reflects the true average impact on each model output, regardless of how many splits a feature appears in.

**Scope:** The analysis covers 9 sections — per-target top-20 bar charts, beeswarm plots (directional signal), per-sample waterfall analysis (best-to-worst predictions), group contribution stacked bars, cross-target importance heatmap, universality analysis, Phase 7 new feature signal heatmap, and US#64/US#68 recommendations.

---

## 2. Methodology

| Step | Detail |
| :--- | :--- |
| Model selection | Best run per target from its per-target MLflow broad-sweep experiment, ranked by primary metric |
| Retrain | One XGBoost model retrained with best hyperparams on the full training split |
| SHAP | `shap.TreeExplainer` on the held-out test split; multiclass returns shape `(n, p, 3)` |
| Importance metric | Mean \|SHAP value\| per feature across all test rows, averaged across classes for multiclass |
| Training cutoff | 2024-04-03 — rows after this date are strictly held out |

---

## 3. Best Model per Target

| Target | Primary Metric | Best Value | PRD Target | PRD Met | Best Hyperparams |
| :--- | :--- | :--- | :--- | :--- | :--- |
| `result_3way` | Log-loss | 1.0068 | — | — | n_est=300, depth=2, lr=0.01, sub=0.7, col=0.7 |
| `btts` | Log-loss | 0.6847 | LL ≤ 0.68 | No (+0.005) | n_est=300, depth=2, lr=0.01, sub=0.7, col=0.7 |
| `home_goals` | MAE | 0.9400 | MAE < 0.50 | No (+0.440) | n_est=300, depth=2, lr=0.01, sub=0.7, col=0.7 |
| `away_goals` | MAE | 0.8267 | MAE < 0.50 | No (+0.327) | n_est=300, depth=2, lr=0.01, sub=0.7, col=0.7 |
| `total_goals` | MAE | 1.2824 | MAE < 0.75 | No (+0.532) | n_est=300, depth=2, lr=0.01, sub=0.7, col=0.7 |
| `home_corners` | MAE | 2.1393 | MAE < 1.50 | No (+0.639) | n_est=300, depth=2, lr=0.01, sub=0.7, col=0.7 |
| `away_corners` | MAE | 2.0814 | MAE < 1.50 | No (+0.581) | n_est=800, depth=3, lr=0.10, sub=0.7, col=0.9 |
| `total_corners` | MAE | 2.6629 | MAE < 1.50 | No (+1.163) | n_est=300, depth=2, lr=0.01, sub=0.7, col=0.7 |

Note: `away_corners` is the only target where a higher learning-rate, deeper model was selected — consistent with SHAP showing it uses more target-specific signal (H2H corners) than the other regressor targets.

---

## 4. Feature Group Taxonomy

Features are grouped by prefix. This grouping is used throughout the analysis.

| Prefix | Group | Feature count |
| :--- | :--- | :--- |
| `MKT_` | Market / Odds | ~6 |
| `OFF_` | Offensive Stats | ~30 |
| `DEF_` | Defensive Stats | ~30 |
| `OPP_ADJ_` | Opponent-Adjusted | ~28 |
| `EFFICIENCY_` | Efficiency Ratios | ~10 |
| `INTERACTION_` | Interaction Terms | ~10 |
| `DIS_` | Discipline (Cards) | ~10 |
| `H2H_` | Head-to-Head | 3 |
| `CTX_` | League Context / Standings | 4 |

---

## 5. Feature Importance by Target

### 5.1 Classifiers

**`result_3way`**  
Market-implied probabilities (`MKT_IMPLIED_HOME`, `MKT_IMPLIED_AWAY`, `MKT_IMPLIED_DRAW`) hold the top-3 SHAP positions by a wide margin. This reflects that closing match odds encode a substantial fraction of all available pre-match information. Below market features, the next meaningful contributors are cumulative points (`CTX_HOME_CUM_PTS`, `CTX_AWAY_CUM_PTS`) and rolling shots-on-target differentials (`OFF_HOME_SOT_R5`, `DEF_AWAY_SOT_R5`). H2H win-rate (`H2H_HOME_WIN_RATE_R5`) appears in the top 20, confirming historical head-to-head context adds classification signal. Beeswarm directional insight: high `MKT_IMPLIED_HOME` (red) strongly pushes P(home) up; high `CTX_AWAY_CUM_PTS` pushes P(away) up.

**`btts`**  
Top features are similarly dominated by market odds. Below the odds tier, `OFF_HOME_GOALS_R5`, `OFF_AWAY_GOALS_R5`, and efficiency features (`EFFICIENCY_HOME_SHOT_CONV`, `EFFICIENCY_AWAY_SHOT_CONV`) matter. Standings features show near-zero SHAP for btts — whether both teams score is driven by recent attacking form, not league position.

### 5.2 Goals Regressors

**`home_goals`** and **`away_goals`**  
`MKT_IMPLIED_HOME` is the strongest single predictor of home goals; `MKT_IMPLIED_AWAY` for away goals. Below odds, the most informative features are offensive rolling averages (`OFF_HOME_GOALS_R3`, `OFF_HOME_GOALS_R5`) and opponent-adjusted conceded stats (`OPP_ADJ_GOALS_CONCEDED_R5`). xG proxies (`OFF_HOME_XG_R5`, `DEF_HOME_XGA_R5`) appear in the top 20 but with noticeably lower SHAP than rolling actual goals — cold-start zero imputation dilutes signal until full Understat coverage is in place. H2H goals history (`H2H_TOTAL_GOALS_R5`) shows near-zero importance for both individual goals targets.

**`total_goals`**  
Pattern mirrors the individual goals targets. The top features are odds-implied totals, then symmetric rolling goals for both teams. `H2H_TOTAL_GOALS_R5` gains slightly more SHAP here (~rank 25–30) but remains low.

### 5.3 Corners Regressors

**`home_corners`** and **`away_corners`**  
Market odds contribute but are weaker than for goals — there are no direct corner-odds features in the database. The dominant contributors are corner-specific rolling stats: `OFF_HOME_HC_R5`, `DEF_AWAY_AC_R5`, `OPP_ADJ_CORNERS_CONCEDED_R5`. `H2H_CORNERS_R5` is in the top 15 for `away_corners` with a positive directional effect (higher historical corner counts → higher predicted corners). Standings features (`CTX_*`) show near-zero SHAP for both corner targets. xG/LUCK features are essentially zero — corners are not xG-driven.

**`total_corners`**  
Combines home and away corner signal. `H2H_CORNERS_R5` is more prominent here than in the individual targets. Rolling home+away corner averages dominate the top slots.

---

## 6. Feature Group Contribution

Aggregated group importance, normalised to 100% within each target:

| Target | Market / Odds | Offensive | Defensive | Opp-Adjusted | H2H | League Context | Other |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| `result_3way` | ~45% | ~15% | ~12% | ~10% | ~5% | ~8% | ~5% |
| `btts` | ~42% | ~18% | ~14% | ~10% | ~2% | ~4% | ~10% |
| `home_goals` | ~38% | ~22% | ~14% | ~12% | ~1% | ~2% | ~11% |
| `away_goals` | ~36% | ~20% | ~16% | ~12% | ~1% | ~2% | ~13% |
| `total_goals` | ~35% | ~22% | ~15% | ~13% | ~2% | ~2% | ~11% |
| `home_corners` | ~20% | ~28% | ~20% | ~18% | ~4% | ~1% | ~9% |
| `away_corners` | ~18% | ~25% | ~22% | ~18% | ~7% | ~1% | ~9% |
| `total_corners` | ~18% | ~27% | ~21% | ~17% | ~8% | ~1% | ~8% |

**Key pattern:** Market/odds group is the most concentrated signal for classifiers and goals targets. Corners targets are more evenly spread — no corner-specific odds exist, so the model distributes weight across rolling stats, opponent-adjusted averages, and H2H corners. H2H contributes most for corners (7–8%) vs near zero for goals (1–2%).

---

## 7. Cross-Target Feature Universality

Analysis of how many of the 8 targets rank a feature in their top 30.

**Universal features (top-30 in ≥ 6 of 8 targets)**:

All `MKT_` features qualify as universal. Below the odds tier, universal features are the core rolling offensive and defensive stats: `OFF_HOME_GOALS_R3/R5`, `OFF_AWAY_GOALS_R3/R5`, `DEF_HOME_GOALS_CONCEDED_R3/R5`, `DEF_AWAY_GOALS_CONCEDED_R3/R5`, and key opponent-adjusted composites (`OPP_ADJ_GOALS_SCORED_R5`, `OPP_ADJ_GOALS_CONCEDED_R5`). These appear universally because scoring and conceding rates correlate with corners, BTTS, and match outcome simultaneously.

**Target-specific features** (top-30 in only 1–2 targets):

- `H2H_CORNERS_R5` — top-30 only for corners targets
- `H2H_HOME_WIN_RATE_R5` — top-30 only for classifiers
- `CTX_HOME_CUM_PTS`, `CTX_AWAY_CUM_PTS` — top-30 only for classifiers (especially `result_3way`)
- `CTX_HOME_PPG_L10`, `CTX_AWAY_PPG_L10` — similar to CUM_PTS but slightly weaker
- `EFFICIENCY_*_SHOT_CONV` — top-30 for btts and goals, not corners
- `DIS_*` (card features) — narrow targets, typically only in 1 target's top-30

**Near-zero features** (max SHAP < 0.001 across all targets): Several xG/LUCK cold-start features and some interaction terms fall into this category. These are candidates for removal in US#68 target-specific feature lists. The exact list is output by notebook Section 6.

---

## 8. Phase 7 New Feature Signal

The 19 Phase 7 features added in the current schema, assessed by SHAP importance:

### 8.1 xG Rolling Features (12 features, US#45)

`OFF_{HOME,AWAY}_XG_{R3,R5}`, `DEF_{HOME,AWAY}_XGA_{R3,R5}`, `OFF_{HOME,AWAY}_LUCK_{R3,R5}`

- **Goals targets**: appear in the top 30 but with low raw SHAP. The cold-start zero imputation (Understat data not yet fetched) means most rows have zero for these features, which suppresses the learned coefficient. Expected to gain importance significantly once real xG data is ingested (US#63).
- **Corners targets**: near-zero across all corner targets — physically expected, as xG measures shot quality, not corner-taking behaviour.
- **Classifiers**: moderate signal for `result_3way` (xG delta between teams correlates with match outcome), near-zero for `btts`.
- **Universality scatter**: low mean importance, high variance — target-specific, not universal. Do not include in corners subsets.

### 8.2 League Standings Features (4 features, US#58)

`CTX_{HOME,AWAY}_CUM_PTS`, `CTX_{HOME,AWAY}_PPG_L10`

- **Classifiers**: `CTX_HOME_CUM_PTS` and `CTX_AWAY_CUM_PTS` are in the top 10 for `result_3way`. Points difference encodes overall team quality in a smooth continuous signal, complementary to the categorical signal in market odds. `PPG_L10` (form over last 10 games) adds recency-adjusted signal.
- **All regressors**: near-zero SHAP. League position does not predict how many goals or corners a team scores; recent rolling averages already capture that.
- **Recommendation**: include in classifiers subset only (US#68).

### 8.3 Head-to-Head Features (3 features, US#60)

`H2H_TOTAL_GOALS_R5`, `H2H_CORNERS_R5`, `H2H_HOME_WIN_RATE_R5`

| Feature | Best targets | SHAP direction | Action |
| :--- | :--- | :--- | :--- |
| `H2H_CORNERS_R5` | `away_corners` (#12), `total_corners` (#18) | Positive — more historical corners → higher predicted corners | Keep for corners subsets |
| `H2H_HOME_WIN_RATE_R5` | `result_3way` (~#18), `btts` (~#22) | Positive for P(home), negative for P(draw) | Keep for classifiers; drop from regressors |
| `H2H_TOTAL_GOALS_R5` | Weak across all targets; #28–35 for goals | Mixed | Can drop from all subsets in US#68 if feature budget is tight |

**Beeswarm directional insight**: For `result_3way`, high `H2H_HOME_WIN_RATE_R5` (red = historically home team dominates) pushes P(home) up and P(draw) down — directionally intuitive. For `away_corners`, high `H2H_CORNERS_R5` pushes predicted corner total up — also intuitive.

---

## 9. Sample-Level Analysis: Best to Worst Predictions

For each target, 5 test samples are examined spanning the error distribution (best, 25th pct, median, 75th pct, worst). Key patterns:

**Classifiers** — Error metric: per-sample log-loss `-log(P(true class))`  
Best-predicted matches have very high market-implied probability for the true outcome (e.g., heavy favourites winning). The model essentially echoes market consensus. Worst-predicted matches are upsets where all market/form signals pointed strongly in the wrong direction — the SHAP waterfall for the worst samples shows a cluster of red bars (features all pushing one outcome) that turns out incorrect.

**Regressors** — Error metric: |predicted − actual|  
Best predictions: the actual score aligns closely with the rolling-average expectation (stable recent form, no volatility). Worst predictions are outlier matches — e.g., `home_goals` worst case is an actual score of 7 with a predicted value of ~1.4. In these samples, SHAP shows that the model's view is not wrong per se (features all suggest a low-scoring match), but the match produced a rare extreme outcome. The dominant SHAP feature for virtually every sample remains `MKT_IMPLIED_HOME`, confirming market odds anchor all predictions.

---

## 10. Recommendations for US#64 and US#68

Based on universality analysis, Phase 7 signal assessment, and SHAP directional findings:

### Proposed Feature Subsets

**Classifiers (`result_3way`, `btts`) — keep all 133 features**  
Market odds, standings (CTX), H2H win-rate, xG rolling, and core rolling stats all contribute signal. Remove only features confirmed near-zero by notebook Section 6 output.

**Goals regressors (`home_goals`, `away_goals`, `total_goals`)**  
Drop: `H2H_HOME_WIN_RATE_R5`, `CTX_HOME_CUM_PTS`, `CTX_AWAY_CUM_PTS`, `CTX_HOME_PPG_L10`, `CTX_AWAY_PPG_L10`  
Keep: `H2H_TOTAL_GOALS_R5` (marginal but non-zero), all `OFF_`, `DEF_`, `OPP_ADJ_`, `MKT_`, xG features  
Expected benefit: recover ~0.01–0.02 MAE lost to feature noise from irrelevant contextual features.

**Corners regressors (`home_corners`, `away_corners`, `total_corners`)**  
Drop: `CTX_*` (all 4 standings features), `H2H_TOTAL_GOALS_R5`, `H2H_HOME_WIN_RATE_R5`, all xG/LUCK features (12 features — near-zero SHAP)  
Keep: `H2H_CORNERS_R5`, all `OFF_*_HC_*`, `DEF_*_AC_*`, `OPP_ADJ_CORNER_*`, `MKT_*`  
Expected benefit: largest potential gain of the three groups, given corners targets degrade most from irrelevant features.

### Implementation Path

1. **US#68 first** — add `target_features` block to `config/schema.yaml`, wire into `ModelManager._load_selected_features()` with fallback to `selected_features`
2. **US#64** — run permutation importance cross-check against these SHAP-derived subsets; validate subset sizes with 288-combo sweeps
3. **US#67** — narrow sweep around best Phase 7 hyperparams, using final per-target feature lists from US#68

---

## 11. Key Takeaways

| Finding | Detail |
| :--- | :--- |
| **Market odds dominate** | `MKT_IMPLIED_*` features hold the top SHAP positions for all 8 targets. The models are largely learning to refine market consensus, not override it. SHAP confirms this is not a gain-bias artefact. |
| **Core rolling stats are universal** | `OFF_*_GOALS_R3/R5`, `DEF_*_GOALS_CONCEDED_R3/R5`, and `OPP_ADJ_*` composites rank in the top 30 for ≥ 6 targets and are safe to include in all feature subsets. |
| **Phase 7 standings (CTX) are classifier-only** | `CTX_CUM_PTS` and `CTX_PPG_L10` are meaningful for `result_3way` (top-10 SHAP) but negligible for all regressors. Including them in goals/corners models adds noise. |
| **H2H is target-specific** | `H2H_CORNERS_R5` provides genuine signal for corners. `H2H_HOME_WIN_RATE_R5` works for classifiers. `H2H_TOTAL_GOALS_R5` is marginal everywhere. |
| **xG features are underperforming** | Cold-start zero imputation suppresses xG/LUCK signal. Once Understat data is ingested (US#63), these features are expected to become meaningful for goals targets. |
| **No single feature explains outliers** | Worst-predicted samples show the model is internally consistent — SHAP values all point the same direction, but the match produced a rare outcome. Error floors require new signal sources (real xG, in-play data), not feature selection. |
| **PRD gap is large** | All 8 targets remain well below PRD thresholds. Feature selection (US#64/68) can recover the noise-induced regression from Phase 7, but closing the structural gap to PRD requires new data (Understat xG, market odds as features). |
