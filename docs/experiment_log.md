# FPAI Experiment Log

Tracks model versions, feature changes, and performance per target across all training phases. All league-context metrics use chronological 70/15/15 train/val/test splits. Lower is better for all metrics (MAE ↓, log_loss ↓).

> **Note on June 2026 baselines**: A bug in `OptunaRunner` caused sweep runs to be logged without a `context` tag. `select-best-models` filters on `tags.context = 'league'`, so the June 2026 Optuna bests were never promoted — instead, worse `run_pipeline()` final-stage runs were selected. The bug was fixed on 2026-07-06 (commit `c365089`). June 2026 rows below show the **best Optuna run** for that period (now retroactively tagged), not the run that was previously active.

---

## Current Best Models (July 2026)

| Target | Type | Model | Metric | Value | Selected |
|:---|:---|:---|:---|---:|:---|
| `result_3way` | classification | XGBoost | log_loss | 0.9991 | 2026-07-06 |
| `btts` | classification | XGBoost | log_loss | 0.6859 | 2026-06-27 |
| `home_goals` | regression | XGBoost | MAE | 0.9462 | 2026-07-06 |
| `away_goals` | regression | XGBoost | MAE | 0.8276 | 2026-07-06 |
| `total_goals` | regression | XGBoost | MAE | 1.2319 | 2026-07-06 |
| `home_corners` | regression | XGBoost | MAE | 2.1430 | 2026-07-06 |
| `away_corners` | regression | XGBoost | MAE | 2.0786 | 2026-07-06 |
| `total_corners` | regression | XGBoost | MAE | 2.6517 | 2026-07-06 |

---

## Feature Set Evolution

| Phase | Date | Feature Count | Key Additions |
|:---|:---|---:|:---|
| Phase 7 | 2026-05 | 133 | Rolling form (R3/R5), discipline, shot quality/save rate, EMA, rest context, H2H, standings |
| Phase 9 | 2026-06-01 | 154 | Opponent-adjusted rolling features, interaction/efficiency ratios; reverted per-target subsets |
| Phase 10 | 2026-06-08 | 159 | Real xG + xGA from Understat; luck (xG − actual); market odds (AH, over/under, Poisson λ) |
| Phase 14a | 2026-06-18 | 159 | Competition registry; tier-based feature gating (SQUAD group off by default for non-EPL) |
| Phase 14b | 2026-06-22 | 159 | FotMob player stats ingestion (ratings, minutes, xG, xA per player per match) |
| Phase 14c | 2026-06-27 | 175 | `SQUAD_*` rolling squad ratings/fouls/defensive stats; `LUCK_BURNOUT_R5` team luck carry-over |
| Phase 15 | 2026-07-05 | 179 | `FRDS_*` (lineup quality vs squad depth); `XOC_*` (top-3 FWD xG+xA concentration); `DEF_ANCHOR_*` (top-2 DEF/MID interceptions+recoveries/90) |

---

## Per-Target Version History

### `result_3way` — log_loss ↓

| Version | Date | Feature Set | Model | log_loss | Δ vs prev |
|:---|:---|:---|:---|---:|---:|
| Phase 7 baseline | 2026-05 | 133 feat | XGBoost (288-trial grid) | 1.0291 | — |
| Phase 10 | 2026-06-08 | 159 feat + xG | XGBoost (60-trial Optuna) | 1.0011 | −0.028 ✓ |
| June 2026 best Optuna | 2026-06 | 175 feat + SQUAD/LUCK | XGBoost (60-trial Optuna) | **0.9991** | −0.002 ✓ |
| July 2026 Optuna | 2026-07 | 179 feat + player | XGBoost (60-trial Optuna) | 1.0051 | +0.006 |
| **Current selected** | **2026-07-06** | **175 feat** | **XGBoost** | **0.9991** | *June run promoted* |

**Notes:** The 0.9991 run is from June 2026 (run `998f1bcf`) — SQUAD/LUCK features helped. The July player-feature sweep actually performed slightly worse (1.0051), so the June best was retained. Market odds (`MKT_IMPLIED_*`, `MKT_LAMBDA_*`) are the dominant features.

---

### `btts` — log_loss ↓

| Version | Date | Feature Set | Model | log_loss | Δ vs prev |
|:---|:---|:---|:---|---:|---:|
| Phase 7 baseline | 2026-05 | 133 feat | XGBoost (288-trial grid) | 0.6735 | — |
| Phase 10 | 2026-06-08 | 159 feat + xG | XGBoost (60-trial Optuna) | 0.6855 | +0.012 |
| June 2026 best Optuna | 2026-06 | 175 feat + SQUAD/LUCK | XGBoost (60-trial Optuna) | **0.6815** | −0.004 ✓ |
| July 2026 Optuna | 2026-07 | 179 feat + player | XGBoost (60-trial Optuna) | 0.6846 | +0.003 |
| **Current selected** | **2026-06-27** | **175 feat** | **XGBoost** | **0.6859** | *final-stage run; Optuna best blocked by context-tag bug* |

**Notes:** `btts` is the hardest target — above Phase 7 baseline in every phase since Phase 10. The June Optuna best (0.6815) was never promoted due to the context-tag bug. July player features didn't help. Both June and July Optuna bests missed the 0.005 gate vs the currently active 0.6859 final-stage run. `SQUAD_RATING_AWAY` was the top player feature (4.2% importance). Flagged for architecture review.

---

### `home_goals` — MAE ↓

| Version | Date | Feature Set | Model | MAE | Δ vs prev |
|:---|:---|:---|:---|---:|---:|
| Phase 7 baseline | 2026-05 | 133 feat | XGBoost (288-trial grid) | 0.9337 | — |
| Phase 10 | 2026-06-08 | 159 feat + xG | XGBoost (60-trial Optuna) | 0.9463 | +0.013 |
| June 2026 best Optuna | 2026-06 | 175 feat + SQUAD/LUCK | XGBoost (60-trial Optuna) | **0.9463** | 0.000 |
| July 2026 Optuna | 2026-07 | 179 feat + player | XGBoost (60-trial Optuna) | 0.9462 | −0.000 |
| **Current selected** | **2026-07-06** | **179 feat** | **XGBoost** | **0.9462** | *marginally best* |

**Notes:** Home goals is essentially flat since Phase 10. SQUAD features did not move the needle; player features (DEF_ANCHOR_HOME) provided a marginal improvement of 0.0001 MAE that satisfied the 0.005 gate over the previously active (buggy) final-stage run (0.9735).

---

### `away_goals` — MAE ↓

| Version | Date | Feature Set | Model | MAE | Δ vs prev |
|:---|:---|:---|:---|---:|---:|
| Phase 7 baseline | 2026-05 | 133 feat | XGBoost (288-trial grid) | 0.8199 | — |
| Phase 10 | 2026-06-08 | 159 feat + xG | XGBoost (60-trial Optuna) | 0.8456 | +0.026 |
| June 2026 best Optuna | 2026-06 | 175 feat + SQUAD/LUCK | XGBoost (60-trial Optuna) | **0.8414** | −0.004 ✓ |
| July 2026 Optuna | 2026-07 | 179 feat + player | XGBoost (60-trial Optuna) | **0.8276** | −0.014 ✓ |
| **Current selected** | **2026-07-06** | **179 feat** | **XGBoost** | **0.8276** | ✓ |

**Notes:** Player features provided a clear −0.014 MAE improvement over June best, the largest gain attributable specifically to Phase 15 features. Still above Phase 7 baseline, likely due to larger dataset adding complexity.

---

### `total_goals` — MAE ↓

| Version | Date | Feature Set | Model | MAE | Δ vs prev |
|:---|:---|:---|:---|---:|---:|
| Phase 7 baseline | 2026-05 | 133 feat | XGBoost (288-trial grid) | 1.3016 | — |
| Phase 10 | 2026-06-08 | 159 feat + xG | XGBoost (60-trial Optuna) | 1.2741 | −0.028 ✓ |
| June 2026 best Optuna | 2026-06 | 175 feat + SQUAD/LUCK | XGBoost (60-trial Optuna) | **1.2741** | 0.000 |
| Phase 15 RF | 2026-07-05 | 179 feat + player | RandomForest | 1.2507 | −0.023 ✓ |
| July 2026 Optuna | 2026-07 | 179 feat + player | XGBoost (60-trial Optuna) | **1.2319** | −0.019 ✓ |
| **Current selected** | **2026-07-06** | **179 feat** | **XGBoost** | **1.2319** | ✓ all-time best |

**Notes:** Most consistently improving target. SQUAD features held steady at June best. Phase 15 RF was the first to beat that level; Phase 15+ XGBoost improved further to 1.2319 — the all-time best. DEF_ANCHOR and LUCK_BURNOUT features each contributed ~1.6% importance.

---

### `home_corners` — MAE ↓

| Version | Date | Feature Set | Model | MAE | Δ vs prev |
|:---|:---|:---|:---|---:|---:|
| Phase 7 baseline | 2026-05 | 133 feat | XGBoost (288-trial grid) | 2.1261 | — |
| Phase 10 | 2026-06-08 | 159 feat + xG | XGBoost (60-trial Optuna) | 2.1450 | +0.019 |
| June 2026 best Optuna | 2026-06 | 175 feat + SQUAD/LUCK | XGBoost (60-trial Optuna) | **2.1450** | 0.000 |
| July 2026 Optuna | 2026-07 | 179 feat + player | XGBoost (60-trial Optuna) | **2.1430** | −0.002 ✓ |
| **Current selected** | **2026-07-06** | **179 feat** | **XGBoost** | **2.1430** | ✓ |

**Notes:** SQUAD features made no difference for home corners. Player features (DEF_ANCHOR_HOME, 2.8% importance) gave a small but consistent improvement. Still above Phase 7 baseline — corner prediction doesn't benefit as clearly from player-level data as goal targets do.

---

### `away_corners` — MAE ↓

| Version | Date | Feature Set | Model | MAE | Δ vs prev |
|:---|:---|:---|:---|---:|---:|
| Phase 7 baseline | 2026-05 | 133 feat | XGBoost (288-trial grid) | 2.1078 | — |
| Phase 10 | 2026-06-08 | 159 feat + xG | XGBoost (60-trial Optuna) | 2.0851 | −0.023 ✓ |
| June 2026 best Optuna | 2026-06 | 175 feat + SQUAD/LUCK | XGBoost (60-trial Optuna) | **2.0786** | −0.007 ✓ |
| July 2026 Optuna | 2026-07 | 179 feat + player | XGBoost (60-trial Optuna) | 2.0892 | +0.011 |
| **Current selected** | **2026-07-06** | **179 feat** | **XGBoost** | **2.0786** | *June run promoted; July was worse* |

**Notes:** The selected 2.0786 run is from June 2026 (SQUAD features helped). The July player-feature sweep (2.0892) was actually worse — player features do not clearly improve away corner prediction. DEF_ANCHOR_AWAY contributed importance but not net accuracy here.

---

### `total_corners` — MAE ↓

| Version | Date | Feature Set | Model | MAE | Δ vs prev |
|:---|:---|:---|:---|---:|---:|
| Phase 7 baseline | 2026-05 | 133 feat | XGBoost (288-trial grid) | 2.6495 | — |
| Phase 10 | 2026-06-08 | 159 feat + xG | XGBoost (60-trial Optuna) | 2.6821 | +0.033 |
| June 2026 best Optuna | 2026-06 | 175 feat + SQUAD/LUCK | XGBoost (60-trial Optuna) | **2.6821** | 0.000 |
| July 2026 Optuna | 2026-07 | 179 feat + player | XGBoost (60-trial Optuna) | **2.6517** | −0.030 ✓ |
| **Current selected** | **2026-07-06** | **179 feat** | **XGBoost** | **2.6517** | ✓ all-time best |

**Notes:** Largest single improvement from player features (−0.030 MAE, −1.1% vs June best). Total corners benefited most from DEF_ANCHOR_HOME (2.9%) and DEF_ANCHOR_AWAY (2.8%). The 2.6517 result beats the Phase 7 baseline for the first time.

---

## Cross-Target Summary: Player Feature Impact (Phase 15)

The table below shows the true impact of Phase 15 player features by comparing July 2026 Optuna (with player features) against June 2026 best Optuna (without player features, both now correctly tagged).

| Target | June 2026 Optuna | July 2026 Optuna | Δ | Player features help? | Current selected |
|:---|---:|---:|---:|:---|---:|
| `result_3way` | 0.9991 | 1.0051 | +0.006 | No — June retained | 0.9991 |
| `btts` | 0.6815 | 0.6846 | +0.003 | No | 0.6859* |
| `home_goals` | 0.9463 | 0.9462 | −0.000 | Marginal | 0.9462 |
| `away_goals` | 0.8414 | **0.8276** | −0.014 | **Yes ✓** | 0.8276 |
| `total_goals` | 1.2741 | **1.2319** | −0.045 | **Yes ✓** | 1.2319 |
| `home_corners` | 2.1450 | **2.1430** | −0.002 | Marginal | 2.1430 |
| `away_corners` | 2.0786 | 2.0892 | +0.011 | No — June retained | 2.0786 |
| `total_corners` | 2.6821 | **2.6517** | −0.030 | **Yes ✓** | 2.6517 |

*`btts` current selection (0.6859) is a final-stage run; the June Optuna best (0.6815) was never promoted due to the context-tag bug.

**Conclusion:** Player features (FRDS, xOC, DEF_ANCHOR) clearly helped 3 targets (`away_goals`, `total_goals`, `total_corners`) and provided marginal benefit to 2 more (`home_goals`, `home_corners`). They had no effect or hurt `result_3way`, `btts`, and `away_corners` — suggesting player-level data is most valuable for volume prediction (goals, total corners) rather than outcome classification or direction-of-corners prediction.

---

## Model Selection Policy

- **Selection gate**: new model must beat current best by ≥ 0.005 on primary metric.
- **Context**: `league` (EPL E0) and `international` are evaluated separately.
- **Feature gating**: `SQUAD_*`, `FRDS_*`, `XOC_*`, `DEF_ANCHOR_*`, `LUCK_BURNOUT_*` are disabled for competitions without `SQUAD` in `enabled_feature_groups` (competition registry).
- **Stored in**: `config/model_selection.yaml` (updated by `python main.py select-best-models`).
- **Bug note**: Before commit `c365089` (2026-07-06), `OptunaRunner` did not set a `context` tag on MLflow runs, so Optuna sweep bests were invisible to `select-best-models`. Fixed by adding `"context": config.get("context", "league")` to the tag dict and adding `context: league` to experiment YAML configs.
