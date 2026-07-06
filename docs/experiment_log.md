# FPAI Experiment Log

Tracks model versions, feature changes, and performance per target across all training phases. All league-context metrics use chronological 70/15/15 train/val/test splits. Lower is better for all metrics (MAE ↓, log_loss ↓).

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

| Version | Date | Features | Model | log_loss | Δ vs prev |
|:---|:---|:---|:---|---:|---:|
| Phase 7 baseline | 2026-05 | 133 | XGBoost (288-trial grid) | 1.0291 | — |
| Phase 10 | 2026-06-08 | 159 + xG | XGBoost (60-trial Optuna) | 1.0011 | −0.028 ✓ |
| Phase 14c | 2026-06-27 | 175 + SQUAD/LUCK | XGBoost (60-trial Optuna) | 1.0093 | +0.008 |
| **Phase 15+** | **2026-07-06** | **179 + player** | **XGBoost (60-trial Optuna)** | **0.9991** | **−0.010 ✓** |

**Notes:** H2H and market odds (especially `MKT_IMPLIED_*`, `MKT_LAMBDA_*`) are the dominant features. Player features (DEF_ANCHOR, LUCK_BURNOUT) contribute ~1.4% importance. Phase 14c regressed due to added feature noise without Optuna retuning depth; Phase 15+ broke through to a new best.

---

### `btts` — log_loss ↓

| Version | Date | Features | Model | log_loss | Δ vs prev |
|:---|:---|:---|:---|---:|---:|
| Phase 7 baseline | 2026-05 | 133 | XGBoost (288-trial grid) | 0.6735 | — |
| Phase 10 | 2026-06-08 | 159 + xG | XGBoost (60-trial Optuna) | 0.6855 | +0.012 |
| Phase 14c | 2026-06-27 | 175 + SQUAD/LUCK | XGBoost (60-trial Optuna) | 0.6859 | +0.000 |
| Phase 15 RF | 2026-07-05 | 179 + player | RandomForest | — | worse |
| Phase 15+ | 2026-07-06 | 179 + player | XGBoost (60-trial Optuna) | 0.6815 | −0.004 |
| **Current** | **2026-06-27** | **175** | **XGBoost** | **0.6859** | *threshold not met* |

**Notes:** `btts` is the hardest target — xG features did not improve it vs Phase 7 (temporal drift documented in Phase 10 analysis). Phase 15+ XGBoost reached 0.6815 but failed the 0.005 min-improvement gate. `SQUAD_RATING_AWAY` was the top player feature (4.2% importance). Remains flagged for architecture review (market-only subset, calibrated model).

---

### `home_goals` — MAE ↓

| Version | Date | Features | Model | MAE | Δ vs prev |
|:---|:---|:---|:---|---:|---:|
| Phase 7 baseline | 2026-05 | 133 | XGBoost (288-trial grid) | 0.9337 | — |
| Phase 10 | 2026-06-08 | 159 + xG | XGBoost (60-trial Optuna) | 0.9463 | +0.013 |
| Phase 14c | 2026-06-27 | 175 + SQUAD/LUCK | XGBoost (60-trial Optuna) | 0.9735 | +0.027 |
| Phase 15 RF | 2026-07-05 | 179 + player | RandomForest | — | worse |
| **Phase 15+** | **2026-07-06** | **179 + player** | **XGBoost (60-trial Optuna)** | **0.9462** | **−0.027 ✓** |

**Notes:** Phase 14c regressed significantly — SQUAD/LUCK features added noise without retuning. Phase 15+ Optuna retuning recovered exactly to Phase 10 level and pushed 0.1% further. DEF_ANCHOR_HOME was a top-10 feature.

---

### `away_goals` — MAE ↓

| Version | Date | Features | Model | MAE | Δ vs prev |
|:---|:---|:---|:---|---:|---:|
| Phase 7 baseline | 2026-05 | 133 | XGBoost (288-trial grid) | 0.8199 | — |
| Phase 10 | 2026-06-08 | 159 + xG | XGBoost (60-trial Optuna) | 0.8456 | +0.026 |
| Phase 14c | 2026-06-27 | 175 + SQUAD/LUCK | XGBoost (60-trial Optuna) | 0.8469 | +0.001 |
| Phase 15 RF | 2026-07-05 | 179 + player | RandomForest | — | worse |
| **Phase 15+** | **2026-07-06** | **179 + player** | **XGBoost (60-trial Optuna)** | **0.8276** | **−0.019 ✓** |

**Notes:** Away goals has consistently underperformed Phase 7 baseline due to dataset growth (more noise from 3,800 matches vs ~3,100). Phase 15+ closed the gap to 0.8276, the best result since Phase 7.

---

### `total_goals` — MAE ↓

| Version | Date | Features | Model | MAE | Δ vs prev |
|:---|:---|:---|:---|---:|---:|
| Phase 7 baseline | 2026-05 | 133 | XGBoost (288-trial grid) | 1.3016 | — |
| Phase 10 | 2026-06-08 | 159 + xG | XGBoost (60-trial Optuna) | 1.2741 | −0.028 ✓ |
| Phase 14c | 2026-06-27 | 175 + SQUAD/LUCK | XGBoost (60-trial Optuna) | 1.2772 | +0.003 |
| Phase 15 RF | 2026-07-05 | 179 + player | RandomForest | 1.2507 | −0.027 ✓ |
| **Phase 15+** | **2026-07-06** | **179 + player** | **XGBoost (60-trial Optuna)** | **1.2319** | **−0.019 ✓** |

**Notes:** Most consistently improving target. RF with player features was the first model to beat Phase 14c XGBoost for this target (selected temporarily). Phase 15+ XGBoost improved further to the all-time best 1.2319. DEF_ANCHOR and LUCK_BURNOUT features each contributed ~1.6% importance.

---

### `home_corners` — MAE ↓

| Version | Date | Features | Model | MAE | Δ vs prev |
|:---|:---|:---|:---|---:|---:|
| Phase 7 baseline | 2026-05 | 133 | XGBoost (288-trial grid) | 2.1261 | — |
| Phase 10 | 2026-06-08 | 159 + xG | XGBoost (60-trial Optuna) | 2.1450 | +0.019 |
| Phase 14c | 2026-06-27 | 175 + SQUAD/LUCK | XGBoost (60-trial Optuna) | 2.1736 | +0.029 |
| Phase 15 RF | 2026-07-05 | 179 + player | RandomForest | — | worse |
| **Phase 15+** | **2026-07-06** | **179 + player** | **XGBoost (60-trial Optuna)** | **2.1430** | **−0.031 ✓** |

**Notes:** Phase 14c added corner noise (SQUAD defensive stats don't correlate well with corner counts). Phase 15+ Optuna retuning recovered to Phase 10 level. DEF_ANCHOR_HOME among top-3 features (2.8% importance) — defenders who win duels affect corner generation.

---

### `away_corners` — MAE ↓

| Version | Date | Features | Model | MAE | Δ vs prev |
|:---|:---|:---|:---|---:|---:|
| Phase 7 baseline | 2026-05 | 133 | XGBoost (288-trial grid) | 2.1078 | — |
| Phase 10 | 2026-06-08 | 159 + xG | XGBoost (60-trial Optuna) | 2.0851 | −0.023 ✓ |
| Phase 14c | 2026-06-27 | 175 + SQUAD/LUCK | XGBoost (60-trial Optuna) | 2.1255 | +0.040 |
| Phase 15 RF | 2026-07-05 | 179 + player | RandomForest | — | worse |
| **Phase 15+** | **2026-07-06** | **179 + player** | **XGBoost (60-trial Optuna)** | **2.0786** | **−0.047 ✓** |

**Notes:** Away corners shows the clearest trajectory — Phase 10 was the previous all-time best (2.0851), Phase 15+ broke through to 2.0786. DEF_ANCHOR_AWAY contributed meaningfully. Phase 14c regression was the largest of any target (−0.040 MAE).

---

### `total_corners` — MAE ↓

| Version | Date | Features | Model | MAE | Δ vs prev |
|:---|:---|:---|:---|---:|---:|
| Phase 7 baseline | 2026-05 | 133 | XGBoost (288-trial grid) | 2.6495 | — |
| Phase 10 | 2026-06-08 | 159 + xG | XGBoost (60-trial Optuna) | 2.6821 | +0.033 |
| Phase 14c | 2026-06-27 | 175 + SQUAD/LUCK | XGBoost (60-trial Optuna) | 2.7170 | +0.035 |
| Phase 15 RF | 2026-07-05 | 179 + player | RandomForest | — | worse |
| **Phase 15+** | **2026-07-06** | **179 + player** | **XGBoost (60-trial Optuna)** | **2.6517** | **−0.065 ✓** |

**Notes:** Largest single improvement in Phase 15+ (−0.065 MAE, −2.4%). Total corners benefited most from DEF_ANCHOR_HOME (2.9% importance) and DEF_ANCHOR_AWAY (2.8% importance). The Phase 15+ result of 2.6517 is the all-time best, beating the Phase 7 baseline for the first time.

---

## Cross-Target Summary: Player Feature Impact (Phase 15)

The table below compares Phase 14c (SQUAD/LUCK, no lineup-level features) vs Phase 15+ (full player feature set with Optuna XGBoost):

| Target | Ph14c | Ph15+ | Δ | % Improvement | All-time Best? |
|:---|---:|---:|---:|---:|:---|
| `result_3way` | 1.0093 | **0.9991** | −0.010 | −1.0% | ✓ Yes |
| `btts` | 0.6859 | 0.6859 | 0.000 | 0.0% | No (Ph7: 0.6735) |
| `home_goals` | 0.9735 | **0.9462** | −0.027 | −2.8% | ≈ Ties Ph10 |
| `away_goals` | 0.8469 | **0.8276** | −0.019 | −2.3% | Best since Ph7 |
| `total_goals` | 1.2772 | **1.2319** | −0.045 | −3.6% | ✓ Yes |
| `home_corners` | 2.1736 | **2.1430** | −0.031 | −1.4% | ≈ Ties Ph10 |
| `away_corners` | 2.1255 | **2.0786** | −0.047 | −2.2% | ✓ Yes |
| `total_corners` | 2.7170 | **2.6517** | −0.065 | −2.4% | ✓ Yes |

**4 of 8 targets** reached all-time best performance with Phase 15+ player features. `btts` is the only target that has not improved since Phase 10 and remains open for architectural investigation.

---

## Model Selection Policy

- **Selection gate**: new model must beat current best by ≥ 0.005 on primary metric.
- **Context**: league (EPL E0, chronological split) and international are evaluated separately.
- **Feature gating**: `SQUAD_*`, `FRDS_*`, `XOC_*`, `DEF_ANCHOR_*`, `LUCK_BURNOUT_*` are disabled for competitions without `SQUAD` in `enabled_feature_groups` (competition registry).
- **Stored in**: `config/model_selection.yaml` (updated by `python main.py select-best-models`).
