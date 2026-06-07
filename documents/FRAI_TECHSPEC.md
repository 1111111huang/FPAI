# Technical Specification - FPAI Forecast Engine

## 1. Architecture Overview
FPAI is a pre-match football forecasting system that serves structured forecast JSON to downstream AI agents and human analysts.

The core architecture is:

```text
raw data -> ingestion -> DuckDB raw tables -> feature factory -> feature store
          -> target registry -> target-specific models -> forecast service
          -> JSON forecast payload -> external AI agent
```

Legacy betting strategy and bankroll backtesting modules may remain in the repository, but they are not the primary product path.

## 2. Core Principles
- Forecasts must be pre-match safe.
- Each target should have an explicit target definition.
- Models should be trained and evaluated per target.
- Probabilistic outputs are preferred over hard labels.
- Uncertainty is required for all forecast families.
- Evaluation measures forecast quality rather than betting profitability.
- Forecast output must be formatted JSON.

## 3. Data Storage
Primary analytical storage remains DuckDB.

### 3.1 `raw_matches`
Stores normalized match-level source data.

Required baseline columns:

| Column | Type | Description |
| :--- | :--- | :--- |
| `match_id` | TEXT | Primary match identifier |
| `league` | TEXT | League code |
| `tier` | INTEGER | League tier |
| `date` | TIMESTAMP | Match date |
| `home_team` | TEXT | Normalized home team |
| `away_team` | TEXT | Normalized away team |
| `fthg` | INTEGER | Full-time home goals |
| `ftag` | INTEGER | Full-time away goals |
| `odds_h` | FLOAT | Home win odds |
| `odds_d` | FLOAT | Draw odds |
| `odds_a` | FLOAT | Away win odds |
| `hs`, `as` | FLOAT | Home and away shots |
| `hst`, `ast` | FLOAT | Home and away shots on target |
| `hc`, `ac` | FLOAT | Home and away corners |
| `hy`, `ay` | FLOAT | Home and away yellow cards |
| `hr`, `ar` | FLOAT | Home and away red cards |
| `avgh`, `avgd`, `avga` | FLOAT | Average 1X2 market odds |
| `xg_h`, `xg_a` | FLOAT | Home and away expected goals |
| `xga_h`, `xga_a` | FLOAT | Home and away expected goals against |

Future columns may add over/under odds, corner odds, Asian handicap lines, and richer provider metadata.

### 3.2 `feature_store`
Stores pre-match-safe engineered features keyed by `match_id`.

Feature families:

- `OFF_*`: attacking form and production.
- `DEF_*`: defensive form and prevention.
- `DIS_*`: discipline/card indicators.
- `CTX_*`: context such as rest days.
- `MKT_*`: market-implied probabilities.
- `STRENGTH_*`: derived matchup strength differences.
- `INTERACTION_*`: shifted home/away matchup deltas.
- `EFFICIENCY_*`: shifted attack-versus-defense matchup ratios.

**Current Feature Availability (v1):** 114 selected features are currently configured for training in `config/schema.yaml`. Expected goals (xG) and luck-derived features (OFF_*_XG_R3/R5, DEF_*_XGA_R3/R5, OFF_*_LUCK_R3/R5) are not included until Understat data integration provides xG values in the ingestion pipeline. The current schema includes rolling form, discipline, shot-quality/save-rate, EMA, rest-context, market probability, strength-differential, interaction, efficiency-ratio, and opponent-adjusted rolling features. Market probability features (`MKT_*`) use `avgh/avgd/avga` where available and fall back to `odds_h/d/a` (B365, corr=0.994) for pre-2020 seasons where market-average odds are absent from source CSVs.

Feature generation must use shifted rolling windows so a match never uses its own result or in-match statistics as features.

### 3.3 Model Artifacts
Model artifacts should be stored per target and versioned by:

- Target name.
- Model type.
- Feature schema version.
- Training data cutoff.
- Created timestamp.

MLflow remains the preferred experiment and artifact tracking backend.

## 4. Target Registry
Add a target registry to define every trainable target in one place.

Recommended config shape:

```yaml
targets:
  result_3way:
    task_type: multiclass_classification
    classes: ["home", "draw", "away"]
    label_columns: ["fthg", "ftag"]
    primary_metric: log_loss
    secondary_metrics: ["accuracy"]

  btts:
    task_type: binary_classification
    classes: ["no", "yes"]
    label_columns: ["fthg", "ftag"]
    primary_metric: log_loss
    secondary_metrics: ["accuracy"]

  home_goals:
    task_type: regression
    label_columns: ["fthg"]
    primary_metric: mae
    secondary_metrics: ["rmse"]

  total_corners:
    task_type: regression
    label_columns: ["hc", "ac"]
    primary_metric: mae
    secondary_metrics: ["rmse"]
```

The registry should drive:

- Label creation.
- Model selection.
- Evaluation metrics.
- Forecast payload shape.
- Artifact naming.

Implementation status:

- `src/logic/target_registry.py` defines the active target contracts.
- Supported targets are `home_win`, `result_3way`, `btts`, `home_goals`, `away_goals`, `total_goals`, `home_corners`, `away_corners`, and `total_corners`.
- `home_win` remains available as a legacy compatibility target for existing strategy and backtest flows.
- Target aliases currently normalize `both_teams_to_score` to `btts` and `3way` / `result` to `result_3way`.

## 5. Model Training
### 5.1 Split Policy
All models must use chronological splits.

Default split:

- 70 percent train.
- 15 percent validation.
- 15 percent test.

Random cross-validation is not allowed for core evaluation because it violates the pre-match forecasting setup.

### 5.2 Target-Specific Models
Separate models should be trained for each target unless a multi-output model is explicitly validated as superior.

Initial recommended targets:

- `result_3way`.
- `home_goals`.
- `away_goals`.
- `total_goals`.
- `btts`.
- `home_corners`.
- `away_corners`.
- `total_corners`.

Implementation status:

- `python main.py train-target --target <name>` trains one registry-backed target model.
- `python main.py train-forecast-suite` trains all forecast targets except legacy `home_win` by default.
- Classification targets default to logistic regression.
- Regression/count targets default to a random forest regressor.
- ModelManager selects required label columns from the registry so goal and corner targets can train from the correct raw columns.

### 5.3 Metrics
Classification:

- Primary: log loss.
- Secondary: accuracy.
- Optional: Brier score and calibration diagnostics.

Regression/count:

- Primary: MAE.
- Secondary: RMSE.
- Optional: interval coverage and mean interval width.

Implementation status:

- Classification target training logs `log_loss` and `accuracy`.
- Binary classification targets also log `precision` for backward compatibility with existing reporting.
- Regression/count target training logs `mae` and `rmse`.
- Train/validation/test split metrics are logged for each target evaluation. The logger writes both prefix and suffix styles, for example `train_log_loss`, `val_log_loss`, `test_log_loss`, and `log_loss_train`, `log_loss_val`, `log_loss_test`, so comparison tools can read older and newer runs.
- Legacy ROI, win-rate, and drawdown backtest metrics are only run for the legacy `home_win` target.

### 5.4 MLflow Experiment Organization
Forecast experiments should be organized by target, model family, sweep stage, and feature schema version.

Default experiment naming:

```text
FPAI_<target>_<model_family>_<sweep_stage>_<version>
```

Examples:

```text
FPAI_result_3way_lr_broad_v1
FPAI_btts_random_forest_broad_v1
FPAI_total_goals_rf_regressor_broad_v1
```

Required tags for every run:

- `target`: registry target name.
- `task_type`: binary classification, multiclass classification, or regression.
- `model_family`: model type from the experiment config.
- `feature_schema_version`: selected feature contract version.
- `split_policy`: `chronological_70_15_15`.
- `league`: league or `all`.
- `sweep_stage`: `smoke`, `broad`, `narrow`, or `final`.
- `experiment_version`: experiment config version.

Sweep stages:

- `smoke`: tiny grid to confirm the target/model path runs.
- `broad`: wide first-pass grid to find promising parameter families.
- `narrow`: smaller grid around the best broad ranges.
- `final`: retrain selected configurations for artifact approval.

Implementation status:

- `python main.py experiment-target --target <name> --config_path <yaml>` runs target-aware MLflow sweeps.
- `experiment-target` logs forecast-quality metrics from the target registry, not legacy betting ROI.
- `python main.py sweep-target --target <name> --config_path <yaml> --sweep_stage <smoke|broad|narrow|final>` runs the unified sweep framework in `src/utils/sweep_runner.py`.
- Broad-grid templates live in `experiments/forecast_lr_broad.yaml`, `experiments/forecast_rf_classifier_broad.yaml`, `experiments/forecast_rf_regressor_broad.yaml`, `experiments/forecast_xgb_classifier_broad.yaml`, and `experiments/forecast_gbm_regressor_broad.yaml`.
- `src/utils/model_comparison.py` powers `python main.py compare-models --target <name>` and exports CSV, JSON, or HTML reports with normalized train/validation/test metric names.
- Local MLflow state currently emits warnings for some malformed file-store experiment directories with missing `meta.yaml`; cleanup or migration is tracked in `documents/bugs.md` before destructive changes are made.

### 5.5 Model Families
Supported registry-aware model families:

- Logistic regression for baseline classification targets.
- Random forest classifier and random forest regressor baselines.
- XGBoost classifier for binary and multiclass targets. String labels are encoded internally and decoded on prediction so registry classes such as `home`, `draw`, and `away` remain stable.
- XGBoost regressor for goals and corners targets.

XGBoost sweep handling sets task-appropriate objectives and evaluation metrics:

- Multiclass classification: `multi:softprob` and `mlogloss`.
- Binary classification: `binary:logistic` and `logloss`.
- Regression/count targets: `reg:squarederror` and `rmse`.

### 5.6 Evaluation Diagnostics
`src/evaluation/diagnostics.py` generates post-training diagnostic reports for a local model artifact:

```bash
python main.py diagnose-model --target btts --model_path models/example.joblib --output_path reports/diagnostics/btts.json
```

Diagnostics include residual summaries for regression, home-odds-bin residual analysis when market columns are available, classifier calibration summaries, and interval-coverage checks when residual quantiles are present.

## 6. Uncertainty
### 6.1 Classification
Classification uncertainty should use entropy.

For a probability vector `p`, entropy is:

```text
H(p) = -sum(p_i * log(p_i))
```

Normalize entropy to `[0, 1]` by dividing by `log(number_of_classes)`.

Suggested levels:

- `low`: normalized entropy < 0.40.
- `medium`: 0.40 to 0.75.
- `high`: > 0.75.

### 6.2 Regression and Counts
Regression/count targets should expose prediction intervals.

Initial implementation options:

- Validation residual quantiles by target.
- Quantile regression models.
- Ensemble disagreement.

The first implementation should use validation residual quantiles because it is simple and target-agnostic.

Example:

```json
{
  "expected": 2.64,
  "prediction_interval": {
    "lower": 1.1,
    "upper": 4.2,
    "coverage": 0.8,
    "method": "validation_residual_quantile"
  }
}
```

## 7. Explainability
Forecast payloads must include top feature values.

Minimum explainability payload:

```json
{
  "top_features": [
    {
      "name": "OFF_HOME_XG_R5",
      "value": 1.72,
      "importance": 0.083
    }
  ]
}
```

Version 1 may use global feature importances from the trained model. A later version may add SHAP or another local explanation method.

## 8. Forecast Service
Add a forecast service responsible for assembling target model outputs into one JSON payload.

Recommended module:

```text
src/forecast/forecast_service.py
```

Responsibilities:

- Load feature rows for requested matches.
- Load latest approved model artifact for each target.
- Produce target-specific predictions.
- Calculate classification entropy.
- Attach regression prediction intervals.
- Attach top feature values.
- Emit formatted JSON.

Implementation status:

- `src/forecast/forecast_service.py` loads requested match feature rows, discovers latest per-target local artifacts, scores classification and regression targets, and validates payloads before returning them.
- `src/forecast/uncertainty.py` provides entropy uncertainty, validation-residual prediction intervals, and count distribution buckets.
- `src/forecast/schema.py` defines the stable payload keys and lightweight validation used by tests and the service.
- Forecast diagnostics include target artifact versions, feature completeness, cold-start risk, and generated timestamp.
- Explainability uses sidecar artifact `feature_importance` metadata with match-level feature values.

## 9. CLI
Recommended new primary commands:

```bash
python main.py train-target --target result_3way
python main.py evaluate-target --target result_3way
python main.py train-forecast-suite
python main.py forecast --league E0 --format json
python main.py sweep-target --target btts --config_path experiments/forecast_lr_broad.yaml --sweep_stage smoke
python main.py compare-models --target btts --format json
python main.py diagnose-model --target btts --model_path models/btts_lr_v1_20260526.joblib
```

Current implementation:

```bash
python main.py train-target --target result_3way
python main.py train-target --target total_goals --model rf_regressor
python main.py train-forecast-suite
python main.py train-forecast-suite --targets result_3way btts total_goals
python main.py forecast --league E0 --limit 20
python main.py experiment-target --target btts --config_path experiments/forecast_lr_broad.yaml
python main.py sweep-target --target result_3way --config_path experiments/forecast_xgb_classifier_broad.yaml --sweep_stage smoke --max_runs 1
python main.py compare-models --target btts --output_path reports/model_comparison/btts_comparison.json --format json
python main.py diagnose-model --target btts --model_path models/btts_lr_v1_20260526.joblib --output_path reports/diagnostics/btts.json
```

Legacy commands such as `backtest` and strategy recommendation commands may remain but should be documented as legacy.

Legacy command status:

- `predict`, `backtest`, and `experiment` are marked as legacy in CLI help.
- New agent-facing forecast access should use `python main.py forecast ...`.

## 10. Testing Requirements
Tests should cover:

- Target label generation for every registry target.
- Chronological split boundaries.
- No leakage in rolling feature generation.
- Entropy calculation for binary and multiclass outputs.
- Prediction interval construction.
- Forecast JSON schema validation.
- Top feature value extraction.

Implementation status:

- Feature-factory tests now validate the expanded `OFF_*`, `DEF_*`, `CTX_*`, `MKT_*`, and `STRENGTH_*` feature contract instead of the retired compact legacy columns.
- The feature tests continue to check shifted rolling windows so same-match outcomes do not leak into pre-match features.
- Target-registry and target-resolver tests cover every registered forecast target.
- Uncertainty tests cover binary/multiclass entropy and validation-residual interval construction.
- Forecast payload tests validate service output shape, diagnostics, explainability, stable target formatting, and count distribution buckets.
- Evaluation tests validate train/validation/test metric logging for classification and regression targets.
- Diagnostics tests validate regression residual-bin summaries and binary classifier calibration output.
- Experiment-config tests validate model aliases, sweep naming, metric normalization, and XGBoost multiclass string-label handling.
- Feature-quality integration tests (`tests/test_feature_quality.py`) run against the real DuckDB feature store and enforce NaN rate ceilings, probability sum invariants, non-negative raw rolling values, OPP_ADJ coverage advantage, and feature-label correlation directions.
- Understat integration tests (`tests/test_understat.py`) cover the JSON API fetcher, team-name mapper, and DuckDB updater.
- Cold-start imputation (US#59) ensures 0% NaN across all 133 schema features in the live feature store.

## 11. Test Execution Record
All test commands should be run with the project virtual environment:

```bash
venv/bin/python -m pytest
```

Current recorded test coverage:

| Test File | Coverage Area | Notes |
| :--- | :--- | :--- |
| `tests/test_feature_factory.py` | Expanded feature contract, feature-store persistence, shifted rolling-window leakage checks, rest-day features. | Validates current `OFF_*`, `DEF_*`, `CTX_*`, `MKT_*`, and derived feature names. |
| `tests/test_experiment_config.py` | Forecast experiment grid expansion, experiment naming, and model factory CLI aliases. | Guards `experiment-target` config plumbing. |
| `tests/test_diagnostics.py` | Evaluation diagnostics for residual bins and classifier calibration summaries. | Guards `diagnose-model` report internals. |
| `tests/test_evaluation.py` | Split metric logging and overfitting/leakage evaluation checks. | Guards train/validation/test metric contracts used by MLflow comparison. |
| `tests/test_forecast_payload.py` | Forecast service payload generation, schema validation, diagnostics, explainability, classification and regression target formatting. | Uses temporary DuckDB data and synthetic local model artifacts. |
| `tests/test_helpers.py` | Match ID generation determinism and normalization. | Guards stable match keys. |
| `tests/test_ingestion.py` | CSV ingestion and raw-match normalization. | Uses DuckDB from the project venv. |
| `tests/test_strategy.py` | Legacy strategy recommendation behavior. | Kept for compatibility while strategy commands are demoted. |
| `tests/test_target_registry.py` | Registered target definitions, aliases, metrics, and stable listing. | Covers all registry-backed forecast targets. |
| `tests/test_target_resolver.py` | Label generation for every target in the registry. | Includes goals, corners, result, BTTS, and legacy `home_win`. |
| `tests/test_uncertainty.py` | Entropy uncertainty, residual prediction intervals, and count bucket distributions. | Covers binary and multiclass entropy behavior. |
| `tests/test_feature_quality.py` | Feature store schema contract, NaN rate ceilings per family, probability sum invariants, non-negative raw rolling values, OPP_ADJ coverage advantage, feature-label correlation directions. | Integration tests against real DuckDB — all 23 pass (LUCK features excluded from non-negativity check). |
| `tests/test_understat.py` | Understat JSON API fetcher (XHR header), team-name mapper (explicit + fuzzy), DuckDB xG updater (match + team-name remap + wrong date + empty input). | 12 tests, all pass. Verified API URL and X-Requested-With header requirement. |

Latest verified run:

```text
Command: python -m pytest tests/test_feature_quality.py tests/test_feature_factory.py tests/test_understat.py -q
Result: 42 passed
```

When adding functionality, update this section with any new test files, changed coverage areas, or intentionally skipped checks.

## 12. Data Ingestion and Feature Status

### 12.1 Current Data Pipeline
- **Source:** Football-Data CSV files with match results, odds, shots, corners, and cards.
- **Processing:** `src/ingestion/data_loader.py` ingests CSVs into `raw_matches` with column mapping and validation.
- **Feature Generation:** `src/features/feature_factory.py` computes rolling averages (R3, R5), exponential moving averages (EMA5), market probabilities, and context features.

### 12.2 Feature Availability (as of 2026-06-07)
- **Current selected features:** 133 features configured in `config/schema.yaml`
  - Rolling stats R3/R5 (FTHG, FTAG, HS, AST, HC, AC, etc.)
  - Discipline features (cards and card rates)
  - Shot-quality, save-rate, and shot-accuracy derived features
  - EMA5 form features (goals scored/conceded, SOT)
  - Market-implied probabilities and margin-removed clean probabilities
  - Context features (rest days, cumulative points, PPG-10)
  - Opponent-adjusted rolling features (OPP_ADJ_, venue-independent)
  - Derived strength, interaction, and efficiency matchup features
  - **xG/xGA/LUCK** features: OFF_HOME/AWAY_XG_R3/R5, DEF_HOME/AWAY_XGA_R3/R5, OFF_HOME/AWAY_LUCK_R3/R5 — populated via `python main.py fetch-understat` (Understat JSON API, 91.6% match coverage)
  - **League standings context**: CTX_HOME/AWAY_CUM_PTS (cumulative points before match), CTX_HOME/AWAY_PPG_L10 (points-per-game last 10)
  - **Head-to-head rolling**: H2H_TOTAL_GOALS_R5, H2H_CORNERS_R5, H2H_HOME_WIN_RATE_R5 (last 5 fixture meetings, pre-match safe)

- **MKT_ odds fallback:** `avgh/avgd/avga` absent pre-2020; filled from `odds_h/d/a` (B365, corr=0.994 with market average). Result: MKT_ NaN rate 0%.
- **Cold-start imputation (US#59):** After rolling computations, NaN values in all rolling feature columns (R3, R5, EMA5, H2H, league standings) are filled with column-wise means. Result: 0% NaN across all 133 features.

### 12.3 Schema Configuration
- **File:** `config/schema.yaml`
- **Current:** `training_setup.selected_features` lists 133 selected features matching the active training contract
- **Update policy:** When new data sources are integrated, expand `selected_features` list and run `python main.py ingest` to rebuild feature store
- **Feature subset support (US#62):** `ModelManager(feature_subset=[...])` trains on a subset of schema features — useful for target-specific top-N feature selection post-sweep

## 13. Experiment Execution Record

The latest implementation pass validated all target/model paths with capped smoke sweeps rather than uncapped broad grids. This covered 18 target/model combinations:

- Classification: `result_3way` and `btts` with logistic regression, random forest classifier, and XGBoost classifier.
- Regression/count: `home_goals`, `away_goals`, `total_goals`, `home_corners`, `away_corners`, and `total_corners` with random forest regressor and XGBoost regressor.

Representative smoke results:

| Target | Model | Metric Snapshot |
| :--- | :--- | :--- |
| `result_3way` | logistic regression | `log_loss` about 1.0339, `accuracy` about 0.4787 |
| `result_3way` | XGBoost classifier | `log_loss` about 1.0170, `accuracy` about 0.5080 |
| `btts` | logistic regression | `log_loss` about 0.6817, `accuracy` about 0.5426 |
| `btts` | XGBoost classifier | `log_loss` about 0.6845, `accuracy` about 0.5452 |
| `home_goals` | XGBoost regressor | `mae` about 0.9428, `rmse` about 1.1536 |
| `away_goals` | XGBoost regressor | `mae` about 0.8260, `rmse` about 1.0436 |
| `total_goals` | XGBoost regressor | `mae` about 1.3047, `rmse` about 1.6291 |
| `total_corners` | random forest regressor | `mae` about 2.6969, `rmse` about 3.3407 |
| `total_corners` | XGBoost regressor | `mae` about 2.6666, `rmse` about 3.3182 |

### 13.2 Broad Sweep Results (2026-06-05)

Full broad-grid XGBoost sweeps completed across all 8 targets (288 runs per target, 2,304 total runs). Best test-set results per target:

| Target | Model | Best Test MAE | Best Test Log Loss | Best Params (key) |
| :--- | :--- | :--- | :--- | :--- |
| `result_3way` | XGBoostClassifier | — | (see MLflow) | — |
| `btts` | XGBoostClassifier | — | 0.6735 | — |
| `home_goals` | XGBoostRegressor | 0.9337 | — | max_depth=2, lr=0.1, subsample=0.7, colsample=0.9 |
| `away_goals` | XGBoostRegressor | 0.8199 | — | max_depth=2, lr=0.1, subsample=0.9, colsample=0.9 |
| `total_goals` | XGBoostRegressor | 1.3016 | — | max_depth=5, lr=0.1, subsample=0.9, colsample=0.9 |
| `home_corners` | XGBoostRegressor | 2.1261 | — | max_depth=5, lr=0.05, subsample=0.9, colsample=0.7 |
| `away_corners` | XGBoostRegressor | 2.1078 | — | max_depth=5, lr=0.1, subsample=0.9, colsample=0.7 |
| `total_corners` | XGBoostRegressor | 2.6495 | — | max_depth=2, lr=0.1, subsample=0.7, colsample=0.9 |

**Key finding:** Broad sweep improved all targets by less than 2% over smoke baselines. Hyperparameter tuning alone is insufficient to reach the PRD performance targets. Feature enrichment (opponent-adjusted features, Poisson objective, and xG integration) is the primary remaining lever.

### 13.3 Phase 6 Feature & Objective Experiments (2026-06-05)

Two follow-up experiment tracks run after Phase 6 US#51 baseline.

**US#54 — Opponent-Adjusted Rolling Features (114 features, XGBoost `reg:squarederror`):**

28 new `OPP_ADJ_*` features added combining home and away appearances into a single venue-independent team timeline (R3/R5 windows for goals scored/conceded, corners scored/conceded, shots-on-target, plus 8 derived matchup deltas). Feature count: 86 → 114.

**US#53 — Poisson Objective (`count:poisson`, 86 features):**

New config `experiments/forecast_poisson_regressor_broad.yaml` with `objective: count:poisson` and `eval_metric: poisson-nloglik`.

**Full comparison (288 runs per target per track):**

| Target | Baseline US#51 | US#54 OPP_ADJ | US#53 Poisson | Best |
| :--- | :--- | :--- | :--- | :--- |
| `home_goals` | MAE 0.9337 | **MAE 0.9298** | MAE 0.9312 | OPP_ADJ −0.4% |
| `away_goals` | **MAE 0.8199** | MAE 0.8276 | MAE 0.8263 | Baseline wins |
| `total_goals` | MAE 1.3016 | MAE 1.2992 | **MAE 1.2941** | Poisson −0.6% |
| `home_corners` | MAE 2.1261 | MAE 2.1234 | **MAE 2.1153** | Poisson −0.5% |
| `away_corners` | MAE 2.1078 | MAE 2.1008 | **MAE 2.0969** | Poisson −0.5% |
| `total_corners` | MAE 2.6495 | MAE 2.6527 | **MAE 2.6436** | Poisson −0.2% |
| `result_3way` | acc 0.5080 | acc 0.5080 | — | No change |
| `btts` | **log_loss 0.6735** | log_loss 0.6837 | — | Baseline wins |

**Key findings:**
- Poisson objective consistently beats `reg:squarederror` for corners and total_goals (distributional fit benefit for count data).
- OPP_ADJ features help `home_goals` and `total_goals` but hurt `away_goals` and `btts` — the combined-venue rolling stats overlap with existing venue-split features and introduce noise for some targets.
- Neither approach materially closes the gap to PRD targets. xG integration (US#43–45) remains the primary lever for goals targets.

### 13.4 Phase 7 Feature Engineering Sweep (2026-06-07)

Phase 7 added 19 new features (114 → 133 total selected) and ran a full 288-combo XGBoost broad sweep per target to measure impact.

**New feature families:**
- `OFF_{HOME,AWAY}_XG_{R3,R5}`, `DEF_{HOME,AWAY}_XGA_{R3,R5}`, `OFF_{HOME,AWAY}_LUCK_{R3,R5}` — xG rolling proxies (US#45); cold-start imputed to zero (Understat fetch pending)
- `CTX_{HOME,AWAY}_CUM_PTS`, `CTX_{HOME,AWAY}_PPG_L10` — league standings context per team before each match (US#58)
- `H2H_TOTAL_GOALS_R5`, `H2H_CORNERS_R5`, `H2H_HOME_WIN_RATE_R5` — head-to-head rolling last-5-meetings (US#60)

**Supporting changes:** cold-start NaN imputation with column means (US#59), isotonic calibration for classifiers (US#61), feature_subset parameter in ModelManager (US#62).

**Phase 7 broad sweep results (288 runs per target, XGBoost, 133 features):**

| Target | Phase 6 Baseline | Phase 7 Best | Delta | PRD Target | Met? |
| :--- | :--- | :--- | :--- | :--- | :--- |
| `result_3way` | acc 0.5080, LL 1.0170 | acc 0.5277, LL 0.9989 | +1.97 pp acc | ≥ 60% acc | No |
| `btts` | LL 0.6845, acc 0.5452 | LL 0.6849, acc 0.5420 | −0.32 pp acc | LL ≤ 0.68 | No |
| `home_goals` | MAE 0.9428 | MAE 0.9561 | +0.013 (worse) | MAE < 0.50 | No |
| `away_goals` | MAE 0.8260 | MAE 0.8476 | +0.022 (worse) | MAE < 0.50 | No |
| `total_goals` | MAE 1.3046 | MAE 1.2807 | −0.024 (better) | MAE < 0.75 | No |
| `home_corners` | MAE 2.1393 | MAE 2.1580 | +0.019 (worse) | MAE < 1.50 | No |
| `away_corners` | MAE 2.1130 | MAE 2.0814 | −0.032 (better) | MAE < 1.50 | No |
| `total_corners` | MAE 2.6652 | MAE 2.6962 | +0.031 (worse) | MAE < 1.50 | No |

**Key findings:**
- New contextual features (H2H, standings) improve `result_3way` accuracy by +1.97 pp — the largest single-phase classification gain.
- New features slightly hurt individual goals targets (`home_goals`, `away_goals`) — H2H/standings are noisy for count prediction.
- This validates US#62: goals/corners models should use target-specific feature subsets excluding H2H and standings features.
- Hyperparameter sweep alone is insufficient — all targets remain well below PRD thresholds, requiring new signal sources.

**Phase 8 priorities:** (1) target-specific feature subsets via permutation importance, (2) real xG from Understat fetch (US#63), (3) market odds integration (US#64 candidate), (4) ensemble/stacking for goals.

## 14. Model Comparison & Evaluation Analysis

### 14.1 Comparison Tool
- **Module:** `src/utils/model_comparison.py`
- **CLI:** `python main.py compare-models --target <name> [--format csv|json|html]`
- **Purpose:** Query MLflow for all runs per target, extract train/val/test metrics, identify best-performing variant per metric

**Supported Comparisons:**
- Across model types (logistic regression, random forest, XGBoost)
- Train/validation/test metric isolation (detect overfitting)
- Parameter sensitivity (which hyperparameters matter most)

**Output:** Comparison report with rankings, best performers marked, and metric deltas highlighted

### 14.2 Evaluation Diagnostics
- **Module:** `src/evaluation/diagnostics.py`
- **CLI:** `python main.py diagnose-model --target <name> --model_path <path>`
- **Purpose:** Analyze residuals, calibration, and prediction interval coverage

**Diagnostics:**
- Residual distribution by match characteristics (home/away, odds bin, tier, date)
- Calibration curves for classifiers
- Prediction interval coverage statistics
- Feature completeness and cold-start risk detection

### 14.3 Model Evaluation Results Summary
All target/model paths passed smoke sweeps. Representative results from latest validation split:

| Target | Model | Test Log Loss | Test Accuracy | Test MAE |
| :--- | :--- | :--- | :--- | :--- |
| result_3way | LogisticRegression | 1.0339 | 0.4787 | - |
| result_3way | RandomForestClassifier | ~1.05 | ~0.475 | - |
| result_3way | XGBoostClassifier | 1.0170 | 0.5080 | - |
| btts | LogisticRegression | 0.6817 | 0.5426 | - |
| btts | RandomForestClassifier | ~0.68 | ~0.54 | - |
| btts | XGBoostClassifier | 0.6845 | 0.5452 | - |
| home_goals | RandomForestRegressor | - | - | 0.9428 |
| home_goals | XGBoostRegressor | - | - | 0.9428 |
| away_goals | RandomForestRegressor | - | - | 0.8260 |
| away_goals | XGBoostRegressor | - | - | 0.8260 |
| total_goals | RandomForestRegressor | - | - | 1.3047 |
| total_goals | XGBoostRegressor | - | - | 1.3047 |
| home_corners | RandomForestRegressor | - | - | 2.1472 |
| home_corners | XGBoostRegressor | - | - | ~2.14 |
| away_corners | RandomForestRegressor | - | - | 2.1140 |
| away_corners | XGBoostRegressor | - | - | ~2.11 |
| total_corners | RandomForestRegressor | - | - | 2.6969 |
| total_corners | XGBoostRegressor | - | - | 2.6666 |

**Observations:**
- XGBoost classifiers marginally improve log loss for 3-way classification
- Random Forest and XGBoost regressors show similar performance on goal/corner targets
- MAE ranges: 0.8-1.3 for goals, 2.1-2.7 for corners
- Logistic regression remains competitive baseline despite simpler architecture

## 15. Learning Curve Analysis (US#64)

### 15.1 Module & CLI
- **Module:** `src/utils/learning_curve.py`
- **CLI:** `python main.py learning-curve --all_targets` or `--target <name>`
- **Output:** Per-target CSV + chart in `reports/learning_curves/`, combined grid chart `learning_curves_all_targets.png`

### 15.2 Method
XGBoost (200 estimators, max_depth=3, lr=0.1) trained on 20/40/60/80/100% of the chronological training split. Val set (15%, 558 matches) held fixed across all fractions. Primary metric evaluated on val set per fraction.

### 15.3 Results (2026-06-07)

| Target | Metric | 20% | 40% | 60% | 80% | 100% | Δ 20→100% | Pattern |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| `result_3way` | log_loss | 0.9588 | 0.9441 | 0.9411 | 0.9291 | 0.9282 | −3.2% | Plateau at 80–100% |
| `btts` | log_loss | 0.6989 | 0.6914 | **0.6876** | 0.6889 | 0.6914 | −1.1% | **Degrades after 60%** |
| `home_goals` | mae | 1.0206 | 1.0096 | 1.0178 | 1.0109 | 0.9951 | −2.5% | Noisy, improving at tail |
| `away_goals` | mae | 0.8819 | 0.8845 | 0.9330 | 0.9064 | 0.8966 | +1.7% | Non-monotonic, noisy |
| `total_goals` | mae | 1.3651 | 1.3578 | 1.3453 | 1.3581 | 1.3558 | −0.7% | **Flat — clear plateau** |
| `home_corners` | mae | 2.4564 | 2.4471 | 2.4388 | 2.4479 | 2.4289 | −1.1% | Noisy, slight improvement |
| `away_corners` | mae | 2.1366 | 2.1418 | 2.1126 | 2.0773 | 2.0590 | −3.6% | Most data-sensitive target |
| `total_corners` | mae | 2.7550 | 2.7561 | 2.7499 | 2.7488 | 2.7522 | −0.1% | **Flat — clear plateau** |

### 15.4 Key Findings
- **Feature ceiling confirmed.** `total_goals`, `total_corners`, and `btts` are essentially flat across the full training range (<1% gain). These targets are feature-limited or irreducibly noisy — more match data will not help.
- **`btts` temporal drift.** Performance is best at 60% of training data (0.6876) and *worsens* at 80% and 100% (0.6914). The most recent training data hurts the val set. This suggests the btts market or match dynamics have shifted over time, and xG features (encoding current team attack quality) may be especially important for correcting this.
- **`away_corners` is the most data-sensitive target** (−3.6%, still improving at 100%). Still small in absolute terms; not sufficient justification for acquiring more match data.
- **No target warrants LSTM investment.** All curves plateau well before 100% of the current training data. The bottleneck is new signal sources, not sequence length.
- **Recommended next step:** US#65 (per-target feature subsets) and US#67 (market odds) remain the primary performance levers.

## 16. Feature Importance & Selection Analysis

### 16.1 Permutation Importance
- **Module:** `src/utils/feature_importance.py`
- **CLI:** `python main.py permutation-importance --target <name> --model_path <path> [--n_repeats 10]`
- **Purpose:** Identify high-impact features using permutation importance methodology

**Outputs:**
- CSV report with feature rankings, importance_mean, importance_std, importance_pct
- Top-N summary for quick reference

**Method:**
- Uses sklearn `permutation_importance()` on validation split
- Ranks 86 features by mean importance decrease
- Computes importance as percentage of max for easy cross-target comparison

### 16.2 Feature Selection Study
- **Module:** `src/utils/feature_selection.py`
- **Purpose:** Determine optimal feature set size and composition per target

**Study Design:**
- Input: Ranked feature list from permutation importance analysis
- Stepwise experiments: Train models with top-10, top-20, top-30, top-40, all features
- Evaluation: Measure train/val/test metrics for each subset
- Output: Recommendation for minimum viable feature set

**Feature Selection Results:**
- Identifies where adding features yields diminishing returns
- Provides improvement percentage over baseline (smallest feature set)
- Supports inference optimization (smaller models) vs accuracy tradeoff

**Recommendation Logic:**
- Start with all features as baseline
- Find knee point where metric improvement < 1% threshold
- Recommend smallest feature set meeting improvement threshold

### 16.3 Top Features (Example from btts target)
When permutation importance analysis is run, look for features like:
- `OFF_*` offensive form metrics (top priority typically)
- `DEF_*` defensive form metrics
- `MKT_*` market-implied probabilities
- `STRENGTH_*` matchup differentials
- `INTERACTION_*` shifted deltas
- `EFFICIENCY_*` attack-vs-defense ratios

**Cross-Target Feature Patterns:**
- Offensive features dominate goals targets
- Defensive features important for BTTS and result_3way
- Market odds and strength differentials universally relevant
- Interaction/efficiency features add incremental value

## 17. MLflow Store Management

### 16.1 Cleanup Tool
- **Module:** `src/evaluation/mlflow_cleanup.py`
- **CLI:** `python main.py cleanup-mlflow [--strategy recover|remove|backup_and_remove] [--report_only]`
- **Purpose:** Repair or remove malformed MLflow experiments

**Malformed Experiments:**
- Missing `meta.yaml` files prevent MLflow from loading experiment metadata
- Symptom: Warnings during experiment queries
- Root cause: Partial experiment initialization or corruption

**Cleanup Strategies:**
1. **recover:** Create minimal meta.yaml for each malformed experiment (preserves runs)
2. **remove:** Delete entire experiment directories (use if empty or corrupt)
3. **backup_and_remove:** Backup to `.mlflow_backup/` then delete

**Action Taken:**
- Successfully recovered 9 malformed experiments (IDs: 1-9)
- Created minimal meta.yaml files with experiment names
- All recovered experiments now queryable via MLflow API

### 16.2 Latest Store Status
- Total experiments: 28
- Valid experiments: 19
- Recovered experiments: 9
- Total runs: (See MLflow UI for count)

## 18. Migration Plan
1. Archive legacy product and technical documents.
2. Introduce target registry.
3. Expand target resolver beyond `home_win`.
4. Add target-specific model training and evaluation.
5. Implement forecast JSON service.
6. Add uncertainty and explainability payloads.
7. Demote strategy/backtest commands in CLI help and docs.
8. Add richer data integrations, including Understat xG and future corner odds.

## 19. Phase 8: Target-Specific Optimization & Signal Expansion (US#65–71)

### 19.1 Per-Target Feature Subsets (US#65 + US#66)

Permutation importance (sklearn, 10 repeats) run on all 8 trained XGBoost models. For `result_3way`, XGBoost gain importance substituted (permutation scoring broke on string class labels). Per-target lists in `config/schema.yaml` under `target_features:`.

| Target | Method | # Features | Top Signal |
| :--- | :--- | :--- | :--- |
| `result_3way` | XGBoost gain | 40 | `MKT_IMPLIED_AWAY`, `MKT_LOG_ODDS_H`, `MKT_LOG_ODDS_H_A_RATIO` |
| `btts` | Permutation | 40 | `CTX_HOME_REST_DAYS`, `MKT_OVERROUND`, `DEF_HOME_FTAG_R3` |
| `home_goals` | Permutation | 25 | `MKT_IMPLIED_HOME`, `MKT_Home_Prob_Real`, `MKT_LOG_ODDS_H` |
| `away_goals` | Permutation | 25 | `MKT_IMPLIED_AWAY`, `MKT_LOG_ODDS_A`, `DIS_HOME_HY_R5` |
| `total_goals` | Permutation | 25 | `MKT_LOG_ODDS_D`, `OFF_HOME_FTHG_R5`, `CTX_HOME_CUM_PTS` |
| `home_corners` | Permutation | 20 | `MKT_LOG_ODDS_H_A_RATIO`, `MKT_LOG_ODDS_A`, `MKT_IMPLIED_HOME` |
| `away_corners` | Permutation | 20 | `MKT_IMPLIED_AWAY`, `MKT_LOG_ODDS_H_A_RATIO`, `MKT_IMPLIED_HOME` |
| `total_corners` | Permutation | 20 | `MKT_IMPLIED_HOME`, `OFF_HOME_HST_EMA5`, `DIS_HOME_DISCIPLINE_SCORE_R3` |

`ModelManager._load_selected_features()` priority: `feature_subset` param > `target_features[name]` > global `selected_features`.

### 19.2 Market Odds Features (US#67)

`_compute_odds_features()` in `FeatureFactory`. 5 new `MKT_` features: `MKT_OVERROUND`, `MKT_LOG_ODDS_H/D/A`, `MKT_LOG_ODDS_H_A_RATIO`. Schema: 154→159 features. Market features rank #1 for goals and corners targets. Raw odds column fallback: `avgh/avgd/avga` → `odds_h/d/a`.

### 19.3 Temporal Features (US#68)

`_compute_temporal_features()` in `FeatureFactory`. 16 new features (schema: 133→159 combined with US#67):
- **Form variance** (6): `CTX_HOME/AWAY_GOALS/CONCEDED/CORNERS_STD_R5` — team consistency signals
- **EMA3 decay** (4): `OFF_HOME_FTHG_EMA3`, `DEF_HOME_FTAG_EMA3`, `OFF_AWAY_FTAG_EMA3`, `DEF_AWAY_FTHG_EMA3`
- **Streaks** (6): `CTX_HOME/AWAY_SCORE/WIN/CS_STREAK` — consecutive match run counts

All features are shift(1)-safe (pre-match only). `CTX_HOME_REST_DAYS` ranks #1 for btts.

### 19.4 Narrow Hyperparameter Sweep (US#69)

Narrow grid configs created for post-permutation-importance sweep:
- `experiments/forecast_xgb_classifier_narrow.yaml` — n_estimators [400,500,600], max_depth [3,4], lr [0.03,0.05,0.08]
- `experiments/forecast_xgb_regressor_narrow.yaml` — adds reg_lambda [1.0,2.0]

Run: `python main.py sweep-target --target <name> --config_path experiments/forecast_xgb_classifier_narrow.yaml --sweep_stage narrow`

### 19.5 Optuna Bayesian Sweep (US#70)

`OptunaRunner` class in `src/utils/sweep_runner.py`. TPE sampler (`optuna.samplers.TPESampler(seed=42)`). Search space config format:
- `{type: float, low: x, high: y, log: bool}` → `trial.suggest_float`
- `{type: int, low: x, high: y}` → `trial.suggest_int`
- `[v1, v2, ...]` → `trial.suggest_categorical`

Configs: `experiments/optuna_xgb_classifier.yaml`, `experiments/optuna_xgb_regressor.yaml` (60 trials each). Backward-compatible with `grid_search`-format configs. CLI: `python main.py optuna-sweep --target <name> --config_path <yaml> [--n_trials N]`.

### 19.6 Ensemble Stacking for Goals (US#71)

`GoalStackerModel` in `src/models/goal_stacker.py`. Architecture:
- **Level-0:** `XGBRegressor(objective="count:poisson")` + `sklearn.PoissonRegressor`
- **Level-1:** `Ridge` meta-learner
- **OOF strategy:** Chronological 50/50 split — base models train on first half, second-half out-of-fold predictions train meta-learner, then base models retrain on full training set

Predictions clipped to `>= 0`. Registered in `ModelFactory` as `goal_stacker`/`stacker`. Saves as joblib payload (XGB + GLM + Ridge + feature columns). Train: `python main.py train-target --target home_goals --model goal_stacker`. 11 unit tests in `tests/test_goal_stacker.py` pass.

## 20. Completed User Story Archive

All user stories completed through Phase 7. Active and blocked stories are tracked in `documents/user_stories.md`.

| # | Story Name | Completion Notes |
| :--- | :--- | :--- |
| 1 | Define Target Registry | Implemented in `src/logic/target_registry.py`. |
| 2 | Expand Target Resolver | Supports `result_3way`, `btts`, `home_goals`, `away_goals`, `total_goals`, `home_corners`, `away_corners`, `total_corners`. Keeps legacy `home_win`. |
| 3 | Add Target-Specific Training Command | `python main.py train-target --target <name>`. |
| 4 | Add Forecast Suite Training Command | `python main.py train-forecast-suite`. Defaults to all forecast targets excluding legacy `home_win`. |
| 5 | Implement Target-Specific Evaluation | Metrics driven by target task type (log loss/accuracy for classification, MAE/RMSE for regression). |
| 6 | Preserve Chronological Splits | Chronological train/val/test enforced across all target training flows. |
| 7 | Build Forecast Service | `src/forecast/forecast_service.py`. |
| 8 | Add Forecast CLI Command | `python main.py forecast ...` emits formatted JSON. |
| 9 | Implement Entropy Uncertainty | Normalized entropy for binary and multiclass targets. Covered by `tests/test_uncertainty.py`. |
| 10 | Implement Prediction Intervals | Validation residual quantiles stored in artifact sidecar metadata. |
| 11 | Add Feature Value Explainability | Top feature names, match-level values, and global importance values in forecast payloads. |
| 12 | Add Forecast JSON Schema | Lightweight validator in `src/forecast/schema.py`. |
| 13 | Train Result 3-Way Model | Logistic regression broad sweep (4 runs): best log_loss=1.0291, accuracy=0.492. 3521 matches, chronological split. |
| 14 | Train Goals Models | Random forest regressor broad sweep. Home MAE=0.943, Away MAE=0.826, Total MAE=1.312. |
| 15 | Train BTTS Model | Logistic regression: log_loss=0.6931, accuracy=0.5319, precision=0.5628. Artifact: `models/btts_lr_v1_20260526.joblib`. |
| 16 | Train Corners Models | Random forest regressor. Home MAE=2.147, Away MAE=2.114, Total MAE=2.677. Corner data 100% available (3721 matches). |
| 17 | Define MLflow Experiment Methodology | Documented in tech spec; local MLflow store reset. |
| 18 | Add Registry-Aware Experiment Command | `python main.py experiment-target --target <name> --config_path <yaml>`. |
| 19 | Add Broad Experiment Configs | `experiments/forecast_*_broad.yaml` templates added. |
| 20 | Add Experiment Helper Tests | Covered by `tests/test_experiment_config.py`. |
| 21 | Add Goal Distribution Output | Poisson count buckets from expected value. |
| 22 | Add Corners Distribution Output | Same count bucket contract as goals. |
| 23 | Add Model Artifact Versioning | `.metadata.json` sidecars beside model files. |
| 24 | Add Forecast Diagnostics | Model version, target versions, feature completeness, cold-start risk, generated timestamp in every payload. |
| 25 | Demote Legacy Betting Commands | `predict`, `backtest`, and `experiment` marked as legacy in CLI help. |
| 26 | Update Existing Tests For New Feature Contract | Full suite passes; validates expanded feature schema. |
| 27 | Add Target Resolver Tests | Covered by `tests/test_target_resolver.py`. |
| 28 | Add Uncertainty Tests | Covered by `tests/test_uncertainty.py`. |
| 29 | Add Forecast Payload Tests | Covered by `tests/test_forecast_payload.py`. |
| 32 | Add Feature Completeness Metric | Included in forecast diagnostics. |
| 33 | Add Cold-Start Risk Metric | Based on missing rolling/EMA features and completeness threshold. |
| 34 | Document Agent Tool Contract | `documents/AGENT_TOOL_CONTRACT.md`. |
| 35 | Add Train/Val/Test Split Metrics | `ModelManager._evaluate_target()` logs train/val/test metrics in both naming styles for compatibility. |
| 36 | Build Model Comparison Tool | `src/utils/model_comparison.py`; CLI: `python main.py compare-models --target <name>`. |
| 37 | Add Evaluation Diagnostics Module | `src/evaluation/diagnostics.py`; CLI: `python main.py diagnose-model`. |
| 38 | Add Overfitting Detection & Validation Tests | Split-metric logging tests plus leakage/split tests in `tests/test_evaluation.py`. |
| 39 | Add XGBoost Classifier Experiments | XGBoost classifier config/support added; all classifier target/model paths passed smoke sweeps. Full broad-grid review via US#51. |
| 40 | Add Gradient Boosting Regressor Experiments | `XGBoostRegressorModel` added; regressor config and smoke coverage across all count targets. Full broad-grid review via US#51. |
| 41 | Build Systematic Hyperparameter Sweep Framework | `SweepRunner` in `src/utils/sweep_runner.py`; `python main.py sweep-target --target <name> --config_path <yaml> --sweep_stage <smoke\|broad\|narrow\|final>`. |
| 46 | Add Feature Interaction Engineering | `INTERACTION_*` and `EFFICIENCY_*` matchup features added to feature factory and schema with regression tests. |
| 47 | Run Permutation Importance Analysis | `PermutationImportanceAnalyzer` in `src/utils/feature_importance.py`; CLI: `python main.py permutation-importance`. |
| 48 | Implement Feature Selection Study | `FeatureSelectionStudy` in `src/utils/feature_selection.py`; stepwise elimination with top-10/20/30/40/all subsets. |
| 49 | Update Technical Specification with Model Findings | Sections 14–16 added to this document: Model Comparison & Evaluation, Feature Importance & Selection Analysis, MLflow Store Management. |
| 50 | Clean or Migrate Malformed Local MLflow File Store | `MLflowStoreCleanup` in `src/evaluation/mlflow_cleanup.py`; recovered 9 malformed experiments (IDs 1–9). |
| 51 | Run Full Broad Experiment Suite After MLflow Store Cleanup | Executed 288-run XGBoost broad sweeps for all 8 targets (2 classifiers + 6 regressors), 2,304 total runs. Best results: home_goals MAE=0.9337, away_goals MAE=0.8199, total_goals MAE=1.3016, home_corners MAE=2.1261, away_corners MAE=2.1078, total_corners MAE=2.6495, btts log_loss=0.6735. See Section 13.2 for full results. |
| 52 | Run Narrow Sweep Around Best Broad-Grid Configurations | Closed without execution. Broad sweep gains were marginal (< 2% across all targets); a narrow follow-up sweep would not meaningfully close the gap to PRD targets. Feature engineering (US#53, US#54, US#43–45) is the higher-priority path. |
| 53 | Add Poisson/Count Objective for Goals and Corners Regressors | Added `experiments/forecast_poisson_regressor_broad.yaml` with `objective: count:poisson` / `eval_metric: poisson-nloglik`. Ran 288-run broad sweeps for all 6 regression targets. Poisson consistently beats `reg:squarederror` for corners and total_goals (−0.2% to −0.6% MAE). Goals targets show mixed results. See Section 13.3. |
| 54 | Add Opponent-Adjusted Rolling Features | Added `_compute_opp_adjusted_rolling()` to `src/features/feature_factory.py`. 28 new `OPP_ADJ_*` features combining home+away appearances into venue-independent rolling windows (R3/R5: goals, corners, SOT, matchup deltas). Schema expanded 86→114 features. Tests: `test_opp_adjusted_features_no_leakage`, `test_opp_adjusted_features_combine_home_and_away_venues`. Swept all 8 targets. Marginal benefit for home_goals/total_goals; slightly hurt away_goals and btts. See Section 13.3. |
| 55 | Feature Quality Integration Tests | Added `tests/test_feature_quality.py` (23 tests) against the real DuckDB feature store. Covers: schema contract (all 114 selected features present), no-inf check, MKT probability sums to 1.0 ± 1e-5, NaN rate ceilings per feature family (OFF_/DEF_/DIS_ < 10%, CTX_ < 5%, STRENGTH_/INTERACTION_/EFFICIENCY_ < 15%, OPP_ADJ_ < 8%, MKT_ < 5%), non-negative values for OFF_ and OPP_ADJ raw rolling features, OPP_ADJ NaN rate lower than venue-split OFF_, 6 feature-label Pearson correlation direction checks, and OPP_ADJ matchup delta mean ≈ 0. `MKT_` NaN test flagged intentional failure at 30.6% — resolved by US#56. |
| 56 | Fix MKT_ Feature NaN Gap for Pre-2020 Seasons | Root cause: Football-Data CSVs before 2020 omit `avgh/avgd/avga` (market-average odds); `odds_h/d/a` (B365, corr=0.994 with market avg) is present for all 3721 matches. Fix: added `odds_h/d/a` to `compute_rolling_stats()` SQL SELECT and applied `fillna(odds_*)` on `avgh/avgd/avga` before MKT_ computation. MKT_ NaN rate dropped from 30.6% to 0.0%. All 23 feature-quality tests now pass. |
| 57 | Add xG Proxy Features from Shots on Target | Superseded by US#63 (real Understat xG now available). xG/xGA/LUCK rolling features added to schema via US#45. Closed without implementing proxy logic. |
| 45 | Add xG Features to Training Schema | Added 12 xG/xGA/LUCK rolling features to `config/schema.yaml`: `OFF_{HOME,AWAY}_XG_{R3,R5}`, `DEF_{HOME,AWAY}_XGA_{R3,R5}`, `OFF_{HOME,AWAY}_LUCK_{R3,R5}`. Cold-start imputation ensures 0% NaN even pre-Understat. Schema grew 114→133. 92 tests pass. |
| 58 | Add League Standings / Current Form Table Features | Implemented `_compute_league_standings()` in `feature_factory.py`. Adds `CTX_HOME/AWAY_CUM_PTS` (shifted cumulative points) and `CTX_HOME/AWAY_PPG_L10` (PPG over last 10 matches) per team before each fixture. All features pre-match safe via `shift(1)`. 0% NaN after cold-start imputation. |
| 59 | Cold-Start Imputation with League Averages | Implemented `_apply_cold_start_imputation()` in `feature_factory.py`. Fills NaN in all rolling feature columns (R3/R5/EMA5/H2H/standings) with column-wise means, excluding `MKT_` features. Runs as final post-processing step. Confirmed: 0% NaN across all 133 schema features. |
| 60 | Add Head-to-Head Rolling Features | Implemented `_compute_h2h_rolling()` in `feature_factory.py`. Builds per-(my_team, opponent) pair perspective view, computes `shift(1)` rolling-5 for `H2H_TOTAL_GOALS_R5`, `H2H_CORNERS_R5`, `H2H_HOME_WIN_RATE_R5`. All pre-match safe. Added to schema; 0% NaN after imputation. |
| 61 | Post-Hoc Model Calibration for Classifiers | Implemented `_fit_and_save_calibrator()` in `model_manager.py`. Fits isotonic regression on val-set probabilities for classifiers. Binary: single calibrator; multiclass: OvR per-class with row renormalization. Saves `.calibration.pkl` sidecar. Logs `val_log_loss_calibrated`, `val_log_loss_uncalibrated`, `calibration_improvement` to MLflow. |
| 62 | Target-Specific Feature Subset Selection | Added `feature_subset: list[str] \| None` parameter to `ModelManager.__init__()`. `_load_selected_features()` filters schema features to the provided intersection, logging subset size to MLflow. Infrastructure in place; subsets populated from permutation importance in Phase 8. |
| 63 | Complete Understat Integration for True xG Features | Implemented `src/ingestion/understat_fetcher.py` (HTTP + HTML parse of Understat datesData JSON), `update_raw_matches_xg()` in `src/ingestion/understat.py` (DuckDB UPDATE with fuzzy team-name matching), and `python main.py fetch-understat` CLI. Populated `config/team_mapping.json` with 35 EPL name mappings. 13 tests in `tests/test_understat.py` all pass. Supersedes US#43/44. |
| 64 | Learning Curve Analysis | Module: `src/utils/learning_curve.py` (`LearningCurveAnalyzer`, `run_all_targets`, `summarise_findings`). CLI: `python main.py learning-curve --all_targets`. Results: feature ceiling confirmed for total_goals/total_corners/btts (curves plateau; <1% gain 20%→100%); away_corners most data-sensitive (−3.6% MAE from 80%→100%); btts degrades after 60% training data (temporal drift). LSTM not warranted. See Section 15 for full results table. 14 unit tests pass. |
| 65 | Per-Target Feature Subset Sweep | Permutation importance computed on all 8 XGBoost models (10 repeats, sklearn `permutation_importance`). Reports in `reports/permutation_importance/`. result_3way importance derived via XGBoost gain (string-label scoring issue with permutation method). Subsets: result_3way top-40 (gain), btts top-40 (permutation), goals top-25, corners top-20. Market features (`MKT_IMPLIED_*`, `MKT_LOG_ODDS_*`) dominate goals and corners targets. `CTX_HOME_REST_DAYS` ranks #1 for btts. |
| 66 | Target-Specific Feature List Configuration | `target_features` block added to `config/schema.yaml` with 8 per-target lists. `ModelManager._load_selected_features()` updated to check `target_features[target_name]` before falling back to global `selected_features`. Priority: `feature_subset` param (US#62) > per-target list (US#66) > global list. All features validated against global list to prevent schema drift. 82 tests pass. |
| 67 | Market Odds as Features | `_compute_odds_features()` static method added to `FeatureFactory.compute_rolling_stats()`. Inputs: `avgh/avgd/avga` (fallback to `odds_h/d/a`). 5 new features: `MKT_OVERROUND`, `MKT_LOG_ODDS_H`, `MKT_LOG_ODDS_D`, `MKT_LOG_ODDS_A`, `MKT_LOG_ODDS_H_A_RATIO`. Schema grew 154→159. `MKT_IMPLIED_HOME` ranks #1 for home_goals, home_corners, away_corners, total_corners importance. |
| 68 | Richer Temporal Features | `_compute_temporal_features()` static method added to `FeatureFactory.compute_rolling_stats()`. 16 new features: 6 form-variance std features (`CTX_*_GOALS/CONCEDED/CORNERS_STD_R5`), 4 EMA3 short-decay variants (`OFF_HOME_FTHG_EMA3`, `DEF_HOME_FTAG_EMA3`, `OFF_AWAY_FTAG_EMA3`, `DEF_AWAY_FTHG_EMA3`), 6 streak indicators (`CTX_HOME/AWAY_SCORE/WIN/CS_STREAK`). All shift(1)-safe. Schema grew 133→159 total (with US#67). |
| 69 | Narrow Hyperparameter Sweep | Config files created: `experiments/forecast_xgb_classifier_narrow.yaml` (n_estimators [400,500,600], max_depth [3,4], lr [0.03,0.05,0.08], subsample/colsample [0.75,0.85], min_child_weight [3,5], gamma [0,0.1]) and `experiments/forecast_xgb_regressor_narrow.yaml` (adds reg_lambda [1.0,2.0]). Run via `python main.py sweep-target --target <name> --config_path experiments/forecast_xgb_classifier_narrow.yaml`. |
| 70 | Optuna Bayesian Sweep | `OptunaRunner` class added to `src/utils/sweep_runner.py`. TPE sampler with seed=42. Search space spec: list → `suggest_categorical`; `{type:float, low, high, log}` → `suggest_float`; `{type:int, low, high}` → `suggest_int`. Configs: `experiments/optuna_xgb_classifier.yaml` and `experiments/optuna_xgb_regressor.yaml` (60 trials, continuous ranges). CLI: `python main.py optuna-sweep --target <name> --config_path <yaml> [--n_trials N]`. Falls back to `grid_search` if `optuna_search` not in config (backward-compatible). |
| 71 | Ensemble / Stacking for Goals Targets | `GoalStackerModel` in `src/models/goal_stacker.py`. Architecture: Level-0 = XGBoost Poisson (`count:poisson`) + sklearn `PoissonRegressor`; Level-1 = `Ridge` meta-learner. OOF strategy: chronological 50/50 split — base models trained on first half, OOF predictions from second half used to fit meta-learner, then base models retrained on full training set. Save/load via joblib payload. Registered in `ModelFactory` as `goal_stacker`/`stacker`. Train: `python main.py train-target --target home_goals --model goal_stacker`. 11 unit tests pass. |
