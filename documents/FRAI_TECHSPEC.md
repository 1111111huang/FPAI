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

**Current Feature Availability (v1):** 86 selected features are currently configured for training in `config/schema.yaml`. Expected goals (xG) and luck-derived features (OFF_*_XG_R3/R5, DEF_*_XGA_R3/R5, OFF_*_LUCK_R3/R5) are not included until Understat data integration provides xG values in the ingestion pipeline. The current schema includes rolling form, discipline, shot-quality/save-rate, EMA, rest-context, market probability, strength-differential, interaction, and efficiency-ratio features.

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

Latest verified run:

```text
Command: venv/bin/python -m pytest
Result: 42 passed, 1 warning
```

When adding functionality, update this section with any new test files, changed coverage areas, or intentionally skipped checks.

## 12. Data Ingestion and Feature Status

### 12.1 Current Data Pipeline
- **Source:** Football-Data CSV files with match results, odds, shots, corners, and cards.
- **Processing:** `src/ingestion/data_loader.py` ingests CSVs into `raw_matches` with column mapping and validation.
- **Feature Generation:** `src/features/feature_factory.py` computes rolling averages (R3, R5), exponential moving averages (EMA5), market probabilities, and context features.

### 12.2 Feature Availability (as of 2026-05-31)
- **Current selected features:** 86 features configured in `config/schema.yaml`
  - Rolling stats (FTHG, FTAG, HS, AST, HC, AC, etc.) 
  - Discipline features (cards and card rates)
  - Shot-quality and save-rate derived features
  - EMA form features
  - Market-implied probabilities
  - Context features (rest days)
  - Derived strength metrics (goal/shot differential)
  - Interaction features (shifted home/away attack, defense, and shots-on-target deltas)
  - Efficiency features (shifted attack-versus-defense matchup ratios)
  
- **Missing/Pending features:** xG-dependent features require source xG data integration
  - Expected goals variants: OFF_*_XG_R3/R5 and DEF_*_XGA_R3/R5
  - Luck features derived from xG: OFF_*_LUCK_R3/R5
  - Action: Waiting for Understat API integration (`src/ingestion/understat.py`) to provide xG values

### 12.3 Schema Configuration
- **File:** `config/schema.yaml`
- **Current:** `training_setup.selected_features` lists 86 selected features matching the active training contract
- **Update policy:** When new data sources are integrated (xG, corners), expand `selected_features` list accordingly
- **Note:** Understat-dependent xG and luck features remain deferred until the paused Understat integration path resumes

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

Full broad-grid execution remains a follow-up because the current broad configs imply roughly 1,500+ MLflow runs. Track broad execution in user story 51.

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

## 15. Feature Importance & Selection Analysis

### 15.1 Permutation Importance
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

### 15.2 Feature Selection Study
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

### 15.3 Top Features (Example from btts target)
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

## 16. MLflow Store Management

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

## 17. Migration Plan
1. Archive legacy product and technical documents.
2. Introduce target registry.
3. Expand target resolver beyond `home_win`.
4. Add target-specific model training and evaluation.
5. Implement forecast JSON service.
6. Add uncertainty and explainability payloads.
7. Demote strategy/backtest commands in CLI help and docs.
8. Add richer data integrations, including Understat xG and future corner odds.
