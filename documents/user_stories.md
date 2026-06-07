# FPAI Forecast Engine User Stories

This document tracks story-level actionable items for the forecast-engine pivot. Default status is `active`. Completed stories are archived in `documents/FRAI_TECHSPEC.md` Section 20.

## Story Dependencies & Execution Order

**PHASE 1–8: All completed** — See Section 20 in `documents/FRAI_TECHSPEC.md` for the full archive.

**PHASE 8: Target-Specific Optimization & Signal Expansion**

Execution order reflects priority and dependency chain:

- **US#64** → Learning Curve Analysis (diagnostic gate — no deps, do first)
- **US#65** → Per-Target Feature Subset Sweep (permutation infra ready; generates feature lists needed by US#66)
- **US#66** → Target-Specific Feature List Configuration (wires US#65 output into config + pipeline; prerequisite for US#69)
- **US#67** → Market Odds as Features (independent, high-signal new source; can run in parallel with US#65–66)
- **US#68** → Richer Temporal Features (independent feature engineering; can run in parallel)
- **US#69** → Narrow Hyperparameter Sweep (depends on US#65 + US#66 completing)
- **US#70** → Optuna Bayesian Sweep (sweep infra upgrade; best after US#69 validates the search space shape)
- **US#71** → Ensemble / Stacking for Goals (most complex; benefits from stable features and tuned baselines)

**PAUSED / BLOCKED**
- **US#42** → Model Selection Logic (PAUSED — manual inspection for now, auto-selection later)

| # | Story Name | Description | Status | Notes |
| :--- | :--- | :--- | :--- | :--- |
| 42 | Implement Model Selection & Deployment Logic | Add logic to select "best model per target" based on primary metric across all variants. Store selection metadata in `config/model_selection.yaml`. Update forecast service to use selected models. Add CLI command `python main.py select-best-models --stage <test\|validate>`. | blocked | PAUSED: For now, manually inspect and compare models. Auto-selection logic deferred to later phase. |
| 64 | Learning Curve Analysis | For each of the 8 targets, train models on growing subsets of the chronological training set (e.g., 20%, 40%, 60%, 80%, 100%) and plot validation metric vs. training set size. If curves plateau early, the bottleneck is signal/features rather than data volume. Output: per-target learning curve chart and a written finding confirming or revising the feature-ceiling hypothesis. This gates the decision to invest in LSTM vs. more features vs. new data sources. | completed | Module: `src/utils/learning_curve.py`. CLI: `python main.py learning-curve --all_targets`. Charts/CSVs: `reports/learning_curves/`. Key findings: feature ceiling confirmed for total_goals/total_corners/btts (curves flat, <1% gain 20%→100%); away_corners most data-sensitive (−3.6%); btts shows degradation after 60% training data (temporal drift signal); no target benefits meaningfully from more raw match data. See tech spec Section 15 for full results. |
| 65 | Per-Target Feature Subset Sweep | Run permutation importance analysis across all 8 targets. Define target-specific top-N feature subsets (e.g., top-40 for classifiers, top-25 for goals excluding H2H/standings, top-20 for corners keeping H2H_CORNERS_R5). Re-run focused 288-combo sweeps using `ModelManager(feature_subset=...)`. Expected: recover goals/corners MAE lost to noise from new Phase 7 contextual features. | completed | Permutation importance run on all 8 XGBoost models. Reports saved to `reports/permutation_importance/`. result_3way used XGBoost gain-based importance (permutation scoring failed on string labels). Feature subsets defined: classifiers top-40, goals top-25, corners top-20. Output wired into US#66. |
| 66 | Target-Specific Feature List Configuration | Add a `target_features` block to `config/schema.yaml` mapping each of the 8 targets to its own named feature list. Update `ModelManager._load_selected_features()` to check per-target list first, fall back to global `selected_features`. | completed | `target_features` block added to `config/schema.yaml` with all 8 targets. `ModelManager._load_selected_features()` updated (lines 68–104) to check target name against `target_features` map before using global list. Feature counts: result_3way=40, btts=40, home/away/total_goals=25, home/away/total_corners=20. |
| 67 | Market Odds as Features | Add pre-match closing odds (odds_h, odds_d, odds_a) as predictive features. Compute implied probabilities and overround. Add to schema.yaml under MKT_ prefix. | completed | `_compute_odds_features()` added to `feature_factory.py`. 5 new features: MKT_OVERROUND, MKT_LOG_ODDS_H, MKT_LOG_ODDS_D, MKT_LOG_ODDS_A, MKT_LOG_ODDS_H_A_RATIO. Added to `schema.yaml`. MKT_IMPLIED_HOME ranks #1 for home_goals/away_corners; MKT_LOG_ODDS_D #1 for total_goals. |
| 68 | Richer Temporal Features | Add form variance (rolling std R5), EMA3 short-decay rolling averages, and win/goal/clean-sheet streak indicators. Add to schema.yaml and verify pre-match safety. | completed | `_compute_temporal_features()` added to `feature_factory.py`. 16 new features in CTX_ and OFF_/DEF_ namespaces. All shift(1)-safe. Schema updated (159 total features). Feature quality tests pass. CTX_HOME_REST_DAYS ranks #1 for btts. |
| 69 | Narrow Hyperparameter Sweep | After per-target feature subsets are defined (US#65+66), run a narrow fine-grained sweep around Phase 7 best configs. Grid: ±20% around best n_estimators, learning_rate, max_depth. | completed | Narrow grid configs created: `experiments/forecast_xgb_classifier_narrow.yaml` and `experiments/forecast_xgb_regressor_narrow.yaml`. Run via `python main.py sweep-target --target <name> --config_path experiments/forecast_xgb_classifier_narrow.yaml --sweep_stage narrow`. |
| 70 | Optuna Bayesian Sweep | Replace Cartesian grid with Optuna TPE sampler. Define search space config format mapping to Optuna's suggest_* API. Expected: 5–10x run reduction for equivalent coverage. | completed | `OptunaRunner` class added to `src/utils/sweep_runner.py`. Supports continuous search space specs (`{type: float, low, high, log}`), categorical lists, and int ranges. Optuna configs: `experiments/optuna_xgb_classifier.yaml` and `experiments/optuna_xgb_regressor.yaml` (60 trials each). CLI: `python main.py optuna-sweep --target <name> --config_path experiments/optuna_xgb_classifier.yaml`. |
| 71 | Ensemble / Stacking for Goals Targets | Implement two-stage stacking for home_goals and away_goals: XGBoost Poisson + Poisson GLM base learners, Ridge meta-learner trained on chronological OOF predictions. | completed | `GoalStackerModel` implemented in `src/models/goal_stacker.py`. OOF split: chronological 50/50. Registered in `ModelFactory` as `goal_stacker`/`stacker`. Save/load via joblib. Train via `python main.py train-target --target home_goals --model goal_stacker`. 11 unit tests pass. |
