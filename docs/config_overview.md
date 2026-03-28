# Configuration Overview

This document describes all configuration files currently used by the project.

## `config.yaml`
Global runtime configuration used by ingestion, features, training, and scraping.

**Top-level keys**
- `project_name`: Project display name used for logging and metadata.
- `version`: Human-readable project version string.
- `paths`: Filesystem locations used by the pipeline.
- `leagues`: List of leagues available to the pipeline (name/code/country metadata).
- `settings`: Core runtime settings (feature windows, data splits, bankroll defaults).
- `scraper`: Scraping configuration for Football-Data.co.uk.

**`paths`**
- `raw_data_dir`: Where CSVs are stored after scraping.
- `processed_data_dir`: Where processed/derived data may be written.
- `database_path`: DuckDB file path used by ingestion and training.

**`leagues`**
- `name`: Display name for the league.
- `code`: League code (e.g., `E0`).
- `country`: Country label.

**`settings`**
- `rolling_window`: Base rolling window length for feature engineering.
- `min_matches_required`: Minimum matches required for certain logic.
- `test_size`: Fraction of data reserved for the test split.
- `initial_bankroll`: Default bankroll for backtesting.

**`scraper`**
- `league_page_url`: Root page used to discover season CSV links.
- `limit_seasons`: Max number of seasons to download.
- `timeout_seconds`: HTTP timeout for requests.
- `leagues`: League codes to include (e.g., `["E0"]`).
- `start_year`: Oldest season start year to keep.

## `experiments/xgb_search.yaml`
Experiment configuration for XGBoost hyperparameter grid search.

**Top-level keys**
- `experiment_name`: MLflow experiment name.
- `model_type`: Model type string passed to the factory (e.g., `xgboost`).
- `league`: League code to filter training data (e.g., `E0`).
- `test_season`: Target season label for evaluation (e.g., `2526`).
- `training_start_year`: Earliest season year to include in training.
- `grid_search`: Hyperparameter grid; every combination becomes a separate MLflow run.
- `fixed_params`: Parameters applied to all runs (not part of the grid).
- `backtest_config`: Backtesting parameters for EV filtering and bankroll simulation.

**`grid_search` (example keys)**
- `n_estimators`: Number of boosting rounds.
- `max_depth`: Tree depth.
- `learning_rate`: Boosting learning rate.
- `subsample`: Row subsampling ratio.
- `colsample_bytree`: Column subsampling ratio.

**`fixed_params`**
- `random_state`: Reproducibility seed.
- `tree_method`: XGBoost tree method (e.g., `hist`).
- `early_stopping_rounds`: Early stopping patience.
- `eval_metric`: Evaluation metric used during training.

**`backtest_config`**
- `ev_threshold`: Minimum expected value to place a bet.
- `initial_bankroll`: Starting bankroll.
- `bet_size`: Fixed bet amount per wager.

## `config/schema.yaml`
Training schema configuration used to explicitly select features for model training.

**Top-level keys**
- `training_setup`: Group for training-related schema settings.

**`training_setup`**
- `selected_features`: Ordered whitelist of feature column names to use in training. Any missing feature in the training DataFrame will raise a `ValueError`.
