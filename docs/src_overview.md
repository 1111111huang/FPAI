# src Overview

## Purpose
The `src/` package contains the data ingestion, feature engineering, model training, and strategy/backtesting logic for FPAI. It is organized by domain (ingestion, features, models, strategy, logic, utils) and is driven by `config.yaml` settings.

## High-Level Architecture
- **Scrape**: Download league CSVs from Football-Data.co.uk into `data/raw/`.
- **Ingest**: Parse CSVs, standardize team names, validate core fields, and upsert into DuckDB (`raw_matches`).
- **Feature Engineering**: Compute rolling team stats, discipline/shot metrics, rest days, and market-implied probabilities, then upsert into DuckDB (`feature_store`).
- **Training/Experimentation**: Join `raw_matches` and `feature_store`, split by time, train models, log to MLflow, and save artifacts.
- **Prediction/Backtesting**: Produce probabilities, apply EV thresholds, and simulate betting performance.

## Module Summaries

### `src/ingestion/`
- **`scraper.py`**: Scrapes season CSV URLs and downloads files into `data/raw/`. Supports `force` mode to clear existing CSVs. Handles league filtering, season windows, and overwriting current or forced seasons.
- **`data_loader.py`**: Loads CSVs, renames and normalizes columns, standardizes team names, validates core schema, and inserts into DuckDB. Tracks processed files to skip unchanged inputs (unless forced). Ensures required DB schema and optional columns.
- **`match_schema.py`**: Pydantic schema for validating minimal match inputs and mapping league code to tier.
- **`schema.py`**: League-to-tier mapping helpers.
- **`__init__.py`**: Exposes ingestion classes.

### `src/features/`
- **`feature_factory.py`**: Computes rolling stats (3/5 match windows) for offense/defense/discipline, rest days, and market-implied probabilities. Joins to `raw_matches` and writes `feature_store` with schema evolution.
- **`feature_inventory.md`**: Human-readable catalog of engineered features and their definitions.
- **`__init__.py`**: Exposes feature utilities.

### `src/models/`
- **`base_model.py`**: Model interfaces and concrete implementations (Logistic Regression, Random Forest, XGBoost). XGBoost supports early stopping with a validation split.
- **`model_factory.py`**: Factory for constructing models by type name.
- **`model_manager.py`**: Prepares training data (DuckDB join), time-based train/test split, optional NaN handling (XGBoost supports NaNs), MLflow logging, evaluation metrics, backtest integration, and artifact versioning.
- **`__init__.py`**: Exposes model classes and factory.

### `src/strategy/`
- **`strategy_engine.py`**: Computes expected value (EV) and filters recommendations above a threshold.
- **`backtester.py`**: Runs a historical backtest using predictions and DuckDB outcomes, logs metrics, and computes ROI/win-rate/drawdown.
- **`__init__.py`**: Exposes strategy utilities.

### `src/logic/`
- **`target_resolver.py`**: Defines label creation and payout logic for supported targets (currently `home_win`).

### `src/utils/`
- **`config_loader.py`**: Loads and validates `config.yaml` into typed settings with caching.
- **`config.py`**: Convenience wrappers to load config as dict and extract DB path.
- **`db_manager.py`**: DuckDB connection manager with context manager support.
- **`helpers.py`**: Match ID generation and team name standardization for consistent keys.
- **`logger.py`**: Centralized logging setup.
- **`__init__.py`**: Exposes shared utilities.

## Data Stores
- **DuckDB**: Primary storage for `raw_matches` (ingested data) and `feature_store` (engineered features).
- **MLflow**: Experiment tracking, metrics logging, and model artifact storage.

## Entry Points (Outside `src/`)
- **`main.py`**: CLI that wires together scraping, ingestion, feature building, training, prediction, and backtesting.

