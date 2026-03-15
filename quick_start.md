# FPAI Quick Start

This guide covers the main CLI commands in `main.py`, configuration files, and how to run experiments with MLflow.

## Setup

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

## Configuration

Key runtime settings live in `config.yaml`:

- `paths.database_path` for DuckDB.
- `settings.rolling_window`, `settings.test_size`, `settings.initial_bankroll`.
- `scraper.*` for Football-Data scraping settings.

Experiment settings live in a YAML config (default: `experiments/xgb_search.yaml`). It contains:

- `experiment_name`: MLflow experiment name.
- `model_type`: model string (for ModelFactory), typically `xgboost`.
- `league`, `test_season`: used for MLflow tags.
- `grid_search`: grid search parameter space.
- `fixed_params`: non-search parameters for the model.
- `backtest_config`: ev threshold, bankroll, and bet size for backtesting.

Example `experiments/xgb_search.yaml`:

```yaml
experiment_name: "XGBoost_Hyperparameter_Search_V1"
model_type: "xgboost"
league: "E0"
test_season: "2526"
grid_search:
  n_estimators: [100, 300, 500]
  max_depth: [3, 4, 5]
  learning_rate: [0.01, 0.05, 0.1]
fixed_params:
  random_state: 42
  tree_method: "hist"
  early_stopping_rounds: 10
  eval_metric: "logloss"
backtest_config:
  ev_threshold: 0.05
  initial_bankroll: 1000
  bet_size: 10
```

## Commands

All commands use:

```bash
python main.py <command> [options]
```

### scrape

Download multi-season CSVs into `data/raw/`.

```bash
python main.py scrape
```

### ingest

Parse all CSVs in `data/raw/`, load into DuckDB, then compute and store features.

```bash
python main.py ingest
```

### train

Train and save a model artifact.

```bash
python main.py train --model lr
python main.py train --model xgb
```

Options:

- `--model` (default `lr`): `lr`, `xgb`, `xgboost`

### predict

Generate value-bet recommendations for recent matches (last 7 days).

```bash
python main.py predict --league E0
```

Options:

- `--league` (default `E0`): `E0`, `E1`, `E2`

### backtest

Run a strict season-isolated backtest.

```bash
python main.py backtest --league E0 --test_season 2425 --ev_threshold 0.05
```

Options:

- `--league` (default `E0`)
- `--test_season` (required, e.g. `2425` for 2024/2025)
- `--ev_threshold` (default `0.05`)
- `--rolling_retrain` (optional flag)

### experiment

Run an MLflow-tracked grid search and log bankroll curve artifacts. The experiment reads its config
from the YAML file you pass to `--config_path`.

```bash
python main.py experiment --config_path experiments/xgb_search.yaml
python main.py experiment --experiment_name XGB_Optimization --config_path experiments/xgb_search.yaml
```

Options:

- `--experiment_name` (default `XGB_Optimization`)
- `--config_path` (default `experiments/xgb_search.yaml`)
- `--test_season` (default `time_split`)

What you get from each experiment run:

- MLflow params: search + fixed params, and `features`.
- MLflow metrics: ROI, Win Rate, Max Drawdown.
- MLflow artifacts: bankroll curve plot.
- Summary table printed in the terminal for the top 3 parameter sets by ROI.

To view MLflow runs:

```bash
mlflow ui
```

## Common Workflow

```bash
python main.py scrape
python main.py ingest
python main.py train --model xgb
python main.py backtest --league E0 --test_season 2425
python main.py predict --league E0
python main.py experiment --config_path experiments/xgb_search.yaml
```
