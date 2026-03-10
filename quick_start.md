# FPAI Quick Start

This guide covers the main CLI commands in `main.py`.

## Setup

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

## Configuration

Key settings live in `config.yaml`:

- `paths.database_path` for DuckDB.
- `settings.rolling_window`, `settings.test_size`, `settings.initial_bankroll`.
- `scraper.*` for Football-Data scraping settings.

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

Run an MLflow-tracked XGBoost grid search and log bankroll curve artifacts.

```bash
python main.py experiment --experiment_name XGB_Optimization --test_season 2425
```

Options:

- `--experiment_name` (default `XGB_Optimization`)
- `--test_season` (default `time_split`)

## Common Workflow

```bash
python main.py scrape
python main.py ingest
python main.py train --model xgb
python main.py backtest --league E0 --test_season 2425
python main.py predict --league E0
```
