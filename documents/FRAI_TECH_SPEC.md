# Technical Specification (Tech Spec) - FPAI System

## 1. System Architecture
Modular design with a clear separation between data ingestion, feature engineering, and modeling.



## 2. Data Storage (DuckDB Schema)
Database: `fpai_core.db`

### Table: `raw_matches`
| Column | Type | Description |
| :--- | :--- | :--- |
| `match_id` | TEXT (PK) | Hash of Date + HomeTeam + AwayTeam |
| `league` | TEXT | League name |
| `date` | TIMESTAMP | Match kick-off time |
| `home_team` | TEXT | Normalized home team name |
| `away_team` | TEXT | Normalized away team name |
| `fthg` | INTEGER | Full-time home goals |
| `ftag` | INTEGER | Full-time away goals |
| `odds_h/d/a` | FLOAT | Closing 1X2 odds |

## 3. Engineering Standards
* **Language**: Python 3.11+ with strict Type Hints.
* **Comments**: All code documentation and docstrings must be in **English**.
* **Testing**: Mandatory `pytest` suites for all data transformation functions.
* **Validation**: Use `Pydantic` models for data ingestion to enforce schema.

## 4. Module Definitions
### 4.1 `data_loader.py`
* **Abstract Class**: `BaseLoader` defining `fetch_data()` and `clean_data()`.
* **Implementation**: `CSVLoader` (for Football-Data.co.uk) and placeholder for `APILoader`.

### 4.2 `feature_factory.py`
* **Persistence**: Computed features must be stored in a `feature_store` table in DuckDB to optimize backtesting speed.
* **Initial Features**: Rolling 5-match averages for goals, shots on target, and corners.

### 4.3 `model_manager.py`
* **Backtest Engine**: Implements a rolling window (1-year) retraining logic.
* **Model**: Initial Baseline using Logistic Regression; extensible to XGBoost.

## 5. Directory Structure
```text
/fpai
├── docs/               # PRD and Tech Spec
├── data/               # Raw CSVs and DuckDB file
├── src/
│   ├── ingestion/      # Data loading logic
│   ├── features/       # Feature engineering
│   ├── models/         # Model training and inference
│   └── utils/          # Helpers (ID hashing, odds conversion)
├── tests/              # Pytest suites
├── config.yaml         # Global configurations
└── main.py             # Execution entry point