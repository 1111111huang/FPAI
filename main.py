"""CLI entry point for the FPAI Prototype 1 pipeline."""

from __future__ import annotations

import argparse
from datetime import datetime
import itertools
import json
from pathlib import Path
from tempfile import TemporaryDirectory

import mlflow
from mlflow.exceptions import MlflowException
import pandas as pd
import matplotlib.pyplot as plt
import yaml

from src.features.feature_factory import FeatureFactory
from src.evaluation import run_diagnostics
from src.evaluation.mlflow_cleanup import MLflowStoreCleanup, save_cleanup_report
from src.forecast import ForecastService
from src.ingestion import CSVLoader, FootballDataScraper
from src.logic.target_registry import get_target_definition, list_target_definitions
from src.models import (
    LRModel,
    ModelFactory,
    ModelManager,
    RandomForestModel,
    RandomForestRegressorModel,
    XGBoostModel,
    XGBoostRegressorModel,
)
from src.strategy import Backtester, StrategyEngine
from src.utils import DuckDBManager, configure_logger, get_logger
from src.utils.config_loader import AppSettings, settings
from src.utils.feature_importance import PermutationImportanceAnalyzer
from src.utils.learning_curve import LearningCurveAnalyzer, run_all_targets, summarise_findings
from src.utils.model_comparison import ModelComparison
from src.utils.sweep_runner import OptunaRunner, SweepRunner

LOGGER = get_logger(__name__)
FEATURE_COLUMNS = [
    "home_avg_goals_scored",
    "home_avg_goals_conceded",
    "away_avg_goals_scored",
    "away_avg_goals_conceded",
    "is_cold_start",
    "relative_tier_change",
    "market_prob_h",
    "elo_rating_diff",
    "home_advantage_trend",
]
LEAGUE_LABELS = {
    "E0": "Premier League",
    "E1": "Championship",
    "E2": "League One",
}
MODEL_REGISTRY = {
    "lr": LRModel,
    "random_forest": RandomForestModel,
    "xgb": XGBoostModel,
    "xgboost": XGBoostModel,
    "xgb_regressor": XGBoostRegressorModel,
    "xgboost_regressor": XGBoostRegressorModel,
    "rf_regressor": RandomForestRegressorModel,
    "goal_stacker": None,  # handled via ModelFactory
    "stacker": None,
}


def _mlflow_log_model_compat(model, flavor: str, name: str = "model") -> None:
    """Log MLflow model with name= (new) or artifact_path (legacy) compatibility."""
    if flavor == "xgboost":
        try:
            mlflow.xgboost.log_model(model, name=name)
            return
        except TypeError:
            mlflow.xgboost.log_model(model, name)
            return
    if flavor == "sklearn":
        try:
            mlflow.sklearn.log_model(model, name=name)
            return
        except TypeError:
            mlflow.sklearn.log_model(model, name)
            return
    raise ValueError(f"Unsupported MLflow model flavor: {flavor}")


def _build_parser() -> argparse.ArgumentParser:
    """Create CLI parser with scrape, ingest, train, predict, and backtest subcommands."""
    parser = argparse.ArgumentParser(description="FPAI command line interface")
    subparsers = parser.add_subparsers(dest="command", required=True)
    scrape_parser = subparsers.add_parser("scrape", help="Download latest multi-season CSV files to raw directory")
    scrape_parser.add_argument(
        "--force",
        action="store_true",
        help="Re-download and overwrite all selected CSV files (ignore existing files).",
    )
    ingest_parser = subparsers.add_parser("ingest", help="Ingest CSV data and pre-compute features")
    ingest_parser.add_argument(
        "--force",
        action="store_true",
        help="Re-ingest all CSV files and overwrite existing database rows.",
    )
    train_parser = subparsers.add_parser("train", help="Train model and save artifact")
    train_parser.add_argument(
        "--model",
        type=str,
        default="lr",
        choices=sorted(MODEL_REGISTRY.keys()),
        help="Model type to train (default: lr).",
    )
    train_parser.add_argument(
        "--target_type",
        type=str,
        default="home_win",
        help="Target type for training labels (default: home_win).",
    )
    train_target_parser = subparsers.add_parser("train-target", help="Train one forecast target model")
    train_target_parser.add_argument(
        "--target",
        type=str,
        required=True,
        choices=sorted(definition.name for definition in list_target_definitions()),
        help="Forecast target to train.",
    )
    train_target_parser.add_argument(
        "--model",
        type=str,
        default=None,
        choices=sorted(MODEL_REGISTRY.keys()),
        help="Optional model override. Defaults to lr for classification and rf_regressor for regression.",
    )
    train_suite_parser = subparsers.add_parser("train-forecast-suite", help="Train all registered forecast target models")
    train_suite_parser.add_argument(
        "--targets",
        type=str,
        nargs="*",
        default=None,
        choices=sorted(definition.name for definition in list_target_definitions()),
        help="Optional subset of forecast targets to train.",
    )
    forecast_parser = subparsers.add_parser("forecast", help="Emit structured forecast JSON")
    forecast_parser.add_argument(
        "--league",
        type=str,
        default=None,
        help="Optional league code filter.",
    )
    forecast_parser.add_argument(
        "--match_id",
        type=str,
        nargs="*",
        default=None,
        help="Optional match_id values to forecast.",
    )
    forecast_parser.add_argument(
        "--target",
        type=str,
        nargs="*",
        default=None,
        choices=sorted(definition.name for definition in list_target_definitions() if definition.name != "home_win"),
        help="Optional forecast target subset.",
    )
    forecast_parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional maximum number of matches to return.",
    )
    predict_parser = subparsers.add_parser("predict", help="[legacy] Load latest model and print value bets")
    predict_parser.add_argument(
        "--league",
        type=str,
        default="E0",
        help="League code filter for prediction output (default: E0).",
    )
    predict_parser.add_argument(
        "--run_id",
        type=str,
        default=None,
        help="Optional MLflow run ID to load a specific model.",
    )
    backtest_parser = subparsers.add_parser("backtest", help="[legacy] Run historical strategy backtest")
    backtest_parser.add_argument(
        "--ev_threshold",
        type=float,
        default=0.05,
        help="Minimum EV threshold to place bets (default: 0.05).",
    )
    backtest_parser.add_argument(
        "--league",
        type=str,
        default="E0",
        help="League code filter for backtest (default: E0).",
    )
    backtest_parser.add_argument(
        "--test_season",
        type=str,
        required=True,
        help="Test season in YYZZ format (example: 2425 for 2024/2025).",
    )
    backtest_parser.add_argument(
        "--rolling_retrain",
        action="store_true",
        help="Retrain model before each backtest match using all prior data.",
    )
    backtest_parser.add_argument(
        "--run_id",
        type=str,
        default=None,
        help="Optional MLflow run ID to load a specific model.",
    )
    experiment_parser = subparsers.add_parser("experiment", help="[legacy] Run MLflow experiment grid search")
    experiment_parser.add_argument(
        "--experiment_name",
        type=str,
        default="XGB_Optimization",
        help="MLflow experiment name (default: XGB_Optimization).",
    )
    experiment_parser.add_argument(
        "--config_path",
        type=str,
        default="experiments/xgb_search.yaml",
        help="Path to experiment config YAML (default: experiments/xgb_search.yaml).",
    )
    experiment_parser.add_argument(
        "--test_season",
        type=str,
        default="time_split",
        help="Tag value for test season used (default: time_split).",
    )
    experiment_parser.add_argument(
        "--target_type",
        type=str,
        default="home_win",
        help="Target type for training labels (default: home_win).",
    )
    experiment_target_parser = subparsers.add_parser(
        "experiment-target",
        help="Run registry-aware MLflow forecast experiments for one target",
    )
    experiment_target_parser.add_argument(
        "--target",
        type=str,
        required=True,
        choices=sorted(definition.name for definition in list_target_definitions()),
        help="Forecast target to experiment on.",
    )
    experiment_target_parser.add_argument(
        "--config_path",
        type=str,
        required=True,
        help="Path to target experiment YAML config.",
    )
    experiment_target_parser.add_argument(
        "--experiment_name",
        type=str,
        default=None,
        help="Optional MLflow experiment name override.",
    )
    experiment_target_parser.add_argument(
        "--max_runs",
        type=int,
        default=None,
        help="Optional cap for smoke-testing broad grids.",
    )
    compare_parser = subparsers.add_parser(
        "compare-models",
        help="Compare MLflow forecast model runs for one target",
    )
    compare_parser.add_argument(
        "--target",
        type=str,
        required=True,
        choices=sorted(definition.name for definition in list_target_definitions()),
        help="Forecast target to compare.",
    )
    compare_parser.add_argument(
        "--experiment_name",
        type=str,
        default=None,
        help="Optional MLflow experiment name. Defaults to all experiments.",
    )
    compare_parser.add_argument(
        "--output_path",
        type=str,
        default=None,
        help="Optional output report path.",
    )
    compare_parser.add_argument(
        "--format",
        type=str,
        default="csv",
        choices=["csv", "json", "html"],
        help="Report format.",
    )
    sweep_parser = subparsers.add_parser(
        "sweep-target",
        help="Run a systematic target sweep with MLflow logging",
    )
    sweep_parser.add_argument(
        "--target",
        type=str,
        required=True,
        choices=sorted(definition.name for definition in list_target_definitions()),
        help="Forecast target to sweep.",
    )
    sweep_parser.add_argument(
        "--config_path",
        type=str,
        required=True,
        help="Path to target experiment YAML config.",
    )
    sweep_parser.add_argument(
        "--sweep_stage",
        type=str,
        default=None,
        choices=["smoke", "broad", "narrow", "final"],
        help="Override sweep stage tag.",
    )
    sweep_parser.add_argument(
        "--experiment_name",
        type=str,
        default=None,
        help="Optional MLflow experiment name override.",
    )
    sweep_parser.add_argument(
        "--max_runs",
        type=int,
        default=None,
        help="Optional cap for smoke-testing broad grids.",
    )
    diagnose_parser = subparsers.add_parser(
        "diagnose-model",
        help="Generate evaluation diagnostics for one model artifact",
    )
    diagnose_parser.add_argument(
        "--target",
        type=str,
        required=True,
        choices=sorted(definition.name for definition in list_target_definitions()),
        help="Forecast target to diagnose.",
    )
    diagnose_parser.add_argument(
        "--model_path",
        type=str,
        required=True,
        help="Path to local model artifact.",
    )
    diagnose_parser.add_argument(
        "--output_path",
        type=str,
        default="reports/diagnostics.json",
        help="Path for diagnostics JSON output.",
    )
    
    # Permutation importance analysis
    importance_parser = subparsers.add_parser(
        "permutation-importance",
        help="Run permutation importance analysis on a trained model",
    )
    importance_parser.add_argument(
        "--target",
        type=str,
        required=True,
        choices=sorted(definition.name for definition in list_target_definitions()),
        help="Forecast target to analyze.",
    )
    importance_parser.add_argument(
        "--model_path",
        type=str,
        required=True,
        help="Path to trained model artifact.",
    )
    importance_parser.add_argument(
        "--n_repeats",
        type=int,
        default=10,
        help="Number of permutation repeats.",
    )
    importance_parser.add_argument(
        "--output_dir",
        type=str,
        default="reports",
        help="Directory for importance reports.",
    )
    
    # Understat xG fetch
    understat_parser = subparsers.add_parser(
        "fetch-understat",
        help="Fetch xG data from understat.com and update raw_matches",
    )
    understat_parser.add_argument(
        "--league",
        type=str,
        default="E0",
        help="Football-Data league code (default: E0 = English Premier League).",
    )
    understat_parser.add_argument(
        "--from_season",
        type=int,
        default=None,
        help="First season start year to fetch (default: auto-detected from raw_matches).",
    )
    understat_parser.add_argument(
        "--to_season",
        type=int,
        default=None,
        help="Last season start year to fetch (default: auto-detected from raw_matches).",
    )
    understat_parser.add_argument(
        "--delay",
        type=float,
        default=1.5,
        help="Polite delay in seconds between requests (default: 1.5).",
    )
    understat_parser.add_argument(
        "--rebuild_features",
        action="store_true",
        default=True,
        help="Rebuild feature store after updating xG (default: True).",
    )

    # Learning curve analysis
    lc_parser = subparsers.add_parser(
        "learning-curve",
        help="Train on growing data subsets to diagnose feature ceiling vs. data ceiling",
    )
    lc_target_group = lc_parser.add_mutually_exclusive_group(required=True)
    lc_target_group.add_argument(
        "--target",
        type=str,
        choices=sorted(definition.name for definition in list_target_definitions() if definition.name != "home_win"),
        help="Single forecast target to analyse.",
    )
    lc_target_group.add_argument(
        "--all_targets",
        action="store_true",
        help="Run analysis for all 8 forecast targets and produce a combined chart.",
    )
    lc_parser.add_argument(
        "--output_dir",
        type=str,
        default="reports/learning_curves",
        help="Directory for output CSVs and charts (default: reports/learning_curves).",
    )

    # Optuna Bayesian sweep (US#70)
    optuna_parser = subparsers.add_parser(
        "optuna-sweep",
        help="Run a Bayesian hyperparameter sweep using Optuna TPE sampler",
    )
    optuna_parser.add_argument(
        "--target",
        type=str,
        required=True,
        choices=sorted(definition.name for definition in list_target_definitions()),
        help="Forecast target to sweep.",
    )
    optuna_parser.add_argument(
        "--config_path",
        type=str,
        required=True,
        help="Path to experiment YAML config (use optuna_search instead of grid_search).",
    )
    optuna_parser.add_argument(
        "--n_trials",
        type=int,
        default=None,
        help="Number of Optuna trials (overrides config n_trials, default 50).",
    )
    optuna_parser.add_argument(
        "--sweep_stage",
        type=str,
        default=None,
        choices=["smoke", "broad", "narrow", "final", "optuna"],
        help="Override sweep stage tag.",
    )
    optuna_parser.add_argument(
        "--experiment_name",
        type=str,
        default=None,
        help="Optional MLflow experiment name override.",
    )

    # MLflow cleanup
    cleanup_parser = subparsers.add_parser(
        "cleanup-mlflow",
        help="Clean up malformed MLflow experiments in local file store",
    )
    cleanup_parser.add_argument(
        "--strategy",
        type=str,
        choices=["recover", "remove", "backup_and_remove"],
        default="recover",
        help="How to handle malformed experiments.",
    )
    cleanup_parser.add_argument(
        "--backup",
        action="store_true",
        default=True,
        help="Backup before destructive operations.",
    )
    cleanup_parser.add_argument(
        "--mlruns_dir",
        type=str,
        default="mlruns",
        help="Path to MLflow runs directory.",
    )
    cleanup_parser.add_argument(
        "--report_only",
        action="store_true",
        help="Only report malformed experiments without fixing them.",
    )
    return parser


def _get_latest_model_path(model_dir: Path) -> Path:
    """Return the latest LR model artifact from disk."""
    candidates = sorted(model_dir.glob("lr_v*_*.joblib"), key=lambda path: path.stat().st_mtime)
    if not candidates:
        raise FileNotFoundError(f"No saved model found in {model_dir}")
    return candidates[-1]


def _get_model_uri(league: str, run_id: str | None = None) -> tuple[str, mlflow.entities.Run]:
    """Resolve model URI and run metadata for a given league and optional run_id."""
    league_code = league.upper()
    if run_id:
        try:
            run = mlflow.get_run(run_id)
        except MlflowException as exc:
            raise ValueError(f"MLflow run_id not found: {run_id}") from exc
        run_league = (run.data.tags or {}).get("league", "").upper()
        if run_league and run_league != league_code:
            raise ValueError(
                f"Run {run_id} is tagged for league {run_league}, not {league_code}."
            )
        run_model_type = (run.data.tags or {}).get("model_type", "").lower()
        if run_model_type and run_model_type not in {"xgboost", "logistic_regression", "random_forest", "lr", "xgb"}:
            raise ValueError(f"Run {run_id} uses unsupported model_type '{run_model_type}'.")
        return f"runs:/{run_id}/model", run

    runs_df = mlflow.search_runs(order_by=["metrics.roi DESC"])
    if runs_df.empty:
        raise ValueError("No MLflow runs found.")
    if "metrics.roi" in runs_df.columns:
        runs_df = runs_df[runs_df["metrics.roi"].notna()]
    if "tags.league" in runs_df.columns:
        runs_df = runs_df[runs_df["tags.league"].str.upper() == league_code]
    if runs_df.empty:
        raise ValueError(f"No MLflow runs found for league {league_code}.")

    best_run_id = runs_df.iloc[0]["run_id"]
    run = mlflow.get_run(best_run_id)
    return f"runs:/{best_run_id}/model", run


def _check_feature_consistency(run: mlflow.entities.Run) -> None:
    """Ensure model features recorded in MLflow match current FEATURE_COLUMNS."""
    features_param = run.data.params.get("features", "")
    recorded = [item.strip() for item in features_param.split(",") if item.strip()]
    if not recorded:
        raise ValueError("Run does not record 'features' metadata; cannot verify consistency.")
    if recorded != FEATURE_COLUMNS:
        raise ValueError(
            "Feature mismatch between model metadata and current pipeline. "
            f"Recorded={recorded}, Current={FEATURE_COLUMNS}"
        )


def _fetch_feature_joined_matches(db_manager: DuckDBManager, days: int | None = None) -> pd.DataFrame:
    """Fetch matches joined with feature_store, optionally limited to recent days."""
    date_filter = ""
    if days is not None:
        date_filter = f"WHERE r.date >= m.max_date - INTERVAL {int(days)} DAY"

    query = f"""
        WITH max_date_cte AS (
            SELECT MAX(date) AS max_date FROM raw_matches
        )
        SELECT
            r.match_id,
            r.league,
            r.home_team,
            r.away_team,
            r.odds_h,
            r.date,
            r.fthg,
            r.ftag,
            f.home_avg_goals_scored,
            f.home_avg_goals_conceded,
            f.away_avg_goals_scored,
            f.away_avg_goals_conceded,
            f.is_cold_start,
            f.relative_tier_change,
            f.market_prob_h,
            f.elo_rating_diff,
            f.home_advantage_trend
        FROM raw_matches r
        INNER JOIN feature_store f ON r.match_id = f.match_id
        CROSS JOIN max_date_cte m
        {date_filter}
        ORDER BY r.date, r.match_id
    """
    with db_manager.connection() as conn:
        return conn.execute(query).fetchdf()


def _build_prediction_frame(model: LRModel, source_df: pd.DataFrame) -> pd.DataFrame:
    """Generate prediction DataFrame required by StrategyEngine and Backtester."""
    inference_df = source_df.dropna(subset=FEATURE_COLUMNS + ["odds_h"]).copy()
    if inference_df.empty:
        return pd.DataFrame(
            columns=["match_id", "league", "home_team", "away_team", "predicted_home_win_prob", "odds_h"]
        )

    probabilities = model.predict_proba(inference_df[FEATURE_COLUMNS])
    predicted_home_win_prob = probabilities[:, 1] if probabilities.ndim == 2 else probabilities.ravel()

    return pd.DataFrame(
        {
            "match_id": inference_df["match_id"].values,
            "league": inference_df["league"].values if "league" in inference_df.columns else "",
            "home_team": inference_df["home_team"].values,
            "away_team": inference_df["away_team"].values,
            "predicted_home_win_prob": predicted_home_win_prob,
            "odds_h": inference_df["odds_h"].values,
        }
    )


def _parse_season_bounds(test_season: str) -> tuple[pd.Timestamp, pd.Timestamp]:
    """Parse YYZZ season code into [start, end] timestamps for July-to-June season windows."""
    normalized = test_season.strip()
    if len(normalized) != 4 or not normalized.isdigit():
        raise ValueError("test_season must be a 4-digit code like '2425'.")

    start_suffix = int(normalized[:2])
    end_suffix = int(normalized[2:])
    expected_end = (start_suffix + 1) % 100
    if end_suffix != expected_end:
        raise ValueError(f"Invalid season code '{test_season}'. Expected trailing year {expected_end:02d}.")

    start_year = 2000 + start_suffix
    end_year = 2000 + end_suffix
    start = pd.Timestamp(datetime(start_year, 7, 1, 0, 0, 0))
    end = pd.Timestamp(datetime(end_year, 6, 30, 23, 59, 59))
    return start, end


def _prepare_backtest_frame(source_df: pd.DataFrame) -> pd.DataFrame:
    """Return cleaned supervised frame with target for backtesting."""
    if source_df.empty:
        return pd.DataFrame()

    required = FEATURE_COLUMNS + ["match_id", "league", "home_team", "away_team", "date", "fthg", "ftag", "odds_h"]
    df = source_df.dropna(subset=required).copy()
    if df.empty:
        return pd.DataFrame()

    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["date"])
    if df.empty:
        return pd.DataFrame()

    df["target"] = (df["fthg"].astype(int) > df["ftag"].astype(int)).astype(int)
    df = df.sort_values(["date", "match_id"]).reset_index(drop=True)
    return df


def run_scrape(app_settings: AppSettings, force: bool = False) -> None:
    """Run scraping only: download season CSV files to data/raw directory."""
    LOGGER.info("Executing command: scrape")
    scraper = FootballDataScraper(
        league_page_url=app_settings.scraper.league_page_url,
        timeout_seconds=app_settings.scraper.timeout_seconds,
    )
    downloaded = scraper.download_all(
        limit_seasons=app_settings.scraper.limit_seasons,
        leagues=app_settings.scraper.leagues,
        start_year=app_settings.scraper.start_year,
        force=force,
    )
    LOGGER.info("Scrape complete | files_downloaded=%s", downloaded)


def run_fetch_understat(
    app_settings: AppSettings,
    db_manager: DuckDBManager,
    league: str = "E0",
    from_season: int | None = None,
    to_season: int | None = None,
    delay: float = 1.5,
    rebuild_features: bool = True,
) -> None:
    """Fetch xG from understat.com, update raw_matches, and optionally rebuild features."""
    from src.ingestion.understat import update_raw_matches_xg
    from src.ingestion.understat_fetcher import fetch_seasons_range

    LOGGER.info("Executing command: fetch-understat | league=%s", league)

    # Auto-detect season range from raw_matches dates if not provided.
    with db_manager.connection() as conn:
        bounds = conn.execute(
            "SELECT YEAR(MIN(date)), YEAR(MAX(date)) FROM raw_matches"
        ).fetchone()

    if bounds is None or bounds[0] is None:
        LOGGER.error("raw_matches is empty — run ingest first.")
        return

    # Season start year = calendar year of August (EPL seasons start Aug-May).
    # A match in Jan 2024 belongs to season 2023; match in Aug 2023 also belongs to 2023.
    # Simple heuristic: min_year - 1 to max_year - 1 covers all start years.
    detected_from = (from_season or bounds[0]) - 1
    detected_to = to_season or (bounds[1] - 1)
    LOGGER.info(
        "Fetching seasons %d – %d for league %s", detected_from, detected_to, league
    )

    understat_df = fetch_seasons_range(
        league=league,
        from_season=detected_from,
        to_season=detected_to,
        delay=delay,
    )

    if understat_df.empty:
        LOGGER.error("No Understat data returned — check league/season args and network.")
        return

    LOGGER.info("Fetched %d Understat match records total.", len(understat_df))

    result = update_raw_matches_xg(understat_df, db_manager)
    LOGGER.info(
        "xG update | matched=%d | updated=%d | unmatched=%d",
        result["matched"],
        result["updated"],
        result["unmatched"],
    )

    if rebuild_features:
        LOGGER.info("Rebuilding feature store with xG data...")
        factory = FeatureFactory()
        features_df = factory.compute_rolling_stats(
            window=app_settings.settings.rolling_window
        )
        factory.save_features(features_df)
        LOGGER.info("Feature store rebuilt with xG features.")


def run_ingest(app_settings: AppSettings, db_manager: DuckDBManager, force: bool = False) -> None:
    """Run directory batch ingestion and feature pre-computation pipeline."""
    LOGGER.info("Executing command: ingest")

    raw_dir = Path(app_settings.paths.raw_data_dir)
    if not raw_dir.exists():
        LOGGER.error("Raw data directory not found: %s", raw_dir)
        return

    if force:
        with db_manager.connection() as conn:
            conn.execute("DELETE FROM processed_files")
            conn.execute("DELETE FROM feature_store")
            conn.execute("DELETE FROM raw_matches")
        LOGGER.info("Force enabled: cleared processed_files, feature_store, and raw_matches.")

    loader = CSVLoader()
    loader.process_directory(pattern="*.csv", force=force)

    factory = FeatureFactory()
    features_df = factory.compute_rolling_stats(window=app_settings.settings.rolling_window)
    factory.save_features(features_df)

    with db_manager.connection() as conn:
        raw_count = conn.execute("SELECT COUNT(*) FROM raw_matches").fetchone()
        feature_count = conn.execute("SELECT COUNT(*) FROM feature_store").fetchone()

    total_raw = int(raw_count[0]) if raw_count is not None else 0
    total_features = int(feature_count[0]) if feature_count is not None else 0
    LOGGER.info("Ingest complete | raw_matches=%s | feature_store=%s", total_raw, total_features)


def run_train(model_name: str, target_type: str) -> None:
    """Train selected model and persist artifact."""
    LOGGER.info("Executing command: train")
    normalized = model_name.strip().lower()
    model_cls = MODEL_REGISTRY.get(normalized)
    if model_cls is None:
        valid_models = ", ".join(sorted(MODEL_REGISTRY.keys()))
        raise ValueError(f"Unsupported model '{model_name}'. Available options: {valid_models}")
    LOGGER.info("Selected model type: %s", normalized)

    model_manager = ModelManager(model=model_cls(), target_config={"target_type": target_type})
    model_path = model_manager.run_pipeline()
    LOGGER.info("Model saved to %s", model_path)


def _default_model_for_target(target_name: str) -> str:
    """Return the default model key for a target definition."""
    definition = get_target_definition(target_name)
    if definition.task_type == "regression":
        return "rf_regressor"
    return "lr"


def _xgb_params_for_target(target_name: str, model_key: str) -> dict:
    """Return XGBoost objective/eval_metric params appropriate for the target."""
    definition = get_target_definition(target_name)
    if model_key not in {"xgb", "xgboost"}:
        return {}
    if definition.task_type == "multiclass_classification":
        return {
            "objective": "multi:softprob",
            "eval_metric": "mlogloss",
            "num_class": len(definition.classes),
        }
    return {"objective": "binary:logistic", "eval_metric": "logloss"}


def run_train_target(target_name: str, model_name: str | None = None) -> Path:
    """Train one registry-backed forecast target model."""
    from src.models import ModelFactory
    definition = get_target_definition(target_name)
    selected_model = (model_name or _default_model_for_target(definition.name)).strip().lower()
    if selected_model not in MODEL_REGISTRY:
        valid_models = ", ".join(sorted(MODEL_REGISTRY.keys()))
        raise ValueError(f"Unsupported model '{selected_model}'. Available options: {valid_models}")
    LOGGER.info(
        "Training forecast target | target=%s | task_type=%s | model=%s",
        definition.name,
        definition.task_type,
        selected_model,
    )
    model_cls = MODEL_REGISTRY.get(selected_model)
    if model_cls is None:
        # Use ModelFactory for models not directly in registry (e.g. goal_stacker)
        model = ModelFactory.get_model(selected_model)
    else:
        xgb_params = _xgb_params_for_target(target_name, selected_model)
        model = model_cls(**xgb_params)
    model_manager = ModelManager(model=model, target_config={"target": definition.name})
    model_path = model_manager.run_pipeline()
    LOGGER.info("Target model saved to %s", model_path)
    return model_path


def run_train_forecast_suite(targets: list[str] | None = None) -> None:
    """Train the full forecast model suite or a selected target subset."""
    selected_targets = targets or [
        definition.name for definition in list_target_definitions() if definition.name != "home_win"
    ]
    LOGGER.info("Training forecast suite targets: %s", ", ".join(selected_targets))
    for target_name in selected_targets:
        run_train_target(target_name)


def run_forecast(
    league: str | None = None,
    match_ids: list[str] | None = None,
    targets: list[str] | None = None,
    limit: int | None = None,
) -> None:
    """Emit forecast JSON for requested matches."""
    service = ForecastService(targets=targets)
    payloads = service.forecast(match_ids=match_ids, league=league, limit=limit)
    print(json.dumps(payloads, indent=2, sort_keys=True))


def run_predict(
    app_settings: AppSettings, db_manager: DuckDBManager, league: str, run_id: str | None
) -> None:
    """Load latest model and output value-bet recommendations for recent matches."""
    LOGGER.info("Executing command: predict")
    league_code = league.upper()
    LOGGER.info(
        "Running prediction for %s (%s) only.",
        LEAGUE_LABELS.get(league_code, "Selected League"),
        league_code,
    )

    model_uri, run = _get_model_uri(league=league_code, run_id=run_id)
    _check_feature_consistency(run)
    model = mlflow.sklearn.load_model(model_uri)
    LOGGER.info(
        "Loaded model from MLflow | run_id=%s | roi=%.4f",
        run.info.run_id,
        run.data.metrics.get("roi", 0.0),
    )
    if run_id:
        LOGGER.info("Run parameters: %s", run.data.params)

    recent_df = _fetch_feature_joined_matches(db_manager, days=7)
    if recent_df.empty:
        LOGGER.warning("No recent matches available for prediction.")
        return

    predictions_df = _build_prediction_frame(model, recent_df)
    if predictions_df.empty:
        LOGGER.warning("No recent matches with complete features for prediction.")
        return
    predictions_df = predictions_df[predictions_df["league"].str.upper() == league_code].copy()
    if predictions_df.empty:
        LOGGER.warning("No recent matches found for league %s.", league_code)
        return

    strategy = StrategyEngine()
    recommendations = strategy.get_recommendations(predictions_df, ev_threshold=0.05)
    LOGGER.info("Value Bets (Current Week):")
    strategy.report_recommendations(recommendations)


def run_backtest(
    app_settings: AppSettings,
    db_manager: DuckDBManager,
    ev_threshold: float,
    league: str,
    test_season: str,
    rolling_retrain: bool,
    run_id: str | None,
) -> None:
    """Run a strict season-isolated backtest with optional rolling retraining."""
    LOGGER.info("Executing command: backtest")
    league_code = league.upper()
    LOGGER.info(
        "Running backtest for %s (%s) only.",
        LEAGUE_LABELS.get(league_code, "Selected League"),
        league_code,
    )

    try:
        start_date, end_date = _parse_season_bounds(test_season)
    except ValueError as exc:
        LOGGER.error("Invalid --test_season value: %s", exc)
        return
    LOGGER.info(
        "Backtest period for season %s | Start Date=%s | End Date=%s",
        test_season,
        start_date.date().isoformat(),
        end_date.date().isoformat(),
    )

    historical_df = _fetch_feature_joined_matches(db_manager)
    if historical_df.empty:
        LOGGER.warning("No historical matches available for backtesting.")
        return
    prepared_df = _prepare_backtest_frame(historical_df)
    if prepared_df.empty:
        LOGGER.warning("No historical matches with complete features for backtesting.")
        return
    prepared_df = prepared_df[prepared_df["league"].str.upper() == league_code].copy()
    if prepared_df.empty:
        LOGGER.warning("No historical matches found for league %s.", league_code)
        return

    training_pool_df = prepared_df[prepared_df["date"] < start_date].copy()
    test_df = prepared_df[(prepared_df["date"] >= start_date) & (prepared_df["date"] <= end_date)].copy()
    if test_df.empty:
        LOGGER.warning("No matches found in league %s for season %s.", league_code, test_season)
        return
    LOGGER.info(
        "Strict split counts | training_matches=%s | backtest_matches=%s",
        len(training_pool_df),
        len(test_df),
    )

    prediction_rows: list[dict[str, object]] = []
    if rolling_retrain:
        LOGGER.info("Using rolling retrain mode for backtest.")
        if run_id:
            raise ValueError("--run_id is not supported with --rolling_retrain.")
        train_sizes: list[int] = []
        for row in test_df.itertuples(index=False):
            train_slice = prepared_df[prepared_df["date"] < row.date]
            if train_slice.empty or train_slice["target"].nunique() < 2:
                continue

            train_sizes.append(int(len(train_slice)))
            model = LRModel()
            model.train(train_slice[FEATURE_COLUMNS], train_slice["target"])
            probability = float(model.predict_proba(pd.DataFrame([row._asdict()])[FEATURE_COLUMNS])[0][1])

            prediction_rows.append(
                {
                    "match_id": row.match_id,
                    "league": row.league,
                    "home_team": row.home_team,
                    "away_team": row.away_team,
                    "predicted_home_win_prob": probability,
                    "odds_h": float(row.odds_h),
                }
            )

        if not prediction_rows:
            LOGGER.warning("No backtest predictions were generated in rolling mode.")
            return
        LOGGER.info(
            "Rolling retrain sample sizes | min=%s | max=%s",
            min(train_sizes),
            max(train_sizes),
        )
        predictions_df = pd.DataFrame(prediction_rows)
    else:
        if training_pool_df.empty:
            LOGGER.warning("No training matches exist before start of season %s.", test_season)
            return
        if training_pool_df["target"].nunique() < 2:
            LOGGER.warning("Training data before %s has a single target class; backtest cannot run.", start_date.date())
            return

        model_uri, run = _get_model_uri(league=league_code, run_id=run_id)
        _check_feature_consistency(run)
        model = mlflow.sklearn.load_model(model_uri)
        LOGGER.info(
            "Loaded model from MLflow | run_id=%s | roi=%.4f",
            run.info.run_id,
            run.data.metrics.get("roi", 0.0),
        )
        if run_id:
            LOGGER.info("Run parameters: %s", run.data.params)

        probabilities = model.predict_proba(test_df[FEATURE_COLUMNS])
        predicted_home_win_prob = probabilities[:, 1] if probabilities.ndim == 2 else probabilities.ravel()

        predictions_df = pd.DataFrame(
            {
                "match_id": test_df["match_id"].values,
                "league": test_df["league"].values,
                "home_team": test_df["home_team"].values,
                "away_team": test_df["away_team"].values,
                "predicted_home_win_prob": predicted_home_win_prob,
                "odds_h": test_df["odds_h"].astype(float).values,
            }
        )

    backtester = Backtester(initial_bankroll=app_settings.settings.initial_bankroll, bet_size=10.0)
    backtester.run_simulation(predictions_df, ev_threshold=ev_threshold)
    metrics = backtester.get_metrics()
    LOGGER.info(
        "Backtest Report | EV Threshold=%.4f | Total ROI=%.4f | Win Rate=%.4f | Max Drawdown=%.4f | Final Bankroll=%.2f",
        ev_threshold,
        metrics.total_roi,
        metrics.win_rate,
        metrics.max_drawdown,
        metrics.final_bankroll,
    )


def run_experiment(
    app_settings: AppSettings,
    experiment_name: str,
    test_season: str,
    config_path: str,
    target_type: str,
) -> None:
    """Run a YAML-configured grid search with MLflow tracking and backtest metrics."""
    LOGGER.info("Executing command: experiment")
    mlflow.set_experiment(experiment_name)

    config_file = Path(config_path)
    if not config_file.exists():
        LOGGER.error("Experiment config not found: %s", config_file)
        return
    with config_file.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle) or {}

    grid = config.get("grid_search")
    if not isinstance(grid, dict) or not grid:
        LOGGER.error("Experiment config missing grid_search parameters.")
        return
    grid_params = config.get("grid_search_params", {})
    if not isinstance(grid_params, dict):
        LOGGER.warning("grid_search_params is not a dict; skipping.")
        grid_params = {}
    fixed_params = config.get("fixed_params", {})
    if not isinstance(fixed_params, dict):
        LOGGER.warning("fixed_params is not a dict; skipping.")
        fixed_params = {}
    league_tag = str(config.get("league", "E0"))
    season_tag = str(config.get("test_season", test_season))
    backtest_config = config.get("backtest_config", {})
    if not isinstance(backtest_config, dict):
        LOGGER.warning("backtest_config is not a dict; skipping.")
        backtest_config = {}

    results: list[dict[str, float | int | str]] = []
    param_keys = list(grid.keys())
    for values in itertools.product(*(grid[key] for key in param_keys)):
        params = dict(zip(param_keys, values))
        model_type = str(config.get("model_type", "xgboost")).strip().lower()
        merged_params = {**fixed_params, **params}
        run_name = f"{model_type}_" + "_".join(f"{key}{value}" for key, value in params.items())
        with mlflow.start_run(run_name=run_name, nested=True):
            mlflow.log_params(grid_params)
            mlflow.log_params(merged_params)
            mlflow.log_param("target_type", target_type)
            mlflow.log_dict(config, "experiment_config.yaml")
            mlflow.set_tag("model_type", model_type)
            mlflow.set_tag("test_season", test_season)
            mlflow.set_tags({"league": league_tag, "season": season_tag})

            model = ModelFactory.get_model(model_type, merged_params)
            manager = ModelManager(
                model=model,
                league_tier="all",
                test_season=test_season,
                feature_version="v1",
                target_config={"target_type": target_type},
            )
            _, test_meta, positive_proba = manager.train()

            predictions_df = pd.DataFrame(
                {
                    "match_id": test_meta["match_id"].values,
                    "predicted_home_win_prob": positive_proba.values,
                    "odds_h": test_meta["odds_h"].astype(float).values,
                }
            )
            initial_bankroll = float(
                backtest_config.get("initial_bankroll", app_settings.settings.initial_bankroll)
            )
            bet_size = float(backtest_config.get("bet_size", 10.0))
            ev_threshold = float(backtest_config.get("ev_threshold", 0.05))
            backtester = Backtester(
                initial_bankroll=initial_bankroll,
                bet_size=bet_size,
                config_path="config.yaml",
            )
            backtester.run_simulation(predictions_df, ev_threshold=ev_threshold)
            metrics = backtester.get_metrics()
            mlflow.log_metrics(
                {
                    "roi": float(metrics.total_roi),
                    "win_rate": float(metrics.win_rate),
                    "max_drawdown": float(metrics.max_drawdown),
                }
            )

            if model_type == "xgboost":
                _mlflow_log_model_compat(model.model, "xgboost", name="model")
            else:
                _mlflow_log_model_compat(model.model, "sklearn", name="model")

            with TemporaryDirectory() as tmpdir:
                plot_path = Path(tmpdir) / "bankroll_curve.png"
                plt.figure(figsize=(8, 4))
                if backtester.bet_history.empty:
                    plt.plot([0, 1], [backtester.initial_bankroll, backtester.initial_bankroll])
                else:
                    plt.plot(backtester.bet_history["bankroll"].values)
                plt.title("Bankroll Curve")
                plt.xlabel("Bet Index")
                plt.ylabel("Bankroll")
                plt.tight_layout()
                plt.savefig(plot_path)
                plt.close()
                mlflow.log_artifact(str(plot_path))

            plots_dir = Path("plots")
            plots_dir.mkdir(parents=True, exist_ok=True)
            stable_plot_path = plots_dir / "bankroll.png"
            plt.figure(figsize=(8, 4))
            if backtester.bet_history.empty:
                plt.plot([0, 1], [backtester.initial_bankroll, backtester.initial_bankroll])
            else:
                plt.plot(backtester.bet_history["bankroll"].values)
            plt.title("Bankroll Curve")
            plt.xlabel("Bet Index")
            plt.ylabel("Bankroll")
            plt.tight_layout()
            plt.savefig(stable_plot_path)
            plt.close()
            if stable_plot_path.exists():
                mlflow.log_artifact(str(stable_plot_path))

            results.append(
                {
                    **params,
                    "roi": float(metrics.total_roi),
                    "win_rate": float(metrics.win_rate),
                    "max_drawdown": float(metrics.max_drawdown),
                }
            )

    if results:
        summary_df = pd.DataFrame(results).sort_values("roi", ascending=False).head(3)
        LOGGER.info("Top 3 parameter sets by ROI:\n%s", summary_df.to_string(index=False))


def _iter_grid_params(grid: dict[str, list[object]]) -> list[dict[str, object]]:
    """Expand a YAML grid into a stable list of parameter dictionaries."""
    param_keys = list(grid.keys())
    return [dict(zip(param_keys, values)) for values in itertools.product(*(grid[key] for key in param_keys))]


def _forecast_experiment_name(target_name: str, model_type: str, stage: str, version: str) -> str:
    """Return the default MLflow experiment name for forecast model sweeps."""
    return f"FPAI_{target_name}_{model_type}_{stage}_{version}"


def _mlflow_flavor_for_model_type(model_type: str) -> str:
    normalized = model_type.strip().lower()
    if normalized in {"xgb", "xgboost", "xgb_regressor", "xgboost_regressor"}:
        return "xgboost"
    return "sklearn"


def run_experiment_target(
    target_name: str,
    config_path: str,
    experiment_name: str | None = None,
    max_runs: int | None = None,
    sweep_stage: str | None = None,
) -> None:
    """Run a target-aware MLflow parameter sweep with forecast metrics."""
    SweepRunner(
        target_name=target_name,
        config_path=config_path,
        experiment_name=experiment_name,
        max_runs=max_runs,
        sweep_stage=sweep_stage,
    ).run()
    return

    definition = get_target_definition(target_name)
    config_file = Path(config_path)
    if not config_file.exists():
        raise FileNotFoundError(f"Experiment config not found: {config_file}")
    with config_file.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle) or {}

    model_type = str(config.get("model_type", _default_model_for_target(definition.name))).strip().lower()
    grid = config.get("grid_search", {})
    if not isinstance(grid, dict) or not grid:
        raise ValueError("Experiment config must contain a non-empty grid_search mapping.")
    for key, values in grid.items():
        if not isinstance(values, list) or not values:
            raise ValueError(f"grid_search.{key} must be a non-empty list.")
    fixed_params = config.get("fixed_params", {})
    if not isinstance(fixed_params, dict):
        raise ValueError("fixed_params must be a mapping when provided.")

    if sweep_stage is not None:
        config["sweep_stage"] = sweep_stage
    stage = str(config.get("sweep_stage", "broad")).strip().lower()
    version = str(config.get("version", "v1")).strip().lower()
    resolved_experiment_name = experiment_name or _forecast_experiment_name(
        definition.name,
        model_type,
        stage,
        version,
    )
    mlflow.set_experiment(resolved_experiment_name)
    run_params = _iter_grid_params(grid)
    if max_runs is not None:
        run_params = run_params[: max(0, int(max_runs))]
    LOGGER.info(
        "Running target experiment | experiment=%s | target=%s | model=%s | runs=%s",
        resolved_experiment_name,
        definition.name,
        model_type,
        len(run_params),
    )

    results: list[dict[str, object]] = []
    for index, params in enumerate(run_params, start=1):
        merged_params = {**fixed_params, **params}
        if model_type in {"xgb", "xgboost"}:
            if definition.task_type == "multiclass_classification":
                merged_params.setdefault("objective", "multi:softprob")
                merged_params.setdefault("eval_metric", "mlogloss")
                merged_params.setdefault("num_class", len(definition.classes))
            elif definition.task_type == "binary_classification":
                merged_params.setdefault("objective", "binary:logistic")
                merged_params.setdefault("eval_metric", "logloss")
        elif model_type in {"xgb_regressor", "xgboost_regressor"}:
            merged_params.setdefault("objective", "reg:squarederror")
            merged_params.setdefault("eval_metric", "rmse")
        run_name = f"{definition.name}_{model_type}_{stage}_{index:04d}"
        with mlflow.start_run(run_name=run_name):
            mlflow.log_dict(config, "experiment_config.yaml")
            mlflow.log_params(merged_params)
            mlflow.log_param("target", definition.name)
            mlflow.log_param("model_type", model_type)
            mlflow.set_tags(
                {
                    "target": definition.name,
                    "task_type": definition.task_type,
                    "model_family": model_type,
                    "feature_schema_version": str(config.get("feature_schema_version", "v1")),
                    "split_policy": "chronological_70_15_15",
                    "league": str(config.get("league", "all")),
                    "sweep_stage": stage,
                    "experiment_version": version,
                }
            )
            extra_tags = config.get("tags", {})
            if isinstance(extra_tags, dict):
                mlflow.set_tags({str(key): str(value) for key, value in extra_tags.items()})

            model = ModelFactory.get_model(model_type, merged_params)
            manager = ModelManager(
                model=model,
                league_tier=str(config.get("league_tier", "all")),
                test_season=str(config.get("test_season", "time_split")),
                feature_version=str(config.get("feature_schema_version", "v1")),
                target_config={"target": definition.name},
            )
            X_train, X_val, X_test, y_train, y_val, y_test, _ = manager.prepare_training_data()
            eval_set = [(X_val, y_val)] if isinstance(model, (XGBoostModel, XGBoostRegressorModel)) else None
            model.train(X_train, y_train, eval_set=eval_set)
            manager._log_feature_importance(list(X_train.columns), model)
            metrics, _ = manager._evaluate_target(X_test, y_test, X_train, y_train, X_val, y_val)
            mlflow.log_metrics({metric_name: float(value) for metric_name, value in metrics.items()})
            mlflow.set_tag("training_cutoff", getattr(manager, "training_cutoff", ""))
            _mlflow_log_model_compat(
                model.model,
                _mlflow_flavor_for_model_type(model_type),
                name="model",
            )
            results.append({**params, **metrics})

    if results:
        primary_metric = definition.primary_metric
        ascending = primary_metric in {"log_loss", "mae", "rmse"}
        summary_df = pd.DataFrame(results).sort_values(primary_metric, ascending=ascending).head(5)
        LOGGER.info("Top 5 parameter sets by %s:\n%s", primary_metric, summary_df.to_string(index=False))




def run_permutation_importance(
    target_name: str,
    model_path: str,
    n_repeats: int = 10,
    output_dir: str = "reports",
) -> Path:
    """Analyze permutation importance for a trained model.
    
    Args:
        target_name: Forecast target name
        model_path: Path to trained model artifact
        n_repeats: Number of permutation repeats
        output_dir: Directory for output reports
    
    Returns:
        Path to saved importance report
    """
    definition = get_target_definition(target_name)
    LOGGER.info(f"Analyzing permutation importance for {target_name}")
    
    # Load model
    analyzer = PermutationImportanceAnalyzer(
        model_path=model_path,
        target_name=target_name,
        output_dir=output_dir,
    )
    
    # Prepare evaluation data
    manager = ModelManager(
        model=analyzer.model,
        target_config={"target": target_name},
    )
    X_train, X_val, X_test, y_train, y_val, y_test, _ = manager.prepare_training_data()
    
    # Use validation set for importance analysis
    LOGGER.info(f"Computing importance on validation set ({len(X_val)} samples)")
    importance_df = analyzer.compute_importance(X_val, y_val, n_repeats=n_repeats)
    
    # Save report
    report_path = analyzer.save_report(importance_df, top_n=30)
    print(f"✓ Permutation importance report: {report_path}")
    
    # Print top 10
    top_10 = importance_df.head(10)
    print("\nTop 10 Important Features:")
    print(top_10[["rank", "feature", "importance_mean", "importance_pct"]].to_string(index=False))
    
    return report_path


def run_learning_curve(
    target: str | None,
    all_targets: bool,
    output_dir: str,
) -> None:
    """Run learning curve analysis for one or all forecast targets."""
    if all_targets:
        LOGGER.info("Running learning curve analysis for all targets...")
        all_results = run_all_targets(output_dir=output_dir)
        summary = summarise_findings(all_results)
        print(summary)
        print(f"\nCharts and CSVs saved to: {output_dir}/")
    else:
        analyzer = LearningCurveAnalyzer(target, output_dir=output_dir)
        result = analyzer.run()
        csv_path = analyzer.save_results(result)
        chart_path = analyzer.save_chart(result)
        metric = result["metric"]
        rows = result["results"]
        print(f"\nLearning Curve — {target} ({metric})")
        print(f"{'Fraction':>10} {'Train N':>10} {metric.upper():>12}")
        for row in rows:
            print(f"{row['fraction']:>10.0%} {row['train_n']:>10d} {row[metric]:>12.4f}")
        print(f"\nCSV: {csv_path}")
        if chart_path:
            print(f"Chart: {chart_path}")


def run_optuna_sweep(
    target_name: str,
    config_path: str,
    n_trials: int | None,
    experiment_name: str | None,
    sweep_stage: str | None,
) -> None:
    """Run a Bayesian hyperparameter sweep using Optuna TPE sampler."""
    runner = OptunaRunner(
        target_name=target_name,
        config_path=config_path,
        experiment_name=experiment_name,
        n_trials=n_trials,
        sweep_stage=sweep_stage,
    )
    results = runner.run()
    if results:
        from src.logic.target_registry import get_target_definition
        definition = get_target_definition(target_name)
        primary_metric = definition.primary_metric
        ascending = primary_metric in {"log_loss", "mae", "rmse"}
        import pandas as pd
        df = pd.DataFrame(results).sort_values(primary_metric, ascending=ascending)
        print(f"\nOptuna Sweep Results — {target_name} (best by {primary_metric}):")
        print(df.head(5).to_string(index=False))


def run_mlflow_cleanup(
    strategy: str = "recover",
    backup: bool = True,
    mlruns_dir: str = "mlruns",
    report_only: bool = False,
) -> None:
    """Clean up malformed MLflow experiments.
    
    Args:
        strategy: How to handle malformed experiments (recover, remove, backup_and_remove)
        backup: Whether to backup before destructive operations
        mlruns_dir: Path to MLflow runs directory
        report_only: If True, only report malformed experiments without fixing
    """
    LOGGER.info("MLflow store cleanup initiated")
    
    cleanup = MLflowStoreCleanup(mlruns_dir=mlruns_dir)
    summary = cleanup.get_cleanup_summary()
    
    print("\n" + "=" * 60)
    print("MLflow Store Status:")
    print("=" * 60)
    print(f"Total experiments: {summary['total_experiments']}")
    print(f"Valid experiments: {summary['valid_experiments']}")
    print(f"Malformed experiments: {summary['malformed_experiments']}")
    print(f"Total runs: {summary['total_runs']}")
    print(f"Runs in malformed experiments: {summary['runs_in_malformed']}")
    
    if summary["malformed_exp_ids"]:
        print(f"\nMalformed experiment IDs: {', '.join(summary['malformed_exp_ids'])}")
    
    if report_only:
        print("\n[Report Only] No changes made.")
        save_cleanup_report(summary)
        return
    
    print(f"\nApplying strategy: {strategy}")
    results = cleanup.cleanup_malformed(strategy=strategy, backup=backup)
    
    # Print results
    print("\nCleanup Results:")
    for exp_id, result in results.items():
        status = "✓" if result in {"recovered", "removed", "backed_up_and_removed"} else "✗"
        print(f"  {status} {exp_id}: {result}")
    
    # Save report
    save_cleanup_report(summary)
    print(f"\n✓ Cleanup report saved to documents/mlflow_cleanup_report.txt")


def main() -> None:
    """Parse CLI args and dispatch the requested command."""
    configure_logger()
    parser = _build_parser()
    args = parser.parse_args()

    app_settings = settings
    db_manager = DuckDBManager()

    if args.command == "scrape":
        run_scrape(app_settings, force=getattr(args, "force", False))
    elif args.command == "ingest":
        run_ingest(app_settings, db_manager, force=getattr(args, "force", False))
    elif args.command == "train":
        run_train(model_name=str(args.model), target_type=str(args.target_type))
    elif args.command == "train-target":
        run_train_target(target_name=str(args.target), model_name=args.model)
    elif args.command == "train-forecast-suite":
        run_train_forecast_suite(targets=args.targets)
    elif args.command == "forecast":
        run_forecast(
            league=args.league,
            match_ids=args.match_id,
            targets=args.target,
            limit=args.limit,
        )
    elif args.command == "predict":
        run_predict(app_settings, db_manager, league=str(args.league), run_id=args.run_id)
    elif args.command == "backtest":
        run_backtest(
            app_settings,
            db_manager,
            ev_threshold=float(args.ev_threshold),
            league=str(args.league),
            test_season=str(args.test_season),
            rolling_retrain=bool(args.rolling_retrain),
            run_id=args.run_id,
        )
    elif args.command == "experiment":
        run_experiment(
            app_settings,
            experiment_name=str(args.experiment_name),
            test_season=str(args.test_season),
            config_path=str(args.config_path),
            target_type=str(args.target_type),
        )
    elif args.command == "experiment-target":
        run_experiment_target(
            target_name=str(args.target),
            config_path=str(args.config_path),
            experiment_name=args.experiment_name,
            max_runs=args.max_runs,
        )
    elif args.command == "compare-models":
        comparer = ModelComparison(experiment_name=args.experiment_name)
        output_path = args.output_path or f"reports/model_comparison/{args.target}_comparison.{args.format}"
        report_path = comparer.export_comparison_report(args.target, output_path, format=args.format)
        best = comparer.identify_best_model(str(args.target))
        print(f"Comparison report written to {report_path}")
        if best:
            print(json.dumps(best, indent=2, sort_keys=True, default=str))
    elif args.command == "sweep-target":
        run_experiment_target(
            target_name=str(args.target),
            config_path=str(args.config_path),
            experiment_name=args.experiment_name,
            max_runs=args.max_runs,
            sweep_stage=args.sweep_stage,
        )
    elif args.command == "diagnose-model":
        report_path = run_diagnostics(
            target_name=str(args.target),
            model_path=str(args.model_path),
            output_path=str(args.output_path),
        )
        print(f"Diagnostics report written to {report_path}")
    elif args.command == "permutation-importance":
        report_path = run_permutation_importance(
            target_name=str(args.target),
            model_path=str(args.model_path),
            n_repeats=args.n_repeats,
            output_dir=str(args.output_dir),
        )
    elif args.command == "learning-curve":
        run_learning_curve(
            target=getattr(args, "target", None),
            all_targets=getattr(args, "all_targets", False),
            output_dir=str(args.output_dir),
        )
    elif args.command == "optuna-sweep":
        run_optuna_sweep(
            target_name=str(args.target),
            config_path=str(args.config_path),
            n_trials=args.n_trials,
            experiment_name=args.experiment_name,
            sweep_stage=args.sweep_stage,
        )
    elif args.command == "cleanup-mlflow":
        run_mlflow_cleanup(
            strategy=str(args.strategy),
            backup=args.backup,
            mlruns_dir=str(args.mlruns_dir),
            report_only=args.report_only,
        )
    elif args.command == "fetch-understat":
        run_fetch_understat(
            app_settings=app_settings,
            db_manager=db_manager,
            league=str(args.league),
            from_season=args.from_season,
            to_season=args.to_season,
            delay=float(args.delay),
            rebuild_features=args.rebuild_features,
        )



if __name__ == "__main__":
    main()
