"""CLI entry point for the FPAI Prototype 1 pipeline."""

from __future__ import annotations

import argparse
from datetime import datetime
import json
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
import mlflow
import pandas as pd
import yaml

load_dotenv()

from src.agent.snapshot_store import DEFAULT_BASE_DIR, league_base_dir
from src.features.feature_factory import FeatureFactory
from src.evaluation import run_diagnostics
from src.evaluation.mlflow_cleanup import MLflowStoreCleanup, save_cleanup_report
from src.forecast import ForecastService
from src.ingestion import CSVLoader, FootballDataScraper
from src.logic.target_registry import get_target_definition, list_target_definitions
from src.logic.competition_registry import (
    get_competition_definition,
    is_target_available,
    list_context_keys,
    resolve_feature_subset_for_tier,
)
from src.models import (
    LRModel,
    ModelFactory,
    ModelManager,
    RandomForestModel,
    RandomForestRegressorModel,
    XGBoostModel,
    XGBoostRegressorModel,
)
from src.utils import DuckDBManager, configure_logger, get_logger
from src.utils.config_loader import AppSettings, settings
from src.utils.feature_importance import PermutationImportanceAnalyzer
from src.utils.learning_curve import LearningCurveAnalyzer, run_all_targets, summarise_findings
from src.utils.model_comparison import ModelComparison
from src.utils.sweep_runner import OptunaRunner, StagedOptunaRunner, SweepRunner
from src.models.dixon_coles import DixonColesModel
from src.models.mlp_model import MLPModel, MLPRegressorModel

LOGGER = get_logger(__name__)
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
    "mlp": None,
    "mlp_regressor": None,
}


def _build_parser() -> argparse.ArgumentParser:
    """Create CLI parser."""
    parser = argparse.ArgumentParser(description="FPAI command line interface")
    subparsers = parser.add_subparsers(dest="command", required=True)

    # scrape
    scrape_parser = subparsers.add_parser("scrape", help="Download latest multi-season CSV files to raw directory")
    scrape_parser.add_argument("--force", action="store_true", help="Re-download and overwrite all selected CSV files.")
    scrape_parser.add_argument(
        "--league-page-url", default=None,
        help="Override config.yaml's scraper.league_page_url for this run (e.g. https://www.football-data.co.uk/spainm.php for La Liga).",
    )
    scrape_parser.add_argument(
        "--league", action="append", default=None,
        help="League code to scrape from --league-page-url (repeatable; overrides config.yaml's scraper.leagues, e.g. --league SP1).",
    )

    # ingest
    ingest_parser = subparsers.add_parser("ingest", help="Ingest CSV data and pre-compute features")
    ingest_parser.add_argument("--force", action="store_true", help="Re-ingest all CSV files and overwrite existing database rows.")

    # refresh-data (US#81)
    refresh_parser = subparsers.add_parser(
        "refresh-data",
        help="Run scrape → ingest → fetch-understat in sequence (standard data update journey)",
    )
    refresh_parser.add_argument("--league", type=str, default="E0", help="League code for understat fetch (default: E0).")
    refresh_parser.add_argument("--force", action="store_true", help="Force re-download and re-ingest of all files.")

    # schedule-refresh (US#109)
    schedule_refresh_parser = subparsers.add_parser(
        "schedule-refresh",
        help="Run refresh-data on a standing weekly schedule (blocks until interrupted).",
    )
    schedule_refresh_parser.add_argument("--league", type=str, default="E0", help="League code for understat fetch (default: E0).")
    schedule_refresh_parser.add_argument(
        "--day-of-week", type=str, default=None,
        help="Cron day-of-week. Defaults per league if omitted: 'sun' for E0, 'tue,fri' for SWE (US#132 -- "
        "Allsvenskan rounds can fall midweek, so a single weekly slot isn't tight enough).",
    )
    schedule_refresh_parser.add_argument("--hour", type=int, default=3, help="Cron hour, 0-23 (default: 3).")
    schedule_refresh_parser.add_argument("--minute", type=int, default=0, help="Cron minute, 0-59 (default: 0).")

    # train-target
    train_target_parser = subparsers.add_parser("train-target", help="Train one forecast target model")
    train_target_parser.add_argument(
        "--target", type=str, required=True,
        choices=sorted(definition.name for definition in list_target_definitions()),
        help="Forecast target to train.",
    )
    train_target_parser.add_argument(
        "--model", type=str, default=None,
        choices=sorted(MODEL_REGISTRY.keys()),
        help="Optional model override. Defaults to lr for classification and rf_regressor for regression.",
    )
    train_target_parser.add_argument(
        "--context", type=str, default="E0",
        help="Training context: a registered competition_id (e.g. E0, international). "
             "'league' is accepted as a deprecated alias for E0.",
    )

    # train-forecast-suite
    train_suite_parser = subparsers.add_parser("train-forecast-suite", help="Train all registered forecast target models")
    train_suite_parser.add_argument(
        "--targets", type=str, nargs="*", default=None,
        choices=sorted(definition.name for definition in list_target_definitions()),
        help="Optional subset of forecast targets to train.",
    )
    train_suite_parser.add_argument(
        "--context", type=str, default="E0",
        help="Training context: a registered competition_id (e.g. E0, international). "
             "'league' is accepted as a deprecated alias for E0.",
    )

    # forecast
    forecast_parser = subparsers.add_parser("forecast", help="Emit structured forecast JSON")
    forecast_parser.add_argument("--league", type=str, default=None, help="Optional league code filter.")
    forecast_parser.add_argument("--match_id", type=str, nargs="*", default=None, help="Optional match_id values to forecast.")
    forecast_parser.add_argument(
        "--target", type=str, nargs="*", default=None,
        choices=sorted(definition.name for definition in list_target_definitions() if definition.name != "home_win"),
        help="Optional forecast target subset.",
    )
    forecast_parser.add_argument("--limit", type=int, default=None, help="Optional maximum number of matches to return.")
    # Spot-inference flags (US#84 / US#86)
    forecast_parser.add_argument("--home", type=str, default=None, help="Home team name for spot inference.")
    forecast_parser.add_argument("--away", type=str, default=None, help="Away team name for spot inference.")
    forecast_parser.add_argument("--date", type=str, default=None, help="Match date (YYYY-MM-DD) for spot inference.")
    forecast_parser.add_argument("--odds_h", type=float, default=None, help="Home win odds.")
    forecast_parser.add_argument("--odds_d", type=float, default=None, help="Draw odds.")
    forecast_parser.add_argument("--odds_a", type=float, default=None, help="Away win odds.")
    forecast_parser.add_argument(
        "--match_type", type=str, default="league", choices=["league", "international"],
        help="Match type: league (full features) or international (market odds only).",
    )

    # compare-models
    compare_parser = subparsers.add_parser("compare-models", help="Compare MLflow forecast model runs for one target")
    compare_parser.add_argument(
        "--target", type=str, required=True,
        choices=sorted(definition.name for definition in list_target_definitions()),
        help="Forecast target to compare.",
    )
    compare_parser.add_argument("--experiment_name", type=str, default=None, help="Optional MLflow experiment name.")
    compare_parser.add_argument("--output_path", type=str, default=None, help="Optional output report path.")
    compare_parser.add_argument("--format", type=str, default="csv", choices=["csv", "json", "html"], help="Report format.")
    compare_parser.add_argument(
        "--context", type=str, default=None,
        help="Filter MLflow runs by context tag (a competition_id, e.g. E0, international).",
    )

    # sweep-target
    sweep_parser = subparsers.add_parser("sweep-target", help="Run a systematic target sweep with MLflow logging")
    sweep_parser.add_argument(
        "--target", type=str, required=True,
        choices=sorted(definition.name for definition in list_target_definitions()),
        help="Forecast target to sweep.",
    )
    sweep_parser.add_argument("--config_path", type=str, required=True, help="Path to target experiment YAML config.")
    sweep_parser.add_argument("--sweep_stage", type=str, default=None, choices=["smoke", "broad", "narrow", "final"], help="Override sweep stage tag.")
    sweep_parser.add_argument("--experiment_name", type=str, default=None, help="Optional MLflow experiment name override.")
    sweep_parser.add_argument("--max_runs", type=int, default=None, help="Optional cap for smoke-testing broad grids.")

    # diagnose-model
    diagnose_parser = subparsers.add_parser("diagnose-model", help="Generate evaluation diagnostics for one model artifact")
    diagnose_parser.add_argument(
        "--target", type=str, required=True,
        choices=sorted(definition.name for definition in list_target_definitions()),
        help="Forecast target to diagnose.",
    )
    diagnose_parser.add_argument("--model_path", type=str, required=True, help="Path to local model artifact.")
    diagnose_parser.add_argument("--output_path", type=str, default="reports/diagnostics.json", help="Path for diagnostics JSON output.")
    diagnose_parser.add_argument(
        "--context", type=str, default="E0",
        help="Competition context the artifact was trained for (e.g. E0, SWE, international). "
        "US#131: previously hardcoded to E0's feature set/training data regardless of which "
        "artifact was passed -- diagnosing a non-E0 model silently used the wrong feature list "
        "and training rows.",
    )

    # permutation-importance
    importance_parser = subparsers.add_parser("permutation-importance", help="Run permutation importance analysis on a trained model")
    importance_parser.add_argument(
        "--target", type=str, required=True,
        choices=sorted(definition.name for definition in list_target_definitions()),
        help="Forecast target to analyze.",
    )
    importance_parser.add_argument("--model_path", type=str, required=True, help="Path to trained model artifact.")
    importance_parser.add_argument("--n_repeats", type=int, default=10, help="Number of permutation repeats.")
    importance_parser.add_argument("--output_dir", type=str, default="reports", help="Directory for importance reports.")

    # fetch-understat
    understat_parser = subparsers.add_parser("fetch-understat", help="Fetch xG data from understat.com and update raw_matches")
    understat_parser.add_argument("--league", type=str, default="E0", help="Football-Data league code (default: E0).")
    understat_parser.add_argument("--from_season", type=int, default=None, help="First season start year to fetch.")
    understat_parser.add_argument("--to_season", type=int, default=None, help="Last season start year to fetch.")
    understat_parser.add_argument("--delay", type=float, default=1.5, help="Polite delay in seconds between requests.")
    understat_parser.add_argument("--rebuild_features", action="store_true", default=True, help="Rebuild feature store after updating xG.")

    # fetch-fotmob (US#95)
    fotmob_parser = subparsers.add_parser(
        "fetch-fotmob", help="Fetch per-match player stats from FotMob and populate raw_player_match_stats"
    )
    fotmob_parser.add_argument("--league", type=str, default="E0", help="Football-Data league code (default: E0).")
    fotmob_parser.add_argument("--from_season", type=int, default=None, help="First season start year to fetch.")
    fotmob_parser.add_argument("--to_season", type=int, default=None, help="Last season start year to fetch.")
    fotmob_parser.add_argument("--delay", type=float, default=1.0, help="Polite delay in seconds between requests.")

    # fetch-lineups (US#101)
    lineup_p = subparsers.add_parser("fetch-lineups", help="Fetch FotMob pre-match lineups into match_lineups table")
    lineup_p.add_argument("--date-from", required=True, help="Start date YYYY-MM-DD (inclusive)")
    lineup_p.add_argument("--date-to", required=True, help="End date YYYY-MM-DD (inclusive)")
    lineup_p.add_argument("--league", default="E0", help="Football-Data league code (default: E0)")
    lineup_p.add_argument("--delay", type=float, default=1.0, help="Polite delay in seconds between requests.")

    # learning-curve
    lc_parser = subparsers.add_parser("learning-curve", help="Train on growing data subsets to diagnose feature ceiling vs. data ceiling")
    lc_target_group = lc_parser.add_mutually_exclusive_group(required=True)
    lc_target_group.add_argument(
        "--target", type=str,
        choices=sorted(definition.name for definition in list_target_definitions() if definition.name != "home_win"),
        help="Single forecast target to analyse.",
    )
    lc_target_group.add_argument("--all_targets", action="store_true", help="Run analysis for all 8 forecast targets.")
    lc_parser.add_argument("--output_dir", type=str, default="reports/learning_curves", help="Directory for output CSVs and charts.")

    # optuna-sweep
    optuna_parser = subparsers.add_parser("optuna-sweep", help="Run a Bayesian hyperparameter sweep using Optuna TPE sampler")
    optuna_parser.add_argument(
        "--target", type=str, required=True,
        choices=sorted(definition.name for definition in list_target_definitions()),
        help="Forecast target to sweep.",
    )
    optuna_parser.add_argument("--config_path", type=str, required=True, help="Path to experiment YAML config.")
    optuna_parser.add_argument("--n_trials", type=int, default=None, help="Number of Optuna trials.")
    optuna_parser.add_argument("--sweep_stage", type=str, default=None, choices=["smoke", "broad", "narrow", "final", "optuna"], help="Override sweep stage tag.")
    optuna_parser.add_argument("--experiment_name", type=str, default=None, help="Optional MLflow experiment name override.")

    # dixon-coles-baseline
    dc_parser = subparsers.add_parser("dixon-coles-baseline", help="Fit Dixon-Coles model and compare against ML results")
    dc_parser.add_argument("--config_path", type=str, default="config/schema.yaml", help="Path to schema config.")
    dc_parser.add_argument("--experiment_name", type=str, default="dixon_coles_baseline", help="MLflow experiment name.")
    dc_parser.add_argument("--output_path", type=str, default="reports/model_comparison/dixon_coles_comparison.csv", help="Where to write the comparison CSV.")

    # cleanup-mlflow
    cleanup_parser = subparsers.add_parser("cleanup-mlflow", help="Clean up malformed MLflow experiments in local file store")
    cleanup_parser.add_argument("--strategy", type=str, choices=["recover", "remove", "backup_and_remove"], default="recover", help="How to handle malformed experiments.")
    cleanup_parser.add_argument("--backup", action="store_true", default=True, help="Backup before destructive operations.")
    cleanup_parser.add_argument("--mlruns_dir", type=str, default="mlruns", help="Path to MLflow runs directory.")
    cleanup_parser.add_argument("--report_only", action="store_true", help="Only report malformed experiments without fixing them.")

    # select-best-models (US#78)
    select_parser = subparsers.add_parser("select-best-models", help="Select best-performing model per target from MLflow and record in model_selection.yaml")
    select_parser.add_argument(
        "--target", type=str, default=None,
        choices=sorted(definition.name for definition in list_target_definitions() if definition.name != "home_win"),
        help="Restrict selection to one target.",
    )
    select_parser.add_argument(
        "--context", type=str, default=None,
        help="Restrict selection to one context (a competition_id, e.g. E0, international, "
             "or the deprecated alias 'league' for E0). Omit to process every registered context.",
    )
    select_parser.add_argument("--dry-run", action="store_true", help="Print proposed changes without writing.")
    select_parser.add_argument("--min_improvement", type=float, default=0.005, help="Minimum metric improvement required to replace current selection.")

    # status (US#80)
    subparsers.add_parser("status", help="Show data freshness, feature store stats, and selected models per context")

    # agent-recommend
    agent_recommend_parser = subparsers.add_parser(
        "agent-recommend",
        help="Run the betting agent to produce a recommendation for an upcoming match",
    )
    agent_recommend_parser.add_argument("--home", required=True, help="Home team name")
    agent_recommend_parser.add_argument("--away", required=True, help="Away team name")
    agent_recommend_parser.add_argument("--date", required=True, help="Match date YYYY-MM-DD")
    agent_recommend_parser.add_argument("--league", default=None, help="League code (e.g. E0). Omit for international matches.")
    agent_recommend_parser.add_argument("--odds-h", type=float, default=None, help="Home win decimal odds from bookmaker")
    agent_recommend_parser.add_argument("--odds-d", type=float, default=None, help="Draw decimal odds from bookmaker")
    agent_recommend_parser.add_argument("--odds-a", type=float, default=None, help="Away win decimal odds from bookmaker")
    agent_recommend_parser.add_argument("--config", default=None, help="Path to agent_config.yaml (default: config/agent_config.yaml)")

    # agent-snapshot
    agent_snapshot_parser = subparsers.add_parser(
        "agent-snapshot",
        help="Collect tool-call snapshots for historical matches (record mode) for later backtesting",
    )
    agent_snapshot_parser.add_argument("--from-date", required=True, help="Start date YYYY-MM-DD (inclusive)")
    agent_snapshot_parser.add_argument("--to-date", required=True, help="End date YYYY-MM-DD (inclusive)")
    agent_snapshot_parser.add_argument("--league", default=None, help="League code (e.g. E0). Omit for all leagues.")
    agent_snapshot_parser.add_argument("--config", default=None, help="Path to agent_config.yaml (default: config/agent_config.yaml)")
    agent_snapshot_parser.add_argument("--dry-run", action="store_true", help="List matches that would be processed without running the agent")

    # agent-backtest
    agent_backtest_parser = subparsers.add_parser(
        "agent-backtest",
        help="Replay recorded snapshots through the agent and report bankroll performance",
    )
    agent_backtest_parser.add_argument("--from-date", required=True, help="Start date YYYY-MM-DD (inclusive)")
    agent_backtest_parser.add_argument("--to-date", required=True, help="End date YYYY-MM-DD (inclusive)")
    agent_backtest_parser.add_argument("--league", default=None, help="League code (e.g. E0). Omit for all leagues.")
    agent_backtest_parser.add_argument("--stake-mode", choices=["flat", "kelly"], default="flat")
    agent_backtest_parser.add_argument("--sample", type=int, default=None, help="Stratified sample size before running the full set")
    agent_backtest_parser.add_argument("--concurrency", type=int, default=5, help="Max concurrent agent runs")
    agent_backtest_parser.add_argument("--config", default=None, help="Path to agent_config.yaml (default: config/agent_config.yaml)")
    agent_backtest_parser.add_argument("--split", choices=["all", "train", "test"], default="all", help="Restrict to a stable train/test partition of the matched corpus (A40). 'test' is the held-out complement of agent-train's 'train' split for the same --test-fraction.")
    agent_backtest_parser.add_argument("--test-fraction", type=float, default=0.2, help="Fraction of matches (by match_id hash) assigned to the 'test' split. Only used when --split != all.")
    agent_backtest_parser.add_argument("--use-lessons", action="store_true", help="Load approved lessons during replay (A41), evaluating them against a held-out split. Requires --split test.")

    # agent-train (A33)
    agent_train_parser = subparsers.add_parser(
        "agent-train",
        help="Critic/train mode: score completed matches and record reviewed lesson candidates in DuckDB",
    )
    agent_train_parser.add_argument("--from-date", required=True, help="Start date YYYY-MM-DD (inclusive)")
    agent_train_parser.add_argument("--to-date", required=True, help="End date YYYY-MM-DD (inclusive)")
    agent_train_parser.add_argument("--league", default=None, help="League code (e.g. E0). Omit for all leagues.")
    agent_train_parser.add_argument("--stake-mode", choices=["flat", "kelly"], default="flat")
    agent_train_parser.add_argument("--sample", type=int, default=None, help="Stratified sample size before running the full set")
    agent_train_parser.add_argument("--concurrency", type=int, default=5, help="Max concurrent agent runs")
    agent_train_parser.add_argument("--config", default=None, help="Path to agent_config.yaml (default: config/agent_config.yaml)")
    agent_train_parser.add_argument("--split", choices=["all", "train", "test"], default="all", help="Restrict to a stable train/test partition of the matched corpus (A40). Use 'train' so the critic never sees the held-out 'test' matches agent-backtest will later report on.")
    agent_train_parser.add_argument("--test-fraction", type=float, default=0.2, help="Fraction of matches (by match_id hash) reserved for the 'test' split. Only used when --split != all.")
    agent_train_parser.add_argument("--batch-size", type=int, default=1, help="Aggregate up to N same-competition/tier matches into one deterministic lesson candidate (A39) instead of one per match. Default 1 preserves the original one-row-per-match behavior.")

    # agent-lessons (A33)
    agent_lessons_parser = subparsers.add_parser(
        "agent-lessons",
        help="Review pending lesson candidates written by agent-train",
    )
    agent_lessons_subparsers = agent_lessons_parser.add_subparsers(dest="lessons_action", required=True)

    agent_lessons_approve_parser = agent_lessons_subparsers.add_parser("approve", help="Approve a pending lesson")
    agent_lessons_approve_parser.add_argument("id", type=int, help="Lesson id")
    agent_lessons_approve_parser.add_argument(
        "--scope", required=True, choices=["competition", "tier"],
        help="competition: applies only to the lesson's source competition. tier: applies to every match in the lesson's tier.",
    )
    agent_lessons_approve_parser.add_argument("--reviewer", default=None, help="Reviewer name (default: current OS user)")
    agent_lessons_approve_parser.add_argument("--rule", default=None, help="Distilled, prompt-ready rule text to store as rule_text (A44). Omit to auto-distill via LLM from the lesson's full text.")
    agent_lessons_approve_parser.add_argument("--config", default=None, help="Path to agent_config.yaml for the LLM used to auto-distill (if --rule omitted) and to check for conflicts with existing approved rules (always). Default: config/agent_config.yaml.")
    agent_lessons_approve_parser.add_argument("--force", action="store_true", help="Approve even if the conflict check (A45) finds a contradiction with an existing approved rule.")

    agent_lessons_reject_parser = agent_lessons_subparsers.add_parser("reject", help="Reject a pending lesson")
    agent_lessons_reject_parser.add_argument("id", type=int, help="Lesson id")
    agent_lessons_reject_parser.add_argument("--reviewer", default=None, help="Reviewer name (default: current OS user)")

    # agent-compare
    agent_compare_parser = subparsers.add_parser(
        "agent-compare",
        help="Compare multiple agent configs over the same backtest snapshot set",
    )
    agent_compare_parser.add_argument("--configs", nargs="+", required=True, help="Paths to two or more agent_config.yaml files")
    agent_compare_parser.add_argument("--from-date", required=True)
    agent_compare_parser.add_argument("--to-date", required=True)
    agent_compare_parser.add_argument("--league", default=None)
    agent_compare_parser.add_argument("--sample", type=int, default=None)
    agent_compare_parser.add_argument("--stake-mode", choices=["flat", "kelly"], default="flat")

    return parser


def _get_latest_model_path(model_dir: Path) -> Path:
    candidates = sorted(model_dir.glob("lr_v*_*.joblib"), key=lambda path: path.stat().st_mtime)
    if not candidates:
        raise FileNotFoundError(f"No saved model found in {model_dir}")
    return candidates[-1]


def run_scrape(
    app_settings: AppSettings,
    force: bool = False,
    league_page_url: str | None = None,
    leagues: list[str] | None = None,
) -> None:
    """US#144: league_page_url/leagues optionally override config.yaml's scraper
    section for one run, so a second football-data.co.uk page (e.g. spainm.php
    for La Liga's SP1, alongside englandm.php's E0) can be scraped without
    editing config.yaml or adding a multi-page config schema -- run `scrape`
    once per page instead. Omitting both preserves the exact pre-US#144 behavior.
    """
    LOGGER.info("Executing command: scrape")
    scraper = FootballDataScraper(
        league_page_url=league_page_url or app_settings.scraper.league_page_url,
        timeout_seconds=app_settings.scraper.timeout_seconds,
    )
    downloaded = scraper.download_all(
        limit_seasons=app_settings.scraper.limit_seasons,
        leagues=leagues or app_settings.scraper.leagues,
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
    from src.ingestion.understat.merge import update_raw_matches_xg
    from src.ingestion.understat.fetcher import fetch_seasons_range

    LOGGER.info("Executing command: fetch-understat | league=%s", league)
    with db_manager.connection() as conn:
        bounds = conn.execute("SELECT YEAR(MIN(date)), YEAR(MAX(date)) FROM raw_matches").fetchone()
    if bounds is None or bounds[0] is None:
        LOGGER.error("raw_matches is empty — run ingest first.")
        return
    detected_from = (from_season or bounds[0]) - 1
    detected_to = to_season or (bounds[1] - 1)
    LOGGER.info("Fetching seasons %d – %d for league %s", detected_from, detected_to, league)
    understat_df = fetch_seasons_range(league=league, from_season=detected_from, to_season=detected_to, delay=delay)
    if understat_df.empty:
        LOGGER.error("No Understat data returned — check league/season args and network.")
        return
    LOGGER.info("Fetched %d Understat match records total.", len(understat_df))
    result = update_raw_matches_xg(understat_df, db_manager, league=league)
    LOGGER.info("xG update | matched=%d | updated=%d | unmatched=%d", result["matched"], result["updated"], result["unmatched"])
    if rebuild_features:
        LOGGER.info("Rebuilding feature store with xG data...")
        factory = FeatureFactory(config_path=db_manager.config_path)  # US#155
        features_df = factory.compute_rolling_stats(window=app_settings.settings.rolling_window)
        factory.save_features(features_df)
        LOGGER.info("Feature store rebuilt with xG features.")


def run_fetch_fotmob(
    app_settings: AppSettings,
    db_manager: DuckDBManager,
    league: str = "E0",
    from_season: int | None = None,
    to_season: int | None = None,
    delay: float = 1.0,
) -> None:
    from datetime import date as _date

    from src.ingestion.fotmob.fetcher import fetch_player_match_stats
    from src.ingestion.fotmob.merge import upsert_player_match_stats

    LOGGER.info("Executing command: fetch-fotmob | league=%s", league)
    with db_manager.connection() as conn:
        bounds = conn.execute("SELECT YEAR(MIN(date)), YEAR(MAX(date)) FROM raw_matches").fetchone()
    if bounds is None or bounds[0] is None:
        LOGGER.error("raw_matches is empty — run ingest first.")
        return
    detected_from = (from_season or bounds[0]) - 1
    detected_to = to_season or (bounds[1] - 1)
    LOGGER.info("Fetching FotMob player stats | seasons %d-%d | league=%s", detected_from, detected_to, league)

    season_frames = []
    for season_start in range(detected_from, detected_to + 1):
        season_from = _date(season_start, 8, 1)
        season_to = _date(season_start + 1, 7, 31)
        season_frames.append(fetch_player_match_stats(league, season_from, season_to, delay=delay))
    fotmob_df = pd.concat(season_frames, ignore_index=True) if season_frames else pd.DataFrame()

    if fotmob_df.empty:
        LOGGER.error("No FotMob data returned — check league/season args and network.")
        return
    LOGGER.info("Fetched %d FotMob player-match rows total.", len(fotmob_df))
    result = upsert_player_match_stats(fotmob_df, db_manager, league=league)
    LOGGER.info(
        "FotMob upsert | matched=%d | unmatched=%d | players=%d | rows=%d",
        result["matched"], result["unmatched"], result["players_upserted"], result["rows_upserted"],
    )


def run_ingest(app_settings: AppSettings, db_manager: DuckDBManager, force: bool = False) -> None:
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
    # US#155: both CSVLoader() and FeatureFactory() used to default to
    # config.yaml unconditionally, silently ignoring the db_manager/
    # app_settings this function was actually called with -- harmless in
    # every real CLI invocation (both are always built from the same
    # default config there), but a real isolation gap for anything that
    # legitimately points run_ingest at a non-default config (e.g. a test).
    # Found live while writing exactly such a test (US#155).
    loader = CSVLoader(config_path=db_manager.config_path)
    loader.process_directory(pattern="*.csv", force=force)
    factory = FeatureFactory(config_path=db_manager.config_path)
    features_df = factory.compute_rolling_stats(window=app_settings.settings.rolling_window)
    factory.save_features(features_df)
    with db_manager.connection() as conn:
        raw_count = conn.execute("SELECT COUNT(*) FROM raw_matches").fetchone()
        feature_count = conn.execute("SELECT COUNT(*) FROM feature_store").fetchone()
    total_raw = int(raw_count[0]) if raw_count is not None else 0
    total_features = int(feature_count[0]) if feature_count is not None else 0
    LOGGER.info("Ingest complete | raw_matches=%s | feature_store=%s", total_raw, total_features)


def run_fetch_lineups(
    db_manager: DuckDBManager,
    date_from: str,
    date_to: str,
    league: str = "E0",
    delay: float = 1.0,
) -> None:
    """Fetch FotMob lineups for a date range and upsert into match_lineups (US#101)."""
    from datetime import date as _date, timedelta

    from src.ingestion.fotmob.fetcher import fetch_finished_match_ids, LEAGUE_IDS
    from src.ingestion.fotmob.lineup import upsert_match_lineups

    LOGGER.info("Executing command: fetch-lineups | league=%s | %s..%s", league, date_from, date_to)
    league_id = LEAGUE_IDS.get(league)
    if league_id is None:
        LOGGER.error("Unsupported league '%s'. Supported: %s", league, sorted(LEAGUE_IDS))
        return
    try:
        from_d = _date.fromisoformat(date_from)
        to_d = _date.fromisoformat(date_to)
    except ValueError as exc:
        LOGGER.error("Invalid date format: %s", exc)
        return

    fotmob_ids: list[int] = []
    current = from_d
    while current <= to_d:
        try:
            matches = fetch_finished_match_ids(current, league_id=league_id, delay=delay)
            fotmob_ids.extend(m["fotmob_match_id"] for m in matches)
        except Exception as exc:  # noqa: BLE001
            LOGGER.warning("Failed to fetch match list for %s: %s", current, exc)
        current += timedelta(days=1)

    LOGGER.info("fetch-lineups: found %d FotMob match IDs", len(fotmob_ids))
    total = upsert_match_lineups(fotmob_ids, db_manager, delay=delay)
    print(f"fetch-lineups complete | matches_scanned={len(fotmob_ids)} | rows_upserted={total}")


def run_refresh_sweden_data(app_settings: AppSettings, db_manager: DuckDBManager) -> None:
    """Fetch + upsert Sweden's Allsvenskan CSV and rebuild the feature store (US#132).

    Sweden's source (football-data.co.uk's new/SWE.csv, US#124) is a single
    static full-history file, not per-season pages -- it doesn't go through
    run_scrape/run_ingest, which are built around EPL's season-link-discovery
    directory-scrape convention. Sweden also has no Understat/FotMob
    integration (declined scope, Phase 20c), so unlike EPL's run_refresh_data
    path this deliberately does NOT chain fetch-understat, fetch-fotmob, or
    lineup backfill -- just fetch, upsert, and a feature-store rebuild (the
    same rebuild step run_ingest performs for EPL after ingesting new rows).
    """
    from src.ingestion.football_data.sweden_fetcher import fetch_sweden_csv
    from src.ingestion.football_data.sweden_loader import upsert_sweden_matches

    LOGGER.info("Executing command: refresh-data | league=SWE")
    sweden_df = fetch_sweden_csv()
    result = upsert_sweden_matches(sweden_df, db_manager)
    LOGGER.info(
        "Sweden CSV upsert | rows_in=%d | skipped=%d | upserted=%d",
        result["rows_in"], result["skipped"], result["upserted"],
    )
    factory = FeatureFactory(config_path=db_manager.config_path)  # US#155
    features_df = factory.compute_rolling_stats(window=app_settings.settings.rolling_window)
    factory.save_features(features_df)
    LOGGER.info("refresh-data (SWE): feature store rebuilt.")
    LOGGER.info("refresh-data complete.")


# US#150: football-data.co.uk hosts each "big five" league on its own scraper
# page (US#144) -- config.yaml's scraper section only ever holds one page's
# defaults (E0/englandm.php). A per-league override table keeps refresh-data
# generic without turning config.yaml into a list-of-pages schema (same
# lightest-option reasoning US#144 already applied to the scrape CLI itself).
_SCRAPE_SOURCE_OVERRIDE_BY_LEAGUE: dict[str, dict[str, object]] = {
    "SP1": {"league_page_url": "https://www.football-data.co.uk/spainm.php", "leagues": ["SP1"]},
}


def run_refresh_data(app_settings: AppSettings, db_manager: DuckDBManager, league: str = "E0", force: bool = False) -> None:
    """Run scrape → ingest → fetch-understat → fetch-fotmob → fetch-lineups in sequence (US#81, US#95, US#101).

    US#132: league="SWE" takes a different, shorter path (run_refresh_sweden_data)
    -- Sweden's single-CSV source and lack of Understat/FotMob integration make
    the EPL scrape/ingest/understat/fotmob/lineup chain below inapplicable.
    """
    if league == "SWE":
        run_refresh_sweden_data(app_settings, db_manager)
        return
    LOGGER.info("Executing command: refresh-data | league=%s | force=%s", league, force)
    run_scrape(app_settings, force=force, **_SCRAPE_SOURCE_OVERRIDE_BY_LEAGUE.get(league, {}))
    run_ingest(app_settings, db_manager, force=force)
    run_fetch_understat(app_settings, db_manager, league=league, rebuild_features=True)
    run_fetch_fotmob(app_settings, db_manager, league=league)
    from src.ingestion.fotmob.lineup import backfill_lineups_from_player_stats
    total = backfill_lineups_from_player_stats(db_manager)
    LOGGER.info("refresh-data: lineup backfill complete | rows_upserted=%d", total)
    LOGGER.info("refresh-data complete.")


def run_schedule_refresh(league: str = "E0", day_of_week: str | None = None, hour: int = 3, minute: int = 0) -> None:
    """Run refresh-data on a standing schedule (US#109). Blocks until interrupted.

    US#132: league="SWE" builds a separate scheduler (build_sweden_refresh_scheduler)
    with its own, tighter default cadence (twice weekly) rather than EPL's weekly
    Sunday one -- see data_refresh_scheduler.py's SWEDEN_JOB_ID comment for the
    live-data evidence justifying this. day_of_week=None (the default) picks the
    right per-league default cadence; pass an explicit value to override either.
    """
    import time

    from src.scheduling.data_refresh_scheduler import (
        DEFAULT_SWEDEN_DAY_OF_WEEK,
        build_sweden_refresh_scheduler,
        build_weekly_refresh_scheduler,
    )

    if league == "SWE":
        effective_day_of_week = day_of_week or DEFAULT_SWEDEN_DAY_OF_WEEK
        scheduler = build_sweden_refresh_scheduler(day_of_week=effective_day_of_week, hour=hour, minute=minute)
    else:
        effective_day_of_week = day_of_week or "sun"
        scheduler = build_weekly_refresh_scheduler(day_of_week=effective_day_of_week, hour=hour, minute=minute, league=league)
    scheduler.start()
    LOGGER.info(
        "schedule-refresh: refresh-data scheduler started | day_of_week=%s hour=%d minute=%d league=%s",
        effective_day_of_week, hour, minute, league,
    )
    try:
        while True:
            time.sleep(3600)
    except (KeyboardInterrupt, SystemExit):
        LOGGER.info("schedule-refresh: shutting down.")
        scheduler.shutdown()


def _default_model_for_target(target_name: str) -> str:
    definition = get_target_definition(target_name)
    if definition.task_type == "regression":
        return "rf_regressor"
    return "lr"


def _xgb_params_for_target(target_name: str, model_key: str) -> dict:
    definition = get_target_definition(target_name)
    if model_key not in {"xgb", "xgboost"}:
        return {}
    if definition.task_type == "multiclass_classification":
        return {"objective": "multi:softprob", "eval_metric": "mlogloss", "num_class": len(definition.classes)}
    return {"objective": "binary:logistic", "eval_metric": "logloss"}


def run_train_target(target_name: str, model_name: str | None = None, context: str = "E0") -> Path:
    """Train one registry-backed forecast target model."""
    definition = get_target_definition(target_name)
    selected_model = (model_name or _default_model_for_target(definition.name)).strip().lower()
    if selected_model not in MODEL_REGISTRY:
        valid_models = ", ".join(sorted(MODEL_REGISTRY.keys()))
        raise ValueError(f"Unsupported model '{selected_model}'. Available options: {valid_models}")
    LOGGER.info("Training forecast target | target=%s | task_type=%s | model=%s | context=%s", definition.name, definition.task_type, selected_model, context)
    model_cls = MODEL_REGISTRY.get(selected_model)
    if model_cls is None:
        model = ModelFactory.get_model(selected_model)
    else:
        xgb_params = _xgb_params_for_target(target_name, selected_model)
        model = model_cls(**xgb_params)

    # US#110: --context IS the competition_id to train for (e.g. "E0", "SWE",
    # "international"), resolved through the registry rather than a hardcoded
    # binary. "league" is kept as a deprecated alias for "E0" -- the one
    # competition_specific competition it used to unambiguously mean -- so it
    # keeps working rather than silently doing the wrong thing.
    competition_id = "E0" if context == "league" else context
    competition_def = get_competition_definition(competition_id)
    if not is_target_available(competition_def, definition.name):
        # US#129: fail fast and explicitly rather than let prepare_training_data's
        # dropna(subset=["target"]) silently drop every row (BUG-001's failure
        # shape, on labels instead of features) when this competition's data
        # source doesn't populate the raw column(s) this target's labels need.
        raise ValueError(
            f"Target '{definition.name}' is not available for competition '{competition_id}': "
            f"its data source does not populate the raw column(s) label_columns={definition.label_columns} "
            "require. See config/competitions.yaml's available_targets for this competition, or use "
            "train-forecast-suite, which skips unavailable targets automatically."
        )
    feature_subset = resolve_feature_subset_for_tier(competition_def.tier)
    model_manager = ModelManager(
        model=model,
        target_config={"target": definition.name},
        feature_subset=feature_subset,
        context=competition_id,
        competition_id=competition_id,
    )
    model_path = model_manager.run_pipeline()
    LOGGER.info("Target model saved to %s", model_path)
    return model_path


def run_train_forecast_suite(targets: list[str] | None = None, context: str = "E0") -> None:
    """Train the full forecast model suite or a selected target subset.

    US#129: some competitions' data sources don't populate every raw column
    a target's labels depend on (e.g. Sweden's football-data.co.uk "New
    Leagues" CSV has no hc/ac corners columns at all -- see US#125's
    _NULL_COLUMNS). Rather than let those targets reach
    ModelManager.prepare_training_data(), where dropna(subset=["target"])
    would silently drop every training row (BUG-001's failure shape, on
    labels instead of features) and raise a confusing "No rows left" error,
    resolve this competition's `available_targets` (config/competitions.yaml,
    via src/logic/competition_registry.py) up front and skip anything not on
    it, with an explicit, readable reason logged per skipped target.
    """
    requested_targets = targets or [
        definition.name for definition in list_target_definitions() if definition.name != "home_win"
    ]

    # --context "league" is a deprecated alias for "E0" -- resolve the same
    # way run_train_target does, so the availability check looks at the
    # right competition's registry entry.
    competition_id = "E0" if context == "league" else context
    competition_def = get_competition_definition(competition_id)

    selected_targets: list[str] = []
    for target_name in requested_targets:
        if is_target_available(competition_def, target_name):
            selected_targets.append(target_name)
            continue
        label_columns = get_target_definition(target_name).label_columns
        LOGGER.info(
            "train-forecast-suite: SKIPPING target=%s for context=%s | reason: this target's labels "
            "require raw_matches column(s) %s, which are not in competition '%s's available_targets "
            "(config/competitions.yaml) -- its data source does not populate them for every row. "
            "Add '%s' to that competition's available_targets once its source provides this data.",
            target_name, context, label_columns, competition_id, target_name,
        )

    if not selected_targets:
        LOGGER.warning(
            "train-forecast-suite: no available targets to train for context=%s -- every requested "
            "target was skipped (see SKIPPING log lines above for reasons).",
            context,
        )
        return

    LOGGER.info("Training forecast suite targets: %s | context=%s", ", ".join(selected_targets), context)
    for target_name in selected_targets:
        run_train_target(target_name, context=context)


def run_forecast(
    league: str | None = None,
    match_ids: list[str] | None = None,
    targets: list[str] | None = None,
    limit: int | None = None,
    home: str | None = None,
    away: str | None = None,
    date: str | None = None,
    odds_h: float | None = None,
    odds_d: float | None = None,
    odds_a: float | None = None,
    match_type: str = "league",
) -> None:
    """Emit forecast JSON for requested matches or a spot-inference match."""
    if home is not None:
        # Spot inference path (US#84 / US#86)
        if odds_h is None or odds_d is None or odds_a is None:
            raise ValueError("--odds_h, --odds_d, and --odds_a are required for spot inference.")
        if match_type == "league" and league is None:
            raise ValueError("--league is required when --match_type league.")
        service = ForecastService(targets=targets)
        payload = service.forecast_upcoming(
            home_team=home,
            away_team=away,
            date=date or datetime.utcnow().date().isoformat(),
            league=league or "",
            odds_h=odds_h,
            odds_d=odds_d,
            odds_a=odds_a,
            match_type=match_type,
        )
        print(json.dumps(payload, indent=2, sort_keys=True))
        return
    service = ForecastService(targets=targets)
    payloads = service.forecast(match_ids=match_ids, league=league, limit=limit)
    print(json.dumps(payloads, indent=2, sort_keys=True))


def run_experiment_target(
    target_name: str,
    config_path: str,
    experiment_name: str | None = None,
    max_runs: int | None = None,
    sweep_stage: str | None = None,
) -> None:
    SweepRunner(
        target_name=target_name,
        config_path=config_path,
        experiment_name=experiment_name,
        max_runs=max_runs,
        sweep_stage=sweep_stage,
    ).run()


def run_permutation_importance(
    target_name: str,
    model_path: str,
    n_repeats: int = 10,
    output_dir: str = "reports",
) -> Path:
    LOGGER.info("Analyzing permutation importance for %s", target_name)
    analyzer = PermutationImportanceAnalyzer(model_path=model_path, target_name=target_name, output_dir=output_dir)
    manager = ModelManager(model=analyzer.model, target_config={"target": target_name})
    X_train, X_val, X_test, y_train, y_val, y_test, _ = manager.prepare_training_data()
    LOGGER.info("Computing importance on validation set (%d samples)", len(X_val))
    importance_df = analyzer.compute_importance(X_val, y_val, n_repeats=n_repeats)
    report_path = analyzer.save_report(importance_df, top_n=30)
    print(f"Permutation importance report: {report_path}")
    top_10 = importance_df.head(10)
    print("\nTop 10 Important Features:")
    print(top_10[["rank", "feature", "importance_mean", "importance_pct"]].to_string(index=False))
    return report_path


def run_learning_curve(target: str | None, all_targets: bool, output_dir: str) -> None:
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
    with open(config_path, "r") as fh:
        probe = yaml.safe_load(fh) or {}
    if "stages" in probe:
        runner: OptunaRunner | StagedOptunaRunner = StagedOptunaRunner(
            target_name=target_name, config_path=config_path, experiment_name=experiment_name,
        )
    else:
        runner = OptunaRunner(
            target_name=target_name, config_path=config_path, experiment_name=experiment_name,
            n_trials=n_trials, sweep_stage=sweep_stage,
        )
    results = runner.run()
    if results:
        from src.logic.target_registry import get_target_definition as _gtd
        definition = _gtd(target_name)
        primary_metric = definition.primary_metric
        ascending = primary_metric in {"log_loss", "mae", "rmse"}
        df = pd.DataFrame(results).sort_values(primary_metric, ascending=ascending)
        print(f"\nOptuna Sweep Results — {target_name} (best by {primary_metric}):")
        print(df.head(5).to_string(index=False))


def run_dixon_coles_baseline(
    config_path: str = "config/schema.yaml",
    experiment_name: str = "dixon_coles_baseline",
    output_path: str = "reports/model_comparison/dixon_coles_comparison.csv",
) -> None:
    from sklearn.metrics import accuracy_score, log_loss, mean_absolute_error
    import numpy as np

    db_manager_local = DuckDBManager(config_path=config_path)
    LOGGER.info("Loading match data for Dixon-Coles baseline...")
    with db_manager_local.connection(read_only=True) as conn:
        df = conn.execute(
            """
            SELECT r.match_id, r.date, r.home_team, r.away_team,
                   r.fthg, r.ftag, r.hc, r.ac
            FROM raw_matches r
            INNER JOIN feature_store f ON r.match_id = f.match_id
            ORDER BY r.date, r.match_id
            """
        ).fetchdf()
    if df.empty:
        raise ValueError("No match data found in DB.")
    df = df.dropna(subset=["fthg", "ftag"]).reset_index(drop=True)
    total = len(df)
    train_end = max(1, int(total * 0.70))
    val_end = max(train_end + 1, int(total * 0.85))
    val_end = min(val_end, total - 1)
    train_df = df.iloc[:train_end].copy()
    test_df = df.iloc[val_end:].copy().reset_index(drop=True)
    LOGGER.info("Split: train=%d, val=%d, test=%d", train_end, val_end - train_end, total - val_end)
    model = DixonColesModel()
    model.fit(train_df)
    preds = model.predict_batch(test_df)
    test_df["y_result_3way"] = test_df.apply(
        lambda r: "home" if r["fthg"] > r["ftag"] else ("draw" if r["fthg"] == r["ftag"] else "away"), axis=1,
    )
    test_df["y_btts"] = ((test_df["fthg"] > 0) & (test_df["ftag"] > 0)).astype(int)
    test_df["y_home_goals"] = test_df["fthg"].astype(float)
    test_df["y_away_goals"] = test_df["ftag"].astype(float)
    test_df["y_total_goals"] = test_df["fthg"].astype(float) + test_df["ftag"].astype(float)
    test_df["y_home_corners"] = test_df["hc"]
    test_df["y_away_corners"] = test_df["ac"]
    test_df["y_total_corners"] = test_df["hc"] + test_df["ac"]
    results: dict[str, dict[str, float]] = {}
    r3_proba = preds[["result_3way_p_home", "result_3way_p_draw", "result_3way_p_away"]].to_numpy()
    r3_true = test_df["y_result_3way"].to_numpy()
    results["result_3way"] = {
        "accuracy": float(accuracy_score(r3_true, preds["result_3way_pred"])),
        "log_loss": float(log_loss(r3_true, r3_proba, labels=["home", "draw", "away"])),
    }
    btts_proba = np.column_stack([1 - preds["btts_prob"].to_numpy(), preds["btts_prob"].to_numpy()])
    btts_true = test_df["y_btts"].to_numpy()
    results["btts"] = {
        "accuracy": float(accuracy_score(btts_true, preds["btts_pred"])),
        "log_loss": float(log_loss(btts_true, btts_proba)),
    }
    for target_col, pred_col in [
        ("y_home_goals", "home_goals"), ("y_away_goals", "away_goals"), ("y_total_goals", "total_goals"),
        ("y_home_corners", "home_corners"), ("y_away_corners", "away_corners"), ("y_total_corners", "total_corners"),
    ]:
        valid = test_df[[target_col]].join(preds[[pred_col]]).dropna()
        if valid.empty:
            results[pred_col] = {"mae": float("nan"), "rmse": float("nan")}
            continue
        y_true = valid[target_col].to_numpy()
        y_pred = valid[pred_col].to_numpy()
        results[pred_col] = {
            "mae": float(mean_absolute_error(y_true, y_pred)),
            "rmse": float(np.sqrt(np.mean((y_true - y_pred) ** 2))),
        }
    mlflow.set_experiment(experiment_name)
    with mlflow.start_run(run_name="dixon_coles_v1"):
        mlflow.set_tag("model_type", "dixon_coles")
        mlflow.log_param("train_matches", train_end)
        mlflow.log_param("test_matches", total - val_end)
        mlflow.log_param("rho", round(model._rho, 4))
        mlflow.log_param("home_adv", round(model._home_adv, 4))
        for target, metrics in results.items():
            for metric, value in metrics.items():
                mlflow.log_metric(f"{target}_{metric}", value)
    model_path = Path("models/dixon_coles_v1.joblib")
    model.save(str(model_path))
    print("\n" + "=" * 70)
    print("Dixon-Coles Baseline — Test Set Results")
    print("=" * 70)
    print(f"{'Target':<22} {'Metric':<12} {'Value':>10}")
    print("-" * 50)
    for target, metrics in results.items():
        for metric, value in metrics.items():
            print(f"{target:<22} {metric:<12} {value:>10.4f}")
    rows = []
    for target, metrics in results.items():
        for metric, value in metrics.items():
            rows.append({"target": target, "metric": metric, "dixon_coles": value})
    out_df = pd.DataFrame(rows)
    out_path = Path(output_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(str(out_path), index=False)
    LOGGER.info("Comparison CSV saved to %s", out_path)
    print(f"\nResults saved to {out_path}")
    print(f"Model saved to {model_path}")


def run_mlflow_cleanup(
    strategy: str = "recover",
    backup: bool = True,
    mlruns_dir: str = "mlruns",
    report_only: bool = False,
) -> None:
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
    print("\nCleanup Results:")
    for exp_id, result in results.items():
        status = "ok" if result in {"recovered", "removed", "backed_up_and_removed"} else "err"
        print(f"  [{status}] {exp_id}: {result}")
    save_cleanup_report(summary)
    print("\nCleanup report saved to documents/mlflow_cleanup_report.txt")


def _current_computable_features() -> set[str] | None:
    """Sample FeatureFactory.build_for_match() on one real fixture to get the
    set of feature columns the live inference path can currently produce
    (BUG-012 layer 3c). Returns None (disabling the promotion-time coverage
    guard) if this can't be determined, e.g. no raw_matches data yet — the
    guard is a safety net, not a hard requirement to run model selection."""
    try:
        from src.features.feature_factory import FeatureFactory
        from src.utils.db_manager import DuckDBManager

        db_manager = DuckDBManager()
        with db_manager.connection(read_only=True) as conn:
            row = conn.execute(
                "SELECT home_team, away_team, date, league FROM raw_matches "
                "WHERE odds_h IS NOT NULL ORDER BY date DESC LIMIT 1"
            ).fetchone()
        if row is None:
            return None
        home_team, away_team, date, league = row
        factory = FeatureFactory()
        sample = factory.build_for_match(
            home_team=home_team, away_team=away_team, match_date=str(date),
            league=league, odds_h=2.0, odds_d=3.0, odds_a=3.5,
        )
        return set(sample.columns)
    except Exception as exc:  # noqa: BLE001 — best-effort guard, never blocks selection
        LOGGER.warning("Could not determine live-computable features for promotion guard: %s", exc)
        return None


def run_select_best_models(
    target: str | None = None,
    context: str | None = None,
    dry_run: bool = False,
    min_improvement: float = 0.005,
) -> None:
    """Select best-performing model per target from MLflow (US#78)."""
    from src.utils.model_selection import ModelSelector
    selector = ModelSelector(computable_features=_current_computable_features())
    selector.run(target=target, context=context, dry_run=dry_run, min_improvement=min_improvement)


def run_agent_recommend(
    home_team: str,
    away_team: str,
    date: str,
    league: str | None,
    config_path: str | None,
    odds_h: float | None = None,
    odds_d: float | None = None,
    odds_a: float | None = None,
) -> None:
    """Run the betting agent for a single upcoming match (A08)."""
    import sys
    from src.agent.agent_config import AgentConfig
    from src.agent.graph import run_agent

    cfg = AgentConfig.from_yaml(config_path) if config_path else AgentConfig.default()
    match_info = {"home_team": home_team, "away_team": away_team, "date": date}
    if league:
        match_info["league"] = league
    if odds_h is not None and odds_d is not None and odds_a is not None:
        match_info["odds"] = {"home": odds_h, "draw": odds_d, "away": odds_a}

    print(f"\nAnalysing: {home_team} vs {away_team} on {date}" + (f" [{league}]" if league else ""))
    print(f"Model: {cfg.provider}/{cfg.model} | max_tool_calls={cfg.max_tool_calls}\n")

    try:
        recommendation = run_agent(match_info=match_info, config=cfg)
    except Exception as exc:
        from src.agent.schema import RecommendationParseError
        print(f"[ERROR] Agent failed: {exc}", file=sys.stderr)
        if isinstance(exc, RecommendationParseError):
            print("\n--- Raw agent output ---", file=sys.stderr)
            print(exc.raw_text, file=sys.stderr)
            print("--- End raw output ---\n", file=sys.stderr)
        sys.exit(1)

    if recommendation is None:
        print("[ERROR] Agent returned no recommendation.", file=sys.stderr)
        sys.exit(1)

    explanation = recommendation.pop("explanation", "")
    print("=== Explanation ===")
    print(explanation)
    print("\n=== Recommendation ===")
    print(json.dumps(recommendation, indent=2))


def run_agent_snapshot(
    from_date: str,
    to_date: str,
    league: str | None,
    config_path: str | None,
    dry_run: bool,
) -> None:
    """Drive the agent in record mode over historical matches to build a snapshot corpus (A11)."""
    import sys
    from datetime import datetime, timezone
    from pathlib import Path

    from src.agent.agent_config import AgentConfig
    from src.agent.backtest import LEAKAGE_GUARD_INSTRUCTIONS
    from src.agent.graph import run_agent
    from src.agent import tools as agent_tools
    from src.utils.db_manager import DuckDBManager

    # A46: shares its actual instruction text with process_match_row's replay
    # path via LEAKAGE_GUARD_INSTRUCTIONS so the two can never drift apart.
    snapshot_addendum = (
        "## SNAPSHOT COLLECTION MODE\n\n"
        "You are collecting training data from a historical match. " + LEAKAGE_GUARD_INSTRUCTIONS
    )

    cfg = AgentConfig.from_yaml(config_path) if config_path else AgentConfig.default()
    db = DuckDBManager()
    query = (
        "SELECT match_id, league, date, home_team, away_team, odds_h, odds_d, odds_a "
        "FROM raw_matches WHERE date >= ? AND date <= ?"
    )
    params: list = [from_date, to_date]
    if league:
        query += " AND UPPER(league) = ?"
        params.append(league.upper())
    query += " ORDER BY date"
    with db.connection() as conn:
        matches = conn.execute(query, params).fetchdf()

    base_dir = DEFAULT_BASE_DIR
    to_process = []
    skipped = 0
    for _, row in matches.iterrows():
        marker = league_base_dir(row["league"], base_dir=base_dir) / row["match_id"] / "_complete.json"
        if marker.exists():
            skipped += 1
            continue
        to_process.append(row)

    print(f"Matches in range: {len(matches)} | already complete: {skipped} | to process: {len(to_process)}")
    if dry_run:
        for row in to_process:
            date_str = str(row["date"].date()) if hasattr(row["date"], "date") else str(row["date"])
            print(f"  {date_str} {row['home_team']} vs {row['away_team']} [{row['league']}]")
        return

    errors = 0
    for i, row in enumerate(to_process, 1):
        match_id = row["match_id"]
        date_str = str(row["date"].date()) if hasattr(row["date"], "date") else str(row["date"])
        match_info = {"home_team": row["home_team"], "away_team": row["away_team"], "date": date_str, "league": row["league"]}
        if row["odds_h"] and row["odds_d"] and row["odds_a"]:
            match_info["odds"] = {"home": row["odds_h"], "draw": row["odds_d"], "away": row["odds_a"]}

        match_base_dir = league_base_dir(row["league"], base_dir=base_dir)
        agent_tools.configure_snapshot_store("record", match_id=match_id, match_date=date_str, base_dir=match_base_dir)
        try:
            run_agent(match_info=match_info, config=cfg, extra_system_instructions=snapshot_addendum)
            marker_path = match_base_dir / match_id / "_complete.json"
            marker_path.parent.mkdir(parents=True, exist_ok=True)
            marker_path.write_text(json.dumps({"completed_at": datetime.now(timezone.utc).isoformat()}))
            print(f"[{i}/{len(to_process)}] OK {match_info['home_team']} vs {match_info['away_team']}")
        except Exception as exc:
            errors += 1
            print(f"[{i}/{len(to_process)}] ERROR {match_info['home_team']} vs {match_info['away_team']}: {exc}", file=sys.stderr)
        finally:
            agent_tools.configure_snapshot_store("live")

    print(f"\nDone. Processed: {len(to_process) - errors} | Errors: {errors} | Skipped: {skipped}")


async def _run_backtest_concurrent(
    matches, config, concurrency: int, capture_state: bool = False, allow_lessons_in_replay: bool = False,
) -> list:
    """Run process_match_row for every match concurrently, bounded by a semaphore.
    Each call runs in its own thread via asyncio.to_thread since the agent graph
    and tools are synchronous; SnapshotStore's thread-local state (A09) keeps
    concurrent replay contexts from clobbering each other. Per-match failures
    (e.g. SnapshotMissingError for an unrecorded match) are caught and skipped
    so one bad match doesn't abort the whole batch — mirrors run_agent_snapshot's
    error-tolerance pattern.

    capture_state (A33): threaded through to process_match_row so agent-train
    can persist each match's raw evidence to DuckDB telemetry.

    allow_lessons_in_replay (A41): threaded through to process_match_row so
    agent-backtest --split test --use-lessons can evaluate the held-out split
    with approved lessons active."""
    import asyncio
    import sys

    from tqdm import tqdm

    from src.agent.backtest import process_match_row

    semaphore = asyncio.Semaphore(concurrency)
    progress = tqdm(total=len(matches), desc="Backtesting")
    rows = [row for _, row in matches.iterrows()]

    async def _run_one(row):
        async with semaphore:
            try:
                record = await asyncio.to_thread(
                    process_match_row, row, config,
                    capture_state=capture_state, allow_lessons_in_replay=allow_lessons_in_replay,
                )
            except Exception as exc:
                match_id = row.get("match_id", "?") if hasattr(row, "get") else "?"
                print(f"  SKIP {match_id}: {exc}", file=sys.stderr)
                record = None
            finally:
                progress.update(1)
            return record

    try:
        results = await asyncio.gather(*[_run_one(row) for row in rows])
    finally:
        progress.close()
    records = [r for r in results if r is not None]
    skipped = len(results) - len(records)
    if skipped:
        print(f"Skipped {skipped}/{len(results)} matches (see stderr for details)")
    return records


def run_agent_backtest(
    from_date: str,
    to_date: str,
    league: str | None,
    stake_mode: str,
    sample: int | None,
    concurrency: int,
    config_path: str | None,
    split: str = "all",
    test_fraction: float = 0.2,
    use_lessons: bool = False,
) -> None:
    """Replay agent recommendations over historical snapshots and report bankroll performance (A14).

    use_lessons (A41): loads approved lessons during this replay, evaluating
    them against a held-out split instead of the live-only default (A33).
    Requires --split test -- applying lessons to --split train or --split all
    would evaluate them against (some of) the very matches that shaped them,
    the exact leakage A40's train/test split exists to prevent."""
    import asyncio

    from src.agent.agent_config import AgentConfig
    from src.agent.backtest import BacktestHarness
    from src.agent.evaluation import build_evaluation_report, print_report, save_report
    from src.agent.staking import simulate_flat_stake, simulate_kelly_stake

    if concurrency < 1:
        raise ValueError(f"--concurrency must be >= 1, got {concurrency}")
    if use_lessons and split != "test":
        raise ValueError("--use-lessons requires --split test (evaluating lessons against their own train/all data leaks)")

    cfg = AgentConfig.from_yaml(config_path) if config_path else AgentConfig.default()
    harness = BacktestHarness(config=cfg)
    matches = harness.load_matches(from_date, to_date, league=league, sample=sample, split=split, test_fraction=test_fraction)
    print(f"Running backtest over {len(matches)} matches (concurrency={concurrency}, split={split}, use_lessons={use_lessons})...")

    records = asyncio.run(_run_backtest_concurrent(matches, cfg, concurrency, allow_lessons_in_replay=use_lessons))

    stake_fn = simulate_kelly_stake if stake_mode == "kelly" else simulate_flat_stake
    bankroll_result = stake_fn(records)
    report = build_evaluation_report(records, bankroll_result)
    print_report(report)
    path = save_report(report, cfg)
    print(f"\nReport saved to {path}")


def _build_llm_invoke(config) -> Any:
    """Wrap this run's configured LLM into a plain str->str callable, so
    src/agent/lessons.py's generate_batch_reflection() stays decoupled from
    langchain (see that function's docstring). Reuses _build_llm/_extract_text
    from src.agent.graph -- the same provider the agent-train run itself used
    (DeepSeek, Anthropic, whichever --config specified) writes the batch's
    reflection too, not a hardcoded second provider."""
    from src.agent.graph import _build_llm, _extract_text

    llm = _build_llm(config)

    def _invoke(prompt: str) -> str:
        response = llm.invoke(prompt)
        return _extract_text(response.content)

    return _invoke


def _write_train_artifacts(
    conn, records: list, run_id: str, batch_size: int = 1, config: Any = None,
) -> tuple[int, int]:
    """Write one telemetry row per scored record that captured full graph
    state, and one pending lesson candidate per record (batch_size <= 1,
    A33's original behavior, left as a fully separate code path so it stays
    byte-identical) or per batch of up to batch_size same-(competition_id,
    tier) records (batch_size > 1, A39). Records without full_state (e.g.
    a per-match failure that skipped capture) are silently skipped -- there's
    nothing to persist. Returns (lessons_written, telemetry_written).

    config (A42-follow-up): when given (and batch_size > 1), each batch's
    lesson also gets an LLM-synthesized reflective narrative appended
    (generate_batch_reflection) on top of the deterministic stats -- the
    stats alone were reviewed and judged "not very sensible" (2026-07-28).
    None (the batch_size <= 1 code path never receives it, and it's optional
    here) skips the reflection entirely, keeping every existing caller that
    doesn't pass config unaffected."""
    from src.agent.lessons import (
        create_lessons_tables,
        extract_competition_scope,
        generate_batch_lesson_text,
        generate_batch_reflection,
        generate_lesson_text,
        insert_lesson_candidate,
        insert_telemetry,
    )

    create_lessons_tables(conn)
    scoped = []  # (record, competition_id, tier), only records with full_state
    telemetry_written = 0
    for record in records:
        if not record.full_state:
            continue
        competition_id, tier = extract_competition_scope(record.full_state)
        insert_telemetry(
            conn,
            match_id=record.match_id,
            run_id=run_id,
            competition_resolution=record.full_state.get("competition_resolution"),
            research_evidence=record.full_state.get("research_evidence"),
            forecast_payload=record.full_state.get("forecast_payload"),
            recommendation=record.recommendation,
        )
        telemetry_written += 1
        scoped.append((record, competition_id, tier))

    if batch_size <= 1:
        lessons_written = 0
        for record, competition_id, tier in scoped:
            lesson_text = generate_lesson_text(record)
            insert_lesson_candidate(conn, lesson_text, competition_id, tier, record.match_id)
            lessons_written += 1
        return lessons_written, telemetry_written

    # A39: chunk consecutive same-(competition_id, tier) records into groups
    # of up to batch_size, never spanning a scope boundary (insert_lesson_candidate
    # takes one competition_id/tier per row) -- relies on `records` already
    # being date-ordered (BacktestHarness.load_matches' ORDER BY date,
    # preserved through asyncio.gather) so "consecutive" is meaningful, not
    # an arbitrary grouping.
    llm_invoke = _build_llm_invoke(config) if config is not None else None
    lessons_written = 0
    current_scope = None
    current_batch: list = []

    def _flush() -> None:
        nonlocal lessons_written
        if not current_batch:
            return
        competition_id, tier = current_scope
        stats_text = generate_batch_lesson_text(current_batch)
        lesson_text = stats_text
        if llm_invoke is not None:
            reflection = generate_batch_reflection(current_batch, stats_text, llm_invoke)
            if reflection:
                lesson_text = f"{stats_text}\n\nReflection: {reflection}"
            else:
                print(f"  note: LLM reflection unavailable for batch of {len(current_batch)} ({competition_id}/{tier})")
        match_ids = ",".join(r.match_id for r in current_batch)
        insert_lesson_candidate(conn, lesson_text, competition_id, tier, match_ids)
        lessons_written += 1

    for record, competition_id, tier in scoped:
        scope = (competition_id, tier)
        if scope != current_scope or len(current_batch) >= batch_size:
            _flush()
            current_batch = []
            current_scope = scope
        current_batch.append(record)
    _flush()

    return lessons_written, telemetry_written


def run_agent_train(
    from_date: str,
    to_date: str,
    league: str | None,
    stake_mode: str,
    sample: int | None,
    concurrency: int,
    config_path: str | None,
    split: str = "all",
    test_fraction: float = 0.2,
    batch_size: int = 1,
) -> None:
    """Critic/train mode (A33): replay completed matches, score them the same
    way agent-backtest does, and additionally write one competition/tier-
    tagged lesson candidate (A39: per batch of up to batch_size matches,
    default 1 -- one per match, A33's original behavior) plus a raw-evidence
    telemetry row per match."""
    import asyncio
    import uuid

    from src.agent.agent_config import AgentConfig
    from src.agent.backtest import BacktestHarness
    from src.agent.evaluation import build_evaluation_report, print_report, save_report
    from src.agent.staking import simulate_flat_stake, simulate_kelly_stake

    if concurrency < 1:
        raise ValueError(f"--concurrency must be >= 1, got {concurrency}")
    if batch_size < 1:
        raise ValueError(f"--batch-size must be >= 1, got {batch_size}")

    cfg = AgentConfig.from_yaml(config_path) if config_path else AgentConfig.default()
    harness = BacktestHarness(config=cfg)
    matches = harness.load_matches(from_date, to_date, league=league, sample=sample, split=split, test_fraction=test_fraction)
    print(f"Running train mode over {len(matches)} matches (concurrency={concurrency}, split={split}, batch_size={batch_size})...")

    records = asyncio.run(_run_backtest_concurrent(matches, cfg, concurrency, capture_state=True))

    stake_fn = simulate_kelly_stake if stake_mode == "kelly" else simulate_flat_stake
    bankroll_result = stake_fn(records)
    report = build_evaluation_report(records, bankroll_result)
    print_report(report)
    path = save_report(report, cfg, base_dir="reports/agent_train")
    print(f"\nReport saved to {path}")

    run_id = uuid.uuid4().hex
    with harness.db.connection() as conn:
        lessons_written, telemetry_written = _write_train_artifacts(conn, records, run_id, batch_size=batch_size, config=cfg)
    print(f"Wrote {lessons_written} lesson candidates and {telemetry_written} telemetry rows (run_id={run_id})")


def run_agent_lessons_approve(
    lesson_id: int, scope: str, reviewer: str | None, rule: str | None = None,
    config_path: str | None = None, force: bool = False,
) -> None:
    """Approve a pending lesson candidate (A33). scope='competition' pins it
    to its source competition; scope='tier' widens it to the whole tier.

    rule (A44): the distilled, prompt-ready rule_text stored on approval --
    required for the lesson to ever reach the live agent (load_approved_lessons
    reads rule_text only). If given, used verbatim, no LLM call for
    distillation. If omitted, auto-distilled from the lesson's full
    lesson_text via generate_rule_from_lesson, using config_path's provider
    (default: AgentConfig.default()) -- printed for visibility before being
    stored, so a bad distillation is easy to notice and redo with an
    explicit --rule.

    force (A45, design 3): before storing, rule_text is checked against
    every already-approved rule that could co-occur with it live (same
    competition_id or same tier) via find_conflicting_rule -- one LLM call,
    run regardless of whether rule_text came from --rule or auto-distillation,
    since the point is protecting live-prompt coherence, not just checking
    auto-generated text. Two independent failure modes, handled differently
    on purpose: if the *check itself* raises (network/provider error), this
    fails open -- warns and still approves, since blocking a reviewer's
    workflow on a transient API error is worse than the check being skipped
    this one time (mirrors A43/A44's fallback philosophy). If the check
    *succeeds and finds a real conflict*, this fails closed -- refuses to
    approve unless force=True, since a silently-approved contradiction is
    exactly what this story exists to prevent."""
    import getpass

    from src.agent.agent_config import AgentConfig
    from src.agent.lessons import (
        approve_lesson, create_lessons_tables, find_conflicting_rule, generate_rule_from_lesson, load_approved_lessons,
    )
    from src.utils.db_manager import DuckDBManager

    db = DuckDBManager()
    with db.connection() as conn:
        create_lessons_tables(conn)
        row = conn.execute(
            "SELECT lesson_text, competition_id, tier FROM agent_lessons WHERE id = ?", [lesson_id]
        ).fetchone()
        if row is None:
            raise ValueError(f"No lesson with id={lesson_id}")
        lesson_text, competition_id, tier = row

        cfg = AgentConfig.from_yaml(config_path) if config_path else AgentConfig.default()
        llm_invoke = _build_llm_invoke(cfg)

        if rule is not None:
            rule_text = rule
        else:
            rule_text = generate_rule_from_lesson(lesson_text, llm_invoke)
            if not rule_text:
                raise ValueError(
                    f"Could not auto-distill a rule for lesson {lesson_id} (LLM call failed or returned "
                    'empty). Re-run with --rule "..." to supply one manually.'
                )
            print(f"Auto-distilled rule: {rule_text}")

        # Reuses load_approved_lessons' own scope-aware query rather than
        # duplicating it: a pending lesson is never 'approved', so no self-
        # exclusion is needed, and this is exactly the same set a real live
        # match with this competition_id/tier would ever be loaded alongside
        # -- unlike a naive "competition_id = ? OR tier = ?" match, which
        # would (wrongly) count e.g. an unrelated competition's own
        # competition-scoped rule as co-occurring just because it happens to
        # share the same tier string.
        existing_rules = load_approved_lessons(conn, competition_id, tier)
        try:
            conflict = find_conflicting_rule(rule_text, existing_rules, llm_invoke)
        except Exception as exc:
            print(f"  warning: conflict check failed ({exc}) -- proceeding without it")
            conflict = None
        if conflict:
            if not force:
                raise ValueError(
                    f"Proposed rule conflicts with an existing approved rule: {conflict} "
                    "Re-run with --force to approve anyway, or reword the rule."
                )
            print(f"  warning: approving despite detected conflict: {conflict}")

        approve_lesson(conn, lesson_id, scope, reviewer or getpass.getuser(), rule_text)
    print(f"Approved lesson {lesson_id} (scope={scope})")


def run_agent_lessons_reject(lesson_id: int, reviewer: str | None) -> None:
    """Reject a pending lesson candidate (A33)."""
    import getpass

    from src.agent.lessons import create_lessons_tables, reject_lesson
    from src.utils.db_manager import DuckDBManager

    db = DuckDBManager()
    with db.connection() as conn:
        create_lessons_tables(conn)
        reject_lesson(conn, lesson_id, reviewer or getpass.getuser())
    print(f"Rejected lesson {lesson_id}")


def run_agent_compare(
    config_paths: list[str],
    from_date: str,
    to_date: str,
    league: str | None,
    sample: int | None,
    stake_mode: str,
) -> None:
    """Compare agent configs over the same backtest snapshot set (A16)."""
    from src.agent.comparison import compare_configs, print_comparison_table, save_comparison

    results = compare_configs(config_paths, from_date, to_date, league=league, sample=sample, stake_mode=stake_mode)
    print_comparison_table(results)
    path = save_comparison(results)
    print(f"\nComparison saved to {path}")


def run_status(db_manager: DuckDBManager) -> None:
    """Display data freshness and model selection status (US#80)."""
    from src.utils.model_selection import ModelSelector

    print("\n" + "=" * 60)
    print("FPAI System Status")
    print("=" * 60)

    # Data layer
    try:
        with db_manager.connection(read_only=True) as conn:
            raw_row = conn.execute("SELECT COUNT(*), MAX(date) FROM raw_matches").fetchone()
            feat_row = conn.execute("SELECT COUNT(*) FROM feature_store").fetchone()
        raw_count = int(raw_row[0]) if raw_row else 0
        max_date = raw_row[1] if raw_row else None
        feat_count = int(feat_row[0]) if feat_row else 0
        if max_date is not None:
            max_date_ts = pd.Timestamp(max_date).tz_localize(None)
            days_since = (pd.Timestamp.now().normalize() - max_date_ts.normalize()).days
        else:
            days_since = None
        print(f"\nData Layer:")
        print(f"  raw_matches:    {raw_count:,} rows | latest={max_date} | days_since={days_since}")
        print(f"  feature_store:  {feat_count:,} rows")
    except Exception as exc:
        print(f"  [error reading data layer: {exc}]")

    # MLflow experiment count
    try:
        experiments = mlflow.search_experiments()
        print(f"  mlflow_experiments: {len(experiments)}")
    except Exception:
        print("  mlflow_experiments: unavailable")

    # Model selections
    print("\nModel Selections:")
    selection_path = Path("config/model_selection.yaml")
    if not selection_path.exists():
        print("  no selection config (run select-best-models to populate)")
    else:
        selector = ModelSelector()
        config = selector.load_config()
        contexts_data = config.get("contexts", {})
        # US#110: show every registered context (e.g. E0, a future SWE,
        # international), plus any legacy bucket still on disk that the
        # current registry no longer accounts for -- rather than the old
        # hardcoded ["league", "international"], which silently hid any
        # second competition_specific competition's selections.
        known_contexts = sorted(set(list_context_keys()) | set(contexts_data.keys()))
        for ctx in known_contexts:
            ctx_data = contexts_data.get(ctx, {})
            if not ctx_data:
                print(f"  [{ctx}] no selections")
                continue
            print(f"  [{ctx}]")
            for tgt, info in ctx_data.items():
                model_type = info.get("model_type", "?")
                metric_val = info.get("metric_value", "?")
                metric_name = info.get("metric_name", "?")
                selected_at = info.get("selected_at", "?")
                print(f"    {tgt:<22} {model_type:<18} {metric_name}={metric_val}  selected={selected_at}")


def main() -> None:
    configure_logger()
    parser = _build_parser()
    args = parser.parse_args()
    app_settings = settings
    db_manager = DuckDBManager()

    if args.command == "scrape":
        run_scrape(
            app_settings,
            force=getattr(args, "force", False),
            league_page_url=getattr(args, "league_page_url", None),
            leagues=getattr(args, "league", None),
        )
    elif args.command == "ingest":
        run_ingest(app_settings, db_manager, force=getattr(args, "force", False))
    elif args.command == "refresh-data":
        run_refresh_data(app_settings, db_manager, league=str(args.league), force=getattr(args, "force", False))
    elif args.command == "schedule-refresh":
        run_schedule_refresh(
            league=str(args.league),
            day_of_week=args.day_of_week,
            hour=int(args.hour),
            minute=int(args.minute),
        )
    elif args.command == "train-target":
        run_train_target(target_name=str(args.target), model_name=args.model, context=str(args.context))
    elif args.command == "train-forecast-suite":
        run_train_forecast_suite(targets=args.targets, context=str(args.context))
    elif args.command == "forecast":
        run_forecast(
            league=args.league,
            match_ids=args.match_id,
            targets=args.target,
            limit=args.limit,
            home=args.home,
            away=args.away,
            date=args.date,
            odds_h=args.odds_h,
            odds_d=args.odds_d,
            odds_a=args.odds_a,
            match_type=str(args.match_type),
        )
    elif args.command == "compare-models":
        comparer = ModelComparison(experiment_name=args.experiment_name)
        output_path = args.output_path or f"reports/model_comparison/{args.target}_comparison.{args.format}"
        report_path = comparer.export_comparison_report(
            args.target, output_path, format=args.format, context=args.context
        )
        best = comparer.identify_best_model(str(args.target), context=args.context)
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
            competition_id=str(args.context),
        )
        print(f"Diagnostics report written to {report_path}")
    elif args.command == "permutation-importance":
        run_permutation_importance(
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
    elif args.command == "dixon-coles-baseline":
        run_dixon_coles_baseline(
            config_path=str(args.config_path),
            experiment_name=str(args.experiment_name),
            output_path=str(args.output_path),
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
    elif args.command == "fetch-fotmob":
        run_fetch_fotmob(
            app_settings, db_manager,
            league=str(args.league), from_season=args.from_season, to_season=args.to_season, delay=float(args.delay),
        )
    elif args.command == "fetch-lineups":
        run_fetch_lineups(
            db_manager,
            date_from=args.date_from,
            date_to=args.date_to,
            league=str(args.league),
            delay=float(args.delay),
        )
    elif args.command == "select-best-models":
        run_select_best_models(
            target=args.target,
            context=args.context,
            dry_run=args.dry_run,
            min_improvement=args.min_improvement,
        )
    elif args.command == "status":
        run_status(db_manager)
    elif args.command == "agent-recommend":
        run_agent_recommend(
            home_team=args.home,
            away_team=args.away,
            date=args.date,
            league=args.league,
            config_path=args.config,
            odds_h=args.odds_h,
            odds_d=args.odds_d,
            odds_a=args.odds_a,
        )
    elif args.command == "agent-snapshot":
        run_agent_snapshot(
            from_date=args.from_date,
            to_date=args.to_date,
            league=args.league,
            config_path=args.config,
            dry_run=args.dry_run,
        )
    elif args.command == "agent-backtest":
        run_agent_backtest(
            from_date=args.from_date,
            to_date=args.to_date,
            league=args.league,
            stake_mode=args.stake_mode,
            sample=args.sample,
            concurrency=args.concurrency,
            config_path=args.config,
            split=args.split,
            test_fraction=args.test_fraction,
            use_lessons=args.use_lessons,
        )
    elif args.command == "agent-train":
        run_agent_train(
            from_date=args.from_date,
            to_date=args.to_date,
            league=args.league,
            stake_mode=args.stake_mode,
            sample=args.sample,
            concurrency=args.concurrency,
            config_path=args.config,
            split=args.split,
            test_fraction=args.test_fraction,
            batch_size=args.batch_size,
        )
    elif args.command == "agent-lessons":
        if args.lessons_action == "approve":
            run_agent_lessons_approve(lesson_id=args.id, scope=args.scope, reviewer=args.reviewer, rule=args.rule, config_path=args.config, force=args.force)
        elif args.lessons_action == "reject":
            run_agent_lessons_reject(lesson_id=args.id, reviewer=args.reviewer)
    elif args.command == "agent-compare":
        run_agent_compare(
            config_paths=args.configs,
            from_date=args.from_date,
            to_date=args.to_date,
            league=args.league,
            sample=args.sample,
            stake_mode=args.stake_mode,
        )


if __name__ == "__main__":
    main()
