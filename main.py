"""CLI entry point for the FPAI Prototype 1 pipeline."""

from __future__ import annotations

import argparse
from datetime import datetime
import itertools
from pathlib import Path
from tempfile import TemporaryDirectory

import mlflow
from mlflow.exceptions import MlflowException
import pandas as pd
import matplotlib.pyplot as plt
import yaml

from src.features.feature_factory import FeatureFactory
from src.ingestion import CSVLoader, FootballDataScraper
from src.models import LRModel, ModelFactory, ModelManager, XGBoostModel
from src.strategy import Backtester, StrategyEngine
from src.utils import DuckDBManager, configure_logger, get_logger
from src.utils.config_loader import AppSettings, settings

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
    "xgb": XGBoostModel,
    "xgboost": XGBoostModel,
}


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
    predict_parser = subparsers.add_parser("predict", help="Load latest model and print value bets")
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
    backtest_parser = subparsers.add_parser("backtest", help="Run historical strategy backtest")
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
    experiment_parser = subparsers.add_parser("experiment", help="Run MLflow experiment grid search")
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
                mlflow.xgboost.log_model(model.model, "model")
            else:
                mlflow.sklearn.log_model(model.model, "model")

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


if __name__ == "__main__":
    main()
