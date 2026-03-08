"""CLI entry point for the FPAI Prototype 1 pipeline."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from src.features.feature_factory import FeatureFactory
from src.ingestion.data_loader import CSVLoader
from src.models import LRModel, ModelManager
from src.strategy import Backtester, StrategyEngine
from src.utils import DuckDBManager, configure_logger, get_logger
from src.utils.config_loader import AppSettings, settings

LOGGER = get_logger(__name__)
FEATURE_COLUMNS = [
    "home_avg_goals_scored",
    "home_avg_goals_conceded",
    "away_avg_goals_scored",
    "away_avg_goals_conceded",
]


def _build_parser() -> argparse.ArgumentParser:
    """Create CLI parser with ingest, train, and predict subcommands."""
    parser = argparse.ArgumentParser(description="FPAI command line interface")
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("ingest", help="Ingest CSV data and pre-compute features")
    subparsers.add_parser("train", help="Train model and save artifact")
    subparsers.add_parser("predict", help="Load latest model and print value bets")
    backtest_parser = subparsers.add_parser("backtest", help="Run historical strategy backtest")
    backtest_parser.add_argument(
        "--ev_threshold",
        type=float,
        default=0.05,
        help="Minimum EV threshold to place bets (default: 0.05).",
    )
    return parser


def _get_latest_model_path(model_dir: Path) -> Path:
    """Return the latest LR model artifact from disk."""
    candidates = sorted(model_dir.glob("lr_v*_*.joblib"), key=lambda path: path.stat().st_mtime)
    if not candidates:
        raise FileNotFoundError(f"No saved model found in {model_dir}")
    return candidates[-1]


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
            r.home_team,
            r.away_team,
            r.odds_h,
            r.date,
            r.fthg,
            r.ftag,
            f.home_avg_goals_scored,
            f.home_avg_goals_conceded,
            f.away_avg_goals_scored,
            f.away_avg_goals_conceded
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
            columns=["match_id", "home_team", "away_team", "predicted_home_win_prob", "odds_h"]
        )

    probabilities = model.predict_proba(inference_df[FEATURE_COLUMNS])
    predicted_home_win_prob = probabilities[:, 1] if probabilities.ndim == 2 else probabilities.ravel()

    return pd.DataFrame(
        {
            "match_id": inference_df["match_id"].values,
            "home_team": inference_df["home_team"].values,
            "away_team": inference_df["away_team"].values,
            "predicted_home_win_prob": predicted_home_win_prob,
            "odds_h": inference_df["odds_h"].values,
        }
    )


def run_ingest(app_settings: AppSettings, db_manager: DuckDBManager) -> None:
    """Run ingestion and feature pre-computation pipeline."""
    LOGGER.info("Executing command: ingest")

    csv_path = Path(app_settings.paths.raw_data_dir) / "E0.csv"
    if not csv_path.exists():
        LOGGER.error("CSV file not found at %s", csv_path)
        return

    loader = CSVLoader()
    loader.process_v1_csv(file_path=str(csv_path), league_code="E0")

    factory = FeatureFactory()
    features_df = factory.compute_rolling_stats(window=app_settings.settings.rolling_window)
    factory.save_features(features_df)

    with db_manager.connection() as conn:
        raw_count = conn.execute("SELECT COUNT(*) FROM raw_matches").fetchone()
        feature_count = conn.execute("SELECT COUNT(*) FROM feature_store").fetchone()

    total_raw = int(raw_count[0]) if raw_count is not None else 0
    total_features = int(feature_count[0]) if feature_count is not None else 0
    LOGGER.info("Ingest complete | raw_matches=%s | feature_store=%s", total_raw, total_features)


def run_train() -> None:
    """Train model and persist artifact."""
    LOGGER.info("Executing command: train")

    model_manager = ModelManager(model=LRModel())
    model_path = model_manager.run_pipeline()
    LOGGER.info("Model saved to %s", model_path)


def run_predict(app_settings: AppSettings, db_manager: DuckDBManager) -> None:
    """Load latest model and output value-bet recommendations for recent matches."""
    LOGGER.info("Executing command: predict")

    model_dir = Path(app_settings.paths.model_dir)
    model_path = _get_latest_model_path(model_dir)
    model = LRModel.load(str(model_path))
    LOGGER.info("Loaded model artifact: %s", model_path)

    recent_df = _fetch_feature_joined_matches(db_manager, days=7)
    if recent_df.empty:
        LOGGER.warning("No recent matches available for prediction.")
        return

    predictions_df = _build_prediction_frame(model, recent_df)
    if predictions_df.empty:
        LOGGER.warning("No recent matches with complete features for prediction.")
        return

    strategy = StrategyEngine()
    recommendations = strategy.get_recommendations(predictions_df, ev_threshold=0.05)
    LOGGER.info("Value Bets (Current Week):")
    strategy.report_recommendations(recommendations)


def run_backtest(app_settings: AppSettings, db_manager: DuckDBManager, ev_threshold: float) -> None:
    """Load latest model and run historical backtest with configurable EV threshold."""
    LOGGER.info("Executing command: backtest")

    model_dir = Path(app_settings.paths.model_dir)
    model_path = _get_latest_model_path(model_dir)
    model = LRModel.load(str(model_path))
    LOGGER.info("Loaded model artifact: %s", model_path)

    historical_df = _fetch_feature_joined_matches(db_manager)
    if historical_df.empty:
        LOGGER.warning("No historical matches available for backtesting.")
        return

    predictions_df = _build_prediction_frame(model, historical_df)
    if predictions_df.empty:
        LOGGER.warning("No historical matches with complete features for backtesting.")
        return

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


def main() -> None:
    """Parse CLI args and dispatch the requested command."""
    configure_logger()
    parser = _build_parser()
    args = parser.parse_args()

    app_settings = settings
    db_manager = DuckDBManager()

    if args.command == "ingest":
        run_ingest(app_settings, db_manager)
    elif args.command == "train":
        run_train()
    elif args.command == "predict":
        run_predict(app_settings, db_manager)
    elif args.command == "backtest":
        run_backtest(app_settings, db_manager, ev_threshold=float(args.ev_threshold))


if __name__ == "__main__":
    main()
