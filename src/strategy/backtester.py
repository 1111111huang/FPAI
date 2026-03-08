"""Backtesting utilities for value-bet strategy simulation."""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from src.utils.db_manager import DuckDBManager
from src.utils.logger import get_logger

LOGGER = get_logger(__name__)


@dataclass
class BacktestMetrics:
    """Container for backtesting performance metrics."""

    total_roi: float
    win_rate: float
    max_drawdown: float
    bets_placed: int
    final_bankroll: float


class Backtester:
    """Simulate fixed-stake betting performance on value-bet signals."""

    def __init__(
        self,
        initial_bankroll: float = 1000.0,
        bet_size: float = 10.0,
        config_path: str = "config.yaml",
    ) -> None:
        """Initialize simulation settings and shared database manager."""
        self.initial_bankroll = float(initial_bankroll)
        self.bet_size = float(bet_size)
        self.db_manager = DuckDBManager(config_path=config_path)
        self.bankroll = float(initial_bankroll)
        self.bet_history = pd.DataFrame()

    def run_simulation(self, predictions_df: pd.DataFrame, ev_threshold: float = 0.05) -> pd.DataFrame:
        """Run a backtest using EV-filtered predictions and match outcomes from DuckDB."""
        required_columns = {"match_id", "predicted_home_win_prob", "odds_h"}
        missing = required_columns.difference(predictions_df.columns)
        if missing:
            missing_text = ", ".join(sorted(missing))
            raise ValueError(f"Missing required columns in predictions_df: {missing_text}")

        df = predictions_df.copy()
        for optional_column in ("home_team", "away_team", "date", "fthg", "ftag"):
            if optional_column in df.columns:
                df = df.drop(columns=[optional_column])
        if "ev" not in df.columns:
            df["ev"] = (df["predicted_home_win_prob"].astype(float) * df["odds_h"].astype(float)) - 1.0

        match_ids = df["match_id"].dropna().astype(str).unique().tolist()
        if not match_ids:
            self.bet_history = pd.DataFrame()
            return self.bet_history

        with self.db_manager.connection() as conn:
            outcomes = conn.execute(
                """
                SELECT match_id, date, home_team, away_team, fthg, ftag
                FROM raw_matches
                WHERE match_id IN (
                    SELECT UNNEST(?)
                )
                """,
                [match_ids],
            ).fetchdf()

        merged = df.merge(outcomes, on="match_id", how="inner")
        merged = merged[merged["ev"] > ev_threshold].copy()
        if merged.empty:
            self.bet_history = pd.DataFrame()
            LOGGER.info("Backtest completed: no bets matched EV threshold %.4f.", ev_threshold)
            return self.bet_history

        merged = merged.sort_values(["date", "match_id"]).reset_index(drop=True)

        bankroll = self.initial_bankroll
        history_rows: list[dict[str, float | int | str]] = []

        for row in merged.itertuples(index=False):
            is_home_win = int(row.fthg) > int(row.ftag)
            if is_home_win:
                pnl = self.bet_size * (float(row.odds_h) - 1.0)
                result = "WIN"
            else:
                pnl = -self.bet_size
                result = "LOSS"

            bankroll += pnl
            history_row = {
                "match_id": row.match_id,
                "date": row.date,
                "home_team": row.home_team,
                "away_team": row.away_team,
                "ev": float(row.ev),
                "odds_h": float(row.odds_h),
                "fthg": int(row.fthg),
                "ftag": int(row.ftag),
                "result": result,
                "pnl": float(pnl),
                "bankroll": float(bankroll),
            }
            history_rows.append(history_row)

            LOGGER.info(
                "BET | %s vs %s | EV=%.4f | Odds=%.2f | Result=%s | PnL=%.2f | Bankroll=%.2f",
                row.home_team,
                row.away_team,
                float(row.ev),
                float(row.odds_h),
                result,
                float(pnl),
                float(bankroll),
            )

        self.bankroll = bankroll
        self.bet_history = pd.DataFrame(history_rows)
        metrics = self.get_metrics()
        LOGGER.info(
            "BACKTEST SUMMARY | Bets=%s | Final Bankroll=%.2f | ROI=%.4f | Win Rate=%.4f | Max Drawdown=%.4f",
            metrics.bets_placed,
            metrics.final_bankroll,
            metrics.total_roi,
            metrics.win_rate,
            metrics.max_drawdown,
        )
        return self.bet_history

    def get_metrics(self) -> BacktestMetrics:
        """Calculate ROI, win rate, and maximum drawdown from the latest simulation."""
        if self.bet_history.empty:
            return BacktestMetrics(
                total_roi=0.0,
                win_rate=0.0,
                max_drawdown=0.0,
                bets_placed=0,
                final_bankroll=self.initial_bankroll,
            )

        final_bankroll = float(self.bet_history["bankroll"].iloc[-1])
        total_roi = (final_bankroll - self.initial_bankroll) / self.initial_bankroll
        win_rate = float((self.bet_history["result"] == "WIN").mean())

        bankroll_series = pd.concat(
            [pd.Series([self.initial_bankroll]), self.bet_history["bankroll"]],
            ignore_index=True,
        )
        rolling_peak = bankroll_series.cummax()
        drawdown_series = (bankroll_series - rolling_peak) / rolling_peak
        max_drawdown = abs(float(drawdown_series.min()))

        return BacktestMetrics(
            total_roi=total_roi,
            win_rate=win_rate,
            max_drawdown=max_drawdown,
            bets_placed=int(len(self.bet_history)),
            final_bankroll=final_bankroll,
        )
