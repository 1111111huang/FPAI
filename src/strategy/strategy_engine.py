"""Strategy logic for expected value filtering and CLI recommendations."""

from __future__ import annotations

import logging

import pandas as pd

LOGGER = logging.getLogger(__name__)


class StrategyEngine:
    """Detect value bets from model probabilities and bookmaker odds."""

    @staticmethod
    def calculate_ev(win_prob: float, odds: float) -> float:
        """Calculate expected value for a single market outcome."""
        return (win_prob * odds) - 1.0

    def get_recommendations(
        self, predictions_df: pd.DataFrame, ev_threshold: float = 0.05
    ) -> pd.DataFrame:
        """Filter and rank matches where EV exceeds the requested threshold."""
        required_columns = {
            "match_id",
            "home_team",
            "away_team",
            "predicted_home_win_prob",
            "odds_h",
        }
        missing = required_columns.difference(predictions_df.columns)
        if missing:
            missing_columns = ", ".join(sorted(missing))
            raise ValueError(f"Missing required columns: {missing_columns}")

        df = predictions_df.copy()
        df["ev"] = df.apply(
            lambda row: self.calculate_ev(
                win_prob=float(row["predicted_home_win_prob"]),
                odds=float(row["odds_h"]),
            ),
            axis=1,
        )

        recommendations = df[df["ev"] > ev_threshold].copy()
        recommendations = recommendations.sort_values("ev", ascending=False).reset_index(drop=True)

        return recommendations

    @staticmethod
    def report_recommendations(recommendations_df: pd.DataFrame) -> None:
        """Print recommendations in a clean CLI table format."""
        if recommendations_df.empty:
            LOGGER.info("No value bet recommendations found.")
            return

        display_columns = [
            "match_id",
            "home_team",
            "away_team",
            "predicted_home_win_prob",
            "odds_h",
            "ev",
        ]
        available_columns = [column for column in display_columns if column in recommendations_df.columns]
        table = recommendations_df[available_columns].copy()
        LOGGER.info("\n%s", table.to_string(index=False))
