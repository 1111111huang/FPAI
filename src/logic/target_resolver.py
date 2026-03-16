"""Target resolution logic for betting labels and payouts."""

from __future__ import annotations

from typing import Any

import pandas as pd


class TargetResolver:
    """Resolve labels and payouts for supported betting targets."""

    @staticmethod
    def get_label(df: pd.DataFrame, target_config: dict[str, Any]) -> pd.Series:
        """Return label series for the requested target."""
        target_type = str(target_config.get("target_type", "home_win")).strip().lower()
        if target_type == "home_win":
            if "FTR" in df.columns:
                return (df["FTR"] == "H").astype(int)
            if {"fthg", "ftag"}.issubset(df.columns):
                return (df["fthg"].astype(int) > df["ftag"].astype(int)).astype(int)
            raise ValueError("Missing columns for home_win label: need FTR or fthg/ftag.")
        raise ValueError(f"Unsupported target_type: {target_type}")

    @staticmethod
    def get_payout(
        df: pd.DataFrame,
        row_index: int,
        prediction: int,
        target_config: dict[str, Any],
    ) -> float:
        """Return payout for a single prediction and row."""
        target_type = str(target_config.get("target_type", "home_win")).strip().lower()
        stake = float(target_config.get("stake", 10.0))
        row = df.iloc[row_index]

        if target_type == "home_win":
            if prediction == 1 and row["FTR"] == "H":
                return float(row["AvgH"]) * stake - stake
            if prediction == 1 and row["FTR"] != "H":
                return -stake
            return 0.0

        # elif target_type == "total_goals":
        #     # Placeholder for future total goals payout logic.
        #     pass

        raise ValueError(f"Unsupported target_type: {target_type}")
