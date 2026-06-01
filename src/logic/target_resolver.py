"""Target resolution logic for forecast labels and legacy payouts."""

from __future__ import annotations

from typing import Any

import pandas as pd

from src.logic.target_registry import get_target_definition


class TargetResolver:
    """Resolve labels and payouts for supported targets."""

    @staticmethod
    def get_label(df: pd.DataFrame, target_config: dict[str, Any]) -> pd.Series:
        """Return label series for the requested target."""
        target_name = target_config.get("target") or target_config.get("target_type", "home_win")
        definition = get_target_definition(str(target_name))
        missing = [column for column in definition.label_columns if column not in df.columns]
        if missing:
            joined = ", ".join(missing)
            raise ValueError(f"Missing columns for {definition.name} label: {joined}.")

        if definition.name == "home_win":
            if "FTR" in df.columns:
                return (df["FTR"] == "H").astype(int)
            return (pd.to_numeric(df["fthg"], errors="coerce") > pd.to_numeric(df["ftag"], errors="coerce")).astype("Int64")
        if definition.name == "result_3way":
            home_goals = pd.to_numeric(df["fthg"], errors="coerce")
            away_goals = pd.to_numeric(df["ftag"], errors="coerce")
            labels = pd.Series(pd.NA, index=df.index, dtype="object")
            labels = labels.mask(home_goals > away_goals, "home")
            labels = labels.mask(home_goals == away_goals, "draw")
            labels = labels.mask(home_goals < away_goals, "away")
            return labels
        if definition.name == "btts":
            home_goals = pd.to_numeric(df["fthg"], errors="coerce")
            away_goals = pd.to_numeric(df["ftag"], errors="coerce")
            return ((home_goals > 0) & (away_goals > 0)).astype("Int64")
        if definition.name == "home_goals":
            return pd.to_numeric(df["fthg"], errors="coerce")
        if definition.name == "away_goals":
            return pd.to_numeric(df["ftag"], errors="coerce")
        if definition.name == "total_goals":
            return pd.to_numeric(df["fthg"], errors="coerce") + pd.to_numeric(df["ftag"], errors="coerce")
        if definition.name == "home_corners":
            return pd.to_numeric(df["hc"], errors="coerce")
        if definition.name == "away_corners":
            return pd.to_numeric(df["ac"], errors="coerce")
        if definition.name == "total_corners":
            return pd.to_numeric(df["hc"], errors="coerce") + pd.to_numeric(df["ac"], errors="coerce")

        raise ValueError(f"Unsupported target_type: {definition.name}")

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
