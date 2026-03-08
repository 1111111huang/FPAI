"""Feature engineering utilities for rolling match statistics."""

from __future__ import annotations

import pandas as pd

from src.utils.db_manager import DuckDBManager


class FeatureFactory:
    """Compute and persist engineered football features in DuckDB."""

    def __init__(self, config_path: str = "config.yaml") -> None:
        """Initialize the feature factory with database config from YAML."""
        self.db_manager = DuckDBManager(config_path=config_path)

    def compute_rolling_stats(self, window: int = 5) -> pd.DataFrame:
        """Compute leakage-safe rolling goal averages for home and away teams."""
        with self.db_manager.connection() as conn:
            raw_df = conn.execute(
                """
                SELECT match_id, date, home_team, away_team, fthg, ftag
                FROM raw_matches
                ORDER BY date, match_id
                """
            ).fetchdf()

        if raw_df.empty:
            return pd.DataFrame(
                columns=[
                    "match_id",
                    "home_avg_goals_scored",
                    "home_avg_goals_conceded",
                    "away_avg_goals_scored",
                    "away_avg_goals_conceded",
                ]
            )

        home_view = raw_df[["match_id", "date", "home_team", "fthg", "ftag"]].rename(
            columns={
                "home_team": "team",
                "fthg": "goals_scored",
                "ftag": "goals_conceded",
            }
        )
        home_view["side"] = "home"

        away_view = raw_df[["match_id", "date", "away_team", "fthg", "ftag"]].rename(
            columns={
                "away_team": "team",
                "ftag": "goals_scored",
                "fthg": "goals_conceded",
            }
        )
        away_view["side"] = "away"

        team_history = pd.concat([home_view, away_view], ignore_index=True)
        team_history = team_history.sort_values(["team", "date", "match_id"]).reset_index(drop=True)

        team_history["avg_goals_scored"] = team_history.groupby("team")["goals_scored"].transform(
            lambda series: series.shift(1).rolling(window=window, min_periods=1).mean()
        )
        team_history["avg_goals_conceded"] = team_history.groupby("team")["goals_conceded"].transform(
            lambda series: series.shift(1).rolling(window=window, min_periods=1).mean()
        )

        home_features = team_history[team_history["side"] == "home"][
            ["match_id", "avg_goals_scored", "avg_goals_conceded"]
        ].rename(
            columns={
                "avg_goals_scored": "home_avg_goals_scored",
                "avg_goals_conceded": "home_avg_goals_conceded",
            }
        )

        away_features = team_history[team_history["side"] == "away"][
            ["match_id", "avg_goals_scored", "avg_goals_conceded"]
        ].rename(
            columns={
                "avg_goals_scored": "away_avg_goals_scored",
                "avg_goals_conceded": "away_avg_goals_conceded",
            }
        )

        features = raw_df[["match_id"]].merge(home_features, on="match_id", how="left")
        features = features.merge(away_features, on="match_id", how="left")

        return features[
            [
                "match_id",
                "home_avg_goals_scored",
                "home_avg_goals_conceded",
                "away_avg_goals_scored",
                "away_avg_goals_conceded",
            ]
        ]

    def save_features(self, df: pd.DataFrame) -> None:
        """Create feature_store if needed and upsert feature rows by match_id."""
        with self.db_manager.connection() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS feature_store (
                    match_id TEXT PRIMARY KEY,
                    home_avg_goals_scored FLOAT,
                    home_avg_goals_conceded FLOAT,
                    away_avg_goals_scored FLOAT,
                    away_avg_goals_conceded FLOAT
                )
                """
            )

            if df.empty:
                return

            rows = [
                (
                    row.match_id,
                    row.home_avg_goals_scored,
                    row.home_avg_goals_conceded,
                    row.away_avg_goals_scored,
                    row.away_avg_goals_conceded,
                )
                for row in df.itertuples(index=False)
            ]

            conn.executemany(
                """
                INSERT OR REPLACE INTO feature_store
                (match_id, home_avg_goals_scored, home_avg_goals_conceded,
                 away_avg_goals_scored, away_avg_goals_conceded)
                VALUES (?, ?, ?, ?, ?)
                """,
                rows,
            )
