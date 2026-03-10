"""Feature engineering utilities for rolling match statistics."""

from __future__ import annotations

import pandas as pd

from src.ingestion.schema import map_league_code_to_tier
from src.utils.db_manager import DuckDBManager
from src.utils.helpers import standardize_team_name
from src.utils.logger import get_logger

LOGGER = get_logger(__name__)


class FeatureFactory:
    """Compute and persist engineered football features in DuckDB."""

    def __init__(self, config_path: str = "config.yaml") -> None:
        """Initialize the feature factory with database config from YAML."""
        self.db_manager = DuckDBManager(config_path=config_path)

    def compute_rolling_stats(self, window: int = 5) -> pd.DataFrame:
        """Compute leakage-safe rolling and market-bias features for each match."""
        with self.db_manager.connection() as conn:
            raw_df = conn.execute(
                """
                SELECT match_id, league, tier, date, home_team, away_team, fthg, ftag, odds_h
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
                    "is_cold_start",
                    "relative_tier_change",
                    "market_prob_h",
                    "elo_rating_diff",
                    "home_advantage_trend",
                ]
            )

        raw_df["season_start_year"] = raw_df["date"].dt.year.where(
            raw_df["date"].dt.month >= 7, raw_df["date"].dt.year - 1
        )
        raw_df["tier"] = raw_df["tier"].fillna(raw_df["league"].map(map_league_code_to_tier)).astype(int)
        raw_df["home_team"] = raw_df["home_team"].astype(str).map(standardize_team_name)
        raw_df["away_team"] = raw_df["away_team"].astype(str).map(standardize_team_name)

        home_view = raw_df[
            ["match_id", "league", "tier", "season_start_year", "date", "home_team", "fthg", "ftag"]
        ].rename(
            columns={
                "home_team": "team",
                "fthg": "goals_scored",
                "ftag": "goals_conceded",
            }
        )
        home_view["side"] = "home"

        away_view = raw_df[
            ["match_id", "league", "tier", "season_start_year", "date", "away_team", "fthg", "ftag"]
        ].rename(
            columns={
                "away_team": "team",
                "ftag": "goals_scored",
                "fthg": "goals_conceded",
            }
        )
        away_view["side"] = "away"

        team_history = pd.concat([home_view, away_view], ignore_index=True)
        team_history = team_history.sort_values(
            ["team", "date", "match_id", "side"]
        ).reset_index(drop=True)

        team_history["prior_games"] = team_history.groupby("team").cumcount()
        team_history["avg_goals_scored"] = pd.NA
        team_history["avg_goals_conceded"] = pd.NA
        team_history["relative_tier_change"] = 0

        # Tier-normalized rolling features: prior matches in different tiers are decayed.
        for _, group in team_history.groupby("team", sort=False):
            indices = group.index.to_list()
            for position, idx in enumerate(indices):
                current_tier = int(team_history.at[idx, "tier"])

                if position > 0:
                    previous_tier = int(team_history.at[indices[position - 1], "tier"])
                    if current_tier < previous_tier:
                        team_history.at[idx, "relative_tier_change"] = 1
                    elif current_tier > previous_tier:
                        team_history.at[idx, "relative_tier_change"] = -1
                    else:
                        team_history.at[idx, "relative_tier_change"] = 0

                history_idx = indices[max(0, position - window) : position]
                if not history_idx:
                    continue

                history = team_history.loc[history_idx, ["goals_scored", "goals_conceded", "tier"]]
                tier_gap = (history["tier"].astype(int) - current_tier).abs()
                tier_decay = (1.0 - 0.15 * tier_gap).clip(lower=0.5, upper=1.0)
                team_history.at[idx, "avg_goals_scored"] = (
                    history["goals_scored"].astype(float) * tier_decay
                ).mean()
                team_history.at[idx, "avg_goals_conceded"] = (
                    history["goals_conceded"].astype(float) * tier_decay
                ).mean()

        # Previous-season promoted-team baseline: bottom-3 teams from prior season.
        home_points = raw_df[["league", "season_start_year", "home_team", "fthg", "ftag"]].copy()
        home_points["team"] = home_points["home_team"]
        home_points["points"] = (
            (home_points["fthg"] > home_points["ftag"]).astype(int) * 3
            + (home_points["fthg"] == home_points["ftag"]).astype(int)
        )
        home_points["goals_for"] = home_points["fthg"]
        home_points["goals_against"] = home_points["ftag"]

        away_points = raw_df[["league", "season_start_year", "away_team", "fthg", "ftag"]].copy()
        away_points["team"] = away_points["away_team"]
        away_points["points"] = (
            (away_points["ftag"] > away_points["fthg"]).astype(int) * 3
            + (away_points["fthg"] == away_points["ftag"]).astype(int)
        )
        away_points["goals_for"] = away_points["ftag"]
        away_points["goals_against"] = away_points["fthg"]

        team_season_table = pd.concat(
            [
                home_points[["league", "season_start_year", "team", "points", "goals_for", "goals_against"]],
                away_points[["league", "season_start_year", "team", "points", "goals_for", "goals_against"]],
            ],
            ignore_index=True,
        )
        season_standings = (
            team_season_table.groupby(["league", "season_start_year", "team"], as_index=False)[
                ["points", "goals_for", "goals_against"]
            ]
            .sum()
        )
        season_standings["goal_diff"] = season_standings["goals_for"] - season_standings["goals_against"]
        season_standings = season_standings.sort_values(
            ["league", "season_start_year", "points", "goal_diff", "goals_for", "team"],
            ascending=[True, True, True, True, True, True],
        )
        bottom_three = season_standings.groupby(["league", "season_start_year"]).head(3).copy()

        team_season_avg = (
            team_history.groupby(["league", "season_start_year", "team"], as_index=False)[
                ["goals_scored", "goals_conceded"]
            ]
            .mean()
        )
        bottom_three_baseline = bottom_three.merge(
            team_season_avg,
            on=["league", "season_start_year", "team"],
            how="left",
        )
        season_bottom_three_avg = (
            bottom_three_baseline.groupby(["league", "season_start_year"], as_index=False)[
                ["goals_scored", "goals_conceded"]
            ]
            .mean()
            .rename(
                columns={
                    "goals_scored": "promoted_baseline_scored",
                    "goals_conceded": "promoted_baseline_conceded",
                }
            )
        )
        prev_season_avg = season_bottom_three_avg.copy()
        prev_season_avg["season_start_year"] = prev_season_avg["season_start_year"] + 1
        prev_season_avg = prev_season_avg.rename(
            columns={
                "promoted_baseline_scored": "prev_season_promoted_baseline_scored",
                "promoted_baseline_conceded": "prev_season_promoted_baseline_conceded",
            }
        )
        team_history = team_history.merge(
            prev_season_avg,
            on=["league", "season_start_year"],
            how="left",
        )

        cold_start_mask = team_history["prior_games"] == 0
        team_history.loc[cold_start_mask, "avg_goals_scored"] = team_history.loc[
            cold_start_mask, "avg_goals_scored"
        ].fillna(team_history.loc[cold_start_mask, "prev_season_promoted_baseline_scored"])
        team_history.loc[cold_start_mask, "avg_goals_conceded"] = team_history.loc[
            cold_start_mask, "avg_goals_conceded"
        ].fillna(team_history.loc[cold_start_mask, "prev_season_promoted_baseline_conceded"])

        # Final fallback for earliest season with no prior-season baseline.
        league_fallback_scored = team_history.groupby("league")["goals_scored"].transform("mean")
        league_fallback_conceded = team_history.groupby("league")["goals_conceded"].transform("mean")
        team_history["avg_goals_scored"] = team_history["avg_goals_scored"].fillna(league_fallback_scored)
        team_history["avg_goals_conceded"] = team_history["avg_goals_conceded"].fillna(
            league_fallback_conceded
        )

        imputed_records = int(cold_start_mask.sum())
        historical_records = int(len(team_history) - imputed_records)
        LOGGER.info(
            "Hybrid feature generation | imputed_records=%s | historical_records=%s",
            imputed_records,
            historical_records,
        )

        home_features = team_history[team_history["side"] == "home"][
            ["match_id", "prior_games", "relative_tier_change", "avg_goals_scored", "avg_goals_conceded"]
        ].rename(
            columns={
                "prior_games": "home_prior_games",
                "relative_tier_change": "home_relative_tier_change",
                "avg_goals_scored": "home_avg_goals_scored",
                "avg_goals_conceded": "home_avg_goals_conceded",
            }
        )

        away_features = team_history[team_history["side"] == "away"][
            ["match_id", "prior_games", "relative_tier_change", "avg_goals_scored", "avg_goals_conceded"]
        ].rename(
            columns={
                "prior_games": "away_prior_games",
                "relative_tier_change": "away_relative_tier_change",
                "avg_goals_scored": "away_avg_goals_scored",
                "avg_goals_conceded": "away_avg_goals_conceded",
            }
        )

        features = raw_df[["match_id"]].merge(home_features, on="match_id", how="left")
        features = features.merge(away_features, on="match_id", how="left")

        # Market implied probability from home odds.
        features = features.merge(raw_df[["match_id", "odds_h"]], on="match_id", how="left")
        odds_h_numeric = pd.to_numeric(features["odds_h"], errors="coerce")
        features["market_prob_h"] = 1.0 / odds_h_numeric.where(odds_h_numeric > 0)

        # Simplified ELO-like score: +1 last match win, -1 last match loss, 0 draw.
        raw_sorted = raw_df.sort_values(["date", "match_id"]).reset_index(drop=True)
        team_score: dict[str, float] = {}
        elo_rows: list[dict[str, float | str]] = []
        for row in raw_sorted.itertuples(index=False):
            home_team = str(row.home_team)
            away_team = str(row.away_team)
            home_pre = float(team_score.get(home_team, 0.0))
            away_pre = float(team_score.get(away_team, 0.0))
            elo_rows.append({"match_id": row.match_id, "elo_rating_diff": home_pre - away_pre})

            if int(row.fthg) > int(row.ftag):
                home_delta, away_delta = 1.0, -1.0
            elif int(row.fthg) < int(row.ftag):
                home_delta, away_delta = -1.0, 1.0
            else:
                home_delta, away_delta = 0.0, 0.0
            team_score[home_team] = home_pre + home_delta
            team_score[away_team] = away_pre + away_delta

        features = features.merge(pd.DataFrame(elo_rows), on="match_id", how="left")

        # Home advantage trend over last 10 games:
        # home_points_avg(last 10 home) - overall_points_avg(last 10 all venues).
        team_points_history: dict[str, list[tuple[str, float]]] = {}
        trend_rows: list[dict[str, float | str]] = []
        for row in raw_sorted.itertuples(index=False):
            home_team = str(row.home_team)
            away_team = str(row.away_team)

            home_history = team_points_history.get(home_team, [])
            overall_last10 = [points for _, points in home_history[-10:]]
            home_last10 = [points for side, points in home_history if side == "home"][-10:]
            overall_avg = float(sum(overall_last10) / len(overall_last10)) if overall_last10 else 0.0
            home_avg = float(sum(home_last10) / len(home_last10)) if home_last10 else overall_avg
            trend_rows.append({"match_id": row.match_id, "home_advantage_trend": home_avg - overall_avg})

            if int(row.fthg) > int(row.ftag):
                home_points, away_points = 3.0, 0.0
            elif int(row.fthg) < int(row.ftag):
                home_points, away_points = 0.0, 3.0
            else:
                home_points, away_points = 1.0, 1.0

            team_points_history.setdefault(home_team, []).append(("home", home_points))
            team_points_history.setdefault(away_team, []).append(("away", away_points))

        features = features.merge(pd.DataFrame(trend_rows), on="match_id", how="left")
        features = features.drop(columns=["odds_h"])

        features["is_cold_start"] = (
            (features["home_prior_games"] < window) | (features["away_prior_games"] < window)
        )
        features["relative_tier_change"] = (
            features["home_relative_tier_change"].astype(float)
            - features["away_relative_tier_change"].astype(float)
        )

        return features[
            [
                "match_id",
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
                    away_avg_goals_conceded FLOAT,
                    is_cold_start BOOLEAN,
                    relative_tier_change FLOAT,
                    market_prob_h FLOAT,
                    elo_rating_diff FLOAT,
                    home_advantage_trend FLOAT
                )
                """
            )
            self._ensure_feature_store_schema(conn)

            if df.empty:
                return

            rows = [
                (
                    row.match_id,
                    row.home_avg_goals_scored,
                    row.home_avg_goals_conceded,
                    row.away_avg_goals_scored,
                    row.away_avg_goals_conceded,
                    bool(row.is_cold_start),
                    float(row.relative_tier_change),
                    float(row.market_prob_h) if pd.notna(row.market_prob_h) else None,
                    float(row.elo_rating_diff) if pd.notna(row.elo_rating_diff) else None,
                    float(row.home_advantage_trend) if pd.notna(row.home_advantage_trend) else None,
                )
                for row in df.itertuples(index=False)
            ]

            conn.executemany(
                """
                INSERT OR REPLACE INTO feature_store
                (match_id, home_avg_goals_scored, home_avg_goals_conceded,
                 away_avg_goals_scored, away_avg_goals_conceded, is_cold_start, relative_tier_change,
                 market_prob_h, elo_rating_diff, home_advantage_trend)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                rows,
            )

    @staticmethod
    def _ensure_feature_store_schema(conn) -> None:
        """Ensure feature_store has all required columns for current pipeline version."""
        existing_columns = {
            row[1] for row in conn.execute("PRAGMA table_info('feature_store')").fetchall()
        }
        required_columns = {
            "home_avg_goals_scored": "FLOAT",
            "home_avg_goals_conceded": "FLOAT",
            "away_avg_goals_scored": "FLOAT",
            "away_avg_goals_conceded": "FLOAT",
            "is_cold_start": "BOOLEAN",
            "relative_tier_change": "FLOAT",
            "market_prob_h": "FLOAT",
            "elo_rating_diff": "FLOAT",
            "home_advantage_trend": "FLOAT",
        }
        for column_name, column_type in required_columns.items():
            if column_name not in existing_columns:
                conn.execute(f"ALTER TABLE feature_store ADD COLUMN {column_name} {column_type}")
