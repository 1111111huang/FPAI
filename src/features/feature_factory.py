"""Feature engineering utilities for rolling match statistics."""

from __future__ import annotations

import pandas as pd
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
        """Compute leakage-safe rolling features with 3- and 5-match windows."""
        with self.db_manager.connection() as conn:
            self._ensure_raw_matches_schema(conn)
            raw_df = conn.execute(
                """
                SELECT match_id, date, home_team, away_team,
                       fthg, ftag, hs, "as", hst, ast, hc, ac, hy, ay, hr, ar,
                       avgh, avgd, avga
                FROM raw_matches
                ORDER BY date, match_id
                """
            ).fetchdf()

        if raw_df.empty:
            return pd.DataFrame(columns=["match_id"])

        raw_df["date"] = pd.to_datetime(raw_df["date"], errors="coerce")
        raw_df = raw_df.dropna(subset=["date"]).reset_index(drop=True)
        raw_df["home_team"] = raw_df["home_team"].astype(str).map(standardize_team_name)
        raw_df["away_team"] = raw_df["away_team"].astype(str).map(standardize_team_name)

        def implied_probabilities(frame: pd.DataFrame) -> pd.DataFrame:
            odds = frame[["avgh", "avgd", "avga"]].apply(pd.to_numeric, errors="coerce")
            inv = 1.0 / odds
            total = inv.sum(axis=1)
            probs = inv.div(total, axis=0)
            return probs.rename(columns={"avgh": "MKT_IMPLIED_HOME", "avgd": "MKT_IMPLIED_DRAW", "avga": "MKT_IMPLIED_AWAY"})

        market_probs = implied_probabilities(raw_df)

        home_df = raw_df[
            ["match_id", "date", "home_team", "fthg", "ftag", "hs", "as", "hst", "ast", "hc", "ac", "hy", "ay", "hr", "ar"]
        ].rename(columns={"home_team": "team"})
        away_df = raw_df[
            ["match_id", "date", "away_team", "fthg", "ftag", "hs", "as", "hst", "ast", "hc", "ac", "hy", "ay", "hr", "ar"]
        ].rename(columns={"away_team": "team"})

        home_df["shot_accuracy"] = home_df["hst"] / (home_df["hs"] + 0.1)
        home_df["discipline_score"] = home_df["hy"] + (home_df["hr"] * 3)
        away_df["shot_accuracy"] = away_df["ast"] / (away_df["as"] + 0.1)
        away_df["discipline_score"] = away_df["ay"] + (away_df["ar"] * 3)

        def add_rollings(frame: pd.DataFrame, prefix: str, stat_map: dict[str, tuple[str, str]]) -> pd.DataFrame:
            frame = frame.sort_values(["team", "date", "match_id"]).reset_index(drop=True)
            for stat, (group_prefix, label) in stat_map.items():
                for win in (3, 5):
                    col_name = f"{group_prefix}_{prefix}_{label}_R{win}"
                    frame[col_name] = (
                        frame.groupby("team")[stat]
                        .transform(lambda s: s.shift(1).rolling(win).mean())
                    )
            return frame

        home_map = {
            "fthg": ("OFF", "FTHG"),
            "ftag": ("DEF", "FTAG"),
            "hs": ("OFF", "HS"),
            "as": ("DEF", "AS"),
            "hst": ("OFF", "HST"),
            "ast": ("DEF", "AST"),
            "hc": ("OFF", "HC"),
            "ac": ("DEF", "AC"),
            "hy": ("DIS", "HY"),
            "ay": ("DIS", "AY"),
            "hr": ("DIS", "HR"),
            "ar": ("DIS", "AR"),
            "shot_accuracy": ("OFF", "SHOT_ACCURACY"),
            "discipline_score": ("DIS", "DISCIPLINE_SCORE"),
        }
        away_map = {
            "ftag": ("OFF", "FTAG"),
            "fthg": ("DEF", "FTHG"),
            "as": ("OFF", "AS"),
            "hs": ("DEF", "HS"),
            "ast": ("OFF", "AST"),
            "hst": ("DEF", "HST"),
            "ac": ("OFF", "AC"),
            "hc": ("DEF", "HC"),
            "ay": ("DIS", "AY"),
            "hy": ("DIS", "HY"),
            "ar": ("DIS", "AR"),
            "hr": ("DIS", "HR"),
            "shot_accuracy": ("OFF", "SHOT_ACCURACY"),
            "discipline_score": ("DIS", "DISCIPLINE_SCORE"),
        }

        home_df = add_rollings(home_df, "HOME", home_map)
        away_df = add_rollings(away_df, "AWAY", away_map)

        home_df["CTX_HOME_REST_DAYS"] = (
            home_df.groupby("team")["date"].transform(lambda s: s.diff().dt.days)
        )
        away_df["CTX_AWAY_REST_DAYS"] = (
            away_df.groupby("team")["date"].transform(lambda s: s.diff().dt.days)
        )

        home_features = home_df[[col for col in home_df.columns if col.startswith(("OFF_", "DEF_", "DIS_", "CTX_"))] + ["match_id"]]
        away_features = away_df[[col for col in away_df.columns if col.startswith(("OFF_", "DEF_", "DIS_", "CTX_"))] + ["match_id"]]

        features = raw_df[["match_id"]].merge(home_features, on="match_id", how="left")
        features = features.merge(away_features, on="match_id", how="left")
        features = features.join(market_probs)
        return features

    @staticmethod
    def _ensure_raw_matches_schema(conn) -> None:
        """Ensure raw_matches has columns required for feature engineering."""
        columns = {row[1] for row in conn.execute("PRAGMA table_info('raw_matches')").fetchall()}
        required = [
            "hs",
            "as",
            "hst",
            "ast",
            "hc",
            "ac",
            "hy",
            "ay",
            "hr",
            "ar",
            "avgh",
            "avgd",
            "avga",
        ]
        for col in required:
            if col not in columns:
                if col == "as":
                    conn.execute('ALTER TABLE raw_matches ADD COLUMN "as" FLOAT')
                else:
                    conn.execute(f"ALTER TABLE raw_matches ADD COLUMN {col} FLOAT")

    def save_features(self, df: pd.DataFrame) -> None:
        """Create feature_store if needed and upsert feature rows by match_id."""
        with self.db_manager.connection() as conn:
            if df.empty:
                return
            column_defs = ["match_id TEXT PRIMARY KEY"]
            feature_columns = [col for col in df.columns if col != "match_id"]
            for col in feature_columns:
                column_defs.append(f"{col} FLOAT")
            conn.execute(
                f"""
                CREATE TABLE IF NOT EXISTS feature_store (
                    {", ".join(column_defs)}
                )
                """
            )
            self._ensure_feature_store_schema(conn, feature_columns)

            rows = [tuple(getattr(row, col) for col in ["match_id"] + feature_columns) for row in df.itertuples(index=False)]
            placeholders = ", ".join(["?"] * (1 + len(feature_columns)))
            conn.executemany(
                f"""
                INSERT OR REPLACE INTO feature_store
                (match_id, {", ".join(feature_columns)})
                VALUES ({placeholders})
                """,
                rows,
            )

    @staticmethod
    def _ensure_feature_store_schema(conn, feature_columns: list[str]) -> None:
        """Ensure feature_store has all required columns for current pipeline version."""
        existing_columns = {
            row[1] for row in conn.execute("PRAGMA table_info('feature_store')").fetchall()
        }
        for column_name in feature_columns:
            if column_name not in existing_columns:
                conn.execute(f"ALTER TABLE feature_store ADD COLUMN {column_name} FLOAT")
