"""Feature engineering utilities for rolling match statistics."""

from __future__ import annotations

import pandas as pd
from src.utils.db_manager import DuckDBManager
from src.utils.helpers import standardize_team_name
from src.utils.logger import get_logger

LOGGER = get_logger(__name__)


def remove_margin(
    home_odds: pd.Series | float,
    draw_odds: pd.Series | float,
    away_odds: pd.Series | float,
) -> pd.DataFrame:
    """Remove bookmaker margin using the multiplicative (normalized implied) method."""
    odds = pd.concat(
        [
            pd.Series(home_odds, name="home"),
            pd.Series(draw_odds, name="draw"),
            pd.Series(away_odds, name="away"),
        ],
        axis=1,
    ).apply(pd.to_numeric, errors="coerce")
    raw = 1.0 / odds
    total = raw.sum(axis=1)
    probs = raw.div(total, axis=0)
    probs.columns = ["MKT_Home_Prob_Real", "MKT_Draw_Prob_Real", "MKT_Away_Prob_Real"]
    return probs


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
                       odds_h, odds_d, odds_a,
                       avgh, avgd, avga, xg_h, xg_a, xga_h, xga_a
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
        for col in ["xg_h", "xg_a", "xga_h", "xga_a"]:
            if col in raw_df.columns:
                raw_df[col] = pd.to_numeric(raw_df[col], errors="coerce")
        if "xg_a" in raw_df.columns and "xga_h" in raw_df.columns:
            raw_df["xga_h"] = raw_df["xga_h"].fillna(raw_df["xg_a"])
        if "xg_h" in raw_df.columns and "xga_a" in raw_df.columns:
            raw_df["xga_a"] = raw_df["xga_a"].fillna(raw_df["xg_h"])
        # avgh/avgd/avga absent from Football-Data CSVs before 2020; fall back
        # to single-bookmaker odds (odds_h/d/a, corr=0.994 with market average).
        raw_df["avgh"] = raw_df["avgh"].fillna(raw_df["odds_h"])
        raw_df["avgd"] = raw_df["avgd"].fillna(raw_df["odds_d"])
        raw_df["avga"] = raw_df["avga"].fillna(raw_df["odds_a"])

        def implied_probabilities(frame: pd.DataFrame) -> pd.DataFrame:
            odds = frame[["avgh", "avgd", "avga"]].apply(pd.to_numeric, errors="coerce")
            inv = 1.0 / odds
            total = inv.sum(axis=1)
            probs = inv.div(total, axis=0)
            return probs.rename(
                columns={
                    "avgh": "MKT_IMPLIED_HOME",
                    "avgd": "MKT_IMPLIED_DRAW",
                    "avga": "MKT_IMPLIED_AWAY",
                }
            )

        market_probs = implied_probabilities(raw_df)
        margin_removed = remove_margin(raw_df["avgh"], raw_df["avgd"], raw_df["avga"])
        margin_removed["MKT_H_Prob_Clean"] = margin_removed["MKT_Home_Prob_Real"]
        margin_removed["MKT_D_Prob_Clean"] = margin_removed["MKT_Draw_Prob_Real"]
        margin_removed["MKT_A_Prob_Clean"] = margin_removed["MKT_Away_Prob_Real"]

        home_df = raw_df[
            [
                "match_id",
                "date",
                "home_team",
                "fthg",
                "ftag",
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
                "xg_h",
                "xga_h",
            ]
        ].rename(columns={"home_team": "team"})
        away_df = raw_df[
            [
                "match_id",
                "date",
                "away_team",
                "fthg",
                "ftag",
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
                "xg_a",
                "xga_a",
            ]
        ].rename(columns={"away_team": "team"})

        home_df["shot_accuracy"] = home_df["hst"] / (home_df["hs"] + 0.1)
        home_df["discipline_score"] = home_df["hy"] + (home_df["hr"] * 3)
        home_df["save_rate"] = (home_df["ast"] - home_df["ftag"]) / (home_df["ast"] + 0.1)
        home_df["home_luck"] = home_df["fthg"] - home_df["xg_h"]
        away_df["shot_accuracy"] = away_df["ast"] / (away_df["as"] + 0.1)
        away_df["discipline_score"] = away_df["ay"] + (away_df["ar"] * 3)
        away_df["away_luck"] = away_df["ftag"] - away_df["xg_a"]

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

        def add_ema(
            frame: pd.DataFrame,
            prefix: str,
            ema_map: dict[str, tuple[str, str]],
            span: int = 5,
        ) -> pd.DataFrame:
            frame = frame.sort_values(["team", "date", "match_id"]).reset_index(drop=True)
            for stat, (group_prefix, label) in ema_map.items():
                col_name = f"{group_prefix}_{prefix}_{label}_EMA{span}"
                frame[col_name] = (
                    frame.groupby("team")[stat]
                    .transform(lambda s: s.shift(1).ewm(span=span, adjust=False).mean())
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
            "xg_h": ("OFF", "XG"),
            "xga_h": ("DEF", "XGA"),
            "home_luck": ("OFF", "LUCK"),
            "shot_accuracy": ("OFF", "SHOT_ACCURACY"),
            "discipline_score": ("DIS", "DISCIPLINE_SCORE"),
            "save_rate": ("DEF", "SAVE_RATE"),
        }
        home_ema_map = {
            "fthg": ("OFF", "FTHG"),
            "ftag": ("DEF", "FTAG"),
            "hst": ("OFF", "HST"),
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
            "xg_a": ("OFF", "XG"),
            "xga_a": ("DEF", "XGA"),
            "away_luck": ("OFF", "LUCK"),
            "shot_accuracy": ("OFF", "SHOT_ACCURACY"),
            "discipline_score": ("DIS", "DISCIPLINE_SCORE"),
        }
        away_ema_map = {
            "ftag": ("OFF", "FTAG"),
            "fthg": ("DEF", "FTHG"),
            "ast": ("OFF", "AST"),
        }

        home_df = add_rollings(home_df, "HOME", home_map)
        away_df = add_rollings(away_df, "AWAY", away_map)
        home_df = add_ema(home_df, "HOME", home_ema_map, span=5)
        away_df = add_ema(away_df, "AWAY", away_ema_map, span=5)

        home_df["CTX_HOME_REST_DAYS"] = (
            home_df.groupby("team")["date"].transform(lambda s: (s - s.shift(1)).dt.days)
        )
        away_df["CTX_AWAY_REST_DAYS"] = (
            away_df.groupby("team")["date"].transform(lambda s: (s - s.shift(1)).dt.days)
        )

        home_features = home_df[[col for col in home_df.columns if col.startswith(("OFF_", "DEF_", "DIS_", "CTX_"))] + ["match_id"]]
        away_features = away_df[[col for col in away_df.columns if col.startswith(("OFF_", "DEF_", "DIS_", "CTX_"))] + ["match_id"]]

        features = raw_df[["match_id"]].merge(home_features, on="match_id", how="left")
        features = features.merge(away_features, on="match_id", how="left")
        features["CTX_REST_DAYS_DIFF"] = (
            features["CTX_HOME_REST_DAYS"] - features["CTX_AWAY_REST_DAYS"]
        )
        features = features.join(market_probs).join(margin_removed)
        if "OFF_HOME_SHOT_ACCURACY_R5" in features.columns:
            features["OFF_Shot_Quality_R5"] = features["OFF_HOME_SHOT_ACCURACY_R5"]
        if "DEF_HOME_SAVE_RATE_R5" in features.columns:
            features["DEF_Save_Rate_R5"] = features["DEF_HOME_SAVE_RATE_R5"]
        if {"OFF_HOME_FTHG_R5", "DEF_AWAY_FTHG_R5"}.issubset(features.columns):
            features["STRENGTH_Goal_Diff"] = (
                features["OFF_HOME_FTHG_R5"] - features["DEF_AWAY_FTHG_R5"]
            )
        if {"OFF_HOME_HST_R5", "DEF_AWAY_HST_R5"}.issubset(features.columns):
            features["STRENGTH_SoT_Diff"] = (
                features["OFF_HOME_HST_R5"] - features["DEF_AWAY_HST_R5"]
            )
        if {"OFF_HOME_FTHG_R5", "OFF_AWAY_FTAG_R5"}.issubset(features.columns):
            features["INTERACTION_ATTACK_GOALS_DIFF_R5"] = (
                features["OFF_HOME_FTHG_R5"] - features["OFF_AWAY_FTAG_R5"]
            )
        if {"DEF_HOME_FTAG_R5", "DEF_AWAY_FTHG_R5"}.issubset(features.columns):
            features["INTERACTION_DEFENSE_GOALS_DIFF_R5"] = (
                features["DEF_HOME_FTAG_R5"] - features["DEF_AWAY_FTHG_R5"]
            )
        if {"OFF_HOME_HST_R5", "OFF_AWAY_AST_R5"}.issubset(features.columns):
            features["INTERACTION_ATTACK_SOT_DIFF_R5"] = (
                features["OFF_HOME_HST_R5"] - features["OFF_AWAY_AST_R5"]
            )
        if {"OFF_HOME_FTHG_R5", "DEF_AWAY_FTHG_R5"}.issubset(features.columns):
            features["EFFICIENCY_HOME_ATTACK_VS_AWAY_DEF_R5"] = (
                features["OFF_HOME_FTHG_R5"] / (features["DEF_AWAY_FTHG_R5"] + 0.1)
            )
        if {"OFF_AWAY_FTAG_R5", "DEF_HOME_FTAG_R5"}.issubset(features.columns):
            features["EFFICIENCY_AWAY_ATTACK_VS_HOME_DEF_R5"] = (
                features["OFF_AWAY_FTAG_R5"] / (features["DEF_HOME_FTAG_R5"] + 0.1)
            )
        if {
            "EFFICIENCY_HOME_ATTACK_VS_AWAY_DEF_R5",
            "EFFICIENCY_AWAY_ATTACK_VS_HOME_DEF_R5",
        }.issubset(features.columns):
            features["EFFICIENCY_ATTACK_MATCHUP_DIFF_R5"] = (
                features["EFFICIENCY_HOME_ATTACK_VS_AWAY_DEF_R5"]
                - features["EFFICIENCY_AWAY_ATTACK_VS_HOME_DEF_R5"]
            )
        # US#67: raw odds, overround, and log-odds features for the current match
        odds_feats = self._compute_odds_features(raw_df)
        features = features.merge(odds_feats, on="match_id", how="left")

        opp_adj = self._compute_opp_adjusted_rolling(raw_df)
        features = features.merge(opp_adj, on="match_id", how="left")

        # US#58: league standings (cumulative points + PPG-10) per team before each match
        league_ctx = self._compute_league_standings(raw_df)
        features = features.merge(league_ctx, on="match_id", how="left")

        # US#60: head-to-head rolling stats for the last 5 fixture meetings
        h2h = self._compute_h2h_rolling(raw_df)
        features = features.merge(h2h, on="match_id", how="left")

        # US#68: form variance (rolling std), decay EMA3, and streak features
        temporal = self._compute_temporal_features(raw_df)
        features = features.merge(temporal, on="match_id", how="left")

        # US#59: cold-start imputation — fill NaN rolling values with column means
        features = self._apply_cold_start_imputation(features)

        return features

    @staticmethod
    def _compute_opp_adjusted_rolling(raw_df: pd.DataFrame) -> pd.DataFrame:
        """Compute venue-independent rolling stats for opponent-adjusted matchup features.

        Combines home and away appearances into a single team timeline so each
        team's rolling form reflects all recent matches, not just home or away games.
        """
        home_rows = raw_df[["match_id", "date", "home_team", "fthg", "ftag", "hc", "ac", "hst"]].copy()
        home_rows["team"] = home_rows["home_team"]
        home_rows = home_rows.rename(columns={
            "fthg": "_gs", "ftag": "_gc", "hc": "_cs", "ac": "_cc", "hst": "_sot",
        }).drop(columns=["home_team"])

        away_rows = raw_df[["match_id", "date", "away_team", "ftag", "fthg", "ac", "hc", "ast"]].copy()
        away_rows["team"] = away_rows["away_team"]
        away_rows = away_rows.rename(columns={
            "ftag": "_gs", "fthg": "_gc", "ac": "_cs", "hc": "_cc", "ast": "_sot",
        }).drop(columns=["away_team"])

        stat_cols = ["_gs", "_gc", "_cs", "_cc", "_sot"]
        labels = ["GOALS_SCORED", "GOALS_CONCEDED", "CORNERS_SCORED", "CORNERS_CONCEDED", "SOT_SCORED"]

        timeline = pd.concat(
            [home_rows[["match_id", "date", "team"] + stat_cols],
             away_rows[["match_id", "date", "team"] + stat_cols]],
            ignore_index=True,
        )
        timeline = timeline.sort_values(["team", "date", "match_id"]).reset_index(drop=True)

        roll_cols = []
        for stat, label in zip(stat_cols, labels):
            for win in (3, 5):
                col = f"_opp_{label}_R{win}"
                timeline[col] = (
                    timeline.groupby("team")[stat]
                    .transform(lambda s, w=win: s.shift(1).rolling(w).mean())
                )
                roll_cols.append(col)

        team_stats = timeline[["match_id", "team"] + roll_cols].copy()

        home_rename = {c: c.replace("_opp_", "OPP_ADJ_HOME_") for c in roll_cols}
        home_stats = (
            raw_df[["match_id", "home_team"]]
            .merge(team_stats, left_on=["match_id", "home_team"], right_on=["match_id", "team"], how="left")
            .drop(columns=["home_team", "team"])
            .rename(columns=home_rename)
        )

        away_rename = {c: c.replace("_opp_", "OPP_ADJ_AWAY_") for c in roll_cols}
        away_stats = (
            raw_df[["match_id", "away_team"]]
            .merge(team_stats, left_on=["match_id", "away_team"], right_on=["match_id", "team"], how="left")
            .drop(columns=["away_team", "team"])
            .rename(columns=away_rename)
        )

        result = home_stats.merge(away_stats, on="match_id", how="left")

        for win in (3, 5):
            h_gs = f"OPP_ADJ_HOME_GOALS_SCORED_R{win}"
            a_gc = f"OPP_ADJ_AWAY_GOALS_CONCEDED_R{win}"
            a_gs = f"OPP_ADJ_AWAY_GOALS_SCORED_R{win}"
            h_gc = f"OPP_ADJ_HOME_GOALS_CONCEDED_R{win}"
            h_cs = f"OPP_ADJ_HOME_CORNERS_SCORED_R{win}"
            a_cc = f"OPP_ADJ_AWAY_CORNERS_CONCEDED_R{win}"
            a_cs = f"OPP_ADJ_AWAY_CORNERS_SCORED_R{win}"
            h_cc = f"OPP_ADJ_HOME_CORNERS_CONCEDED_R{win}"
            if {h_gs, a_gc}.issubset(result.columns):
                result[f"OPP_ADJ_GOAL_MATCHUP_HOME_R{win}"] = result[h_gs] - result[a_gc]
            if {a_gs, h_gc}.issubset(result.columns):
                result[f"OPP_ADJ_GOAL_MATCHUP_AWAY_R{win}"] = result[a_gs] - result[h_gc]
            if {h_cs, a_cc}.issubset(result.columns):
                result[f"OPP_ADJ_CORNER_MATCHUP_HOME_R{win}"] = result[h_cs] - result[a_cc]
            if {a_cs, h_cc}.issubset(result.columns):
                result[f"OPP_ADJ_CORNER_MATCHUP_AWAY_R{win}"] = result[a_cs] - result[h_cc]

        return result

    @staticmethod
    def _compute_league_standings(raw_df: pd.DataFrame) -> pd.DataFrame:
        """Compute pre-match cumulative points and recent PPG for home and away teams.

        Uses a shifted team timeline so all values reflect the state BEFORE each match.
        """
        import numpy as np

        home_rows = raw_df[["match_id", "date", "home_team", "fthg", "ftag"]].copy()
        home_rows["team"] = home_rows["home_team"]
        home_rows["pts"] = np.where(
            home_rows["fthg"] > home_rows["ftag"], 3,
            np.where(home_rows["fthg"] == home_rows["ftag"], 1, 0),
        )

        away_rows = raw_df[["match_id", "date", "away_team", "fthg", "ftag"]].copy()
        away_rows["team"] = away_rows["away_team"]
        away_rows["pts"] = np.where(
            away_rows["ftag"] > away_rows["fthg"], 3,
            np.where(away_rows["fthg"] == away_rows["ftag"], 1, 0),
        )

        timeline = pd.concat(
            [
                home_rows[["match_id", "date", "team", "pts"]],
                away_rows[["match_id", "date", "team", "pts"]],
            ],
            ignore_index=True,
        ).sort_values(["team", "date", "match_id"]).reset_index(drop=True)

        # Cumulative points before the current match (shift by 1, start from 0)
        timeline["cum_pts"] = (
            timeline.groupby("team")["pts"]
            .transform(lambda s: s.shift(1).cumsum().fillna(0))
        )

        # Points per game in last 10 matches before current (shift by 1)
        timeline["ppg_l10"] = (
            timeline.groupby("team")["pts"]
            .transform(lambda s: s.shift(1).rolling(10, min_periods=1).mean())
        )

        team_stats = timeline[["match_id", "team", "cum_pts", "ppg_l10"]]

        home_stats = (
            raw_df[["match_id", "home_team"]]
            .merge(
                team_stats,
                left_on=["match_id", "home_team"],
                right_on=["match_id", "team"],
                how="left",
            )
            .drop(columns=["home_team", "team"])
            .rename(columns={"cum_pts": "CTX_HOME_CUM_PTS", "ppg_l10": "CTX_HOME_PPG_L10"})
        )

        away_stats = (
            raw_df[["match_id", "away_team"]]
            .merge(
                team_stats,
                left_on=["match_id", "away_team"],
                right_on=["match_id", "team"],
                how="left",
            )
            .drop(columns=["away_team", "team"])
            .rename(columns={"cum_pts": "CTX_AWAY_CUM_PTS", "ppg_l10": "CTX_AWAY_PPG_L10"})
        )

        return home_stats.merge(away_stats, on="match_id", how="left")

    @staticmethod
    def _compute_h2h_rolling(raw_df: pd.DataFrame) -> pd.DataFrame:
        """Compute rolling H2H stats over the last 5 fixture meetings between each team pair.

        Tracks total goals, total corners, and current home-team win rate across all
        prior H2H meetings (regardless of venue). Uses shift(1) to exclude the current match.
        """
        df = raw_df[["match_id", "date", "home_team", "away_team", "fthg", "ftag", "hc", "ac"]].copy()
        df["total_goals"] = df["fthg"] + df["ftag"]
        df["total_corners"] = df["hc"].fillna(0) + df["ac"].fillna(0)

        # Build a per-team perspective: each team tracks stats vs each opponent
        home_view = df[["match_id", "date", "home_team", "away_team", "total_goals", "total_corners", "fthg", "ftag"]].copy()
        home_view["my_team"] = home_view["home_team"]
        home_view["opponent"] = home_view["away_team"]
        home_view["my_win"] = (home_view["fthg"] > home_view["ftag"]).astype(float)

        away_view = df[["match_id", "date", "home_team", "away_team", "total_goals", "total_corners", "fthg", "ftag"]].copy()
        away_view["my_team"] = away_view["away_team"]
        away_view["opponent"] = away_view["home_team"]
        away_view["my_win"] = (away_view["ftag"] > away_view["fthg"]).astype(float)

        pair_view = pd.concat(
            [
                home_view[["match_id", "date", "my_team", "opponent", "total_goals", "total_corners", "my_win"]],
                away_view[["match_id", "date", "my_team", "opponent", "total_goals", "total_corners", "my_win"]],
            ],
            ignore_index=True,
        ).sort_values(["my_team", "opponent", "date", "match_id"]).reset_index(drop=True)

        for col, out_col in [
            ("total_goals", "_h2h_goals"),
            ("total_corners", "_h2h_corners"),
            ("my_win", "_h2h_win"),
        ]:
            pair_view[out_col] = (
                pair_view.groupby(["my_team", "opponent"])[col]
                .transform(lambda s: s.shift(1).rolling(5, min_periods=1).mean())
            )

        # Join to home team: my_team = home_team, opponent = away_team
        h2h_stats = pair_view[["match_id", "my_team", "opponent", "_h2h_goals", "_h2h_corners", "_h2h_win"]]

        result = (
            raw_df[["match_id", "home_team", "away_team"]]
            .merge(
                h2h_stats,
                left_on=["match_id", "home_team", "away_team"],
                right_on=["match_id", "my_team", "opponent"],
                how="left",
            )
            .drop(columns=["home_team", "away_team", "my_team", "opponent"])
            .rename(columns={
                "_h2h_goals": "H2H_TOTAL_GOALS_R5",
                "_h2h_corners": "H2H_CORNERS_R5",
                "_h2h_win": "H2H_HOME_WIN_RATE_R5",
            })
        )
        return result

    @staticmethod
    def _compute_odds_features(raw_df: pd.DataFrame) -> pd.DataFrame:
        """Compute raw odds, overround, and log-odds features for the current match (US#67).

        These differ from the rolling MKT_ features: they encode the market's view
        of THIS specific fixture, not form derived from past match odds.
        """
        import numpy as np

        df = raw_df[["match_id", "avgh", "avgd", "avga"]].copy()
        for col in ["avgh", "avgd", "avga"]:
            df[col] = pd.to_numeric(df[col], errors="coerce")

        # Overround: sum of raw implied probabilities before normalisation — a proxy for
        # market confidence (lower overround = market is more decisive).
        df["MKT_OVERROUND"] = (1.0 / df["avgh"]) + (1.0 / df["avgd"]) + (1.0 / df["avga"])

        # Log-odds are more linearly useful for gradient models than raw decimal odds.
        df["MKT_LOG_ODDS_H"] = np.log(df["avgh"].clip(lower=1.01))
        df["MKT_LOG_ODDS_D"] = np.log(df["avgd"].clip(lower=1.01))
        df["MKT_LOG_ODDS_A"] = np.log(df["avga"].clip(lower=1.01))

        # Home-to-away log-odds ratio encodes relative market strength in one value.
        df["MKT_LOG_ODDS_H_A_RATIO"] = df["MKT_LOG_ODDS_H"] - df["MKT_LOG_ODDS_A"]

        return df[["match_id", "MKT_OVERROUND", "MKT_LOG_ODDS_H", "MKT_LOG_ODDS_D",
                   "MKT_LOG_ODDS_A", "MKT_LOG_ODDS_H_A_RATIO"]]

    @staticmethod
    def _compute_temporal_features(raw_df: pd.DataFrame) -> pd.DataFrame:
        """Compute form variance, EMA3, and streak features (US#68).

        Form variance: rolling std over R5 measures team consistency.
        EMA3: shorter decay window captures very recent form more aggressively.
        Streaks: consecutive matches where team scored / won / kept clean sheet.
        """
        import numpy as np

        # --- Build per-team timelines ---
        home_rows = raw_df[["match_id", "date", "home_team", "fthg", "ftag", "hc"]].copy()
        home_rows["team"] = home_rows["home_team"]
        home_rows = home_rows.rename(columns={"fthg": "gs", "ftag": "gc", "hc": "cs_corners"})

        away_rows = raw_df[["match_id", "date", "away_team", "ftag", "fthg", "ac"]].copy()
        away_rows["team"] = away_rows["away_team"]
        away_rows = away_rows.rename(columns={"ftag": "gs", "fthg": "gc", "ac": "cs_corners"})

        def _per_team(rows: pd.DataFrame) -> pd.DataFrame:
            rows = rows.sort_values(["team", "date", "match_id"]).reset_index(drop=True)
            # Form variance R5 (rolling std, shift 1 to exclude current match)
            rows["_gs_std_r5"] = rows.groupby("team")["gs"].transform(
                lambda s: s.shift(1).rolling(5, min_periods=2).std()
            )
            rows["_gc_std_r5"] = rows.groupby("team")["gc"].transform(
                lambda s: s.shift(1).rolling(5, min_periods=2).std()
            )
            rows["_corners_std_r5"] = rows.groupby("team")["cs_corners"].transform(
                lambda s: s.shift(1).rolling(5, min_periods=2).std()
            )
            # EMA3 for goals (shorter, more decay-sensitive than EMA5)
            rows["_gs_ema3"] = rows.groupby("team")["gs"].transform(
                lambda s: s.shift(1).ewm(span=3, adjust=False).mean()
            )
            rows["_gc_ema3"] = rows.groupby("team")["gc"].transform(
                lambda s: s.shift(1).ewm(span=3, adjust=False).mean()
            )
            # Streaks: consecutive matches where condition holds (shift 1)
            def streak(s: pd.Series, condition_fn) -> pd.Series:
                cond = condition_fn(s.shift(1))
                out = pd.Series(0.0, index=s.index)
                cnt = 0
                for i in range(len(cond)):
                    if pd.isna(cond.iloc[i]):
                        cnt = 0
                    elif cond.iloc[i]:
                        cnt += 1
                    else:
                        cnt = 0
                    out.iloc[i] = float(cnt)
                return out

            rows["_score_streak"] = rows.groupby("team")["gs"].transform(
                lambda s: streak(s, lambda x: x > 0)
            )
            rows["_won"] = (rows["gs"] > rows["gc"]).astype(float)
            rows["_win_streak"] = rows.groupby("team")["_won"].transform(
                lambda s: streak(s, lambda x: x > 0)
            )
            rows.drop(columns=["_won"], inplace=True)
            rows["_cs_streak"] = rows.groupby("team")["gc"].transform(
                lambda s: streak(s, lambda x: x == 0)
            )
            return rows

        home_t = _per_team(home_rows)
        away_t = _per_team(away_rows)

        rename_home = {
            "_gs_std_r5": "CTX_HOME_GOALS_STD_R5",
            "_gc_std_r5": "CTX_HOME_CONCEDED_STD_R5",
            "_corners_std_r5": "CTX_HOME_CORNERS_STD_R5",
            "_gs_ema3": "OFF_HOME_FTHG_EMA3",
            "_gc_ema3": "DEF_HOME_FTAG_EMA3",
            "_score_streak": "CTX_HOME_SCORE_STREAK",
            "_win_streak": "CTX_HOME_WIN_STREAK",
            "_cs_streak": "CTX_HOME_CS_STREAK",
        }
        rename_away = {
            "_gs_std_r5": "CTX_AWAY_GOALS_STD_R5",
            "_gc_std_r5": "CTX_AWAY_CONCEDED_STD_R5",
            "_corners_std_r5": "CTX_AWAY_CORNERS_STD_R5",
            "_gs_ema3": "OFF_AWAY_FTAG_EMA3",
            "_gc_ema3": "DEF_AWAY_FTHG_EMA3",
            "_score_streak": "CTX_AWAY_SCORE_STREAK",
            "_win_streak": "CTX_AWAY_WIN_STREAK",
            "_cs_streak": "CTX_AWAY_CS_STREAK",
        }
        new_cols = list(rename_home.keys())
        home_feat = home_t[["match_id"] + new_cols].rename(columns=rename_home)
        away_feat = away_t[["match_id"] + new_cols].rename(columns=rename_away)
        return home_feat.merge(away_feat, on="match_id", how="left")

    @staticmethod
    def _apply_cold_start_imputation(features: pd.DataFrame) -> pd.DataFrame:
        """Fill rolling-feature NaN values (cold-start rows) with column-wise means.

        Teams with fewer prior matches than the window size produce NaN rolling
        features. We impute with the overall column mean as a league-average prior.
        MKT_ features are excluded — their NaN encodes genuine missing odds data.
        """
        skip_prefix = ("MKT_", "match_id")
        imputable = [
            col for col in features.columns
            if not any(col.startswith(p) for p in skip_prefix)
            and pd.api.types.is_float_dtype(features[col])
            and features[col].isna().any()
        ]
        if imputable:
            col_means = features[imputable].mean()
            features[imputable] = features[imputable].fillna(col_means)
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
            "xg_h",
            "xg_a",
            "xga_h",
            "xga_a",
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

    def generate_feature_report(self) -> dict[str, object]:
        """Generate a lightweight report describing current engineered features."""
        df = self.compute_rolling_stats()
        feature_columns = [col for col in df.columns if col != "match_id"]
        report = {
            "feature_count": len(feature_columns),
            "features": feature_columns,
        }
        LOGGER.info("Feature report generated | feature_count=%s", len(feature_columns))
        return report

    @staticmethod
    def _ensure_feature_store_schema(conn, feature_columns: list[str]) -> None:
        """Ensure feature_store has all required columns for current pipeline version."""
        existing_columns = {
            row[1] for row in conn.execute("PRAGMA table_info('feature_store')").fetchall()
        }
        for column_name in feature_columns:
            if column_name not in existing_columns:
                conn.execute(f"ALTER TABLE feature_store ADD COLUMN {column_name} FLOAT")
