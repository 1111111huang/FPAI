"""Feature engineering utilities for rolling match statistics."""

from __future__ import annotations

from pathlib import Path

import duckdb
import pandas as pd
from src.ingestion.common.team_mapping import TeamNameMapper
from src.utils.db_manager import DuckDBManager
from src.utils.helpers import standardize_team_name
from src.utils.logger import get_logger

LOGGER = get_logger(__name__)

# W198 (2026-09-22): a team can have real raw_matches rows that are years
# stale -- e.g. a club relegated out of a tracked competition long ago, then
# fictionally listed as "current season" by a fixtures vendor. 400 days
# covers any normal in-season/off-season gap (a team playing every season
# straight through never goes 400 days between matches in the same
# competition) while still catching a team that's been gone at least a full
# season -- picked to be a round number safely above football's ~3-4 month
# off-season, not backtested/tuned.
_MAX_STALE_HISTORY_DAYS = 400


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

    def __init__(
        self,
        config_path: str = "config.yaml",
        default_max_retries: int = 5,
        default_retry_delay_seconds: float = 1.0,
    ) -> None:
        """Initialize the feature factory with database config from YAML.

        US#159: default_max_retries/default_retry_delay_seconds forwarded
        straight to the DuckDBManager this factory builds internally -- see
        CSVLoader.__init__'s identical parameter for the same reasoning."""
        self.db_manager = DuckDBManager(
            config_path=config_path,
            default_max_retries=default_max_retries,
            default_retry_delay_seconds=default_retry_delay_seconds,
        )

    def compute_rolling_stats(self, window: int = 5) -> pd.DataFrame:
        """Compute leakage-safe rolling features with 3- and 5-match windows."""
        with self.db_manager.connection() as conn:
            self._ensure_raw_matches_schema(conn)
            raw_df = conn.execute(
                """
                SELECT match_id, league, date, home_team, away_team,
                       fthg, ftag, hs, "as", hst, ast, hc, ac, hy, ay, hr, ar,
                       odds_h, odds_d, odds_a,
                       avgh, avgd, avga, maxch, maxcd, maxca, avgch, avgcd, avgca,
                       xg_h, xg_a, xga_h, xga_a,
                       over25_odds, under25_odds, ah_line, ah_home_odds, ah_away_odds
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

        # US#174: Dixon-Coles walk-forward stacking features
        dc_feats = self._compute_dixon_coles_features(raw_df)
        if not dc_feats.empty:
            features = features.merge(dc_feats, on="match_id", how="left")

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

        # US#96: squad-level rolling features (skipped when raw_player_match_stats absent)
        squad = self._compute_squad_features(raw_df)
        if not squad.empty:
            features = features.merge(squad, on="match_id", how="left")

        # US#208: squad-level rolling market value (skipped when raw_player_match_stats or player_market_values absent)
        squad_mkt_value = self._compute_squad_market_value_features(raw_df)
        if not squad_mkt_value.empty:
            features = features.merge(squad_mkt_value, on="match_id", how="left")

        # US#106: team-level luck burnout (skipped when raw_player_match_stats absent)
        luck = self._compute_luck_burnout_features(raw_df)
        if not luck.empty:
            features = features.merge(luck, on="match_id", how="left")

        # US#103: xOC — Top-3 Offensive Concentration (skipped when lineup tables absent)
        xoc = self._compute_xoc_features(raw_df)
        if not xoc.empty:
            features = features.merge(xoc, on="match_id", how="left")

        # US#102: FRDS — FotMob Rating Dominance Share (skipped when lineup tables absent)
        frds = self._compute_frds_features(raw_df)
        if not frds.empty:
            features = features.merge(frds, on="match_id", how="left")

        # US#104: Defensive Anchor (skipped when lineup/interceptions data absent)
        def_anchor = self._compute_defensive_anchor_features(raw_df)
        if not def_anchor.empty:
            features = features.merge(def_anchor, on="match_id", how="left")

        # US#175: key-attacker-absence flags (skipped when lineup tables absent)
        key_starter_absence = self._compute_key_starter_absence_features(raw_df)
        if not key_starter_absence.empty:
            features = features.merge(key_starter_absence, on="match_id", how="left")

        # US#59/US#134: cold-start imputation — fill NaN rolling values with
        # column means, computed per competition so cross-league data never
        # contaminates another league's fill values (see docstring for detail).
        league_by_match_id = raw_df.set_index("match_id")["league"]
        features = self._apply_cold_start_imputation(
            features, league=features["match_id"].map(league_by_match_id)
        )

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
    def _compute_dixon_coles_features(raw_df: pd.DataFrame, min_train_matches: int = 40) -> pd.DataFrame:
        """US#174/US#177/US#178: walk-forward Dixon-Coles stacking features.

        `DixonColesModel` is otherwise only fit once on the entire history
        for the standalone comparison baseline -- doing that here would leak
        every match's own future results into its own attack/defence
        ratings. Instead, refits once per (league, calendar month) on
        strictly prior matches only, then applies the resulting model to
        every fixture in that month. Matches in a league's first
        `min_train_matches` (default 40 -- enough rows for the MLE to be
        meaningfully identified, not just converge) get NaN, same as any
        other cold-start feature in this file.

        US#177: each month's fit is warm-started from the previous month's
        converged parameters (per league) instead of a cold zero vector --
        team strengths drift gradually, so this both converges faster and
        more reliably (fewer "did not fully converge" fits).

        US#178: a second, independent sub-model reuses the exact same
        generic Poisson machinery on `hc`/`ac` (corners are count data just
        like goals) to produce `DC_CORNER_*` features. Gated on its own
        per-month corner-data floor -- a league with hc/ac entirely absent
        (e.g. Sweden) or genuinely sparse for a stretch gets NaN for the
        corner columns only, independent of whether the goals sub-model has
        enough data.
        """
        import numpy as np
        from src.models.dixon_coles import DixonColesModel

        required = {"match_id", "league", "date", "home_team", "away_team", "fthg", "ftag"}
        if raw_df.empty or not required.issubset(raw_df.columns):
            return pd.DataFrame(columns=["match_id"])

        has_corners = {"hc", "ac"}.issubset(raw_df.columns)
        cols = list(required) + (["hc", "ac"] if has_corners else [])
        df = raw_df[cols].copy()
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
        df = df.dropna(subset=["date"])
        if df.empty:
            return pd.DataFrame(columns=["match_id"])
        df["_month"] = df["date"].values.astype("datetime64[M]")

        nan_goals = {"DC_LAMBDA_HOME": np.nan, "DC_LAMBDA_AWAY": np.nan, "DC_ATTACK_HOME": np.nan,
                     "DC_DEFENSE_HOME": np.nan, "DC_ATTACK_AWAY": np.nan, "DC_DEFENSE_AWAY": np.nan}
        nan_corner = {"DC_CORNER_LAMBDA_HOME": np.nan, "DC_CORNER_LAMBDA_AWAY": np.nan,
                      "DC_CORNER_ATTACK_HOME": np.nan, "DC_CORNER_DEFENSE_HOME": np.nan,
                      "DC_CORNER_ATTACK_AWAY": np.nan, "DC_CORNER_DEFENSE_AWAY": np.nan}

        rows = []
        for _league, league_df in df.groupby("league"):
            league_df = league_df.sort_values(["date", "match_id"])
            goals_warm_state: dict | None = None
            corner_warm_state: dict | None = None
            for month in sorted(league_df["_month"].unique()):
                train = league_df[league_df["date"] < month]
                test = league_df[league_df["_month"] == month]
                if len(train) < min_train_matches:
                    for match_id in test["match_id"]:
                        rows.append({"match_id": match_id, **nan_goals, **nan_corner})
                    continue

                goals_model = DixonColesModel().fit(
                    train[["home_team", "away_team", "fthg", "ftag"]], warm_start=goals_warm_state,
                )
                goals_warm_state = goals_model.get_state()

                corner_model = None
                if has_corners:
                    train_corners = train.dropna(subset=["hc", "ac"])
                    if len(train_corners) >= min_train_matches:
                        corner_train = train_corners[["home_team", "away_team", "hc", "ac"]].rename(
                            columns={"hc": "fthg", "ac": "ftag"}
                        )
                        corner_model = DixonColesModel().fit(corner_train, warm_start=corner_warm_state)
                        corner_warm_state = corner_model.get_state()

                for _, m in test.iterrows():
                    pred = goals_model.predict_match(m["home_team"], m["away_team"])
                    atk_h, dfc_h = goals_model.team_strengths(m["home_team"])
                    atk_a, dfc_a = goals_model.team_strengths(m["away_team"])
                    row = {
                        "match_id": m["match_id"],
                        "DC_LAMBDA_HOME": pred["home_goals"], "DC_LAMBDA_AWAY": pred["away_goals"],
                        "DC_ATTACK_HOME": atk_h, "DC_DEFENSE_HOME": dfc_h,
                        "DC_ATTACK_AWAY": atk_a, "DC_DEFENSE_AWAY": dfc_a,
                    }
                    if corner_model is not None:
                        c_pred = corner_model.predict_match(m["home_team"], m["away_team"])
                        c_atk_h, c_dfc_h = corner_model.team_strengths(m["home_team"])
                        c_atk_a, c_dfc_a = corner_model.team_strengths(m["away_team"])
                        row.update({
                            "DC_CORNER_LAMBDA_HOME": c_pred["home_goals"],
                            "DC_CORNER_LAMBDA_AWAY": c_pred["away_goals"],
                            "DC_CORNER_ATTACK_HOME": c_atk_h, "DC_CORNER_DEFENSE_HOME": c_dfc_h,
                            "DC_CORNER_ATTACK_AWAY": c_atk_a, "DC_CORNER_DEFENSE_AWAY": c_dfc_a,
                        })
                    else:
                        row.update(nan_corner)
                    rows.append(row)
        return pd.DataFrame(rows) if rows else pd.DataFrame(columns=["match_id"])

    @staticmethod
    def _compute_odds_features(raw_df: pd.DataFrame) -> pd.DataFrame:
        """Compute raw odds, overround, over/under 2.5, AH, and Poisson-decomposed features.

        US#67/BUG-009: market features for THIS specific fixture (not rolling form).
        US#76: Poisson lambda back-solved from O/U market; decomposed into team-level
               lambdas via the AH line for use across all 8 forecast targets.
        """
        import numpy as np
        from scipy.optimize import brentq

        src_cols = ["match_id", "avgh", "avgd", "avga"]
        for col in [
            "over25_odds", "under25_odds", "ah_line", "ah_home_odds", "ah_away_odds",
            "maxch", "maxcd", "maxca", "avgch", "avgcd", "avgca",
        ]:
            if col in raw_df.columns:
                src_cols.append(col)

        df = raw_df[src_cols].copy()
        for col in [
            "avgh", "avgd", "avga", "over25_odds", "under25_odds", "ah_home_odds", "ah_away_odds",
            "maxch", "maxcd", "maxca", "avgch", "avgcd", "avgca",
        ]:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")

        # Overround: sum of raw implied probabilities before normalisation.
        df["MKT_OVERROUND"] = (1.0 / df["avgh"]) + (1.0 / df["avgd"]) + (1.0 / df["avga"])

        # Over/under 2.5 goals: implied probabilities from market odds (BUG-009).
        if "over25_odds" in df.columns:
            df["MKT_IMPLIED_OVER25"] = (1.0 / df["over25_odds"].clip(lower=1.01)).where(
                df["over25_odds"].notna()
            )
        else:
            df["MKT_IMPLIED_OVER25"] = np.nan

        # Asian handicap: line and implied probabilities from average AH odds (BUG-009).
        if "ah_line" in df.columns:
            df["MKT_AH_LINE"] = pd.to_numeric(df["ah_line"], errors="coerce")
        else:
            df["MKT_AH_LINE"] = np.nan

        if "ah_home_odds" in df.columns:
            df["MKT_AH_HOME_ODDS"] = df["ah_home_odds"]
        else:
            df["MKT_AH_HOME_ODDS"] = np.nan

        if "ah_away_odds" in df.columns:
            df["MKT_AH_AWAY_ODDS"] = df["ah_away_odds"]
        else:
            df["MKT_AH_AWAY_ODDS"] = np.nan

        # US#76: Poisson lambda back-solved from P(Poisson(λ)≥3) = implied_over25_prob.
        # Team-level lambdas via AH line: λ_home = (λ+|AH|)/2, λ_away = (λ-|AH|)/2.
        def _poisson_cdf2(lam: float) -> float:
            """P(Poisson(λ) ≤ 2) = e^-λ (1 + λ + λ²/2)."""
            return np.exp(-lam) * (1.0 + lam + lam * lam / 2.0)

        def _solve_lambda(p_over25: float) -> float:
            if np.isnan(p_over25) or p_over25 <= 0.0 or p_over25 >= 1.0:
                return np.nan
            # P(X≥3) = p_over25  ⟺  _poisson_cdf2(λ) = 1 - p_over25
            target = 1.0 - p_over25
            try:
                return brentq(lambda lam: _poisson_cdf2(lam) - target, 0.01, 20.0)
            except ValueError:
                return np.nan

        p_over25_arr = df["MKT_IMPLIED_OVER25"].to_numpy(dtype=float)
        lambda_total = np.array([_solve_lambda(p) for p in p_over25_arr])
        df["MKT_LAMBDA_TOTAL"] = lambda_total

        ah_abs = df["MKT_AH_LINE"].abs().to_numpy(dtype=float)
        lam_home = np.where(
            np.isnan(lambda_total) | np.isnan(ah_abs),
            np.nan,
            (lambda_total + ah_abs) / 2.0,
        )
        lam_away = np.where(
            np.isnan(lambda_total) | np.isnan(ah_abs),
            np.nan,
            np.clip((lambda_total - ah_abs) / 2.0, 0.0, None),
        )
        df["MKT_LAMBDA_HOME"] = lam_home
        df["MKT_LAMBDA_AWAY"] = lam_away

        # Market-implied BTTS probability under independent Poisson assumptions.
        df["MKT_POISSON_BTTS_PROB"] = np.where(
            np.isnan(lam_home) | np.isnan(lam_away),
            np.nan,
            (1.0 - np.exp(-lam_home)) * (1.0 - np.exp(-lam_away)),
        )

        # Excess-goals signal: how much total scoring exceeds the handicap margin.
        df["MKT_LAMBDA_AH_DIFF"] = np.where(
            np.isnan(lambda_total) | np.isnan(ah_abs),
            np.nan,
            lambda_total - ah_abs,
        )

        # US#176: line movement (closing margin-removed implied prob minus
        # opening) and cross-bookmaker disagreement (raw implied prob at the
        # average closing price minus at the best/max closing price -- a
        # proxy for std-dev-across-books that needs only the Max/Avg columns
        # football-data.co.uk already ships, not one column per bookmaker).
        # NaN whenever avgch/maxch weren't ingested for this row (older
        # seasons, a gated-off league, or a match still awaiting kickoff).
        if {"avgch", "avgcd", "avgca"}.issubset(df.columns):
            opening_probs = remove_margin(df["avgh"], df["avgd"], df["avga"])
            closing_probs = remove_margin(df["avgch"], df["avgcd"], df["avgca"])
            df["MKT_LINE_MOVE_HOME"] = closing_probs["MKT_Home_Prob_Real"].to_numpy() - opening_probs["MKT_Home_Prob_Real"].to_numpy()
            df["MKT_LINE_MOVE_DRAW"] = closing_probs["MKT_Draw_Prob_Real"].to_numpy() - opening_probs["MKT_Draw_Prob_Real"].to_numpy()
            df["MKT_LINE_MOVE_AWAY"] = closing_probs["MKT_Away_Prob_Real"].to_numpy() - opening_probs["MKT_Away_Prob_Real"].to_numpy()
        else:
            df["MKT_LINE_MOVE_HOME"] = np.nan
            df["MKT_LINE_MOVE_DRAW"] = np.nan
            df["MKT_LINE_MOVE_AWAY"] = np.nan

        if {"maxch", "maxcd", "maxca", "avgch", "avgcd", "avgca"}.issubset(df.columns):
            df["MKT_BOOK_DISAGREEMENT_HOME"] = (1.0 / df["avgch"]) - (1.0 / df["maxch"])
            df["MKT_BOOK_DISAGREEMENT_DRAW"] = (1.0 / df["avgcd"]) - (1.0 / df["maxcd"])
            df["MKT_BOOK_DISAGREEMENT_AWAY"] = (1.0 / df["avgca"]) - (1.0 / df["maxca"])
        else:
            df["MKT_BOOK_DISAGREEMENT_HOME"] = np.nan
            df["MKT_BOOK_DISAGREEMENT_DRAW"] = np.nan
            df["MKT_BOOK_DISAGREEMENT_AWAY"] = np.nan

        return df[["match_id", "MKT_OVERROUND",
                   "MKT_IMPLIED_OVER25",
                   "MKT_AH_LINE", "MKT_AH_HOME_ODDS", "MKT_AH_AWAY_ODDS",
                   "MKT_LAMBDA_TOTAL", "MKT_LAMBDA_HOME", "MKT_LAMBDA_AWAY",
                   "MKT_POISSON_BTTS_PROB", "MKT_LAMBDA_AH_DIFF",
                   "MKT_LINE_MOVE_HOME", "MKT_LINE_MOVE_DRAW", "MKT_LINE_MOVE_AWAY",
                   "MKT_BOOK_DISAGREEMENT_HOME", "MKT_BOOK_DISAGREEMENT_DRAW", "MKT_BOOK_DISAGREEMENT_AWAY"]]

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
    def _apply_cold_start_imputation(
        features: pd.DataFrame, league: pd.Series | None = None
    ) -> pd.DataFrame:
        """Fill rolling-feature NaN values (cold-start rows) with column-wise means.

        Teams with fewer prior matches than the window size produce NaN rolling
        features. We impute with a column mean as a league-average prior.
        MKT_ features are excluded — their NaN encodes genuine missing odds data.

        US#134: the fill value is computed *per competition* when ``league`` is
        supplied (a Series aligned positionally to ``features``, e.g.
        ``features["match_id"].map(match_id_to_league)``). Before this, the mean
        was taken across the whole DataFrame regardless of competition, so once a
        second competition's rows enter the same table, a genuinely-cold-start gap
        in one competition would get diluted by another competition's statistics
        -- or, for a feature family a competition structurally can't populate,
        would silently import another competition's typical values wholesale.
        Grouping by league also means a competition whose entire column is NaN
        stays NaN (no cross-competition fallback), which is the correct signal.
        When ``league`` is omitted (or entirely null) we fall back to a single
        global mean -- correct for call sites already scoped to one competition,
        such as ``build_for_match``'s single-team-pair history.
        """
        skip_prefix = ("MKT_", "match_id")
        imputable = [
            col for col in features.columns
            if not any(col.startswith(p) for p in skip_prefix)
            and pd.api.types.is_float_dtype(features[col])
            and features[col].isna().any()
        ]
        if not imputable:
            return features
        if league is not None and league.notna().any():
            group_means = features[imputable].groupby(league.to_numpy()).transform("mean")
            features[imputable] = features[imputable].fillna(group_means)
        else:
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
            "maxch",
            "maxcd",
            "maxca",
            "avgch",
            "avgcd",
            "avgca",
            "xg_h",
            "xg_a",
            "xga_h",
            "xga_a",
            "over25_odds",
            "under25_odds",
            "ah_line",
            "ah_home_odds",
            "ah_away_odds",
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

    def build_for_match(
        self,
        home_team: str,
        away_team: str,
        match_date: str,
        league: str,
        odds_h: float,
        odds_d: float,
        odds_a: float,
        over25_odds: float | None = None,
        ah_line: float | None = None,
        ah_home_odds: float | None = None,
        ah_away_odds: float | None = None,
    ) -> pd.DataFrame:
        """Compute features for a single upcoming match without writing to DB (US#84).

        Fetches recent raw_matches history for both teams, appends a synthetic row for
        the requested match, computes all rolling/MKT/derived features, then returns the
        synthetic row's features as a single-row DataFrame.
        """
        import json
        import numpy as np

        home_norm = standardize_team_name(home_team)
        away_norm = standardize_team_name(away_team)

        # W06: resolve via the shared TeamNameMapper (same one ingestion uses)
        # instead of a bespoke inline lookup -- logs a warning on a mismatch
        # rather than silently passing an unresolved name through.
        mapping_path = Path(__file__).parent.parent.parent / "config" / "team_mapping.json"
        team_mapper = TeamNameMapper(mapping_path=str(mapping_path))
        home_norm = team_mapper.map_team(home_norm)
        away_norm = team_mapper.map_team(away_norm)

        LOGGER.info("build_for_match | home=%s away=%s date=%s league=%s", home_norm, away_norm, match_date, league)

        SYNTHETIC_ID = "__spot__"

        with self.db_manager.connection(read_only=True) as conn:
            self._ensure_raw_matches_schema(conn)
            raw_df = conn.execute(
                """
                SELECT match_id, league, date, home_team, away_team,
                       fthg, ftag, hs, "as", hst, ast, hc, ac, hy, ay, hr, ar,
                       odds_h, odds_d, odds_a,
                       avgh, avgd, avga, maxch, maxcd, maxca, avgch, avgcd, avgca,
                       xg_h, xg_a, xga_h, xga_a,
                       over25_odds, under25_odds, ah_line, ah_home_odds, ah_away_odds
                FROM raw_matches
                WHERE home_team = ? OR away_team = ? OR home_team = ? OR away_team = ?
                ORDER BY date, match_id
                """,
                [home_norm, home_norm, away_norm, away_norm],
            ).fetchdf()

        # US#108: detect a team with zero rows in raw_matches at all -- distinct
        # from the existing cold_start_risk/feature_completeness signal, which
        # only catches sparse/missing individual feature values for a team that
        # does have some history. Checked here (against the raw, unfiltered
        # fetch) before any cold-start imputation runs.
        home_mask = (raw_df["home_team"] == home_norm) | (raw_df["away_team"] == home_norm)
        away_mask = (raw_df["home_team"] == away_norm) | (raw_df["away_team"] == away_norm)
        home_has_history = bool(home_mask.any())
        away_has_history = bool(away_mask.any())

        # W198 (2026-09-22): found live -- football-data.org listed "Real
        # Racing Club de Santander"/"Malaga"/"La Coruna" as current-season
        # La Liga fixtures despite football-data.co.uk's own current-season
        # CSV (raw_matches' real source) never tracking them there this
        # season. Malaga/La Coruna both DO have real raw_matches rows (so
        # the zero-history check above didn't catch them) -- but every row
        # dates to 2016-2018, their actual last season in this competition.
        # Without this check they were silently treated as fully "known"
        # (feature_completeness ~0.92), computing rolling-window features
        # from an 8-year-old roster/era with no relationship to today's
        # team -- confirmed to have already reached a live direct_bet
        # recommendation on that basis before this was found.
        match_date_ts = pd.Timestamp(match_date)
        home_stale = home_has_history and (match_date_ts - raw_df.loc[home_mask, "date"].max()).days > _MAX_STALE_HISTORY_DAYS
        away_stale = away_has_history and (match_date_ts - raw_df.loc[away_mask, "date"].max()).days > _MAX_STALE_HISTORY_DAYS
        unknown_team = not home_has_history or not away_has_history or home_stale or away_stale

        # A flag alone isn't the fix -- a stale team's own rows must also be
        # dropped from the rolling-feature input, the same way a genuinely
        # zero-history team already has none, so its rolling features come
        # back NaN (then cold-start imputed) instead of silently reflecting
        # an 8-year-old roster/era. The other side's real, fresh rows (if
        # any) are untouched.
        if home_stale:
            raw_df = raw_df[~((raw_df["home_team"] == home_norm) | (raw_df["away_team"] == home_norm))]
        if away_stale:
            raw_df = raw_df[~((raw_df["home_team"] == away_norm) | (raw_df["away_team"] == away_norm))]

        if raw_df.empty:
            # No history — build empty history, synthetic row only; cold-start imputation covers NaNs
            raw_df = pd.DataFrame(columns=[
                "match_id", "league", "date", "home_team", "away_team", "fthg", "ftag",
                "hs", "as", "hst", "ast", "hc", "ac", "hy", "ay", "hr", "ar",
                "odds_h", "odds_d", "odds_a", "avgh", "avgd", "avga",
                "maxch", "maxcd", "maxca", "avgch", "avgcd", "avgca",
                "xg_h", "xg_a", "xga_h", "xga_a",
                "over25_odds", "under25_odds", "ah_line", "ah_home_odds", "ah_away_odds",
            ])

        # Build synthetic row (no result/in-match stats; MKT features from supplied odds)
        avgh_val = odds_h
        avgd_val = odds_d
        avga_val = odds_a
        synthetic_row: dict = {
            "match_id": SYNTHETIC_ID,
            "league": league,
            "date": pd.Timestamp(match_date),
            "home_team": home_norm,
            "away_team": away_norm,
            "fthg": np.nan, "ftag": np.nan,
            "hs": np.nan, "as": np.nan,
            "hst": np.nan, "ast": np.nan,
            "hc": np.nan, "ac": np.nan,
            "hy": np.nan, "ay": np.nan,
            "hr": np.nan, "ar": np.nan,
            "odds_h": odds_h, "odds_d": odds_d, "odds_a": odds_a,
            "avgh": avgh_val, "avgd": avgd_val, "avga": avga_val,
            # US#176: closing-line odds genuinely don't exist yet for a match
            # being forecast pre-kickoff -- always NaN here, same as
            # fthg/hs/etc above. MKT_LINE_MOVE_*/MKT_BOOK_DISAGREEMENT_* come
            # back NaN for this row by design, not a bug (existing MKT_
            # cold-start-imputation skip already tolerates this).
            "maxch": np.nan, "maxcd": np.nan, "maxca": np.nan,
            "avgch": np.nan, "avgcd": np.nan, "avgca": np.nan,
            "xg_h": np.nan, "xg_a": np.nan, "xga_h": np.nan, "xga_a": np.nan,
            "over25_odds": over25_odds, "under25_odds": np.nan,
            "ah_line": ah_line, "ah_home_odds": ah_home_odds, "ah_away_odds": ah_away_odds,
        }

        synthetic_df = pd.DataFrame([synthetic_row])
        # over25_odds/ah_line/ah_home_odds/ah_away_odds default to Python
        # None (not np.nan) when the caller doesn't supply them -- a column
        # built from a dict literal containing None infers `object` dtype,
        # not float64, which is the other half of the dtype mismatch this
        # function's pd.concat calls trigger pandas' "empty or all-NA
        # entries" FutureWarning for (the raw_df-side half, int32/float32
        # vs float64, is upcast in the `else` branch below). Confirmed via
        # direct column-by-column bisection against real data -- every
        # numeric synthetic column triggers it, always at a dtype mismatch,
        # never merely from being all-NaN at matching dtypes.
        _numeric_synthetic_cols = [
            "fthg", "ftag", "hs", "as", "hst", "ast", "hc", "ac", "hy", "ay", "hr", "ar",
            "odds_h", "odds_d", "odds_a", "avgh", "avgd", "avga",
            "maxch", "maxcd", "maxca", "avgch", "avgcd", "avgca",
            "xg_h", "xg_a", "xga_h", "xga_a",
            "over25_odds", "under25_odds", "ah_line", "ah_home_odds", "ah_away_odds",
        ]
        synthetic_df[_numeric_synthetic_cols] = synthetic_df[_numeric_synthetic_cols].astype("float64")
        if raw_df.empty:
            # A cold-start team (raw_df was just replaced with an empty,
            # explicitly-columned frame above) has no real dtypes of its
            # own to concat with -- pd.concat([empty_df, synthetic_df])
            # triggers pandas' "concatenation with empty or all-NA entries
            # is deprecated" FutureWarning every single time, since the
            # empty frame's per-column dtype is ambiguous. Nothing is lost
            # by skipping the concat entirely here: synthetic_row already
            # carries every column raw_df's manufactured empty frame would
            # have had.
            combined = synthetic_df.copy()
        else:
            raw_df["date"] = pd.to_datetime(raw_df["date"], errors="coerce")
            raw_df = raw_df.dropna(subset=["date"])
            # DuckDB returns fthg/ftag as int32 and most other stat columns
            # as float32, while synthetic_row's np.nan literals force
            # float64 -- concatenating a real int32/float32 column against a
            # same-named column that's entirely NaN (the synthetic row) at a
            # different numeric dtype is pandas' other trigger for the
            # "empty or all-NA entries" FutureWarning, distinct from (and
            # not covered by) the raw_df.empty branch above. Upcast raw_df's
            # numeric columns to float64 up front -- harmless, since
            # everything downstream already goes through
            # pd.to_numeric(..., errors="coerce") and cold-start imputation
            # regardless -- so both sides agree on dtype before concat.
            numeric_cols = raw_df.select_dtypes(include=["number"]).columns
            raw_df[numeric_cols] = raw_df[numeric_cols].astype("float64")
            combined = pd.concat([raw_df, synthetic_df], ignore_index=True)
        combined = combined.sort_values(["date", "match_id"]).reset_index(drop=True)
        combined["home_team"] = combined["home_team"].astype(str).map(standardize_team_name)
        combined["away_team"] = combined["away_team"].astype(str).map(standardize_team_name)

        # Run full feature computation on combined data
        for col in ["xg_h", "xg_a", "xga_h", "xga_a"]:
            if col in combined.columns:
                combined[col] = pd.to_numeric(combined[col], errors="coerce")
        if "xg_a" in combined.columns and "xga_h" in combined.columns:
            combined["xga_h"] = combined["xga_h"].fillna(combined["xg_a"])
        if "xg_h" in combined.columns and "xga_a" in combined.columns:
            combined["xga_a"] = combined["xga_a"].fillna(combined["xg_h"])
        combined["avgh"] = combined["avgh"].fillna(combined["odds_h"])
        combined["avgd"] = combined["avgd"].fillna(combined["odds_d"])
        combined["avga"] = combined["avga"].fillna(combined["odds_a"])

        # Re-use internal computation helpers (same as compute_rolling_stats)
        def implied_probabilities(frame: pd.DataFrame) -> pd.DataFrame:
            odds = frame[["avgh", "avgd", "avga"]].apply(pd.to_numeric, errors="coerce")
            inv = 1.0 / odds
            total = inv.sum(axis=1)
            probs = inv.div(total, axis=0)
            return probs.rename(columns={"avgh": "MKT_IMPLIED_HOME", "avgd": "MKT_IMPLIED_DRAW", "avga": "MKT_IMPLIED_AWAY"})

        market_probs = implied_probabilities(combined)
        margin_removed = remove_margin(combined["avgh"], combined["avgd"], combined["avga"])
        margin_removed["MKT_H_Prob_Clean"] = margin_removed["MKT_Home_Prob_Real"]
        margin_removed["MKT_D_Prob_Clean"] = margin_removed["MKT_Draw_Prob_Real"]
        margin_removed["MKT_A_Prob_Clean"] = margin_removed["MKT_Away_Prob_Real"]

        home_df = combined[["match_id", "date", "home_team", "fthg", "ftag", "hs", "as", "hst", "ast", "hc", "ac", "hy", "ay", "hr", "ar", "xg_h", "xga_h"]].rename(columns={"home_team": "team"})
        away_df = combined[["match_id", "date", "away_team", "fthg", "ftag", "hs", "as", "hst", "ast", "hc", "ac", "hy", "ay", "hr", "ar", "xg_a", "xga_a"]].rename(columns={"away_team": "team"})

        home_df["shot_accuracy"] = home_df["hst"] / (home_df["hs"] + 0.1)
        home_df["discipline_score"] = home_df["hy"] + (home_df["hr"] * 3)
        home_df["save_rate"] = (home_df["ast"] - home_df["ftag"]) / (home_df["ast"] + 0.1)
        home_df["home_luck"] = home_df["fthg"] - home_df["xg_h"]
        away_df["shot_accuracy"] = away_df["ast"] / (away_df["as"] + 0.1)
        away_df["discipline_score"] = away_df["ay"] + (away_df["ar"] * 3)
        away_df["away_luck"] = away_df["ftag"] - away_df["xg_a"]

        def add_rollings(frame: pd.DataFrame, prefix: str, stat_map: dict) -> pd.DataFrame:
            frame = frame.sort_values(["team", "date", "match_id"]).reset_index(drop=True)
            for stat, (group_prefix, label) in stat_map.items():
                for win in (3, 5):
                    col_name = f"{group_prefix}_{prefix}_{label}_R{win}"
                    frame[col_name] = frame.groupby("team")[stat].transform(lambda s, w=win: s.shift(1).rolling(w).mean())
            return frame

        def add_ema(frame: pd.DataFrame, prefix: str, ema_map: dict, span: int = 5) -> pd.DataFrame:
            frame = frame.sort_values(["team", "date", "match_id"]).reset_index(drop=True)
            for stat, (group_prefix, label) in ema_map.items():
                col_name = f"{group_prefix}_{prefix}_{label}_EMA{span}"
                frame[col_name] = frame.groupby("team")[stat].transform(lambda s, sp=span: s.shift(1).ewm(span=sp, adjust=False).mean())
            return frame

        home_map = {
            "fthg": ("OFF", "FTHG"), "ftag": ("DEF", "FTAG"), "hs": ("OFF", "HS"), "as": ("DEF", "AS"),
            "hst": ("OFF", "HST"), "ast": ("DEF", "AST"), "hc": ("OFF", "HC"), "ac": ("DEF", "AC"),
            "hy": ("DIS", "HY"), "ay": ("DIS", "AY"), "hr": ("DIS", "HR"), "ar": ("DIS", "AR"),
            "xg_h": ("OFF", "XG"), "xga_h": ("DEF", "XGA"), "home_luck": ("OFF", "LUCK"),
            "shot_accuracy": ("OFF", "SHOT_ACCURACY"), "discipline_score": ("DIS", "DISCIPLINE_SCORE"), "save_rate": ("DEF", "SAVE_RATE"),
        }
        home_ema_map = {"fthg": ("OFF", "FTHG"), "ftag": ("DEF", "FTAG"), "hst": ("OFF", "HST")}
        away_map = {
            "ftag": ("OFF", "FTAG"), "fthg": ("DEF", "FTHG"), "as": ("OFF", "AS"), "hs": ("DEF", "HS"),
            "ast": ("OFF", "AST"), "hst": ("DEF", "HST"), "ac": ("OFF", "AC"), "hc": ("DEF", "HC"),
            "ay": ("DIS", "AY"), "hy": ("DIS", "HY"), "ar": ("DIS", "AR"), "hr": ("DIS", "HR"),
            "xg_a": ("OFF", "XG"), "xga_a": ("DEF", "XGA"), "away_luck": ("OFF", "LUCK"),
            "shot_accuracy": ("OFF", "SHOT_ACCURACY"), "discipline_score": ("DIS", "DISCIPLINE_SCORE"),
        }
        away_ema_map = {"ftag": ("OFF", "FTAG"), "fthg": ("DEF", "FTHG"), "ast": ("OFF", "AST")}

        home_df = add_rollings(home_df, "HOME", home_map)
        away_df = add_rollings(away_df, "AWAY", away_map)
        home_df = add_ema(home_df, "HOME", home_ema_map, span=5)
        away_df = add_ema(away_df, "AWAY", away_ema_map, span=5)
        home_df["CTX_HOME_REST_DAYS"] = home_df.groupby("team")["date"].transform(lambda s: (s - s.shift(1)).dt.days)
        away_df["CTX_AWAY_REST_DAYS"] = away_df.groupby("team")["date"].transform(lambda s: (s - s.shift(1)).dt.days)

        home_features = home_df[[col for col in home_df.columns if col.startswith(("OFF_", "DEF_", "DIS_", "CTX_"))] + ["match_id"]]
        away_features = away_df[[col for col in away_df.columns if col.startswith(("OFF_", "DEF_", "DIS_", "CTX_"))] + ["match_id"]]

        features = combined[["match_id"]].merge(home_features, on="match_id", how="left")
        features = features.merge(away_features, on="match_id", how="left")
        features["CTX_REST_DAYS_DIFF"] = features.get("CTX_HOME_REST_DAYS", pd.Series(dtype=float)) - features.get("CTX_AWAY_REST_DAYS", pd.Series(dtype=float))
        features = features.join(market_probs.reset_index(drop=True)).join(margin_removed.reset_index(drop=True))

        if "OFF_HOME_SHOT_ACCURACY_R5" in features.columns:
            features["OFF_Shot_Quality_R5"] = features["OFF_HOME_SHOT_ACCURACY_R5"]
        if "DEF_HOME_SAVE_RATE_R5" in features.columns:
            features["DEF_Save_Rate_R5"] = features["DEF_HOME_SAVE_RATE_R5"]
        if {"OFF_HOME_FTHG_R5", "DEF_AWAY_FTHG_R5"}.issubset(features.columns):
            features["STRENGTH_Goal_Diff"] = features["OFF_HOME_FTHG_R5"] - features["DEF_AWAY_FTHG_R5"]
        if {"OFF_HOME_HST_R5", "DEF_AWAY_HST_R5"}.issubset(features.columns):
            features["STRENGTH_SoT_Diff"] = features["OFF_HOME_HST_R5"] - features["DEF_AWAY_HST_R5"]
        if {"OFF_HOME_FTHG_R5", "OFF_AWAY_FTAG_R5"}.issubset(features.columns):
            features["INTERACTION_ATTACK_GOALS_DIFF_R5"] = features["OFF_HOME_FTHG_R5"] - features["OFF_AWAY_FTAG_R5"]
        if {"DEF_HOME_FTAG_R5", "DEF_AWAY_FTHG_R5"}.issubset(features.columns):
            features["INTERACTION_DEFENSE_GOALS_DIFF_R5"] = features["DEF_HOME_FTAG_R5"] - features["DEF_AWAY_FTHG_R5"]
        if {"OFF_HOME_HST_R5", "OFF_AWAY_AST_R5"}.issubset(features.columns):
            features["INTERACTION_ATTACK_SOT_DIFF_R5"] = features["OFF_HOME_HST_R5"] - features["OFF_AWAY_AST_R5"]
        if {"OFF_HOME_FTHG_R5", "DEF_AWAY_FTHG_R5"}.issubset(features.columns):
            features["EFFICIENCY_HOME_ATTACK_VS_AWAY_DEF_R5"] = features["OFF_HOME_FTHG_R5"] / (features["DEF_AWAY_FTHG_R5"] + 0.1)
        if {"OFF_AWAY_FTAG_R5", "DEF_HOME_FTAG_R5"}.issubset(features.columns):
            features["EFFICIENCY_AWAY_ATTACK_VS_HOME_DEF_R5"] = features["OFF_AWAY_FTAG_R5"] / (features["DEF_HOME_FTAG_R5"] + 0.1)
        if {"EFFICIENCY_HOME_ATTACK_VS_AWAY_DEF_R5", "EFFICIENCY_AWAY_ATTACK_VS_HOME_DEF_R5"}.issubset(features.columns):
            features["EFFICIENCY_ATTACK_MATCHUP_DIFF_R5"] = features["EFFICIENCY_HOME_ATTACK_VS_AWAY_DEF_R5"] - features["EFFICIENCY_AWAY_ATTACK_VS_HOME_DEF_R5"]

        odds_feats = self._compute_odds_features(combined)
        features = features.merge(odds_feats, on="match_id", how="left")
        opp_adj = self._compute_opp_adjusted_rolling(combined)
        features = features.merge(opp_adj, on="match_id", how="left")
        league_ctx = self._compute_league_standings(combined)
        features = features.merge(league_ctx, on="match_id", how="left")
        h2h = self._compute_h2h_rolling(combined)
        features = features.merge(h2h, on="match_id", how="left")
        temporal = self._compute_temporal_features(combined)
        features = features.merge(temporal, on="match_id", how="left")

        # BUG-012 layer 1: mirror compute_rolling_stats' US#96/#106/#103/#102/#104
        # feature blocks so the live spot-forecast path stays in sync with the
        # offline feature_store pipeline (parity with lines ~306-329 above).
        squad = self._compute_squad_features(combined)
        if not squad.empty:
            features = features.merge(squad, on="match_id", how="left")
        squad_mkt_value = self._compute_squad_market_value_features(combined)
        if not squad_mkt_value.empty:
            features = features.merge(squad_mkt_value, on="match_id", how="left")
        luck = self._compute_luck_burnout_features(combined)
        if not luck.empty:
            features = features.merge(luck, on="match_id", how="left")
        xoc = self._compute_xoc_features(combined)
        if not xoc.empty:
            features = features.merge(xoc, on="match_id", how="left")
        frds = self._compute_frds_features(combined)
        if not frds.empty:
            features = features.merge(frds, on="match_id", how="left")
        def_anchor = self._compute_defensive_anchor_features(combined)
        if not def_anchor.empty:
            features = features.merge(def_anchor, on="match_id", how="left")
        # US#175: key-attacker-absence flags -- correctly NaN here for the
        # synthetic upcoming-match row (no confirmed lineup yet pre-kickoff),
        # same fallback the story called for.
        key_starter_absence = self._compute_key_starter_absence_features(combined)
        if not key_starter_absence.empty:
            features = features.merge(key_starter_absence, on="match_id", how="left")

        # US#175/US#174: Dixon-Coles stacking needs the WHOLE league's
        # history to fit meaningfully (MLE team strengths are only
        # identified from a full round-robin), not just these two teams'
        # own rows -- `combined` above is deliberately filtered to
        # home_norm/away_norm only, so a separate, unfiltered query is
        # required here rather than reusing it.
        dc_feats = self._compute_dixon_coles_features_for_spot_match(
            league=league, home_team=home_norm, away_team=away_norm,
            match_date=match_date, synthetic_id=SYNTHETIC_ID,
        )
        if not dc_feats.empty:
            features = features.merge(dc_feats, on="match_id", how="left")

        # US#134: group by league (see _apply_cold_start_imputation docstring).
        # combined's "league" column covers each historical row's own
        # competition plus the synthetic row's target league, so a team-name
        # collision across competitions can't cross-contaminate fill values.
        league_by_match_id = combined.set_index("match_id")["league"]
        features = self._apply_cold_start_imputation(
            features, league=features["match_id"].map(league_by_match_id)
        )

        # Return the synthetic match row only
        row = features[features["match_id"] == SYNTHETIC_ID].copy()
        if row.empty:
            raise RuntimeError("build_for_match: synthetic row missing after feature computation.")
        row = row.drop(columns=["match_id"], errors="ignore")
        row["_unknown_team"] = unknown_team
        return row.reset_index(drop=True)

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

    def _compute_squad_features(self, raw_df: pd.DataFrame) -> pd.DataFrame:
        """Query raw_player_match_stats and delegate to the pure rolling helper.

        Returns empty DataFrame (with only match_id column) when the table does
        not exist yet — feature_factory callers must check for emptiness before
        merging.
        """
        try:
            with self.db_manager.connection(read_only=True) as conn:
                player_df = conn.execute(
                    "SELECT match_id, team_name, xg, xa, rating FROM raw_player_match_stats"
                ).fetchdf()
        except duckdb.CatalogException:
            return pd.DataFrame(columns=["match_id"])
        return self._squad_rolling_from_data(player_df, raw_df)

    @staticmethod
    def _squad_rolling_from_data(
        player_df: pd.DataFrame, raw_df: pd.DataFrame
    ) -> pd.DataFrame:
        """Aggregate per-match player stats to rolling squad-level features.

        Applies shifted R3/R5 windows to ensure all SQUAD_* values reflect
        information available *before* the current match (pre-match safe).

        Args:
            player_df: Rows from raw_player_match_stats (match_id, team_name,
                xg, xa, rating). FotMob-abbreviated team names are normalised
                via standardize_team_name before joining.
            raw_df: Rows from raw_matches (match_id, date, home_team, away_team)
                with already-canonical team names.

        Returns:
            DataFrame keyed by match_id with 12 SQUAD_* columns, one row per
            match. Returns a single-column match_id DataFrame (empty) when
            player_df is empty.
        """
        from src.utils.helpers import standardize_team_name

        if player_df.empty:
            return pd.DataFrame(columns=["match_id"])

        # Normalise FotMob-abbreviated team names (e.g. "Man City" → "Manchester City")
        player_df = player_df.copy()
        player_df["team_std"] = player_df["team_name"].map(standardize_team_name)

        # Per-match per-team aggregate (NaN-safe: skip null xg/xa/rating values)
        agg = (
            player_df.groupby(["match_id", "team_std"])
            .agg(squad_xg=("xg", "mean"), squad_xa=("xa", "mean"), squad_rating=("rating", "mean"))
            .reset_index()
        )

        match_info = raw_df[["match_id", "date", "home_team", "away_team"]].copy()
        match_info["date"] = pd.to_datetime(match_info["date"])

        # BUG-012 layer 2: build each team's FULL per-match timeline (every
        # match_info row they appear in, home or away) rather than only the
        # matches where raw_player_match_stats happens to have a row for
        # them. A prior inner-join-by-match_id here silently dropped any
        # fixture lacking its own player-stats row (including
        # build_for_match()'s synthetic upcoming-match row, and any
        # historical match recorded before player-stat ingestion started)
        # from the rolling window entirely, producing NaN even when the
        # team's recent real history was available to carry forward.
        home_timeline = match_info[["match_id", "date", "home_team"]].rename(columns={"home_team": "team_std"})
        away_timeline = match_info[["match_id", "date", "away_team"]].rename(columns={"away_team": "team_std"})
        timeline = pd.concat([home_timeline, away_timeline], ignore_index=True)
        timeline = timeline.merge(
            agg[["match_id", "team_std", "squad_xg", "squad_xa", "squad_rating"]],
            on=["match_id", "team_std"],
            how="left",
        )
        timeline = timeline.sort_values(["team_std", "date", "match_id"]).reset_index(drop=True)

        # Shifted rolling windows per team (shift=1 guarantees pre-match safety).
        # Rolling .mean() already skips NaN gaps (matches with no stats row),
        # averaging over whichever real values fall inside the window.
        for metric in ["squad_xg", "squad_xa", "squad_rating"]:
            for window in [3, 5]:
                col = f"_{metric}_r{window}"
                timeline[col] = timeline.groupby("team_std")[metric].transform(
                    lambda s, w=window: s.shift(1).rolling(w, min_periods=1).mean()
                )

        stat_cols = [
            "_squad_xg_r3", "_squad_xg_r5",
            "_squad_xa_r3", "_squad_xa_r5",
            "_squad_rating_r3", "_squad_rating_r5",
        ]

        def _join_side(side: str) -> pd.DataFrame:
            rename_map = {
                "_squad_xg_r3":     f"SQUAD_{side}_XG_MEAN_R3",
                "_squad_xg_r5":     f"SQUAD_{side}_XG_MEAN_R5",
                "_squad_xa_r3":     f"SQUAD_{side}_XA_MEAN_R3",
                "_squad_xa_r5":     f"SQUAD_{side}_XA_MEAN_R5",
                "_squad_rating_r3": f"SQUAD_{side}_RATING_MEAN_R3",
                "_squad_rating_r5": f"SQUAD_{side}_RATING_MEAN_R5",
            }
            team_col = "home_team" if side == "HOME" else "away_team"
            joined = match_info.merge(
                timeline[["match_id", "team_std"] + stat_cols],
                left_on=["match_id", team_col],
                right_on=["match_id", "team_std"],
                how="left",
            )
            return joined.rename(columns=rename_map)[
                ["match_id"] + list(rename_map.values())
            ]

        home_feats = _join_side("HOME")
        away_feats = _join_side("AWAY")
        return home_feats.merge(away_feats, on="match_id", how="left")

    def _compute_squad_market_value_features(self, raw_df: pd.DataFrame) -> pd.DataFrame:
        """Query raw_player_match_stats + player_market_values and delegate
        to the pure rolling helper (US#208 Phase 2). Returns empty (match_id
        only) when either table doesn't exist yet -- same degradation
        contract as _compute_squad_features."""
        try:
            with self.db_manager.connection(read_only=True) as conn:
                player_df = conn.execute(
                    "SELECT match_id, team_name, player_id FROM raw_player_match_stats"
                ).fetchdf()
                values_df = conn.execute(
                    "SELECT fotmob_player_id, snapshot_date, market_value_eur FROM player_market_values"
                ).fetchdf()
        except duckdb.CatalogException:
            return pd.DataFrame(columns=["match_id"])
        return self._squad_market_value_rolling_from_data(player_df, values_df, raw_df)

    @staticmethod
    def _squad_market_value_rolling_from_data(
        player_df: pd.DataFrame, values_df: pd.DataFrame, raw_df: pd.DataFrame
    ) -> pd.DataFrame:
        """Aggregate per-match player market values to rolling squad-level
        features -- SQUAD_HOME/AWAY_MKT_VALUE_MEAN_R3/R5, the market-value
        analog of _squad_rolling_from_data's SQUAD_*_RATING_MEAN_R3/R5.

        Point-in-time correctness (the load-bearing requirement, design
        spec): each player-match row is joined to the LATEST
        player_market_values snapshot dated at-or-before that match's own
        date via pd.merge_asof(direction="backward") -- never a later
        snapshot, the same lookahead-bias class of bug W179 already had to
        fix for closing-line odds features.

        Args:
            player_df: Rows from raw_player_match_stats (match_id,
                team_name, player_id). FotMob-abbreviated team names are
                normalised via standardize_team_name before joining.
            values_df: Rows from player_market_values (fotmob_player_id,
                snapshot_date, market_value_eur) -- every dated snapshot,
                not pre-filtered to "latest".
            raw_df: Rows from raw_matches (match_id, date, home_team,
                away_team) with already-canonical team names.

        Returns:
            DataFrame keyed by match_id with 4 SQUAD_*_MKT_VALUE_MEAN_R3/R5
            columns, one row per match. Returns a single-column match_id
            DataFrame (empty) when player_df is empty.
        """
        from src.utils.helpers import standardize_team_name

        if player_df.empty:
            return pd.DataFrame(columns=["match_id"])

        match_info = raw_df[["match_id", "date", "home_team", "away_team"]].copy()
        match_info["date"] = pd.to_datetime(match_info["date"]).astype("datetime64[us]")

        player_df = player_df.copy()
        # Found live: some historical rows have a null player_id (a real
        # data-quality gap) -- merge_asof's by= key can't contain nulls at
        # all and raises, so these rows simply can't be matched to any
        # market value and are dropped, not assumed away.
        player_df = player_df[player_df["player_id"].notna()]
        if player_df.empty:
            return pd.DataFrame(columns=["match_id"])
        player_df["player_id"] = player_df["player_id"].astype("int64")
        player_df["team_std"] = player_df["team_name"].map(standardize_team_name)
        player_df = player_df.merge(match_info[["match_id", "date"]], on="match_id", how="left")
        # Found live: a handful of raw_player_match_stats rows reference a
        # match_id that isn't in raw_matches at all (orphaned historical
        # data) -- the left-join above leaves date as NaT for those, which
        # merge_asof also rejects outright. Drop them, same "can't be
        # matched" reasoning as the null player_id filter above.
        player_df = player_df[player_df["date"].notna()]
        if player_df.empty:
            return pd.DataFrame(columns=["match_id"])

        values = values_df.copy()
        # Explicit dtype (not just pd.to_datetime's own default resolution):
        # found live that DuckDB-sourced raw_matches.date comes back as
        # datetime64[us] while an independently-loaded snapshot_date column
        # can resolve to datetime64[ns] -- merge_asof raises on that
        # mismatch rather than coercing, so both sides must agree explicitly.
        values["snapshot_date"] = pd.to_datetime(values["snapshot_date"]).astype("datetime64[us]")
        values = values.sort_values("snapshot_date")

        # Point-in-time as-of join: for each player-match row, the latest
        # snapshot dated at-or-before that match's own date.
        player_df = player_df.sort_values("date")
        resolved = pd.merge_asof(
            player_df, values,
            left_on="date", right_on="snapshot_date",
            left_by="player_id", right_by="fotmob_player_id",
            direction="backward",
        )

        agg = (
            resolved.groupby(["match_id", "team_std"])
            .agg(squad_mkt_value=("market_value_eur", "mean"))
            .reset_index()
        )

        home_timeline = match_info[["match_id", "date", "home_team"]].rename(columns={"home_team": "team_std"})
        away_timeline = match_info[["match_id", "date", "away_team"]].rename(columns={"away_team": "team_std"})
        timeline = pd.concat([home_timeline, away_timeline], ignore_index=True)
        timeline = timeline.merge(agg[["match_id", "team_std", "squad_mkt_value"]], on=["match_id", "team_std"], how="left")
        timeline = timeline.sort_values(["team_std", "date", "match_id"]).reset_index(drop=True)

        for window in [3, 5]:
            col = f"_squad_mkt_value_r{window}"
            timeline[col] = timeline.groupby("team_std")["squad_mkt_value"].transform(
                lambda s, w=window: s.shift(1).rolling(w, min_periods=1).mean()
            )

        stat_cols = ["_squad_mkt_value_r3", "_squad_mkt_value_r5"]

        def _join_side(side: str) -> pd.DataFrame:
            rename_map = {
                "_squad_mkt_value_r3": f"SQUAD_{side}_MKT_VALUE_MEAN_R3",
                "_squad_mkt_value_r5": f"SQUAD_{side}_MKT_VALUE_MEAN_R5",
            }
            team_col = "home_team" if side == "HOME" else "away_team"
            joined = match_info.merge(
                timeline[["match_id", "team_std"] + stat_cols],
                left_on=["match_id", team_col],
                right_on=["match_id", "team_std"],
                how="left",
            )
            return joined.rename(columns=rename_map)[["match_id"] + list(rename_map.values())]

        home_feats = _join_side("HOME")
        away_feats = _join_side("AWAY")
        return home_feats.merge(away_feats, on="match_id", how="left")

    def _compute_luck_burnout_features(self, raw_df: pd.DataFrame) -> pd.DataFrame:
        """Query raw_player_match_stats and compute team-level luck burnout features.

        Returns empty DataFrame (with only match_id column) when the table does
        not exist yet — callers must check for emptiness before merging.
        """
        try:
            with self.db_manager.connection(read_only=True) as conn:
                player_df = conn.execute(
                    "SELECT match_id, team_name, goals, assists, xg, xa FROM raw_player_match_stats"
                ).fetchdf()
        except duckdb.CatalogException:
            return pd.DataFrame(columns=["match_id"])
        return self._luck_burnout_from_data(player_df, raw_df)

    @staticmethod
    def _luck_burnout_from_data(
        player_df: pd.DataFrame, raw_df: pd.DataFrame
    ) -> pd.DataFrame:
        """Compute team-level luck burnout: rolling 5-match (G+A) − (xG+xA) per team.

        Applies shift(1).rolling(5, min_periods=1).mean() so all values reflect
        information available before the current match (pre-match safe).

        Returns:
            DataFrame keyed by match_id with columns
            [match_id, LUCK_HOME_BURNOUT_R5, LUCK_AWAY_BURNOUT_R5].
            Returns a single-column match_id DataFrame (empty) when player_df is empty.
        """
        from src.utils.helpers import standardize_team_name

        if player_df.empty:
            return pd.DataFrame(columns=["match_id"])

        player_df = player_df.copy()
        for col in ["goals", "assists", "xg", "xa"]:
            player_df[col] = pd.to_numeric(player_df[col], errors="coerce").fillna(0.0)
        player_df["team_std"] = player_df["team_name"].map(standardize_team_name)

        agg = (
            player_df.groupby(["match_id", "team_std"])[["goals", "assists", "xg", "xa"]]
            .sum()
            .eval("luck = goals + assists - xg - xa")[["luck"]]
            .reset_index()
        )

        match_info = raw_df[["match_id", "date", "home_team", "away_team"]].copy()
        match_info["date"] = pd.to_datetime(match_info["date"])

        # BUG-012 layer 2: build each team's full per-match timeline (home or
        # away appearances in match_info), not just matches where player_df
        # has a row for them — see _squad_rolling_from_data for the identical
        # rationale (an exact-match_id join otherwise drops any fixture
        # lacking its own stats row, including build_for_match()'s synthetic
        # upcoming-match row, from the rolling window entirely).
        home_timeline = match_info[["match_id", "date", "home_team"]].rename(columns={"home_team": "team_std"})
        away_timeline = match_info[["match_id", "date", "away_team"]].rename(columns={"away_team": "team_std"})
        timeline = pd.concat([home_timeline, away_timeline], ignore_index=True)
        timeline = timeline.merge(agg[["match_id", "team_std", "luck"]], on=["match_id", "team_std"], how="left")
        timeline = timeline.sort_values(["team_std", "date", "match_id"]).reset_index(drop=True)

        timeline["_luck_r5"] = timeline.groupby("team_std")["luck"].transform(
            lambda s: s.shift(1).rolling(5, min_periods=1).mean()
        )

        def _join_side(side: str, team_col: str, out_col: str) -> pd.DataFrame:
            return (
                match_info.merge(
                    timeline[["match_id", "team_std", "_luck_r5"]],
                    left_on=["match_id", team_col],
                    right_on=["match_id", "team_std"],
                    how="left",
                )
                .rename(columns={"_luck_r5": out_col})[["match_id", out_col]]
            )

        home_feats = _join_side("HOME", "home_team", "LUCK_HOME_BURNOUT_R5")
        away_feats = _join_side("AWAY", "away_team", "LUCK_AWAY_BURNOUT_R5")
        return home_feats.merge(away_feats, on="match_id", how="left")

    def _compute_xoc_features(self, raw_df: pd.DataFrame) -> pd.DataFrame:
        """Query match_lineups + raw_player_match_stats and compute xOC features.

        Returns empty DataFrame (with only match_id column) when either table is absent.
        """
        try:
            with self.db_manager.connection(read_only=True) as conn:
                lineups_df = conn.execute(
                    "SELECT fotmob_match_id, player_id, team_name, side, position_group"
                    " FROM match_lineups"
                ).fetchdf()
                player_df = conn.execute(
                    "SELECT match_id, player_id, team_name, minutes_played, xg, xa"
                    " FROM raw_player_match_stats"
                ).fetchdf()
        except duckdb.CatalogException:
            return pd.DataFrame(columns=["match_id"])
        from src.features.lineup_features import compute_xoc
        return compute_xoc(lineups_df, player_df, raw_df)

    def _compute_dixon_coles_features_for_spot_match(
        self, league: str, home_team: str, away_team: str, match_date: str, synthetic_id: str,
    ) -> pd.DataFrame:
        """US#174: live-serving counterpart to `_compute_dixon_coles_features`.

        `build_for_match`'s own `combined` frame is deliberately filtered to
        the two requested teams' own rows (see its query), which is the
        wrong input for Dixon-Coles -- MLE team strengths need the whole
        league's round robin to be identified at all. Fetches that league's
        full history fresh, appends a single synthetic future row for the
        requested fixture, and reuses `_compute_dixon_coles_features`
        unchanged (its own per-month grouping already treats the synthetic
        row's month as strictly-future relative to every real row, so this
        is leakage-safe by construction, not by a second implementation).
        """
        with self.db_manager.connection(read_only=True) as conn:
            league_df = conn.execute(
                "SELECT match_id, league, date, home_team, away_team, fthg, ftag, hc, ac"
                " FROM raw_matches WHERE league = ?",
                [league],
            ).fetchdf()
        if league_df.empty:
            return pd.DataFrame(columns=["match_id"])
        league_df["home_team"] = league_df["home_team"].astype(str).map(standardize_team_name)
        league_df["away_team"] = league_df["away_team"].astype(str).map(standardize_team_name)

        # fthg/ftag/hc/ac deliberately omitted here (left for the reindex
        # below), not set to np.nan directly -- pandas' fetchdf() gives
        # league_df these columns as int32/float32 (not its own float64
        # default), and a plain dict literal's np.nan is always float64.
        # Concatenating a real int32 column against a same-named column
        # that's *entirely* NaN in this one-row frame hits pandas' own
        # deprecated "empty or all-NA" dtype-inference path (FutureWarning:
        # "DataFrame concatenation with empty or all-NA entries..."),
        # exactly the case pandas' own message recommends fixing by
        # "exclud[ing] the relevant entries before the concat operation" --
        # verified byte-for-byte identical output/dtypes to the previous
        # np.nan-in-the-literal approach, zero warnings.
        synthetic_row = pd.DataFrame([{
            "match_id": synthetic_id, "league": league, "date": pd.Timestamp(match_date),
            "home_team": home_team, "away_team": away_team,
        }])
        combined_league_df = pd.concat([league_df, synthetic_row], ignore_index=True)
        for col in ("fthg", "ftag", "hc", "ac"):
            target_dtype = league_df[col].dtype
            # An integer dtype (fthg/ftag) can never hold the synthetic
            # row's NaN -- upcast to float64, matching what the previous
            # np.nan-in-the-literal approach's own dtype inference already
            # landed on for these two columns (verified). hc/ac are already
            # float in league_df and stay at their own (float32) dtype.
            if pd.api.types.is_integer_dtype(target_dtype):
                target_dtype = "float64"
            combined_league_df[col] = combined_league_df[col].astype(target_dtype)
        result = self._compute_dixon_coles_features(combined_league_df)
        return result[result["match_id"] == synthetic_id]

    def _compute_key_starter_absence_features(self, raw_df: pd.DataFrame) -> pd.DataFrame:
        """US#175: query match_lineups + raw_player_match_stats and compute
        the key-attacker-absence flags.

        Returns empty DataFrame (with only match_id column) when either table is absent.
        """
        try:
            with self.db_manager.connection(read_only=True) as conn:
                lineups_df = conn.execute(
                    "SELECT fotmob_match_id, player_id, team_name, side, position_group"
                    " FROM match_lineups"
                ).fetchdf()
                player_df = conn.execute(
                    "SELECT match_id, player_id, team_name, minutes_played, xg, xa"
                    " FROM raw_player_match_stats"
                ).fetchdf()
        except duckdb.CatalogException:
            return pd.DataFrame(columns=["match_id"])
        from src.features.lineup_features import compute_key_starter_absence
        return compute_key_starter_absence(lineups_df, player_df, raw_df)

    def _compute_frds_features(self, raw_df: pd.DataFrame) -> pd.DataFrame:
        """Query match_lineups + raw_player_match_stats and compute FRDS features.

        Returns empty DataFrame (with only match_id column) when either table is absent.
        """
        try:
            with self.db_manager.connection(read_only=True) as conn:
                lineups_df = conn.execute(
                    "SELECT fotmob_match_id, player_id, team_name, side"
                    " FROM match_lineups"
                ).fetchdf()
                player_df = conn.execute(
                    "SELECT match_id, player_id, team_name, rating"
                    " FROM raw_player_match_stats"
                ).fetchdf()
        except duckdb.CatalogException:
            return pd.DataFrame(columns=["match_id"])
        from src.features.lineup_features import compute_frds
        return compute_frds(lineups_df, player_df, raw_df)

    def _compute_defensive_anchor_features(self, raw_df: pd.DataFrame) -> pd.DataFrame:
        """Query match_lineups + raw_player_match_stats and compute Defensive Anchor features.

        Returns empty DataFrame (with only match_id column) when tables or columns are absent.
        """
        try:
            with self.db_manager.connection(read_only=True) as conn:
                lineups_df = conn.execute(
                    "SELECT fotmob_match_id, player_id, team_name, side, position_group"
                    " FROM match_lineups"
                ).fetchdf()
                player_df = conn.execute(
                    "SELECT match_id, player_id, team_name, minutes_played,"
                    "       interceptions, recoveries"
                    " FROM raw_player_match_stats"
                ).fetchdf()
        except duckdb.CatalogException:
            return pd.DataFrame(columns=["match_id"])
        except duckdb.BinderException:
            # interceptions/recoveries columns not yet added to existing DB
            return pd.DataFrame(columns=["match_id"])
        from src.features.lineup_features import compute_defensive_anchor
        return compute_defensive_anchor(lineups_df, player_df, raw_df)
