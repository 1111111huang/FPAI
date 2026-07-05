"""Lineup-gated feature engineering utilities (US#103+).

Functions here require data from both ``match_lineups`` (FWD starters) and
``raw_player_match_stats`` (per-player xG/xA) which are only available for
FotMob-covered competitions.  Callers must handle the empty-DataFrame sentinel
returned when the underlying tables are absent.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
from src.utils.helpers import standardize_team_name
from src.utils.logger import get_logger

LOGGER = get_logger(__name__)

# ---------------------------------------------------------------------------
# Coefficient loader
# ---------------------------------------------------------------------------

def _load_coefficient(league_code: str) -> float:
    """Look up the UEFA/FIFA league coefficient from config/league_coefficients.yaml.

    Returns 1.0 when the file is absent or the league_code is not found.
    """
    path = Path("config/league_coefficients.yaml")
    if not path.exists():
        return 1.0
    try:
        import yaml  # optional — only needed at feature-compute time
        data = yaml.safe_load(path.read_text())
        return float(data.get("coefficients", {}).get(league_code, 1.0))
    except Exception:  # noqa: BLE001
        return 1.0


# ---------------------------------------------------------------------------
# xOC — Top-3 Offensive Concentration (US#103)
# ---------------------------------------------------------------------------

def compute_xoc(
    match_lineups_df: pd.DataFrame,
    player_stats_df: pd.DataFrame,
    raw_df: pd.DataFrame,
    league_code: str = "E0",
) -> pd.DataFrame:
    """Compute XOC_HOME and XOC_AWAY per match.

    xOC = sum of rolling (xG+xA)/90 for the top-3 FWD starters, normalised by
    the league's UEFA/FIFA coefficient.

    Parameters
    ----------
    match_lineups_df:
        Rows from ``match_lineups`` (fotmob_match_id, player_id, team_name,
        side, position_group).  team_name values are FotMob-abbreviated and
        will be standardised internally.
    player_stats_df:
        Rows from ``raw_player_match_stats`` (match_id, player_id, team_name,
        minutes_played, xg, xa).  match_id is the Football-Data hashed key.
    raw_df:
        Rows from ``raw_matches`` (match_id, date, home_team, away_team).
        team_name values must already be canonical (standardize_team_name).
    league_code:
        League code used to look up the normalisation coefficient, e.g. ``"E0"``.

    Returns
    -------
    pd.DataFrame
        One row per match with columns [match_id, XOC_HOME, XOC_AWAY].
        Returns an empty DataFrame (match_id column only) when inputs are empty
        or cannot be joined.
    """
    if match_lineups_df.empty or player_stats_df.empty or raw_df.empty:
        return pd.DataFrame(columns=["match_id"])

    coeff = _load_coefficient(league_code)

    # --- 1. Build FWD-only starters list from match_lineups ---
    lineups = match_lineups_df.copy()
    lineups["team_std"] = lineups["team_name"].map(standardize_team_name)
    fwd = lineups[lineups["position_group"] == "FWD"].copy()
    if fwd.empty:
        return pd.DataFrame(columns=["match_id"])

    # --- 2. Enrich player stats with match date (needed for rolling) ---
    stats = player_stats_df.copy()
    for col in ["xg", "xa", "minutes_played"]:
        stats[col] = pd.to_numeric(stats[col], errors="coerce").fillna(0.0)

    # Filter out zero-minute entries to avoid division noise
    stats = stats[stats["minutes_played"] > 0].copy()
    stats["xgxa_p90"] = (stats["xg"] + stats["xa"]) / stats["minutes_played"] * 90.0
    stats["team_std"] = stats["team_name"].map(standardize_team_name)

    match_info = raw_df[["match_id", "date", "home_team", "away_team"]].copy()
    match_info["date"] = pd.to_datetime(match_info["date"])

    # Attach date to player stats for chronological rolling
    stats = stats.merge(match_info[["match_id", "date"]], on="match_id", how="inner")
    stats = stats.sort_values(["player_id", "date", "match_id"]).reset_index(drop=True)

    # --- 3. Compute rolling R5 (xG+xA)/90 per player (shift=1 → pre-match safe) ---
    stats["_roll_xgxa_p90"] = stats.groupby("player_id")["xgxa_p90"].transform(
        lambda s: s.shift(1).rolling(5, min_periods=1).mean()
    )

    # --- 4. Resolve fotmob_match_id → match_id via date + normalised team name ---
    # match_lineups uses fotmob_match_id; we need the Football-Data match_id.
    # Strategy: join on date + (home or away) team name.
    raw_key = match_info.rename(columns={"home_team": "_home", "away_team": "_away"})

    # We'll build a lookup: (date, team_std) → match_id
    home_lkp = raw_key[["match_id", "date", "_home"]].rename(columns={"_home": "team_std"})
    away_lkp = raw_key[["match_id", "date", "_away"]].rename(columns={"_away": "team_std"})
    team_date_to_match = pd.concat([home_lkp, away_lkp], ignore_index=True).drop_duplicates(
        subset=["date", "team_std"]
    )

    # Attach date to lineups via fotmob_match_id → team_std day lookup
    # We don't have fotmob match dates directly, but we can recover them from
    # player_stats (which has match_id + date) joined via player_id overlap.
    # Faster path: join lineups → player_stats on player_id, then use match date.
    # Note: player_stats has match_id (Football-Data), so we pivot through that.

    # player_id appears in both tables — merge on player_id and filter same team
    fwd_with_stats = fwd.merge(
        stats[["match_id", "player_id", "team_std", "date", "_roll_xgxa_p90"]],
        on="player_id",
        how="inner",
        suffixes=("_lineup", "_stats"),
    )
    # Keep rows where team matches (standardised names should align after map)
    fwd_with_stats = fwd_with_stats[
        fwd_with_stats["team_std_lineup"] == fwd_with_stats["team_std_stats"]
    ].copy()
    fwd_with_stats["team_std"] = fwd_with_stats["team_std_lineup"]

    if fwd_with_stats.empty:
        return pd.DataFrame(columns=["match_id"])

    # --- 5. For each (match_id, team_std, side), pick top-3 FWDs by rolling xgxa_p90 ---
    # We know which match_id each player appeared in (from raw_player_match_stats).
    # We also know the side from match_lineups.  But fotmob_match_id ≠ match_id,
    # so we must use the player's stat match_id (Football-Data) as the key,
    # and confirm the date matches between lineups and stats.
    #
    # To associate each lineup appearance with the correct match_id:
    # For each player-lineup row, we use the player_stats match_id where the
    # dates align within the same season. Since we joined on player_id above,
    # match_id now refers to the Football-Data row.

    # Identify match_id for each lineup player
    # The inner join already brought in match_id from raw_player_match_stats.
    # We need to determine which match in raw_matches corresponds to this player's
    # lineup entry. We join by fotmob team name and date from the stats side.

    # Group by match_id + team_std + side → top-3 FWDs
    def _top3_sum(group: pd.DataFrame) -> float:
        top3 = group["_roll_xgxa_p90"].nlargest(3)
        return top3.sum() if not top3.empty else 0.0

    team_side_agg = (
        fwd_with_stats
        .groupby(["match_id", "team_std", "side"])
        .apply(_top3_sum, include_groups=False)
        .reset_index(name="_raw_xoc")
    )

    team_side_agg["_xoc"] = team_side_agg["_raw_xoc"] / coeff

    # --- 6. Join to match_info to produce XOC_HOME / XOC_AWAY ---
    result = match_info[["match_id", "home_team", "away_team"]].copy()

    home_xoc = (
        result.merge(
            team_side_agg[["match_id", "team_std", "side", "_xoc"]],
            on="match_id",
            how="left",
        )
    )
    # Filter rows where team_std matches home_team and side is "home"
    home_xoc = home_xoc[
        (home_xoc["team_std"] == home_xoc["home_team"]) & (home_xoc["side"] == "home")
    ][["match_id", "_xoc"]].rename(columns={"_xoc": "XOC_HOME"})

    away_xoc = (
        result.merge(
            team_side_agg[["match_id", "team_std", "side", "_xoc"]],
            on="match_id",
            how="left",
        )
    )
    away_xoc = away_xoc[
        (away_xoc["team_std"] == away_xoc["away_team"]) & (away_xoc["side"] == "away")
    ][["match_id", "_xoc"]].rename(columns={"_xoc": "XOC_AWAY"})

    result = result.merge(home_xoc, on="match_id", how="left")
    result = result.merge(away_xoc, on="match_id", how="left")

    return result[["match_id", "XOC_HOME", "XOC_AWAY"]]


# ---------------------------------------------------------------------------
# FRDS — FotMob Rating Dominance Share (US#102)
# ---------------------------------------------------------------------------

SQUAD_POOL_DAYS: int = 90


def _resolve_fotmob_to_raw(
    lineups: pd.DataFrame,
    stats: pd.DataFrame,
    match_dates: pd.DataFrame,
) -> pd.DataFrame:
    """Resolve fotmob_match_id → raw match_id via player co-occurrence voting.

    For each (fotmob_match_id, team_std), find the raw match_id where the most
    starters from that lineup appear together.  Ties broken by latest match date.

    Args:
        lineups: columns fotmob_match_id (str), player_id, team_std.
        stats: columns match_id, player_id, team_std.
        match_dates: columns match_id, date (for tie-breaking).

    Returns:
        DataFrame with columns [fotmob_match_id, team_std, match_id].
    """
    joined = lineups[["fotmob_match_id", "player_id", "team_std"]].merge(
        stats[["match_id", "player_id", "team_std"]],
        on=["player_id", "team_std"],
        how="inner",
    )
    if joined.empty:
        return pd.DataFrame(columns=["fotmob_match_id", "team_std", "match_id"])

    vote = (
        joined.groupby(["fotmob_match_id", "team_std", "match_id"])["player_id"]
        .count()
        .reset_index()
        .rename(columns={"player_id": "_votes"})
    )
    vote = vote.merge(match_dates[["match_id", "date"]], on="match_id", how="left")
    best = (
        vote.sort_values(["_votes", "date"], ascending=[False, False])
        .drop_duplicates(subset=["fotmob_match_id", "team_std"], keep="first")
        [["fotmob_match_id", "team_std", "match_id"]]
        .reset_index(drop=True)
    )
    return best


def compute_frds(
    match_lineups_df: pd.DataFrame,
    player_stats_df: pd.DataFrame,
    raw_df: pd.DataFrame,
) -> pd.DataFrame:
    """Compute FRDS_HOME and FRDS_AWAY per match.

    FRDS = sum(rolling_avg_rating of starting 11)
           / sum(rolling_avg_rating of all players who appeared for this team
                 in the last SQUAD_POOL_DAYS days before the match).
    Clamped to [0, 1].
    """
    if match_lineups_df.empty or player_stats_df.empty or raw_df.empty:
        return pd.DataFrame(columns=["match_id"])

    match_info = raw_df[["match_id", "date", "home_team", "away_team"]].copy()
    match_info["date"] = pd.to_datetime(match_info["date"], errors="coerce")
    match_info = match_info.dropna(subset=["date"]).reset_index(drop=True)
    match_info["home_std"] = match_info["home_team"].map(standardize_team_name)
    match_info["away_std"] = match_info["away_team"].map(standardize_team_name)

    lineups = match_lineups_df.copy()
    lineups["team_std"] = lineups["team_name"].map(standardize_team_name)
    lineups["fotmob_match_id"] = lineups["fotmob_match_id"].astype(str)

    stats = player_stats_df.copy()
    stats["team_std"] = stats["team_name"].map(standardize_team_name)

    fotmob_to_raw = _resolve_fotmob_to_raw(lineups, stats, match_info[["match_id", "date"]])
    if fotmob_to_raw.empty:
        return pd.DataFrame(columns=["match_id"])

    stats_with_date = stats.merge(match_info[["match_id", "date"]], on="match_id", how="inner")
    stats_with_date = stats_with_date.sort_values(
        ["player_id", "team_std", "date", "match_id"]
    ).reset_index(drop=True)
    stats_with_date["rolling_avg_rating"] = stats_with_date.groupby(
        ["player_id", "team_std"]
    )["rating"].transform(lambda s: s.shift(1).rolling(5, min_periods=1).mean())
    stats_with_date["rolling_avg_rating"] = stats_with_date["rolling_avg_rating"].fillna(
        stats_with_date["rating"]
    )

    lineups_resolved = lineups.merge(
        fotmob_to_raw, on=["fotmob_match_id", "team_std"], how="left"
    ).dropna(subset=["match_id"])
    if lineups_resolved.empty:
        return pd.DataFrame(columns=["match_id"])

    lineups_resolved["match_id"] = lineups_resolved["match_id"].astype(str)
    lineups_resolved = lineups_resolved.merge(
        match_info[["match_id", "date"]], on="match_id", how="left"
    )

    starter_ratings = lineups_resolved[
        ["match_id", "fotmob_match_id", "player_id", "team_std", "side", "date"]
    ].merge(
        stats_with_date[["player_id", "team_std", "match_id", "rolling_avg_rating"]],
        on=["player_id", "team_std", "match_id"],
        how="left",
    )
    starting_sum = (
        starter_ratings.groupby(["match_id", "team_std"])["rolling_avg_rating"]
        .sum()
        .reset_index()
        .rename(columns={"rolling_avg_rating": "starting_11_rating_sum"})
    )

    frds_match_teams = lineups_resolved[["match_id", "team_std", "date"]].drop_duplicates().reset_index(drop=True)
    stats_lookup = stats_with_date[["player_id", "team_std", "date", "match_id", "rolling_avg_rating"]].copy()

    pool_records = []
    for _, focal in frds_match_teams.iterrows():
        cutoff = focal["date"] - pd.Timedelta(days=SQUAD_POOL_DAYS)
        pool = stats_lookup[
            (stats_lookup["team_std"] == focal["team_std"])
            & (stats_lookup["date"] >= cutoff)
            & (stats_lookup["date"] < focal["date"])
        ]
        if pool.empty:
            pool_records.append({
                "match_id": focal["match_id"],
                "team_std": focal["team_std"],
                "squad_pool_rating_sum": float("nan"),
            })
        else:
            latest = pool.sort_values("date").groupby("player_id")["rolling_avg_rating"].last()
            pool_records.append({
                "match_id": focal["match_id"],
                "team_std": focal["team_std"],
                "squad_pool_rating_sum": latest.sum(),
            })

    if not pool_records:
        return pd.DataFrame(columns=["match_id"])

    squad_pool = pd.DataFrame(pool_records)
    frds_by_team = starting_sum.merge(squad_pool, on=["match_id", "team_std"], how="left")
    frds_by_team["frds"] = (
        frds_by_team["starting_11_rating_sum"] / frds_by_team["squad_pool_rating_sum"]
    ).clip(0.0, 1.0)

    result = match_info[["match_id", "home_std", "away_std"]].copy()
    result = result.merge(
        frds_by_team[["match_id", "team_std", "frds"]].rename(
            columns={"team_std": "_ts", "frds": "FRDS_HOME"}
        ),
        left_on=["match_id", "home_std"],
        right_on=["match_id", "_ts"],
        how="left",
    ).drop(columns=["_ts"], errors="ignore")
    result = result.merge(
        frds_by_team[["match_id", "team_std", "frds"]].rename(
            columns={"team_std": "_ts", "frds": "FRDS_AWAY"}
        ),
        left_on=["match_id", "away_std"],
        right_on=["match_id", "_ts"],
        how="left",
    ).drop(columns=["_ts"], errors="ignore")

    n_populated = result[["FRDS_HOME", "FRDS_AWAY"]].notna().any(axis=1).sum()
    LOGGER.info("compute_frds complete | matches=%d | with_frds=%d", len(result), n_populated)
    return result[["match_id", "FRDS_HOME", "FRDS_AWAY"]]


# ---------------------------------------------------------------------------
# Defensive Anchor (US#104)
# ---------------------------------------------------------------------------

def compute_defensive_anchor(
    match_lineups_df: pd.DataFrame,
    player_stats_df: pd.DataFrame,
    raw_df: pd.DataFrame,
) -> pd.DataFrame:
    """Compute DEF_ANCHOR_HOME and DEF_ANCHOR_AWAY per match.

    Finds top-2 DEF/MID starters by rolling (interceptions + recoveries)/90
    over last R5 appearances and returns their mean as the anchor signal.
    """
    if match_lineups_df.empty or player_stats_df.empty:
        return pd.DataFrame(columns=["match_id"])

    lineups = match_lineups_df.copy()
    lineups["fotmob_match_id"] = pd.to_numeric(lineups["fotmob_match_id"], errors="coerce")
    lineups["player_id"] = pd.to_numeric(lineups["player_id"], errors="coerce")

    stats = player_stats_df.copy()
    stats["player_id"] = pd.to_numeric(stats["player_id"], errors="coerce")
    stats["minutes_played"] = pd.to_numeric(stats["minutes_played"], errors="coerce")
    stats["interceptions"] = pd.to_numeric(
        stats.get("interceptions", pd.Series(0, index=stats.index)), errors="coerce"
    ).fillna(0.0)
    stats["recoveries"] = pd.to_numeric(
        stats.get("recoveries", pd.Series(0, index=stats.index)), errors="coerce"
    ).fillna(0.0)

    match_info = raw_df[["match_id", "date", "home_team", "away_team"]].copy()
    match_info["date"] = pd.to_datetime(match_info["date"])

    stats_dated = stats.merge(match_info[["match_id", "date"]], on="match_id", how="inner")
    stats_dated["def_rec"] = stats_dated["interceptions"] + stats_dated["recoveries"]
    stats_dated["def_rec_p90"] = stats_dated.apply(
        lambda r: r["def_rec"] / r["minutes_played"] * 90.0
        if (r["minutes_played"] is not None and r["minutes_played"] > 0)
        else float("nan"),
        axis=1,
    )
    stats_dated = stats_dated.sort_values(["player_id", "date", "match_id"]).reset_index(drop=True)
    stats_dated["def_anchor_roll"] = stats_dated.groupby("player_id")["def_rec_p90"].transform(
        lambda s: s.shift(1).rolling(5, min_periods=1).mean()
    )

    starters = lineups[lineups["position_group"].isin(["DEF", "MID"])].copy()
    if starters.empty:
        return pd.DataFrame(columns=["match_id"])

    starter_cols = starters[["fotmob_match_id", "player_id", "side"]].drop_duplicates()
    player_match_map = stats_dated[["player_id", "match_id", "date"]].drop_duplicates()
    bridge = starter_cols.merge(player_match_map, on="player_id", how="inner")
    if bridge.empty:
        return pd.DataFrame(columns=["match_id"])

    cooccur = (
        bridge.groupby(["fotmob_match_id", "match_id"])
        .agg(player_count=("player_id", "count"), max_date=("date", "max"))
        .reset_index()
    )
    cooccur = cooccur.sort_values(
        ["fotmob_match_id", "player_count", "max_date"], ascending=[True, False, False]
    )
    fotmob_to_raw = cooccur.drop_duplicates(subset=["fotmob_match_id"], keep="first")[
        ["fotmob_match_id", "match_id"]
    ]

    starters_with_match = starters.merge(fotmob_to_raw, on="fotmob_match_id", how="inner")
    anchor_data = starters_with_match.merge(
        stats_dated[["match_id", "player_id", "def_anchor_roll"]],
        on=["match_id", "player_id"],
        how="inner",
    )
    if anchor_data.empty:
        return pd.DataFrame(columns=["match_id"])

    def _top2_mean(group: pd.DataFrame) -> float:
        vals = group["def_anchor_roll"].dropna().sort_values(ascending=False)
        top2 = vals.iloc[:2]
        return float(top2.mean()) if not top2.empty else float("nan")

    anchor_by_side = (
        anchor_data.groupby(["match_id", "side"])
        .apply(_top2_mean, include_groups=False)
        .reset_index(name="_anchor")
    )

    home_anchor = anchor_by_side[anchor_by_side["side"] == "home"][
        ["match_id", "_anchor"]
    ].rename(columns={"_anchor": "DEF_ANCHOR_HOME"})
    away_anchor = anchor_by_side[anchor_by_side["side"] == "away"][
        ["match_id", "_anchor"]
    ].rename(columns={"_anchor": "DEF_ANCHOR_AWAY"})

    return (
        match_info[["match_id"]]
        .merge(home_anchor, on="match_id", how="left")
        .merge(away_anchor, on="match_id", how="left")
    )
