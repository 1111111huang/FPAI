"""Helpers for integrating Understat xG data with CSV match records."""

from __future__ import annotations

from typing import TYPE_CHECKING, Iterable

import pandas as pd

from src.ingestion.common.team_mapping import TeamNameMapper
from src.utils.helpers import standardize_team_name
from src.utils.logger import get_logger

if TYPE_CHECKING:
    from src.utils.db_manager import DuckDBManager

LOGGER = get_logger(__name__)


def _resolve_column(df: pd.DataFrame, options: Iterable[str]) -> str | None:
    for option in options:
        if option in df.columns:
            return option
    return None


def _resolve_xg_columns(df: pd.DataFrame) -> tuple[str, str]:
    home_col = _resolve_column(
        df,
        [
            "xG_H",
            "xg_h",
            "xg_home",
            "xG_home",
            "home_xg",
            "h_xg",
        ],
    )
    away_col = _resolve_column(
        df,
        [
            "xG_A",
            "xg_a",
            "xg_away",
            "xG_away",
            "away_xg",
            "a_xg",
        ],
    )
    if home_col is None or away_col is None:
        raise ValueError("Understat dataframe must include home/away xG columns.")
    return home_col, away_col


def merge_understat_data(
    main_df: pd.DataFrame,
    understat_df: pd.DataFrame,
    mapping_path: str = "config/team_mapping.json",
) -> pd.DataFrame:
    """Merge Understat xG metrics into the main CSV dataframe."""
    if main_df.empty:
        return main_df.copy()

    main_date_col = _resolve_column(main_df, ["Date", "date"])
    main_home_col = _resolve_column(main_df, ["HomeTeam", "home_team"])
    main_away_col = _resolve_column(main_df, ["AwayTeam", "away_team"])
    if main_date_col is None or main_home_col is None or main_away_col is None:
        raise ValueError("main_df must include Date/HomeTeam/AwayTeam columns.")

    under_date_col = _resolve_column(understat_df, ["Date", "date"])
    under_home_col = _resolve_column(understat_df, ["HomeTeam", "home_team"])
    under_away_col = _resolve_column(understat_df, ["AwayTeam", "away_team"])
    if under_date_col is None or under_home_col is None or under_away_col is None:
        raise ValueError("understat_df must include Date/HomeTeam/AwayTeam columns.")

    xg_home_col, xg_away_col = _resolve_xg_columns(understat_df)

    main_working = main_df.copy()
    main_working["_match_date"] = pd.to_datetime(
        main_working[main_date_col], dayfirst=True, errors="coerce"
    ).dt.normalize()
    main_working["_home_team"] = (
        main_working[main_home_col].astype(str).map(standardize_team_name)
    )
    main_working["_away_team"] = (
        main_working[main_away_col].astype(str).map(standardize_team_name)
    )

    team_pool = set(main_working["_home_team"]).union(set(main_working["_away_team"]))
    mapper = TeamNameMapper(mapping_path=mapping_path)

    under_working = understat_df.copy()
    under_working["_match_date"] = pd.to_datetime(
        under_working[under_date_col], errors="coerce"
    ).dt.normalize()
    under_working["_home_team"] = under_working[under_home_col].astype(str).map(
        lambda name: mapper.map_team(name, team_pool)
    )
    under_working["_away_team"] = under_working[under_away_col].astype(str).map(
        lambda name: mapper.map_team(name, team_pool)
    )
    under_working["xg_h"] = pd.to_numeric(under_working[xg_home_col], errors="coerce")
    under_working["xg_a"] = pd.to_numeric(under_working[xg_away_col], errors="coerce")
    under_working["xga_h"] = under_working["xg_a"]
    under_working["xga_a"] = under_working["xg_h"]

    understat_core = under_working[["_match_date", "_home_team", "_away_team", "xg_h", "xg_a", "xga_h", "xga_a"]].drop_duplicates(
        subset=["_match_date", "_home_team", "_away_team"]
    )

    merged = main_working.merge(
        understat_core,
        on=["_match_date", "_home_team", "_away_team"],
        how="left",
    )

    merged["xG_H"] = merged["xg_h"]
    merged["xG_A"] = merged["xg_a"]
    merged["xGA_H"] = merged["xga_h"]
    merged["xGA_A"] = merged["xga_a"]

    xg_home_mean = understat_core["xg_h"].mean()
    xg_away_mean = understat_core["xg_a"].mean()

    missing_mask = merged["xG_H"].isna() | merged["xG_A"].isna()
    if missing_mask.any():
        missing_rows = merged.loc[missing_mask, [main_date_col, main_home_col, main_away_col]].head(5)
        LOGGER.warning(
            "Understat match missing for %s rows. Example rows: %s",
            int(missing_mask.sum()),
            missing_rows.to_dict(orient="records"),
        )
        merged.loc[missing_mask, "xG_H"] = xg_home_mean
        merged.loc[missing_mask, "xG_A"] = xg_away_mean
        merged.loc[missing_mask, "xGA_H"] = merged.loc[missing_mask, "xG_A"]
        merged.loc[missing_mask, "xGA_A"] = merged.loc[missing_mask, "xG_H"]

    fthg_col = _resolve_column(main_df, ["FTHG", "fthg"])
    ftag_col = _resolve_column(main_df, ["FTAG", "ftag"])
    if fthg_col is not None and ftag_col is not None:
        merged["Home_Luck"] = pd.to_numeric(merged[fthg_col], errors="coerce") - merged["xG_H"]
        merged["Away_Luck"] = pd.to_numeric(merged[ftag_col], errors="coerce") - merged["xG_A"]
    else:
        LOGGER.warning("FTHG/FTAG columns missing; luck features not computed.")

    return merged.drop(columns=["_match_date", "_home_team", "_away_team", "xg_h", "xg_a", "xga_h", "xga_a"], errors="ignore")


def update_raw_matches_xg(
    understat_df: pd.DataFrame,
    db_manager: "DuckDBManager",
    mapping_path: str = "config/team_mapping.json",
    league: str | None = None,
) -> dict[str, int]:
    """Match Understat xG rows to raw_matches by date+team and UPDATE the database.

    Args:
        understat_df: DataFrame with columns date, home_team, away_team, xg_h, xg_a.
        db_manager: DuckDBManager instance for database access.
        mapping_path: Path to team_mapping.json for name normalisation.
        league: If provided, scope both the raw_matches candidate rows and the
            fuzzy team-name match pool to this league only (US#141). Without
            this, once more than one competition's rows exist in raw_matches,
            the fuzzy-match candidate pool would silently mix every
            competition's team names together. Understat only covers EPL
            today (see LEAGUE_MAP in fetcher.py), so this has no practical
            effect yet, but closes the same structural gap resolve_match_ids
            (fotmob/merge.py) has, for whenever that changes. Omitted (None)
            preserves the pre-US#141 behavior (matches every row).

    Returns:
        Dict with keys 'matched', 'updated', 'unmatched'.
    """
    if understat_df.empty:
        return {"matched": 0, "updated": 0, "unmatched": 0}

    query = "SELECT match_id, date, home_team, away_team FROM raw_matches"
    params: list[str] = []
    if league is not None:
        query += " WHERE league = ?"
        params.append(league)
    with db_manager.connection() as conn:
        raw = conn.execute(query, params).fetchdf()

    if raw.empty:
        return {"matched": 0, "updated": 0, "unmatched": len(understat_df)}

    mapper = TeamNameMapper(mapping_path=mapping_path)
    team_pool = set(raw["home_team"]).union(set(raw["away_team"]))

    work = understat_df.copy()
    work["_home"] = work["home_team"].map(lambda n: mapper.map_team(n, team_pool))
    work["_away"] = work["away_team"].map(lambda n: mapper.map_team(n, team_pool))
    work["_date"] = pd.to_datetime(work["date"]).dt.normalize()

    raw["_date"] = pd.to_datetime(raw["date"]).dt.normalize()

    merged = raw.merge(
        work[["_date", "_home", "_away", "xg_h", "xg_a"]],
        left_on=["_date", "home_team", "away_team"],
        right_on=["_date", "_home", "_away"],
        how="left",
    )

    has_xg = merged["xg_h"].notna()
    unmatched = int((~has_xg).sum())
    if unmatched:
        LOGGER.warning(
            "%d raw_matches rows had no Understat match — xG left as NULL.", unmatched
        )

    to_update = merged.loc[has_xg, ["match_id", "xg_h", "xg_a"]].copy()
    to_update["xga_h"] = to_update["xg_a"]
    to_update["xga_a"] = to_update["xg_h"]

    if to_update.empty:
        return {"matched": 0, "updated": 0, "unmatched": unmatched}

    with db_manager.connection() as conn:
        conn.register("_xg_upd", to_update)
        conn.execute("""
            UPDATE raw_matches
            SET xg_h  = u.xg_h,
                xg_a  = u.xg_a,
                xga_h = u.xga_h,
                xga_a = u.xga_a
            FROM _xg_upd u
            WHERE raw_matches.match_id = u.match_id
        """)
        conn.unregister("_xg_upd")

    LOGGER.info(
        "xG update complete | updated=%d | unmatched=%d", len(to_update), unmatched
    )
    return {"matched": int(has_xg.sum()), "updated": len(to_update), "unmatched": unmatched}
