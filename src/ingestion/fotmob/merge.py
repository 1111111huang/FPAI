"""Resolve FotMob player-match rows to raw_matches and persist them.

Joins by (date, home_team, away_team) rather than recomputing raw_matches'
match_id hash directly — this mirrors the existing, already-tested approach
in src/ingestion/understat/merge.py's update_raw_matches_xg, and avoids any
risk of a date/team-string formatting mismatch against the hash function.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import pandas as pd

from src.ingestion.common.team_mapping import TeamNameMapper
from src.utils.logger import get_logger

if TYPE_CHECKING:
    from src.utils.db_manager import DuckDBManager

LOGGER = get_logger(__name__)

_EMPTY_RESULT = {"matched": 0, "unmatched": 0, "players_upserted": 0, "rows_upserted": 0}

_STATS_COLUMNS = [
    "match_id", "player_id", "team_name", "minutes_played",
    "rating", "goals", "assists", "xg", "xa", "xgot", "shots",
]


def _create_tables(conn) -> None:
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS player_dim (
            player_id BIGINT PRIMARY KEY,
            player_name TEXT,
            opta_id TEXT
        )
        """
    )
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS raw_player_match_stats (
            match_id TEXT,
            player_id BIGINT,
            team_name TEXT,
            minutes_played INTEGER,
            rating FLOAT,
            goals INTEGER,
            assists INTEGER,
            xg FLOAT,
            xa FLOAT,
            xgot FLOAT,
            shots INTEGER,
            PRIMARY KEY (match_id, player_id)
        )
        """
    )


def resolve_match_ids(
    fotmob_df: pd.DataFrame,
    db_manager: "DuckDBManager",
    mapping_path: str = "config/team_mapping.json",
) -> pd.DataFrame:
    """Join FotMob rows to raw_matches by date+team to recover match_id.

    Args:
        fotmob_df: DataFrame with columns match_date, home_team, away_team
            (plus arbitrary per-player stat columns to carry through).
        db_manager: DuckDBManager instance for database access.
        mapping_path: Path to team_mapping.json for name normalisation.

    Returns:
        Copy of fotmob_df with a match_id column added (NaN where unmatched).
    """
    with db_manager.connection() as conn:
        raw = conn.execute("SELECT match_id, date, home_team, away_team FROM raw_matches").fetchdf()

    if raw.empty:
        return fotmob_df.assign(match_id=None)

    mapper = TeamNameMapper(mapping_path=mapping_path)
    team_pool = set(raw["home_team"]).union(set(raw["away_team"]))

    work = fotmob_df.copy()
    work["_home"] = work["home_team"].map(lambda name: mapper.map_team(name, team_pool))
    work["_away"] = work["away_team"].map(lambda name: mapper.map_team(name, team_pool))
    work["_date"] = pd.to_datetime(work["match_date"]).dt.normalize()

    raw = raw.rename(columns={"home_team": "_raw_home", "away_team": "_raw_away"})
    raw["_date"] = pd.to_datetime(raw["date"]).dt.normalize()

    merged = work.merge(
        raw[["match_id", "_date", "_raw_home", "_raw_away"]],
        left_on=["_date", "_home", "_away"],
        right_on=["_date", "_raw_home", "_raw_away"],
        how="left",
    )
    return merged


def upsert_player_match_stats(
    fotmob_df: pd.DataFrame,
    db_manager: "DuckDBManager",
    mapping_path: str = "config/team_mapping.json",
) -> dict[str, int]:
    """Resolve match_id and upsert player_dim + raw_player_match_stats rows.

    Args:
        fotmob_df: DataFrame of FotMob per-player, per-match rows (see
            src.ingestion.fotmob.fetcher.PLAYER_MATCH_COLUMNS for schema).
        db_manager: DuckDBManager instance for database access.
        mapping_path: Path to team_mapping.json for name normalisation.

    Returns:
        Dict with keys 'matched', 'unmatched', 'players_upserted', 'rows_upserted'.
    """
    if fotmob_df.empty:
        return dict(_EMPTY_RESULT)

    resolved = resolve_match_ids(fotmob_df, db_manager, mapping_path=mapping_path)
    has_match = resolved["match_id"].notna()
    unmatched = int((~has_match).sum())
    if unmatched:
        LOGGER.warning("%d FotMob player rows had no raw_matches match — skipped.", unmatched)

    matched = resolved.loc[has_match].copy()
    if matched.empty:
        with db_manager.connection() as conn:
            _create_tables(conn)
        return {**_EMPTY_RESULT, "unmatched": unmatched}

    players = matched[["player_id", "player_name", "opta_id"]].drop_duplicates(subset=["player_id"])
    stats = matched[_STATS_COLUMNS].drop_duplicates(subset=["match_id", "player_id"])

    with db_manager.connection() as conn:
        _create_tables(conn)

        conn.register("_players_upd", players)
        conn.execute(
            """
            INSERT INTO player_dim (player_id, player_name, opta_id)
            SELECT player_id, player_name, opta_id FROM _players_upd
            ON CONFLICT (player_id) DO UPDATE SET
                player_name = excluded.player_name,
                opta_id = excluded.opta_id
            """
        )
        conn.unregister("_players_upd")

        conn.register("_stats_upd", stats)
        conn.execute(
            f"""
            INSERT INTO raw_player_match_stats ({", ".join(_STATS_COLUMNS)})
            SELECT {", ".join(_STATS_COLUMNS)} FROM _stats_upd
            ON CONFLICT (match_id, player_id) DO UPDATE SET
                team_name = excluded.team_name,
                minutes_played = excluded.minutes_played,
                rating = excluded.rating,
                goals = excluded.goals,
                assists = excluded.assists,
                xg = excluded.xg,
                xa = excluded.xa,
                xgot = excluded.xgot,
                shots = excluded.shots
            """
        )
        conn.unregister("_stats_upd")

    LOGGER.info(
        "FotMob player stats upsert complete | matched=%d | unmatched=%d | players=%d | rows=%d",
        len(matched), unmatched, len(players), len(stats),
    )
    return {
        "matched": len(matched),
        "unmatched": unmatched,
        "players_upserted": len(players),
        "rows_upserted": len(stats),
    }
