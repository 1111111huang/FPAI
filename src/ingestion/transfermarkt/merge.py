"""Persists Transfermarkt player market values into a dated-snapshot table.

Append-only by design (design spec's "Storage schema" section): each refresh
run inserts a new row set stamped with that run's snapshot_date, never
overwriting an earlier one. Costs nothing for the agent-tool consumer (Phase
1 only ever wants "most recent snapshot <= today"), but is exactly what a
future point-in-time backtest needs -- a historical match must read the
value that existed *at* that match's date, never a later one (the same
lookahead-bias class of bug W179 already had to fix for closing-line odds).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import pandas as pd

from src.utils.logger import get_logger

if TYPE_CHECKING:
    from src.utils.db_manager import DuckDBManager

LOGGER = get_logger(__name__)

_VALUE_COLUMNS = ["fotmob_player_id", "transfermarkt_player_id", "market_value_eur"]


def _create_tables(conn) -> None:
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS player_market_values (
            fotmob_player_id BIGINT,
            transfermarkt_player_id BIGINT,
            snapshot_date TEXT,
            market_value_eur BIGINT
        )
        """
    )


def insert_market_value_snapshot(values_df: pd.DataFrame, db_manager: "DuckDBManager", snapshot_date: str) -> int:
    """Inserts one dated snapshot's worth of market values. Returns the row
    count inserted. Deliberately plain INSERT ... SELECT, no ON CONFLICT
    clause -- this table is append-only, never upserted (see module
    docstring)."""
    if values_df.empty:
        with db_manager.connection() as conn:
            _create_tables(conn)
        return 0

    to_insert = values_df[_VALUE_COLUMNS].copy()
    to_insert["snapshot_date"] = snapshot_date

    with db_manager.connection() as conn:
        _create_tables(conn)
        conn.register("_values_upd", to_insert)
        conn.execute(
            """
            INSERT INTO player_market_values (fotmob_player_id, transfermarkt_player_id, snapshot_date, market_value_eur)
            SELECT fotmob_player_id, transfermarkt_player_id, snapshot_date, market_value_eur
            FROM _values_upd
            """
        )
        conn.unregister("_values_upd")

    LOGGER.info("Market value snapshot inserted | snapshot_date=%s | rows=%d", snapshot_date, len(to_insert))
    return len(to_insert)
