"""Fetch and store pre-match / post-match starting XI from FotMob's matchDetails API.

Stores one row per starter (home + away combined) in the `match_lineups` DuckDB
table. Substitutes are intentionally excluded: the `subs[]` array in FotMob's
lineup payload omits the `positionId` field, so position-group assignment is
unreliable.

Endpoints
---------
Same ``matchDetails`` endpoint already used by src.ingestion.fotmob.fetcher:
  https://www.fotmob.com/api/data/matchDetails?matchId={id}

positionId ranges (observed in live data):
  11          -> GK
  30-39       -> DEF
  60-69       -> MID
  80-89, 110+ -> FWD (ATT / WNG / ST)
"""

from __future__ import annotations

import time
from typing import TYPE_CHECKING

import requests

from src.ingestion.fotmob.fetcher import _HEADERS, _MATCH_DETAILS_URL
from src.utils.logger import get_logger

if TYPE_CHECKING:
    from src.utils.db_manager import DuckDBManager

LOGGER = get_logger(__name__)

_LINEUP_COLUMNS = [
    "fotmob_match_id",
    "player_id",
    "team_name",
    "side",
    "position_group",
    "position_id",
    "shirt_number",
    "player_name",
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _position_group(position_id: int | None) -> str:
    """Map a FotMob positionId to a coarse position group string."""
    if position_id is None:
        return "UNK"
    if position_id == 11:
        return "GK"
    if 30 <= position_id <= 39:
        return "DEF"
    if 60 <= position_id <= 69:
        return "MID"
    if 80 <= position_id <= 89 or position_id >= 110:
        return "FWD"
    return "UNK"


def _create_lineup_table(conn) -> None:
    """Create ``match_lineups`` table if it does not already exist."""
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS match_lineups (
            fotmob_match_id  BIGINT  NOT NULL,
            player_id        BIGINT  NOT NULL,
            team_name        TEXT    NOT NULL,
            side             TEXT    NOT NULL,
            position_group   TEXT    NOT NULL,
            position_id      INTEGER,
            shirt_number     TEXT,
            player_name      TEXT,
            PRIMARY KEY (fotmob_match_id, player_id)
        )
        """
    )


# ---------------------------------------------------------------------------
# Fetch
# ---------------------------------------------------------------------------

def fetch_match_lineup(fotmob_match_id: int, delay: float = 1.0) -> list[dict]:
    """Fetch starting XI rows for one match from FotMob.

    Parameters
    ----------
    fotmob_match_id:
        FotMob's internal integer match identifier.
    delay:
        Seconds to sleep after the HTTP request (polite crawl delay).

    Returns
    -------
    list[dict]
        One dict per starter (both teams combined), with keys matching
        ``_LINEUP_COLUMNS``.  Returns an empty list when the lineup block
        is absent from the payload.
    """
    url = _MATCH_DETAILS_URL.format(match_id=fotmob_match_id)
    LOGGER.info("Fetching FotMob lineup | match_id=%s -> %s", fotmob_match_id, url)
    try:
        resp = requests.get(url, headers=_HEADERS, timeout=30)
        resp.raise_for_status()
    except requests.RequestException as exc:
        LOGGER.error("HTTP error fetching lineup for match_id=%s: %s", fotmob_match_id, exc)
        time.sleep(delay)
        return []
    time.sleep(delay)

    payload = resp.json()
    lineup = payload.get("content", {}).get("lineup")
    if not lineup:
        LOGGER.debug("No lineup block found | match_id=%s", fotmob_match_id)
        return []

    lineup_type = lineup.get("lineupType")
    if lineup_type is None:
        LOGGER.debug("lineupType is None -- skipping | match_id=%s", fotmob_match_id)
        return []

    rows: list[dict] = []
    for side, team_key in (("home", "homeTeam"), ("away", "awayTeam")):
        team = lineup.get(team_key, {})
        team_name: str = team.get("name", "")
        for player in team.get("starters", []):
            try:
                pid = int(player["id"])
            except (KeyError, TypeError, ValueError) as exc:
                LOGGER.warning(
                    "Skipping malformed starter entry match_id=%s side=%s: %s",
                    fotmob_match_id, side, exc,
                )
                continue

            pos_id_raw = player.get("positionId")
            try:
                pos_id: int | None = int(pos_id_raw) if pos_id_raw is not None else None
            except (TypeError, ValueError):
                pos_id = None

            rows.append(
                {
                    "fotmob_match_id": fotmob_match_id,
                    "player_id": pid,
                    "team_name": team_name,
                    "side": side,
                    "position_group": _position_group(pos_id),
                    "position_id": pos_id,
                    "shirt_number": str(player.get("shirtNumber", "") or ""),
                    "player_name": player.get("name"),
                }
            )

    LOGGER.info("Got %d starter rows | match_id=%s", len(rows), fotmob_match_id)
    return rows


# ---------------------------------------------------------------------------
# Upsert
# ---------------------------------------------------------------------------

def upsert_match_lineups(
    fotmob_match_ids: list[int],
    db_manager: "DuckDBManager",
    delay: float = 1.0,
) -> int:
    """Fetch and upsert lineup rows for a list of FotMob match IDs.

    Parameters
    ----------
    fotmob_match_ids:
        FotMob integer match identifiers to process.
    db_manager:
        DuckDBManager instance for database access.
    delay:
        Polite delay in seconds passed to ``fetch_match_lineup``.

    Returns
    -------
    int
        Total rows upserted across all matches.
    """
    if not fotmob_match_ids:
        LOGGER.info("upsert_match_lineups: no match IDs provided -- nothing to do.")
        return 0

    all_rows: list[dict] = []
    for mid in fotmob_match_ids:
        rows = fetch_match_lineup(mid, delay=delay)
        all_rows.extend(rows)

    if not all_rows:
        LOGGER.info("upsert_match_lineups: no lineup rows fetched -- nothing to upsert.")
        with db_manager.connection() as conn:
            _create_lineup_table(conn)
        return 0

    import pandas as pd

    df = pd.DataFrame(all_rows, columns=_LINEUP_COLUMNS)

    with db_manager.connection() as conn:
        _create_lineup_table(conn)
        conn.register("_lineups_upd", df)
        conn.execute(
            f"""
            INSERT INTO match_lineups ({", ".join(_LINEUP_COLUMNS)})
            SELECT {", ".join(_LINEUP_COLUMNS)} FROM _lineups_upd
            ON CONFLICT (fotmob_match_id, player_id) DO UPDATE SET
                team_name      = excluded.team_name,
                side           = excluded.side,
                position_group = excluded.position_group,
                position_id    = excluded.position_id,
                shirt_number   = excluded.shirt_number,
                player_name    = excluded.player_name
            """
        )
        conn.unregister("_lineups_upd")

    total = len(df)
    LOGGER.info("match_lineups upsert complete | rows=%d", total)
    return total


# ---------------------------------------------------------------------------
# Backfill helper
# ---------------------------------------------------------------------------

def backfill_lineups_from_player_stats(db_manager: "DuckDBManager", delay: float = 1.0) -> int:
    """Backfill ``match_lineups`` using date bounds inferred from ``raw_matches``.

    The ``raw_player_match_stats`` table stores an internal hashed ``match_id``
    (not a FotMob ID), so we cannot resolve FotMob IDs from it directly.
    Instead we derive the full date range covered by ``raw_matches`` and then
    use ``fetch_finished_match_ids`` -- the same mechanism as the forward-going
    ``fetch-lineups`` command -- to discover FotMob match IDs.

    This will re-issue HTTP requests for match-list pages but avoids storing
    duplicate state.  For a targeted historical backfill, prefer the
    ``fetch-lineups`` CLI with explicit ``--date-from`` / ``--date-to`` flags.

    Returns
    -------
    int
        Total rows upserted, or 0 if raw_matches is empty.
    """
    from datetime import date as _date, timedelta

    from src.ingestion.fotmob.fetcher import fetch_finished_match_ids, LEAGUE_IDS

    with db_manager.connection(read_only=True) as conn:
        bounds = conn.execute(
            "SELECT CAST(MIN(date) AS DATE), CAST(MAX(date) AS DATE) FROM raw_matches"
        ).fetchone()

    if bounds is None or bounds[0] is None:
        LOGGER.warning(
            "backfill_lineups_from_player_stats: raw_matches is empty -- nothing to backfill."
        )
        return 0

    date_from: _date = bounds[0]
    date_to: _date = bounds[1]
    LOGGER.info(
        "backfill_lineups_from_player_stats: scanning %s .. %s for FotMob match IDs",
        date_from,
        date_to,
    )

    league_id = LEAGUE_IDS["E0"]
    fotmob_ids: list[int] = []
    current = date_from
    while current <= date_to:
        try:
            matches = fetch_finished_match_ids(current, league_id=league_id, delay=delay)
            fotmob_ids.extend(m["fotmob_match_id"] for m in matches)
        except Exception as exc:  # noqa: BLE001
            LOGGER.warning("backfill: failed to fetch match list for %s: %s", current, exc)
        current += timedelta(days=1)

    LOGGER.info(
        "backfill_lineups_from_player_stats: found %d FotMob match IDs", len(fotmob_ids)
    )
    return upsert_match_lineups(fotmob_ids, db_manager, delay=delay)
