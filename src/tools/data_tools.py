"""Data access tools for agent/MCP consumption (US#82)."""

from __future__ import annotations

from typing import Any

import pandas as pd

from src.utils.db_manager import DuckDBManager


def get_data_freshness() -> dict[str, Any]:
    """Return data freshness metadata from the raw_matches table.

    Returns:
        Dict with keys:
            latest_match_date: ISO date string of the most recent match (or None).
            days_since_update: Number of days since the latest match.
            match_count: Total number of rows in raw_matches.
            is_stale: True if latest_match_date is more than 7 days ago.
    """
    db = DuckDBManager()
    try:
        with db.connection(read_only=True) as conn:
            row = conn.execute("SELECT COUNT(*), MAX(date) FROM raw_matches").fetchone()
    except Exception:
        return {"latest_match_date": None, "days_since_update": None, "match_count": 0, "is_stale": True}

    match_count = int(row[0]) if row else 0
    max_date = row[1] if row else None
    if max_date is not None:
        latest_ts = pd.Timestamp(max_date).tz_localize(None)
        days_since = (pd.Timestamp.now().normalize() - latest_ts.normalize()).days
        latest_str = latest_ts.date().isoformat()
    else:
        days_since = None
        latest_str = None

    return {
        "latest_match_date": latest_str,
        "days_since_update": days_since,
        "match_count": match_count,
        "is_stale": (days_since is None or days_since > 7),
    }


def list_matches(
    league: str | None = None,
    from_date: str | None = None,
    to_date: str | None = None,
    limit: int | None = None,
) -> list[dict[str, Any]]:
    """List historical matches from the feature store (historical only — no upcoming matches).

    NOTE: This returns matches that are already in the feature store. Upcoming matches not yet
    played are not included. For forecasting upcoming matches use forecast_upcoming().

    Args:
        league: Optional league code filter (e.g. 'E0').
        from_date: Optional ISO date string lower bound (inclusive).
        to_date: Optional ISO date string upper bound (inclusive).
        limit: Optional maximum number of matches to return.

    Returns:
        List of dicts with keys: match_id, date, home_team, away_team, league.
    """
    db = DuckDBManager()
    filters: list[str] = []
    params: list[object] = []

    if league:
        filters.append("UPPER(r.league) = ?")
        params.append(league.upper())
    if from_date:
        filters.append("r.date >= ?")
        params.append(from_date)
    if to_date:
        filters.append("r.date <= ?")
        params.append(to_date)

    where = f"WHERE {' AND '.join(filters)}" if filters else ""
    limit_clause = f"LIMIT {int(limit)}" if limit is not None else ""

    query = f"""
        SELECT r.match_id, r.date, r.home_team, r.away_team, r.league
        FROM raw_matches r
        INNER JOIN feature_store f ON r.match_id = f.match_id
        {where}
        ORDER BY r.date DESC, r.match_id
        {limit_clause}
    """
    try:
        with db.connection(read_only=True) as conn:
            df = conn.execute(query, params).fetchdf()
    except Exception:
        return []

    if df.empty:
        return []

    df["date"] = pd.to_datetime(df["date"], errors="coerce").dt.date.astype(str)
    return df.to_dict(orient="records")
