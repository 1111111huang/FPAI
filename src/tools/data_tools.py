"""Data access tools for agent/MCP consumption (US#82)."""

from __future__ import annotations

from typing import Any, Callable

import pandas as pd

from src.utils.db_manager import DuckDBManager


def _freshness_from_row(match_count: int, max_date: Any, now_fn: Callable[[], pd.Timestamp]) -> dict[str, Any]:
    if max_date is not None:
        latest_ts = pd.Timestamp(max_date).tz_localize(None)
        days_since = (now_fn().normalize() - latest_ts.normalize()).days
        latest_str = latest_ts.date().isoformat()
    else:
        days_since = None
        latest_str = None
    return {
        "latest_match_date": latest_str,
        "days_since_update": days_since,
        "match_count": int(match_count),
        "is_stale": (days_since is None or days_since > 7),
    }


def get_data_freshness(now_fn: Callable[[], pd.Timestamp] = pd.Timestamp.now) -> dict[str, Any]:
    """Return data freshness metadata from the raw_matches table.

    US#136: the top-level keys are a single number blended across every
    competition in raw_matches. That was accurate when only E0 existed, but
    once a second competition with a different season calendar and refresh
    cadence exists (e.g. Sweden's Allsvenskan, Mar-Nov + twice-weekly vs.
    EPL's Aug-May + weekly), a blended MAX(date) can mask one competition
    going stale behind the other staying fresh -- e.g. if EPL is deep in its
    off-season with no new matches for months but Sweden's weekly refresh
    keeps running, the blended `is_stale` reads "fresh" even though EPL's own
    data hasn't moved. The top-level keys are kept exactly as they were
    (byte-identical for a single-competition table, so no existing caller's
    behavior changes) and a new `by_league` breakdown is added alongside so a
    caller that cares about a specific competition's own freshness doesn't
    have to guess from the blended number.

    Args:
        now_fn: Returns the current time; injectable for testing the
            staleness boundary without monkeypatching pandas globally.
            Defaults to the real wall clock.

    Returns:
        Dict with keys:
            latest_match_date: ISO date string of the most recent match across
                every competition (or None).
            days_since_update: Number of days since that latest match.
            match_count: Total number of rows in raw_matches, every competition.
            is_stale: True if latest_match_date is more than 7 days ago.
            by_league: Dict keyed by league code, each value the same shape
                as the top level but scoped to just that competition's rows.
    """
    db = DuckDBManager()
    try:
        with db.connection(read_only=True) as conn:
            rows = conn.execute("SELECT league, COUNT(*), MAX(date) FROM raw_matches GROUP BY league").fetchall()
    except Exception:
        return {
            "latest_match_date": None, "days_since_update": None, "match_count": 0,
            "is_stale": True, "by_league": {},
        }

    if not rows:
        result = _freshness_from_row(0, None, now_fn)
        result["by_league"] = {}
        return result

    by_league = {league: _freshness_from_row(count, max_date, now_fn) for league, count, max_date in rows}
    total_count = sum(count for _, count, _ in rows)
    overall_max_date = max((max_date for _, _, max_date in rows if max_date is not None), default=None)

    result = _freshness_from_row(total_count, overall_max_date, now_fn)
    result["by_league"] = by_league
    return result


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
