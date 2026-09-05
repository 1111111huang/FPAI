"""Fetch per-match player stats from FotMob's internal JSON API.

Endpoints verified live (2026-06-27): plain HTTP JSON, HTTP 200, no auth and
no anti-bot challenge with only a browser-like User-Agent header. This is an
undocumented internal API (not an official product), same access-method
caveat as scraping any other football stats site.
"""

from __future__ import annotations

from datetime import date, timedelta
import time

import pandas as pd
import requests

from src.utils.logger import get_logger

LOGGER = get_logger(__name__)

LEAGUE_IDS: dict[str, int] = {
    "E0": 47,  # Premier League
    "SP1": 87,  # LaLiga (US#146, live-verified 2026-08-06 against /api/data/matches)
    # US#165, live-verified 2026-08-15 against /api/data/matches across
    # several real matchday dates (each name/ccode confirmed, not guessed
    # from the id-pattern alone -- e.g. Germany's men's top-flight
    # "Bundesliga"/id 54 had to be distinguished from the same endpoint's
    # "2. Bundesliga"/id 146 and Austria's own "Bundesliga"/id 938366).
    "I1": 55,  # Serie A (ccode ITA)
    "D1": 54,  # Bundesliga (ccode GER)
    "F1": 53,  # Ligue 1 (ccode FRA)
}

_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0 Safari/537.36"
    ),
    "Accept": "application/json",
}

_MATCHES_URL = "https://www.fotmob.com/api/data/matches?date={date}"
_MATCH_DETAILS_URL = "https://www.fotmob.com/api/data/matchDetails?matchId={match_id}"

# Maps our column name to FotMob's human-readable "top_stats" label.
_TOP_STAT_FIELDS: dict[str, str] = {
    "rating": "FotMob rating",
    "minutes_played": "Minutes played",
    "goals": "Goals",
    "assists": "Assists",
    "xg": "Expected goals (xG)",
    "xa": "Expected assists (xA)",
    "xgot": "Expected goals on target (xGOT)",
    "shots": "Total shots",
}

PLAYER_MATCH_COLUMNS: list[str] = [
    "fotmob_match_id", "match_date", "home_team", "away_team",
    "player_id", "player_name", "opta_id", "team_name",
    "rating", "minutes_played", "goals", "assists", "xg", "xa", "xgot", "shots",
    "interceptions", "recoveries",
]


def _date_range(date_from: date, date_to: date) -> list[date]:
    days = (date_to - date_from).days
    return [date_from + timedelta(days=offset) for offset in range(days + 1)]


def _parse_finished_matches(matches_payload: list[dict]) -> list[dict]:
    """Shared parsing for one league's raw 'matches' list -> our own finished-
    match dicts (fotmob_match_id, match_date, home_team, away_team). Used by
    both fetch_finished_match_ids (single league) and fetch_matches_for_leagues
    (US#190, several leagues from one shared request) so the two never drift."""
    matches: list[dict] = []
    for match in matches_payload:
        status = match.get("status", {})
        if not status.get("finished"):
            continue
        utc_time = status.get("utcTime")
        if not utc_time:
            continue
        try:
            matches.append(
                {
                    "fotmob_match_id": match["id"],
                    "match_date": pd.to_datetime(utc_time).tz_localize(None).normalize(),
                    "home_team": match["home"]["name"],
                    "away_team": match["away"]["name"],
                }
            )
        except (KeyError, TypeError, ValueError) as exc:
            LOGGER.warning("Skipping malformed match entry id=%s: %s", match.get("id"), exc)
    return matches


def _fetch_matches_payload(day: date, delay: float) -> dict | None:
    """One HTTP request for one date's /api/data/matches payload. Returns
    None (not a dict) for the BUG-041 malformed-payload case; caller decides
    what "no matches" looks like for its own return shape."""
    url = _MATCHES_URL.format(date=day.strftime("%Y%m%d"))
    resp = requests.get(url, headers=_HEADERS, timeout=30)
    resp.raise_for_status()
    time.sleep(delay)
    payload = resp.json()
    if not isinstance(payload, dict):
        # BUG-041: found live -- a 200 OK response with a non-dict body
        # (observed: literal `null`, for a date FotMob's matches endpoint
        # doesn't have data for, e.g. one far enough in the past) crashed
        # with an uncaught AttributeError on the .get() calls below. The
        # caller's own per-day try/except (fetch_player_match_stats) only
        # catches requests.RequestException, not this -- so this must
        # degrade gracefully itself, the same way a malformed individual
        # match entry already does, not raise and take out the whole
        # multi-day/multi-season loop over one date with no data.
        LOGGER.warning(
            "FotMob matches endpoint returned an unexpected payload shape (%s) for date=%s -- treating as no matches.",
            type(payload).__name__, day,
        )
        return None
    return payload


def fetch_finished_match_ids(day: date, league_id: int, delay: float = 1.0) -> list[dict]:
    """Return finished matches for one league on one date.

    Each dict has keys: fotmob_match_id, match_date, home_team, away_team.
    """
    LOGGER.info("Fetching FotMob matches | league_id=%s date=%s", league_id, day)
    payload = _fetch_matches_payload(day, delay)
    if payload is None:
        return []
    leagues = [entry for entry in payload.get("leagues", []) if entry.get("id") == league_id]

    matches: list[dict] = []
    for league in leagues:
        matches.extend(_parse_finished_matches(league.get("matches", [])))

    LOGGER.info("Got %d finished matches | league_id=%s date=%s", len(matches), league_id, day)
    return matches


def fetch_matches_for_leagues(day: date, league_ids: dict[str, int], delay: float = 1.0) -> dict[str, list[dict]]:
    """Like fetch_finished_match_ids, but for several leagues at once via a
    single HTTP request (US#190) -- /api/data/matches?date=... already
    returns every league's fixtures for that date in one payload, so
    backfilling N leagues by calling fetch_finished_match_ids once per
    league per day (the original pattern) made N redundant identical
    requests for the same date. Found live: day-level requests were
    roughly half of a real multi-league backfill's total request volume --
    entirely eliminable since they don't vary by league at all.

    Returns {league_code: [match, ...]}, one key per requested league code,
    using the same per-match dict shape as fetch_finished_match_ids.
    """
    LOGGER.info("Fetching FotMob matches (multi-league) | leagues=%s date=%s", sorted(league_ids), day)
    result: dict[str, list[dict]] = {code: [] for code in league_ids}
    payload = _fetch_matches_payload(day, delay)
    if payload is None:
        return result

    id_to_codes: dict[int, list[str]] = {}
    for code, lid in league_ids.items():
        id_to_codes.setdefault(lid, []).append(code)

    for league_entry in payload.get("leagues", []):
        codes = id_to_codes.get(league_entry.get("id"))
        if not codes:
            continue
        parsed = _parse_finished_matches(league_entry.get("matches", []))
        for code in codes:
            result[code].extend(parsed)

    LOGGER.info(
        "Got matches (multi-league) | date=%s | %s", day,
        {code: len(matches) for code, matches in result.items()},
    )
    return result


def _extract_top_stat(top_stats: dict, label: str) -> float | int | None:
    entry = top_stats.get(label)
    if entry is None:
        return None
    return entry.get("stat", {}).get("value")


def _extract_defense_stat(stat_groups: list, label: str) -> float | None:
    """Extract a stat from the 'Defense' stat group (separate from Top Stats)."""
    for group in stat_groups:
        if group.get("title") == "Defense":
            stats = group.get("stats", {})
            entry = stats.get(label)
            if entry is None:
                return None
            return entry.get("stat", {}).get("value")
    return None


def fetch_match_player_stats(fotmob_match_id: int, delay: float = 1.0) -> list[dict]:
    """Fetch per-player stats for one finished FotMob match."""
    url = _MATCH_DETAILS_URL.format(match_id=fotmob_match_id)
    LOGGER.info("Fetching FotMob match details | match_id=%s -> %s", fotmob_match_id, url)
    resp = requests.get(url, headers=_HEADERS, timeout=30)
    resp.raise_for_status()
    time.sleep(delay)

    payload = resp.json()
    player_stats = payload.get("content", {}).get("playerStats") or {}

    rows: list[dict] = []
    for player_id_str, player in player_stats.items():
        try:
            stat_groups = player.get("stats", [])
            top_stats = stat_groups[0]["stats"] if stat_groups else {}
            row = {
                "player_id": int(player_id_str),
                "player_name": player.get("name"),
                "opta_id": player.get("optaId"),
                "team_name": player.get("teamName"),
            }
            for column, label in _TOP_STAT_FIELDS.items():
                row[column] = _extract_top_stat(top_stats, label)
            row["interceptions"] = _extract_defense_stat(stat_groups, "Interceptions")
            row["recoveries"] = _extract_defense_stat(stat_groups, "Recoveries")
            rows.append(row)
        except (KeyError, TypeError, ValueError) as exc:
            LOGGER.warning(
                "Skipping malformed player entry id=%s match_id=%s: %s",
                player_id_str, fotmob_match_id, exc,
            )

    LOGGER.info("Got %d player rows | match_id=%s", len(rows), fotmob_match_id)
    return rows


def fetch_player_match_stats(
    league: str, date_from: date, date_to: date, delay: float = 1.0
) -> pd.DataFrame:
    """Fetch per-player, per-match stats for a league across a date range.

    Returns a flat DataFrame: one row per player per finished match, with
    match-identifying columns (fotmob_match_id, match_date, home_team,
    away_team) alongside the per-player stat columns.
    """
    league_id = LEAGUE_IDS.get(league)
    if league_id is None:
        raise ValueError(f"Unsupported league '{league}'. Supported: {sorted(LEAGUE_IDS)}")

    all_rows: list[dict] = []
    for day in _date_range(date_from, date_to):
        try:
            matches = fetch_finished_match_ids(day, league_id=league_id, delay=delay)
        except requests.RequestException as exc:
            LOGGER.error("Failed to fetch matches for %s: %s", day, exc)
            continue

        for match in matches:
            try:
                player_rows = fetch_match_player_stats(match["fotmob_match_id"], delay=delay)
            except requests.RequestException as exc:
                LOGGER.error(
                    "Failed to fetch player stats for match_id=%s: %s",
                    match["fotmob_match_id"], exc,
                )
                continue
            for player_row in player_rows:
                all_rows.append({**match, **player_row})

    LOGGER.info(
        "Got %d player-match rows | league=%s %s..%s", len(all_rows), league, date_from, date_to
    )
    return pd.DataFrame(all_rows, columns=PLAYER_MATCH_COLUMNS)


def fetch_player_match_stats_multi_league(
    leagues: dict[str, int], date_from: date, date_to: date, delay: float = 1.0
) -> dict[str, pd.DataFrame]:
    """Like fetch_player_match_stats, but backfills several leagues' player
    stats together (US#190), sharing one day-level request across all of
    them (fetch_matches_for_leagues) instead of one per league per day --
    the match-detail requests below are still genuinely one-per-match-per-
    league (no way to share those), but the day-scan, roughly half of a
    real backfill's total request volume, collapses from N requests/day to
    exactly 1.

    Returns {league_code: DataFrame}, one key per requested league code,
    each shaped like fetch_player_match_stats's own return value.
    """
    all_rows: dict[str, list[dict]] = {code: [] for code in leagues}
    for day in _date_range(date_from, date_to):
        try:
            matches_by_league = fetch_matches_for_leagues(day, leagues, delay=delay)
        except requests.RequestException as exc:
            LOGGER.error("Failed to fetch matches for %s: %s", day, exc)
            continue

        for code, matches in matches_by_league.items():
            for match in matches:
                try:
                    player_rows = fetch_match_player_stats(match["fotmob_match_id"], delay=delay)
                except requests.RequestException as exc:
                    LOGGER.error(
                        "Failed to fetch player stats for match_id=%s: %s",
                        match["fotmob_match_id"], exc,
                    )
                    continue
                for player_row in player_rows:
                    all_rows[code].append({**match, **player_row})

    for code, rows in all_rows.items():
        LOGGER.info(
            "Got %d player-match rows | league=%s %s..%s", len(rows), code, date_from, date_to
        )
    return {code: pd.DataFrame(rows, columns=PLAYER_MATCH_COLUMNS) for code, rows in all_rows.items()}
