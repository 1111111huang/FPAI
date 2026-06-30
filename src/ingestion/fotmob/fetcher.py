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
]


def _date_range(date_from: date, date_to: date) -> list[date]:
    days = (date_to - date_from).days
    return [date_from + timedelta(days=offset) for offset in range(days + 1)]


def fetch_finished_match_ids(day: date, league_id: int, delay: float = 1.0) -> list[dict]:
    """Return finished matches for one league on one date.

    Each dict has keys: fotmob_match_id, match_date, home_team, away_team.
    """
    url = _MATCHES_URL.format(date=day.strftime("%Y%m%d"))
    LOGGER.info("Fetching FotMob matches | league_id=%s date=%s -> %s", league_id, day, url)
    resp = requests.get(url, headers=_HEADERS, timeout=30)
    resp.raise_for_status()
    time.sleep(delay)

    payload = resp.json()
    leagues = [entry for entry in payload.get("leagues", []) if entry.get("id") == league_id]

    matches: list[dict] = []
    for league in leagues:
        for match in league.get("matches", []):
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

    LOGGER.info("Got %d finished matches | league_id=%s date=%s", len(matches), league_id, day)
    return matches


def _extract_top_stat(top_stats: dict, label: str) -> float | int | None:
    entry = top_stats.get(label)
    if entry is None:
        return None
    return entry.get("stat", {}).get("value")


def fetch_match_player_stats(fotmob_match_id: int, delay: float = 1.0) -> list[dict]:
    """Fetch per-player stats for one finished FotMob match."""
    url = _MATCH_DETAILS_URL.format(match_id=fotmob_match_id)
    LOGGER.info("Fetching FotMob match details | match_id=%s -> %s", fotmob_match_id, url)
    resp = requests.get(url, headers=_HEADERS, timeout=30)
    resp.raise_for_status()
    time.sleep(delay)

    payload = resp.json()
    player_stats = payload.get("content", {}).get("playerStats", {})

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
