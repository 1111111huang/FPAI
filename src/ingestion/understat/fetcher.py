"""Fetch xG match data from understat.com for raw_matches enrichment."""

from __future__ import annotations

import time

import pandas as pd
import requests

from src.utils.logger import get_logger

LOGGER = get_logger(__name__)

LEAGUE_MAP: dict[str, str] = {
    "E0": "EPL",
    "EPL": "EPL",
    "SP1": "La_liga",  # US#146, live-verified 2026-08-06 against getLeagueData
    # US#165, live-verified 2026-08-15 against getLeagueData -- real,
    # non-empty team payloads for all three (e.g. Bundesliga's included
    # "Bayern Munich", Ligue_1's included "Lille"/"Marseille").
    "I1": "Serie_A",
    "D1": "Bundesliga",
    "F1": "Ligue_1",
}

_API_URL = "https://understat.com/getLeagueData/{league}/{season}"

# understat.com requires X-Requested-With to return JSON instead of 404.
_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
    ),
    "Accept": "application/json, text/javascript, */*; q=0.01",
    "X-Requested-With": "XMLHttpRequest",
}


def fetch_league_season(league: str, season: int, delay: float = 1.5) -> pd.DataFrame:
    """Fetch completed match xG values for one league-season from understat.com.

    Args:
        league: Football-Data code (e.g. 'E0') or Understat name ('EPL').
        season: Start year of season (e.g. 2023 for 2023/24).
        delay: Polite wait in seconds after the HTTP request.

    Returns:
        DataFrame with columns: date, home_team, away_team, xg_h, xg_a, season.
    """
    understat_league = LEAGUE_MAP.get(league, league)
    url = _API_URL.format(league=understat_league, season=season)
    headers = {**_HEADERS, "Referer": f"https://understat.com/league/{understat_league}/{season}"}

    LOGGER.info("Fetching Understat | %s/%s -> %s", league, season, url)
    resp = requests.get(url, headers=headers, timeout=30)
    resp.raise_for_status()
    time.sleep(delay)

    records = resp.json().get("dates", [])
    rows = []
    for m in records:
        if not m.get("isResult"):
            continue
        try:
            rows.append({
                "date": pd.to_datetime(m["datetime"]).normalize(),
                "home_team": m["h"]["title"],
                "away_team": m["a"]["title"],
                "xg_h": float(m["xG"]["h"]),
                "xg_a": float(m["xG"]["a"]),
                "season": season,
            })
        except (KeyError, ValueError, TypeError) as exc:
            LOGGER.warning("Skipping malformed match id=%s: %s", m.get("id"), exc)

    df = pd.DataFrame(rows, columns=["date", "home_team", "away_team", "xg_h", "xg_a", "season"])
    LOGGER.info("Got %d completed matches | %s/%s", len(df), league, season)
    return df


def fetch_seasons_range(
    league: str, from_season: int, to_season: int, delay: float = 1.5
) -> pd.DataFrame:
    """Fetch a range of seasons and concatenate results.

    Args:
        league: Football-Data league code or Understat league name.
        from_season: First season start year (inclusive).
        to_season: Last season start year (inclusive).
        delay: Polite wait between requests.

    Returns:
        Combined DataFrame for all successfully fetched seasons.
    """
    frames: list[pd.DataFrame] = []
    for season in range(from_season, to_season + 1):
        try:
            df = fetch_league_season(league, season, delay=delay)
            frames.append(df)
        except Exception as exc:
            LOGGER.error("Failed to fetch %s/%s: %s", league, season, exc)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
