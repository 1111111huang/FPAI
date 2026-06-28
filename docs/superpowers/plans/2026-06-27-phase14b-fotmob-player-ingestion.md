# Phase 14b Part 2: FotMob Player Data Ingestion Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a new `src/ingestion/fotmob/` package that fetches per-match player stats (rating, xG, xA, xGOT, goals, assists, shots, minutes) from FotMob's internal JSON API, resolves them to existing `raw_matches` rows, and persists them into two new DuckDB tables (`raw_player_match_stats`, `player_dim`). Wires a new `fetch-fotmob` CLI command and extends `refresh-data`. Corresponds to US#92–95 in `documents/user_stories.md`.

**Prerequisite:** This plan assumes `docs/superpowers/plans/2026-06-27-phase14b-ingestion-restructuring.md` (US#91) has already been executed, so `src/ingestion/common/team_mapping.py` exists.

**Architecture:** `src/ingestion/fotmob/fetcher.py` is a pure HTTP+parsing module with no DB dependency: `fetch_finished_match_ids()` hits `fotmob.com/api/data/matches?date=...` to discover finished matches for a league/date, `fetch_match_player_stats()` hits `fotmob.com/api/data/matchDetails?matchId=...` to pull `content.playerStats`, and `fetch_player_match_stats()` combines both across a date range into one flat DataFrame. `src/ingestion/fotmob/merge.py` resolves that DataFrame's `(match_date, home_team, away_team)` to `raw_matches.match_id` by date+team join (mirroring `update_raw_matches_xg`'s existing join pattern — not by recomputing the match_id hash), then upserts `player_dim` and `raw_player_match_stats` via DuckDB's `ON CONFLICT ... DO UPDATE`.

**Tech Stack:** Python, `requests`, `pandas`, DuckDB, pytest, `unittest.mock`.

**API details verified live (2026-06-27):**
- `GET https://www.fotmob.com/api/data/matches?date=YYYYMMDD` → `{"leagues": [{"id": 47, "matches": [{"id": 4193901, "leagueId": 47, "home": {"name": "Arsenal", ...}, "away": {"name": "Everton", ...}, "status": {"finished": true, "utcTime": "2024-05-19T15:00:00.000Z"}}, ...]}]}`. Premier League's `leagueId` is `47`.
- `GET https://www.fotmob.com/api/data/matchDetails?matchId=4193901` → `content.playerStats` is a dict keyed by player ID string, e.g. `{"23354": {"id": 23354, "name": "Ashley Young", "optaId": "18892", "teamName": "Everton", "stats": [{"key": "top_stats", "title": "Top stats", "stats": {"FotMob rating": {"stat": {"value": 5.58}}, "Minutes played": {"stat": {"value": 90}}, "Expected assists (xA)": {"stat": {"value": 0.01}}, ...}}, ...]}}`. Fields like `"Expected goals (xG)"` are absent entirely for players with no attacking involvement (not present as zero) — extraction must tolerate missing keys and produce `None`, not `0.0`.
- Both endpoints return plain JSON with HTTP 200 given only a browser-like `User-Agent` header — no auth, no anti-bot challenge encountered.

---

### Task 1: Build the FotMob fetcher (TDD)

**Files:**
- Create: `src/ingestion/fotmob/__init__.py`
- Create: `src/ingestion/fotmob/fetcher.py`
- Test: `tests/test_fotmob_fetcher.py`

- [ ] **Step 1: Create the package directory**

```bash
mkdir -p src/ingestion/fotmob
touch src/ingestion/fotmob/__init__.py
```

- [ ] **Step 2: Write the failing tests**

```python
"""Tests for FotMob player-stats fetching: match discovery and player-stat extraction."""

from __future__ import annotations

from datetime import date
from pathlib import Path
import sys
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.ingestion.fotmob.fetcher import (
    fetch_finished_match_ids,
    fetch_match_player_stats,
    fetch_player_match_stats,
)


def _matches_payload(matches: list[dict], league_id: int = 47) -> dict:
    return {"leagues": [{"id": league_id, "matches": matches}]}


def _match_entry(
    match_id: int = 4193901,
    home: str = "Arsenal",
    away: str = "Everton",
    finished: bool = True,
    utc_time: str = "2024-05-19T15:00:00.000Z",
) -> dict:
    return {
        "id": match_id,
        "leagueId": 47,
        "home": {"id": 9825, "name": home},
        "away": {"id": 8668, "name": away},
        "status": {"finished": finished, "utcTime": utc_time},
    }


def _match_details_payload(player_stats: dict) -> dict:
    return {"content": {"playerStats": player_stats}}


def _player_entry(
    player_id: int = 23354,
    name: str = "Ashley Young",
    opta_id: str = "18892",
    team_name: str = "Everton",
    top_stats: dict | None = None,
) -> dict:
    stats = top_stats if top_stats is not None else {
        "FotMob rating": {"stat": {"value": 5.58}},
        "Minutes played": {"stat": {"value": 90}},
        "Goals": {"stat": {"value": 0}},
        "Assists": {"stat": {"value": 0}},
        "Expected assists (xA)": {"stat": {"value": 0.01}},
        # "Expected goals (xG)" deliberately absent, matching real FotMob payloads
        # for players with no attacking involvement.
    }
    return {
        str(player_id): {
            "id": player_id,
            "name": name,
            "optaId": opta_id,
            "teamName": team_name,
            "stats": [{"key": "top_stats", "title": "Top stats", "stats": stats}],
        }
    }


def _mock_resp(payload: dict) -> MagicMock:
    mock = MagicMock()
    mock.raise_for_status = MagicMock()
    mock.json.return_value = payload
    return mock


# ---------------------------------------------------------------------------
# fetch_finished_match_ids
# ---------------------------------------------------------------------------

def test_fetch_finished_match_ids_includes_finished_matches():
    payload = _matches_payload([_match_entry()])
    with patch("src.ingestion.fotmob.fetcher.requests.get", return_value=_mock_resp(payload)), \
         patch("src.ingestion.fotmob.fetcher.time.sleep"):
        matches = fetch_finished_match_ids(date(2024, 5, 19), league_id=47, delay=0)

    assert len(matches) == 1
    assert matches[0]["fotmob_match_id"] == 4193901
    assert matches[0]["home_team"] == "Arsenal"
    assert matches[0]["away_team"] == "Everton"
    assert matches[0]["match_date"] == pd.Timestamp("2024-05-19")


def test_fetch_finished_match_ids_excludes_unfinished_matches():
    payload = _matches_payload([_match_entry(finished=False)])
    with patch("src.ingestion.fotmob.fetcher.requests.get", return_value=_mock_resp(payload)), \
         patch("src.ingestion.fotmob.fetcher.time.sleep"):
        matches = fetch_finished_match_ids(date(2024, 5, 19), league_id=47, delay=0)

    assert matches == []


def test_fetch_finished_match_ids_filters_to_requested_league():
    payload = {
        "leagues": [
            {"id": 47, "matches": [_match_entry(match_id=1)]},
            {"id": 87, "matches": [_match_entry(match_id=2)]},
        ]
    }
    with patch("src.ingestion.fotmob.fetcher.requests.get", return_value=_mock_resp(payload)), \
         patch("src.ingestion.fotmob.fetcher.time.sleep"):
        matches = fetch_finished_match_ids(date(2024, 5, 19), league_id=47, delay=0)

    assert len(matches) == 1
    assert matches[0]["fotmob_match_id"] == 1


# ---------------------------------------------------------------------------
# fetch_match_player_stats
# ---------------------------------------------------------------------------

def test_fetch_match_player_stats_extracts_present_fields():
    payload = _match_details_payload(_player_entry())
    with patch("src.ingestion.fotmob.fetcher.requests.get", return_value=_mock_resp(payload)), \
         patch("src.ingestion.fotmob.fetcher.time.sleep"):
        rows = fetch_match_player_stats(4193901, delay=0)

    assert len(rows) == 1
    row = rows[0]
    assert row["player_id"] == 23354
    assert row["player_name"] == "Ashley Young"
    assert row["opta_id"] == "18892"
    assert row["team_name"] == "Everton"
    assert row["rating"] == pytest.approx(5.58)
    assert row["minutes_played"] == 90
    assert row["xa"] == pytest.approx(0.01)


def test_fetch_match_player_stats_returns_none_for_missing_fields():
    """A field absent from FotMob's payload (e.g. xG for a non-attacking player)
    must become None, not 0.0 — the two mean different things downstream."""
    payload = _match_details_payload(_player_entry())
    with patch("src.ingestion.fotmob.fetcher.requests.get", return_value=_mock_resp(payload)), \
         patch("src.ingestion.fotmob.fetcher.time.sleep"):
        rows = fetch_match_player_stats(4193901, delay=0)

    assert rows[0]["xg"] is None


def test_fetch_match_player_stats_handles_multiple_players():
    player_stats = {**_player_entry(player_id=1, name="Player One"), **_player_entry(player_id=2, name="Player Two")}
    payload = _match_details_payload(player_stats)
    with patch("src.ingestion.fotmob.fetcher.requests.get", return_value=_mock_resp(payload)), \
         patch("src.ingestion.fotmob.fetcher.time.sleep"):
        rows = fetch_match_player_stats(4193901, delay=0)

    assert {row["player_id"] for row in rows} == {1, 2}


# ---------------------------------------------------------------------------
# fetch_player_match_stats (end-to-end across a date range)
# ---------------------------------------------------------------------------

def test_fetch_player_match_stats_combines_match_and_player_rows():
    matches_payload = _matches_payload([_match_entry()])
    details_payload = _match_details_payload(_player_entry())

    def fake_get(url, **kwargs):
        if "matchDetails" in url:
            return _mock_resp(details_payload)
        return _mock_resp(matches_payload)

    with patch("src.ingestion.fotmob.fetcher.requests.get", side_effect=fake_get), \
         patch("src.ingestion.fotmob.fetcher.time.sleep"):
        df = fetch_player_match_stats("E0", date(2024, 5, 19), date(2024, 5, 19), delay=0)

    assert len(df) == 1
    assert df.iloc[0]["player_name"] == "Ashley Young"
    assert df.iloc[0]["home_team"] == "Arsenal"
    assert df.iloc[0]["fotmob_match_id"] == 4193901


def test_fetch_player_match_stats_rejects_unsupported_league():
    with pytest.raises(ValueError, match="Unsupported league"):
        fetch_player_match_stats("XX", date(2024, 5, 19), date(2024, 5, 19), delay=0)
```

- [ ] **Step 3: Run tests to verify they fail**

Run: `pytest tests/test_fotmob_fetcher.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'src.ingestion.fotmob.fetcher'`

- [ ] **Step 4: Write the implementation**

```python
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
            matches.append(
                {
                    "fotmob_match_id": match["id"],
                    "match_date": pd.to_datetime(utc_time).normalize(),
                    "home_team": match["home"]["name"],
                    "away_team": match["away"]["name"],
                }
            )
    return matches


def _extract_top_stat(top_stats: dict, label: str) -> float | int | None:
    entry = top_stats.get(label)
    if entry is None:
        return None
    return entry.get("stat", {}).get("value")


def fetch_match_player_stats(fotmob_match_id: int, delay: float = 1.0) -> list[dict]:
    """Fetch per-player stats for one finished FotMob match."""
    url = _MATCH_DETAILS_URL.format(match_id=fotmob_match_id)
    resp = requests.get(url, headers=_HEADERS, timeout=30)
    resp.raise_for_status()
    time.sleep(delay)

    payload = resp.json()
    player_stats = payload.get("content", {}).get("playerStats", {})

    rows: list[dict] = []
    for player_id_str, player in player_stats.items():
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
        matches = fetch_finished_match_ids(day, league_id=league_id, delay=delay)
        for match in matches:
            player_rows = fetch_match_player_stats(match["fotmob_match_id"], delay=delay)
            for player_row in player_rows:
                all_rows.append({**match, **player_row})

    return pd.DataFrame(all_rows, columns=PLAYER_MATCH_COLUMNS)
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `pytest tests/test_fotmob_fetcher.py -v`
Expected: 8 passed

- [ ] **Step 6: Commit**

```bash
git add src/ingestion/fotmob/__init__.py src/ingestion/fotmob/fetcher.py tests/test_fotmob_fetcher.py
git commit -m "feat: add FotMob player-stats fetcher (US#92)"
```

---

### Task 2: Build the merge/persistence layer (TDD)

**Files:**
- Create: `src/ingestion/fotmob/merge.py`
- Test: `tests/test_fotmob_merge.py`

- [ ] **Step 1: Write the failing tests**

```python
"""Tests for resolving FotMob player rows to raw_matches and persisting them."""

from __future__ import annotations

from pathlib import Path
import sys

import pandas as pd
import pytest
import yaml

sys.path.append(str(Path(__file__).resolve().parents[1]))

duckdb = pytest.importorskip("duckdb")

from src.ingestion.fotmob.merge import upsert_player_match_stats
from src.utils.db_manager import DuckDBManager


def _make_db_manager(tmp_path: Path) -> DuckDBManager:
    db_path = tmp_path / "test_fpai.db"
    config_path = tmp_path / "config.yaml"
    config_path.write_text(yaml.safe_dump({"paths": {"database_path": str(db_path)}}), encoding="utf-8")
    return DuckDBManager(config_path=str(config_path))


def _seed_raw_matches(db_manager: DuckDBManager) -> None:
    with db_manager.connection() as conn:
        conn.execute(
            """
            CREATE TABLE raw_matches (
                match_id TEXT PRIMARY KEY, date TIMESTAMP, home_team TEXT, away_team TEXT
            )
            """
        )
        conn.execute(
            "INSERT INTO raw_matches VALUES ('match-abc', '2024-05-19', 'Arsenal', 'Everton')"
        )


def _fotmob_rows() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "fotmob_match_id": 4193901, "match_date": pd.Timestamp("2024-05-19"),
                "home_team": "Arsenal", "away_team": "Everton",
                "player_id": 23354, "player_name": "Ashley Young", "opta_id": "18892", "team_name": "Everton",
                "rating": 5.58, "minutes_played": 90, "goals": 0, "assists": 0,
                "xg": None, "xa": 0.01, "xgot": None, "shots": 0,
            },
            {
                "fotmob_match_id": 4193901, "match_date": pd.Timestamp("2024-05-19"),
                "home_team": "Arsenal", "away_team": "Everton",
                "player_id": 99001, "player_name": "Idrissa Gana Gueye", "opta_id": "55001", "team_name": "Everton",
                "rating": 8.16, "minutes_played": 90, "goals": 1, "assists": 0,
                "xg": 0.06, "xa": 0.02, "xgot": 0.31, "shots": 1,
            },
        ]
    )


def test_upsert_player_match_stats_matches_by_date_and_team(tmp_path: Path) -> None:
    db_manager = _make_db_manager(tmp_path)
    _seed_raw_matches(db_manager)

    result = upsert_player_match_stats(_fotmob_rows(), db_manager)

    assert result == {"matched": 2, "unmatched": 0, "players_upserted": 2, "rows_upserted": 2}

    with db_manager.connection() as conn:
        stats_rows = conn.execute(
            "SELECT match_id, player_id, rating, xg, xa FROM raw_player_match_stats ORDER BY player_id"
        ).fetchall()
        player_rows = conn.execute("SELECT player_id, player_name, opta_id FROM player_dim ORDER BY player_id").fetchall()

    assert stats_rows == [
        ("match-abc", 23354, pytest.approx(5.58), None, pytest.approx(0.01)),
        ("match-abc", 99001, pytest.approx(8.16), pytest.approx(0.06), pytest.approx(0.02)),
    ]
    assert player_rows == [
        (23354, "Ashley Young", "18892"),
        (99001, "Idrissa Gana Gueye", "55001"),
    ]


def test_upsert_player_match_stats_counts_unmatched_rows(tmp_path: Path) -> None:
    db_manager = _make_db_manager(tmp_path)
    _seed_raw_matches(db_manager)

    fotmob_df = _fotmob_rows()
    fotmob_df["home_team"] = "Some Other Team"  # won't match the seeded raw_matches row

    result = upsert_player_match_stats(fotmob_df, db_manager)

    assert result["matched"] == 0
    assert result["unmatched"] == 2
    assert result["rows_upserted"] == 0

    with db_manager.connection() as conn:
        count = conn.execute("SELECT COUNT(*) FROM raw_player_match_stats").fetchone()[0]
    assert count == 0


def test_upsert_player_match_stats_is_idempotent_on_rerun(tmp_path: Path) -> None:
    db_manager = _make_db_manager(tmp_path)
    _seed_raw_matches(db_manager)

    upsert_player_match_stats(_fotmob_rows(), db_manager)

    updated_rows = _fotmob_rows()
    updated_rows.loc[updated_rows["player_id"] == 99001, "rating"] = 9.0
    result = upsert_player_match_stats(updated_rows, db_manager)

    assert result["matched"] == 2
    with db_manager.connection() as conn:
        count = conn.execute("SELECT COUNT(*) FROM raw_player_match_stats").fetchone()[0]
        rating = conn.execute(
            "SELECT rating FROM raw_player_match_stats WHERE player_id = 99001"
        ).fetchone()[0]
    assert count == 2  # no duplicate rows from the second run
    assert rating == pytest.approx(9.0)  # second run's value won


def test_upsert_player_match_stats_handles_empty_input(tmp_path: Path) -> None:
    db_manager = _make_db_manager(tmp_path)
    _seed_raw_matches(db_manager)

    result = upsert_player_match_stats(pd.DataFrame(columns=[
        "fotmob_match_id", "match_date", "home_team", "away_team",
        "player_id", "player_name", "opta_id", "team_name",
        "rating", "minutes_played", "goals", "assists", "xg", "xa", "xgot", "shots",
    ]), db_manager)

    assert result == {"matched": 0, "unmatched": 0, "players_upserted": 0, "rows_upserted": 0}
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_fotmob_merge.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'src.ingestion.fotmob.merge'`

- [ ] **Step 3: Write the implementation**

```python
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
    """Join FotMob rows to raw_matches by date+team to recover match_id."""
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
    """Resolve match_id and upsert player_dim + raw_player_match_stats rows."""
    if fotmob_df.empty:
        return dict(_EMPTY_RESULT)

    resolved = resolve_match_ids(fotmob_df, db_manager, mapping_path=mapping_path)
    has_match = resolved["match_id"].notna()
    unmatched = int((~has_match).sum())
    if unmatched:
        LOGGER.warning("%d FotMob player rows had no raw_matches match — skipped.", unmatched)

    matched = resolved.loc[has_match].copy()
    if matched.empty:
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
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_fotmob_merge.py -v`
Expected: 4 passed

- [ ] **Step 5: Commit**

```bash
git add src/ingestion/fotmob/merge.py tests/test_fotmob_merge.py
git commit -m "feat: add raw_player_match_stats/player_dim upsert from FotMob data (US#93, US#94)"
```

---

### Task 3: Wire the `fetch-fotmob` CLI command and extend `refresh-data`

**Files:**
- Modify: `main.py` (CLI parser, around line 70-77; new handler function near line 313-346; `run_refresh_data` at line 374-380; dispatch block near line 1095)

- [ ] **Step 1: Add the CLI subparser**

In `main.py`'s `_build_parser()`, immediately after the existing `fetch-understat` subparser block, add:

```python
    # fetch-fotmob (US#95)
    fotmob_parser = subparsers.add_parser(
        "fetch-fotmob", help="Fetch per-match player stats from FotMob and populate raw_player_match_stats"
    )
    fotmob_parser.add_argument("--league", type=str, default="E0", help="Football-Data league code (default: E0).")
    fotmob_parser.add_argument("--from_season", type=int, default=None, help="First season start year to fetch.")
    fotmob_parser.add_argument("--to_season", type=int, default=None, help="Last season start year to fetch.")
    fotmob_parser.add_argument("--delay", type=float, default=1.0, help="Polite delay in seconds between requests.")
```

- [ ] **Step 2: Add the `run_fetch_fotmob` handler**

In `main.py`, immediately after `run_fetch_understat` (after its closing line, before `def run_ingest`), add:

```python
def run_fetch_fotmob(
    app_settings: AppSettings,
    db_manager: DuckDBManager,
    league: str = "E0",
    from_season: int | None = None,
    to_season: int | None = None,
    delay: float = 1.0,
) -> None:
    from datetime import date as _date

    from src.ingestion.fotmob.fetcher import fetch_player_match_stats
    from src.ingestion.fotmob.merge import upsert_player_match_stats

    LOGGER.info("Executing command: fetch-fotmob | league=%s", league)
    with db_manager.connection() as conn:
        bounds = conn.execute("SELECT YEAR(MIN(date)), YEAR(MAX(date)) FROM raw_matches").fetchone()
    if bounds is None or bounds[0] is None:
        LOGGER.error("raw_matches is empty — run ingest first.")
        return
    detected_from = (from_season or bounds[0]) - 1
    detected_to = to_season or (bounds[1] - 1)
    LOGGER.info("Fetching FotMob player stats | seasons %d-%d | league=%s", detected_from, detected_to, league)

    season_frames = []
    for season_start in range(detected_from, detected_to + 1):
        season_from = _date(season_start, 8, 1)
        season_to = _date(season_start + 1, 7, 31)
        season_frames.append(fetch_player_match_stats(league, season_from, season_to, delay=delay))
    fotmob_df = pd.concat(season_frames, ignore_index=True) if season_frames else pd.DataFrame()

    if fotmob_df.empty:
        LOGGER.error("No FotMob data returned — check league/season args and network.")
        return
    LOGGER.info("Fetched %d FotMob player-match rows total.", len(fotmob_df))
    result = upsert_player_match_stats(fotmob_df, db_manager)
    LOGGER.info(
        "FotMob upsert | matched=%d | unmatched=%d | players=%d | rows=%d",
        result["matched"], result["unmatched"], result["players_upserted"], result["rows_upserted"],
    )
```

- [ ] **Step 3: Extend `run_refresh_data`**

In `main.py`, change:

```python
def run_refresh_data(app_settings: AppSettings, db_manager: DuckDBManager, league: str = "E0", force: bool = False) -> None:
    """Run scrape → ingest → fetch-understat in sequence (US#81)."""
    LOGGER.info("Executing command: refresh-data | league=%s | force=%s", league, force)
    run_scrape(app_settings, force=force)
    run_ingest(app_settings, db_manager, force=force)
    run_fetch_understat(app_settings, db_manager, league=league, rebuild_features=True)
    LOGGER.info("refresh-data complete.")
```

to:

```python
def run_refresh_data(app_settings: AppSettings, db_manager: DuckDBManager, league: str = "E0", force: bool = False) -> None:
    """Run scrape → ingest → fetch-understat → fetch-fotmob in sequence (US#81, US#95)."""
    LOGGER.info("Executing command: refresh-data | league=%s | force=%s", league, force)
    run_scrape(app_settings, force=force)
    run_ingest(app_settings, db_manager, force=force)
    run_fetch_understat(app_settings, db_manager, league=league, rebuild_features=True)
    run_fetch_fotmob(app_settings, db_manager, league=league)
    LOGGER.info("refresh-data complete.")
```

- [ ] **Step 4: Add CLI dispatch**

In `main.py`, immediately after the existing dispatch block:

```python
    elif args.command == "fetch-understat":
```

(find its body, which calls `run_fetch_understat(...)`) add a new `elif` branch right after it:

```python
    elif args.command == "fetch-fotmob":
        run_fetch_fotmob(
            app_settings, db_manager,
            league=str(args.league), from_season=args.from_season, to_season=args.to_season, delay=float(args.delay),
        )
```

- [ ] **Step 5: Run the full test suite**

Run: `pytest -q`
Expected: all tests pass, including the new `test_fotmob_fetcher.py` and `test_fotmob_merge.py` suites

- [ ] **Step 6: Manual smoke test against real data**

Run: `python main.py fetch-fotmob --league E0 --from_season 2024 --to_season 2024 --delay 1.0`
Expected: log lines `Fetching FotMob player stats | seasons 2024-2024 | league=E0`, `Fetched <N> FotMob player-match rows total.`, and `FotMob upsert | matched=<N> | unmatched=<N> | players=<N> | rows=<N>` with `matched` close to the full row count (some unmatched rows are expected if team-name mapping needs a `config/team_mapping.json` entry — check the WARNING log lines for any unmapped team names and add them if so).

- [ ] **Step 7: Commit**

```bash
git add main.py
git commit -m "feat: wire fetch-fotmob CLI command and extend refresh-data (US#95)"
```

---

### Task 4: Mark US#92–95 complete in documentation

**Files:**
- Modify: `documents/FRAI_TECHSPEC.md` (Section 27 status line)
- Modify: `documents/user_stories.md` (Phase 14b heading)

- [ ] **Step 1: Update the techspec status line**

In `documents/FRAI_TECHSPEC.md` Section 27, change:

```
**Status: Phase 14a (tier reorg) implemented — see 27.2 for the design and `config/competitions.yaml` / `src/logic/competition_registry.py` for the implementation. Phase 14b and 14c remain planned.** Story breakdown lives in `documents/user_stories.md` Phase 14.
```

to:

```
**Status: Phase 14a and Phase 14b implemented.** Phase 14a: `config/competitions.yaml` / `src/logic/competition_registry.py` (see 27.2). Phase 14b: `src/ingestion/fotmob/` for fetch+merge, `raw_player_match_stats` / `player_dim` tables, `fetch-fotmob` CLI command (see 27.3). **Phase 14c remains planned.** Story breakdown lives in `documents/user_stories.md` Phase 14.
```

- [ ] **Step 2: Update the user stories heading**

In `documents/user_stories.md`, change:

```
### Phase 14b: Player Data Sourcing & Ingestion
```

to:

```
### Phase 14b: Player Data Sourcing & Ingestion — Completed
```

- [ ] **Step 3: Commit**

```bash
git add documents/FRAI_TECHSPEC.md documents/user_stories.md
git commit -m "docs: mark Phase 14b FotMob player data ingestion complete (US#92-95)"
```
