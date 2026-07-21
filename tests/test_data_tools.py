"""W34: injectable clock for get_data_freshness(), plus 7/8-day staleness
boundary tests. Previously untested -- pd.Timestamp.now() was called
directly inline, with no seam to simulate day-by-day advancement without
globally monkeypatching pandas.Timestamp.now (risky, used throughout the
codebase).

US#136: get_data_freshness() now GROUP BYs league and returns a per-league
`by_league` breakdown alongside the pre-existing blended top-level keys --
see that function's docstring for why a single blended number can mask one
competition going stale behind another staying fresh. _mock_manager below
was updated from a single-row .fetchone() mock to a multi-row .fetchall()
mock to match the new query shape."""

from __future__ import annotations

from pathlib import Path
import sys
from unittest.mock import MagicMock, patch

import pandas as pd

sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.tools.data_tools import get_data_freshness


def _mock_manager(*league_rows: tuple[str, int, object]) -> MagicMock:
    """Each row is (league, match_count, max_date), one per league -- matches
    the real `SELECT league, COUNT(*), MAX(date) FROM raw_matches GROUP BY league` shape."""
    conn = MagicMock()
    conn.execute.return_value.fetchall.return_value = list(league_rows)
    manager = MagicMock()
    manager.connection.return_value.__enter__.return_value = conn
    manager.connection.return_value.__exit__.return_value = False
    return manager


def test_exactly_seven_days_old_is_not_stale() -> None:
    max_date = pd.Timestamp("2026-07-01")
    now = pd.Timestamp("2026-07-08")  # exactly 7 days later

    with patch("src.tools.data_tools.DuckDBManager", return_value=_mock_manager(("E0", 100, max_date))):
        result = get_data_freshness(now_fn=lambda: now)

    assert result["days_since_update"] == 7
    assert result["is_stale"] is False


def test_exactly_eight_days_old_is_stale() -> None:
    max_date = pd.Timestamp("2026-07-01")
    now = pd.Timestamp("2026-07-09")  # exactly 8 days later

    with patch("src.tools.data_tools.DuckDBManager", return_value=_mock_manager(("E0", 100, max_date))):
        result = get_data_freshness(now_fn=lambda: now)

    assert result["days_since_update"] == 8
    assert result["is_stale"] is True


def test_staleness_flips_correctly_as_simulated_days_advance() -> None:
    max_date = pd.Timestamp("2026-07-01")
    manager = _mock_manager(("E0", 100, max_date))

    with patch("src.tools.data_tools.DuckDBManager", return_value=manager):
        for offset, expected_stale in [(0, False), (6, False), (7, False), (8, True), (30, True)]:
            now = max_date + pd.Timedelta(days=offset)
            result = get_data_freshness(now_fn=lambda now=now: now)
            assert result["is_stale"] is expected_stale, f"offset={offset}"
            assert result["days_since_update"] == offset


def test_real_clock_default_is_unchanged() -> None:
    """Calling with no now_fn must still use the real wall clock -- zero
    behavior change for every existing caller."""
    max_date = pd.Timestamp.now().normalize() - pd.Timedelta(days=3)

    with patch("src.tools.data_tools.DuckDBManager", return_value=_mock_manager(("E0", 50, max_date))):
        result = get_data_freshness()

    assert result["days_since_update"] == 3
    assert result["is_stale"] is False


def test_single_league_table_by_league_matches_top_level() -> None:
    """A single-competition table (e.g. today's real E0-only history before
    US#125 added Sweden) must produce a by_league entry identical in shape
    and value to the pre-existing top-level keys -- zero behavior change."""
    max_date = pd.Timestamp("2026-07-01")
    now = pd.Timestamp("2026-07-05")

    with patch("src.tools.data_tools.DuckDBManager", return_value=_mock_manager(("E0", 3800, max_date))):
        result = get_data_freshness(now_fn=lambda: now)

    assert result["by_league"] == {
        "E0": {
            "latest_match_date": "2026-07-01",
            "days_since_update": 4,
            "match_count": 3800,
            "is_stale": False,
        }
    }
    # Blended top-level == the single league's own values when there's only one.
    assert result["latest_match_date"] == "2026-07-01"
    assert result["match_count"] == 3800


def test_stale_competition_not_masked_by_a_fresh_one() -> None:
    """The exact real-world case this story is about: EPL deep in its
    off-season (stale) while Sweden's weekly refresh keeps its own data
    current. The blended top-level `is_stale` reads False (masking EPL) --
    that's pre-existing, documented behavior, not what changed. What's new
    is that by_league still surfaces EPL's own staleness distinctly."""
    now = pd.Timestamp("2026-07-21")
    e0_stale_date = pd.Timestamp("2026-05-24")  # EPL season ended, >7 days stale
    swe_fresh_date = pd.Timestamp("2026-07-20")  # Sweden refreshed yesterday

    with patch(
        "src.tools.data_tools.DuckDBManager",
        return_value=_mock_manager(("E0", 3800, e0_stale_date), ("SWE", 3489, swe_fresh_date)),
    ):
        result = get_data_freshness(now_fn=lambda: now)

    assert result["by_league"]["E0"]["is_stale"] is True
    assert result["by_league"]["SWE"]["is_stale"] is False
    # Blended top level takes the overall MAX(date) across every competition,
    # so it reads fresh even though EPL's own data hasn't moved -- exactly
    # the masking this story's by_league addition exists to let a caller see
    # past, without changing the pre-existing top-level number itself.
    assert result["is_stale"] is False
    assert result["match_count"] == 3800 + 3489


def test_empty_table_returns_empty_by_league() -> None:
    with patch("src.tools.data_tools.DuckDBManager", return_value=_mock_manager()):
        result = get_data_freshness()

    assert result["by_league"] == {}
    assert result["match_count"] == 0
    assert result["is_stale"] is True


def test_db_connection_error_returns_safe_default_with_empty_by_league() -> None:
    manager = MagicMock()
    manager.connection.side_effect = RuntimeError("db unavailable")

    with patch("src.tools.data_tools.DuckDBManager", return_value=manager):
        result = get_data_freshness()

    assert result == {
        "latest_match_date": None,
        "days_since_update": None,
        "match_count": 0,
        "is_stale": True,
        "by_league": {},
    }
