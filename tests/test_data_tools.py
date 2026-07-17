"""W34: injectable clock for get_data_freshness(), plus 7/8-day staleness
boundary tests. Previously untested -- pd.Timestamp.now() was called
directly inline, with no seam to simulate day-by-day advancement without
globally monkeypatching pandas.Timestamp.now (risky, used throughout the
codebase)."""

from __future__ import annotations

from pathlib import Path
import sys
from unittest.mock import MagicMock, patch

import pandas as pd

sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.tools.data_tools import get_data_freshness


def _mock_manager(match_count: int, max_date) -> MagicMock:
    conn = MagicMock()
    conn.execute.return_value.fetchone.return_value = (match_count, max_date)
    manager = MagicMock()
    manager.connection.return_value.__enter__.return_value = conn
    manager.connection.return_value.__exit__.return_value = False
    return manager


def test_exactly_seven_days_old_is_not_stale() -> None:
    max_date = pd.Timestamp("2026-07-01")
    now = pd.Timestamp("2026-07-08")  # exactly 7 days later

    with patch("src.tools.data_tools.DuckDBManager", return_value=_mock_manager(100, max_date)):
        result = get_data_freshness(now_fn=lambda: now)

    assert result["days_since_update"] == 7
    assert result["is_stale"] is False


def test_exactly_eight_days_old_is_stale() -> None:
    max_date = pd.Timestamp("2026-07-01")
    now = pd.Timestamp("2026-07-09")  # exactly 8 days later

    with patch("src.tools.data_tools.DuckDBManager", return_value=_mock_manager(100, max_date)):
        result = get_data_freshness(now_fn=lambda: now)

    assert result["days_since_update"] == 8
    assert result["is_stale"] is True


def test_staleness_flips_correctly_as_simulated_days_advance() -> None:
    max_date = pd.Timestamp("2026-07-01")
    manager = _mock_manager(100, max_date)

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

    with patch("src.tools.data_tools.DuckDBManager", return_value=_mock_manager(50, max_date)):
        result = get_data_freshness()

    assert result["days_since_update"] == 3
    assert result["is_stale"] is False
