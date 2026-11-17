"""Tests for scripts/calibrate_goal_market_fallback.py -- see
docs/superpowers/specs/2026-11-16-goal-market-lambda-fallback-design.md."""
from __future__ import annotations

import contextlib
from pathlib import Path
import sys

import duckdb
import pytest

sys.path.append(str(Path(__file__).resolve().parents[1]))

from scripts import calibrate_goal_market_fallback as script


def test_fit_home_share_coefficients_recovers_a_known_linear_relationship():
    # real_home_share = 0.54 + 0.2 * gap, exactly, for 5 synthetic rows.
    rows = []
    for gap in (-0.4, -0.2, 0.0, 0.2, 0.4):
        share = 0.54 + 0.2 * gap
        lambda_total = 2.6
        lambda_home = lambda_total * share
        implied_home = 0.4 + gap / 2.0
        implied_away = 0.4 - gap / 2.0
        rows.append((lambda_total, lambda_home, implied_home, implied_away))

    slope, intercept = script.fit_home_share_coefficients(rows)

    assert slope == pytest.approx(0.2, abs=1e-6)
    assert intercept == pytest.approx(0.54, abs=1e-6)


def test_fetch_rows_by_league_reads_only_rows_with_a_real_ah_line():
    conn = duckdb.connect(":memory:")
    conn.execute("CREATE TABLE raw_matches (match_id VARCHAR, league VARCHAR)")
    conn.execute(
        'CREATE TABLE feature_store (match_id VARCHAR, "MKT_LAMBDA_TOTAL" FLOAT, '
        '"MKT_LAMBDA_HOME" FLOAT, "MKT_AH_LINE" FLOAT, "MKT_IMPLIED_HOME" FLOAT, '
        '"MKT_IMPLIED_AWAY" FLOAT)'
    )
    conn.execute("INSERT INTO raw_matches VALUES ('e1', 'E0'), ('e2', 'E0')")
    conn.execute(
        "INSERT INTO feature_store VALUES "
        "('e1', 2.6, 1.4, -0.5, 0.5, 0.3), "  # real AH line -> included
        "('e2', 2.6, 1.4, NULL, 0.5, 0.3)"  # missing AH line -> excluded
    )

    class _FakeDb:
        @contextlib.contextmanager
        def connection(self, read_only=False):
            yield conn

    rows_by_league = script._fetch_rows_by_league(_FakeDb())

    assert len(rows_by_league) == 1
    assert "E0" in rows_by_league
    assert len(rows_by_league["E0"]) == 1
    lambda_total, lambda_home, implied_home, implied_away = rows_by_league["E0"][0]
    assert lambda_total == pytest.approx(2.6, abs=1e-6)
    assert lambda_home == pytest.approx(1.4, abs=1e-6)
    assert implied_home == pytest.approx(0.5, abs=1e-6)
    assert implied_away == pytest.approx(0.3, abs=1e-6)
