"""Tests for _resolve_probe_row (US#190 follow-up), the match-selection query
behind _current_computable_features()'s BUG-012 layer 3c live-computability
guard.

Found live: a real select-best-models run for SP1/I1/D1/F1 refused to
promote every corner/goal target, all citing the same 6 SQUAD-gated
features as "not computable by the live feature pipeline" -- even though
FeatureFactory.build_for_match() was independently confirmed (this same
session) to compute all 6 correctly for real SP1/I1/D1/F1 fixtures. Root
cause: the probe match was chosen as the single most-recent row across
*all* of raw_matches with no league filter at all, and Sweden's calendar
(2026-07-20) happened to run later than any "big five" league's -- so
every context's promotion check was silently validated against a
Swedish match that structurally never has FotMob player data, a false
negative with nothing to do with the context actually being evaluated.
"""

from __future__ import annotations

from pathlib import Path
import sys

import duckdb
import pytest

sys.path.append(str(Path(__file__).resolve().parents[1]))

from main import _resolve_probe_row


def _make_raw_matches_db(tmp_path: Path) -> duckdb.DuckDBPyConnection:
    conn = duckdb.connect(str(tmp_path / "test.db"))
    conn.execute(
        "CREATE TABLE raw_matches (match_id VARCHAR, league VARCHAR, date TIMESTAMP, "
        "home_team VARCHAR, away_team VARCHAR, odds_h FLOAT)"
    )
    # SWE's own most recent match is later than SP1's -- exactly the real
    # shape found live.
    conn.execute("INSERT INTO raw_matches VALUES ('m1', 'SP1', '2026-05-24', 'A', 'B', 2.0)")
    conn.execute("INSERT INTO raw_matches VALUES ('m2', 'SWE', '2026-07-20', 'C', 'D', 2.0)")
    return conn


def test_context_none_falls_back_to_unscoped_global_most_recent(tmp_path: Path):
    """Unchanged legacy behavior for the bare 'all contexts' CLI invocation."""
    conn = _make_raw_matches_db(tmp_path)
    row = _resolve_probe_row(conn, context=None)
    assert row[3] == "SWE"  # the real global-most-recent row, unscoped


def test_context_scopes_probe_to_that_context_own_league(tmp_path: Path):
    """The actual bug fix: SP1's own promotion check must probe SP1's own
    most recent match, never a different competition's, even one that
    happens to be globally more recent."""
    conn = _make_raw_matches_db(tmp_path)
    row = _resolve_probe_row(conn, context="SP1")
    assert row[3] == "SP1"


def test_international_context_has_no_single_league_falls_back_unscoped(tmp_path: Path):
    """'international' has league_code=None in the registry (pools across
    competitions by design, per US#138) -- there's no single league to
    scope by, so this stays the pre-existing unscoped behavior."""
    conn = _make_raw_matches_db(tmp_path)
    row = _resolve_probe_row(conn, context="international")
    assert row[3] == "SWE"


def test_unknown_context_falls_back_unscoped_rather_than_crashing(tmp_path: Path):
    conn = _make_raw_matches_db(tmp_path)
    row = _resolve_probe_row(conn, context="NOT_A_REAL_CONTEXT")
    assert row[3] == "SWE"
