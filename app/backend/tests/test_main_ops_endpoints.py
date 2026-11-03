"""Tests for the /api/admin/ops/* endpoints -- the four-tab /ops dashboard's
backend. 2026-11-02 design:
docs/superpowers/specs/2026-11-02-background-job-safety-net-design.md"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

from fastapi.testclient import TestClient

from app.backend import recommendations
from app.backend.main import app
from app.backend.run_isolated import get_failures_store


def test_get_ops_failures_returns_recorded_failures_and_breaker_status(tmp_path: Path) -> None:
    # conftest.py's autouse _default_failures_store fixture already points
    # run_isolated's DEFAULT_DB_PATH at a tmp_path and resets the singleton
    # before this test body runs -- no need to redo that here.
    get_failures_store().record(
        subsystem="t30", unit_id="match_id=558542", exc=ValueError("vendor exploded"),
        alerted=True, league="I1",
    )

    with TestClient(app) as client:
        response = client.get("/api/admin/ops/failures")

    assert response.status_code == 200
    body = response.json()
    assert len(body["failures"]) == 1
    assert body["failures"][0]["subsystem"] == "t30"
    assert body["failures"][0]["league"] == "I1"
    assert set(body["circuit_breakers"].keys()) == {"the_odds_api", "oddspapi"}


def test_get_ops_failures_filters_by_league(tmp_path: Path) -> None:
    store = get_failures_store()
    store.record(subsystem="t30", unit_id="match_id=1", exc=ValueError("a"), alerted=False, league="E0")
    store.record(subsystem="t30", unit_id="match_id=2", exc=ValueError("b"), alerted=False, league="I1")

    with TestClient(app) as client:
        response = client.get("/api/admin/ops/failures", params={"league": "I1"})

    body = response.json()
    assert len(body["failures"]) == 1
    assert body["failures"][0]["unit_id"] == "match_id=2"


def test_get_ops_failures_filters_by_since(tmp_path: Path) -> None:
    get_failures_store().record(
        subsystem="t30", unit_id="match_id=1", exc=ValueError("a"), alerted=False, league="E0",
    )

    with TestClient(app) as client:
        response = client.get("/api/admin/ops/failures", params={"since": "2099-01-01T00:00:00+00:00"})

    assert response.json()["failures"] == []


def test_get_ops_odds_availability_reports_missing_markets_with_cause(tmp_path: Path) -> None:
    from app.backend.recommendation_cache import CacheEntry

    entry = CacheEntry(
        match_id="558542", date="2026-11-02", agent_config_hash="abc",
        odds={"home": 2.1, "draw": 3.4, "away": 3.6, "total_goals": None, "asian_handicap": None, "corners": None},
        recommendation={"match": {"home": "Udinese", "away": "Roma", "date": "2026-11-02", "league": "I1"}},
        generated_at="2026-11-02T10:00:00+00:00", triggered_by="scheduled",
    )
    mock_cache = MagicMock()
    mock_cache.list_latest_per_match.return_value = [entry]

    # NOTE on patch target: main.py imports get_circuit_breaker_status via
    # `from app.backend.scheduler_wiring import (..., get_circuit_breaker_status)`,
    # which binds the name directly into main's own module namespace at
    # import time. The new endpoint below calls the bare name
    # `get_circuit_breaker_status()`, which resolves through main's
    # namespace, not scheduler_wiring's -- so patching
    # `app.backend.scheduler_wiring.get_circuit_breaker_status` would leave
    # main's already-bound reference untouched. Patch where it's USED:
    # `app.backend.main.get_circuit_breaker_status`.
    with patch("app.backend.main.get_circuit_breaker_status", return_value={
        "the_odds_api": {"state": "open"}, "oddspapi": {"state": "closed"},
    }):
        with TestClient(app) as client:
            app.dependency_overrides[recommendations.get_cache] = lambda: mock_cache
            try:
                response = client.get(
                    "/api/admin/ops/odds-availability",
                    params={"date_from": "2026-11-02", "date_to": "2026-11-02"},
                )
            finally:
                app.dependency_overrides.pop(recommendations.get_cache, None)

    assert response.status_code == 200
    body = response.json()
    assert len(body["fixtures"]) == 1
    row = body["fixtures"][0]
    assert row["match_id"] == "558542"
    assert row["league"] == "I1"
    assert "over/under 2.5" in row["missing_markets"]
    assert row["likely_cause"] == "circuit_open"


def test_odds_availability_trend_buckets_by_week_and_ranks_markets(tmp_path: Path) -> None:
    from app.backend.recommendation_cache import CacheEntry

    complete_odds = {
        "home": 2.0, "draw": 3.0, "away": 3.5,
        "total_goals": {"over_2.5": 1.9},
        "asian_handicap": {"line": -0.5, "home": 1.9, "away": 1.9},
        "corners": {"over_9.5": 1.9},
    }
    missing_corners_odds = {**complete_odds, "corners": None}

    entries = [
        # Week of 2026-10-08 (Thu-Wed): 2 fixtures, one missing total_corners.
        CacheEntry(
            match_id="m1", date="2026-10-09", agent_config_hash="h",
            odds=missing_corners_odds, recommendation={"match": {"league": "E0", "home": "A", "away": "B"}},
            generated_at="2026-10-09T10:00:00+00:00", triggered_by="scheduled",
        ),
        CacheEntry(
            match_id="m2", date="2026-10-10", agent_config_hash="h",
            odds=complete_odds, recommendation={"match": {"league": "E0", "home": "C", "away": "D"}},
            generated_at="2026-10-10T10:00:00+00:00", triggered_by="scheduled",
        ),
        # Week of 2026-10-15: 1 fixture, missing total_corners too.
        CacheEntry(
            match_id="m3", date="2026-10-16", agent_config_hash="h",
            odds=missing_corners_odds, recommendation={"match": {"league": "E0", "home": "E", "away": "F"}},
            generated_at="2026-10-16T10:00:00+00:00", triggered_by="scheduled",
        ),
    ]
    mock_cache = MagicMock()
    mock_cache.list_latest_per_match.return_value = entries

    with patch("app.backend.main.get_circuit_breaker_status", return_value={
        "the_odds_api": {"state": "closed"}, "oddspapi": {"state": "closed"},
    }):
        with TestClient(app) as client:
            app.dependency_overrides[recommendations.get_cache] = lambda: mock_cache
            try:
                response = client.get(
                    "/api/admin/ops/odds-availability-trend", params={"weeks": 2, "today": "2026-10-15"},
                )
            finally:
                app.dependency_overrides.pop(recommendations.get_cache, None)

    assert response.status_code == 200
    body = response.json()
    assert [w["week_start"] for w in body["weeks"]] == ["2026-10-08", "2026-10-15"]
    week1 = body["weeks"][0]
    assert week1["total_fixtures"] == 2
    assert week1["by_market"]["total_corners"]["missing"] == 1
    assert week1["by_market"]["total_corners"]["not_posted"] == 1
    assert week1["by_market"]["total_corners"]["circuit_open"] == 0
    week2 = body["weeks"][1]
    assert week2["total_fixtures"] == 1
    assert week2["by_market"]["total_corners"]["missing"] == 1


def test_odds_availability_trend_cause_attribution_matches_fixture_level_rule(tmp_path: Path) -> None:
    """2026-11-03: the real get_ops_odds_availability rule computes ONE
    likely_cause per fixture (total_corners+oddspapi checked first, then
    any the_odds_api-backed market), not an independent check per market.
    With oddspapi closed and the_odds_api open, a fixture missing BOTH
    total_corners and 1X2 gets likely_cause="circuit_open" for the whole
    fixture (via the elif branch) -- so total_corners must be counted
    circuit_open here too, even though its own vendor breaker (oddspapi) is
    closed. A naive "total_corners checks oddspapi, everything else checks
    the_odds_api" per-market rule would wrongly call it not_posted."""
    from app.backend.recommendation_cache import CacheEntry

    entry = CacheEntry(
        match_id="m1", date="2026-10-09", agent_config_hash="h",
        odds={"home": None, "draw": None, "away": None, "total_goals": {"over_2.5": 1.9},
              "asian_handicap": {"line": -0.5, "home": 1.9, "away": 1.9}, "corners": None},
        recommendation={"match": {"league": "E0", "home": "A", "away": "B"}},
        generated_at="2026-10-09T10:00:00+00:00", triggered_by="scheduled",
    )
    mock_cache = MagicMock()
    mock_cache.list_latest_per_match.return_value = [entry]

    with patch("app.backend.main.get_circuit_breaker_status", return_value={
        "the_odds_api": {"state": "open"}, "oddspapi": {"state": "closed"},
    }):
        with TestClient(app) as client:
            app.dependency_overrides[recommendations.get_cache] = lambda: mock_cache
            try:
                response = client.get(
                    "/api/admin/ops/odds-availability-trend", params={"weeks": 1, "today": "2026-10-09"},
                )
            finally:
                app.dependency_overrides.pop(recommendations.get_cache, None)

    body = response.json()
    by_market = body["weeks"][0]["by_market"]
    assert by_market["1X2"]["circuit_open"] == 1
    assert by_market["total_corners"]["circuit_open"] == 1
    assert by_market["total_corners"]["not_posted"] == 0


def test_get_ops_feature_health_aggregates_by_league(tmp_path: Path) -> None:
    from unittest.mock import MagicMock, patch
    from app.backend.recommendation_cache import CacheEntry

    entries = [
        CacheEntry(
            match_id="1", date="2026-11-02", agent_config_hash="a",
            odds={}, recommendation={"match": {"league": "I1"}},
            generated_at="2026-11-02T10:00:00+00:00", triggered_by="scheduled",
            forecast_payload={"diagnostics": {"cold_start_risk": True, "feature_completeness": 0.4}, "data_quality": {"unknown_team": False}},
        ),
        CacheEntry(
            match_id="2", date="2026-11-02", agent_config_hash="a",
            odds={}, recommendation={"match": {"league": "I1"}},
            generated_at="2026-11-02T11:00:00+00:00", triggered_by="scheduled",
            forecast_payload={"diagnostics": {"cold_start_risk": False, "feature_completeness": 0.9}, "data_quality": {"unknown_team": False}},
        ),
    ]
    mock_cache = MagicMock()
    mock_cache.list_latest_per_match.return_value = entries

    with TestClient(app) as client:
        app.dependency_overrides[recommendations.get_cache] = lambda: mock_cache
        try:
            response = client.get("/api/admin/ops/feature-health", params={"date_from": "2026-11-02", "date_to": "2026-11-02"})
        finally:
            app.dependency_overrides.pop(recommendations.get_cache, None)

    body = response.json()
    assert body["by_league"]["I1"]["sample_size"] == 2
    assert body["by_league"]["I1"]["cold_start_risk_rate"] == 0.5
    assert body["by_league"]["I1"]["avg_feature_completeness"] == 0.65
    assert body["by_league"]["I1"]["unknown_team_rate"] == 0.0


def test_get_ops_feature_health_averages_only_entries_with_a_reported_completeness(tmp_path: Path) -> None:
    """2026-11-02: avg_feature_completeness used to divide by sample_size
    (every entry in the league) instead of only entries that actually
    reported a non-None completeness -- an entry with a failed/errored
    forecast (feature_completeness=None) silently dragged the average
    toward zero instead of being excluded."""
    from unittest.mock import MagicMock
    from app.backend.recommendation_cache import CacheEntry

    entries = [
        CacheEntry(
            match_id="1", date="2026-11-02", agent_config_hash="a",
            odds={}, recommendation={"match": {"league": "I1"}},
            generated_at="2026-11-02T10:00:00+00:00", triggered_by="scheduled",
            forecast_payload={"diagnostics": {"cold_start_risk": False, "feature_completeness": 0.4}, "data_quality": {"unknown_team": False}},
        ),
        CacheEntry(
            match_id="2", date="2026-11-02", agent_config_hash="a",
            odds={}, recommendation={"match": {"league": "I1"}},
            generated_at="2026-11-02T11:00:00+00:00", triggered_by="scheduled",
            forecast_payload={"diagnostics": {"cold_start_risk": False, "feature_completeness": None}, "data_quality": {"unknown_team": False}},
        ),
    ]
    mock_cache = MagicMock()
    mock_cache.list_latest_per_match.return_value = entries

    with TestClient(app) as client:
        app.dependency_overrides[recommendations.get_cache] = lambda: mock_cache
        try:
            response = client.get("/api/admin/ops/feature-health", params={"date_from": "2026-11-02", "date_to": "2026-11-02"})
        finally:
            app.dependency_overrides.pop(recommendations.get_cache, None)

    body = response.json()
    # sample_size still counts both entries (both are real, cached matches)
    assert body["by_league"]["I1"]["sample_size"] == 2
    # but the average is over the ONE entry that actually reported a value, not 0.4/2=0.2
    assert body["by_league"]["I1"]["avg_feature_completeness"] == 0.4


def test_get_ops_feature_health_avg_completeness_is_none_when_no_entry_reported_one(tmp_path: Path) -> None:
    from unittest.mock import MagicMock
    from app.backend.recommendation_cache import CacheEntry

    entry = CacheEntry(
        match_id="1", date="2026-11-02", agent_config_hash="a",
        odds={}, recommendation={"match": {"league": "I1"}},
        generated_at="2026-11-02T10:00:00+00:00", triggered_by="scheduled",
        forecast_payload={"diagnostics": {"cold_start_risk": False, "feature_completeness": None}, "data_quality": {"unknown_team": False}},
    )
    mock_cache = MagicMock()
    mock_cache.list_latest_per_match.return_value = [entry]

    with TestClient(app) as client:
        app.dependency_overrides[recommendations.get_cache] = lambda: mock_cache
        try:
            response = client.get("/api/admin/ops/feature-health", params={"date_from": "2026-11-02", "date_to": "2026-11-02"})
        finally:
            app.dependency_overrides.pop(recommendations.get_cache, None)

    body = response.json()
    assert body["by_league"]["I1"]["sample_size"] == 1
    assert body["by_league"]["I1"]["avg_feature_completeness"] is None
