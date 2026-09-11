"""W103: pre-generates recommendations for real upcoming fixtures right
now, reusing eod_batch.py's own bounded-concurrency/per-match-error-
isolation machinery (run_eod_batch's `fixtures=` override) instead of a
separate mechanism. Two callers: the admin trigger endpoint, and
lifespan's own startup hook (gated behind ENABLE_SCHEDULER, same as the
scheduler registration itself -- see main.py's own comment for why an
unconditional hook would break every test's TestClient(app) boot). That
lifespan-level wiring is deliberately NOT re-tested here via a full app
boot -- W08/W09's own scheduler registration has no such test either,
for the identical non-determinism reason; this file covers the
deterministic logic pieces only."""

from __future__ import annotations

from pathlib import Path
import sys
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

sys.path.append(str(Path(__file__).resolve().parents[3]))

from fastapi.testclient import TestClient

from app.backend import main
from app.backend.eod_batch import EodBatchResult
from app.backend.football_data_client import NormalizedMatch


def _fixture(match_id: str, competition: str) -> NormalizedMatch:
    return NormalizedMatch(
        match_id=match_id, utc_date="2026-08-15T15:00:00Z", status="SCHEDULED",
        home_team="Home", away_team="Away", home_goals=None, away_goals=None, competition=competition,
    )


def test_default_days_ahead_is_3_not_5():
    """W165: the automatic boot-time default shrank from 5 -> 3, sized to
    match the frontend's own "next 10 upcoming fixtures" display cap
    (MatchUI.tsx) rather than the memory-safety-only value it started as --
    a fixed constant, not logic, but a real behavior change worth a
    regression check against silent reverts. The admin endpoint's own
    explicit-override tests below are unaffected (they never rely on this
    default)."""
    assert main._PREGENERATE_DEFAULT_DAYS_AHEAD == 3


def test_pregenerate_recently_ran_false_when_cache_empty(tmp_path):
    from app.backend.recommendation_cache import RecommendationCache

    cache = RecommendationCache(db_path=tmp_path / "cache.db")
    assert main._pregenerate_recently_ran(cache) is False


def test_pregenerate_recently_ran_true_within_cooldown(tmp_path):
    from datetime import datetime, timedelta, timezone

    from app.backend.recommendation_cache import RecommendationCache

    cache = RecommendationCache(db_path=tmp_path / "cache.db")
    recent = (datetime.now(timezone.utc) - timedelta(minutes=5)).isoformat()
    cache.record_generation(
        match_id="m1", date="2026-08-22", agent_config_hash="cfg-hash",
        odds={}, recommendation={}, triggered_by="scheduled", generated_at=recent,
    )
    assert main._pregenerate_recently_ran(cache) is True


def test_pregenerate_recently_ran_false_once_cooldown_elapsed(tmp_path):
    from datetime import datetime, timedelta, timezone

    from app.backend.recommendation_cache import RecommendationCache

    cache = RecommendationCache(db_path=tmp_path / "cache.db")
    stale = (datetime.now(timezone.utc) - timedelta(minutes=25)).isoformat()
    cache.record_generation(
        match_id="m1", date="2026-08-22", agent_config_hash="cfg-hash",
        odds={}, recommendation={}, triggered_by="scheduled", generated_at=stale,
    )
    assert main._pregenerate_recently_ran(cache) is False


def test_groups_fixtures_by_league_and_runs_one_batch_per_league(monkeypatch):
    monkeypatch.delenv("APP_ACCESS_TOKEN", raising=False)
    fixtures = [_fixture("m1", "E0"), _fixture("m2", "E0"), _fixture("m3", "SWE")]
    batch_calls = []

    async def _fake_run_eod_batch(**kwargs):
        batch_calls.append(kwargs)
        return EodBatchResult(fixtures=kwargs["fixtures"], generated=len(kwargs["fixtures"]), skipped=0)

    with patch("app.backend.main.get_fixtures", new=AsyncMock(return_value=fixtures)), \
         patch("app.backend.main.build_odds_client", return_value=None), \
         patch("app.backend.main.recommendations.get_cache", return_value=MagicMock()), \
         patch("app.backend.eod_batch.run_eod_batch", side_effect=_fake_run_eod_batch):
        import asyncio
        results = asyncio.run(main._pregenerate_recommendations(days_ahead=5, scheduler=None))

    assert len(batch_calls) == 2  # one per league, not one per fixture
    called_leagues = {call["league"] for call in batch_calls}
    assert called_leagues == {"E0", "SWE"}
    e0_call = next(c for c in batch_calls if c["league"] == "E0")
    assert len(e0_call["fixtures"]) == 2
    assert results["E0"] == {"generated": 2, "skipped": 0, "unchanged": 0}
    assert results["SWE"] == {"generated": 1, "skipped": 0, "unchanged": 0}


def test_force_flag_reaches_run_eod_batch(monkeypatch):
    """W204: direct user request -- an admin-triggered force-refresh must
    actually reach run_eod_batch's own force param (already_fresh() bypass),
    not just exist as an unused kwarg."""
    monkeypatch.delenv("APP_ACCESS_TOKEN", raising=False)
    fixtures = [_fixture("m1", "E0")]
    captured = {}

    async def _fake_run_eod_batch(**kwargs):
        captured["force"] = kwargs["force"]
        return EodBatchResult(fixtures=kwargs["fixtures"], generated=1, skipped=0)

    with patch("app.backend.main.get_fixtures", new=AsyncMock(return_value=fixtures)), \
         patch("app.backend.main.build_odds_client", return_value=None), \
         patch("app.backend.main.recommendations.get_cache", return_value=MagicMock()), \
         patch("app.backend.eod_batch.run_eod_batch", side_effect=_fake_run_eod_batch):
        import asyncio
        asyncio.run(main._pregenerate_recommendations(days_ahead=5, scheduler=None, force=True))

    assert captured["force"] is True


def test_force_defaults_to_false_preserving_existing_behavior(monkeypatch):
    monkeypatch.delenv("APP_ACCESS_TOKEN", raising=False)
    fixtures = [_fixture("m1", "E0")]
    captured = {}

    async def _fake_run_eod_batch(**kwargs):
        captured["force"] = kwargs["force"]
        return EodBatchResult(fixtures=kwargs["fixtures"], generated=1, skipped=0)

    with patch("app.backend.main.get_fixtures", new=AsyncMock(return_value=fixtures)), \
         patch("app.backend.main.build_odds_client", return_value=None), \
         patch("app.backend.main.recommendations.get_cache", return_value=MagicMock()), \
         patch("app.backend.eod_batch.run_eod_batch", side_effect=_fake_run_eod_batch):
        import asyncio
        asyncio.run(main._pregenerate_recommendations(days_ahead=5, scheduler=None))

    assert captured["force"] is False


def test_explicit_date_range_overrides_days_ahead(monkeypatch):
    """W204: date_from/date_to (both given) target any window, not just
    "today onward" -- e.g. backfilling a specific day a prior pass failed
    on (the football-data.org 429 scenario this session hit live)."""
    monkeypatch.delenv("APP_ACCESS_TOKEN", raising=False)
    captured = {}

    async def _fake_get_fixtures(date_from, date_to):
        captured["date_from"] = date_from
        captured["date_to"] = date_to
        return [_fixture("m1", "E0")]

    with patch("app.backend.main.get_fixtures", new=AsyncMock(side_effect=_fake_get_fixtures)), \
         patch("app.backend.main.build_odds_client", return_value=None), \
         patch("app.backend.main.recommendations.get_cache", return_value=MagicMock()), \
         patch("app.backend.eod_batch.run_eod_batch", new=AsyncMock(
             return_value=EodBatchResult(fixtures=[], generated=0, skipped=0)
         )):
        import asyncio
        asyncio.run(main._pregenerate_recommendations(
            days_ahead=5, scheduler=None, date_from="2026-09-05", date_to="2026-09-06",
        ))

    # days_ahead=5 would compute a completely different range -- confirms
    # the explicit override actually won, not just that the call succeeded.
    assert captured == {"date_from": "2026-09-05", "date_to": "2026-09-06"}


def test_one_sided_date_range_falls_back_to_days_ahead(monkeypatch):
    """Only date_from OR only date_to (not both) is treated as no override --
    a half-open range can't be resolved unambiguously against days_ahead,
    same "both required together" contract as main.py's own
    _split_fixture_date_range precedent elsewhere in this file."""
    monkeypatch.delenv("APP_ACCESS_TOKEN", raising=False)
    captured = {}

    async def _fake_get_fixtures(date_from, date_to):
        captured["date_from"] = date_from
        captured["date_to"] = date_to
        return []

    with patch("app.backend.main.get_fixtures", new=AsyncMock(side_effect=_fake_get_fixtures)), \
         patch("app.backend.main.build_odds_client", return_value=None), \
         patch("app.backend.main.recommendations.get_cache", return_value=MagicMock()):
        import asyncio
        asyncio.run(main._pregenerate_recommendations(days_ahead=5, scheduler=None, date_from="2026-09-05"))

    assert captured["date_from"] != "2026-09-05"


def test_league_filter_narrows_to_one_competition(monkeypatch):
    """W204: direct user request -- restrict a refresh to one league,
    without generating for every other enabled one."""
    monkeypatch.delenv("APP_ACCESS_TOKEN", raising=False)
    fixtures = [_fixture("m1", "E0"), _fixture("m2", "SWE")]
    batch_calls = []

    async def _fake_run_eod_batch(**kwargs):
        batch_calls.append(kwargs)
        return EodBatchResult(fixtures=kwargs["fixtures"], generated=len(kwargs["fixtures"]), skipped=0)

    with patch("app.backend.main.get_fixtures", new=AsyncMock(return_value=fixtures)), \
         patch("app.backend.main.build_odds_client", return_value=None), \
         patch("app.backend.main.recommendations.get_cache", return_value=MagicMock()), \
         patch("app.backend.eod_batch.run_eod_batch", side_effect=_fake_run_eod_batch):
        import asyncio
        results = asyncio.run(main._pregenerate_recommendations(days_ahead=5, scheduler=None, league_filter="E0"))

    assert len(batch_calls) == 1
    assert batch_calls[0]["league"] == "E0"
    assert "SWE" not in results


def test_league_filter_for_a_league_with_no_fixtures_in_range_is_a_no_op(monkeypatch):
    monkeypatch.delenv("APP_ACCESS_TOKEN", raising=False)
    fixtures = [_fixture("m1", "E0")]

    with patch("app.backend.main.get_fixtures", new=AsyncMock(return_value=fixtures)), \
         patch("app.backend.main.build_odds_client", return_value=None), \
         patch("app.backend.main.recommendations.get_cache", return_value=MagicMock()), \
         patch("app.backend.eod_batch.run_eod_batch") as mock_batch:
        import asyncio
        results = asyncio.run(main._pregenerate_recommendations(days_ahead=5, scheduler=None, league_filter="SWE"))

    mock_batch.assert_not_called()
    assert results == {}


def test_bug_045_defaults_to_a_lower_concurrency_than_eod_batchs_own_default(monkeypatch):
    """Confirmed live (2026-08-13): boot-time pregenerate stacking
    eod_batch's own default concurrency=5 directly on top of a freshly
    booted process's import-time memory OOM-crashed the deployed instance
    (Railway: "Deploy Ran Out Of Memory", 4GB limit) -- and since pregenerate
    re-fires on every boot, one OOM became an infinite crash loop. Asserts
    the safer default actually reaches run_eod_batch, not just that a
    `_PREGENERATE_DEFAULT_CONCURRENCY` constant exists somewhere unused."""
    monkeypatch.delenv("APP_ACCESS_TOKEN", raising=False)
    fixtures = [_fixture("m1", "E0")]
    captured = {}

    async def _fake_run_eod_batch(**kwargs):
        captured["concurrency"] = kwargs["concurrency"]
        return EodBatchResult(fixtures=kwargs["fixtures"], generated=1, skipped=0)

    with patch("app.backend.main.get_fixtures", new=AsyncMock(return_value=fixtures)), \
         patch("app.backend.main.build_odds_client", return_value=None), \
         patch("app.backend.main.recommendations.get_cache", return_value=MagicMock()), \
         patch("app.backend.eod_batch.run_eod_batch", side_effect=_fake_run_eod_batch):
        import asyncio
        asyncio.run(main._pregenerate_recommendations(days_ahead=5, scheduler=None))

    assert captured["concurrency"] == main._PREGENERATE_DEFAULT_CONCURRENCY
    assert captured["concurrency"] < 5  # eod_batch.run_eod_batch()'s own default


def test_degrades_to_a_no_op_schedule_t30_when_no_scheduler_is_running(monkeypatch):
    """scheduler=None (ENABLE_SCHEDULER off, or a manual trigger before
    it's ever been turned on) must still generate and cache -- just
    without a T-30 refresh registered for the pregenerated matches."""
    monkeypatch.delenv("APP_ACCESS_TOKEN", raising=False)
    fixtures = [_fixture("m1", "E0")]
    captured_schedule_t30 = {}

    async def _fake_run_eod_batch(**kwargs):
        captured_schedule_t30["fn"] = kwargs["schedule_t30"]
        return EodBatchResult(fixtures=kwargs["fixtures"], generated=1, skipped=0)

    with patch("app.backend.main.get_fixtures", new=AsyncMock(return_value=fixtures)), \
         patch("app.backend.main.build_odds_client", return_value=None), \
         patch("app.backend.main.recommendations.get_cache", return_value=MagicMock()), \
         patch("app.backend.eod_batch.run_eod_batch", side_effect=_fake_run_eod_batch):
        import asyncio
        asyncio.run(main._pregenerate_recommendations(days_ahead=5, scheduler=None))

    # calling the no-op schedule_t30 with a real fixture must not raise
    captured_schedule_t30["fn"](fixtures[0])


def test_a_scheduler_present_gets_a_real_schedule_t30_wired_through(monkeypatch):
    monkeypatch.delenv("APP_ACCESS_TOKEN", raising=False)
    fixtures = [_fixture("m1", "E0")]
    fake_scheduler = MagicMock()

    async def _fake_run_eod_batch(**kwargs):
        # a real schedule_t30 (from build_schedule_t30) is a closure, not
        # our no-op lambda -- calling it should reach the real scheduler.
        kwargs["schedule_t30"](fixtures[0])
        return EodBatchResult(fixtures=kwargs["fixtures"], generated=1, skipped=0)

    with patch("app.backend.main.get_fixtures", new=AsyncMock(return_value=fixtures)), \
         patch("app.backend.main.build_odds_client", return_value=None), \
         patch("app.backend.main.recommendations.get_cache", return_value=MagicMock()), \
         patch("app.backend.eod_batch.run_eod_batch", side_effect=_fake_run_eod_batch):
        import asyncio
        asyncio.run(main._pregenerate_recommendations(days_ahead=5, scheduler=fake_scheduler))

    fake_scheduler.schedule_once.assert_called_once()  # build_schedule_t30's own real behavior


def test_one_leagues_batch_failure_does_not_block_the_others(monkeypatch):
    monkeypatch.delenv("APP_ACCESS_TOKEN", raising=False)
    fixtures = [_fixture("m1", "E0"), _fixture("m2", "SWE")]

    async def _fake_run_eod_batch(**kwargs):
        if kwargs["league"] == "E0":
            raise RuntimeError("boom")
        return EodBatchResult(fixtures=kwargs["fixtures"], generated=1, skipped=0)

    with patch("app.backend.main.get_fixtures", new=AsyncMock(return_value=fixtures)), \
         patch("app.backend.main.build_odds_client", return_value=None), \
         patch("app.backend.main.recommendations.get_cache", return_value=MagicMock()), \
         patch("app.backend.eod_batch.run_eod_batch", side_effect=_fake_run_eod_batch):
        import asyncio
        results = asyncio.run(main._pregenerate_recommendations(days_ahead=5, scheduler=None))

    assert "E0" not in results
    assert results["SWE"] == {"generated": 1, "skipped": 0, "unchanged": 0}


def test_get_fixtures_failure_skips_the_whole_pass_instead_of_crashing_unretrieved(monkeypatch):
    """W179 regression: found live (2026-08-27) -- get_fixtures() raises
    HTTPException(503) on an upstream 429/outage, the right contract for its
    HTTP-route caller but fatal here: this runs as a fire-and-forget
    background task with no request to turn that into a response, so it
    surfaced only as an unhandled "Task exception was never retrieved"."""
    monkeypatch.delenv("APP_ACCESS_TOKEN", raising=False)
    from fastapi import HTTPException

    with patch(
        "app.backend.main.get_fixtures",
        new=AsyncMock(side_effect=HTTPException(status_code=503, detail="Fixture data is temporarily unavailable")),
    ), patch("app.backend.eod_batch.run_eod_batch") as mock_run_eod_batch:
        import asyncio
        results = asyncio.run(main._pregenerate_recommendations(days_ahead=5, scheduler=None))

    assert results == {}
    mock_run_eod_batch.assert_not_called()


def test_admin_endpoint_returns_immediately_without_waiting_for_pregenerate(monkeypatch):
    monkeypatch.delenv("APP_ACCESS_TOKEN", raising=False)
    # _fire_and_forget itself deliberately left real here (not mocked) --
    # it genuinely schedules the mocked-but-still-a-coroutine
    # _pregenerate_recommendations() call as a real asyncio.Task, which the
    # event loop properly consumes; mocking _fire_and_forget too would
    # leave that coroutine object dangling, unawaited (patch() auto-detects
    # _pregenerate_recommendations is `async def` and returns an AsyncMock
    # accordingly, so calling it always produces a real coroutine to
    # schedule, mocked or not).
    with patch("app.backend.main._pregenerate_recommendations", return_value=MagicMock()) as mock_pregenerate:
        with TestClient(main.app) as client:
            response = client.post("/api/admin/pregenerate-recommendations", params={"days_ahead": 7})

    assert response.status_code == 200
    assert response.json() == {
        "days_ahead": 7, "date_from": None, "date_to": None, "league": None, "force": False, "status": "started",
    }
    mock_pregenerate.assert_called_once_with(
        days_ahead=7, scheduler=None, date_from=None, date_to=None, league_filter=None, force=False,
    )


def test_fire_and_forget_keeps_a_reference_until_the_task_completes():
    """Regression guard for asyncio.create_task()'s own documented
    gotcha -- a task with no other reference can be garbage-collected
    mid-execution."""
    import asyncio

    async def _tiny_coro():
        return "done"

    async def _run():
        assert len(main._background_tasks) == 0
        main._fire_and_forget(_tiny_coro())
        assert len(main._background_tasks) == 1
        await asyncio.sleep(0)  # let the task run to completion
        await asyncio.sleep(0)  # let the done-callback fire
        assert len(main._background_tasks) == 0

    asyncio.run(_run())
