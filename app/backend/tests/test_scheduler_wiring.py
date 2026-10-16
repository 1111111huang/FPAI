"""W08/W09/W10 wiring: connects RecoverableScheduler (W08) to the EOD batch
job (W09) and each fixture's T-30 job (W10). Focused on the pure/testable
seams -- next_day_date_str's NY-timezone date math, t30_run_at's kickoff-
minus-30-minutes parsing, and PersistingOddsClient's credit-counter
persistence (W07's OddsAPIClient/CreditCounter/FileCreditCounterStore
trio leaves persistence to the caller by design)."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from pathlib import Path
import sys
import time
from unittest.mock import MagicMock, patch

sys.path.append(str(Path(__file__).resolve().parents[3]))

import pytest
import requests

from app.backend import fixture_cache
from app.backend.football_data_client import NormalizedMatch
from app.backend.odds_api_client import CreditCounter, FileCreditCounterStore
from app.backend.push_subscriptions import PushSubscriptionStore
from app.backend.recommendation_cache import RecommendationCache
from app.backend.recommendation_outcomes import RecommendationOutcomeStore
from app.backend.scheduler import NY_TZ, JobRunLog, RecoverableScheduler
from app.backend.scheduler_wiring import (
    DATA_REFRESH_HOUR,
    DATA_REFRESH_JOB_ID,
    DATA_REFRESH_MINUTE,
    EOD_HOUR,
    EOD_JOB_ID,
    FallbackOddsClient,
    FallbackOddsPapiClient,
    LESSONS_HOUR,
    LESSONS_JOB_ID,
    LESSONS_WEEKLY_DAY_OF_WEEK,
    LESSONS_WEEKLY_HOUR,
    LESSONS_WEEKLY_JOB_ID,
    LESSONS_WEEKLY_MINUTE,
    RECOMMENDATION_CACHE_PRUNE_DAY_OF_WEEK,
    RECOMMENDATION_CACHE_PRUNE_HOUR,
    RECOMMENDATION_CACHE_PRUNE_JOB_ID,
    RECOMMENDATION_CACHE_PRUNE_MINUTE,
    RECOMMENDATION_CACHE_RETENTION_DAYS,
    RECONCILIATION_HOURS,
    RECONCILIATION_MINUTE,
    PersistingOddsClient,
    PersistingOddsPapiClient,
    build_batch_notification_body,
    build_schedule_t30,
    cluster_fixtures_by_kickoff,
    next_day_date_str,
    register_data_refresh_job,
    register_eod_job,
    register_fixture_reconciliation_job,
    register_lessons_job,
    register_recommendation_cache_prune_job,
    t30_run_at,
)
from src.agent.agent_config import AgentConfig
from src.utils.db_manager import DuckDBManager

# eod_batch.has_kicked_off() (reused by run_eod_batch, called under the hood
# by register_eod_job below) compares each fixture's utc_date against real
# wall-clock time (sandbox_now()) -- independent of the `now_fn` these tests
# inject into RecoverableScheduler for its own catch-up-trigger math. A fixed
# calendar date rots the moment real time passes it, so every fixture/`now`
# pair below is anchored to this real-future day instead.
_FUTURE_DAY = (datetime.now(timezone.utc) + timedelta(days=2)).date()
_FUTURE_DAY_STR = _FUTURE_DAY.isoformat()
_FUTURE_DAY_PLUS_1_STR = (_FUTURE_DAY + timedelta(days=1)).isoformat()


def _wait_until(predicate, timeout: float = 2.0, interval: float = 0.01) -> bool:
    """Polls predicate() until it's true or timeout elapses. W159 follow-up:
    RecoverableScheduler's immediate catch-up path now runs the job body on
    a background thread without waiting (schedule_daily()/schedule_once(),
    wait=False -- a real production hang-on-startup fix), so register_eod_job()
    returns before the EOD job body has necessarily finished. Tests that mock
    run_agent/clients via `with patch(...):` must wait for completion *before*
    the patch context exits, or the background thread executes against the
    real (unpatched) objects once it does -- confirmed live: this is exactly
    what "Real network call attempted during an app/backend test" meant."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(interval)
    return predicate()


def _fire_scheduled_job_now(scheduler: RecoverableScheduler, job_id: str) -> None:
    """W251: register_fixture_reconciliation_job's 6 hourly slots register
    with catch_up=False (see its own comment -- a boot-time redeploy used
    to stampede every already-passed slot at once), so there's no longer a
    catch-up path for these tests to lean on to exercise the job body.
    Fetches the real registered APScheduler job and calls its own func()
    directly instead -- that's the same lambda schedule_daily() would hand
    to the CronTrigger, which calls _run_and_mark(..., wait=True) by
    default, so this blocks until the job body genuinely finishes (no
    _wait_until polling needed afterward)."""
    job = scheduler._scheduler.get_job(job_id)
    assert job is not None, f"job {job_id!r} was never registered"
    job.func()


def test_next_day_date_str_in_ny_timezone() -> None:
    now = datetime(2026, 7, 12, 10, 0, tzinfo=NY_TZ)
    assert next_day_date_str(lambda: now) == "2026-07-13"


def test_next_day_date_str_rolls_over_at_midnight_ny() -> None:
    now = datetime(2026, 7, 12, 23, 59, tzinfo=NY_TZ)
    assert next_day_date_str(lambda: now) == "2026-07-13"


def test_t30_run_at_is_30_minutes_before_kickoff() -> None:
    fixture = NormalizedMatch(
        match_id="m1", utc_date=f"{_FUTURE_DAY_STR}T15:00:00Z", status="SCHEDULED",
        home_team="Arsenal", away_team="Everton", home_goals=None, away_goals=None,
    )
    run_at = t30_run_at(fixture)
    assert run_at.isoformat() == f"{_FUTURE_DAY_STR}T14:30:00+00:00"


def test_persisting_odds_client_saves_counter_after_call(tmp_path: Path) -> None:
    store = FileCreditCounterStore(tmp_path / "counter.json")
    counter = CreditCounter()
    inner_client = MagicMock()
    inner_client.get_odds.return_value = []

    client = PersistingOddsClient(client=inner_client, counter=counter, store=store)
    result = client.get_odds()

    assert result == []
    inner_client.get_odds.assert_called_once()
    assert (tmp_path / "counter.json").exists()


def test_persisting_odds_client_saves_counter_even_when_call_returns_none(tmp_path: Path) -> None:
    store = FileCreditCounterStore(tmp_path / "counter.json")
    counter = CreditCounter()
    inner_client = MagicMock()
    inner_client.get_odds.return_value = None  # budget exhausted

    client = PersistingOddsClient(client=inner_client, counter=counter, store=store)
    result = client.get_odds()

    assert result is None
    assert (tmp_path / "counter.json").exists()


def test_persisting_odds_client_forwards_date_to_the_inner_client(tmp_path: Path) -> None:
    """W99: found live on the first real (non-sandbox) production deploy --
    main.py's manual regenerate path and eod_batch.py's per-fixture-date
    fetch both call get_odds(sport_key=..., date=...); this wrapper must
    accept and forward it exactly like the two real client classes it
    wraps (OddsAPIClient/HistoricalOddsClient) already do, not silently
    drop it or blow up with a TypeError."""
    store = FileCreditCounterStore(tmp_path / "counter.json")
    counter = CreditCounter()
    inner_client = MagicMock()
    inner_client.get_odds.return_value = []

    client = PersistingOddsClient(client=inner_client, counter=counter, store=store)
    client.get_odds(sport_key="soccer_sweden_allsvenskan", date="2026-08-10")

    inner_client.get_odds.assert_called_once_with(sport_key="soccer_sweden_allsvenskan", date="2026-08-10")


def test_persisting_odds_client_get_event_odds_saves_counter_after_call(tmp_path: Path) -> None:
    store = FileCreditCounterStore(tmp_path / "counter.json")
    counter = CreditCounter()
    inner_client = MagicMock()
    inner_client.get_event_odds.return_value = "secondary odds"

    client = PersistingOddsClient(client=inner_client, counter=counter, store=store)
    result = client.get_event_odds(sport_key="soccer_epl", event_id="evt1")

    assert result == "secondary odds"
    inner_client.get_event_odds.assert_called_once_with(
        sport_key="soccer_epl", event_id="evt1", markets=("totals", "btts"),
        regions=None, home_team=None, away_team=None,
    )
    assert (tmp_path / "counter.json").exists()


def test_persisting_odds_client_get_event_odds_forwards_regions_and_team_names(tmp_path: Path) -> None:
    """W199 (post-W99 lesson): regions/home_team/away_team must be forwarded
    the same way markets already is -- W99 already found this exact wrapper
    class silently dropping a newly-added get_odds() param (`date`) with no
    test catching it until production crashed on it for real."""
    store = FileCreditCounterStore(tmp_path / "counter.json")
    counter = CreditCounter()
    inner_client = MagicMock()
    inner_client.get_event_odds.return_value = "team totals odds"

    client = PersistingOddsClient(client=inner_client, counter=counter, store=store)
    result = client.get_event_odds(
        sport_key="soccer_epl", event_id="evt1", markets=("team_totals",),
        regions=("uk", "us", "us2"), home_team="Arsenal", away_team="Everton",
    )

    assert result == "team totals odds"
    inner_client.get_event_odds.assert_called_once_with(
        sport_key="soccer_epl", event_id="evt1", markets=("team_totals",),
        regions=("uk", "us", "us2"), home_team="Arsenal", away_team="Everton",
    )


def test_persisting_odds_client_marks_unavailable_on_401_then_reraises(tmp_path: Path) -> None:
    """W254: direct user request -- a revoked/invalid key (BUG-056) must not
    be retried first on every subsequent call. The 401 itself must still
    propagate unchanged (FallbackOddsClient's own except clause is what
    actually moves to the next key)."""
    store = FileCreditCounterStore(tmp_path / "counter.json")
    counter = CreditCounter()
    inner_client = MagicMock()
    inner_client.get_odds.side_effect = requests.HTTPError(response=MagicMock(status_code=401, headers={}))

    client = PersistingOddsClient(client=inner_client, counter=counter, store=store)

    assert client.is_available() is True
    with pytest.raises(requests.HTTPError):
        client.get_odds()
    assert client.is_available() is False
    assert client.unavailable_reason() == "401"

    reloaded = FileCreditCounterStore(tmp_path / "counter.json").load()
    assert reloaded.is_available() is False


def test_persisting_odds_client_marks_unavailable_on_429_using_retry_after_header(tmp_path: Path) -> None:
    store = FileCreditCounterStore(tmp_path / "counter.json")
    counter = CreditCounter(now_fn=lambda: datetime(2026, 10, 14, 12, 0, tzinfo=timezone.utc))
    inner_client = MagicMock()
    inner_client.get_odds.side_effect = requests.HTTPError(
        response=MagicMock(status_code=429, headers={"Retry-After": "120"}),
    )

    client = PersistingOddsClient(client=inner_client, counter=counter, store=store)
    with pytest.raises(requests.HTTPError):
        client.get_odds()

    assert client.is_available() is False
    assert client.unavailable_reason() == "429"


def test_persisting_odds_client_does_not_mark_unavailable_on_an_unrelated_5xx(tmp_path: Path) -> None:
    """Only 401/403/429 are about the key itself -- a 500/502/503 is the
    provider's own outage, not evidence this particular key is bad, so it
    must not get cooled down over it."""
    store = FileCreditCounterStore(tmp_path / "counter.json")
    counter = CreditCounter()
    inner_client = MagicMock()
    inner_client.get_odds.side_effect = requests.HTTPError(response=MagicMock(status_code=503, headers={}))

    client = PersistingOddsClient(client=inner_client, counter=counter, store=store)
    with pytest.raises(requests.HTTPError):
        client.get_odds()

    assert client.is_available() is True


def test_fallback_odds_client_tries_an_available_key_before_a_cooling_down_one() -> None:
    """W254: the first key in the list being in cooldown must not mean it's
    tried first anyway -- an available later key must be preferred,
    regardless of list position."""
    cooling_down, available = MagicMock(), MagicMock()
    cooling_down.is_available.return_value = False
    cooling_down.unavailable_reason.return_value = "401"
    available.is_available.return_value = True
    available.get_odds.return_value = ["available key's odds"]

    result = FallbackOddsClient([cooling_down, available]).get_odds()

    assert result == ["available key's odds"]
    cooling_down.get_odds.assert_not_called()


def test_fallback_odds_client_tries_a_cooling_down_key_as_a_last_resort_if_nothing_else_works() -> None:
    """Reorder, never hard-skip (same discipline as FootballDataClient's own
    would_block() sort, W250) -- if every other key has also failed, a
    cooling-down key still gets one attempt rather than returning None
    outright on a possibly-overcautious cooldown estimate."""
    cooling_down, exhausted = MagicMock(), MagicMock()
    cooling_down.is_available.return_value = False
    cooling_down.unavailable_reason.return_value = "429"
    cooling_down.get_odds.return_value = ["rescued despite cooldown"]
    exhausted.is_available.return_value = True
    exhausted.get_odds.return_value = None

    result = FallbackOddsClient([cooling_down, exhausted]).get_odds()

    assert result == ["rescued despite cooldown"]
    cooling_down.get_odds.assert_called_once()


def test_fallback_odds_client_uses_first_client_when_it_succeeds() -> None:
    primary, secondary = MagicMock(), MagicMock()
    primary.get_odds.return_value = ["primary odds"]

    result = FallbackOddsClient([primary, secondary]).get_odds()

    assert result == ["primary odds"]
    secondary.get_odds.assert_not_called()


def test_fallback_odds_client_falls_back_when_first_client_is_locally_exhausted() -> None:
    primary, secondary = MagicMock(), MagicMock()
    primary.get_odds.return_value = None  # local CreditCounter predicts exhaustion
    secondary.get_odds.return_value = ["secondary odds"]

    result = FallbackOddsClient([primary, secondary]).get_odds()

    assert result == ["secondary odds"]


def test_fallback_odds_client_falls_back_when_first_client_raises() -> None:
    primary, secondary = MagicMock(), MagicMock()
    primary.get_odds.side_effect = requests.HTTPError("401 out of credits")
    secondary.get_odds.return_value = ["secondary odds"]

    result = FallbackOddsClient([primary, secondary]).get_odds()

    assert result == ["secondary odds"]


def test_fallback_odds_client_returns_none_when_every_client_fails() -> None:
    primary, secondary = MagicMock(), MagicMock()
    primary.get_odds.return_value = None
    secondary.get_odds.side_effect = requests.ConnectionError("network down")

    result = FallbackOddsClient([primary, secondary]).get_odds()

    assert result is None


def test_fallback_odds_client_logs_a_final_warning_only_once_every_key_is_exhausted(caplog) -> None:
    """W250: found live -- OddsAPIClient's own "skipping call" warning fires
    on every exhausted key and, read in isolation, looks identical whether
    a fallback key rescues the call or not, making a genuinely working
    fallback chain indistinguishable in the logs from a fully-exhausted one.
    This is the one place that can say "keeping last-known odds" truthfully
    -- at WARNING, so it survives a warning-level-only log filter -- and it
    must NOT fire when a later key actually rescues the call."""
    import logging

    primary, secondary = MagicMock(), MagicMock()
    primary.get_odds.return_value = None
    secondary.get_odds.return_value = ["secondary odds"]

    with caplog.at_level(logging.WARNING):
        rescued = FallbackOddsClient([primary, secondary]).get_odds()
    assert rescued == ["secondary odds"]
    assert "exhausted or failed" not in caplog.text

    caplog.clear()
    secondary.get_odds.return_value = None
    with caplog.at_level(logging.WARNING):
        exhausted = FallbackOddsClient([primary, secondary]).get_odds()
    assert exhausted is None
    assert "all 2 configured key(s) exhausted or failed" in caplog.text


def test_fallback_odds_client_get_event_odds_falls_back_the_same_way() -> None:
    """W164: get_event_odds() shares _try_each_client with get_odds() --
    one test proving the shared plumbing forwards args/falls back correctly
    for this second method too, not a full re-run of every get_odds() case."""
    primary, secondary = MagicMock(), MagicMock()
    primary.get_event_odds.return_value = None
    secondary.get_event_odds.return_value = "secondary event odds"

    result = FallbackOddsClient([primary, secondary]).get_event_odds(sport_key="soccer_epl", event_id="evt1")

    assert result == "secondary event odds"
    secondary.get_event_odds.assert_called_once_with(
        sport_key="soccer_epl", event_id="evt1", markets=("totals", "btts"),
        regions=None, home_team=None, away_team=None,
    )


def test_fallback_odds_client_get_event_odds_forwards_regions_and_team_names() -> None:
    primary, secondary = MagicMock(), MagicMock()
    primary.get_event_odds.return_value = None
    secondary.get_event_odds.return_value = "team totals odds"

    result = FallbackOddsClient([primary, secondary]).get_event_odds(
        sport_key="soccer_epl", event_id="evt1", markets=("team_totals",),
        regions=("uk", "us", "us2"), home_team="Arsenal", away_team="Everton",
    )

    assert result == "team totals odds"
    secondary.get_event_odds.assert_called_once_with(
        sport_key="soccer_epl", event_id="evt1", markets=("team_totals",),
        regions=("uk", "us", "us2"), home_team="Arsenal", away_team="Everton",
    )


def test_persisting_oddspapi_client_saves_counter_after_get_corners_odds(tmp_path: Path) -> None:
    store = FileCreditCounterStore(tmp_path / "counter.json")
    counter = CreditCounter()
    inner_client = MagicMock()
    inner_client.get_corners_odds.return_value = {"over_9.5": 1.9, "under_9.5": 1.95}

    client = PersistingOddsPapiClient(client=inner_client, counter=counter, store=store)
    result = client.get_corners_odds(fixture_id="555")

    assert result == {"over_9.5": 1.9, "under_9.5": 1.95}
    inner_client.get_corners_odds.assert_called_once_with(fixture_id="555")
    assert (tmp_path / "counter.json").exists()


def test_persisting_oddspapi_client_get_fixtures_forwards_args(tmp_path: Path) -> None:
    store = FileCreditCounterStore(tmp_path / "counter.json")
    counter = CreditCounter()
    inner_client = MagicMock()
    inner_client.get_fixtures.return_value = ["fixture"]

    client = PersistingOddsPapiClient(client=inner_client, counter=counter, store=store)
    result = client.get_fixtures(tournament_id=8, status_id=1)

    assert result == ["fixture"]
    inner_client.get_fixtures.assert_called_once_with(tournament_id=8, status_id=1)


def test_persisting_oddspapi_client_marks_unavailable_on_429_from_get_corners_odds(tmp_path: Path) -> None:
    """W254: same cooldown contract as PersistingOddsClient -- confirmed live
    (this exact session): OddsPapi's key #1 hit 429 on get_corners_odds."""
    store = FileCreditCounterStore(tmp_path / "counter.json")
    counter = CreditCounter()
    inner_client = MagicMock()
    inner_client.get_corners_odds.side_effect = requests.HTTPError(response=MagicMock(status_code=429, headers={}))

    client = PersistingOddsPapiClient(client=inner_client, counter=counter, store=store)
    with pytest.raises(requests.HTTPError):
        client.get_corners_odds(fixture_id="555")

    assert client.is_available() is False
    assert client.unavailable_reason() == "429"


def test_persisting_oddspapi_client_marks_unavailable_on_401_from_get_fixtures(tmp_path: Path) -> None:
    """get_fixtures() never saved the counter at all pre-W254 (no success
    path needs the credit count persisted) -- a 401 here must still be
    recorded onto the same per-key counter get_corners_odds reads, since
    both share one underlying key."""
    store = FileCreditCounterStore(tmp_path / "counter.json")
    counter = CreditCounter()
    inner_client = MagicMock()
    inner_client.get_fixtures.side_effect = requests.HTTPError(response=MagicMock(status_code=401, headers={}))

    client = PersistingOddsPapiClient(client=inner_client, counter=counter, store=store)
    with pytest.raises(requests.HTTPError):
        client.get_fixtures(tournament_id=8)

    assert client.is_available() is False
    reloaded = FileCreditCounterStore(tmp_path / "counter.json").load()
    assert reloaded.is_available() is False


def test_fallback_oddspapi_client_tries_an_available_key_before_a_cooling_down_one() -> None:
    cooling_down, available = MagicMock(), MagicMock()
    cooling_down.is_available.return_value = False
    cooling_down.unavailable_reason.return_value = "429"
    available.is_available.return_value = True
    available.get_corners_odds.return_value = {"over_9.5": 1.9, "under_9.5": 1.95}

    result = FallbackOddsPapiClient([cooling_down, available]).get_corners_odds(fixture_id="555")

    assert result == {"over_9.5": 1.9, "under_9.5": 1.95}
    cooling_down.get_corners_odds.assert_not_called()


def test_fallback_oddspapi_client_uses_first_client_when_it_succeeds() -> None:
    primary, secondary = MagicMock(), MagicMock()
    primary.get_corners_odds.return_value = {"over_9.5": 1.9, "under_9.5": 1.95}

    result = FallbackOddsPapiClient([primary, secondary]).get_corners_odds(fixture_id="555")

    assert result == {"over_9.5": 1.9, "under_9.5": 1.95}
    secondary.get_corners_odds.assert_not_called()


def test_fallback_oddspapi_client_falls_back_when_first_client_is_locally_exhausted() -> None:
    primary, secondary = MagicMock(), MagicMock()
    primary.get_corners_odds.return_value = None  # local CreditCounter predicts exhaustion
    secondary.get_corners_odds.return_value = {"over_9.5": 1.85, "under_9.5": 2.0}

    result = FallbackOddsPapiClient([primary, secondary]).get_corners_odds(fixture_id="555")

    assert result == {"over_9.5": 1.85, "under_9.5": 2.0}


def test_fallback_oddspapi_client_falls_back_when_first_client_raises() -> None:
    primary, secondary = MagicMock(), MagicMock()
    primary.get_corners_odds.side_effect = requests.HTTPError("401 out of credits")
    secondary.get_corners_odds.return_value = {"over_9.5": 1.85, "under_9.5": 2.0}

    result = FallbackOddsPapiClient([primary, secondary]).get_corners_odds(fixture_id="555")

    assert result == {"over_9.5": 1.85, "under_9.5": 2.0}


def test_fallback_oddspapi_client_returns_none_when_every_client_fails() -> None:
    primary, secondary = MagicMock(), MagicMock()
    primary.get_corners_odds.return_value = None
    secondary.get_corners_odds.side_effect = requests.ConnectionError("network down")

    result = FallbackOddsPapiClient([primary, secondary]).get_corners_odds(fixture_id="555")

    assert result is None


def test_fallback_oddspapi_client_get_fixtures_falls_back_on_raise() -> None:
    primary, secondary = MagicMock(), MagicMock()
    primary.get_fixtures.side_effect = requests.HTTPError("401")
    secondary.get_fixtures.return_value = ["fixture"]

    result = FallbackOddsPapiClient([primary, secondary]).get_fixtures(tournament_id=8)

    assert result == ["fixture"]


def test_register_eod_job_generates_recommendations_and_schedules_t30(tmp_path: Path) -> None:
    """End-to-end wiring check (no real network/LLM): a fixture for
    tomorrow gets a cached recommendation from the EOD job, and its T-30
    job is registered on the same scheduler (not merely called and
    discarded) -- confirmed via the run_log gaining an entry once that
    T-30 job's own catch-up fires (its run_at is already in the past
    relative to 'now', matching a realistic same-run scheduling case)."""
    fixture = NormalizedMatch(
        match_id="m1", utc_date=f"{_FUTURE_DAY_STR}T15:00:00Z", status="SCHEDULED",
        home_team="Arsenal", away_team="Everton", home_goals=None, away_goals=None,
    )
    fixtures_client = MagicMock()
    fixtures_client.get_fixtures.return_value = [fixture]
    cache = RecommendationCache(db_path=tmp_path / "cache.db")
    config = AgentConfig.default()
    run_log = JobRunLog(db_path=tmp_path / "job_runs.db")

    # "now" is 2026-08-22 23:30 NY (EDT, UTC-4) = 2026-08-23 03:30 UTC --
    # past both today's 23:00 EOD trigger and the fixture's T-30 trigger
    # (2026-08-22T14:30:00Z), so registering triggers immediate catch-up
    # runs for both rather than waiting.
    now = datetime(_FUTURE_DAY.year, _FUTURE_DAY.month, _FUTURE_DAY.day, 23, 30, tzinfo=NY_TZ)
    scheduler = RecoverableScheduler(run_log=run_log, now_fn=lambda: now)

    recommendation = {
        "match": {"home": "Arsenal", "away": "Everton", "date": "2026-08-22", "league": "E0"},
        "overall": "no_bet", "markets": [], "explanation": "test", "confidence": "low",
        "limitations": [], "prediction_basis": "market_odds_only",
    }
    with patch("app.backend.recommendations.run_agent", return_value=recommendation):
        register_eod_job(
            scheduler, fixtures_client=fixtures_client, odds_client=None,
            cache=cache, config=config, now_fn=lambda: now,
        )
        # W159 follow-up: catch-up now runs on a background thread without
        # waiting -- must finish before the patch above is torn down, or it
        # executes against the real (unpatched) run_agent.
        assert _wait_until(lambda: run_log.has_run(EOD_JOB_ID, _FUTURE_DAY_STR))

    assert run_log.has_run("t30_m1", t30_run_at(fixture).isoformat())


def test_register_eod_job_processes_both_e0_and_swe_when_sweden_client_configured(tmp_path: Path) -> None:
    """W62: register_eod_job is no longer structurally single-league -- when
    a sweden_fixtures_client is supplied, both E0 (football-data.org) and
    SWE (The Odds API, W57) fixtures are discovered and generate correctly
    league-tagged recommendations in the same nightly run."""
    e0_fixture = NormalizedMatch(
        match_id="m1", utc_date=f"{_FUTURE_DAY_STR}T15:00:00Z", status="SCHEDULED",
        home_team="Arsenal", away_team="Everton", home_goals=None, away_goals=None,
    )
    swe_fixture = NormalizedMatch(
        match_id="sw1", utc_date=f"{_FUTURE_DAY_STR}T17:00:00Z", status="SCHEDULED",
        home_team="Malmo FF", away_team="AIK", home_goals=None, away_goals=None,
    )
    fixtures_client = MagicMock()
    fixtures_client.get_fixtures.return_value = [e0_fixture]
    sweden_fixtures_client = MagicMock()
    sweden_fixtures_client.get_fixtures.return_value = [swe_fixture]
    cache = RecommendationCache(db_path=tmp_path / "cache.db")
    config = AgentConfig.default()
    run_log = JobRunLog(db_path=tmp_path / "job_runs.db")
    now = datetime(_FUTURE_DAY.year, _FUTURE_DAY.month, _FUTURE_DAY.day, 23, 30, tzinfo=NY_TZ)
    scheduler = RecoverableScheduler(run_log=run_log, now_fn=lambda: now)

    recommendation = {
        "match": {"home": "A", "away": "B", "date": "2026-08-22", "league": "E0"},
        "overall": "no_bet", "markets": [], "explanation": "test", "confidence": "low",
        "limitations": [], "prediction_basis": "market_odds_only",
    }
    with patch("app.backend.recommendations.run_agent", return_value=recommendation) as mock_run_agent, patch(
        "app.backend.scheduler_wiring.list_display_enabled_competition_ids", return_value=["E0", "SWE", "SP1"]
    ):
        register_eod_job(
            scheduler, fixtures_client=fixtures_client, odds_client=None,
            cache=cache, config=config, now_fn=lambda: now,
            sweden_fixtures_client=sweden_fixtures_client,
        )
        assert _wait_until(lambda: run_log.has_run(EOD_JOB_ID, _FUTURE_DAY_STR))

    leagues_seen = {call.kwargs["match_info"]["league"] for call in mock_run_agent.call_args_list}
    assert leagues_seen == {"E0", "SWE"}
    assert run_log.has_run("t30_m1", t30_run_at(e0_fixture).isoformat())
    assert run_log.has_run("t30_sw1", t30_run_at(swe_fixture).isoformat())


def test_register_eod_job_skips_sweden_gracefully_when_no_sweden_client_configured(tmp_path: Path) -> None:
    """Backward compatibility: omitting sweden_fixtures_client (its default)
    must behave exactly like before W62 -- only E0 is processed, no crash."""
    e0_fixture = NormalizedMatch(
        match_id="m1", utc_date=f"{_FUTURE_DAY_STR}T15:00:00Z", status="SCHEDULED",
        home_team="Arsenal", away_team="Everton", home_goals=None, away_goals=None,
    )
    fixtures_client = MagicMock()
    fixtures_client.get_fixtures.return_value = [e0_fixture]
    cache = RecommendationCache(db_path=tmp_path / "cache.db")
    config = AgentConfig.default()
    run_log = JobRunLog(db_path=tmp_path / "job_runs.db")
    now = datetime(_FUTURE_DAY.year, _FUTURE_DAY.month, _FUTURE_DAY.day, 23, 30, tzinfo=NY_TZ)
    scheduler = RecoverableScheduler(run_log=run_log, now_fn=lambda: now)

    recommendation = {
        "match": {"home": "A", "away": "B", "date": "2026-08-22", "league": "E0"},
        "overall": "no_bet", "markets": [], "explanation": "test", "confidence": "low",
        "limitations": [], "prediction_basis": "market_odds_only",
    }
    with patch("app.backend.recommendations.run_agent", return_value=recommendation) as mock_run_agent:
        register_eod_job(
            scheduler, fixtures_client=fixtures_client, odds_client=None,
            cache=cache, config=config, now_fn=lambda: now,
        )
        assert _wait_until(lambda: run_log.has_run(EOD_JOB_ID, _FUTURE_DAY_STR))

    assert mock_run_agent.call_count == 1
    assert mock_run_agent.call_args.kwargs["match_info"]["league"] == "E0"


def test_register_eod_job_skips_a_display_disabled_competition_even_with_its_client_configured(
    tmp_path: Path,
) -> None:
    """A competition flipped to display_enabled=False (config/competitions.yaml)
    must be skipped even when its fixtures client *is* supplied -- the flag
    wins over client wiring, so turning a competition off doesn't require
    also ripping out its client registration."""
    e0_fixture = NormalizedMatch(
        match_id="m1", utc_date=f"{_FUTURE_DAY_STR}T15:00:00Z", status="SCHEDULED",
        home_team="Arsenal", away_team="Everton", home_goals=None, away_goals=None,
    )
    swe_fixture = NormalizedMatch(
        match_id="sw1", utc_date=f"{_FUTURE_DAY_STR}T17:00:00Z", status="SCHEDULED",
        home_team="Malmo FF", away_team="AIK", home_goals=None, away_goals=None,
    )
    fixtures_client = MagicMock()
    fixtures_client.get_fixtures.return_value = [e0_fixture]
    sweden_fixtures_client = MagicMock()
    sweden_fixtures_client.get_fixtures.return_value = [swe_fixture]
    cache = RecommendationCache(db_path=tmp_path / "cache.db")
    config = AgentConfig.default()
    run_log = JobRunLog(db_path=tmp_path / "job_runs.db")
    now = datetime(_FUTURE_DAY.year, _FUTURE_DAY.month, _FUTURE_DAY.day, 23, 30, tzinfo=NY_TZ)
    scheduler = RecoverableScheduler(run_log=run_log, now_fn=lambda: now)

    recommendation = {
        "match": {"home": "A", "away": "B", "date": "2026-08-22", "league": "E0"},
        "overall": "no_bet", "markets": [], "explanation": "test", "confidence": "low",
        "limitations": [], "prediction_basis": "market_odds_only",
    }
    with patch("app.backend.recommendations.run_agent", return_value=recommendation) as mock_run_agent, patch(
        "app.backend.scheduler_wiring.list_display_enabled_competition_ids", return_value=["E0"]
    ):
        register_eod_job(
            scheduler, fixtures_client=fixtures_client, odds_client=None,
            cache=cache, config=config, now_fn=lambda: now,
            sweden_fixtures_client=sweden_fixtures_client,
        )
        assert _wait_until(lambda: run_log.has_run(EOD_JOB_ID, _FUTURE_DAY_STR))

    assert mock_run_agent.call_count == 1
    assert mock_run_agent.call_args.kwargs["match_info"]["league"] == "E0"
    sweden_fixtures_client.get_fixtures.assert_not_called()


def test_register_eod_job_one_competitions_fixture_fetch_failure_does_not_block_the_other(tmp_path: Path) -> None:
    """W62 acceptance: one competition's fixture-fetch failure must not
    block the other's batch."""
    e0_fixture = NormalizedMatch(
        match_id="m1", utc_date=f"{_FUTURE_DAY_STR}T15:00:00Z", status="SCHEDULED",
        home_team="Arsenal", away_team="Everton", home_goals=None, away_goals=None,
    )
    fixtures_client = MagicMock()
    fixtures_client.get_fixtures.return_value = [e0_fixture]
    sweden_fixtures_client = MagicMock()
    sweden_fixtures_client.get_fixtures.side_effect = RuntimeError("The Odds API is unreachable")
    cache = RecommendationCache(db_path=tmp_path / "cache.db")
    config = AgentConfig.default()
    run_log = JobRunLog(db_path=tmp_path / "job_runs.db")
    now = datetime(_FUTURE_DAY.year, _FUTURE_DAY.month, _FUTURE_DAY.day, 23, 30, tzinfo=NY_TZ)
    scheduler = RecoverableScheduler(run_log=run_log, now_fn=lambda: now)

    recommendation = {
        "match": {"home": "A", "away": "B", "date": "2026-08-22", "league": "E0"},
        "overall": "no_bet", "markets": [], "explanation": "test", "confidence": "low",
        "limitations": [], "prediction_basis": "market_odds_only",
    }
    with patch("app.backend.recommendations.run_agent", return_value=recommendation) as mock_run_agent:
        register_eod_job(
            scheduler, fixtures_client=fixtures_client, odds_client=None,
            cache=cache, config=config, now_fn=lambda: now,
            sweden_fixtures_client=sweden_fixtures_client,
        )
        assert _wait_until(lambda: run_log.has_run(EOD_JOB_ID, _FUTURE_DAY_STR))

    assert mock_run_agent.call_count == 1
    assert mock_run_agent.call_args.kwargs["match_info"]["league"] == "E0"


def test_register_eod_job_processes_e0_swe_and_sp1_when_all_clients_configured(tmp_path: Path) -> None:
    """W81: register_eod_job extends cleanly to a third competition -- SP1
    shares E0's football-data.org provider/class (unlike SWE) but is
    dispatched through its own la_liga_fixtures_client parameter, mirroring
    sweden_fixtures_client's opt-in shape exactly, purely for test/mock
    isolation (production wiring can pass the same underlying client for
    both)."""
    e0_fixture = NormalizedMatch(
        match_id="m1", utc_date=f"{_FUTURE_DAY_STR}T15:00:00Z", status="SCHEDULED",
        home_team="Arsenal", away_team="Everton", home_goals=None, away_goals=None,
    )
    swe_fixture = NormalizedMatch(
        match_id="sw1", utc_date=f"{_FUTURE_DAY_STR}T17:00:00Z", status="SCHEDULED",
        home_team="Malmo FF", away_team="AIK", home_goals=None, away_goals=None,
    )
    sp1_fixture = NormalizedMatch(
        match_id="sp1", utc_date=f"{_FUTURE_DAY_STR}T19:00:00Z", status="SCHEDULED",
        home_team="Real Madrid", away_team="Sevilla FC", home_goals=None, away_goals=None,
    )
    fixtures_client = MagicMock()
    fixtures_client.get_fixtures.return_value = [e0_fixture]
    sweden_fixtures_client = MagicMock()
    sweden_fixtures_client.get_fixtures.return_value = [swe_fixture]
    la_liga_fixtures_client = MagicMock()
    la_liga_fixtures_client.get_fixtures.return_value = [sp1_fixture]
    cache = RecommendationCache(db_path=tmp_path / "cache.db")
    config = AgentConfig.default()
    run_log = JobRunLog(db_path=tmp_path / "job_runs.db")
    now = datetime(_FUTURE_DAY.year, _FUTURE_DAY.month, _FUTURE_DAY.day, 23, 30, tzinfo=NY_TZ)
    scheduler = RecoverableScheduler(run_log=run_log, now_fn=lambda: now)

    recommendation = {
        "match": {"home": "A", "away": "B", "date": "2026-08-22", "league": "E0"},
        "overall": "no_bet", "markets": [], "explanation": "test", "confidence": "low",
        "limitations": [], "prediction_basis": "market_odds_only",
    }
    with patch("app.backend.recommendations.run_agent", return_value=recommendation) as mock_run_agent, patch(
        "app.backend.scheduler_wiring.list_display_enabled_competition_ids", return_value=["E0", "SWE", "SP1"]
    ):
        register_eod_job(
            scheduler, fixtures_client=fixtures_client, odds_client=None,
            cache=cache, config=config, now_fn=lambda: now,
            sweden_fixtures_client=sweden_fixtures_client,
            la_liga_fixtures_client=la_liga_fixtures_client,
        )
        assert _wait_until(lambda: run_log.has_run(EOD_JOB_ID, _FUTURE_DAY_STR))

    leagues_seen = {call.kwargs["match_info"]["league"] for call in mock_run_agent.call_args_list}
    assert leagues_seen == {"E0", "SWE", "SP1"}
    la_liga_fixtures_client.get_fixtures.assert_called_once_with(
        competition_code="PD", date_from=_FUTURE_DAY_PLUS_1_STR, date_to=_FUTURE_DAY_PLUS_1_STR
    )
    assert run_log.has_run("t30_sp1", t30_run_at(sp1_fixture).isoformat())


def test_register_eod_job_skips_la_liga_gracefully_when_no_la_liga_client_configured(tmp_path: Path) -> None:
    """Backward compatibility: omitting la_liga_fixtures_client (its default)
    behaves exactly like before W81 -- only E0 (and SWE, if configured) run,
    no crash."""
    e0_fixture = NormalizedMatch(
        match_id="m1", utc_date=f"{_FUTURE_DAY_STR}T15:00:00Z", status="SCHEDULED",
        home_team="Arsenal", away_team="Everton", home_goals=None, away_goals=None,
    )
    fixtures_client = MagicMock()
    fixtures_client.get_fixtures.return_value = [e0_fixture]
    cache = RecommendationCache(db_path=tmp_path / "cache.db")
    config = AgentConfig.default()
    run_log = JobRunLog(db_path=tmp_path / "job_runs.db")
    now = datetime(_FUTURE_DAY.year, _FUTURE_DAY.month, _FUTURE_DAY.day, 23, 30, tzinfo=NY_TZ)
    scheduler = RecoverableScheduler(run_log=run_log, now_fn=lambda: now)

    recommendation = {
        "match": {"home": "A", "away": "B", "date": "2026-08-22", "league": "E0"},
        "overall": "no_bet", "markets": [], "explanation": "test", "confidence": "low",
        "limitations": [], "prediction_basis": "market_odds_only",
    }
    with patch("app.backend.recommendations.run_agent", return_value=recommendation) as mock_run_agent:
        register_eod_job(
            scheduler, fixtures_client=fixtures_client, odds_client=None,
            cache=cache, config=config, now_fn=lambda: now,
        )
        assert _wait_until(lambda: run_log.has_run(EOD_JOB_ID, _FUTURE_DAY_STR))

    assert mock_run_agent.call_count == 1
    assert mock_run_agent.call_args.kwargs["match_info"]["league"] == "E0"


def test_next_day_date_str_respects_sandbox_override(monkeypatch) -> None:
    monkeypatch.setenv("SANDBOX_MODE", "1")
    monkeypatch.setenv("SANDBOX_DATE", "2026-03-01")
    assert next_day_date_str() == "2026-03-02"


def test_next_day_date_str_uses_real_clock_when_sandbox_off(monkeypatch) -> None:
    monkeypatch.delenv("SANDBOX_MODE", raising=False)
    real_tomorrow = (datetime.now(NY_TZ) + timedelta(days=1)).date().isoformat()
    assert next_day_date_str() == real_tomorrow


def test_register_lessons_job_runs_at_a_different_hour_than_eod() -> None:
    assert LESSONS_HOUR != EOD_HOUR


def test_register_lessons_job_daily_resolves_outcomes_but_writes_no_lesson_yet(tmp_path: Path) -> None:
    """A109 (2026-09-12): lesson-text generation moved off the daily
    cadence onto the weekly job (see register_lessons_job's own
    docstring) -- the daily job now only resolves outcomes. `now` is a
    fixed Monday, well past LESSONS_HOUR but nowhere near the following
    Sunday's weekly trigger, so only the daily job's catch-up fires."""
    cache = RecommendationCache(db_path=tmp_path / "cache.db")
    # No store.insert() here -- RecommendationOutcomeStore.insert() only
    # ever creates an ALREADY-resolved row (correct: bool is required, not
    # optional). A genuinely-unresolved recommendation only exists in the
    # cache until resolve_pending_recommendations() itself creates its
    # first recommendation_outcomes row -- same setup
    # test_generate_daily_lessons_resolves_pending_recommendations_first
    # (test_live_lessons.py) uses for this identical scenario.
    cache.record_generation("m1", "2026-08-22", "hash1", {}, {
        "match": {"home": "Arsenal", "away": "Everton", "date": "2026-08-22", "league": "E0"},
        "overall": "direct_bet",
        "candidates": [{
            "market": "result_3way", "selection": "home", "recommendation_type": "direct_bet",
            "current_odds": 2.0, "value_edge": 0.1, "composite_score": 0.6, "reason": "Test fixture.",
        }],
        "recommendation_pick": {"market": "result_3way", "selection": "home"},
        "confidence": "medium", "explanation": ["good value"], "limitations": [],
        "prediction_basis": "team_history_and_market",
    }, "scheduled")
    store = RecommendationOutcomeStore(db_path=tmp_path / "outcomes.db")
    client = MagicMock()
    client.get_results.return_value = [
        NormalizedMatch(
            match_id="m1", utc_date="2026-08-22T15:00:00Z", status="FINISHED",
            home_team="Arsenal", away_team="Everton", home_goals=2, away_goals=1,
        ),
    ]
    duckdb_manager = DuckDBManager()
    duckdb_manager.db_path = tmp_path / "fpai_core.db"
    config = AgentConfig.default()
    run_log = JobRunLog(db_path=tmp_path / "job_runs.db")

    assert LESSONS_WEEKLY_DAY_OF_WEEK == 6  # Sunday
    now = datetime(2026, 8, 24, LESSONS_HOUR + 1, 0, tzinfo=NY_TZ)  # Monday, past LESSONS_HOUR
    scheduler = RecoverableScheduler(run_log=run_log, now_fn=lambda: now)

    with patch("app.backend.scheduler_wiring._build_lessons_llm_invoke", return_value=None):
        register_lessons_job(
            scheduler, cache=cache, store=store, client=client,
            duckdb_manager=duckdb_manager, config=config,
        )
        assert _wait_until(lambda: run_log.has_run(LESSONS_JOB_ID, now.date().isoformat()))

    resolved = store.list_unbatched_for_lessons()
    assert len(resolved) == 1
    assert resolved[0].correct is True  # the daily job did resolve it

    # The weekly job (which creates agent_lessons via create_lessons_tables)
    # never ran -- the table shouldn't exist at all yet.
    assert not duckdb_manager.db_path.exists() or not _agent_lessons_table_exists(duckdb_manager)


def _agent_lessons_table_exists(duckdb_manager: DuckDBManager) -> bool:
    with duckdb_manager.connection(read_only=True) as conn:
        tables = {row[0] for row in conn.execute("SELECT table_name FROM information_schema.tables").fetchall()}
    return "agent_lessons" in tables


def test_register_lessons_job_weekly_job_generates_the_lesson_from_resolved_outcomes(tmp_path: Path) -> None:
    """A109 (2026-09-12): the weekly job now owns lesson generation (not
    just judging) -- an already-resolved-but-unbatched outcome (as the
    daily job above would leave it) gets turned into a real agent_lessons
    row once the weekly trigger fires, with no human/daily-job step
    required in between."""
    cache = RecommendationCache(db_path=tmp_path / "cache.db")
    store = RecommendationOutcomeStore(db_path=tmp_path / "outcomes.db")
    store.insert(
        match_id="m1", date="2026-08-22", competition="Premier League", market="result_3way",
        selection="home", recommendation_type="direct_bet", confidence="medium", odds=2.0,
        value_edge=0.1, correct=True, generated_at="2026-08-22T10:00:00+00:00",
        competition_id="E0", home_goals=2, away_goals=1,
    )
    client = MagicMock()
    client.get_results.return_value = []
    duckdb_manager = DuckDBManager()
    duckdb_manager.db_path = tmp_path / "fpai_core.db"
    config = AgentConfig.default()
    run_log = JobRunLog(db_path=tmp_path / "job_runs.db")

    assert LESSONS_WEEKLY_DAY_OF_WEEK == 6  # Sunday -- 2026-08-23 below must match
    now = datetime(2026, 8, 23, LESSONS_WEEKLY_HOUR, LESSONS_WEEKLY_MINUTE + 5, tzinfo=NY_TZ)
    scheduler = RecoverableScheduler(run_log=run_log, now_fn=lambda: now)

    with patch("app.backend.scheduler_wiring._build_lessons_llm_invoke", return_value=None):
        register_lessons_job(
            scheduler, cache=cache, store=store, client=client,
            duckdb_manager=duckdb_manager, config=config,
        )
        assert _wait_until(lambda: run_log.has_run(LESSONS_WEEKLY_JOB_ID, "2026-08-23"))

    with duckdb_manager.connection(read_only=True) as conn:
        assert conn.execute("SELECT COUNT(*) FROM agent_lessons").fetchone()[0] == 1
    assert store.list_unbatched_for_lessons() == []  # marked batched by the weekly job


def test_register_lessons_job_weekly_review_judges_accumulated_candidates(tmp_path: Path) -> None:
    """End-to-end: a candidate already sitting pending (as it would after
    a week of daily-only generation, since the daily job no longer judges)
    gets judged once the weekly job's own trigger day/time arrives --
    proves the weekly review is actually reachable from register_lessons_job,
    not just unit-tested in isolation. Pre-seeding the pending row directly
    (rather than relying on the daily job's own same-run catch-up to create
    it) avoids a real race between two independently-threaded catch-up
    fires that can happen when 'now' is past both jobs' trigger times at
    once, exactly as it is here."""
    cache = RecommendationCache(db_path=tmp_path / "cache.db")
    store = RecommendationOutcomeStore(db_path=tmp_path / "outcomes.db")
    client = MagicMock()
    client.get_results.return_value = []
    duckdb_manager = DuckDBManager()
    duckdb_manager.db_path = tmp_path / "fpai_core.db"
    config = AgentConfig.default()
    run_log = JobRunLog(db_path=tmp_path / "job_runs.db")

    from src.agent.lessons import create_lessons_tables, insert_lesson_candidate
    with duckdb_manager.connection() as conn:
        create_lessons_tables(conn)
        insert_lesson_candidate(conn, "Live-sourced batch: pattern.", "E0", "competition_specific", "m1", source="live")

    assert LESSONS_WEEKLY_DAY_OF_WEEK == 6  # Sunday -- 2026-08-23 below must match
    now = datetime(2026, 8, 23, LESSONS_WEEKLY_HOUR, LESSONS_WEEKLY_MINUTE + 5, tzinfo=NY_TZ)
    scheduler = RecoverableScheduler(run_log=run_log, now_fn=lambda: now)

    def fake_llm_invoke(prompt: str) -> str:
        return '{"approve": false, "scope": null, "reasoning": "Single-match sample, too thin to judge."}'

    with patch("app.backend.scheduler_wiring._build_lessons_llm_invoke", return_value=fake_llm_invoke):
        register_lessons_job(
            scheduler, cache=cache, store=store, client=client,
            duckdb_manager=duckdb_manager, config=config,
        )
        assert _wait_until(lambda: run_log.has_run(LESSONS_WEEKLY_JOB_ID, "2026-08-23"))

    with duckdb_manager.connection(read_only=True) as conn:
        row = conn.execute("SELECT status, source, auto_decision_reasoning FROM agent_lessons").fetchone()
    assert row[0] == "rejected"
    assert row[1] == "live"
    assert row[2] == "Single-match sample, too thin to judge."


def test_register_lessons_job_degrades_to_stats_only_when_llm_build_fails(tmp_path: Path) -> None:
    """Task 4 code-quality review, re-targeted 2026-09-12 at the weekly job
    since it now owns generation: a broken/misconfigured LLM provider must
    not fail the whole week's run -- _weekly_review_job() catches the build
    failure and proceeds with llm_invoke=None (a stats-only lesson, and a
    no-op auto-judge pass) instead."""
    cache = RecommendationCache(db_path=tmp_path / "cache.db")
    store = RecommendationOutcomeStore(db_path=tmp_path / "outcomes.db")
    store.insert(
        match_id="m1", date="2026-08-22", competition="Premier League", market="result_3way",
        selection="home", recommendation_type="direct_bet", confidence="medium", odds=2.0,
        value_edge=0.1, correct=True, generated_at="2026-08-22T10:00:00+00:00",
        competition_id="E0", home_goals=2, away_goals=1,
    )
    client = MagicMock()
    client.get_results.return_value = []
    duckdb_manager = DuckDBManager()
    duckdb_manager.db_path = tmp_path / "fpai_core.db"
    config = AgentConfig.default()
    run_log = JobRunLog(db_path=tmp_path / "job_runs.db")

    assert LESSONS_WEEKLY_DAY_OF_WEEK == 6  # Sunday -- 2026-08-23 below must match
    now = datetime(2026, 8, 23, LESSONS_WEEKLY_HOUR, LESSONS_WEEKLY_MINUTE + 5, tzinfo=NY_TZ)
    scheduler = RecoverableScheduler(run_log=run_log, now_fn=lambda: now)

    with patch("app.backend.scheduler_wiring._build_lessons_llm_invoke", side_effect=RuntimeError("boom")):
        register_lessons_job(
            scheduler, cache=cache, store=store, client=client,
            duckdb_manager=duckdb_manager, config=config,
        )
        assert _wait_until(lambda: run_log.has_run(LESSONS_WEEKLY_JOB_ID, "2026-08-23"))

    with duckdb_manager.connection(read_only=True) as conn:
        rows = conn.execute("SELECT status FROM agent_lessons").fetchall()
    assert len(rows) == 1
    assert rows[0][0] == "pending"  # auto-judge never ran (llm_invoke=None -> no-op)


def test_register_data_refresh_job_launches_a_subprocess_per_enabled_league(tmp_path: Path) -> None:
    """W247: the daily automated replacement for the manual
    POST /api/admin/trigger-data-refresh endpoint -- same subprocess
    launch (main.py refresh-data --league <L>), one per configured league,
    run sequentially (not concurrently) since they all write to the same
    DuckDB file and a concurrent second writer would hit its exclusive
    file lock. W252: each per-league call now also carries
    --skip-feature-rebuild, and one extra consolidated `main.py ingest`
    call follows the whole per-league loop -- see register_data_refresh_job's
    own updated docstring."""
    run_log = JobRunLog(db_path=tmp_path / "job_runs.db")
    now = datetime(_FUTURE_DAY.year, _FUTURE_DAY.month, _FUTURE_DAY.day, DATA_REFRESH_HOUR, DATA_REFRESH_MINUTE + 5, tzinfo=NY_TZ)
    scheduler = RecoverableScheduler(run_log=run_log, now_fn=lambda: now)

    mock_process = MagicMock()
    mock_process.wait.return_value = 0
    with patch("app.backend.scheduler_wiring.subprocess.Popen", return_value=mock_process) as mock_popen:
        register_data_refresh_job(scheduler, leagues=("E0", "SP1"))
        assert _wait_until(lambda: run_log.has_run(DATA_REFRESH_JOB_ID, _FUTURE_DAY_STR))

    assert mock_popen.call_count == 3  # 2 per-league calls + 1 consolidated rebuild
    per_league_calls = mock_popen.call_args_list[:2]
    leagues_seen = []
    for call in per_league_calls:
        command = call.args[0]
        assert "refresh-data" in command and "--league" in command and "--skip-feature-rebuild" in command
        leagues_seen.append(command[command.index("--league") + 1])
    assert leagues_seen == ["E0", "SP1"]
    assert mock_process.wait.call_count == 3

    rebuild_command = mock_popen.call_args_list[2].args[0]
    assert rebuild_command[-1] == "ingest"
    assert "--league" not in rebuild_command


def test_register_data_refresh_job_skips_a_display_disabled_league(tmp_path: Path) -> None:
    run_log = JobRunLog(db_path=tmp_path / "job_runs.db")
    now = datetime(_FUTURE_DAY.year, _FUTURE_DAY.month, _FUTURE_DAY.day, DATA_REFRESH_HOUR, DATA_REFRESH_MINUTE + 5, tzinfo=NY_TZ)
    scheduler = RecoverableScheduler(run_log=run_log, now_fn=lambda: now)

    mock_process = MagicMock()
    mock_process.wait.return_value = 0
    with patch("app.backend.scheduler_wiring.subprocess.Popen", return_value=mock_process) as mock_popen, patch(
        "app.backend.scheduler_wiring.list_display_enabled_competition_ids", return_value=["E0"]
    ):
        register_data_refresh_job(scheduler, leagues=("E0", "SP1"))
        assert _wait_until(lambda: run_log.has_run(DATA_REFRESH_JOB_ID, _FUTURE_DAY_STR))

    assert mock_popen.call_count == 2  # 1 enabled league + 1 consolidated rebuild
    assert "E0" in mock_popen.call_args_list[0].args[0]
    assert mock_popen.call_args_list[1].args[0][-1] == "ingest"


def test_register_data_refresh_job_continues_after_one_leagues_subprocess_fails_to_launch(tmp_path: Path) -> None:
    """One league's Popen() call raising (e.g. FileNotFoundError) must not
    stop the rest of the leagues from being attempted, and must not stop
    the overall job from being marked as ran -- same "one failure doesn't
    block the others" contract as register_eod_job's own per-competition
    try/except. W252: SP1 still succeeding means the consolidated rebuild
    at the end still runs (a 3rd Popen call) -- only a league that ALSO
    fails to launch leaves nothing fresh to rebuild from (see the other
    new test below for that case)."""
    run_log = JobRunLog(db_path=tmp_path / "job_runs.db")
    now = datetime(_FUTURE_DAY.year, _FUTURE_DAY.month, _FUTURE_DAY.day, DATA_REFRESH_HOUR, DATA_REFRESH_MINUTE + 5, tzinfo=NY_TZ)
    scheduler = RecoverableScheduler(run_log=run_log, now_fn=lambda: now)

    mock_process = MagicMock()
    mock_process.wait.return_value = 0
    mock_rebuild_process = MagicMock()
    mock_rebuild_process.wait.return_value = 0
    with patch(
        "app.backend.scheduler_wiring.subprocess.Popen",
        side_effect=[OSError("boom"), mock_process, mock_rebuild_process],
    ) as mock_popen:
        register_data_refresh_job(scheduler, leagues=("E0", "SP1"))
        assert _wait_until(lambda: run_log.has_run(DATA_REFRESH_JOB_ID, _FUTURE_DAY_STR))

    assert mock_popen.call_count == 3
    assert mock_process.wait.call_count == 1
    assert mock_rebuild_process.wait.call_count == 1
    assert mock_popen.call_args_list[2].args[0][-1] == "ingest"


def test_register_data_refresh_job_skips_the_rebuild_when_every_league_fails_to_launch(tmp_path: Path) -> None:
    """If nothing fresh was fetched (every league's subprocess failed to
    even launch), the consolidated rebuild has nothing new to rebuild
    from -- skip it rather than burning a rebuild cycle for no reason."""
    run_log = JobRunLog(db_path=tmp_path / "job_runs.db")
    now = datetime(_FUTURE_DAY.year, _FUTURE_DAY.month, _FUTURE_DAY.day, DATA_REFRESH_HOUR, DATA_REFRESH_MINUTE + 5, tzinfo=NY_TZ)
    scheduler = RecoverableScheduler(run_log=run_log, now_fn=lambda: now)

    with patch(
        "app.backend.scheduler_wiring.subprocess.Popen", side_effect=[OSError("boom"), OSError("boom")],
    ) as mock_popen:
        register_data_refresh_job(scheduler, leagues=("E0", "SP1"))
        assert _wait_until(lambda: run_log.has_run(DATA_REFRESH_JOB_ID, _FUTURE_DAY_STR))

    assert mock_popen.call_count == 2  # both leagues attempted, no 3rd (rebuild) call


@pytest.fixture(autouse=True)
def _clear_fixture_cache_between_reconciliation_tests():
    fixture_cache.clear()
    yield
    fixture_cache.clear()


def test_register_fixture_reconciliation_job_does_not_stampede_every_passed_slot_on_boot(tmp_path: Path) -> None:
    """W251, found live: a 16:16 ET redeploy used to independently
    catch-up-fire every one of the 6 hourly slots whose time had already
    passed that day (00:30/04:30/08:30/12:30 -- 4 simultaneous
    reconciliation runs), all competing for the same rate-limited client.
    `now` here is deliberately past 4 of 5 configured slots -- none may
    fire at registration time."""
    run_log = JobRunLog(db_path=tmp_path / "job_runs.db")
    now = datetime(_FUTURE_DAY.year, _FUTURE_DAY.month, _FUTURE_DAY.day, 16, 16, tzinfo=NY_TZ)
    scheduler = RecoverableScheduler(run_log=run_log, now_fn=lambda: now)
    fixtures_client = MagicMock()
    fixtures_client.get_fixtures.return_value = []
    fixtures_client.get_results.return_value = []
    cache = RecommendationCache(db_path=tmp_path / "cache.db")
    config = AgentConfig.default()

    register_fixture_reconciliation_job(
        scheduler, fixtures_client=fixtures_client, odds_client=None, cache=cache, config=config,
        now_fn=lambda: now, hours=(0, 4, 8, 12, 16),
    )

    # No background catch-up thread exists in this path at all (see
    # schedule_daily()'s catch_up=False short-circuit), so there's nothing
    # to poll for -- a direct assertion is already deterministic.
    fixtures_client.get_fixtures.assert_not_called()
    fixtures_client.get_results.assert_not_called()
    for hour in (0, 4, 8, 12, 16):
        assert not run_log.has_run(f"fixture_reconciliation_{hour:02d}", _FUTURE_DAY_STR)

    # Still genuinely registered for its own future trigger, just not
    # firing yet -- catch_up=False must not mean "never runs at all".
    job_ids = {job.id for job in scheduler._scheduler.get_jobs()}
    for hour in (0, 4, 8, 12, 16):
        assert f"fixture_reconciliation_{hour:02d}" in job_ids


def test_register_fixture_reconciliation_job_registers_one_job_per_hour(tmp_path: Path) -> None:
    run_log = JobRunLog(db_path=tmp_path / "job_runs.db")
    now = datetime(_FUTURE_DAY.year, _FUTURE_DAY.month, _FUTURE_DAY.day, 1, 0, tzinfo=NY_TZ)
    scheduler = RecoverableScheduler(run_log=run_log, now_fn=lambda: now)
    fixtures_client = MagicMock()
    fixtures_client.get_fixtures.return_value = []
    fixtures_client.get_results.return_value = []
    cache = RecommendationCache(db_path=tmp_path / "cache.db")
    config = AgentConfig.default()

    register_fixture_reconciliation_job(
        scheduler, fixtures_client=fixtures_client, odds_client=None, cache=cache, config=config,
        now_fn=lambda: now, hours=(0, 4),
    )

    job_ids = {job.id for job in scheduler._scheduler.get_jobs()}
    assert "fixture_reconciliation_00" in job_ids
    assert "fixture_reconciliation_04" in job_ids


def test_register_fixture_reconciliation_job_warms_the_fixture_cache(tmp_path: Path) -> None:
    """The merged warming half of W249: a reconciliation run's own live
    fetch must leave fixture_cache warm, so a subsequent get_range() call
    for an overlapping day is a cache hit."""
    fixture = NormalizedMatch(
        match_id="m1", utc_date=f"{_FUTURE_DAY_STR}T15:00:00Z", status="SCHEDULED",
        home_team="Arsenal", away_team="Everton", home_goals=None, away_goals=None,
    )
    run_log = JobRunLog(db_path=tmp_path / "job_runs.db")
    now = datetime(_FUTURE_DAY.year, _FUTURE_DAY.month, _FUTURE_DAY.day, 0, 30, tzinfo=NY_TZ)
    scheduler = RecoverableScheduler(run_log=run_log, now_fn=lambda: now)
    fixtures_client = MagicMock()
    fixtures_client.get_fixtures.return_value = [fixture]
    fixtures_client.get_results.return_value = []
    cache = RecommendationCache(db_path=tmp_path / "cache.db")
    config = AgentConfig.default()

    register_fixture_reconciliation_job(
        scheduler, fixtures_client=fixtures_client, odds_client=None, cache=cache, config=config,
        now_fn=lambda: now, hours=(0,),
    )
    _fire_scheduled_job_now(scheduler, "fixture_reconciliation_00")

    import asyncio as _asyncio
    cached = _asyncio.run(fixture_cache.get_range(
        "fixtures", MagicMock(side_effect=AssertionError("must not refetch -- already warm")),
        date_from=_FUTURE_DAY_STR, date_to=_FUTURE_DAY_STR,
    ))
    assert len(cached) == 1
    assert cached[0].match_id == "m1"


def test_register_fixture_reconciliation_job_resyncs_t30_for_every_not_yet_kicked_off_fixture(tmp_path: Path) -> None:
    fixture = NormalizedMatch(
        match_id="m1", utc_date=f"{_FUTURE_DAY_STR}T15:00:00Z", status="SCHEDULED",
        home_team="Arsenal", away_team="Everton", home_goals=None, away_goals=None,
    )
    run_log = JobRunLog(db_path=tmp_path / "job_runs.db")
    now = datetime(_FUTURE_DAY.year, _FUTURE_DAY.month, _FUTURE_DAY.day, 23, 30, tzinfo=NY_TZ)
    scheduler = RecoverableScheduler(run_log=run_log, now_fn=lambda: now)
    fixtures_client = MagicMock()
    fixtures_client.get_fixtures.return_value = [fixture]
    fixtures_client.get_results.return_value = []
    cache = RecommendationCache(db_path=tmp_path / "cache.db")
    config = AgentConfig.default()

    register_fixture_reconciliation_job(
        scheduler, fixtures_client=fixtures_client, odds_client=None, cache=cache, config=config,
        now_fn=lambda: now, hours=(23,),
    )
    _fire_scheduled_job_now(scheduler, "fixture_reconciliation_23")
    # The T-30 job this triggers is its own separate schedule_once()
    # catch-up (wait=False, fire-and-forget on its own thread) --
    # unaffected by W251/catch_up=False (that's schedule_daily()-only),
    # but still async relative to _fire_scheduled_job_now() returning.
    assert _wait_until(lambda: run_log.has_run("t30_m1", t30_run_at(fixture).isoformat()))


def test_register_fixture_reconciliation_job_keys_a_future_day_fixture_under_its_own_date_not_todays(
    tmp_path: Path,
) -> None:
    """Regression guard: a fixture kicking off the day AFTER the cycle runs
    must get its T-30 refresh keyed under ITS OWN date, not today's --
    date_str flows into cache.record_generation(date=...), and
    RecommendationCache.get_latest (plus the frontend's own
    kickoffIso.slice(0,10) lookup) keys strictly on the fixture's actual
    date, so a mis-dated write is silently unreachable rather than merely
    wrong.

    Kickoff is 02:00Z rather than the usual 15:00Z for a reason: the T-30
    instant has to be in the past relative to `now` for schedule_once's
    catch-up to actually run the job body (the only way to observe which
    date_str was baked in), and `now` can't advance past _FUTURE_DAY
    without making the fixture "today". An early-morning kickoff is also
    precisely the case the bug does NOT self-heal for -- no later cycle
    runs on the fixture's own day before it kicks off."""
    next_day_str = (_FUTURE_DAY + timedelta(days=1)).isoformat()
    fixture = NormalizedMatch(
        match_id="m1", utc_date=f"{next_day_str}T02:00:00Z", status="SCHEDULED",
        home_team="Arsenal", away_team="Everton", home_goals=None, away_goals=None,
    )
    run_log = JobRunLog(db_path=tmp_path / "job_runs.db")
    now = datetime(_FUTURE_DAY.year, _FUTURE_DAY.month, _FUTURE_DAY.day, 23, 30, tzinfo=NY_TZ)
    scheduler = RecoverableScheduler(run_log=run_log, now_fn=lambda: now)
    fixtures_client = MagicMock()
    fixtures_client.get_fixtures.return_value = [fixture]
    fixtures_client.get_results.return_value = []
    cache = RecommendationCache(db_path=tmp_path / "cache.db")
    config = AgentConfig.default()

    with patch("app.backend.scheduler_wiring.refresh_match_at_t30") as mock_refresh:
        register_fixture_reconciliation_job(
            scheduler, fixtures_client=fixtures_client, odds_client=None, cache=cache, config=config,
            now_fn=lambda: now, hours=(23,),
        )
        _fire_scheduled_job_now(scheduler, "fixture_reconciliation_23")
        # The T-30 job this triggers is its own separate schedule_once()
        # catch-up (wait=False, fire-and-forget on its own thread) --
        # unaffected by W251/catch_up=False (that's schedule_daily()-only),
        # but still async relative to _fire_scheduled_job_now() returning,
        # so this one still needs to poll.
        assert _wait_until(lambda: mock_refresh.called)

    assert mock_refresh.call_args.kwargs["date_str"] == next_day_str
    # mock_refresh.called flips True slightly before run_log.mark_ran()
    # does (the latter happens right after fn() returns, in the same
    # background thread) -- wait for this one too rather than assume the
    # two are close enough to coincide.
    assert _wait_until(lambda: run_log.has_run("t30_m1", t30_run_at(fixture).isoformat()))


def test_register_fixture_reconciliation_job_does_not_resync_an_already_kicked_off_fixture(tmp_path: Path) -> None:
    """has_kicked_off's filter must genuinely exclude a fixture whose
    kickoff has passed. `now` is anchored to a REAL-past day here, unlike
    every other case in this group: has_kicked_off compares against real
    wall-clock time (sandbox_now), not the injected now_fn, so nothing
    inside a _FUTURE_DAY-centred window could ever be "already kicked off"
    at all and the filter would never actually be exercised. Both fixtures
    land inside the fetched [today, today+2] window, so the only thing
    separating them is the filter itself."""
    yesterday = (datetime.now(timezone.utc) - timedelta(days=1)).date()
    kicked_off = NormalizedMatch(
        match_id="m2", utc_date=f"{yesterday.isoformat()}T15:00:00Z", status="FINISHED",
        home_team="A", away_team="B", home_goals=1, away_goals=0,
    )
    upcoming = NormalizedMatch(
        match_id="m1", utc_date=f"{(yesterday + timedelta(days=2)).isoformat()}T15:00:00Z", status="SCHEDULED",
        home_team="Arsenal", away_team="Everton", home_goals=None, away_goals=None,
    )
    run_log = JobRunLog(db_path=tmp_path / "job_runs.db")
    now = datetime(yesterday.year, yesterday.month, yesterday.day, 23, 30, tzinfo=NY_TZ)
    scheduler = RecoverableScheduler(run_log=run_log, now_fn=lambda: now)
    fixtures_client = MagicMock()
    fixtures_client.get_fixtures.return_value = [kicked_off, upcoming]
    fixtures_client.get_results.return_value = []
    cache = RecommendationCache(db_path=tmp_path / "cache.db")
    config = AgentConfig.default()

    register_fixture_reconciliation_job(
        scheduler, fixtures_client=fixtures_client, odds_client=None, cache=cache, config=config,
        now_fn=lambda: now, hours=(0,),
    )
    _fire_scheduled_job_now(scheduler, "fixture_reconciliation_00")

    job_ids = {job.id for job in scheduler._scheduler.get_jobs()}
    assert "t30_m1" in job_ids  # the still-upcoming fixture WAS re-synced
    assert "t30_m2" not in job_ids  # the already-kicked-off one was not
    # m2's own T-30 instant is in the past, so had it been scheduled at all
    # its catch-up would have fired immediately and left this marker.
    assert not run_log.has_run("t30_m2", t30_run_at(kicked_off).isoformat())


def test_register_fixture_reconciliation_job_one_leagues_failure_does_not_block_the_others(tmp_path: Path) -> None:
    e0_fixture = NormalizedMatch(
        match_id="m1", utc_date=f"{_FUTURE_DAY_STR}T15:00:00Z", status="SCHEDULED",
        home_team="Arsenal", away_team="Everton", home_goals=None, away_goals=None,
    )
    run_log = JobRunLog(db_path=tmp_path / "job_runs.db")
    # 23:30 NY (= next-day 03:30 UTC), not the 00:30 the other cases use:
    # past BOTH the 00:30 reconciliation trigger AND m1's own T-30 instant
    # (_FUTURE_DAY 14:30Z), so the re-registered T-30 job's catch-up
    # actually fires and leaves the run_log marker this asserts on. Same
    # anchor as test_..._resyncs_t30_for_every_not_yet_kicked_off_fixture.
    now = datetime(_FUTURE_DAY.year, _FUTURE_DAY.month, _FUTURE_DAY.day, 23, 30, tzinfo=NY_TZ)
    scheduler = RecoverableScheduler(run_log=run_log, now_fn=lambda: now)
    fixtures_client = MagicMock()
    fixtures_client.get_fixtures.return_value = [e0_fixture]
    fixtures_client.get_results.return_value = []
    sweden_fixtures_client = MagicMock()
    sweden_fixtures_client.get_fixtures.side_effect = requests.exceptions.ConnectionError("down")
    sweden_fixtures_client.get_results.side_effect = requests.exceptions.ConnectionError("down")
    cache = RecommendationCache(db_path=tmp_path / "cache.db")
    config = AgentConfig.default()

    with patch(
        "app.backend.scheduler_wiring.list_display_enabled_competition_ids", return_value=["E0", "SWE"]
    ):
        register_fixture_reconciliation_job(
            scheduler, fixtures_client=fixtures_client, odds_client=None, cache=cache, config=config,
            now_fn=lambda: now, sweden_fixtures_client=sweden_fixtures_client, hours=(0,),
        )
        _fire_scheduled_job_now(scheduler, "fixture_reconciliation_00")

    # T-30's own schedule_once() catch-up is fire-and-forget (wait=False)
    # on its own thread, async relative to _fire_scheduled_job_now()
    # returning -- same reasoning as the other reconciliation tests above.
    assert _wait_until(lambda: run_log.has_run("t30_m1", t30_run_at(e0_fixture).isoformat()))


def test_register_recommendation_cache_prune_job_deletes_only_rows_older_than_retention(tmp_path: Path) -> None:
    """W253: recommendation_cache.db has no other retention mechanism (see
    recommendation_cache.py's module docstring) -- this job is what keeps it
    bounded. A row older than RECOMMENDATION_CACHE_RETENTION_DAYS must be
    gone after the weekly trigger fires; a recent one must survive."""
    cache = RecommendationCache(db_path=tmp_path / "cache.db")
    run_log = JobRunLog(db_path=tmp_path / "job_runs.db")

    assert RECOMMENDATION_CACHE_PRUNE_DAY_OF_WEEK == 6  # Sunday -- 2026-08-23 below must match
    now = datetime(2026, 8, 23, RECOMMENDATION_CACHE_PRUNE_HOUR, RECOMMENDATION_CACHE_PRUNE_MINUTE + 5, tzinfo=NY_TZ)
    old_date = (now - timedelta(days=RECOMMENDATION_CACHE_RETENTION_DAYS + 1)).date().isoformat()
    recent_date = (now - timedelta(days=1)).date().isoformat()
    cache.record_generation("old", old_date, "cfg", {}, {"overall": "no_bet"}, "scheduled")
    cache.record_generation("recent", recent_date, "cfg", {}, {"overall": "no_bet"}, "scheduled")

    scheduler = RecoverableScheduler(run_log=run_log, now_fn=lambda: now)
    register_recommendation_cache_prune_job(scheduler, cache=cache)

    assert _wait_until(lambda: run_log.has_run(RECOMMENDATION_CACHE_PRUNE_JOB_ID, "2026-08-23"))
    remaining = cache.export_rows()
    assert [entry.match_id for entry in remaining] == ["recent"]


def test_register_recommendation_cache_prune_job_registers_the_future_weekly_trigger(tmp_path: Path) -> None:
    """Mirrors test_daily_job_with_catch_up_false_still_registers_the_future_cron_trigger's
    own reasoning for the daily case -- the weekly cron trigger itself must
    be registered regardless of whether catch-up also fires immediately."""
    run_log = JobRunLog(db_path=tmp_path / "job_runs.db")
    now = datetime(2026, 8, 24, 0, 0, tzinfo=NY_TZ)  # a Monday, well before Sunday's slot
    scheduler = RecoverableScheduler(run_log=run_log, now_fn=lambda: now)

    register_recommendation_cache_prune_job(scheduler, cache=RecommendationCache(db_path=tmp_path / "cache.db"))

    job = scheduler._scheduler.get_job(RECOMMENDATION_CACHE_PRUNE_JOB_ID)
    assert job is not None
    assert str(job.trigger) == "cron[day_of_week='6', hour='6', minute='30']"


def test_register_eod_job_notifies_subscribers_once_after_the_nightly_run(tmp_path: Path) -> None:
    """New in this task: register_eod_job accepts an optional push_store --
    when supplied, every existing per-league test above (which omit it) is
    completely unaffected (push_store defaults to None -> notification step
    skipped entirely, no PushSubscriptionStore ever constructed for those
    tests). Only a test that explicitly opts in, like this one, exercises
    the notify path."""
    fixture = NormalizedMatch(
        match_id="m1", utc_date=f"{_FUTURE_DAY_STR}T15:00:00Z", status="SCHEDULED",
        home_team="Arsenal", away_team="Everton", home_goals=None, away_goals=None,
    )
    fixtures_client = MagicMock()
    fixtures_client.get_fixtures.return_value = [fixture]
    cache = RecommendationCache(db_path=tmp_path / "cache.db")
    config = AgentConfig.default()
    run_log = JobRunLog(db_path=tmp_path / "job_runs.db")
    push_store = PushSubscriptionStore(db_path=tmp_path / "push.db")
    push_store.add(user_id=1, endpoint="https://push.example/a", p256dh="k", auth="a")

    now = datetime(_FUTURE_DAY.year, _FUTURE_DAY.month, _FUTURE_DAY.day, 23, 30, tzinfo=NY_TZ)
    scheduler = RecoverableScheduler(run_log=run_log, now_fn=lambda: now)

    recommendation = {
        "match": {"home": "Arsenal", "away": "Everton", "date": "2026-08-22", "league": "E0"},
        "overall": "no_bet", "markets": [], "explanation": "test", "confidence": "low",
        "limitations": [], "prediction_basis": "market_odds_only",
    }
    with patch("app.backend.recommendations.run_agent", return_value=recommendation), \
         patch("app.backend.scheduler_wiring.send_to_all_subscribers") as mock_send:
        register_eod_job(
            scheduler, fixtures_client=fixtures_client, odds_client=None,
            cache=cache, config=config, now_fn=lambda: now, push_store=push_store,
        )
        assert _wait_until(lambda: run_log.has_run(EOD_JOB_ID, _FUTURE_DAY_STR))

    # Task 3 (batched T-30 push): this single fixture is also its own
    # kickoff cluster, so send_to_all_subscribers now fires twice -- once
    # for the nightly "New recommendations ready" push (checked here,
    # first call) and once for that fixture's own batch-notify cluster
    # (covered by test_register_eod_job_schedules_one_batched_notify_per_kickoff_cluster
    # above). This test's own concern is only the nightly push firing
    # exactly once, so it pins down the first call rather than the total.
    assert mock_send.call_count == 2
    _, kwargs = mock_send.call_args_list[0]
    assert kwargs["store"] is push_store
    assert kwargs["title"] == "New recommendations ready"


def _match(match_id: str, kickoff: str) -> NormalizedMatch:
    return NormalizedMatch(
        match_id=match_id, utc_date=kickoff, status="SCHEDULED",
        home_team=f"Home{match_id}", away_team=f"Away{match_id}", home_goals=None, away_goals=None,
    )


def test_cluster_fixtures_by_kickoff_empty_list_returns_empty() -> None:
    assert cluster_fixtures_by_kickoff([]) == []


def test_cluster_fixtures_by_kickoff_single_fixture_returns_one_cluster() -> None:
    fixture = _match("m1", "2026-10-17T15:00:00Z")
    assert cluster_fixtures_by_kickoff([fixture]) == [[fixture]]


def test_cluster_fixtures_by_kickoff_groups_fixtures_within_window() -> None:
    a = _match("a", "2026-10-17T15:00:00Z")
    b = _match("b", "2026-10-17T15:10:00Z")
    clusters = cluster_fixtures_by_kickoff([b, a], window_minutes=15)
    assert clusters == [[a, b]]


def test_cluster_fixtures_by_kickoff_splits_fixtures_outside_window() -> None:
    a = _match("a", "2026-10-17T15:00:00Z")
    b = _match("b", "2026-10-17T15:30:00Z")
    clusters = cluster_fixtures_by_kickoff([a, b], window_minutes=15)
    assert clusters == [[a], [b]]


def test_cluster_fixtures_by_kickoff_chains_a_cluster_beyond_the_window_end_to_end() -> None:
    """14:55, 15:05, 15:18 stay one cluster -- each CONSECUTIVE gap is <=15
    min (10, then 13) -- even though first-to-last is 23 min, over the
    window. A 15:40 fixture starts a new cluster (22 min since 15:18)."""
    a = _match("a", "2026-10-17T14:55:00Z")
    b = _match("b", "2026-10-17T15:05:00Z")
    c = _match("c", "2026-10-17T15:18:00Z")
    d = _match("d", "2026-10-17T15:40:00Z")
    clusters = cluster_fixtures_by_kickoff([a, b, c, d], window_minutes=15)
    assert clusters == [[a, b, c], [d]]


def test_build_batch_notification_body_lists_all_when_at_or_under_limit() -> None:
    a = _match("a", "2026-10-17T15:00:00Z")
    b = _match("b", "2026-10-17T15:05:00Z")
    body = build_batch_notification_body([a, b])
    assert body == "Homea v Awaya, Homeb v Awayb"


def test_build_batch_notification_body_truncates_with_remaining_count() -> None:
    fixtures = [_match(str(i), "2026-10-17T15:00:00Z") for i in range(5)]
    body = build_batch_notification_body(fixtures, max_shown=3)
    assert body == "Home0 v Away0, Home1 v Away1, Home2 v Away2 +2 more"


def test_register_eod_job_schedules_one_batched_notify_per_kickoff_cluster(tmp_path: Path) -> None:
    """Two fixtures 10 minutes apart (one cluster) + a third fixture 40
    minutes later (its own cluster) should produce exactly 2 batch-notify
    pushes, not 3 (one per fixture) and not 1 (all fixtures lumped
    together)."""
    close_a = NormalizedMatch(
        match_id="m1", utc_date=f"{_FUTURE_DAY_STR}T15:00:00Z", status="SCHEDULED",
        home_team="Arsenal", away_team="Everton", home_goals=None, away_goals=None,
    )
    close_b = NormalizedMatch(
        match_id="m2", utc_date=f"{_FUTURE_DAY_STR}T15:10:00Z", status="SCHEDULED",
        home_team="Chelsea", away_team="Fulham", home_goals=None, away_goals=None,
    )
    far = NormalizedMatch(
        match_id="m3", utc_date=f"{_FUTURE_DAY_STR}T15:50:00Z", status="SCHEDULED",
        home_team="Liverpool", away_team="Brentford", home_goals=None, away_goals=None,
    )
    fixtures_client = MagicMock()
    fixtures_client.get_fixtures.return_value = [close_a, close_b, far]
    cache = RecommendationCache(db_path=tmp_path / "cache.db")
    config = AgentConfig.default()
    run_log = JobRunLog(db_path=tmp_path / "job_runs.db")
    push_store = PushSubscriptionStore(db_path=tmp_path / "push.db")
    push_store.add(user_id=1, endpoint="https://push.example/a", p256dh="k", auth="a")

    # "now" is well after all 3 fixtures' own T-30 times AND after the
    # batch-notify jobs' (T-30 + buffer) times, so registering triggers
    # immediate catch-up for everything -- same pattern as this file's
    # other EOD/T-30 catch-up tests.
    now = datetime(_FUTURE_DAY.year, _FUTURE_DAY.month, _FUTURE_DAY.day, 23, 30, tzinfo=NY_TZ)
    scheduler = RecoverableScheduler(run_log=run_log, now_fn=lambda: now)

    recommendation = {
        "match": {"home": "Arsenal", "away": "Everton", "date": "2026-08-22", "league": "E0"},
        "overall": "no_bet", "markets": [], "explanation": "test", "confidence": "low",
        "limitations": [], "prediction_basis": "market_odds_only",
    }

    # Computed ahead of time to match _schedule_t30_batch_notify's own
    # run_at formula (max t30_run_at in the cluster + a 3-minute buffer) --
    # hardcoded as timedelta(minutes=3) here rather than importing
    # T30_NOTIFY_BUFFER_MINUTES, since that constant doesn't exist until
    # this task's Step 3 (this test is written and confirmed-failing
    # first, per TDD).
    close_cluster_run_at = t30_run_at(close_b) + timedelta(minutes=3)  # later of close_a/close_b
    far_cluster_run_at = t30_run_at(far) + timedelta(minutes=3)
    # _eod_job's own date_str is next_day_date_str(now_fn), i.e. tomorrow
    # relative to "now" (23:30 on _FUTURE_DAY) -- _FUTURE_DAY_PLUS_1_STR,
    # same next-day shift this file already relies on at its PD
    # fixtures-fetch assertion above. The fixtures themselves are dated
    # _FUTURE_DAY_STR only because the mocked fixtures_client ignores the
    # date range it's called with -- that's independent of what date_str
    # _eod_job computes internally and bakes into the notify job_id.
    close_job_id = f"t30_notify_{_FUTURE_DAY_PLUS_1_STR}_m1"
    far_job_id = f"t30_notify_{_FUTURE_DAY_PLUS_1_STR}_m3"

    with patch("app.backend.recommendations.run_agent", return_value=recommendation), \
         patch("app.backend.scheduler_wiring.send_to_all_subscribers") as mock_send:
        register_eod_job(
            scheduler, fixtures_client=fixtures_client, odds_client=None,
            cache=cache, config=config, now_fn=lambda: now, push_store=push_store,
        )
        assert _wait_until(lambda: run_log.has_run(EOD_JOB_ID, _FUTURE_DAY_STR))
        # Both batch-notify jobs' run_at is already in the past relative to
        # "now" (same as T-30's own catch-up above), so registering them
        # triggers an immediate catch-up run on a background thread --
        # must wait for both to actually finish *before* the patch above
        # is torn down, or a late-firing job calls the real (unpatched)
        # send_to_all_subscribers instead of the mock (same gotcha this
        # file's other catch-up tests already guard against).
        assert _wait_until(lambda: run_log.has_run(close_job_id, close_cluster_run_at.isoformat()))
        assert _wait_until(lambda: run_log.has_run(far_job_id, far_cluster_run_at.isoformat()))

    # 1 nightly "New recommendations ready" call + 2 batch-notify calls
    # (one for [close_a, close_b], one for [far]) = 3 total.
    assert mock_send.call_count == 3
    bodies = [kwargs["body"] for _, kwargs in mock_send.call_args_list]
    assert "Tomorrow's picks" in bodies[0]
    batch_bodies = bodies[1:]
    assert any("Arsenal v Everton, Chelsea v Fulham" in b for b in batch_bodies)
    assert any("Liverpool v Brentford" in b for b in batch_bodies)
    titles = [kwargs["title"] for _, kwargs in mock_send.call_args_list]
    assert titles.count("Kickoff soon") == 2


def test_build_schedule_t30_shares_one_odds_fetch_across_a_kickoff_cluster() -> None:
    """BUG-082: odds_client.get_odds() already returns the whole league's
    odds in one response -- two fixtures 10 minutes apart (one cluster)
    must share a single call between their two T-30 jobs, not one each."""
    close_a = _match("a", f"{_FUTURE_DAY_STR}T15:00:00Z")
    close_b = _match("b", f"{_FUTURE_DAY_STR}T15:10:00Z")
    odds_client = MagicMock()
    odds_client.get_odds.return_value = []
    cache = MagicMock()
    config = AgentConfig.default()

    captured = {}
    fake_scheduler = MagicMock()
    fake_scheduler.schedule_once.side_effect = lambda job_id, fn, run_at: captured.__setitem__(job_id, fn)

    schedule_t30 = build_schedule_t30(
        fake_scheduler, odds_client, cache, config, _FUTURE_DAY_STR, league="E0",
        fixtures=[close_a, close_b],
    )
    schedule_t30(close_a)
    schedule_t30(close_b)

    with patch("app.backend.scheduler_wiring.refresh_match_at_t30") as mock_refresh:
        captured["t30_a"]()
        captured["t30_b"]()
        assert mock_refresh.call_count == 2
        for call in mock_refresh.call_args_list:
            call.kwargs["get_odds_events"]()

    assert odds_client.get_odds.call_count == 1


def test_build_schedule_t30_does_not_share_odds_across_different_clusters() -> None:
    """Fixtures 40 minutes apart (two separate clusters, outside the
    default 15-minute window) must each still fetch their own odds --
    confirms the sharing above is scoped to a cluster, not every fixture
    build_schedule_t30 was ever called with."""
    early = _match("a", f"{_FUTURE_DAY_STR}T15:00:00Z")
    late = _match("b", f"{_FUTURE_DAY_STR}T15:40:00Z")
    odds_client = MagicMock()
    odds_client.get_odds.return_value = []
    cache = MagicMock()
    config = AgentConfig.default()

    captured = {}
    fake_scheduler = MagicMock()
    fake_scheduler.schedule_once.side_effect = lambda job_id, fn, run_at: captured.__setitem__(job_id, fn)

    schedule_t30 = build_schedule_t30(
        fake_scheduler, odds_client, cache, config, _FUTURE_DAY_STR, league="E0",
        fixtures=[early, late],
    )
    schedule_t30(early)
    schedule_t30(late)

    with patch("app.backend.scheduler_wiring.refresh_match_at_t30") as mock_refresh:
        captured["t30_a"]()
        captured["t30_b"]()
        for call in mock_refresh.call_args_list:
            call.kwargs["get_odds_events"]()

    assert odds_client.get_odds.call_count == 2
