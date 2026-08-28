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

import requests

from app.backend.football_data_client import NormalizedMatch
from app.backend.odds_api_client import CreditCounter, FileCreditCounterStore
from app.backend.recommendation_cache import RecommendationCache
from app.backend.recommendation_outcomes import RecommendationOutcomeStore
from app.backend.scheduler import NY_TZ, JobRunLog, RecoverableScheduler
from app.backend.scheduler_wiring import (
    EOD_HOUR,
    EOD_JOB_ID,
    FallbackOddsClient,
    LESSONS_HOUR,
    LESSONS_JOB_ID,
    LESSONS_WEEKLY_DAY_OF_WEEK,
    LESSONS_WEEKLY_HOUR,
    LESSONS_WEEKLY_JOB_ID,
    LESSONS_WEEKLY_MINUTE,
    PersistingOddsClient,
    next_day_date_str,
    register_eod_job,
    register_lessons_job,
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
    inner_client.get_event_odds.assert_called_once_with(sport_key="soccer_epl", event_id="evt1", markets=("totals", "btts"))
    assert (tmp_path / "counter.json").exists()


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


def test_fallback_odds_client_get_event_odds_falls_back_the_same_way() -> None:
    """W164: get_event_odds() shares _try_each_client with get_odds() --
    one test proving the shared plumbing forwards args/falls back correctly
    for this second method too, not a full re-run of every get_odds() case."""
    primary, secondary = MagicMock(), MagicMock()
    primary.get_event_odds.return_value = None
    secondary.get_event_odds.return_value = "secondary event odds"

    result = FallbackOddsClient([primary, secondary]).get_event_odds(sport_key="soccer_epl", event_id="evt1")

    assert result == "secondary event odds"
    secondary.get_event_odds.assert_called_once_with(sport_key="soccer_epl", event_id="evt1", markets=("totals", "btts"))


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


def test_register_lessons_job_generates_a_candidate_and_marks_the_scheduler_run(tmp_path: Path) -> None:
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

    now = datetime(_FUTURE_DAY.year, _FUTURE_DAY.month, _FUTURE_DAY.day, 23, 30, tzinfo=NY_TZ)
    scheduler = RecoverableScheduler(run_log=run_log, now_fn=lambda: now)

    with patch("app.backend.scheduler_wiring._build_lessons_llm_invoke", return_value=None):
        register_lessons_job(
            scheduler, cache=cache, store=store, client=client,
            duckdb_manager=duckdb_manager, config=config,
        )
        assert _wait_until(lambda: run_log.has_run(LESSONS_JOB_ID, now.date().isoformat()))

    with duckdb_manager.connection(read_only=True) as conn:
        assert conn.execute("SELECT COUNT(*) FROM agent_lessons").fetchone()[0] == 1


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
    """Task 4 code-quality review: a broken/misconfigured LLM provider must
    not fail the whole day's run -- _lessons_job() catches the build
    failure and proceeds with llm_invoke=None (a stats-only lesson) instead."""
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

    now = datetime(_FUTURE_DAY.year, _FUTURE_DAY.month, _FUTURE_DAY.day, 23, 30, tzinfo=NY_TZ)
    scheduler = RecoverableScheduler(run_log=run_log, now_fn=lambda: now)

    with patch("app.backend.scheduler_wiring._build_lessons_llm_invoke", side_effect=RuntimeError("boom")):
        register_lessons_job(
            scheduler, cache=cache, store=store, client=client,
            duckdb_manager=duckdb_manager, config=config,
        )
        assert _wait_until(lambda: run_log.has_run(LESSONS_JOB_ID, now.date().isoformat()))

    with duckdb_manager.connection(read_only=True) as conn:
        assert conn.execute("SELECT COUNT(*) FROM agent_lessons").fetchone()[0] == 1
