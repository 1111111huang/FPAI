"""W08/W09/W10 wiring: connects RecoverableScheduler (W08) to the EOD batch
job (W09) and each fixture's T-30 job (W10). Focused on the pure/testable
seams -- next_day_date_str's NY-timezone date math, t30_run_at's kickoff-
minus-30-minutes parsing, and PersistingOddsClient's credit-counter
persistence (W07's OddsAPIClient/CreditCounter/FileCreditCounterStore
trio leaves persistence to the caller by design)."""

from __future__ import annotations

from datetime import datetime, timedelta
from pathlib import Path
import sys
from unittest.mock import MagicMock, patch

sys.path.append(str(Path(__file__).resolve().parents[3]))

from app.backend.football_data_client import NormalizedMatch
from app.backend.odds_api_client import CreditCounter, FileCreditCounterStore
from app.backend.recommendation_cache import RecommendationCache
from app.backend.scheduler import NY_TZ, JobRunLog, RecoverableScheduler
from app.backend.scheduler_wiring import (
    EOD_JOB_ID,
    PersistingOddsClient,
    next_day_date_str,
    register_eod_job,
    t30_run_at,
)
from src.agent.agent_config import AgentConfig


def test_next_day_date_str_in_ny_timezone() -> None:
    now = datetime(2026, 7, 12, 10, 0, tzinfo=NY_TZ)
    assert next_day_date_str(lambda: now) == "2026-07-13"


def test_next_day_date_str_rolls_over_at_midnight_ny() -> None:
    now = datetime(2026, 7, 12, 23, 59, tzinfo=NY_TZ)
    assert next_day_date_str(lambda: now) == "2026-07-13"


def test_t30_run_at_is_30_minutes_before_kickoff() -> None:
    fixture = NormalizedMatch(
        match_id="m1", utc_date="2026-08-22T15:00:00Z", status="SCHEDULED",
        home_team="Arsenal", away_team="Everton", home_goals=None, away_goals=None,
    )
    run_at = t30_run_at(fixture)
    assert run_at.isoformat() == "2026-08-22T14:30:00+00:00"


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


def test_register_eod_job_generates_recommendations_and_schedules_t30(tmp_path: Path) -> None:
    """End-to-end wiring check (no real network/LLM): a fixture for
    tomorrow gets a cached recommendation from the EOD job, and its T-30
    job is registered on the same scheduler (not merely called and
    discarded) -- confirmed via the run_log gaining an entry once that
    T-30 job's own catch-up fires (its run_at is already in the past
    relative to 'now', matching a realistic same-run scheduling case)."""
    fixture = NormalizedMatch(
        match_id="m1", utc_date="2026-08-22T15:00:00Z", status="SCHEDULED",
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
    now = datetime(2026, 8, 22, 23, 30, tzinfo=NY_TZ)
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

    assert run_log.has_run(EOD_JOB_ID, "2026-08-22")
    assert run_log.has_run("t30_m1", t30_run_at(fixture).isoformat())


def test_register_eod_job_processes_both_e0_and_swe_when_sweden_client_configured(tmp_path: Path) -> None:
    """W62: register_eod_job is no longer structurally single-league -- when
    a sweden_fixtures_client is supplied, both E0 (football-data.org) and
    SWE (The Odds API, W57) fixtures are discovered and generate correctly
    league-tagged recommendations in the same nightly run."""
    e0_fixture = NormalizedMatch(
        match_id="m1", utc_date="2026-08-22T15:00:00Z", status="SCHEDULED",
        home_team="Arsenal", away_team="Everton", home_goals=None, away_goals=None,
    )
    swe_fixture = NormalizedMatch(
        match_id="sw1", utc_date="2026-08-22T17:00:00Z", status="SCHEDULED",
        home_team="Malmo FF", away_team="AIK", home_goals=None, away_goals=None,
    )
    fixtures_client = MagicMock()
    fixtures_client.get_fixtures.return_value = [e0_fixture]
    sweden_fixtures_client = MagicMock()
    sweden_fixtures_client.get_fixtures.return_value = [swe_fixture]
    cache = RecommendationCache(db_path=tmp_path / "cache.db")
    config = AgentConfig.default()
    run_log = JobRunLog(db_path=tmp_path / "job_runs.db")
    now = datetime(2026, 8, 22, 23, 30, tzinfo=NY_TZ)
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

    leagues_seen = {call.kwargs["match_info"]["league"] for call in mock_run_agent.call_args_list}
    assert leagues_seen == {"E0", "SWE"}
    assert run_log.has_run("t30_m1", t30_run_at(e0_fixture).isoformat())
    assert run_log.has_run("t30_sw1", t30_run_at(swe_fixture).isoformat())


def test_register_eod_job_skips_sweden_gracefully_when_no_sweden_client_configured(tmp_path: Path) -> None:
    """Backward compatibility: omitting sweden_fixtures_client (its default)
    must behave exactly like before W62 -- only E0 is processed, no crash."""
    e0_fixture = NormalizedMatch(
        match_id="m1", utc_date="2026-08-22T15:00:00Z", status="SCHEDULED",
        home_team="Arsenal", away_team="Everton", home_goals=None, away_goals=None,
    )
    fixtures_client = MagicMock()
    fixtures_client.get_fixtures.return_value = [e0_fixture]
    cache = RecommendationCache(db_path=tmp_path / "cache.db")
    config = AgentConfig.default()
    run_log = JobRunLog(db_path=tmp_path / "job_runs.db")
    now = datetime(2026, 8, 22, 23, 30, tzinfo=NY_TZ)
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

    assert mock_run_agent.call_count == 1
    assert mock_run_agent.call_args.kwargs["match_info"]["league"] == "E0"


def test_register_eod_job_one_competitions_fixture_fetch_failure_does_not_block_the_other(tmp_path: Path) -> None:
    """W62 acceptance: one competition's fixture-fetch failure must not
    block the other's batch."""
    e0_fixture = NormalizedMatch(
        match_id="m1", utc_date="2026-08-22T15:00:00Z", status="SCHEDULED",
        home_team="Arsenal", away_team="Everton", home_goals=None, away_goals=None,
    )
    fixtures_client = MagicMock()
    fixtures_client.get_fixtures.return_value = [e0_fixture]
    sweden_fixtures_client = MagicMock()
    sweden_fixtures_client.get_fixtures.side_effect = RuntimeError("The Odds API is unreachable")
    cache = RecommendationCache(db_path=tmp_path / "cache.db")
    config = AgentConfig.default()
    run_log = JobRunLog(db_path=tmp_path / "job_runs.db")
    now = datetime(2026, 8, 22, 23, 30, tzinfo=NY_TZ)
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
        match_id="m1", utc_date="2026-08-22T15:00:00Z", status="SCHEDULED",
        home_team="Arsenal", away_team="Everton", home_goals=None, away_goals=None,
    )
    swe_fixture = NormalizedMatch(
        match_id="sw1", utc_date="2026-08-22T17:00:00Z", status="SCHEDULED",
        home_team="Malmo FF", away_team="AIK", home_goals=None, away_goals=None,
    )
    sp1_fixture = NormalizedMatch(
        match_id="sp1", utc_date="2026-08-22T19:00:00Z", status="SCHEDULED",
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
    now = datetime(2026, 8, 22, 23, 30, tzinfo=NY_TZ)
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
            la_liga_fixtures_client=la_liga_fixtures_client,
        )

    leagues_seen = {call.kwargs["match_info"]["league"] for call in mock_run_agent.call_args_list}
    assert leagues_seen == {"E0", "SWE", "SP1"}
    la_liga_fixtures_client.get_fixtures.assert_called_once_with(
        competition_code="PD", date_from="2026-08-23", date_to="2026-08-23"
    )
    assert run_log.has_run("t30_sp1", t30_run_at(sp1_fixture).isoformat())


def test_register_eod_job_skips_la_liga_gracefully_when_no_la_liga_client_configured(tmp_path: Path) -> None:
    """Backward compatibility: omitting la_liga_fixtures_client (its default)
    behaves exactly like before W81 -- only E0 (and SWE, if configured) run,
    no crash."""
    e0_fixture = NormalizedMatch(
        match_id="m1", utc_date="2026-08-22T15:00:00Z", status="SCHEDULED",
        home_team="Arsenal", away_team="Everton", home_goals=None, away_goals=None,
    )
    fixtures_client = MagicMock()
    fixtures_client.get_fixtures.return_value = [e0_fixture]
    cache = RecommendationCache(db_path=tmp_path / "cache.db")
    config = AgentConfig.default()
    run_log = JobRunLog(db_path=tmp_path / "job_runs.db")
    now = datetime(2026, 8, 22, 23, 30, tzinfo=NY_TZ)
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
