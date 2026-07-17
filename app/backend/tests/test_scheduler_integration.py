"""W21: scheduled-job test strategy. The EOD batch (W09) and T-30 refresh
(W10) fire on real-world clock time by design; every piece in this
subsystem (RecoverableScheduler, register_eod_job, build_schedule_t30)
takes an injectable `now_fn` rather than reading `datetime.now()`
internally, so a full day/kickoff cycle can be simulated in milliseconds
by controlling that clock directly -- no freezegun or other time-mocking
library needed, and no sleeping through real hours.

This file's acceptance-literal scenario: the backend was down from 22:00
to 23:30 America/New_York (spanning the 23:00 EOD trigger) -- on restart
at 23:30, the missed EOD batch must catch up immediately rather than
silently waiting for tomorrow's window. A parallel scenario covers W10's
T-30 job catching up the same way after a fixture's window was missed.
"""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
import sys
from unittest.mock import MagicMock, patch

sys.path.append(str(Path(__file__).resolve().parents[3]))

from app.backend.agent_config_hash import compute_agent_config_hash
from app.backend.football_data_client import NormalizedMatch
from app.backend.recommendation_cache import RecommendationCache
from app.backend.scheduler import NY_TZ, JobRunLog, RecoverableScheduler
from app.backend.scheduler_wiring import EOD_JOB_ID, build_schedule_t30, register_eod_job, t30_run_at
from src.agent.agent_config import AgentConfig

_RECOMMENDATION = {
    "match": {"home": "Arsenal", "away": "Everton", "date": "2026-07-13", "league": "E0"},
    "overall": "no_bet", "markets": [], "explanation": "test", "confidence": "low",
    "limitations": [], "prediction_basis": "market_odds_only",
}


def test_missed_eod_batch_catches_up_after_an_outage_spanning_the_trigger(tmp_path: Path) -> None:
    """Backend goes down at 22:00 NY (before the 23:00 EOD trigger) and
    restarts at 23:30 NY (after it) -- the EOD batch must have run by the
    time the restarted process finishes registering the job, not be
    silently deferred to tomorrow's 23:00 fire."""
    fixture = NormalizedMatch(
        match_id="m1", utc_date="2026-07-13T15:00:00Z", status="SCHEDULED",
        home_team="Arsenal", away_team="Everton", home_goals=None, away_goals=None,
    )
    fixtures_client = MagicMock()
    fixtures_client.get_fixtures.return_value = [fixture]
    cache = RecommendationCache(db_path=tmp_path / "cache.db")
    config = AgentConfig.default()
    job_runs_db = tmp_path / "job_runs.db"

    # --- process A: still healthy at 22:00, before the trigger ---
    down_since = datetime(2026, 7, 12, 22, 0, tzinfo=NY_TZ)
    scheduler_a = RecoverableScheduler(run_log=JobRunLog(db_path=job_runs_db), now_fn=lambda: down_since)
    with patch("app.backend.recommendations.run_agent", return_value=_RECOMMENDATION) as mock_run_agent:
        register_eod_job(scheduler_a, fixtures_client=fixtures_client, odds_client=None, cache=cache, config=config, now_fn=lambda: down_since)
    mock_run_agent.assert_not_called()  # too early -- backend "goes down" right after this

    # --- process B: restarts at 23:30, after the missed 23:00 trigger ---
    restart_at = datetime(2026, 7, 12, 23, 30, tzinfo=NY_TZ)
    scheduler_b = RecoverableScheduler(run_log=JobRunLog(db_path=job_runs_db), now_fn=lambda: restart_at)
    with patch("app.backend.recommendations.run_agent", return_value=_RECOMMENDATION) as mock_run_agent:
        register_eod_job(scheduler_b, fixtures_client=fixtures_client, odds_client=None, cache=cache, config=config, now_fn=lambda: restart_at)

    mock_run_agent.assert_called_once()  # caught up immediately on restart
    agent_config_hash = compute_agent_config_hash(config)
    assert cache.get_latest("m1", "2026-07-13", agent_config_hash) is not None

    # --- a third restart at the same moment must not re-run it again ---
    scheduler_c = RecoverableScheduler(run_log=JobRunLog(db_path=job_runs_db), now_fn=lambda: restart_at)
    with patch("app.backend.recommendations.run_agent") as mock_run_agent_c:
        register_eod_job(scheduler_c, fixtures_client=fixtures_client, odds_client=None, cache=cache, config=config, now_fn=lambda: restart_at)
    mock_run_agent_c.assert_not_called()


def test_missed_t30_job_catches_up_after_an_outage_spanning_the_kickoff_window(tmp_path: Path) -> None:
    """A fixture's T-30 window (kickoff minus 30 minutes) falls during a
    separate outage -- the next process to register that fixture's T-30
    job must catch it up immediately, same as the EOD case. Uses
    odds_client=None (refresh_match_at_t30 then always returns
    'skipped_no_odds') so the thing under test is purely *whether the job
    ran at all* (proven via JobRunLog, which is marked regardless of that
    outcome), not what it decided to do once it ran."""
    fixture = NormalizedMatch(
        match_id="m2", utc_date="2026-07-13T15:00:00Z", status="SCHEDULED",  # T-30 trigger = 14:30 UTC
        home_team="Chelsea", away_team="Fulham", home_goals=None, away_goals=None,
    )
    cache = RecommendationCache(db_path=tmp_path / "cache.db")
    config = AgentConfig.default()
    job_runs_db = tmp_path / "job_runs.db"
    run_log = JobRunLog(db_path=job_runs_db)
    expected_run_key = t30_run_at(fixture).isoformat()

    # before the T-30 window: 14:00 UTC -- registering must not catch up yet
    before_utc = datetime(2026, 7, 13, 14, 0, tzinfo=timezone.utc)
    scheduler_a = RecoverableScheduler(run_log=JobRunLog(db_path=job_runs_db), now_fn=lambda: before_utc)
    schedule_t30_a = build_schedule_t30(scheduler_a, odds_client=None, cache=cache, config=config, date_str="2026-07-13")
    schedule_t30_a(fixture)
    assert not run_log.has_run("t30_m2", expected_run_key)

    # restart after the outage, at 15:00 UTC (past the 14:30 T-30 trigger)
    after_utc = datetime(2026, 7, 13, 15, 0, tzinfo=timezone.utc)
    scheduler_b = RecoverableScheduler(run_log=JobRunLog(db_path=job_runs_db), now_fn=lambda: after_utc)
    schedule_t30_b = build_schedule_t30(scheduler_b, odds_client=None, cache=cache, config=config, date_str="2026-07-13")
    schedule_t30_b(fixture)

    assert run_log.has_run("t30_m2", expected_run_key)  # caught up immediately on restart
