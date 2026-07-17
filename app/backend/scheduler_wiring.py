"""W08/W09/W10 wiring: connects RecoverableScheduler (W08) to the EOD batch
job (W09), whose schedule_t30 callback registers each fixture's T-30 job
(W10) on the same scheduler instance. Kept out of main.py to keep the route
module focused on HTTP concerns.
"""

from __future__ import annotations

import asyncio
from datetime import datetime, timedelta
import os
from pathlib import Path
from typing import Callable

from app.backend.eod_batch import run_eod_batch
from app.backend.football_data_client import FootballDataClient, NormalizedMatch
from app.backend.historical_odds_client import HistoricalOddsClient
from app.backend.odds_api_client import CreditCounter, FileCreditCounterStore, OddsAPIClient
from app.backend.recommendation_cache import RecommendationCache
from app.backend.scheduler import NY_TZ, RecoverableScheduler
from app.backend.sandbox_clock import sandbox_date, sandbox_now
from app.backend.t30_refresh import refresh_match_at_t30
from src.agent.agent_config import AgentConfig
from src.utils.logger import get_logger

LOGGER = get_logger(__name__)

CREDIT_COUNTER_PATH = Path(__file__).parent.parent / "data" / "odds_api_credit_counter.json"
EOD_JOB_ID = "eod_batch_generation"
EOD_HOUR = 23
EOD_MINUTE = 0


class PersistingOddsClient:
    """Wraps OddsAPIClient so its CreditCounter is persisted to disk after
    every call. W07's OddsAPIClient/CreditCounter/FileCreditCounterStore
    trio deliberately leaves persistence to the caller (see
    test_odds_api_client.py) -- this is that caller, used only here so a
    backend restart doesn't lose track of the current month's credit usage.
    """

    def __init__(self, client: OddsAPIClient, counter: CreditCounter, store: FileCreditCounterStore) -> None:
        self._client = client
        self._counter = counter
        self._store = store

    def get_odds(self, sport_key: str = "soccer_epl"):
        result = self._client.get_odds(sport_key=sport_key)
        self._store.save(self._counter)
        return result


def build_odds_client() -> OddsAPIClient | HistoricalOddsClient | None:
    """Returns W28's HistoricalOddsClient when sandbox mode is active with a
    SANDBOX_DATE set (a real historical odds source, since The Odds API is
    live-current-odds-only); otherwise the real, live OddsAPIClient -- None
    if no ODDS_API_KEY is configured."""
    override_date = sandbox_date()
    if override_date is not None:
        return HistoricalOddsClient(sandbox_date=override_date.isoformat())

    api_key = os.environ.get("ODDS_API_KEY", "")
    if not api_key:
        return None
    store = FileCreditCounterStore(CREDIT_COUNTER_PATH)
    counter = store.load()
    return PersistingOddsClient(client=OddsAPIClient(api_key=api_key, credit_counter=counter), counter=counter, store=store)


def next_day_date_str(now_fn: Callable[[], datetime] = lambda: sandbox_now(NY_TZ)) -> str:
    """Tomorrow's date in America/New_York, as the EOD job (fired at 23:00
    NY time) needs the *next* day's fixtures, not today's."""
    return (now_fn() + timedelta(days=1)).date().isoformat()


def t30_run_at(fixture: NormalizedMatch) -> datetime:
    kickoff = datetime.fromisoformat(fixture.utc_date.replace("Z", "+00:00"))
    return kickoff - timedelta(minutes=30)


def build_schedule_t30(
    scheduler: RecoverableScheduler,
    odds_client: OddsAPIClient | None,
    cache: RecommendationCache,
    config: AgentConfig,
    date_str: str,
) -> Callable[[NormalizedMatch], None]:
    def _schedule(fixture: NormalizedMatch) -> None:
        def _job() -> None:
            refresh_match_at_t30(fixture, odds_client=odds_client, cache=cache, config=config, date_str=date_str)

        scheduler.schedule_once(f"t30_{fixture.match_id}", _job, run_at=t30_run_at(fixture))

    return _schedule


def register_eod_job(
    scheduler: RecoverableScheduler,
    fixtures_client: FootballDataClient,
    odds_client: OddsAPIClient | None,
    cache: RecommendationCache,
    config: AgentConfig,
    now_fn: Callable[[], datetime] = lambda: sandbox_now(NY_TZ),
) -> None:
    """Registers the daily EOD batch job (W09) on the given scheduler.
    RecoverableScheduler.schedule_daily itself handles the restart/catch-up
    guarantee (W08) -- this just supplies the job body."""

    def _eod_job() -> None:
        date_str = next_day_date_str(now_fn)
        schedule_t30 = build_schedule_t30(scheduler, odds_client, cache, config, date_str)
        asyncio.run(
            run_eod_batch(
                fixtures_client=fixtures_client, odds_client=odds_client, cache=cache, config=config,
                schedule_t30=schedule_t30, date_str=date_str,
            )
        )

    scheduler.schedule_daily(EOD_JOB_ID, _eod_job, hour=EOD_HOUR, minute=EOD_MINUTE)
