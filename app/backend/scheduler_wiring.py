"""W08/W09/W10 wiring: connects RecoverableScheduler (W08) to the EOD batch
job (W09), whose schedule_t30 callback registers each fixture's T-30 job
(W10) on the same scheduler instance. Kept out of main.py to keep the route
module focused on HTTP concerns.
"""

from __future__ import annotations

import asyncio
from collections import Counter
from datetime import datetime, timedelta
import os
from pathlib import Path
from typing import Callable

import requests

from app.backend.eod_batch import COMPETITION_CODE, LEAGUE_CODE, run_eod_batch
from app.backend.football_data_client import FootballDataClient, NormalizedMatch
from app.backend.historical_odds_client import HistoricalOddsClient
from app.backend.live_lessons import auto_judge_live_lessons, commit_lesson_batches, prepare_lesson_batches
from app.backend.odds_api_client import CreditCounter, FileCreditCounterStore, OddsAPIClient
from app.backend.recommendation_cache import RecommendationCache
from app.backend.recommendation_outcomes import RecommendationOutcomeStore
from app.backend.scheduler import NY_TZ, RecoverableScheduler
from app.backend.sandbox_clock import sandbox_date, sandbox_now
from app.backend.sweden_fixtures_client import SwedenFixturesClient
from app.backend.t30_refresh import refresh_match_at_t30
from src.agent.agent_config import AgentConfig
from src.agent.lessons import create_lessons_tables
from src.logic.competition_registry import list_display_enabled_competition_ids
from src.utils.db_manager import DuckDBManager
from src.utils.logger import get_logger

LOGGER = get_logger(__name__)

# W163: repo-root data/, not app/data/ -- see recommendation_cache.py's W163 note.
CREDIT_COUNTER_PATH = Path(__file__).parent.parent.parent / "data" / "odds_api_credit_counter.json"
CREDIT_COUNTER_PATH_2 = Path(__file__).parent.parent.parent / "data" / "odds_api_credit_counter_2.json"
CREDIT_COUNTER_PATH_3 = Path(__file__).parent.parent.parent / "data" / "odds_api_credit_counter_3.json"
EOD_JOB_ID = "eod_batch_generation"
EOD_HOUR = 23
EOD_MINUTE = 0
LESSONS_JOB_ID = "daily_live_lessons"
LESSONS_HOUR = 6
LESSONS_MINUTE = 0
# W183-W185: judging moved off the daily job onto its own weekly one (see
# docs/superpowers/specs/2026-08-27-weekly-lesson-judging-design.md) --
# 10 minutes after the daily job's own slot, on the daily job's own
# schedule_daily trigger day, so a Sunday's own freshly-generated candidate
# is always included in that same week's review rather than deferred to
# the following week, and the weekly job's read never races the daily
# job's write for the same morning.
LESSONS_WEEKLY_JOB_ID = "weekly_live_lesson_review"
LESSONS_WEEKLY_DAY_OF_WEEK = 6  # Sunday (0=Monday..6=Sunday)
LESSONS_WEEKLY_HOUR = 6
LESSONS_WEEKLY_MINUTE = 10

# W62/W81/W140: every competition_specific league (match_info.py's
# COMPETITION_ALLOWLIST) the nightly EOD batch/T-30 refresh knows how to
# process at all. "SWE" is only actually processed when the caller supplies
# a sweden_fixtures_client; "SP1"/"I1"/"D1"/"F1" only when the caller
# supplies their own football-data.org-backed client -- omitting any (the
# default) preserves the exact pre-existing behavior for that competition.
# A competition also flipped off via config/competitions.yaml's
# display_enabled is skipped regardless of client wiring -- see _eod_job
# below.
COMPETITIONS: tuple[str, ...] = (LEAGUE_CODE, "SWE", "SP1", "I1", "D1", "F1")

# W81/W140: football-data.org's competition codes (W74/W76/W134 -- confirmed
# live, same provider as E0, unlike SWE which needs a whole separate client).
LA_LIGA_COMPETITION_CODE = "PD"
SERIE_A_COMPETITION_CODE = "SA"
BUNDESLIGA_COMPETITION_CODE = "BL1"
LIGUE_1_COMPETITION_CODE = "FL1"


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

    def get_odds(self, sport_key: str = "soccer_epl", date: str | None = None):
        # W99: found live -- the wrapped OddsAPIClient/HistoricalOddsClient
        # both accept `date` (BUG-031/W54), but this wrapper never did,
        # never forwarded it, and had no test exercising it -- so every
        # caller that passes date=... (main.py's manual regenerate path,
        # eod_batch.py's per-fixture-date fetch) crashed with a real
        # TypeError the very first time this exact production (non-sandbox)
        # code path ever ran, on this app's first live deployment. Local/
        # sandbox testing structurally never exercises this class at all --
        # build_odds_client() only returns it when SANDBOX_MODE is off.
        result = self._client.get_odds(sport_key=sport_key, date=date)
        self._store.save(self._counter)
        return result

    def get_event_odds(self, sport_key: str, event_id: str, markets: tuple[str, ...] = ("totals", "btts")):
        # W164: same persist-after-call contract as get_odds() above --
        # self._client here is OddsAPIClient specifically (not
        # HistoricalOddsClient, which has no per-event odds concept at all;
        # build_odds_client() only ever wraps OddsAPIClient in this class).
        result = self._client.get_event_odds(sport_key=sport_key, event_id=event_id, markets=markets)
        self._store.save(self._counter)
        return result


class FallbackOddsClient:
    """Tries each wrapped PersistingOddsClient (one per ODDS_API_KEY[_2] env
    var) in order, moving to the next when one is exhausted -- either
    predicted exhausted by its own local CreditCounter (get_odds() returns
    None) or actually rejected by the API (a raised RequestException, e.g.
    a truly out-of-credits or revoked key) -- so a second key can be added
    purely via env vars and picked up automatically once the first runs dry
    mid-month. Returns None (the same "keep last-known odds" convention as a
    single exhausted client) only once every client has failed."""

    def __init__(self, clients: list[PersistingOddsClient]) -> None:
        self._clients = clients

    def _try_each_client(self, call: Callable[[PersistingOddsClient], object], op_name: str):
        for i, client in enumerate(self._clients):
            try:
                result = call(client)
            except requests.RequestException as exc:
                LOGGER.warning("FallbackOddsClient.%s: key #%d failed (%s) -- trying next key.", op_name, i + 1, exc)
                continue
            if result is not None:
                return result
            LOGGER.info("FallbackOddsClient.%s: key #%d exhausted (local budget) -- trying next key.", op_name, i + 1)
        return None

    def get_odds(self, sport_key: str = "soccer_epl", date: str | None = None):
        return self._try_each_client(lambda client: client.get_odds(sport_key=sport_key, date=date), "get_odds")

    def get_event_odds(self, sport_key: str, event_id: str, markets: tuple[str, ...] = ("totals", "btts")):
        return self._try_each_client(
            lambda client: client.get_event_odds(sport_key=sport_key, event_id=event_id, markets=markets),
            "get_event_odds",
        )


def _build_persisting_odds_client(api_key: str, counter_path: Path) -> PersistingOddsClient:
    store = FileCreditCounterStore(counter_path)
    counter = store.load()
    return PersistingOddsClient(client=OddsAPIClient(api_key=api_key, credit_counter=counter), counter=counter, store=store)


def build_odds_client() -> OddsAPIClient | HistoricalOddsClient | FallbackOddsClient | None:
    """Returns W28's HistoricalOddsClient when sandbox mode is active with a
    SANDBOX_DATE set (a real historical odds source, since The Odds API is
    live-current-odds-only); otherwise the real, live OddsAPIClient(s) --
    None if no ODDS_API_KEY is configured.

    ODDS_API_KEY_2/_3, when also set, are wired in as fallbacks in order:
    each key gets its own CreditCounter file (CREDIT_COUNTER_PATH / _PATH_2 /
    _PATH_3), since credits are tracked per-key by the API itself, not shared
    across keys. ODDS_API_KEY_3 (2026-08-30, after both configured keys hit a
    real 401 Unauthorized on Serie A/Ligue 1 -- BUG-056, documents/bugs.md)
    falls back to ODDS_API_KEY's own value when unset, so local/dev setups that only
    ever configure one real key don't need a second env var just to exercise
    the 3-key code path; production sets ODDS_API_KEY_3 to a genuinely
    distinct key."""
    override_date = sandbox_date()
    if override_date is not None:
        return HistoricalOddsClient(sandbox_date=override_date.isoformat())

    primary_key = os.environ.get("ODDS_API_KEY", "")
    keys_and_paths = [
        (primary_key, CREDIT_COUNTER_PATH),
        (os.environ.get("ODDS_API_KEY_2", ""), CREDIT_COUNTER_PATH_2),
        (os.environ.get("ODDS_API_KEY_3") or primary_key, CREDIT_COUNTER_PATH_3),
    ]
    clients = [_build_persisting_odds_client(key, path) for key, path in keys_and_paths if key]
    if not clients:
        return None
    return clients[0] if len(clients) == 1 else FallbackOddsClient(clients)


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
    league: str = LEAGUE_CODE,
) -> Callable[[NormalizedMatch], None]:
    def _schedule(fixture: NormalizedMatch) -> None:
        def _job() -> None:
            refresh_match_at_t30(
                fixture, odds_client=odds_client, cache=cache, config=config,
                date_str=date_str, league=league,
            )

        scheduler.schedule_once(f"t30_{fixture.match_id}", _job, run_at=t30_run_at(fixture))

    return _schedule


def _fetch_fixtures_for_league(
    league: str,
    fixtures_client: FootballDataClient,
    sweden_fixtures_client: SwedenFixturesClient | None,
    date_str: str,
    la_liga_fixtures_client: FootballDataClient | None = None,
    serie_a_fixtures_client: FootballDataClient | None = None,
    bundesliga_fixtures_client: FootballDataClient | None = None,
    ligue1_fixtures_client: FootballDataClient | None = None,
) -> list[NormalizedMatch] | None:
    """Returns None (distinct from an empty list) when this league can't be
    attempted at all this run -- e.g. SWE requested but no
    sweden_fixtures_client configured, or SP1/I1/D1/F1 requested but their
    own football-data.org-backed client isn't configured -- so the caller
    can skip it silently rather than treating "not configured" the same as
    "0 fixtures today"."""
    if league == "SWE":
        if sweden_fixtures_client is None:
            return None
        return sweden_fixtures_client.get_fixtures(date_from=date_str, date_to=date_str)
    if league == "SP1":
        if la_liga_fixtures_client is None:
            return None
        return la_liga_fixtures_client.get_fixtures(
            competition_code=LA_LIGA_COMPETITION_CODE, date_from=date_str, date_to=date_str
        )
    if league == "I1":
        if serie_a_fixtures_client is None:
            return None
        return serie_a_fixtures_client.get_fixtures(
            competition_code=SERIE_A_COMPETITION_CODE, date_from=date_str, date_to=date_str
        )
    if league == "D1":
        if bundesliga_fixtures_client is None:
            return None
        return bundesliga_fixtures_client.get_fixtures(
            competition_code=BUNDESLIGA_COMPETITION_CODE, date_from=date_str, date_to=date_str
        )
    if league == "F1":
        if ligue1_fixtures_client is None:
            return None
        return ligue1_fixtures_client.get_fixtures(
            competition_code=LIGUE_1_COMPETITION_CODE, date_from=date_str, date_to=date_str
        )
    return fixtures_client.get_fixtures(competition_code=COMPETITION_CODE, date_from=date_str, date_to=date_str)


def register_eod_job(
    scheduler: RecoverableScheduler,
    fixtures_client: FootballDataClient,
    odds_client: OddsAPIClient | None,
    cache: RecommendationCache,
    config: AgentConfig,
    now_fn: Callable[[], datetime] = lambda: sandbox_now(NY_TZ),
    sweden_fixtures_client: SwedenFixturesClient | None = None,
    la_liga_fixtures_client: FootballDataClient | None = None,
    serie_a_fixtures_client: FootballDataClient | None = None,
    bundesliga_fixtures_client: FootballDataClient | None = None,
    ligue1_fixtures_client: FootballDataClient | None = None,
) -> None:
    """Registers the daily EOD batch job (W09) on the given scheduler.
    RecoverableScheduler.schedule_daily itself handles the restart/catch-up
    guarantee (W08) -- this just supplies the job body.

    W62/W81/W140: loops over every competition in COMPETITIONS (currently
    E0, SWE, SP1, I1, D1, and F1), fetching each one's own fixtures via the
    client that actually covers it (football-data.org for E0/SP1/I1/D1/F1 --
    the same provider/class, just a different competition_code, W74/W76/
    W134/W136 -- the Odds-API-backed sweden_fixtures_client for SWE,
    W55/W57) and running a separate run_eod_batch() per competition,
    correctly league-tagged. A fixture-fetch failure for one competition is
    caught and logged, not allowed to block the others' batch. Omitting any
    of sweden_fixtures_client/la_liga_fixtures_client/serie_a_fixtures_client/
    bundesliga_fixtures_client/ligue1_fixtures_client (all default to None)
    preserves the exact pre-existing behavior for that competition --
    silently skipped, not attempted. Confirmed live (W81) that this loop
    shape needs zero further generalization to extend from 2 to 6
    competitions -- only the client-resolution branches grow."""

    def _eod_job() -> None:
        date_str = next_day_date_str(now_fn)
        enabled = set(list_display_enabled_competition_ids())
        for league in COMPETITIONS:
            if league not in enabled:
                continue
            try:
                fixtures = _fetch_fixtures_for_league(
                    league, fixtures_client, sweden_fixtures_client, date_str,
                    la_liga_fixtures_client=la_liga_fixtures_client,
                    serie_a_fixtures_client=serie_a_fixtures_client,
                    bundesliga_fixtures_client=bundesliga_fixtures_client,
                    ligue1_fixtures_client=ligue1_fixtures_client,
                )
            except Exception:
                LOGGER.warning(
                    "EOD batch: fixture discovery failed for league=%s -- skipping this "
                    "competition for tonight, other competitions unaffected.", league, exc_info=True,
                )
                continue
            if fixtures is None:
                continue  # this league isn't configured this run (e.g. no sweden_fixtures_client)

            schedule_t30 = build_schedule_t30(scheduler, odds_client, cache, config, date_str, league=league)
            asyncio.run(
                run_eod_batch(
                    fixtures_client=fixtures_client, odds_client=odds_client, cache=cache, config=config,
                    schedule_t30=schedule_t30, date_str=date_str, fixtures=fixtures, league=league,
                )
            )

    scheduler.schedule_daily(EOD_JOB_ID, _eod_job, hour=EOD_HOUR, minute=EOD_MINUTE)


def register_lessons_job(
    scheduler: RecoverableScheduler,
    cache: RecommendationCache,
    store: RecommendationOutcomeStore,
    client: FootballDataClient,
    duckdb_manager: DuckDBManager,
    config: AgentConfig,
    sweden_client: object | None = None,
) -> None:
    """Registers two jobs (W175-W185): the daily live-lessons job resolves
    pending recommendation_outcomes (W167) and batches whatever's newly
    unbatched into agent_lessons candidates via live_lessons.py's
    prepare_lesson_batches/commit_lesson_batches, at LESSONS_HOUR (06:00 ET,
    distinct from EOD_HOUR's 23:00, and after football-data.org has
    typically posted the prior day's results) -- same schedule_daily
    restart/catch-up guarantee as the EOD job. It no longer judges anything
    itself (W183-W185): a separate weekly job (LESSONS_WEEKLY_*) judges
    every candidate still pending, grouped by (competition_id, tier), via
    auto_judge_live_lessons -- see docs/superpowers/specs/
    2026-08-27-weekly-lesson-judging-design.md for why judging moved off
    the daily cadence.

    duckdb_manager: a write-mode DuckDBManager (matches main.py's own
    `agent-lessons approve` CLI pattern) -- distinct from lessons_node's
    own read_only=True live-serving connection, since both jobs write.
    Deliberately not sandbox-routed, unlike its sibling dependencies here --
    agent_lessons is one persistent human-review queue regardless of
    SANDBOX_MODE, matching lessons_node's own always-real-path read
    behavior.

    Each job's own DuckDB connection is opened only around its brief write
    step -- prepare_lesson_batches/auto_judge_live_lessons both do all
    their network-bound (football-data.org results lookups, possibly
    rate-limited) and LLM-bound (reflection, judging) work first, with no
    DuckDB connection open at all, so data/fpai_core.db's exclusive file
    lock is never held across either (Task 4 code-quality review finding,
    unchanged by this split)."""

    def _lessons_job() -> None:
        try:
            llm_invoke = _build_lessons_llm_invoke(config)
        except Exception:
            LOGGER.warning(
                "live_lessons: could not build an LLM client -- generating stats-only "
                "lessons for today instead of failing the whole run.", exc_info=True,
            )
            llm_invoke = None

        batches = prepare_lesson_batches(cache, store, client, sweden_client, llm_invoke)

        with duckdb_manager.connection() as conn:
            create_lessons_tables(conn)
            lesson_ids = commit_lesson_batches(conn, store, batches)
        LOGGER.info("Daily live lessons: %d candidate(s) generated.", len(lesson_ids))

    def _weekly_review_job() -> None:
        try:
            llm_invoke = _build_lessons_llm_invoke(config)
        except Exception:
            LOGGER.warning(
                "live_lessons: could not build an LLM client -- skipping this week's "
                "auto-judge review (every pending candidate waits for next week's run).",
                exc_info=True,
            )
            llm_invoke = None

        judged = auto_judge_live_lessons(duckdb_manager, llm_invoke)
        action_counts = Counter(j["action"] for j in judged)
        LOGGER.info(
            "Weekly live-lesson review: %d candidate(s) auto-judged (approved=%d, rejected=%d, deferred=%d).",
            len(judged), action_counts["approve"], action_counts["reject"], action_counts["defer"],
        )

    scheduler.schedule_daily(LESSONS_JOB_ID, _lessons_job, hour=LESSONS_HOUR, minute=LESSONS_MINUTE)
    scheduler.schedule_weekly(
        LESSONS_WEEKLY_JOB_ID, _weekly_review_job,
        day_of_week=LESSONS_WEEKLY_DAY_OF_WEEK, hour=LESSONS_WEEKLY_HOUR, minute=LESSONS_WEEKLY_MINUTE,
    )


def _build_lessons_llm_invoke(config: AgentConfig) -> Callable[[str], str]:
    """Deliberate small duplication of main.py's own _build_llm_invoke --
    app/backend/ has never imported from the root main.py CLI script (nor
    vice versa); a 6-line copy is a smaller, safer diff than making this
    job the first thing to cross that boundary. Keep in sync with
    main.py's _build_llm_invoke if its shape ever changes."""
    from src.agent.graph import _build_llm, _extract_text

    llm = _build_llm(config)

    def _invoke(prompt: str) -> str:
        response = llm.invoke(prompt)
        return _extract_text(response.content)

    return _invoke
