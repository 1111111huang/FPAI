"""FastAPI backend for the FPAI web app (W01).

Imports src/agent and src/forecast directly as Python libraries -- no MCP,
no subprocess, no HTTP hop to src/mcp_server.py (that server is for external
third-party agent consumers, not this first-party app)."""

from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager
import dataclasses
from datetime import date, datetime, timedelta, timezone
import os
from pathlib import Path
import subprocess
import sys
import time
from typing import Callable, Literal

import duckdb
from dotenv import load_dotenv
from fastapi import Depends, FastAPI, HTTPException, Query, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import requests
from starlette.concurrency import run_in_threadpool
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import JSONResponse

load_dotenv()

from app.backend import bets, eod_batch, recommendations, sandbox_clock
from app.backend.agent_config_hash import compute_agent_config_hash
from app.backend.bet_tracker import BetTracker
from app.backend.bets import BetFromRecommendationRequest, BetManualRequest, BetOut
from app.backend.football_data_client import FootballDataClient, NormalizedMatch
from app.backend.odds_sport_keys import DEFAULT_SPORT_KEY, ODDS_SPORT_KEY_BY_COMPETITION
from app.backend.sweden_fixtures_client import (
    SwedenFixturesClient,
    historical_results_from_raw_matches,
)
from app.backend.llm_check import check_llm_reachable
from app.backend.recommendation_cache import DEFAULT_DB_PATH as RECOMMENDATION_CACHE_DB_PATH, RecommendationCache
from app.backend.recommendation_outcomes import (
    RecommendationOutcomeStore,
    get_recommendation_outcome_store,
    resolve_pending_recommendations,
)
from app.backend.agent_performance_dashboard import compute_agent_performance_dashboard
from app.backend.recommendation_stats import compute_recommendation_stats
from app.backend.bet_stats import compute_bet_stats
from app.backend.recommendations import MatchRecommendationOut, RecommendationRequest, validate_and_degrade
from app.backend.scheduler import JobRunLog, RecoverableScheduler
from app.backend.scheduler_wiring import build_odds_client, build_schedule_t30, register_eod_job, register_lessons_job
from app.backend.settlement import settle_open_bets
from src.agent.agent_config import AgentConfig
from src.logic.competition_registry import list_display_enabled_competition_ids
from src.tools.data_tools import get_data_freshness
from src.tools.model_tools import get_model_status
from src.utils.db_manager import DuckDBManager
from src.utils.logger import get_logger

LOGGER = get_logger(__name__)

_SANDBOX_JOB_RUNS_DB_PATH = sandbox_clock.sandbox_scoped_path("job_runs.db")

_fixtures_client: FootballDataClient | None = None


def get_fixtures_client() -> FootballDataClient:
    """FastAPI dependency -- overridden in tests via patching this function."""
    global _fixtures_client
    if _fixtures_client is None:
        _fixtures_client = FootballDataClient(api_key=os.environ.get("FOOTBALL_DATA_API_KEY", ""))
    return _fixtures_client


def get_la_liga_fixtures_client() -> FootballDataClient:
    """W76: La Liga uses the *same* football-data.org provider/class as E0 --
    live-verified (W74) the `PD` competition code returns real La Liga
    fixtures, unlike SWE which needed an entirely separate Odds-API-backed
    client (football-data.org has no Allsvenskan coverage at all). This
    thin wrapper around the same singleton exists purely so tests can mock
    La Liga's calls independently of E0's own (mirroring
    get_sweden_fixtures_client()'s test-isolation rationale) without paying
    for a second real HTTP client/session -- both accessors return the same
    underlying FootballDataClient instance in production."""
    return get_fixtures_client()


LA_LIGA_COMPETITION_CODE = "PD"


def get_serie_a_fixtures_client() -> FootballDataClient:
    """W136: Serie A uses the same football-data.org provider/class as
    E0/SP1 (live-confirmed, W134) -- a thin wrapper purely for test/cache
    isolation, mirroring get_la_liga_fixtures_client() exactly."""
    return get_fixtures_client()


SERIE_A_COMPETITION_CODE = "SA"


def get_bundesliga_fixtures_client() -> FootballDataClient:
    """W136: Bundesliga uses the same football-data.org provider/class as
    E0/SP1/I1 (live-confirmed, W134)."""
    return get_fixtures_client()


BUNDESLIGA_COMPETITION_CODE = "BL1"


def get_ligue1_fixtures_client() -> FootballDataClient:
    """W136: Ligue 1 uses the same football-data.org provider/class as
    E0/SP1/I1/D1 (live-confirmed, W134)."""
    return get_fixtures_client()


LIGUE_1_COMPETITION_CODE = "FL1"


_sweden_fixtures_client: SwedenFixturesClient | None = None


def get_sweden_fixtures_client() -> SwedenFixturesClient:
    """W57: Sweden (Allsvenskan) fixtures/results, sourced from The Odds API
    rather than football-data.org -- W55 confirmed football-data.org's free
    tier has no Allsvenskan coverage at all. FastAPI dependency -- overridden
    in tests via patching this function, same pattern as get_fixtures_client."""
    global _sweden_fixtures_client
    if _sweden_fixtures_client is None:
        _sweden_fixtures_client = SwedenFixturesClient(api_key=os.environ.get("ODDS_API_KEY", ""))
    return _sweden_fixtures_client


# W52: football-data.org's free tier (~10 req/min) is shared across every
# request through the single `_fixtures_client` singleton above. Three
# independent frontend call sites (Dashboard, Match Explorer, manual bet
# form) each fetch fixtures fresh on every mount with no de-duplication, so
# normal navigation within a session can burst well past the budget and trip
# a 429. This is a short-lived request-dedup cache (not a data-freshness
# cache) -- 60s is short enough that staleness is a non-issue, long enough to
# absorb that exact repeated-navigation pattern. Module-level state, matching
# the existing `_fixtures_client` singleton pattern already in this file.
_FIXTURE_CACHE_TTL_SECONDS = 60.0
_fixture_cache: dict[tuple[str, str | None, str | None], tuple[float, list[NormalizedMatch]]] = {}

# Code review follow-up (post-21e6bf9): a bare cache dict only de-dupes
# requests that arrive *after* an earlier one has already completed and
# populated the cache -- two genuinely concurrent cache-miss requests for
# the same key (e.g. React StrictMode's double-effect-invocation in dev, or
# two browser tabs loading at once) would both race past the cache check
# and both hit the upstream client for real. This tracks the in-flight
# asyncio.Task for each key so a second concurrent request awaits the same
# task instead of starting its own.
_fixture_cache_pending: dict[tuple[str, str | None, str | None], "asyncio.Task[list[NormalizedMatch]]"] = {}


def _fixture_cache_now() -> float:
    """Split out from _cached_fixture_call so tests can monkeypatch the
    clock -- mirrors this file's existing `_current_real_date()` patchable-
    function pattern -- to deterministically exercise TTL expiry without a
    real 60-second sleep."""
    return time.monotonic()


async def _fetch_and_cache_fixtures(
    cache_key: tuple[str, str | None, str | None],
    fetch: Callable[..., list[NormalizedMatch]],
    fetch_kwargs: dict[str, str | None],
) -> list[NormalizedMatch]:
    """Runs the actual upstream call (in the threadpool) and populates the
    cache on success. A requests.HTTPError raised by `fetch` (e.g.
    football-data.org's 429 rate-limit response) is turned into a clean 503
    HTTPException rather than left to propagate as an unhandled 500 -- and
    is NOT cached, so it doesn't wrongly suppress the next genuine
    request."""
    try:
        matches = await run_in_threadpool(fetch, **fetch_kwargs)
    except requests.exceptions.HTTPError as exc:
        LOGGER.warning("Upstream fixture provider call failed for %s: %s", cache_key, exc, exc_info=True)
        raise HTTPException(
            status_code=503,
            detail=(
                "Fixture data is temporarily unavailable (the upstream provider is rate-limited "
                "or unreachable). Please try again in a minute."
            ),
        ) from exc

    _fixture_cache[cache_key] = (_fixture_cache_now() + _FIXTURE_CACHE_TTL_SECONDS, matches)
    return matches


async def _cached_fixture_call(
    cache_key: tuple[str, str | None, str | None],
    fetch: Callable[..., list[NormalizedMatch]],
    **fetch_kwargs: str | None,
) -> list[NormalizedMatch]:
    """Look up `cache_key` in the module-level TTL cache; on a miss, either
    join an already-in-flight call for the same key (`_fixture_cache_pending`)
    or kick off a new one. The pending task is registered *before* the first
    `await` inside it runs -- since asyncio is single-threaded/cooperative,
    a second concurrent call checking `_fixture_cache_pending` between that
    registration and the task's completion is guaranteed to see it, closing
    the race a bare cache dict would leave open."""
    cached = _fixture_cache.get(cache_key)
    if cached is not None:
        expires_at, matches = cached
        if expires_at > _fixture_cache_now():
            return matches

    pending = _fixture_cache_pending.get(cache_key)
    if pending is None:
        pending = asyncio.ensure_future(_fetch_and_cache_fixtures(cache_key, fetch, fetch_kwargs))
        _fixture_cache_pending[cache_key] = pending

    try:
        return await pending
    finally:
        if _fixture_cache_pending.get(cache_key) is pending:
            del _fixture_cache_pending[cache_key]


_PREGENERATE_DEFAULT_DAYS_AHEAD = 3
# BUG-045: eod_batch.run_eod_batch()'s own default (5) is fine for the
# scheduled nightly EOD job, which runs hours into a long-stable process.
# Pregenerate (below) instead runs immediately inside lifespan's startup
# hook, stacking up to `concurrency` concurrent asyncio.to_thread() agent
# calls -- each loading its own ForecastService models/feature frames --
# directly on top of the process's own fresh, not-yet-settled import-time
# memory. Confirmed live (2026-08-13, Railway "Deploy Ran Out Of Memory"
# crash-loop, container memory limit 4GB): every boot re-fires pregenerate,
# so a single OOM becomes an infinite crash loop, never completing one
# pregenerate pass. Lower, not equal to eod_batch's default -- deliberately
# narrower than the value already proven safe for the nightly path.
#
# W165: 5 -> 3 (was memory-safety-only, never actually sized to what's
# shown). Direct user point: no reason to eagerly regenerate a fixture that
# won't reach the frontend soon -- MatchUI.tsx's Dashboard caps its own
# display at the next 10 upcoming fixtures regardless of date
# ("DashboardPage/MatchExplorerPage... next 10 matches going forward").
# Confirmed live (2026-08-25) that fixture density right now needs exactly
# 3 days to reach 10 fixtures across every league (2/4/10/47 fixtures at
# days_ahead=1/2/3/5 respectively) -- 3 covers what's actually visible
# without the other 37 fixtures 4-5 days out never getting looked at
# regardless (EOD picks each of them up the night before their own date;
# T-30 covers each again right before its own kickoff). Cuts pregenerate's
# real-money LLM call count and Odds-API credit spend by ~79% on a typical
# boot without leaving a currently-visible card blank. The admin endpoint
# (POST /api/admin/pregenerate-recommendations?days_ahead=N) still accepts
# a larger explicit override for a deliberate wider backfill (e.g. ahead of
# a busy weekend) -- only the automatic boot-time default shrank.
_PREGENERATE_DEFAULT_CONCURRENCY = 2
_background_tasks: set[asyncio.Task] = set()


def _fire_and_forget(coro) -> None:
    """W103: asyncio.create_task()'s own docs warn a task with no other
    reference can be garbage-collected mid-execution -- this keeps one
    (in a module-level set, discarded via the task's own done-callback)
    so a fire-and-forget background job (pregenerate, below) can't
    silently vanish partway through."""
    task = asyncio.create_task(coro)
    _background_tasks.add(task)
    task.add_done_callback(_background_tasks.discard)


async def _pregenerate_recommendations(
    days_ahead: int = _PREGENERATE_DEFAULT_DAYS_AHEAD,
    scheduler: RecoverableScheduler | None = None,
    concurrency: int = _PREGENERATE_DEFAULT_CONCURRENCY,
) -> dict:
    """W103: generates and caches recommendations for real upcoming
    fixtures across every competition right now, reusing eod_batch.py's
    own bounded-concurrency/per-match-error-isolation machinery (the exact
    same code the real nightly EOD job uses, via run_eod_batch's own
    `fixtures=` override -- W50's sandbox-precompute caller established
    this same pattern) instead of a separate mechanism.

    Two callers: the admin trigger endpoint (any time, independent of
    whether the scheduler is on) and lifespan's own startup hook (every
    boot -- i.e. every redeploy, direct user request, so a freshly
    deployed instance is never left showing blank cards for fixtures
    already close enough to matter). `scheduler=None` (the startup hook's
    own case when ENABLE_SCHEDULER is off) degrades gracefully -- matches
    are still generated and cached now, just without the later T-30
    refresh safety net registered for them, rather than failing outright.

    BUG-045: `concurrency` defaults lower than eod_batch.run_eod_batch()'s
    own default (5) -- see _PREGENERATE_DEFAULT_CONCURRENCY's comment.
    Every league still runs sequentially (this loop `await`s one league's
    whole batch before starting the next), so this bounds how many
    concurrent agent calls stack on top of a freshly booted process at any
    one instant, not how many run in total."""
    from datetime import timedelta

    today = _current_real_date()
    date_from = today.isoformat()
    date_to = (today + timedelta(days=days_ahead)).isoformat()
    # W179: found live (2026-08-27) -- get_fixtures() deliberately raises a
    # clean HTTPException(503) on an upstream 429/outage (see
    # _fetch_and_cache_fixtures's own docstring), the right contract for its
    # primary caller (the /api/fixtures HTTP route). But this caller is a
    # fire-and-forget background task (_fire_and_forget), not a request --
    # there's no handler to turn that HTTPException into a response, so it
    # propagated as an unhandled exception ("Task exception was never
    # retrieved"), skipping every league's pregenerate for the whole boot
    # even though only one upstream competition call actually failed.
    # Isolated the same way the per-league loop below already isolates a
    # single league's own eod_batch failure from every other league's.
    try:
        fixtures = await get_fixtures(date_from=date_from, date_to=date_to)
    except Exception:
        LOGGER.warning(
            "Pregenerate: get_fixtures failed (days_ahead=%d) -- upstream fixture provider "
            "unavailable, skipping this pregenerate pass entirely.", days_ahead, exc_info=True,
        )
        return {}

    fixtures_by_league: dict[str, list] = {}
    for fixture in fixtures:
        fixtures_by_league.setdefault(fixture.competition or "E0", []).append(fixture)

    odds_client = build_odds_client()
    cache = recommendations.get_cache()
    config = AgentConfig.default()
    results: dict[str, dict] = {}
    for league, league_fixtures in fixtures_by_league.items():
        if not league_fixtures:
            continue
        schedule_t30 = (
            build_schedule_t30(scheduler, odds_client, cache, config, date_from, league=league)
            if scheduler is not None
            else (lambda fixture: None)
        )
        try:
            result = await eod_batch.run_eod_batch(
                fixtures_client=get_fixtures_client(), odds_client=odds_client, cache=cache, config=config,
                schedule_t30=schedule_t30, date_str=date_from, fixtures=league_fixtures, league=league,
                concurrency=concurrency,
            )
        except Exception:
            LOGGER.warning("Pregenerate: batch failed for league=%s -- other leagues unaffected.", league, exc_info=True)
            continue
        results[league] = {"generated": result.generated, "skipped": result.skipped, "unchanged": result.unchanged}
    LOGGER.info("Pregenerate complete | days_ahead=%d | concurrency=%d | results=%s", days_ahead, concurrency, results)
    return results


@asynccontextmanager
async def lifespan(app: FastAPI):
    config = AgentConfig.default()
    if not check_llm_reachable(config):
        LOGGER.warning(
            "LLM provider '%s' (model=%s) does not appear reachable at startup -- "
            "recommendation generation will fail until this is resolved.",
            config.provider, config.model,
        )

    # W08/W09/W10: off by default. Registering the EOD job runs
    # RecoverableScheduler's restart/catch-up check immediately (by design,
    # W08's whole point) -- if wired in unconditionally, every test file's
    # `with TestClient(app)` would non-deterministically trigger a real live
    # EOD batch (real fixtures/odds/LLM calls) depending on whatever wall-clock
    # time tests happen to run at relative to the 23:00 NY trigger. Gated
    # behind an explicit opt-in until this wave is verified live and the app
    # is actually going to production -- see app_user_stories.md W08 sequencing
    # note ("built last, right before going live").
    scheduler: RecoverableScheduler | None = None
    if os.environ.get("ENABLE_SCHEDULER", "").lower() in ("1", "true", "yes"):
        # W29: sandbox mode routes JobRunLog to a scratch path so it never touches real dev data.
        run_log = JobRunLog(db_path=_SANDBOX_JOB_RUNS_DB_PATH) if sandbox_clock.is_sandbox_mode() else None
        scheduler = RecoverableScheduler(run_log=run_log)
        register_eod_job(
            scheduler,
            fixtures_client=get_fixtures_client(),
            odds_client=build_odds_client(),
            cache=recommendations.get_cache(),
            config=config,
            # W62: Sweden (Allsvenskan) processed alongside EPL in the same
            # nightly EOD batch / T-30 refresh, sourced from The Odds API
            # (W55/W57) rather than football-data.org.
            sweden_fixtures_client=get_sweden_fixtures_client(),
            # W81: La Liga processed alongside E0/SWE in the same nightly
            # EOD batch / T-30 refresh, sourced from football-data.org --
            # the same provider/class as E0 (W74/W76), unlike SWE.
            la_liga_fixtures_client=get_la_liga_fixtures_client(),
            # W140: Serie A/Bundesliga/Ligue 1 processed the same way as
            # La Liga -- football-data.org, same provider/class (W134/W136).
            serie_a_fixtures_client=get_serie_a_fixtures_client(),
            bundesliga_fixtures_client=get_bundesliga_fixtures_client(),
            ligue1_fixtures_client=get_ligue1_fixtures_client(),
        )
        register_lessons_job(
            scheduler,
            cache=recommendations.get_cache(),
            store=get_recommendation_outcome_store(),
            client=get_fixtures_client(),
            duckdb_manager=DuckDBManager(),
            config=config,
            sweden_client=get_sweden_fixtures_client(),
        )
        scheduler.start()
        LOGGER.info("W08/W09/W10 scheduler started (ENABLE_SCHEDULER=1).")

        # W103: direct user request -- every boot with the scheduler on
        # (i.e. every future redeploy, since ENABLE_SCHEDULER persists as a
        # Railway env var once set) pre-generates recommendations for the
        # next few days' real fixtures immediately, rather than leaving a
        # freshly deployed instance showing blank "not yet generated" cards
        # until whichever fixture's own EOD/T-30 cycle happens to fire.
        # Deliberately gated behind the same ENABLE_SCHEDULER check as the
        # scheduler itself (not unconditional) -- an unconditional hook here
        # would fire on every test file's `with TestClient(app)` too, the
        # exact class of problem that made the scheduler registration above
        # opt-in in the first place (W08/W09's own comment, unchanged).
        _fire_and_forget(_pregenerate_recommendations(scheduler=scheduler))
    else:
        LOGGER.info("Scheduler disabled -- set ENABLE_SCHEDULER=1 to enable the EOD/T-30 pipeline.")

    app.state.scheduler = scheduler

    yield

    if scheduler is not None:
        scheduler.shutdown()


app = FastAPI(title="FPAI Web App Backend", lifespan=lifespan)


class RequireAppTokenMiddleware(BaseHTTPMiddleware):
    """W97: gates every request behind a shared-secret header once the app
    is reachable from the public internet, not just localhost. Off by
    default (APP_ACCESS_TOKEN unset) so every existing test and local-dev
    run is completely unaffected -- same "off unless explicitly opted in"
    pattern already used for ENABLE_SCHEDULER. /api/health is exempt so a
    hosting platform's own health check (which never sends this header)
    doesn't get treated as a deploy failure.

    Deliberately added to the middleware stack *before* CORSMiddleware --
    Starlette wraps middleware in the reverse of add_middleware() call
    order (confirmed empirically, not assumed: the *last*-added one ends
    up outermost), so CORSMiddleware being added after this one makes it
    the outermost layer. That matters twice: (1) a real browser's CORS
    preflight (OPTIONS) request never reaches this check at all -- it's
    intercepted and answered by CORSMiddleware first; (2) a 401 this
    middleware returns still gets proper CORS headers attached on the way
    back out, so a browser reports the real 401 instead of a confusing,
    header-less CORS error masking it."""

    async def dispatch(self, request, call_next):
        token = os.environ.get("APP_ACCESS_TOKEN")
        if not token or request.method == "OPTIONS" or request.url.path == "/api/health":
            return await call_next(request)
        if request.headers.get("x-app-token") != token:
            return JSONResponse({"detail": "Unauthorized"}, status_code=401)
        return await call_next(request)


app.add_middleware(RequireAppTokenMiddleware)


def _parse_cors_origins(raw: str) -> list[str]:
    """Comma-separated env value -> allow_origins list, dropping blanks (a
    trailing comma or empty value shouldn't silently become a "" origin,
    which CORSMiddleware would never match anyway but is still worth not
    emitting)."""
    return [origin.strip() for origin in raw.split(",") if origin.strip()]


# D7: standard two-process local dev -- Next.js dev server + uvicorn, talking
# over HTTP with CORS rather than a shared process. W97: origins now
# env-driven (CORS_ALLOWED_ORIGINS, comma-separated) so a deployed frontend's
# real origin can be allowed without a code change -- defaults to the
# original localhost-only value, unchanged behavior for local dev. Read once
# at import time (like any other env-driven app config) -- not re-read
# per-request, unlike RequireAppTokenMiddleware's token check above.
app.add_middleware(
    CORSMiddleware,
    allow_origins=_parse_cors_origins(os.environ.get("CORS_ALLOWED_ORIGINS", "http://localhost:3000")),
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/api/health")
def health() -> dict:
    return {"status": "ok"}


_RESTORE_TARGET_NAMES = Literal["fpai_core", "recommendation_cache"]


def _restore_target_path(target: _RESTORE_TARGET_NAMES) -> Path:
    """Lazy, not module-level -- DuckDBManager() reads config.yaml at
    construction time, and this keeps that (and any failure from a missing/
    invalid config) scoped to an actual call of this endpoint, not import
    time for every other route."""
    if target == "fpai_core":
        return DuckDBManager().db_path
    return RECOMMENDATION_CACHE_DB_PATH


@app.post("/api/admin/restore-database")
async def restore_database(target: _RESTORE_TARGET_NAMES, source_url: str) -> dict:
    """W100: first deployment has no reliable way to push a large local
    file directly onto a hosting platform's remote persistent volume, so
    this instead has the already-running, already-deployed app pull one
    itself from a URL you control (e.g. a GitHub Release asset -- handles
    large files, unlike a git commit, which data/fpai_core.db is
    deliberately excluded from, see .gitignore). Already protected by
    RequireAppTokenMiddleware like every other non-health route -- no
    separate auth needed. `target` is a fixed allowlist, not an arbitrary
    path, and `source_url` must be https -- this can't be pointed at an
    unintended file or source."""
    if not source_url.startswith("https://"):
        raise HTTPException(status_code=400, detail="source_url must start with https://")

    target_path = _restore_target_path(target)
    target_path.parent.mkdir(parents=True, exist_ok=True)

    def _download() -> int:
        with requests.get(source_url, stream=True, timeout=600) as response:
            response.raise_for_status()
            with open(target_path, "wb") as f:
                for chunk in response.iter_content(chunk_size=8 * 1024 * 1024):
                    f.write(chunk)
        return target_path.stat().st_size

    bytes_written = await run_in_threadpool(_download)
    return {"target": target, "path": str(target_path), "bytes_written": bytes_written}


class _LessonSyncItem(BaseModel):
    """Fields needed to reproduce an already-approved lesson on a different
    database -- deliberately not the full agent_lessons row shape (id,
    created_at, etc. are meaningless across databases); these four are
    exactly what load_approved_lessons() ever reads at serving time."""
    competition_id: str
    tier: str
    scope: Literal["competition", "tier"]
    rule_text: str
    lesson_text: str | None = None
    source_match_id: str = "synced-from-local-review"
    reviewer: str = "sync"


@app.post("/api/admin/sync-lessons")
def sync_lessons(lessons: list[_LessonSyncItem]) -> dict:
    """A lesson approved against a local/dev database has no path to a
    deployed instance's own agent_lessons table -- it's not part of git
    (data/fpai_core.db is gitignored), and /api/admin/restore-database's
    full-file overwrite is the wrong tool: agent_lessons lives in the same
    physical file as raw_matches/feature_store, so overwriting the whole
    file to sync one lesson row would also clobber production's own
    (fresher) match data. This inserts/upserts *only* into agent_lessons,
    reusing insert_lesson_candidate/approve_lesson (src/agent/lessons.py)
    rather than re-deriving the table's insert shape here.

    Idempotent by content, not by any transferred id (ids aren't meaningful
    across databases) -- skips a lesson if an approved row with the exact
    same (scope, competition_id, tier, rule_text) already exists, so
    re-running this with the same payload is always safe."""
    from src.agent.lessons import approve_lesson, create_lessons_tables, insert_lesson_candidate

    db = DuckDBManager()
    inserted = 0
    skipped = 0
    with db.connection() as conn:
        create_lessons_tables(conn)
        for lesson in lessons:
            duplicate = conn.execute(
                "SELECT COUNT(*) FROM agent_lessons WHERE status = 'approved' "
                "AND scope = ? AND competition_id = ? AND tier = ? AND rule_text = ?",
                [lesson.scope, lesson.competition_id, lesson.tier, lesson.rule_text],
            ).fetchone()[0]
            if duplicate:
                skipped += 1
                continue
            lesson_id = insert_lesson_candidate(
                conn, lesson.lesson_text or lesson.rule_text,
                lesson.competition_id, lesson.tier, lesson.source_match_id,
            )
            approve_lesson(conn, lesson_id, lesson.scope, lesson.reviewer, lesson.rule_text)
            inserted += 1
    return {"inserted": inserted, "skipped_duplicates": skipped}


@app.get("/api/admin/lessons")
def list_lessons(
    status: Literal["pending", "approved", "rejected"] | None = None,
    source: Literal["train", "live"] | None = None,
    limit: int = Query(50, ge=1, le=500),
) -> dict:
    """Read-only browse of agent_lessons, newest first -- there was no way to
    see what the daily live-lessons job (register_lessons_job) has actually
    been writing in a deployed instance short of a raw DB query; this mirrors
    /api/admin/sync-lessons' own DuckDBManager/create_lessons_tables pattern
    but only ever issues a SELECT -- not opened read_only since
    create_lessons_tables' CREATE TABLE IF NOT EXISTS/ALTER TABLE guard (a
    no-op once the table already exists, which it always will in a real
    deployment) needs a writable connection to run at all. status/source
    filter with plain SQL equality (both are already-columns, no derived
    logic); omitting either returns every value including legacy NULL
    source rows (pre-2026-08-26 rows never had one set)."""
    from src.agent.lessons import create_lessons_tables

    db = DuckDBManager()
    with db.connection() as conn:
        create_lessons_tables(conn)
        clauses, params = [], []
        if status is not None:
            clauses.append("status = ?")
            params.append(status)
        if source is not None:
            clauses.append("source = ?")
            params.append(source)
        where = f"WHERE {' AND '.join(clauses)}" if clauses else ""
        params.append(limit)
        rows = conn.execute(
            f"""
            SELECT id, lesson_text, rule_text, status, competition_id, tier, scope,
                   source_match_id, source, created_at, reviewed_at, reviewer,
                   auto_decision_reasoning
            FROM agent_lessons {where} ORDER BY created_at DESC LIMIT ?
            """,
            params,
        ).fetchall()
    columns = (
        "id", "lesson_text", "rule_text", "status", "competition_id", "tier", "scope",
        "source_match_id", "source", "created_at", "reviewed_at", "reviewer",
        "auto_decision_reasoning",
    )
    return {"lessons": [dict(zip(columns, row)) for row in rows]}


_REFRESH_LEAGUE_NAMES = Literal["E0", "SP1", "SWE", "I1", "D1", "F1"]
_REPO_ROOT = Path(__file__).resolve().parents[2]


@app.post("/api/admin/trigger-data-refresh")
def trigger_data_refresh(league: _REFRESH_LEAGUE_NAMES) -> dict:
    """W101: manually triggers the ML-engine's own standalone refresh-data
    chain (scrape -> ingest -> fetch-understat -> fetch-fotmob ->
    lineup-backfill -- main.py's run_refresh_data) for one league, from
    the deployed app itself. There's no cron/scheduler wired up for this
    yet on this deployment (main.py's own schedule-refresh is a separate
    standalone process, never deployed here) -- this is a manual trigger
    for now, not a replacement for setting that up properly later.

    Launched as a genuinely separate OS process (subprocess), deliberately
    NOT an in-process function call: main.py's own top-level imports
    (mlflow, Optuna, and other training-only machinery needed for its
    other subcommands) would otherwise get pulled into this always-running
    web server process just by importing the file -- exactly the class of
    avoidable memory bloat W100's OOM investigation just confirmed isn't
    currently present in this server's own import graph. The subprocess
    gets that memory and releases all of it back on exit; the server's own
    footprint never grows. Returns immediately (refresh-data itself can
    take several minutes of real scraping/xG-matching/lineup-fetching --
    too long for one HTTP request/response); check GET /api/status
    afterward to see when data_freshness reflects the update. Already
    protected by RequireAppTokenMiddleware.

    Deliberately does NOT pass stdout=/stderr=PIPE: found live (2026-08-10)
    that doing so silently redirects the subprocess's output away from
    Railway's own log capture (nothing ever reads that pipe, so nothing
    shows up in Deploy Logs), and worse, is a classic subprocess deadlock
    risk -- once the child writes enough output to fill the OS pipe buffer
    (commonly ~64KB) with nobody draining it, its next write() call blocks
    forever, silently hanging the refresh indefinitely. Leaving stdout/
    stderr unset makes the subprocess inherit this process's own file
    descriptors instead, so its output flows straight into the same
    stream Railway already captures, with no pipe to fill."""
    process = subprocess.Popen(
        [sys.executable, "main.py", "refresh-data", "--league", league],
        cwd=_REPO_ROOT,
    )
    return {"league": league, "pid": process.pid, "status": "started"}


@app.post("/api/admin/pregenerate-recommendations")
async def pregenerate_recommendations_endpoint(
    request: Request, days_ahead: int = _PREGENERATE_DEFAULT_DAYS_AHEAD
) -> dict:
    """W103: manual trigger for _pregenerate_recommendations (see its own
    docstring) -- independent of whether the scheduler is on, and doesn't
    wait for the (multi-minute, one real LLM+odds call per fixture) work
    to finish before responding, same reasoning as the other admin
    triggers. Reads the scheduler lifespan stored on app.state (None if
    ENABLE_SCHEDULER is off) so a pregenerated match still gets its T-30
    refresh scheduled when a scheduler is actually running. Already
    protected by RequireAppTokenMiddleware."""
    scheduler = getattr(request.app.state, "scheduler", None)
    _fire_and_forget(_pregenerate_recommendations(days_ahead=days_ahead, scheduler=scheduler))
    return {"days_ahead": days_ahead, "status": "started"}


@app.get("/api/sandbox/status")
def get_sandbox_status() -> dict:
    """W27: lets the frontend and test scripts introspect the active
    sandbox date instead of each needing their own access to the env vars."""
    return sandbox_clock.sandbox_status()


@app.get("/api/status")
def get_status() -> dict:
    """W17: system status surface (data staleness + current model
    selections) -- pure reuse of already-exposed src/tools functions, no
    new engine work.

    W104 follow-up: get_data_freshness()'s by_league breakdown is
    engine-side and league-complete by design (src/tools/ is also consumed
    by the agent/CLI, which must see every competition regardless of the
    app's own display toggle) -- so a display-disabled competition (e.g.
    SWE) is filtered out here, at the app boundary, the same "app keeps its
    own view independent of the engine's registry" split this app already
    uses elsewhere (Section 4, app_techspec.md). Only by_league is touched
    -- the only field that renders a literal competition label in the UI
    (AppShell's sidebar); the blended top-level freshness numbers are left
    as engine-wide, matching the existing "top-level = whole table" contract."""
    freshness = get_data_freshness()
    if "by_league" in freshness:
        enabled = set(list_display_enabled_competition_ids())
        freshness = {**freshness, "by_league": {k: v for k, v in freshness["by_league"].items() if k in enabled}}
    return {"data_freshness": freshness, "model_status": get_model_status()}


def _current_real_date() -> date:
    """Genuine wall-clock UTC 'today' -- deliberately NOT
    sandbox_clock.sandbox_now(). football-data.org's own SCHEDULED/FINISHED
    match status reflects whether a match has actually been played in the
    real world, independent of what date SANDBOX_MODE/SANDBOX_DATE is
    pretending "today" is, so the split below must be computed against real
    time, not the app's sandbox-overridable clock."""
    return datetime.now(timezone.utc).date()


def _split_fixture_date_range(
    date_from: str | None, date_to: str | None, today: date,
) -> tuple[tuple[str, str] | None, tuple[str | None, str | None] | None]:
    """Splits a requested [date_from, date_to] range into a (results_range,
    fixtures_range) pair relative to real wall-clock `today`, so the caller
    can source the past portion from get_results() (status=FINISHED) and the
    future portion from get_fixtures() (status=SCHEDULED) -- either element
    is None if that side contributes nothing. A range that includes `today`
    queries `today` from *both* sides, since a same-day match may already be
    FINISHED or still SCHEDULED depending on kickoff time relative to right
    now -- there is no way to know which without asking both.

    Either bound omitted (or unparseable) falls back to the pre-W45
    single-call behavior: everything goes through fixtures_range unchanged,
    since a half-open/unbounded range can't be meaningfully split against a
    boundary, and this preserves however football-data.org's API itself
    already interprets missing date params by default.
    """
    if date_from is None or date_to is None:
        return None, (date_from, date_to)
    try:
        parsed_from = date.fromisoformat(date_from)
        parsed_to = date.fromisoformat(date_to)
    except ValueError:
        return None, (date_from, date_to)

    if parsed_to < today:
        return (date_from, date_to), None
    if parsed_from > today:
        return None, (date_from, date_to)

    # Range includes today: query today from *both* sides. A match kicking
    # off today may already be FINISHED (e.g. an early kickoff, viewed later
    # the same day) or still SCHEDULED, depending on kickoff time relative to
    # right now -- excluding today from get_results() would silently drop
    # already-finished same-day matches, which is exactly the gap this fix
    # exists to close (found in code review of the first version of this
    # function, which stopped the results side at yesterday).
    return (date_from, today.isoformat()), (today.isoformat(), date_to)


@app.get("/api/fixtures")
async def get_fixtures(date_from: str | None = None, date_to: str | None = None) -> list[NormalizedMatch]:
    """Thin wrapper over W05's FootballDataClient -- gives the frontend a real
    fixture list to render (Dashboard/Match Explorer). Still used directly
    even after W09 shipped -- W09 populates the recommendation cache in the
    background, but the frontend still needs a live fixture list to render
    cards for, cached recommendation or not.

    W45: a date range entirely in the past needs get_results()
    (status=FINISHED) -- get_fixtures() (status=SCHEDULED) structurally can
    never return anything for a range that's already happened, regardless of
    real data availability. A range spanning real "today" is split and the
    two sides merged, past-then-future (already chronological, since the
    split is on whole days and each side is independently date-ordered by
    the API)."""
    client = get_fixtures_client()
    sweden_client = get_sweden_fixtures_client()
    la_liga_client = get_la_liga_fixtures_client()
    serie_a_client = get_serie_a_fixtures_client()
    bundesliga_client = get_bundesliga_fixtures_client()
    ligue1_client = get_ligue1_fixtures_client()
    results_range, fixtures_range = _split_fixture_date_range(date_from, date_to, _current_real_date())
    # display_enabled gate (config/competitions.yaml): a competition flipped
    # off there is skipped here entirely -- not just filtered out after the
    # fact -- so a disabled competition's fixture/odds calls are never even
    # made. Same registry flag also gates the nightly EOD/T-30 batch
    # (scheduler_wiring.COMPETITIONS filtering), so one edit turns a
    # competition off everywhere it's featured to users.
    enabled = set(list_display_enabled_competition_ids())

    def _tag(matches: list[NormalizedMatch], competition: str) -> list[NormalizedMatch]:
        # W64: explicit tagging here (not inside each client's own
        # normalize function) is deliberate -- this is the one place every
        # return path through this endpoint passes through, real or
        # test-mocked, so it's the only place a tag is guaranteed to stick.
        return [dataclasses.replace(m, competition=competition) for m in matches]

    matches: list[NormalizedMatch] = []
    if results_range is not None:
        past_from, past_to = results_range
        if "E0" in enabled:
            matches += _tag(
                await _cached_fixture_call(
                    ("results", past_from, past_to), client.get_results, date_from=past_from, date_to=past_to
                ),
                "E0",
            )
        if "SWE" in enabled:
            # W71: sourced from raw_matches directly, not sweden_client.get_results()
            # (The Odds API's /scores endpoint can only see the last few real
            # days -- it has no arbitrary-historical-date capability at all,
            # unlike football-data.org's get_results() for E0). Still
            # cache-keyed as "results_swe" -- same TTL-cache slot as before,
            # just backed by a different underlying source.
            matches += _tag(
                await _cached_fixture_call(
                    ("results_swe", past_from, past_to),
                    historical_results_from_raw_matches, date_from=past_from, date_to=past_to,
                ),
                "SWE",
            )
        if "SP1" in enabled:
            matches += _tag(
                await _cached_fixture_call(
                    ("results_sp1", past_from, past_to), la_liga_client.get_results,
                    competition_code=LA_LIGA_COMPETITION_CODE, date_from=past_from, date_to=past_to,
                ),
                "SP1",
            )
        if "I1" in enabled:
            matches += _tag(
                await _cached_fixture_call(
                    ("results_i1", past_from, past_to), serie_a_client.get_results,
                    competition_code=SERIE_A_COMPETITION_CODE, date_from=past_from, date_to=past_to,
                ),
                "I1",
            )
        if "D1" in enabled:
            matches += _tag(
                await _cached_fixture_call(
                    ("results_d1", past_from, past_to), bundesliga_client.get_results,
                    competition_code=BUNDESLIGA_COMPETITION_CODE, date_from=past_from, date_to=past_to,
                ),
                "D1",
            )
        if "F1" in enabled:
            matches += _tag(
                await _cached_fixture_call(
                    ("results_f1", past_from, past_to), ligue1_client.get_results,
                    competition_code=LIGUE_1_COMPETITION_CODE, date_from=past_from, date_to=past_to,
                ),
                "F1",
            )
    if fixtures_range is not None:
        future_from, future_to = fixtures_range
        if "E0" in enabled:
            matches += _tag(
                await _cached_fixture_call(
                    ("fixtures", future_from, future_to), client.get_fixtures, date_from=future_from, date_to=future_to
                ),
                "E0",
            )
        if "SWE" in enabled:
            matches += _tag(
                await _cached_fixture_call(
                    ("fixtures_swe", future_from, future_to), sweden_client.get_fixtures, date_from=future_from, date_to=future_to
                ),
                "SWE",
            )
        if "SP1" in enabled:
            matches += _tag(
                await _cached_fixture_call(
                    ("fixtures_sp1", future_from, future_to), la_liga_client.get_fixtures,
                    competition_code=LA_LIGA_COMPETITION_CODE, date_from=future_from, date_to=future_to,
                ),
                "SP1",
            )
        if "I1" in enabled:
            matches += _tag(
                await _cached_fixture_call(
                    ("fixtures_i1", future_from, future_to), serie_a_client.get_fixtures,
                    competition_code=SERIE_A_COMPETITION_CODE, date_from=future_from, date_to=future_to,
                ),
                "I1",
            )
        if "D1" in enabled:
            matches += _tag(
                await _cached_fixture_call(
                    ("fixtures_d1", future_from, future_to), bundesliga_client.get_fixtures,
                    competition_code=BUNDESLIGA_COMPETITION_CODE, date_from=future_from, date_to=future_to,
                ),
                "D1",
            )
        if "F1" in enabled:
            matches += _tag(
                await _cached_fixture_call(
                    ("fixtures_f1", future_from, future_to), ligue1_client.get_fixtures,
                    competition_code=LIGUE_1_COMPETITION_CODE, date_from=future_from, date_to=future_to,
                ),
                "F1",
            )
    return matches


def _fetch_odds_for_manual_request(request: RecommendationRequest, league: str | None) -> dict[str, float] | None:
    """W49: best-effort odds lookup for the manual 'regenerate now' path,
    reusing eod_batch.py's exact odds_client + match_odds team-matching
    logic (BUG-015's canonical-name matching included) rather than
    reimplementing it. Unlike run_eod_batch's build_odds_client() call
    (fetched once for a whole batch of same-day fixtures), this fetches
    fresh odds per manual request -- acceptable here since it's a single
    user-triggered click, not a batch loop.

    `league` is the already-gated competition id (match_info.get("league"),
    W03's gate_league output) -- W58: picks the matching Odds-API sport_key
    instead of relying on get_odds()'s own "soccer_epl" default, so a
    Swedish fixture's fetch actually queries Sweden's odds feed rather than
    silently querying EPL's.

    Every failure mode degrades to None (no odds attached) rather than
    raising: no odds client configured (build_odds_client() returns None
    when no ODDS_API_KEY/sandbox override is set), no matching odds event
    for this fixture, or the odds client call itself raising (e.g. a
    network error). This must never turn a previously-working no-odds
    request into a 500 -- it's strictly additive over the pre-W49
    behavior.

    Cost note (code review, W49): in production this draws from the same
    CreditCounter-guarded monthly Odds API budget the scheduled EOD/T-30
    jobs use (build_odds_client() -> the same on-disk counter path) --
    manual regenerate previously cost zero Odds API credits. would_exceed()
    still protects the safety margin (degrades to no-odds, never errors),
    but frequent manual clicks now measurably compete with the scheduler
    for the same budget within a given month -- worth watching if usage
    patterns change, not something this fix can size in advance."""
    try:
        odds_client = build_odds_client()
        if odds_client is None:
            return None
        sport_key = ODDS_SPORT_KEY_BY_COMPETITION.get(league, DEFAULT_SPORT_KEY)
        # BUG-031: date=request.date, not the client's own default -- without
        # it, HistoricalOddsClient.get_odds() (sandbox mode) falls back to
        # the sandbox's own as_of date, silently querying *today's* odds
        # events instead of the requested fixture's, for any match not dated
        # exactly on as_of (i.e. most of what the Dashboard shows since W86's
        # "next 10 matches" window). odds_lookup/match_odds then correctly
        # find no matching pair among the wrong day's fixtures and this
        # degrades to no-odds -- not a crash, but a silent, wrong "no odds
        # available" for a fixture that genuinely has real odds recorded.
        # OddsAPIClient (live) accepts and ignores `date` -- interface parity
        # only, same as eod_batch.py's own per-fixture-date odds fetch.
        odds_events = odds_client.get_odds(sport_key=sport_key, date=request.date)
        odds_by_teams = eod_batch.odds_lookup(odds_events or [])
        fixture = NormalizedMatch(
            match_id=request.effective_match_id(), utc_date="", status="",
            home_team=request.home_team, away_team=request.away_team,
            home_goals=None, away_goals=None,
        )
        return eod_batch.match_odds(fixture, odds_by_teams)
    except Exception:
        LOGGER.warning(
            "Manual recommendation odds fetch failed for %s v %s (%s)",
            request.home_team, request.away_team, request.date, exc_info=True,
        )
        return None


@app.post("/api/recommendations")
async def create_recommendation(
    request: RecommendationRequest,
    cache: RecommendationCache = Depends(recommendations.get_cache),
) -> MatchRecommendationOut:
    """The explicit 'regenerate now' escape hatch (W11) -- always calls the
    agent and writes the result into the cache, tagged manual_regenerate so
    it's distinguishable from a scheduled (W09/W10) generation.

    W49: unlike the scheduled EOD/T-30 pipeline (eod_batch.py), this path
    used to never fetch odds itself -- match_info["odds"] was purely
    caller-supplied, and neither MatchCard nor MatchAnalysisPage ever
    populate it, leaving the agent to rely on its own web_search (which
    structurally can't succeed for a historical/sandboxed match, and
    frequently fails for real matches too). Now mirrors run_eod_batch's
    odds-fetch-before-run_agent sequence whenever the caller didn't already
    supply odds explicitly -- an explicit request.odds always wins."""
    match_info = request.to_match_info()
    if request.odds is None:
        # build_odds_client()/get_odds() make a real synchronous HTTP or DB
        # call (HistoricalOddsClient/OddsAPIClient) -- off the event loop,
        # same convention as the run_agent call directly below.
        # _fetch_odds_for_manual_request already has its own broad
        # except-and-degrade-to-None (including a hypothetical
        # duckdb.IOException from HistoricalOddsClient's sandbox-mode DB
        # read, W93 -- confirmed live, not assumed) -- nothing further
        # needed here for that side.
        fetched_odds = await run_in_threadpool(_fetch_odds_for_manual_request, request, match_info.get("league"))
        if fetched_odds is not None:
            match_info["odds"] = fetched_odds

    # W93: unlike the odds fetch above, run_agent (via ForecastService,
    # reading data/fpai_core.db) had no protection against DuckDB's real
    # exclusive file lock -- confirmed live that a second process opening
    # *any* connection (even read-only) while a read-write connection is
    # open elsewhere (e.g. the ML-engine's own scheduled data refresh,
    # documents/user_stories.md Phase 23, mid-write) fails immediately with
    # duckdb.IOException rather than blocking or corrupting data. Was
    # previously unhandled here, surfacing as a raw 500 unlike every other
    # transient-external-condition path in this app
    # (_fetch_and_cache_fixtures's own HTTPError-to-503 precedent, above).
    try:
        # run_agent is a real ~10-30s synchronous call (LLM + Tavily) --
        # must run off the event loop or it blocks every other request.
        raw = await run_in_threadpool(recommendations.run_agent, match_info)
    except duckdb.IOException as exc:
        LOGGER.warning("Recommendation generation hit a locked database (%s): %s", request.effective_match_id(), exc)
        raise HTTPException(
            status_code=503,
            detail=(
                "The match database is temporarily locked by a scheduled data refresh. "
                "Please try again in a minute."
            ),
        ) from exc
    result = validate_and_degrade(raw, request.home_team, request.away_team)

    cache.record_generation(
        match_id=request.effective_match_id(),
        date=request.date,
        agent_config_hash=compute_agent_config_hash(AgentConfig.default()),
        # W49: record what was actually used for generation, not just what
        # the caller explicitly supplied -- match_info["odds"] reflects the
        # server-side fetch too. Recording request.odds alone would persist
        # {} even when real odds were fetched and used, spuriously tricking
        # t30_refresh.py's "odds unchanged, skip regeneration" dedup check
        # into always seeing a change on the next comparison.
        odds=match_info.get("odds", {}),
        recommendation=result.model_dump(),
        triggered_by="manual_regenerate",
    )
    return result


class RecommendationOutcomeOut(BaseModel):
    id: int
    match_id: str
    date: str
    competition: str | None
    market: str
    selection: str
    recommendation_type: str
    confidence: str | None
    odds: float | None
    value_edge: float | None
    correct: bool
    generated_at: str
    resolved_at: str


@app.post("/api/recommendations/settle-open")
async def settle_open_recommendations(
    cache: RecommendationCache = Depends(recommendations.get_cache),
    store: RecommendationOutcomeStore = Depends(get_recommendation_outcome_store),
) -> list[RecommendationOutcomeOut]:
    """W167: on-demand resolution trigger, same trigger story as
    /api/bets/settle-open -- not scheduler-tied. Diagnostics only, for the
    user's own querying; the frontend never calls this. Calls
    get_fixtures_client()/get_sweden_fixtures_client() as plain function
    calls (not Depends()), exactly mirroring settle_open()'s own existing
    shape immediately below -- both reuse the same fixtures/results client
    and its shared rate-limit budget."""
    client = get_fixtures_client()
    sweden_client = get_sweden_fixtures_client()
    resolved = await run_in_threadpool(resolve_pending_recommendations, cache, store, client, sweden_client)
    return [RecommendationOutcomeOut(**dataclasses.asdict(r)) for r in resolved]


@app.get("/api/recommendations/stats")
async def get_recommendation_stats(
    days: int = Query(30, ge=0, le=3650),
    store: RecommendationOutcomeStore = Depends(get_recommendation_outcome_store),
) -> dict:
    """W168: hit-rate breakdown + Kelly ROI simulation over resolved
    recommendation_outcomes, denominated in UB. Diagnostics only -- queried
    directly by the user, no frontend surface.

    Registered ahead of GET /api/recommendations/{match_id} below: FastAPI/
    Starlette matches routes in registration order, and {match_id} is a
    single-path-segment pattern that would otherwise swallow the literal
    "stats" segment as match_id (then 422 on the missing required `date`
    query param) if it came first."""
    cutoff = (datetime.now(timezone.utc) - timedelta(days=days)).strftime("%Y-%m-%d")
    outcomes = store.list_all(since=cutoff)
    return compute_recommendation_stats(outcomes)


@app.get("/api/recommendations/performance-dashboard")
async def get_agent_performance_dashboard(
    days: int = Query(30, ge=0, le=3650),
    top_n: int = Query(5, ge=1, le=50),
    cache: RecommendationCache = Depends(recommendations.get_cache),
    store: RecommendationOutcomeStore = Depends(get_recommendation_outcome_store),
) -> dict:
    """W171/W172: local-only diagnostics dashboard -- main metrics, segment
    breakdowns, distributions, and top/bottom staked-bet examples, all in
    one response. Not called by the deployed frontend's nav-linked pages;
    reachable only via app/agent-performance/page.tsx, which is itself
    unlinked from AppShell's nav (W174).

    Registered ahead of GET /api/recommendations/{match_id} below, same
    reason /stats already had to be: {match_id} is a single-path-segment
    pattern that would otherwise swallow "performance-dashboard" as its
    own match_id value."""
    cutoff = (datetime.now(timezone.utc) - timedelta(days=days)).strftime("%Y-%m-%d")
    outcomes = store.list_all(since=cutoff)
    return compute_agent_performance_dashboard(outcomes, cache, top_n=top_n)


@app.get("/api/recommendations/{match_id}")
async def get_cached_recommendation(
    match_id: str,
    date: str,
    cache: RecommendationCache = Depends(recommendations.get_cache),
) -> MatchRecommendationOut:
    """Reads exclusively from the cache (W11) -- never calls run_agent. The
    normal path for an already-scheduled fixture; a cache miss means nothing
    has generated a recommendation for this match/date yet.

    BUG-028: routes through validate_and_degrade (no home_team/away_team --
    this endpoint only has match_id/date, so the match-mismatch check is
    skipped, but per-market validation still gracefully drops a malformed
    market) instead of a raw MatchRecommendationOut.model_validate(), which
    raised an uncaught ValidationError (a 500) for any pre-existing cached
    row using a market/selection value that predates BUG-027's Literal
    constraints -- routine for local-model-hallucinated rows.

    A65/A66 follow-up: falls back to get_latest_any_config() (ignores
    agent_config_hash) when there's no entry for today's exact config --
    a still-good prior recommendation, generated under an older config,
    beats a hard miss, especially when a config bump (busts every match's
    hash at once) coincides with an unrelated regeneration failure
    (confirmed live: a DeepSeek billing outage) that leaves nothing
    regenerated under the new hash yet."""
    agent_config_hash = compute_agent_config_hash(AgentConfig.default())
    entry = cache.get_latest(match_id, date, agent_config_hash) or cache.get_latest_any_config(match_id, date)
    if entry is None:
        raise HTTPException(status_code=404, detail="No cached recommendation for this match/date yet.")
    return validate_and_degrade(entry.recommendation)


@app.post("/api/bets/from-recommendation")
async def create_bet_from_recommendation(
    request: BetFromRecommendationRequest,
    tracker: BetTracker = Depends(bets.get_bet_tracker),
) -> BetOut:
    """Every field but stake is locked -- derived from the recommendation
    snapshot itself, which is also stored verbatim (recommendations aren't
    reproducible run-to-run, agent_techspec.md sec18.6)."""
    try:
        resolved = bets.resolve_from_recommendation(request)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    bet = tracker.create_bet(
        match_id=resolved["match_id"], date=resolved["date"],
        home_team=resolved["home_team"], away_team=resolved["away_team"],
        market=resolved["market"], selection=resolved["selection"],
        odds=resolved["odds"], stake=resolved["stake"],
        source="from_recommendation", recommendation_snapshot=request.recommendation,
    )
    return BetOut.from_bet(bet)


@app.post("/api/bets/manual")
async def create_bet_manual(
    request: BetManualRequest,
    tracker: BetTracker = Depends(bets.get_bet_tracker),
) -> BetOut:
    """User-provided fields, but match_id must be a resolved fixture reference
    (enforced by the frontend's Match Explorer search, not free-typed team
    names) -- Pydantic requires it non-empty at minimum."""
    bet = tracker.create_bet(
        match_id=request.match_id, date=request.date,
        home_team=request.home_team, away_team=request.away_team,
        market=request.market, selection=request.selection,
        odds=request.odds, stake=request.stake,
        source="manual", recommendation_snapshot=None,
    )
    return BetOut.from_bet(bet)


@app.get("/api/bets")
async def list_bets(tracker: BetTracker = Depends(bets.get_bet_tracker)) -> list[BetOut]:
    return [BetOut.from_bet(bet) for bet in tracker.list_bets()]


@app.get("/api/bets/stats")
async def get_bet_stats(tracker: BetTracker = Depends(bets.get_bet_tracker)) -> dict:
    """W14: ROI/hit-rate/bankroll summary, computed fresh over settled bets
    on every call -- no persisted running total."""
    return compute_bet_stats(tracker.list_bets())


@app.post("/api/bets/settle-open")
async def settle_open(tracker: BetTracker = Depends(bets.get_bet_tracker)) -> list[BetOut]:
    """On-demand settlement trigger (W13) -- intentionally not folded into
    W08's scheduler; bet settlement isn't tied to recommendation-generation
    timing the way W09/W10 are. Reuses get_fixtures_client (W05's
    FootballDataClient) since results/fixtures share the same API and rate
    limit budget. W57: also consults get_sweden_fixtures_client so a
    Swedish bet's match_id (unknown to football-data.org) can settle too."""
    client = get_fixtures_client()
    sweden_client = get_sweden_fixtures_client()
    settled = await run_in_threadpool(settle_open_bets, tracker, client, sweden_client)
    return [BetOut.from_bet(bet) for bet in settled]
