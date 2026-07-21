# Technical Specification — FPAI Web App

Authoritative implementation reference for the bettor-facing web app described in `app_prd.md`. This document reflects the actual code as built through **47 completed stories** (`W01`–`W50`, excluding `W18`/`W19`/`W24`/`W26`/`W32`, which remain `future`, and `W40`/`W41`, which are `active`/not yet started — see [Implementation Status](#14-implementation-status)). All open design discussions in `documents/app_user_stories.md` (D1–D7) are resolved; that document's "Confirmed So Far" section is the settled-decisions record this spec formalizes, and its per-story "Completion notes" are the primary source for everything below. Where this app depends on the agent's or forecast engine's internal behavior, this document points at `agent_techspec.md`/`FRAI_TECHSPEC.md` rather than re-explaining it.

The app wraps `src/agent` and `src/forecast` directly as Python libraries — no MCP, no subprocess, no HTTP hop through `src/mcp_server.py` (that server is for external third-party agent consumers, not this first-party app; see `app_user_stories.md` W01).

---

## 1. Module Structure

```
app/
  README.md
  data/                          # gitignored — SQLite/JSON scratch stores, plus data/sandbox/ (W29)

  backend/
    __init__.py
    main.py                      # FastAPI app, lifespan (startup LLM check + scheduler wiring), all endpoints
    llm_check.py                 # check_llm_reachable() -- soft startup probe (W01)
    match_info.py                # COMPETITION_ALLOWLIST / gate_league() -- app-side league routing (W03)
    football_data_client.py      # FootballDataClient -- fixtures/results, rate limiter (W05)
    odds_api_client.py           # OddsAPIClient, CreditCounter, FileCreditCounterStore (W07)
    historical_odds_client.py    # HistoricalOddsClient -- sandbox real-historical odds from raw_matches (W28)
    recommendations.py           # RecommendationRequest/MatchRecommendationOut, validate_and_degrade,
                                  #   sandbox-aware run_agent() wrapper (W02, W37, W43)
    recommendation_cache.py      # RecommendationCache -- SQLite append-only generation history (W11)
    agent_config_hash.py         # compute_agent_config_hash() -- cache-key component (W11)
    eod_batch.py                 # run_eod_batch(), odds_lookup()/match_odds() (W09, BUG-015 fix)
    t30_refresh.py               # refresh_match_at_t30() -- per-fixture T-30 odds-diff refresh (W10)
    scheduler.py                 # RecoverableScheduler, JobRunLog, NY_TZ (W08)
    scheduler_wiring.py          # register_eod_job(), build_schedule_t30(), build_odds_client(),
                                  #   PersistingOddsClient (W08→W09→W10 wiring)
    bet_tracker.py                # BetTracker -- SQLite CRUD + settle_bet() (W12)
    bets.py                      # BetFromRecommendationRequest/BetManualRequest/BetOut, resolve_from_recommendation() (W12)
    settlement.py                # settle_open_bets() -- auto-settlement against live results (W13)
    bet_stats.py                 # compute_bet_stats() -- ROI/hit-rate/bankroll (W14)
    sandbox_clock.py             # is_sandbox_mode()/sandbox_date()/sandbox_now()/sandbox_status()/
                                  #   sandbox_scoped_path() (W27)
    tests/                       # 29 test modules -- see Section 12

  frontend/
    app/
      layout.tsx                 # Root layout -- mounts <StatusFooter/> on every page
      page.tsx                   # "/" -> DashboardPage
      matches/page.tsx           # "/matches" -> MatchExplorerPage
      matches/[id]/page.tsx      # "/matches/[id]" -> MatchAnalysisPage
      bets/page.tsx               # "/bets" -> BetTrackerPage
      globals.css                 # dark-mode design tokens (CSS custom properties consumed by tailwind.config.js)
    components/
      MatchUI.tsx                 # ported sandbox/ prototype (1020 lines): types, atoms (TeamBadge, StatusBadge,
                                  #   TierTag, TrustSignal), MatchCard, DashboardPage, MatchExplorerPage,
                                  #   MatchAnalysisPage, LogBetButton, ProbabilityRow, DraftNav
      BetTracker.tsx              # ManualBetForm, BetRow, StatsBar, BetTrackerPage
      StatusFooter.tsx            # W17 data-staleness/model-status footer
      __tests__/                  # Vitest + RTL component/race/boundary tests -- see Section 12
    lib/
      api.ts                     # typed fetch wrappers for every backend endpoint
      types.ts                   # wire types mirroring the backend's Pydantic models exactly
      useSandboxAsOf.ts           # W30 -- resolves "today" to the sandbox as_of date when active
    tailwind.config.js / postcss.config.js / next.config.js / vitest.config.ts / vitest.setup.ts

scripts/
  sandbox_runbook.py              # W31 -- repeatable end-to-end sandbox scenario driver (backend-only, no server)
  launch_sandbox.py               # W44 -- one-command interactive launch: preflight + real backend/frontend servers

documents/
  app_prd.md
  app_user_stories.md             # story tracking (W01–W44) -- primary source for this document
  app_techspec.md                 # this document
  prelaunch_smoke_test_checklist.md   # W23 -- manual pre-launch checklist, run and recorded
  sandbox_testing_runbook.md          # W31 -- recorded runbook results
```

---

## 2. Backend Architecture

`app/backend/main.py` builds a single `FastAPI(title="FPAI Web App Backend", lifespan=lifespan)` instance.

**Startup (`lifespan`, an `asynccontextmanager`, not the deprecated `on_event("startup")`):**

1. Loads `AgentConfig.default()` and calls `check_llm_reachable(config)` (`llm_check.py`). For `provider: ollama`, this is a real 2-second-timeout probe against `http://localhost:11434/api/tags` — any exception is treated as unreachable and never raises. For `provider: anthropic`, it only checks `ANTHROPIC_API_KEY` presence (a live call would spend real credits just to check connectivity). A failed check logs a warning and does **not** block startup — recommendation generation degrades, the rest of the app still serves requests.
2. If `ENABLE_SCHEDULER` is truthy (`1`/`true`/`yes`), builds a `RecoverableScheduler` — sandbox-scoped `JobRunLog` if `sandbox_clock.is_sandbox_mode()` is active — and calls `register_eod_job()` (Section 6), then `.start()`s it. **Off by default.** Registering the EOD job immediately runs `RecoverableScheduler`'s restart/catch-up check (Section 6.1) — leaving this unconditional would make every `TestClient(app)` instantiation non-deterministically trigger a real live EOD batch (real fixtures/odds/LLM calls) depending on wall-clock time relative to the 23:00 NY trigger. This gate is a deliberate engineering call beyond what any story text required.
3. `yield`s; on shutdown, calls `scheduler.shutdown()` if one was started.

`load_dotenv()` runs at module import time, before any `app.backend`/`src.agent` import — this was a real gap found during W04 (`FOOTBALL_DATA_API_KEY` silently unloaded when running the FastAPI app directly, unlike `main.py` at the repo root) and is now fixed at the top of `app/backend/main.py`.

**CORS (D7):** `CORSMiddleware` allows `http://localhost:3000` only, all methods/headers — the standard two-process local-dev shape (Next.js dev server + `uvicorn --reload`), no Docker, no shared process.

**Dependency injection:** `get_fixtures_client()` (module-level singleton, `main.py`), `recommendations.get_cache()`, and `bets.get_bet_tracker()` are FastAPI `Depends(...)`-injected and overridable in tests via `app.dependency_overrides` or by monkeypatching the function directly. Each of the latter two routes to a sandbox-scoped SQLite path (`sandbox_clock.sandbox_scoped_path(...)`) when `is_sandbox_mode()` is true (W29) — see Section 8.

---

## 3. API Reference

All routes are under `app/backend/main.py`. Response models are the Pydantic classes noted; unannotated dict returns are plain JSON.

| Method & Path | Request | Response | Notes |
|---|---|---|---|
| `GET /api/health` | — | `{"status": "ok"}` | W01. |
| `GET /api/sandbox/status` | — | `SandboxStatus {sandbox_mode: bool, as_of: str \| null}` | W27 — introspects the active sandbox override for the frontend and test scripts. |
| `GET /api/status` | — | `{data_freshness: ..., model_status: ...}` | W17 — direct pass-through of `src/tools/data_tools.get_data_freshness()` and `src/tools/model_tools.get_model_status()` (`AGENT_TOOL_CONTRACT.md`-documented, no new engine work). |
| `GET /api/fixtures` | query `date_from?`, `date_to?` | `list[NormalizedMatch]` | Wraps `FootballDataClient` (W05), added during W04 as a discovered gap. Still used directly even after W09 — the frontend needs a live fixture list to render cards regardless of whether a recommendation is cached yet. **W45:** sources the requested range from `get_results()` (status=`FINISHED`) and/or `get_fixtures()` (status=`SCHEDULED`) depending on where it falls relative to real wall-clock "today" — see Section 5.1. |
| `POST /api/recommendations` | `RecommendationRequest {home_team, away_team, date, league?, odds?, match_id?}` | `MatchRecommendationOut` | W02/W11 — the explicit "regenerate now" escape hatch. Always calls the real agent (off the event loop via `run_in_threadpool`) and writes into the cache tagged `triggered_by="manual_regenerate"`. **W47:** the frontend now genuinely treats this as a fallback — see Section 6.3. **W49:** when the request doesn't already supply `odds`, the handler now fetches them itself (same `build_odds_client()`/`match_odds()` logic as the scheduled batch) before calling the agent — see Section 6.2. |
| `GET /api/recommendations/{match_id}` | query `date` (required) | `MatchRecommendationOut`, `404` on cache miss | W11 — reads **exclusively** from `RecommendationCache`; never calls `run_agent()`. The normal read path for an already-scheduled fixture. **W47:** as of this story, the frontend actually calls this before falling back to `POST /api/recommendations` — previously implemented but unused by any UI code, see Section 6.3. |
| `POST /api/bets/from-recommendation` | `BetFromRecommendationRequest {match_id, recommendation, market, selection, stake}` | `BetOut` | W12 — every field but `stake` is derived from the recommendation snapshot itself (`resolve_from_recommendation()`); `400` if the market/selection isn't in the snapshot or has `current_odds: null`. |
| `POST /api/bets/manual` | `BetManualRequest {match_id, date, home_team, away_team, market, selection, odds, stake}` | `BetOut` | W12 — `match_id` must be non-empty (a resolved fixture reference from Match Explorer search, not free text); `422` otherwise. |
| `GET /api/bets` | — | `list[BetOut]` | W12 — all logged bets, ascending `id` order. |
| `GET /api/bets/stats` | — | `compute_bet_stats()` dict (ROI/hit-rate/bankroll, see Section 7.3) | W14 — recomputed fresh from the tracker on every call, no persisted running total. |
| `POST /api/bets/settle-open` | — | `list[BetOut]` (only bets settled by this call) | W13 — on-demand, no scheduler; reuses `get_fixtures_client()` since results and fixtures share the same rate-limit budget. |

`MatchRecommendationOut`/`MarketRecommendationOut` (`app/backend/recommendations.py`) are the app's own Pydantic models, independent of the agent's `src/agent/schema.py` TypedDicts — see Section 10.

---

## 4. Competition/League Gating (W03)

`app/backend/match_info.py`:

```python
COMPETITION_ALLOWLIST: frozenset[str] = frozenset({"E0"})

def gate_league(competition_id: str | None) -> str | None:
    ...  # returns competition_id if allowlisted, else None
```

Wired into `RecommendationRequest.to_match_info()` (`recommendations.py`) — the request's `league` field is never passed through verbatim; it is gated first, so `match_info["league"]` is only ever set for actual Premier League fixtures. Every other competition omits `league` entirely, matching the existing agent CLI convention ("omit `--league` for international fixtures — agent uses `forecast_international`").

**Why this exists as app-side defense-in-depth, not just engine-side routing.** `ForecastService.forecast_upcoming()`'s domestic branch, before US#107 landed, unconditionally called `FeatureFactory.build_for_match(league=league, ...)` and loaded the flat `"league"` model context regardless of what `league` string was passed — a non-EPL fixture routed this way got `prediction_basis: "team_history_and_market"` with every rolling/team feature cold-start-zero-filled, the only honest signal being a buried `cold_start_risk: true`. US#107 (`FRAI_TECHSPEC.md`/`documents/user_stories.md`) now makes `forecast_upcoming()` consult `config/competitions.yaml` itself and route to `general_purpose` automatically for unregistered leagues; A27 (`agent_techspec.md` §4.4) separately gives the agent a `resolve_competition` tool so its own tool selection doesn't have to guess. **This app keeps `COMPETITION_ALLOWLIST` regardless of either fix** — a hardcoded literal, deliberately *not* a live read of `config/competitions.yaml`, so a registry misconfiguration on the engine side can't silently cascade into the app's own routing too. `config/competitions.yaml` registers exactly two entries as of this writing: `E0` (`competition_specific`) and a generic `international` (`general_purpose`) — there is no La Liga, Serie A, Bundesliga, or Champions League registration yet (W18, Section 13).

---

## 5. Fixture & Odds Data Sourcing

### 5.1 Fixtures/results — `football_data_client.py` (W05)

`FootballDataClient.get_fixtures()`/`get_results()` (status `SCHEDULED`/`FINISHED`) wrap `GET /v4/competitions/{code}/matches` (default `PL`) and return a frozen `NormalizedMatch` dataclass (`match_id`, `utc_date`, `status`, `home_team`, `away_team`, `home_goals`, `away_goals`) built from `homeTeam.shortName`/`awayTeam.shortName` — independent of the provider's own internal field names.

`_RateLimiter` tracks the real free-tier headers (`x-requests-available-minute`, `X-RequestCounter-Reset`, confirmed live: 10 req/min) and proactively sleeps until the window resets once exhausted, rather than waiting for a 429. Both `sleep_fn`/`time_fn` are constructor-injectable for deterministic tests (exercised end-to-end by W36's sequence test, Section 6.4).

**W45 — `GET /api/fixtures`'s date-range split (`app/backend/main.py`).** Both `FootballDataClient` methods existed from W05 onward, but the `/api/fixtures` endpoint originally called only `get_fixtures()` (status `SCHEDULED`) regardless of the requested range — so it structurally returned nothing for any date range entirely in the past (a historical sandbox date, or real fixtures that had already finished), independent of whether real data actually existed. `_current_real_date()` (a genuine `datetime.now(timezone.utc).date()` call, deliberately **not** `sandbox_clock.sandbox_now()`, since football-data.org's SCHEDULED/FINISHED status reflects real-world match completion regardless of what date the app is pretending "today" is) and `_split_fixture_date_range()` now split the requested `[date_from, date_to]` into a results-sub-range and a fixtures-sub-range relative to real wall-clock today, calling `get_results()`/`get_fixtures()` respectively and concatenating. **A range spanning or landing on today queries today from *both* sides** — a same-day match may already be `FINISHED` (an early kickoff, viewed later) or still `SCHEDULED`, and there's no way to know which without asking both; this was a real gap in the first version of the fix (caught by code review — the initial split treated today as future-only, silently dropping already-finished same-day matches from the Dashboard's own actual hot path) fixed in a same-day follow-up.

### 5.2 Team-name mapping (W06)

`config/team_mapping.json` + `src/ingestion/common/team_mapping.py`'s `TeamNameMapper` (source-agnostic, already shared with Understat/FotMob ingestion) were extended with football-data.org's naming variants. Verified live in two passes against real API responses — 7 of an initial draft's guesses were wrong, including 3 not even flagged low-confidence (Newcastle, West Ham, Wolves). `FeatureFactory.build_for_match()`'s previously bespoke, silent inline lookup was replaced with the shared, logging `TeamNameMapper`, which surfaced a second latent gap: 11 canonical names (`Brighton`, `Cardiff`, `Huddersfield`, `Hull`, `Ipswich`, `Leeds`, `Luton`, `Nott'm Forest`, `Sheff Wed`, `West Brom`, `Wolves`) had no identity entry of their own — fixed by adding them, not by special-casing the code. `documents/football_data_org_team_mapping_draft.json` records the draft/verification history.

### 5.3 Odds — `odds_api_client.py` (W07, D2b)

football-data.org has **no odds on its free tier** (verified: a €15/month add-on requiring a paid base plan) — a second provider, The Odds API, supplies 1X2 odds, scoped to a single market (`h2h`) and single region (`uk`) to stay inside the ~500-credit/month free tier.

```python
class CreditCounter:
    def would_exceed(self, cost: int, limit: int, safety_margin: int) -> bool: ...
    def record_usage(self, cost: int) -> None: ...
```

Cost is computed client-side as `len(markets) × len(regions)` per the provider's documented formula — never read from a response header. `OddsAPIClient.get_odds()` checks `would_exceed(...)` **before** calling; if within the safety margin (default 50 credits), it logs a warning and returns `None` (caller keeps last-known odds) without attempting the request. `CreditCounter` auto-rolls over the first time it's consulted in a new calendar month. `FileCreditCounterStore` persists state to a JSON file so a restart doesn't lose the running monthly count — `scheduler_wiring.py`'s `PersistingOddsClient` is the sole caller that wires save-after-every-call, since `OddsAPIClient`/`CreditCounter`/`FileCreditCounterStore` deliberately leave persistence to the caller rather than baking it in.

`_normalize()` maps `bookmakers[0].markets[].outcomes[]` to home/draw/away decimal odds. **Live-verified against a real key (W25, 2026-07-15):** no changes needed — real structure matched the public v4 schema. Two follow-on caveats recorded in the module docstring rather than fixed: `bookmakers[]` ordering is confirmed *not* stable across events (`_normalize` only ever reads index 0, no fallback if it lacks `h2h`), and real responses carry authoritative `x-requests-remaining`/`x-requests-used` headers that `CreditCounter` never reads (relies purely on the client-side estimate).

### 5.4 BUG-015 — odds-to-fixture team-name matching

The same W25 live check found 6 of 10 real fixtures with differently-spelled team names between The Odds API and football-data.org (`Man United` vs `Manchester United`, etc.) — `eod_batch.py`'s `odds_lookup()`/`match_odds()` originally matched by exact case-insensitive string equality, silently dropping real odds for the majority of real fixtures. Fixed by routing both sides through the same `TeamNameMapper`/`config/team_mapping.json` ingestion already uses, before comparing canonical names — re-verified 10/10 fixtures resolve correctly post-fix. `HistoricalOddsClient` (Section 9) inherits this fix for free since it feeds the same `odds_lookup()`/`match_odds()` functions.

---

## 6. Scheduled Recommendation Generation

### 6.1 Scheduler infrastructure — `scheduler.py` (W08)

`RecoverableScheduler` wraps APScheduler's `BackgroundScheduler(timezone=NY_TZ)` (`NY_TZ = ZoneInfo("America/New_York")` — a real IANA zone, correctly DST-aware, not a fixed UTC offset) and deliberately does **not** rely on APScheduler's own in-memory jobstore surviving a restart. `JobRunLog` (SQLite, `app/data/job_runs.db`) persists `(job_id, run_key)` "already ran" markers.

```python
def schedule_daily(self, job_id, fn, hour, minute) -> None
def schedule_once(self, job_id, fn, run_at: datetime) -> None
```

Both register the normal future APScheduler trigger **and** immediately check `JobRunLog`: if the trigger time has already passed and the job hasn't been marked as run for that `run_key`, it runs synchronously, right there, exactly once. `schedule_daily`'s `run_key` is the calendar date (`now.date().isoformat()`); `schedule_once`'s `run_key` is `run_at.isoformat()` — **not** a constant (see W33 below). `_run_and_mark` never lets a job's own exception propagate out of this immediate catch-up path (it runs on the *caller's* thread, unlike APScheduler's own later background-thread fires, which are already exception-isolated) — a failure is logged and the run is *not* marked done, so the next registration retries it.

### 6.2 EOD batch — `eod_batch.py` (W09) and per-fixture refresh — `t30_refresh.py` (W10)

`register_eod_job()` (`scheduler_wiring.py`) registers `_eod_job` on the scheduler via `schedule_daily(EOD_JOB_ID, ..., hour=23, minute=0)` (NY time). The job body:

1. Computes tomorrow's date via `next_day_date_str(now_fn)` (defaults to `sandbox_now(NY_TZ)` — sandbox-aware, Section 9).
2. Calls `asyncio.run(run_eod_batch(...))`.

`run_eod_batch()` mirrors `agent_techspec.md` §13's `_run_backtest_concurrent()` shape exactly: fetches the day's E0 fixtures (W05) and current odds (W07), builds an `odds_by_teams` lookup via canonical-name matching (Section 5.4), then for each fixture — bounded by `asyncio.Semaphore(concurrency)` (default 5), dispatched via `asyncio.to_thread(recommendations.run_agent, ...)` — generates a recommendation. A per-match `try/except` logs and increments `EodBatchResult.skipped` rather than aborting the whole batch. Every fixture gets `schedule_t30(fixture)` called unconditionally afterward, regardless of whether its own EOD generation succeeded — the T-30 refresh is independently best-effort.

**W49 — `POST /api/recommendations` never fetched odds itself.** Unlike `run_eod_batch()` above, the manual "regenerate now" endpoint (`create_recommendation()`, `main.py`) only ever used caller-supplied `odds` — and neither `MatchCard` nor `MatchAnalysisPage` populate that field, leaving the agent to rely on its own `web_search` for odds, which structurally can't succeed for a historical/sandboxed match and often fails for real ones too. Diagnosed live (not inferred): a real click-through with network capture showed a genuinely-completed (~25s) request with no `odds` in the payload, resulting in an avoidable `insufficient_data`, while `raw_matches` had real odds for that exact fixture. Fixed with a new `_fetch_odds_for_manual_request()` (`main.py`) that reuses `build_odds_client()`/`odds_lookup()`/`match_odds()` verbatim — the same functions above, not a reimplementation — via `run_in_threadpool`, only when `request.odds is None`; every failure mode (no client configured, no matching event, client raising) degrades to the pre-existing no-odds behavior. The cache write records `match_info.get("odds", {})` (what was actually used), not the original `request.odds` — a code-review-caught fix, since recording the latter would falsely show `odds={}` for a fetched-odds generation, spuriously tripping T-30's "odds unchanged, skip" dedup check below. Manual regeneration now draws from the same `CreditCounter`-guarded Odds API budget the scheduler uses (previously zero cost) — safety-margin-protected, not a crash risk, but a real, documented shift in credit consumption worth watching.

**W50 — sandbox testing gained an opt-in precompute step, and a second instance of `/api/fixtures`'s own bug was found and fixed.** `run_eod_batch()` itself calls `fixtures_client.get_fixtures()` (status=`SCHEDULED` only) — the exact root cause W45 fixed for `/api/fixtures`, just via a separate code path W45 never touched, invisible until now because the live scheduler has only ever run for real, always-future dates. Fixed by giving `run_eod_batch()` an optional `fixtures: list[NormalizedMatch] | None = None` parameter (bypasses the internal `get_fixtures()` call when supplied; omitting it preserves the exact pre-W50 live-scheduler behavior) and an `on_progress` callback for CLI reporting. `scripts/launch_sandbox.py`'s new `--precompute` flag (Section 9.8) uses this to pre-populate the sandbox-scoped cache before the servers start — see there for the full flow.

`refresh_match_at_t30()` (W10) fires once per fixture, 30 minutes before kickoff (`t30_run_at()` = `utc_date - 30min`, scheduled via `schedule_once`). It fetches fresh odds first and compares them against the cached recommendation's stored odds (`RecommendationCache.get_latest(...).odds`, exact dict equality):

- **Odds unchanged** → `skipped_no_change`, zero new `run_agent()` calls, zero new cache rows.
- **Odds changed, or no prior cache entry at all** → re-runs `run_agent()`, writes a new cache row tagged `triggered_by="scheduled"`.
- **No odds available** (credit budget exhausted, no client configured, or fixture not found in the feed — e.g. postponed) → `skipped_no_odds`, prior recommendation untouched.
- **`run_agent()` raises** → `skipped_error`, prior recommendation untouched.

This both saves LLM/Tavily cost and reduces how often the agent's known run-to-run non-reproducibility (`agent_techspec.md` §18.6) is exercised for no reason. `refresh_match_at_t30` is deliberately synchronous, not `async def` — making it async caused a real `RuntimeError: asyncio.run() cannot be called from a running event loop` the first time it was wired end-to-end inside W09's own `asyncio.run()`-driven catch-up path; fixed by dropping `async`, not by adding a workaround.

### 6.3 Recommendation cache — `recommendation_cache.py` (W11)

**Not** a reuse of `SnapshotStore` (`src/agent/snapshot_store.py`) — that component's record/replay semantics are purpose-built for backtest determinism (SHA-256 tool-call keys, gitignored corpus); repurposing it for live caching would conflate two different concerns. `RecommendationCache` is a new, SQLite-backed, append-only `recommendation_generations` table keyed by `(match_id, date, agent_config_hash)`. Every generation is kept as its own row — `get_latest()` returns the most recent, `get_history()` returns all of them, doubling as an audit trail. `agent_config_hash.compute_agent_config_hash()` hashes the tunable `AgentConfig` fields (model, provider, temperature, thresholds, markets sorted, prompt version) so a config change naturally produces a new cache key rather than colliding with entries generated under a different configuration.

The **API** never calls `run_agent()` synchronously in the normal path — `GET /api/recommendations/{match_id}` reads exclusively from this cache; `POST /api/recommendations` is the explicit, distinctly-tagged "regenerate now" escape hatch. **The frontend, however, did not actually honor this distinction until W47** — see below.

**W47 — the frontend never read the cache.** `getCachedRecommendation()` (`lib/api.ts`, wrapping `GET /api/recommendations/{match_id}`) was implemented correctly from early on but was never imported or called by any component. `MatchCard`'s expand handler and `MatchAnalysisPage`'s `load()` (`MatchUI.tsx`) both called `POST /api/recommendations` (the live-agent path) unconditionally on every view — so, contrary to decision D2a's design and contrary to the paragraph above (which was only ever true of the backend, not of how the frontend actually used it), every card expansion or analysis-page load made a fresh ~20-30s live Ollama+Tavily call and wrote a new cache row, regardless of whether the EOD batch or a prior view had already generated one. Fixed by having both call sites call `getCachedRecommendation(matchId, date)` first and fall back to `generateRecommendation` only on a `null` (cache miss) — matching W04's own original stated intent for `MatchCard` ("triggers the real 'regenerate now' call *if nothing's cached yet*"), which the code had never actually implemented. A thrown cache-check error is treated the same as a miss (falls through to the live call) rather than surfaced as an error, since the cache check is a speed optimization, not the source of truth. Note this fix is independent of `ENABLE_SCHEDULER` (Section 2) actually being on — without it, the cache stays empty and every view still live-generates exactly as before, just now correctly recorded as the fallback path rather than the only path.

### 6.4 Correctness hardening (W21, W33, W35, W36, W39)

- **W21** — deterministic scheduled-job tests via the already-injectable `now_fn` (no `freezegun` needed; W08/W09/W10 were built test-first with an injectable clock specifically for this). Covers the literal "backend down 22:00→23:30 NY" outage scenario using the real `register_eod_job` wiring across simulated process restarts sharing one on-disk `JobRunLog`.
- **W33 — multi-day soak + DST + reschedule bug.** Found and fixed a real bug, not just a test gap: `schedule_once()` originally keyed its "already ran" marker on the constant `ONCE_RUN_KEY = "once"` rather than `run_at` — a postponed/rescheduled fixture re-registering the same `job_id` (`f"t30_{match_id}"`) with a new `run_at` was permanently blocked from ever firing at its new time by the old marker. Fixed by keying on `run_at.isoformat()` (reflected in the code shown in Section 6.1). Verified across a real 2026-03-08 EST→EDT transition and a 10-simulated-day soak (exactly one EOD fire per day). **Known, accepted gap:** the soak test proves `JobRunLog`'s registration-time catch-up/dedup logic, never APScheduler's own live `CronTrigger` actually firing — that gap is what W41 (Section 13) targets.
- **W35** — integration test for `FileCreditCounterStore`'s save→restart→first-post-boundary-call sequence (the lazy `_roll_month_if_needed()` check was previously only unit-tested per-method, not as a realistic month-boundary restart).
- **W36** — multi-step `_RateLimiter` sequence test (exhaustion → sleep → simulated time crossing the real reset instant → refreshed headers → proceeds) — the three pre-existing tests all used a frozen instant and never exercised the reset-crossing arithmetic for real.
- **W39** — confirms `RecoverableScheduler` and the separate, still-unwired weekly data-refresh scheduler (`src/scheduling/data_refresh_scheduler.py`, US#109) can coexist in one process with no job-id collision, in case they're ever run together.

---

## 7. Bet Tracker & Settlement

### 7.1 Storage — `bet_tracker.py` (W12, D3a)

SQLite-backed (`app/data/user_bets.db`), a `user_bets` table independent of both `SnapshotStore` and the recommendation cache. Schema: `id`, `match_id`, `date`, `home_team`, `away_team`, `market`, `selection`, `odds`, `stake`, `outcome` (`open`|`won`|`lost`), `profit_loss` (`NULL` until settled), `source` (`from_recommendation`|`manual`), `recommendation_snapshot_json` (`NULL` for manual bets), `created_at`.

```python
def settle_bet(self, bet_id: int, outcome: Literal["won", "lost"]) -> Bet:
    profit_loss = bet.stake * (bet.odds - 1) if outcome == "won" else -bet.stake
```

### 7.2 Two logging paths — `bets.py` (W12)

- **From a recommendation** (`POST /api/bets/from-recommendation`) — `BetFromRecommendationRequest` has **no fields at all** for odds/home_team/away_team/date; `resolve_from_recommendation()` derives every one of them from the supplied recommendation snapshot's matching `markets[]` entry, raising `ValueError` (→ HTTP 400) if the market/selection isn't present or its `current_odds` is `null`. The only thing structurally submittable besides which market/selection to bet on is `stake`. The full recommendation dict is stored verbatim in `recommendation_snapshot_json` — necessary because recommendations aren't reproducible run-to-run (`agent_techspec.md` §18.6); the bet must be tied to the recommendation the user actually saw, not a fresh regeneration.
- **Manual** (`POST /api/bets/manual`) — the user supplies every field, but `match_id` must be non-empty (`Field(min_length=1)`) — sourced from Match Explorer's real fixture search, not free-typed team names, so auto-settlement (Section 7.3) still works. No snapshot stored.

### 7.3 Auto-settlement — `settlement.py` (W13)

Sourced from **on-demand `FootballDataClient.get_results()` calls, not a DuckDB `raw_matches` query** — `raw_matches` is a batch-refreshed table (stale relative to real time in-season) and structurally unsuited to near-real-time settlement of a match that just finished. `settle_open_bets(tracker, client)`:

1. Filters `list_open_bets()` down to `RESOLVABLE_MARKETS` (`result_3way`, `btts`, `total_goals`) — a corners bet never triggers even a results API call.
2. Groups the rest by `date` — one `get_results()` call per distinct date, not per bet, respecting the ~10-req/min budget.
3. For each match found `FINISHED`, builds the `actual` outcome via `build_actual_outcome(home_goals, away_goals)` (a new helper in `src/agent/market_resolution.py`) and resolves correctness via `market_correct()` — both extracted out of `src/agent/backtest.py` (which now imports them rather than duplicating), so settlement and backtesting share exactly one resolution implementation. Only `load_outcome()` itself is *not* reused as-is, since it's shaped around a DuckDB row.
4. Calls `BetTracker.settle_bet(...)` for anything that resolves non-`None`.

**Corners markets (`home_corners`/`away_corners`) are excluded from auto-settlement permanently, not just for v1** — `market_correct()` always returns `None` for them (`agent_techspec.md` §11.1: no numeric line field on `MarketRecommendation`, only a free-text `selection` string). This is a cross-team-owned limitation (accepted, revisit deferred indefinitely — see `app_user_stories.md` Integration Gaps #8).

### 7.4 Summary stats — `bet_stats.py` (W14)

`compute_bet_stats(bets, starting_bankroll=1000.0)` reuses `src/agent/evaluation.py`'s `compute_max_drawdown()` verbatim against a notional equity curve built from settled bets' already-computed `profit_loss`, in list/creation order. `build_evaluation_report()` itself is *not* reused as-is — it's shaped around backtest's `BankrollResult`, not the `Bet` dataclass — so ROI/hit-rate are recomputed with the same formulas against the different data shape. Computed only over `won`/`lost` bets on every call (no persisted running total); `bets_open` is reported separately.

---

## 8. Trust & Data-Quality Surfacing (W15, W16)

### 8.1 Structured trust signals (W15)

Before W15, `MatchRecommendation` (`src/agent/schema.py`) had no structured `cold_start_risk`/`feature_completeness`/`unknown_team` fields — only free-text `limitations`, which the system prompt never asked the model to populate with this data. Fixed **at the agent source**, not the app layer (confirmed with the user before crossing into `src/agent/` territory): `graph.py`'s `output_node` was refactored into a testable `_build_recommendation()` + new `_extract_forecast_diagnostics()`, which reads these three fields deterministically from the most recent `forecast_league`/`forecast_international` tool call's own result (already computed by `ForecastService`, previously just never propagated to the final answer) and merges them onto the recommendation **regardless of what the LLM's own JSON says** — the same code-over-prompt philosophy as A28/A29 (`agent_techspec.md` §2, §17).

`app/backend/recommendations.py`'s `MatchRecommendationOut` carries the same three fields, defaulting safely (`cold_start_risk=False`, `feature_completeness=None`, `unknown_team=False`) for any pre-W15 cached data. The frontend's `TrustSignal` component (`MatchUI.tsx`) renders "Unseen team — no history" or "Cold start — thin history" whenever `unknown_team`/`cold_start_risk` is true, independent of the `overall`/`prediction_basis`-driven `StatusBadge` — i.e. a cold-start match visibly reads as lower-trust even if `prediction_basis` itself claims `team_history_and_market` (see US#108, `documents/user_stories.md`).

### 8.2 Defensive rendering for known agent-output quirks (W16)

- **`direct_bet` with `current_odds: null`** (BUG-013, `agent_techspec.md` §18.3) — `isAnomalousDirectBet()` in `MatchUI.tsx` detects this combination and renders an explicit "Data issue" state instead of a misleading green "Direct Bet" badge. Belt-and-suspenders alongside A28's extraction-time downgrade (`agent_techspec.md` §2), since the app shouldn't assume that fix holds for every recommendation it ever sees, including ones cached before A28 shipped.
- **W02-layer type-validation failures** — `Match` carries `invalidMarketCount` (from the API's `invalid_market_count`), surfaced as an explicit "N market(s) omitted — malformed data" note rather than silently rendering fewer markets with no explanation.
- **`insufficient_data`** — was already handled safely by existing `STATUS_META`/empty-markets guards; confirmed, no change needed.

---

## 9. Sandbox / Point-in-Time Testing Environment

Built as Phase 7 (2026-07-16 addition) once off-season conditions made testing full user journeys against real wall-clock "today" impractical (no live fixtures, no live-odds volume). **Not a literal container** — "sandbox" here means an isolated *configuration and data* mode inside the existing app, not Docker.

### 9.1 The clock — `sandbox_clock.py` (W27)

Driven by two env vars, both absent by default (purely additive — normal operation is byte-for-byte unaffected):

```python
SANDBOX_MODE=1
SANDBOX_DATE=YYYY-MM-DD
```

```python
def is_sandbox_mode() -> bool
def sandbox_date() -> date | None
def sandbox_now(tz: tzinfo | None = None) -> datetime   # real now() unless overridden
def sandbox_status() -> dict                              # {"sandbox_mode": bool, "as_of": str | None}
def sandbox_scoped_path(filename: str) -> Path             # app/data/sandbox/<filename>
```

Every backend call site that computes "today" for date-window purposes routes through `sandbox_now()` instead of a bare `datetime.now()` — `scheduler_wiring.py`'s `next_day_date_str()`/`register_eod_job()` default clock, in particular. `sandbox_scoped_path()` was added after W29 review caught a real bug (below) — centralizing the `app/backend/` vs `app/` parent-depth arithmetic in one place so a fourth call site can't repeat the mistake. `GET /api/sandbox/status` exposes `sandbox_status()` so both the frontend (W30) and `scripts/sandbox_runbook.py` (W31) can introspect the active override.

### 9.2 Historical odds — `historical_odds_client.py` (W28)

The Odds API is live-current-odds-only, with no historical replay. `raw_matches` already carries real `odds_h`/`odds_d`/`odds_a` from football-data.co.uk (2016-08-13 through the table's last refresh). `HistoricalOddsClient` implements the **exact same** `get_odds() -> list[NormalizedOdds] | None` shape `OddsAPIClient` does, querying `raw_matches` for the sandbox date instead — `eod_batch.py`/`t30_refresh.py` need zero changes to consume it. A real bug found via TDD: `odds_h`/`odds_d`/`odds_a` are 32-bit `FLOAT` in `raw_matches`, so `1.80` round-trips as `1.7999999523162842` — fixed with `ROUND(CAST(... AS DOUBLE), 2)` (lossless, since football-data.co.uk odds are always quoted to 2dp). Deliberately excludes odds *movement* — a single closing-line snapshot, not a time series (see W32, Section 13).

### 9.3 Wiring — data isolation (W29)

`build_odds_client()` (`scheduler_wiring.py`) returns `HistoricalOddsClient` instead of the live `OddsAPIClient` whenever `sandbox_date()` is set. `RecommendationCache`/`BetTracker`/`JobRunLog` singletons route to `sandbox_scoped_path(...)` (e.g. `app/data/sandbox/recommendation_cache.db`) instead of their real `app/data/*.db` paths whenever `is_sandbox_mode()` is true, so sandbox runs never touch real dev data — resettable by deleting the `sandbox/` directory. Two real bugs caught by review before merge: (1) `main.py`'s original `JobRunLog` path constant used one `.parent` too few, silently escaping `.gitignore`'s `app/data/` pattern — fixed and centralized into `sandbox_scoped_path()`; (2) a test assertion collided with this project's own worktree directory happening to be named `sandbox-testing-environment` — fixed to check the specific path segment, not a raw substring.

### 9.4 Frontend awareness — `useSandboxAsOf.ts` (W30)

```typescript
function useSandboxAsOf(): { asOf: Date; sandboxMode: boolean }
```

Fetches `GET /api/sandbox/status` once per mount; returns the real `new Date()` until/unless sandbox mode is confirmed active, in which case it returns the `as_of` date. Wired into the Dashboard's fixture query, Match Explorer's 90-day window, and `ManualBetForm`'s fixture search. Two real, symmetric timezone bugs were found and fixed during this story (both independent of sandbox correctness per se, but surfaced by it):

1. Constructing `as_of` via local midnight (`new Date(`${as_of}T00:00:00`)`) while every consumer re-serializes via `.toISOString()` (UTC) silently showed the *previous* day's fixtures in positive-UTC-offset timezones — fixed to construct UTC midnight directly (`new Date(as_of)`, correct per ECMA-262 for a bare date string).
2. That fix then exposed the inverse in the 90-day window's date arithmetic (local getters against a now-UTC `Date`) — fixed to use `setUTCDate`/`getUTCDate` uniformly.

A follow-up (surfaced during final whole-branch review, not this story's original scope) found `MatchCard`'s `formatDay()` still called `new Date()` directly for its "today"/"tomorrow" label; fixing it introduced a *third* bug (real, non-sandboxed users needing local getters, not UTC ones, for the real-clock case) — resolved by having `useSandboxAsOf()` return `{ asOf, sandboxMode }` and `formatDay()` branch explicitly on `sandboxMode` (UTC getters only when true). This dual-getter branching is exactly what W40 (Section 13) plans to collapse once the real-clock default is also pinned to a single canonical zone.

### 9.5 SnapshotStore integration for the sandbox agent path (W37)

The app's real agent-invocation path never called `configure_snapshot_store()` before this story — `SnapshotStore.mode` always defaulted to `"live"`, the one mode `web_search`'s `before:<match_date>` leakage filter (`agent_techspec.md` §9.1) does **not** cover. This matters once sandbox mode makes the agent reason about a chosen past date while the real world has moved on. `recommendations.run_agent()` (Section 6.3's `_run_agent_in_mode()`) now:

- Records (`configure_snapshot_store("record", match_key, match_date, base_dir=.../sandbox/)`) the **first** run of a given sandboxed match — one real, date-filtered live call.
- Replays every **subsequent** run of the same match — zero live calls at all, nothing left to leak.

Recordings live in a sandbox-specific namespace (`data/agent_snapshots/sandbox/`), separate from the real evaluation corpus (`agent_techspec.md` §18's A20/A21 pilot). Two real concurrency bugs were found and fixed during implementation: (1) the plan's literal "reset `base_dir` to default when omitted" would have broken an existing autouse test fixture and leaked real snapshot files during ordinary test runs — fixed by making `base_dir` sticky-if-omitted, mirroring `match_id`'s existing convention; (2) a genuine thread-safety race on the `base_dir` global swap, reachable via concurrent `/api/recommendations` requests, could silently drop a sandboxed request back into unfiltered `"live"` mode — fixed with a `threading.Lock` scoped tightly around the fast configure step only, not the slow LLM call.

### 9.6 Frontend date-boundary tests (W38)

`MatchUI.dateboundary.test.tsx` confirms Dashboard's/Match Explorer's date-window queries are anchored to the sandbox `as_of`, not the real browser date, and pick up a new simulated day correctly (via `unmount()`/fresh `render()` — `rerender()` cannot re-trigger `useSandboxAsOf()`'s fetch-once-per-mount effect, since `SANDBOX_DATE` has no live "advance" endpoint). Verified across 4 timezones.

### 9.7 Runbook — `scripts/sandbox_runbook.py` (W31)

A flat, linear `main()` driving the full real user journey against `SANDBOX_MODE=1`/`SANDBOX_DATE=<date>`: Dashboard shows that date's real fixtures with real historical odds; generate a real recommendation (real Ollama + Tavily call, real point-in-time ML features — `FeatureFactory.build_for_match()` was already built for backtesting, so it computes rolling features strictly before the match date, no lookahead); log one bet from the recommendation and one manually; settle both against the real historical result. Non-determinism in the agent's output is expected and accepted — this validates plumbing end-to-end for an arbitrary day, not that the agent's predictions are correct.

The real run (`SANDBOX_DATE=2026-05-24`) surfaced a genuine plumbing bug: bets were initially logged under `RecommendationRequest.effective_match_id()`'s synthetic `home__away__date` key, but `settle_open_bets()` matches by football-data.org's real numeric `match_id` — settlement silently matched 0 of 2 bets on the first attempt. Fixed by passing `match_id=fixture.match_id` explicitly (the same convention `eod_batch.py`/`t30_refresh.py` already use). Results recorded in `documents/sandbox_testing_runbook.md`: a real Sunderland vs Chelsea fixture (2-1), 10 real odds events, a real `llama3.1:8b` + Tavily call producing `overall="conditional"`, two bets logged and correctly settled (`lost`/`-10.0`, `won`/`+5.0`) — real `app/data/*.db` mtimes confirmed byte-identical before/after, only `app/data/sandbox/*.db` written.

### 9.8 One-command interactive launch — `scripts/launch_sandbox.py` (W44)

`scripts/sandbox_runbook.py` (Section 9.7) drives the backend-only pipeline with no server involved — good for a scripted regression check, useless for a human who wants to click through the actual UI. `launch_sandbox.py` fills that gap: `python scripts/launch_sandbox.py <date>` runs a preflight, then boots real `uvicorn`/`next dev` servers for interactive browsing.

**Preflight** (`find_preflight_info()`, also reachable standalone via `--dry-run`, no live calls):

- Queries `raw_matches` for real fixtures on the requested date; if none exist, finds and reports the nearest date that does (day-difference arithmetic over the distinct dates in `raw_matches`, not a hardcoded lookback window).
- Reports a **best-effort** count of existing `data/agent_snapshots/sandbox/` directories for that date (`*__*__<date>` suffix match) — explicitly *not* an exact per-fixture match, since snapshot directory names are keyed on whatever team-name spelling the live caller used at record time, which does not reliably resolve against `raw_matches`'s own Football-Data CSV spelling. Confirmed directly in this repo's own snapshot corpus: `"Nott'm Forest"` (raw_matches), `"Nottingham Forest"`, and `"Nottingham"` all exist as real, different snapshot-directory prefixes for the same team.
- Checks Ollama reachability and `FOOTBALL_DATA_API_KEY`/`TAVILY_API_KEY` presence — reported as warnings, not blockers, since some pages don't need the agent at all.

**Launch:** starts backend (`SANDBOX_MODE=1 SANDBOX_DATE=<date> uvicorn app.backend.main:app`) and frontend (`NEXT_PUBLIC_API_BASE=...` `next dev`) as **separate process groups** (`preexec_fn=os.setsid`), polls each for HTTP health, confirms `GET /api/sandbox/status` reports the requested `as_of`, then writes a `{backend_pid, frontend_pid, ports, started_at}` state file to `/tmp/fpai_sandbox_launch_state.json`. A second launch attempt while the state file is present is rejected rather than silently double-starting.

**Teardown (`--stop`):** reads the state file and `SIGTERM`s both recorded process groups (via `os.killpg`, not the parent PID alone) before clearing the file — verified this avoids the orphaned-child problem a plain PID kill has with `npm run dev` spawning `next dev` as a separate child process; a bare `kill <npm-pid>` leaves `next dev` running.

TDD: `scripts/test_launch_sandbox.py` (23 tests) covers the deterministic, non-networked logic — preflight queries against a temp DuckDB (exact match, nearest-matchday suggestion, no-data case), state-file read/write/clear, `--stop`'s teardown including the already-gone-process case, argument parsing, and (W50) `--precompute`'s fixture-sourcing and env-var-ordering logic — matching `sandbox_runbook.py`'s own precedent (Section 9.7's test file only covers its argument guard; real server launches are verified live, not simulated).

**Verified live** against `SANDBOX_DATE=2025-03-08`: preflight correctly listed the date's 6 real fixtures; both servers passed their health checks; `/api/sandbox/status` confirmed `as_of: "2025-03-08"`; a concurrent second-launch attempt was correctly rejected; `--stop` cleanly terminated both process groups with zero orphaned processes (confirmed via `ps aux`). **One bug this live run caught that the unit tests didn't**: a leftover reference to a renamed `datetime` import crashed `main()` at the final state-write step — *after* both servers had already started successfully, which would have left them running, untracked, with no way to `--stop` them cleanly. Fixed and re-verified end-to-end.

**`--precompute` (W50):** opt-in, default off so a quick launch/dry-run stays quick (backend/agent modules are imported lazily, only when the flag is set). `precompute_recommendations(date_str)` sets `SANDBOX_MODE`/`SANDBOX_DATE` in this script's own process *before* constructing anything sandbox-aware — `recommendations.get_cache()` and `scheduler_wiring.build_odds_client()` both read those env vars at call time, so results land in the exact same on-disk sandbox-scoped cache file the backend subprocess (started afterward) will read from. Fetches the date's real fixtures via `fetch_sandbox_fixtures()` (`get_results()`, not `get_fixtures()` — see Section 6.2's W50 note on why `run_eod_batch()` itself needed a fix for this), then calls `run_eod_batch(..., fixtures=..., on_progress=...)`, printing an incremental per-fixture tally to the terminal as generation proceeds (not buffered to the end). Code-quality review flagged that this ordering property — env vars set before dependency construction — had no direct test, only its two building blocks tested individually; added a dedicated test that mocks the full chain and snapshots `os.environ` at the moment each dependency is actually invoked, empirically verified to catch a real reordering regression before being merged (deliberately broke the ordering, confirmed the test failed with the exact wrong values, restored, confirmed it passed).

**Not included** (out of scope for what was actually requested — a way to launch and browse the app manually, not to automate verifying what renders): a Playwright-driven frontend check (screenshot + fixture-list assertion) was floated as a possible shape for this story before it was scoped, but was not built into this script.

### 9.9 Bugs found by actually running the app (W42, W43)

Found 2026-07-18 by launching the app with `SANDBOX_DATE=2025-03-05` and driving it manually — neither was caught by any prior automated test.

**W42 — stale fixture-fetch response race.** `DashboardPage`, `MatchExplorerPage`, and `ManualBetForm` each re-fetch fixtures on `asOf` changes but, unlike `useSandboxAsOf()`'s own effect, had no guard against a stale response landing after a newer one. Since `useSandboxAsOf()` resolves asynchronously (real `new Date()` first, then the corrected sandbox value), `load()` fires twice, and the real-clock request's response can land *after* the correct one — reproduced via Playwright with captured network timing, the final page showed 60 real off-season fixtures instead of the correct empty sandbox-date result. Fixed by applying the same `cancelled`-flag guard pattern from `useSandboxAsOf.ts` to all three call sites; verified by checking out the pre-fix code and confirming the new race tests (`MatchUI.race.test.tsx`, `BetTracker.race.test.tsx`, using a `deferred<T>()` helper to resolve the earlier request after the later one) fail exactly as expected, then pass after the fix.

**W43 — snapshot replay key-miss crashes with a 500.** Calling `POST /api/recommendations` twice for the identical sandboxed match raised an unhandled `SnapshotMissingError` all the way to a raw FastAPI 500 — the already-documented, accepted limitation that LLM output isn't reproducible run-to-run (`agent_techspec.md` §18.6: a second real LLM call can phrase a `web_search` query differently, missing the SHA-256-keyed recording) had simply never been triggered end-to-end before (every W37 test mocked `_real_run_agent`). Fixed in `recommendations.run_agent()`: a replay-mode `SnapshotMissingError` now triggers exactly one fresh `"record"`-mode retry for that request, matching this codebase's established "degrade gracefully, don't assume the optimization holds" philosophy (W02/W15/W16's `validate_and_degrade`). Any other exception — including a second failure from the retry itself — still propagates uncaught, so this isn't a silent catch-all. A `_LOG.warning(...)` was added at the retry site after code review flagged the original fix as silent, unlike this codebase's precedent for logging comparable fallbacks (the LLM-unreachable startup warning, the league-model-absent tool fallback).

### 9.10 More bugs found by actually using the app (W45–W50)

Found 2026-07-19/20, in the session immediately following W44's `launch_sandbox.py` landing — with a one-command way to actually browse a sandbox date interactively, six real gaps surfaced that no prior test had caught. W45–W47 and W49 are not sandbox-exclusive (they affect real, non-sandboxed usage too, just less visibly); W48 is sandbox-specific by nature; W50 is a sandbox-only feature addition that, in scoping it, surfaced a second instance of W45's own root cause — see below.

**W45 — `/api/fixtures` never sourced `FINISHED` results.** See Section 5.1 for the full fix. Symptom: browsing `SANDBOX_DATE=2025-03-08` showed a blank Dashboard despite 6 real fixtures that day, and searching "Liverpool"/"Chelsea" in Match Explorer or the manual bet-log search bar returned nothing despite both teams having real fixtures in the queried window — all three traced to the identical root cause (`/api/fixtures` only ever called `get_fixtures()`, status `SCHEDULED`). Fixed by splitting the requested range against real wall-clock today and querying `get_results()`/`get_fixtures()` for the respective portions, both sides for the boundary day itself (a same-day match may be already-finished or not-yet-started).

**W46 — blank Dashboard on an empty fixture window.** Once W45 correctly returns nothing for a genuinely fixture-less window (real off-season, or a sandbox date that lands on a rest day), the Dashboard had no fallback beyond a static "No E0 fixtures today" message. `DashboardPage` now fetches a 90-day-forward window (reusing `MatchExplorerPage`'s existing precedent) when the same-day query is empty, sorts and caps at 10, and renders them under an explicit "next matches" label distinct from today's fixtures. The new fallback fetch is guarded by the same `cancelled`-flag pattern (W42) a second, independent time, since it's itself a second in-flight async call that could race a superseded `asOf`/`retryTick` change.

**W47 — the frontend never read the recommendation cache.** See Section 6.3 for the full fix and the corrected framing of what "the dashboard/API reads exclusively from cache" actually meant (true of the backend's `GET /api/recommendations/{match_id}` from day one; never true of how the frontend used it, until this story). Every card expansion or Match Analysis page load made a fresh ~20-30s live agent call regardless of whether a recommendation was already cached — `getCachedRecommendation()` existed correctly in `lib/api.ts` since early in the project but was never imported by any component.

**W48 — sandbox mode leaked real results for fixtures still "in the future" relative to its own pretend `asOf`.** Found immediately after W45/W46 shipped, testing `SANDBOX_DATE=2026-03-08` against real wall-clock time later than that: `fixtureToMatch()` (`MatchUI.tsx`) derived `Match.status` purely from the real-world `FINISHED`/`SCHEDULED` status, with no awareness of `asOf`/`sandboxMode` at all. Since a sandbox date in the past has real completed matches after it too (up through real wall-clock today), the Dashboard's W46 fallback and Match Explorer's search results commonly included fixtures the sandbox is supposed to treat as not-yet-played, but which showed a real final score anyway — exactly the future-outcome leakage the sandbox environment and the agent's own two leakage defenses (`agent_techspec.md` §9.1/§10, §18.7) exist to prevent, just via a surface neither defense covers. A second, related symptom — the recommendation-generation UI *looking* unavailable for these matches (the pre-expand label read "Settled" instead of "Not yet generated") — turned out to be cosmetic only (`handleExpand()` never actually gated on completion status), but this was **explicitly verified, not assumed**: the fix's test suite includes a dedicated test proving a recommendation can still be generated and cached for a future-in-sandbox fixture through the real `handleExpand()` → `getCachedRecommendation()` → `generateRecommendation()`-fallback flow. Fixed by computing an `isFutureInSandbox` check in `fixtureToMatch()` (a fixture's kickoff date strictly after `asOf`'s date, in sandbox mode, forces `status: "upcoming"` and `result: undefined` regardless of real-world status) — reusing `formatDay()`'s pre-existing UTC-vs-local getter branching on `sandboxMode` via an extracted `dayDiff()` helper, rather than hand-writing a second date-comparison implementation and risking a fresh instance of the getter-mismatch bug class W30 (Section 9.4) already found three times in this same file. Deliberately scoped to display/interaction only — real match-completion status remains correct and authoritative for auto-settlement (Section 7.3) and for the sandbox date's own same-day matches, which the fix's strict `> 0` boundary leaves untouched.

**W49 — the manual "regenerate now" path never fetched odds.** See Section 6.2 for the full fix. Diagnosed live, not inferred: a real click-through with captured network timing showed a genuinely-completed (~25s) generation with no `odds` in the request payload, producing an avoidable `insufficient_data` while `raw_matches` had real odds for that exact fixture sitting unused. Fixed by reusing `run_eod_batch()`'s own `build_odds_client()`/`match_odds()` logic in the manual endpoint too.

**W50 — sandbox testing gained an opt-in precompute step, matching the live EOD batch.** Requested directly ("I hope to have them pre-computed, just like in the live case") after W49 made clear that even a fixed manual path still costs one real ~10-30s call per card. See Section 9.8's `--precompute` flag for the full flow. Scoping this surfaced a second, previously-invisible instance of W45's root cause inside `run_eod_batch()` itself (Section 6.2) — fixed as part of the same story, since the precompute feature would otherwise have silently done nothing for any sandbox date.

All six followed the same pattern already established by W42–W44: an implementer subagent's first version was caught with a real, code-review- or spec-review-verified correctness gap before merge for four of them (W45's same-day `FINISHED`-status blind spot; W46's fallback-failure error message overwriting an already-correct empty state; W49's cache write recording the wrong odds value; W50's untested env-var-ordering property) — all fixed and re-verified with their own failing-then-passing regression tests, not just noted and left. W47 and W48's implementations were each approved without a required follow-up; their code-quality reviews noted only non-blocking nitpicks (W47: precedented-elsewhere duplication and an error-masking observability gap; W48: minor parameter-signature asymmetry) — all explicitly assessed as "not a bug," left as-is. W49's review additionally flagged a real, deliberately-not-blocking operational concern (manual regeneration now shares the same Odds API credit budget the scheduler depends on) — documented, not code-changed, since no test could meaningfully "fix" a cost tradeoff.

---

## 10. Data-Validation Layer (W02)

Independent of the agent's own `extract_recommendation()`, which only validates key *presence*, not value types (`agent_techspec.md` §8, §17 — a model could emit `value_edge: "high"` and it would still pass). `app/backend/recommendations.py`'s `validate_and_degrade(raw: dict) -> MatchRecommendationOut`:

- Validates each `markets[]` entry against its own `MarketRecommendationOut` Pydantic model (a genuinely separate model, not reused from `src.agent.schema`); a market that fails validation is **dropped**, not raised — `invalid_market_count` is incremented and a limitations note appended, rather than failing the whole request.
- Every top-level field defaults safely (`overall` → `"insufficient_data"`, `confidence` → `"low"`, etc.) so even a badly malformed payload can't crash the endpoint.

This is deliberately still built even though A28 (`agent_techspec.md` §2) already added strong validation on the agent side — the app should never assume a recommendation it receives now, or from a future cache predating a fix, is trustworthy. Verified live: a real `POST /api/recommendations` call hit tool errors on that run and fell back to `insufficient_data`; the model's raw output had `confidence: ""` (invalid), which `extract_recommendation` correctly rejected and `graph.py`'s own fallback substituted `confidence: "low"` for — all of which passed cleanly through this endpoint's independent validation layer with no special-casing needed.

---

## 11. Frontend Architecture

### 11.1 Pages and routing

Next.js 14 App Router. Four routes, each a thin wrapper delegating to a component in `MatchUI.tsx`/`BetTracker.tsx`:

| Route | Component | Purpose |
|---|---|---|
| `/` | `DashboardPage` | Today's (or the sandbox `as_of` date's) fixtures as `MatchCard`s. **W46:** falls back to up to 10 nearest upcoming/next matches, clearly labeled, when the same-day query is empty rather than rendering a blank page — see Section 9.10. |
| `/matches` | `MatchExplorerPage` | Search real fixtures across a rolling 90-day window. |
| `/matches/[id]` | `MatchAnalysisPage` | Full recommendation detail for one fixture, auto-triggers generation on load. |
| `/bets` | `BetTrackerPage` | Logged-bet list, `StatsBar`, manual-entry form, "Settle open bets" action. |

`app/layout.tsx` (a server component) renders `<StatusFooter/>` (a client component) globally so data-staleness/model-status is visible on every page.

### 11.2 Component library — `MatchUI.tsx`, `BetTracker.tsx`

Ported from the `sandbox/` prototype during W04, validated there first across all three original pages (Dashboard, Match Explorer, Match Analysis) before being wired to real data. Key exported pieces: `TeamBadge`, `StatusBadge`, `TierTag`, `TrustSignal` (Section 8.1), `MatchCard` (expand-to-lazily-generate interaction — triggers `POST /api/recommendations` if nothing's cached, shows a skeleton, then result or an inline retry-capable error), `LogBetButton` (locked-except-stake, Section 7.2), `ProbabilityRow`, `DraftNav`. `BetTracker.tsx` adds `ManualBetForm`, `BetRow`, `StatsBar`. Player/squad/top-features data from the original `DraftUI.tsx` prototype ("Agent Intelligence" section) is **not** returned by the real API — left as an honest empty/"not yet exposed" state rather than inventing new backend surface.

### 11.3 API/type layer — `lib/api.ts`, `lib/types.ts`

`lib/api.ts` exposes one typed async function per backend endpoint (`getFixtures`, `generateRecommendation`, `getCachedRecommendation`, `logBetFromRecommendation`, `logBetManual`, `getBets`, `settleOpenBets`, `getBetStats`, `getStatus`, `getSandboxStatus`), each throwing a shared `ApiError` on a non-2xx response. `lib/types.ts` mirrors the backend's Pydantic models field-for-field as the wire types (`Fixture`, `MatchRecommendationOut`, `Bet`, `BetStats`, `SandboxStatus`, `StatusResponse`) — distinct from `MatchUI.tsx`'s own UI-facing `Match` type, which an adapter layer (`fixtureToMatch`/`applyRecommendation`) maps onto.

### 11.4 Visual design system (D6)

Locked decisions, validated in the `sandbox/` prototype before being ported: dark mode only (`tailwind.config.js`'s `darkMode: "media"`, no light-mode toggle), Tailwind with hand-rolled components (no shadcn/Radix), high visual density with card-based match displays, real club-color identity badges (`teamColor()`/`badgeColor()` helpers in `MatchUI.tsx`), system-ui sans, Phosphor icons, plain CSS transitions (150ms, no animation library). Color tokens (`--page-plane`, `--surface-1`, `--text-primary`, `--status-good`/`warning`/`serious`/`critical`, etc.) are defined as CSS custom properties in `app/globals.css` and consumed via Tailwind's `theme.extend.colors` — so status semantics (`good`/`warning`/`serious`/`critical`) are named consistently across `StatusBadge`, `TrustSignal`, and `StatusFooter`.

### 11.5 Frontend test infrastructure (W22)

No frontend test tooling existed before this story (deliberately deferred through W04/W12/W15/W16, per this codebase's convention of verifying UI changes live in a browser first). Added Vitest 2 + React Testing Library + `@testing-library/user-event` + jsdom, pinned below Vitest 4/Vite 8 to avoid an `@types/node` peer-dependency conflict with the Next 14/React 18 stack. `MatchCard`/`StatusBadge`/`LogBetButton` and their supporting types were made exported (visibility only) so they're directly testable. `components/__tests__/MatchUI.test.tsx` (14 tests) parametrizes `StatusBadge`/`MatchCard` across all four `overall` values plus the `hasRecommendation: false` state, and proves `LogBetButton`'s locked-except-stake behavior via `getAllByRole("textbox")` length checks. All 14 tests passed on the first run against already-shipped components — this story added coverage/tooling, not new behavior.

---

## 12. Testing & Release Readiness

### 12.1 Backend test strategy (W20)

Every HTTP-boundary client (`FootballDataClient`, `OddsAPIClient`) and every `run_agent` call site is mocked at its module-qualified name in the tests that exercise it — audited and confirmed already the case project-wide. `app/backend/tests/conftest.py` adds an **autouse** fixture that monkeypatches `socket.socket.connect`/`connect_ex` to raise for the whole `app/backend/tests/` tree, so an accidental real network call fails loudly instead of hanging or flaking; a registered (currently unused) `@pytest.mark.live` marker is the documented opt-out for a genuinely-required real call. `app/backend/tests/` currently comprises 29 modules (health, LLM check, match-info gating, football-data client, odds client, historical-odds client, recommendation schema/cache/endpoints, EOD batch, T-30 refresh, scheduler + soak + integration + wiring, bets schema/endpoints, bet stats, settlement, status, sandbox clock/status/wiring/agent-snapshot, network guard, fixtures endpoint).

### 12.2 Scheduled-job determinism (W21, W33) and rate-limiter/credit-counter sequencing (W35, W36)

See Section 6.4 — all four stories close gaps between "each piece unit-tested in isolation" and "the realistic multi-step sequence," and W33 found a genuine bug (Section 6.1/6.4) along the way.

### 12.3 Frontend test infrastructure (W22)

See Section 11.5. `app/frontend/components/__tests__/` also carries `MatchUI.dateboundary.test.tsx` (W38), `MatchUI.race.test.tsx`/`BetTracker.race.test.tsx` (W42), and `StatusFooter.test.tsx`; `lib/useSandboxAsOf.test.ts` covers the hook directly.

### 12.4 Pre-launch smoke test (W23) and its follow-up (W26)

`documents/prelaunch_smoke_test_checklist.md` — authored and **run for real** (2026-07-12), not simulated: a real EOD batch against real Ollama + real football-data.org for the actual next scheduled E0 fixture; a real T-30 refresh (correctly `skipped_no_odds`, no Odds API key at the time); one from-recommendation and one manual bet, both settled via a real `settle_open_bets()` call against a real completed historical fixture (substituted for a same-day live match, which wasn't achievable in one session — explicitly noted as a substitution); Odds-API-unavailable graceful degradation (confirmed true in-environment, not simulated). All four items recorded PASS with real values. Two genuinely-blocked items (a true live-match-completes-today settlement, and the T-30 "odds changed" path against a real Odds API response) were split into **W26** (`future` — blocked on timing and a real Odds API key landing first), rather than left silently undone.

---

## 13. Known Limitations / Future Work

Stories still `future` or `active` as of this writing — described briefly, since none is built yet.

- **W18 (future) — Multi-league expansion.** Blocked on ML-engine-side work, not app work: ingesting raw data for new leagues, training real per-competition model sets, and wiring `config/competitions.yaml` into `ForecastService.forecast_upcoming()`'s live path for leagues beyond E0. Once that lands, W03's hardcoded `COMPETITION_ALLOWLIST` (Section 4) becomes a registry lookup instead of a literal.
- **W19 (future) — Batch/weekend fixture view.** A UI view over recommendations W09's scheduled batch already generated and W11 already cached, grouped for a weekend-at-a-glance read. Per the "Agent invocation ownership" decision (`app_user_stories.md`), this is never a call into `A18`/any agent-side batch mechanism — the app never delegates orchestration to the agent side.
- **W24 (future) — Odds provider exploration.** Compares The Odds API's free tier against football-data.org's own Odds Add-on, The Odds API's paid tiers, and other providers, on cost-at-real-volume, region/market coverage, kickoff-proximity update frequency, and multi-league readiness for when W18 unblocks. Exploratory — no rush while the free tier suffices.
- **W26 (future) — Remaining W23 checklist items.** See Section 12.4.
- **W32 (future) — Sandbox odds-movement scenario.** `raw_matches` carries a single pre-match/closing odds snapshot, not a time series — exercising T-30's "odds changed → agent re-runs" path against a chosen sandbox date would need a synthetic odds-change sequence layered on `HistoricalOddsClient`. Explicitly deferred, not part of the W27–W31 batch.
- **W40 (active) — Frontend timezone standardization.** Pins production's non-sandboxed frontend "now" to `America/New_York` everywhere (a single `nowInEasternTime()` helper), matching the backend's `NY_TZ` convention throughout `scheduler.py`, rather than falling back to the browser's ambient local timezone. Intended to also collapse `formatDay()`'s `sandboxMode` UTC/local-getter branching (Section 9.4) to one code path, since there would then be only one canonical reference frame, real or sandboxed.
- **W41 (active) — Scheduler live-fire smoke test.** Every existing scheduler test (W08/W21/W33) exercises `RecoverableScheduler`'s own immediate catch-up check, never APScheduler's live `CronTrigger` actually firing on a background thread — a `CronTrigger`-specific bug (version quirk, timezone-object mismatch) would slip through today. Planned as a short, real-wall-clock-wait test marked `@pytest.mark.live` (excluded from the default fast run).

**Standing, accepted limitations (not tracked as stories — see `app_user_stories.md` Integration Gaps for the full table):**

- Corners markets (`home_corners`/`away_corners`) cannot be auto-scored (Section 7.3) — a cross-team-owned gap requiring a `MarketRecommendation` schema change, deferred indefinitely.
- LLM output isn't reproducible run-to-run (`agent_techspec.md` §18.6) — mitigated but not eliminated by W10's skip-if-odds-unchanged logic and W12's snapshot-at-log-time design; W43's replay-miss retry (Section 9.9) is the sandbox-specific mitigation.
- Result-leakage defenses (`web_search`'s `before:<date>` filter, the system-prompt instruction) are a mitigation, not a guarantee (`agent_techspec.md` §18.7) — accepted, low relevance for genuinely-upcoming live fixtures, worth revisiting before any "review past agent calls" feature is built on `SnapshotStore`.
- The forecast engine requires a local Ollama daemon (or an Anthropic API key) to answer anything — confirmed as an accepted operational dependency, not a gap, per the 2026-07-11 decision to stay on local Ollama for now.
- The `Odds API`'s free tier (~500 credits/month) leaves no slack for adding markets, regions, or leagues without a paid tier — the credit counter (Section 5.3) makes this visible and gates gracefully, but doesn't create headroom.

---

## 14. Implementation Status

Reflects `documents/app_user_stories.md` as of this writing.

| Phase | Stories | Status |
|---|---|---|
| 1 — Backend Foundation & Single-Match Wiring | W01–W04 | ✅ Implemented — FastAPI scaffold, `POST /api/recommendations`, W03 gating, real-data frontend wiring (Sections 2–4, 11) |
| 2 — Fixture, Odds Discovery & Scheduled Batch Generation | W05–W11, W25 | ✅ Implemented — football-data.org + Odds API clients, team-name mapping, recommendation cache, EOD/T-30 scheduler (Sections 5, 6) |
| 3 — Bet Tracker & Settlement | W12–W14 | ✅ Implemented (Section 7) |
| 4 — Trust & Data-Quality Surfacing | W15–W16 | ✅ Implemented (Section 8) |
| 5 — Future Extensions | W17 (done), W18, W19, W24 | W17 ✅ Implemented (status footer); W18/W19/W24 ⬜ Future (Section 13) |
| 6 — Testing & Release Readiness | W20–W23, W26 | W20–W23 ✅ Implemented (Section 12); W26 ⬜ Future, blocked on timing + a real Odds API key |
| 7 — Sandbox / Point-in-Time Testing Environment | W27–W31, W32 | W27–W31 ✅ Implemented (Section 9); W32 ⬜ Future, deliberately deferred |
| 8 — Time-Related Correctness Testing | W33–W39 | ✅ Implemented (Section 6.4) |
| 9 — Frontend Timezone Standardization & Scheduler Live-Fire Coverage | W40–W41 | 🟡 Active — not yet started (Section 13) |
| 10 — Bugs Found By Actually Running The App | W42–W44 | ✅ Implemented — W42–W43 (Section 9.9), W44 (Section 9.8) |
| 11 — More Bugs Found By Actually Using The App | W45–W50 | ✅ Implemented (Section 9.10) |

**Test counts as of the most recent completed stories (W45–W50, 2026-07-20):** backend (`tests/` + `app/backend/tests/` + `scripts/`) 540 passed / 23 skipped / 1 deselected, zero regressions since W44's 533/1/1 baseline (the skip count reverted to 23 in this worktree; not investigated as part of this story, same caveat noted at W44). Frontend (Vitest + RTL) unchanged from W48 at 42 tests across 7 files (`tsc --noEmit` clean) — W49/W50 are backend/CLI-only, no frontend files touched.

---

**Cross-references:** agent-internal behavior (LangGraph state machine, tool wrapping, snapshot record/replay semantics, backtest harness, model-selection findings) is documented in `agent_techspec.md`, not repeated here. Forecast-engine-internal behavior (feature store, competitions registry, model selection, cold-start/`unknown_team` diagnostics) is documented in `FRAI_TECHSPEC.md`. This app's own story tracking, including the full "Integration Gaps & Concerns" table this spec draws its cross-team callouts from, lives in `app_user_stories.md`.
