# Background Job Safety Net & Ops Dashboard

## 1. Problem

Three separate production incidents hit the same day (2026-10-31), each a different shape of the same underlying gap:

- **BUG-099**: `FallbackOddsClient`/`FallbackOddsPapiClient`'s exhausted-key fallback loop is synchronous (plain `requests.get` calls, up to N keys sequentially), called directly from async code. When every key was genuinely down, this froze the whole process's single asyncio event loop for the full fallback duration — not just the request that triggered it, every other concurrent request too, including unrelated ones like `/api/health`.
- **BUG-100**: when the agent's tool-call budget ran out on the exact turn the model also requested a tool call, the resulting message sequence sent to the LLM violated the provider's contract (an assistant `tool_calls` message with no matching `ToolMessage`), producing a deterministic 400 on all 3 retry attempts — silently skipping the match every single regen attempt, with retrying never helping.
- **BUG-101**: an empty/corrupt credit-counter JSON file raised an unhandled `JSONDecodeError` out of `build_odds_client()`, called once in `_pregenerate_recommendations`, outside any per-league `try/except` — one corrupt file silently killed the *entire* pregenerate pass for every league in that request.

A follow-up audit of the whole background/async surface (scheduled jobs, admin-triggered batches, T-30 refresh, the shared helpers they all call) found this wasn't three unrelated bugs — it's one recurring pattern surfacing in three different spots:

1. **Isolation granularity is inconsistent and is a property of the call site, not the function.** The same function (`add_secondary_odds`) is wrapped in a `try/except` in `eod_batch.py` and completely unguarded in `t30_refresh.py`. Isolation silently regresses whenever a shared helper gets reused somewhere new.
2. **"Shared one-shot builder, called fresh, unprotected" recurs at least 3 times**: `build_odds_client()`/`build_oddspapi_client()` (BUG-101), and the same shape exists unprotected in `lifespan()`'s boot sequence (`DuckDBManager()`, every `register_*` job, `scheduler.start()` — none of it has a `try/except`; any exception there fails the entire app's boot, not just one subsystem).
3. **Three different containment tiers coexist with no stated rule**: a narrow catch with a structured, logged outcome; a catch only via `asyncio.gather(..., return_exceptions=True)`'s broad umbrella (contained, but logged as "failed outside the normal skip path" and never counted in the batch summary — invisible to anything reading those counts); and, for T-30 specifically, no umbrella at all — just the scheduler's generic per-job log line.
4. **No structured failure record exists anywhere.** `documents/bugs.md` is hand-written, after the fact, by a human (or this session) manually grepping live Railway logs — which is exactly what today's investigation needed three separate rounds of to find each root cause.

A fourth, explicitly *separate* incident (**BUG-102**, 2026-11-01) surfaced during this same investigation: the scheduled `ingest` subprocess's walk-forward Dixon-Coles feature rebuild starved the live web server of CPU on the shared Railway container. This is cross-process CPU contention on a shared container, not an isolation gap within the web server's own event loop — out of scope for this design, logged separately in `documents/bugs.md`.

## 2. Goals

- **Containment**: no single item's failure (one fixture, one league, one key, one scheduled job's unit of work) can block, slow, or cancel any sibling unit — anywhere in the background/async surface, with one deliberate exception (below).
- **Observability**: every caught failure is recorded as a structured, queryable row — not just a log line — so reviewing "what broke, where, how often" doesn't require live-log archaeology.
- **Faster recovery from a known-dead vendor.** Today, every fixture re-discovers a full vendor outage from scratch; a circuit breaker lets the system remember.
- **A reviewable surface** for odds availability, feature health, and recommendation performance, in one place, actually linked into the app (not another orphaned admin page).

## 3. Non-goals (explicitly out of scope)

- **Graceful boot-time degradation.** Direct decision: a broken boot stays a hard, loud crash, visible in Railway immediately. The isolation principle below applies to steady-state background work, not startup integrity — a half-booted app silently serving some routes and not others is a worse failure mode than a crash-loop Railway surfaces clearly.
- **Cross-process/CPU-level isolation** (BUG-102's class). This design only covers isolation within the web server's own process/event loop. Running `ingest` on a separate service, a CPU cap, or confirming it can't overlap daytime traffic is a different, deliberately deferred problem — tracked as BUG-102.
- **A new auth/role system.** `/ops` reuses the existing Google-login allowlist and the existing `NEXT_PUBLIC_APP_ACCESS_TOKEN` → `x-app-token` → `/api/admin/*` mechanism unchanged. No admin/non-admin role distinction — the allowlist today is one person; a tiered-role system would solve a problem that doesn't exist yet.
- **A stale-vs-fresh indicator on `MatchCard`.** Found along the way: a cached recommendation from 2 minutes ago and one from 3 days ago (served via `get_latest_any_config()`'s fallback) render identically today. Real product question, not a backend safety-net question — flagged, not solved here.
- **Smarter retry policy for `_invoke_with_retry`** beyond what BUG-100's own fix already provides. That fix was at the source (the message sequence itself); this design doesn't change the 3x-no-backoff retry policy itself.

## 4. `run_isolated` — the core primitive

```python
def run_isolated(subsystem: str, unit_id: str, fn: Callable[[], T]) -> T | None:
    """Runs fn(), catching any exception. On failure: writes one row to the
    failures table, fires a debounced alert if this is the first occurrence
    of this (subsystem, exc_type) pair within the alert cooldown window, logs
    as today, and returns None instead of raising. Never re-raises -- the
    caller's loop always continues to the next unit."""
```

Async callers use an `async def run_isolated_async(...)` wrapping the same contract around `await fn()` (needed for the LLM/odds calls already behind `asyncio.to_thread`).

Applied at every per-item boundary the audit found, replacing today's inconsistent per-caller `try/except`:

| Call site | Today | Target |
|---|---|---|
| `eod_batch.py` `_generate_one`: `add_secondary_odds` | narrow catch (BUG-093) | `run_isolated` — same behavior, now also writes a failures row |
| `eod_batch.py` `_generate_one`: `run_agent` | narrow catch + `to_thread` | `run_isolated_async` |
| `eod_batch.py` `_generate_one`: `has_kicked_off`, `already_fresh`, `validate_and_degrade`, `cache.record_generation` | broad-umbrella only (`gather(return_exceptions=True)`), uncounted | `run_isolated` each — now narrowly caught, logged by name, counted |
| `t30_refresh.py`: `add_secondary_odds`, `cache.record_generation` | **unguarded** | `run_isolated` — closes the live gap |
| `scheduler_wiring.py` job bodies: push-notify step (EOD job, fixture-reconciliation job) | unguarded | `run_isolated` |
| `main.py` `_pregenerate_recommendations`: `build_odds_client()`/`build_oddspapi_client()` | unguarded (BUG-101's site; the `JSONDecodeError` itself is now fixed at the source in `FileCreditCounterStore.load()`, but the call site has no general protection) | `run_isolated` — defense in depth for any *other* exception that function could raise |
| `main.py` `lifespan()`: `register_*` job bodies | unguarded | **not wrapped** — boot stays fail-loud, by design (§3) |

`run_isolated` itself must never be the thing that breaks: the failures-table write and the alert both follow `slack_alerts.post_slack_alert`'s existing "never raises" contract — a failure to *record* a failure degrades to a log line, never to a second unhandled exception.

## 5. Circuit breaker — one per vendor fallback client

Not per key (the existing per-key cooldown in `CreditCounter.mark_unavailable`/`is_available` already handles "this one key is bad"). This is for "the whole chain is bad right now" — the gap that made today's OddsPapi incident take ~6-7 minutes per league: its fallback loop's global 5-second throttle (`_MIN_REQUEST_INTERVAL_SECONDS`, `oddspapi_client.py`) means cycling all 10 keys costs ≥50 seconds, paid again by every fixture, with no memory that every key was already proven dead moments earlier. `asyncio.to_thread` doesn't parallelize this away — the throttle is one global lock, so per-fixture costs stack rather than overlap.

```python
class CircuitBreaker:
    """CLOSED (normal) -> OPEN (3 consecutive full-chain exhaustions trip it;
    further calls return None instantly, zero network calls) -> HALF-OPEN
    (after a cooldown, e.g. 3 min, exactly one real call is let through) ->
    CLOSED on success, or back to OPEN with a doubled cooldown on failure."""
```

One instance per `FallbackOddsClient`/`FallbackOddsPapiClient`, gating the top of `_try_each_client`. `CLOSED` and `HALF-OPEN` run today's exact per-key loop unchanged — this is purely additive, not a rewrite of the fallback logic itself.

## 6. Failures table

New store, same convention as every other per-concern store in this codebase (`job_runs.db`, `recommendation_cache.db`, `fixture_cache.db` — one small SQLite/DuckDB file per concern, not a shared catch-all).

```sql
CREATE TABLE IF NOT EXISTS failures (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    ts TEXT NOT NULL,
    subsystem TEXT NOT NULL,       -- e.g. "pregenerate", "t30", "eod_batch:I1"
    unit_id TEXT NOT NULL,         -- e.g. "match_id=558542", "league=I1"
    exc_type TEXT NOT NULL,
    message TEXT NOT NULL,
    traceback TEXT NOT NULL,
    retryable INTEGER,             -- always NULL for now -- see note below
    alerted INTEGER NOT NULL DEFAULT 0
)
-- index on (subsystem, ts) for the /ops "failures" tab's date+league filter
```

`retryable` is reserved, not populated, by this design. Automatically distinguishing "transient" from "deterministic" (the BUG-100 vs. ordinary-timeout distinction in Fig. 2 of the Blast Radius Map artifact) needs its own classifier this spec doesn't define — guessing at one here would be scope creep without a real mechanism behind it. The column exists so a future pass can backfill it without a schema migration; until then, the `/ops` failures tab reads `exc_type` and `message` directly, same as a log line would.

`data/failures.db`, same `Path(__file__).resolve().parents[2] / "data"` convention as the rest.

## 7. `/ops` dashboard

New route in the existing Next.js frontend, four tabs behind one shared filter bar (date range + league, applied to all four — not four independently-filterable views that can drift out of sync):

| Tab | Status | Shows | Source |
|---|---|---|---|
| Failures & circuit breakers | new | Failure rows by subsystem/unit; live breaker state (CLOSED/OPEN/HALF-OPEN) per vendor; trip history | `failures` table + a small in-memory breaker-status endpoint |
| Odds availability | new | Per fixture/market: live, or missing with cause — circuit open (transient) vs. line not posted (normal) vs. never attempted | Live query over current cache + breaker state; reuses `missing_live_odds()`'s logic as a query instead of only firing once at generation time |
| Feature health | new | `feature_completeness` trend, `cold_start_risk` rate, `unknown_team` rate, by league/date | Scans `forecast_payload`/`data_quality`, already stored on every cached recommendation — nothing new to compute, only to aggregate |
| Recommendation performance | **link, don't rebuild** | Hit rate, sample size, Kelly-simulated ROI, segment breakdowns | Already built: `GET /api/recommendations/stats`, `/performance-dashboard`, the existing (currently unlinked) `agent-performance` page |

**Access**: `/ops`, behind the same Google-login gate every other page already uses. No new auth code — the frontend's `lib/api.ts` already attaches `NEXT_PUBLIC_APP_ACCESS_TOKEN` as `x-app-token` to admin calls; this page only needs route wiring, not a login flow of its own.

## 8. Testing

- `run_isolated`/`run_isolated_async`: success path (returns `fn()`'s result, no failures-table write), failure path (returns `None`, one failures row written, no re-raise), alert debounce (second failure within the cooldown window doesn't re-alert) — mirrors the existing `test_scheduler_wiring.py`/`test_eod_batch.py` conventions already in this codebase.
- `CircuitBreaker`: state-transition tests (CLOSED→OPEN after 3 exhaustions, OPEN short-circuits with zero calls to the wrapped loop, HALF-OPEN allows exactly one call through, success closes it, failure re-opens with a doubled cooldown) — unit-level, no real vendor calls, same `monkeypatch`/fake-clock style as `test_oddspapi_credit_burn.py`.
- `/ops` endpoints: filter correctness (date range + league applied consistently across all four data sources), empty-state rendering (no failures yet, no cold-start data yet).
- Regression: re-run `test_eod_batch.py`, `test_scheduler_wiring.py`, `test_oddspapi_client.py`, `test_odds_api_client.py` after wiring `run_isolated` into existing call sites — behavior must be unchanged for every already-passing case, only the isolation/recording mechanism changes.

## 9. Deferred / tracked separately

- BUG-102 (cross-process CPU contention, `ingest` vs. the live web server) — own investigation, own fix.
- BUG-099/100/101 are fixed in code but not yet logged as their own rows in `documents/bugs.md`.
- The stale-vs-fresh indicator on `MatchCard` (§3).
