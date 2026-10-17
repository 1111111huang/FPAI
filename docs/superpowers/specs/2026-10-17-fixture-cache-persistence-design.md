# Fixture Cache Persistence

## 1. Problem

`app/backend/fixture_cache.py`'s per-day cache (`_cache`, a module-level Python dict) is purely in-memory. Every process restart — every Railway redeploy — wipes it completely, forcing the app back to a cold start: every league's fixtures/results have to be re-won against football-data.org's shared, rate-limited budget from scratch.

This was directly observed live on 2026-10-17: several deploys in a row (each shipping an unrelated fix) each triggered the same cold-start cascade — competitions missing entirely, then appearing one at a time as each won its turn at the rate limit, sometimes taking several minutes to fully recover. BUG-084/085 (same day) made a *failed* refetch degrade gracefully by falling back to stale in-memory data — but right after a restart there is no stale data to fall back to at all, since the cache itself no longer exists.

football-data.org RESULTS (finished matches) already have exactly this kind of cross-restart persistence via `results_cache.py`'s `ResultsCache` (SQLite, `data/match_results_cache.db`) — but that only covers `FootballDataClient.get_results()`, not `get_fixtures()` (upcoming/live matches), and not the day-level TTL cache `fixture_cache.py` itself maintains on top of both.

## 2. Scope

**In scope:**
- A new SQLite-backed store (`app/backend/fixture_cache_store.py`) persisting `fixture_cache.py`'s own `(call_type, day) -> matches` entries, mirroring `results_cache.py`'s `ResultsCache` pattern exactly (same stdlib `sqlite3`, same per-store dedicated file under `data/`, same fresh-connection-per-call thread safety).
- Boot-time hydration: load everything persisted (after pruning anything older than a 2-week retention window) into the in-memory `_cache`, each entry marked already-expired.
- Write-through: every `store_day()` call also persists to the new store.

**Out of scope (explicitly not doing):**
- Changing `ResultsCache` itself, or merging the two stores into one. They serve different consumers (`ResultsCache` backs `FootballDataClient.get_results()` directly, used by settlement/recommendation-outcomes callers independent of `fixture_cache.py`; this new store backs `fixture_cache.py`'s own merged results+fixtures view for `/api/fixtures` and the reconciliation job). Some result data ends up persisted in both stores — accepted, minor, consistent with this codebase's existing "each concern gets its own small store" convention (`push_subscriptions.db`, `user_bets.db`, `recommendation_cache.db`, `match_results_cache.db` are already four separate files for four separate concerns).
- A dedicated scheduled prune job (the `recommendation_cache.db` pattern, `register_recommendation_cache_prune_job`). Rejected per direct user decision — a boot-time prune is enough given how often this app actually redeploys in practice; revisit only if a single deployment is ever expected to stay up for weeks without a restart.
- Reconstructing a "real" TTL across the restart boundary. A monotonic clock means nothing across a process boundary anyway — marking every loaded entry as already-expired is sufficient and correct, not a simplification that loses anything: `_missing_days()` will still attempt a background refresh on first use, `_read_days()` (BUG-084) still serves the loaded data immediately in the meantime.
- Any change to `get_range()`/`force_refresh_range()`/`_read_days()`/`_missing_days()`'s own logic. All of today's BUG-084/085 fallback behavior is reused completely unchanged — this only changes where `_cache`'s contents come from at boot and where `store_day()` additionally writes to.

## 3. Storage: `fixture_cache_store.py`

New file, new class `FixtureCacheStore`, new default path `data/fixture_cache.db` (`Path(__file__).resolve().parents[2] / "data" / "fixture_cache.db"`, same convention as every other per-store `DEFAULT_DB_PATH` in this codebase).

Two tables, mirroring `ResultsCache`'s own `match_results`/`fetched_queries` split (needed for the same reason: a genuinely empty day must be distinguishable from a never-fetched one):

```sql
CREATE TABLE IF NOT EXISTS cached_days (
    call_type TEXT NOT NULL,
    day TEXT NOT NULL,
    PRIMARY KEY (call_type, day)
)

CREATE TABLE IF NOT EXISTS cached_matches (
    call_type TEXT NOT NULL,
    day TEXT NOT NULL,
    match_id TEXT NOT NULL,
    utc_date TEXT NOT NULL,
    status TEXT NOT NULL,
    home_team TEXT NOT NULL,
    away_team TEXT NOT NULL,
    home_goals INTEGER,
    away_goals INTEGER,
    PRIMARY KEY (call_type, day, match_id)
)
-- plus an index on (call_type, day), same as ResultsCache's idx_match_results_lookup
```

No `competition` column -- `NormalizedMatch.competition` is still at its dataclass default (`"E0"`) at the point `fixture_cache.store_day()` is called (tagging happens later, in `main.py`'s own `_tag()` helper, after `get_range()` returns), so persisting it would just be a wrong value. Same reasoning `ResultsCache` already documents for leaving it unset.

Three methods:
- `store_day(call_type: str, day: str, matches: list[NormalizedMatch]) -> None` — upserts `cached_days`, replaces `cached_matches` for that key (delete-then-insert, simplest correct way to handle a day shrinking from N matches to fewer on a refetch).
- `prune_older_than(cutoff_day: str) -> None` — deletes every `(call_type, day)` (both tables) with `day < cutoff_day`.
- `load_all() -> dict[tuple[str, str], list[NormalizedMatch]]` — everything currently persisted, grouped by key. An entry with zero matches still appears (as an empty list) if its `(call_type, day)` has a `cached_days` row.

## 4. Wiring into `fixture_cache.py`

Module-level `_store = FixtureCacheStore()`, constructed eagerly at import time (cheap -- just `CREATE TABLE IF NOT EXISTS`, no meaningful I/O).

New function:

```python
RETENTION_DAYS = 14

def load_from_disk() -> None:
    cutoff = (date.today() - timedelta(days=RETENTION_DAYS)).isoformat()
    _store.prune_older_than(cutoff)
    persisted = _store.load_all()
    now = now_monotonic()
    with _cache_lock:
        for (call_type, day), matches in persisted.items():
            _cache[(call_type, day)] = (now - 1, matches)
```

`store_day()` gains one line, after the existing in-memory write (outside `_cache_lock` -- the lock only needs to guard the dict, not block on disk I/O):

```python
def store_day(call_type: str, day: str, matches: list[NormalizedMatch]) -> None:
    ttl = compute_ttl_seconds(matches, wall_clock_now())
    with _cache_lock:
        _cache[(call_type, day)] = (now_monotonic() + ttl, matches)
    _store.store_day(call_type, day, matches)
```

## 5. Calling `load_from_disk()` at boot

`main.py`'s `lifespan`, inside the existing `if os.environ.get("ENABLE_SCHEDULER", "").lower() in ("1", "true", "yes"):` block -- same gate, same justification already documented right next to it for the boot-time pregenerate-recommendations hook (W103): an unconditional call here would fire on every test file's `with TestClient(app)` too. Production already runs with `ENABLE_SCHEDULER=1`, so this costs nothing there.

`store_day()`'s write-through is NOT gated -- it runs unconditionally, same as `ResultsCache`/`RecommendationCache`/etc. already do today when a test doesn't explicitly override their default real-file path. This is an accepted, pre-existing class of risk in this codebase, not a new one.

## 6. Testing

- `FixtureCacheStore` unit tests (new `test_fixture_cache_store.py`, `tmp_path`-scoped, same convention as `JobRunLog(db_path=tmp_path / ...)`/`RecommendationCache(db_path=tmp_path / ...)` elsewhere): store-then-load round-trips matches correctly; an empty day round-trips as an empty list, distinguishable from a day that was never stored (`load_all()` has no key for it at all); `prune_older_than` removes old entries from both tables and leaves newer ones; storing the same `(call_type, day)` twice replaces rather than accumulates.
- `fixture_cache.py` tests (`test_fixture_cache.py`, which already has an autouse `_clear_cache` fixture): extend that fixture to also monkeypatch the module's `_store` to a fresh `tmp_path`-scoped `FixtureCacheStore`, fully isolating these tests from the real `data/fixture_cache.db`. New tests: `store_day()` persists (read back via the test's own `_store` instance, bypassing `fixture_cache.py`'s own in-memory dict); `load_from_disk()` seeds `_cache` from whatever the store has, with every entry already expired (confirm via `_missing_days()` treating it as missing right after load, while `_read_days()` still returns it); `load_from_disk()` prunes before loading (an old entry seeded directly into the store is gone from `_cache` after `load_from_disk()`).
- No new test needed for the `ENABLE_SCHEDULER` gating itself -- it's the exact same gate `main.py`'s existing pregenerate-recommendations hook already uses, with its own existing test coverage for "off means skipped."
