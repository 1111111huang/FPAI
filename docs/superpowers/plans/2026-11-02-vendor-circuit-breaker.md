# Vendor Circuit Breaker Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Stop `FallbackOddsClient`/`FallbackOddsPapiClient` from re-discovering a fully-dead vendor from scratch on every single call. A circuit breaker gates the top of each fallback chain's per-key loop: after 3 consecutive full-chain exhaustions it opens, short-circuiting further calls to an instant `None` (zero network calls, zero throttle wait) for a cooldown window, then lets exactly one real call through to probe recovery.

**Architecture:** A standalone, dependency-free `CircuitBreaker` state machine (`app/backend/circuit_breaker.py`) — CLOSED / OPEN / HALF-OPEN, no I/O of its own. One instance per vendor, held as a **process-level singleton** (not rebuilt on every `build_odds_client()`/`build_oddspapi_client()` call) — this is the part that actually matters: those two functions are called fresh on *every* pregenerate/EOD invocation, so a breaker instantiated inside `FallbackOddsClient.__init__` would lose all memory of the outage the instant the next call rebuilt it, reproducing exactly the "re-discover the outage every time" problem this plan exists to fix. `CLOSED` and `HALF-OPEN` run today's exact per-key loop unchanged — this is additive, not a rewrite.

**Tech Stack:** Python 3.13, stdlib `time.monotonic`, pytest with a fake clock (`now_fn` injection, same pattern `CreditCounter` already uses for `now_fn`).

---

### Task 1: `CircuitBreaker` state machine

**Files:**
- Create: `app/backend/circuit_breaker.py`
- Test: `app/backend/tests/test_circuit_breaker.py`

- [ ] **Step 1: Write the failing tests**

```python
"""Tests for CircuitBreaker -- CLOSED/OPEN/HALF-OPEN gate in front of
FallbackOddsClient/FallbackOddsPapiClient's per-key fallback loop.
2026-11-02 design: docs/superpowers/specs/2026-11-02-background-job-safety-net-design.md"""

from __future__ import annotations

from app.backend.circuit_breaker import CircuitBreaker


def _clock(start: float = 1000.0):
    """A controllable fake clock -- .now is the current time, call it to
    advance. Matches CreditCounter's own now_fn injection pattern."""
    state = {"now": start}

    def now_fn() -> float:
        return state["now"]

    def advance(seconds: float) -> None:
        state["now"] += seconds

    return now_fn, advance


def test_circuit_breaker_starts_closed_and_allows_calls() -> None:
    breaker = CircuitBreaker()
    assert breaker.state == "closed"
    assert breaker.allow() is True


def test_circuit_breaker_stays_closed_below_the_failure_threshold() -> None:
    breaker = CircuitBreaker(failure_threshold=3)
    breaker.record_failure()
    breaker.record_failure()
    assert breaker.state == "closed"
    assert breaker.allow() is True


def test_circuit_breaker_opens_after_threshold_consecutive_failures() -> None:
    breaker = CircuitBreaker(failure_threshold=3)
    breaker.record_failure()
    breaker.record_failure()
    breaker.record_failure()
    assert breaker.state == "open"
    assert breaker.allow() is False


def test_circuit_breaker_a_success_resets_the_consecutive_failure_count() -> None:
    breaker = CircuitBreaker(failure_threshold=3)
    breaker.record_failure()
    breaker.record_failure()
    breaker.record_success()
    breaker.record_failure()
    breaker.record_failure()
    # two more failures after the reset -- still below threshold again
    assert breaker.state == "closed"


def test_circuit_breaker_blocks_calls_while_open_and_cooldown_not_elapsed() -> None:
    now_fn, advance = _clock()
    breaker = CircuitBreaker(failure_threshold=1, cooldown_seconds=180.0, now_fn=now_fn)
    breaker.record_failure()
    assert breaker.state == "open"

    advance(60.0)
    assert breaker.allow() is False
    assert breaker.state == "open"


def test_circuit_breaker_allows_exactly_one_probe_after_cooldown_elapses() -> None:
    now_fn, advance = _clock()
    breaker = CircuitBreaker(failure_threshold=1, cooldown_seconds=180.0, now_fn=now_fn)
    breaker.record_failure()
    assert breaker.state == "open"

    advance(181.0)
    assert breaker.allow() is True
    assert breaker.state == "half_open"


def test_circuit_breaker_closes_on_a_successful_probe() -> None:
    now_fn, advance = _clock()
    breaker = CircuitBreaker(failure_threshold=1, cooldown_seconds=180.0, now_fn=now_fn)
    breaker.record_failure()
    advance(181.0)
    breaker.allow()  # transitions to half_open

    breaker.record_success()
    assert breaker.state == "closed"
    assert breaker.allow() is True


def test_circuit_breaker_reopens_with_a_doubled_cooldown_on_a_failed_probe() -> None:
    now_fn, advance = _clock()
    breaker = CircuitBreaker(failure_threshold=1, cooldown_seconds=180.0, now_fn=now_fn)
    breaker.record_failure()
    advance(181.0)
    breaker.allow()  # half_open

    breaker.record_failure()
    assert breaker.state == "open"

    advance(181.0)
    assert breaker.allow() is False  # old cooldown would have elapsed, doubled one hasn't

    advance(180.0)
    assert breaker.allow() is True  # now the doubled 360s cooldown has elapsed


def test_circuit_breaker_caps_the_doubled_cooldown() -> None:
    now_fn, advance = _clock()
    breaker = CircuitBreaker(failure_threshold=1, cooldown_seconds=1000.0, max_cooldown_seconds=1800.0, now_fn=now_fn)
    breaker.record_failure()
    advance(1001.0)
    breaker.allow()
    breaker.record_failure()  # would double to 2000s without the cap

    advance(1801.0)
    assert breaker.allow() is True  # capped at 1800s, not 2000s
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `python -m pytest app/backend/tests/test_circuit_breaker.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'app.backend.circuit_breaker'`

- [ ] **Step 3: Implement `CircuitBreaker`**

```python
"""CLOSED/OPEN/HALF-OPEN gate in front of a vendor fallback chain's own
per-key loop (FallbackOddsClient/FallbackOddsPapiClient, scheduler_wiring.py).

Not per-key -- CreditCounter.mark_unavailable/is_available already handles
"this one key is bad". This is for "the whole chain is bad right now": a
vendor fully exhausted (every key revoked or genuinely out of credits)
costs a full per-key pass through requests.get(timeout=10) calls (OddsPapi:
also its own 5s global throttle per attempt) on every single call, with no
memory that the last call already proved every key dead. Opening the
circuit after repeated full exhaustions turns that into "discover it once,
remember it for a cooldown window."

2026-11-02 design: docs/superpowers/specs/2026-11-02-background-job-safety-net-design.md

ponytail: HALF-OPEN allows every call through, not strictly exactly one --
avoiding a lock for the rare race of two concurrent calls landing exactly
on a cooldown expiry. Worst case under real concurrency is a few probe
attempts instead of one, still far cheaper than CLOSED's full fan-out;
revisit only if that race is observed to matter in practice."""

from __future__ import annotations

import time
from typing import Callable

DEFAULT_FAILURE_THRESHOLD = 3
DEFAULT_COOLDOWN_SECONDS = 180.0
DEFAULT_MAX_COOLDOWN_SECONDS = 1800.0


class CircuitBreaker:
    def __init__(
        self,
        failure_threshold: int = DEFAULT_FAILURE_THRESHOLD,
        cooldown_seconds: float = DEFAULT_COOLDOWN_SECONDS,
        max_cooldown_seconds: float = DEFAULT_MAX_COOLDOWN_SECONDS,
        now_fn: Callable[[], float] = time.monotonic,
    ) -> None:
        self._failure_threshold = failure_threshold
        self._base_cooldown = cooldown_seconds
        self._max_cooldown = max_cooldown_seconds
        self._now_fn = now_fn
        self._state = "closed"
        self._consecutive_failures = 0
        self._opened_at: float | None = None
        self._current_cooldown = cooldown_seconds

    @property
    def state(self) -> str:
        return self._state

    def allow(self) -> bool:
        """True if a real call should be attempted now. Called at the top
        of _try_each_client, before touching any key."""
        if self._state in ("closed", "half_open"):
            return True
        # state == "open"
        assert self._opened_at is not None
        if self._now_fn() - self._opened_at >= self._current_cooldown:
            self._state = "half_open"
            return True
        return False

    def record_success(self) -> None:
        """A real call (CLOSED's normal path, or HALF-OPEN's probe)
        returned a usable result."""
        self._state = "closed"
        self._consecutive_failures = 0
        self._current_cooldown = self._base_cooldown

    def record_failure(self) -> None:
        """A real call exhausted every configured key. From HALF-OPEN this
        is a failed probe -- reopen with a doubled (capped) cooldown rather
        than re-counting toward the threshold again."""
        if self._state == "half_open":
            self._current_cooldown = min(self._current_cooldown * 2, self._max_cooldown)
            self._state = "open"
            self._opened_at = self._now_fn()
            return
        self._consecutive_failures += 1
        if self._consecutive_failures >= self._failure_threshold:
            self._state = "open"
            self._opened_at = self._now_fn()
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `python -m pytest app/backend/tests/test_circuit_breaker.py -v`
Expected: PASS (9 passed)

- [ ] **Step 5: Commit**

```bash
git add app/backend/circuit_breaker.py app/backend/tests/test_circuit_breaker.py
git commit -m "feat: add CircuitBreaker state machine"
```

---

### Task 2: Process-level singleton breakers + wire into `FallbackOddsClient`

**Files:**
- Modify: `app/backend/scheduler_wiring.py`
- Test: `app/backend/tests/test_scheduler_wiring.py`

- [ ] **Step 1: Write the failing tests**

Check the existing `FallbackOddsClient` test construction pattern first:

Run: `grep -n "class TestFallbackOddsClient\|FallbackOddsClient(\[" app/backend/tests/test_scheduler_wiring.py | head -10`

Add tests using that same construction pattern (list of fake `PersistingOddsClient`-like objects with `.is_available()`, `.unavailable_reason()`, and a `.get_odds(...)` that raises or returns a value):

```python
def test_fallback_odds_client_short_circuits_when_breaker_is_open() -> None:
    """2026-11-02: when the circuit is OPEN, _try_each_client must not
    touch a single client -- that's the whole point (zero network calls,
    zero throttle wait) vs. today's full pass through every key."""
    from app.backend.circuit_breaker import CircuitBreaker
    from app.backend.scheduler_wiring import FallbackOddsClient

    breaker = CircuitBreaker(failure_threshold=1)
    breaker.record_failure()  # opens it
    client = MagicMock()
    fallback = FallbackOddsClient([client], circuit_breaker=breaker)

    result = fallback.get_odds(sport_key="soccer_epl")

    assert result is None
    client.get_odds.assert_not_called()


def test_fallback_odds_client_records_success_on_the_breaker() -> None:
    from app.backend.circuit_breaker import CircuitBreaker
    from app.backend.scheduler_wiring import FallbackOddsClient

    breaker = CircuitBreaker()
    client = MagicMock()
    client.is_available.return_value = True
    client.get_odds.return_value = ["some odds"]
    fallback = FallbackOddsClient([client], circuit_breaker=breaker)

    fallback.get_odds(sport_key="soccer_epl")

    assert breaker.state == "closed"


def test_fallback_odds_client_records_failure_on_full_exhaustion() -> None:
    from app.backend.circuit_breaker import CircuitBreaker
    from app.backend.scheduler_wiring import FallbackOddsClient

    breaker = CircuitBreaker(failure_threshold=3)
    client = MagicMock()
    client.is_available.return_value = True
    client.get_odds.return_value = None  # exhausted, same as today's existing exhaustion tests

    fallback = FallbackOddsClient([client], circuit_breaker=breaker)
    fallback.get_odds(sport_key="soccer_epl")

    assert breaker.state == "closed"  # 1 of 3
    fallback.get_odds(sport_key="soccer_epl")
    fallback.get_odds(sport_key="soccer_epl")
    assert breaker.state == "open"  # 3 of 3 -- tripped
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `python -m pytest app/backend/tests/test_scheduler_wiring.py -k "breaker" -v`
Expected: FAIL with `TypeError: FallbackOddsClient.__init__() got an unexpected keyword argument 'circuit_breaker'`

- [ ] **Step 3: Add the singleton accessor and wire the breaker into `FallbackOddsClient`**

In `app/backend/scheduler_wiring.py`, add near the other module-level singletons (alongside `_last_slack_alert_at` or similar):

```python
from app.backend.circuit_breaker import CircuitBreaker

# 2026-11-02: one breaker per vendor, held at module level so it survives
# across build_odds_client()/build_oddspapi_client() being called fresh on
# every pregenerate/EOD invocation -- a breaker created inside
# FallbackOddsClient.__init__ would lose all memory of an ongoing outage
# the instant the next call rebuilt the client, reproducing exactly the
# "re-discover the outage every time" problem this exists to fix.
_odds_api_circuit_breaker = CircuitBreaker()
_oddspapi_circuit_breaker = CircuitBreaker()
```

Modify `FallbackOddsClient.__init__` and `_try_each_client`:

```python
    def __init__(self, clients: list[PersistingOddsClient], circuit_breaker: CircuitBreaker | None = None) -> None:
        self._clients = clients
        self._circuit = circuit_breaker if circuit_breaker is not None else _odds_api_circuit_breaker

    def _try_each_client(self, call: Callable[[PersistingOddsClient], object], op_name: str):
        if not self._circuit.allow():
            LOGGER.info("FallbackOddsClient.%s: circuit open -- short-circuiting, no keys attempted.", op_name)
            return None
        # W254: stable sort by "currently in cooldown" -- available keys
        # keep their relative order and go first, cooling-down ones keep
        # theirs and go last, same "reorder, never hard-skip" discipline as
        # FootballDataClient._get_matches' own would_block() sort (W250):
        # a cooling-down key is still tried as a last resort if every other
        # key has also failed, in case the cooldown estimate was wrong.
        ordered = sorted(enumerate(self._clients), key=lambda pair: not pair[1].is_available())
        for i, client in ordered:
            if not client.is_available():
                LOGGER.info(
                    "FallbackOddsClient.%s: key #%d still cooling down (%s) -- trying anyway as a last resort.",
                    op_name, i + 1, client.unavailable_reason(),
                )
            try:
                result = call(client)
            except requests.RequestException as exc:
                LOGGER.warning("FallbackOddsClient.%s: key #%d failed (%s) -- trying next key.", op_name, i + 1, exc)
                continue
            if result is not None:
                self._circuit.record_success()
                return result
            LOGGER.info("FallbackOddsClient.%s: key #%d exhausted (local budget) -- trying next key.", op_name, i + 1)
        # W250: the per-key client's own "skipping call" warning fires on
        # every one of these attempts and, read alone, looks like the app
        # just gave up -- this is the one place that's actually true, so it's
        # the one place that says so, at WARNING (not the per-key loop's INFO)
        # so it survives a warning-level-only log filter.
        LOGGER.warning(
            "FallbackOddsClient.%s: all %d configured key(s) exhausted or failed -- keeping last-known odds.",
            op_name, len(self._clients),
        )
        self._circuit.record_failure()
        _alert_all_keys_exhausted("The Odds API", op_name, len(self._clients))
        return None
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `python -m pytest app/backend/tests/test_scheduler_wiring.py -k "breaker" -v`
Expected: PASS (3 passed)

- [ ] **Step 5: Run the full scheduler_wiring test file to confirm no regressions**

Run: `python -m pytest app/backend/tests/test_scheduler_wiring.py -v`
Expected: all tests PASS — the existing `FallbackOddsClient` exhaustion/fallback tests (W254, cooldown ordering, etc.) construct the client without passing `circuit_breaker`, so they exercise the module-level singleton; confirm none of them assert on `_odds_api_circuit_breaker`'s state leaking between tests (if any do, that test needs to reset it — see Step 6)

- [ ] **Step 6: Guard against cross-test state leakage on the module-level singleton**

Since `_odds_api_circuit_breaker` is now a shared module-level instance, a test that trips it (3 consecutive exhaustions) would leave it OPEN for every subsequent test in the same process that doesn't pass its own `circuit_breaker=`. Add an autouse fixture at the top of `app/backend/tests/test_scheduler_wiring.py` (check first whether one already exists for a similar reason — `grep -n "autouse=True" app/backend/tests/test_scheduler_wiring.py`):

```python
@pytest.fixture(autouse=True)
def _reset_circuit_breakers():
    """The module-level odds/oddspapi circuit breakers are process-global
    singletons by design (2026-11-02) -- reset them before each test so one
    test tripping a breaker can't leave it OPEN for an unrelated later test
    in the same run."""
    import app.backend.scheduler_wiring as scheduler_wiring_module
    from app.backend.circuit_breaker import CircuitBreaker
    scheduler_wiring_module._odds_api_circuit_breaker = CircuitBreaker()
    scheduler_wiring_module._oddspapi_circuit_breaker = CircuitBreaker()
    yield
```

- [ ] **Step 7: Run the full scheduler_wiring test file again**

Run: `python -m pytest app/backend/tests/test_scheduler_wiring.py -v`
Expected: all tests PASS, order-independent (run with `-p no:randomly` and default order both green if `pytest-randomly` is installed — check with `pip show pytest-randomly`)

- [ ] **Step 8: Commit**

```bash
git add app/backend/scheduler_wiring.py app/backend/tests/test_scheduler_wiring.py
git commit -m "feat: gate FallbackOddsClient with a process-level circuit breaker"
```

---

### Task 3: Wire the breaker into `FallbackOddsPapiClient`

**Files:**
- Modify: `app/backend/scheduler_wiring.py`
- Test: `app/backend/tests/test_scheduler_wiring.py`

Identical shape to Task 2, applied to `FallbackOddsPapiClient`/`_oddspapi_circuit_breaker`.

- [ ] **Step 1: Write the failing tests**

Mirror Task 2 Step 1's three tests exactly, substituting `FallbackOddsPapiClient`, `get_fixtures`/`get_corners_odds`, and `_oddspapi_circuit_breaker`:

```python
def test_fallback_oddspapi_client_short_circuits_when_breaker_is_open() -> None:
    from app.backend.circuit_breaker import CircuitBreaker
    from app.backend.scheduler_wiring import FallbackOddsPapiClient

    breaker = CircuitBreaker(failure_threshold=1)
    breaker.record_failure()
    client = MagicMock()
    fallback = FallbackOddsPapiClient([client], circuit_breaker=breaker)

    result = fallback.get_corners_odds(fixture_id="abc123")

    assert result is None
    client.get_corners_odds.assert_not_called()


def test_fallback_oddspapi_client_records_success_on_the_breaker() -> None:
    from app.backend.circuit_breaker import CircuitBreaker
    from app.backend.scheduler_wiring import FallbackOddsPapiClient

    breaker = CircuitBreaker()
    client = MagicMock()
    client.is_available.return_value = True
    client.get_corners_odds.return_value = {"over_9.5": 1.9, "under_9.5": 1.9}
    fallback = FallbackOddsPapiClient([client], circuit_breaker=breaker)

    fallback.get_corners_odds(fixture_id="abc123")

    assert breaker.state == "closed"


def test_fallback_oddspapi_client_records_failure_on_full_exhaustion() -> None:
    from app.backend.circuit_breaker import CircuitBreaker
    from app.backend.scheduler_wiring import FallbackOddsPapiClient

    breaker = CircuitBreaker(failure_threshold=1)
    client = MagicMock()
    client.is_available.return_value = True
    client.get_corners_odds.return_value = None

    fallback = FallbackOddsPapiClient([client], circuit_breaker=breaker)
    fallback.get_corners_odds(fixture_id="abc123")

    assert breaker.state == "open"
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `python -m pytest app/backend/tests/test_scheduler_wiring.py -k "oddspapi_client_short_circuits or oddspapi_client_records" -v`
Expected: FAIL with `TypeError`

- [ ] **Step 3: Wire the breaker into `FallbackOddsPapiClient`**

Same pattern as Task 2 Step 3, applied to `FallbackOddsPapiClient.__init__`/`_try_each_client`, using `_oddspapi_circuit_breaker` as the default and calling `self._circuit.record_success()`/`self._circuit.record_failure()` at the same two points (successful result / full exhaustion, right before the existing `_alert_all_keys_exhausted("OddsPapi", ...)` call).

- [ ] **Step 4: Run the tests to verify they pass**

Run: `python -m pytest app/backend/tests/test_scheduler_wiring.py -k "oddspapi_client_short_circuits or oddspapi_client_records" -v`
Expected: PASS (3 passed)

- [ ] **Step 5: Extend the autouse fixture's reset to cover both breakers**

Already done in Task 2 Step 6 (`_reset_circuit_breakers` resets both module-level singletons) — confirm no further change needed here.

- [ ] **Step 6: Run the full scheduler_wiring test file**

Run: `python -m pytest app/backend/tests/test_scheduler_wiring.py -v`
Expected: all tests PASS

- [ ] **Step 7: Commit**

```bash
git add app/backend/scheduler_wiring.py app/backend/tests/test_scheduler_wiring.py
git commit -m "feat: gate FallbackOddsPapiClient with a process-level circuit breaker"
```

---

### Task 4: Expose breaker state for the future `/ops` dashboard

**Files:**
- Modify: `app/backend/scheduler_wiring.py`
- Test: `app/backend/tests/test_scheduler_wiring.py`

Plan 3 (`/ops` backend endpoints) needs a way to read "what's the current state of each vendor's circuit" without reaching into module internals directly.

- [ ] **Step 1: Write the failing test**

```python
def test_get_circuit_breaker_status_reports_both_vendors() -> None:
    from app.backend.scheduler_wiring import get_circuit_breaker_status

    status = get_circuit_breaker_status()

    assert set(status.keys()) == {"the_odds_api", "oddspapi"}
    assert status["the_odds_api"]["state"] == "closed"
    assert status["oddspapi"]["state"] == "closed"
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `python -m pytest app/backend/tests/test_scheduler_wiring.py -k get_circuit_breaker_status -v`
Expected: FAIL with `ImportError`

- [ ] **Step 3: Implement `get_circuit_breaker_status`**

Add to `app/backend/scheduler_wiring.py`, near the two singleton breaker instances from Task 2:

```python
def get_circuit_breaker_status() -> dict[str, dict[str, str]]:
    """Read-only snapshot of both vendor circuit breakers' current state --
    consumed by the /ops dashboard's failures tab (2026-11-02 design)."""
    return {
        "the_odds_api": {"state": _odds_api_circuit_breaker.state},
        "oddspapi": {"state": _oddspapi_circuit_breaker.state},
    }
```

- [ ] **Step 4: Run the test to verify it passes**

Run: `python -m pytest app/backend/tests/test_scheduler_wiring.py -k get_circuit_breaker_status -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add app/backend/scheduler_wiring.py app/backend/tests/test_scheduler_wiring.py
git commit -m "feat: expose get_circuit_breaker_status for the ops dashboard"
```

---

### Task 5: Full regression pass

**Files:** none (verification only)

- [ ] **Step 1: Run the full backend test suite**

Run: `python -m pytest app/backend/tests -q`
Expected: no new failures beyond the same 8 pre-existing, unrelated ones documented in `documents/bugs.md`'s BUG-092 notes

- [ ] **Step 2: Run the odds-vendor-specific files for a clean summary**

Run: `python -m pytest app/backend/tests/test_circuit_breaker.py app/backend/tests/test_scheduler_wiring.py app/backend/tests/test_oddspapi_credit_burn.py app/backend/tests/test_odds_api_client.py -v`
Expected: all PASS

---

## Self-review notes

- **Spec coverage**: implements §5 of the design spec in full (the `CircuitBreaker` state machine, wired into both `FallbackOddsClient` and `FallbackOddsPapiClient`), plus the breaker-status read needed by §7's failures tab.
- **The one real design decision this plan makes beyond a literal reading of the spec**: the breaker must be a process-level singleton, not constructed inside `FallbackOddsClient.__init__`, because `build_odds_client()` is called fresh on every pregenerate/EOD invocation (confirmed by reading `main.py`'s `_pregenerate_recommendations` and `scheduler_wiring.py`'s own job registration) — an instance-scoped breaker would silently fail to persist across exactly the multi-call scenario (several separate admin `force=true` calls) that the 2026-10-31 incident actually was.
- **Type consistency**: `CircuitBreaker.allow()`/`record_success()`/`record_failure()` signatures are used identically in both `FallbackOddsClient` and `FallbackOddsPapiClient`.
