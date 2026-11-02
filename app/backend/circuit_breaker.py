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
