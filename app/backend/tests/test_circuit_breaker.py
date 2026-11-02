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
