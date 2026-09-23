"""W20: backend test strategy -- actively enforces (not just documents)
that the app/backend test suite never depends on a real network call or a
running Ollama daemon. run_agent(), FootballDataClient, and OddsAPIClient
are all mocked at their boundaries throughout this suite (W02/W05/W07's
own test files establish that convention); this fixture makes a violation
fail loudly with a clear message instead of silently hanging or flaking in
CI, rather than relying on every future test remembering to mock.

A test that must make a real call opts in via @pytest.mark.live --
registered below so pytest doesn't warn about an unknown marker. W41
(2026-09-22): the marker itself only controlled the network-block bypass
above -- nothing actually excluded a live-marked test from a normal `pytest`
run (confirmed live: test_scheduler_live_fire.py's real-wall-clock test ran
by default with no opt-in flag, the exact opposite of the "excluded from
the default fast run" contract W41 itself documents). Skipped below unless
RUN_LIVE_TESTS is truthy, same opt-in-by-env-var shape as this project's
other real-call gates (e.g. scripts that check for an API key before
calling out) -- run explicitly via `RUN_LIVE_TESTS=1 pytest -m live`.
"""

from __future__ import annotations

import os
import socket

import pytest


def pytest_configure(config: pytest.Config) -> None:
    config.addinivalue_line(
        "markers", "live: allow real network/wall-clock calls for this test (opt-in, skipped by default -- see module docstring)"
    )


def pytest_collection_modifyitems(config: pytest.Config, items: list[pytest.Item]) -> None:
    if os.environ.get("RUN_LIVE_TESTS"):
        return
    skip_live = pytest.mark.skip(reason="live test -- set RUN_LIVE_TESTS=1 to run")
    for item in items:
        if item.get_closest_marker("live"):
            item.add_marker(skip_live)


def _blocked(*args, **kwargs):
    raise RuntimeError(
        "Real network call attempted during an app/backend test. This suite must not "
        "depend on live network/Ollama (W20) -- mock the client/run_agent boundary "
        "instead, or mark the test @pytest.mark.live if a real call is genuinely required."
    )


@pytest.fixture(autouse=True)
def _block_real_network(request: pytest.FixtureRequest, monkeypatch: pytest.MonkeyPatch):
    if request.node.get_closest_marker("live"):
        yield
        return
    monkeypatch.setattr(socket.socket, "connect", _blocked)
    monkeypatch.setattr(socket.socket, "connect_ex", _blocked)
    yield
