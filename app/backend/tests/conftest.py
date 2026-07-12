"""W20: backend test strategy -- actively enforces (not just documents)
that the app/backend test suite never depends on a real network call or a
running Ollama daemon. run_agent(), FootballDataClient, and OddsAPIClient
are all mocked at their boundaries throughout this suite (W02/W05/W07's
own test files establish that convention); this fixture makes a violation
fail loudly with a clear message instead of silently hanging or flaking in
CI, rather than relying on every future test remembering to mock.

A test that must make a real call (there are none today) opts in via
@pytest.mark.live -- registered below so pytest doesn't warn about an
unknown marker.
"""

from __future__ import annotations

import socket

import pytest


def pytest_configure(config: pytest.Config) -> None:
    config.addinivalue_line(
        "markers", "live: allow real network calls for this test (opt-in; none of this suite's tests need it today)"
    )


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
