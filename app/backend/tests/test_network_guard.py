"""W20: proves conftest.py's network-blocking fixture actually blocks a
real connection attempt (not just documents an intention), and that a
test marked @pytest.mark.live is exempted."""

from __future__ import annotations

import socket

import pytest


def test_a_real_connection_attempt_is_blocked_by_default() -> None:
    with pytest.raises(RuntimeError, match="Real network call attempted"):
        socket.socket(socket.AF_INET, socket.SOCK_STREAM).connect(("127.0.0.1", 80))


@pytest.mark.live
def test_live_marker_opts_out_of_the_network_guard() -> None:
    """A real connect() attempt against a closed local port (no internet
    needed -- just loopback) raises a real connection error, proving the
    guard's patch was never applied for this @pytest.mark.live test --
    not the guard's synthetic RuntimeError."""
    with pytest.raises(OSError) as exc_info:
        socket.socket(socket.AF_INET, socket.SOCK_STREAM).connect(("127.0.0.1", 1))
    assert "Real network call attempted" not in str(exc_info.value)
