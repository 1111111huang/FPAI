"""W03: the app decides the 'league' value passed to run_agent() itself,
via a hardcoded allowlist independent of the live competition registry --
defense in depth, so a registry misconfiguration or a not-yet-shipped
US#107/A27 doesn't silently cascade into the app's own routing decision."""

from __future__ import annotations

from pathlib import Path
import sys

sys.path.append(str(Path(__file__).resolve().parents[3]))

from app.backend.match_info import COMPETITION_ALLOWLIST, gate_league
from app.backend.recommendations import RecommendationRequest


def test_e0_is_in_the_allowlist():
    assert "E0" in COMPETITION_ALLOWLIST


def test_gate_league_keeps_allowlisted_competition():
    assert gate_league("E0") == "E0"


def test_gate_league_omits_non_allowlisted_competition():
    assert gate_league("La Liga") is None
    assert gate_league("SP1") is None
    assert gate_league("Champions League") is None


def test_gate_league_omits_none():
    assert gate_league(None) is None


def test_request_to_match_info_includes_league_for_e0():
    request = RecommendationRequest(home_team="Arsenal", away_team="Everton", date="2026-08-22", league="E0")
    match_info = request.to_match_info()
    assert match_info["league"] == "E0"


def test_request_to_match_info_omits_league_for_non_epl_fixture():
    """Acceptance: a non-EPL fixture's constructed match_info has no league key."""
    request = RecommendationRequest(
        home_team="Real Madrid", away_team="Barcelona", date="2026-08-22", league="La Liga",
    )
    match_info = request.to_match_info()
    assert "league" not in match_info


def test_request_to_match_info_omits_league_when_not_provided():
    request = RecommendationRequest(home_team="Team A", away_team="Team B", date="2026-08-22")
    match_info = request.to_match_info()
    assert "league" not in match_info
