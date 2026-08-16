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


def test_swe_is_in_the_allowlist():
    """W56: Sweden's Allsvenskan is fully built engine-side (data, trained
    models, config/competitions.yaml registration) -- the allowlist is the
    only remaining reason it doesn't route through run_agent() correctly."""
    assert "SWE" in COMPETITION_ALLOWLIST


def test_gate_league_keeps_swe():
    assert gate_league("SWE") == "SWE"


def test_sp1_is_in_the_allowlist():
    """W75: La Liga is fully built engine-side (data, config/competitions.yaml
    registration, US#143-147) -- the allowlist is the only remaining reason
    it doesn't route through run_agent() correctly."""
    assert "SP1" in COMPETITION_ALLOWLIST


def test_gate_league_keeps_sp1():
    assert gate_league("SP1") == "SP1"


def test_i1_d1_f1_are_in_the_allowlist():
    """W135: Serie A/Bundesliga/Ligue 1 are fully built engine-side (data,
    config/competitions.yaml registration, US#161-171) -- the allowlist is
    the only remaining reason they don't route through run_agent() correctly."""
    assert {"I1", "D1", "F1"} <= COMPETITION_ALLOWLIST


def test_gate_league_keeps_i1_d1_f1():
    assert gate_league("I1") == "I1"
    assert gate_league("D1") == "D1"
    assert gate_league("F1") == "F1"


def test_request_to_match_info_includes_league_for_swe():
    request = RecommendationRequest(home_team="Malmo FF", away_team="AIK", date="2026-07-25", league="SWE")
    match_info = request.to_match_info()
    assert match_info["league"] == "SWE"


def test_gate_league_keeps_allowlisted_competition():
    assert gate_league("E0") == "E0"


def test_gate_league_omits_non_allowlisted_competition():
    # "La Liga" (the free-text name) stays gated out even after SP1 (the
    # code) joined the allowlist (W75) -- gate_league matches by code, not
    # name, so this is still a genuinely correct "not allowlisted" case, not
    # a stale example (see the agent_user_stories.md Phase 15 note on the
    # same free-text-vs-code distinction for resolve_competition).
    assert gate_league("La Liga") is None
    # N1 (Eredivisie) is a still-genuinely-unregistered competition code, used
    # here in D1's old place now that I1/D1/F1 themselves are allowlisted
    # (W135) -- D1 (Bundesliga) is retired as the stock example the same way
    # W75 retired "La Liga" for SP1's own registration.
    assert gate_league("N1") is None
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
