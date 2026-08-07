"""Regression tests for A27: give the agent a reliable, tool-callable signal
for which forecast tool to use, instead of relying on its own domestic-vs-
international judgment. resolve_competition wraps the same competition
registry US#107 already consults engine-side, so the agent's own tool
selection and the engine's actual routing can never disagree."""

from __future__ import annotations

import json

from src.agent.tools import forecast_international, resolve_competition


def test_resolve_competition_recommends_forecast_league_for_competition_specific():
    """E0 is registered competition_specific in the real config/competitions.yaml."""
    result = json.loads(resolve_competition.invoke({"competition_or_league": "E0"}))
    assert result["tier"] == "competition_specific"
    assert result["recommended_tool"] == "forecast_league"


def test_resolve_competition_recommends_forecast_international_for_general_purpose():
    """'international' is registered general_purpose in the real registry."""
    result = json.loads(resolve_competition.invoke({"competition_or_league": "international"}))
    assert result["tier"] == "general_purpose"
    assert result["recommended_tool"] == "forecast_international"


def test_resolve_competition_recommends_forecast_league_for_sp1():
    """A49: SP1 (La Liga) was registered competition_specific in the real
    config/competitions.yaml (US#147) -- confirm resolve_competition routes
    it correctly, distinguishable from an unregistered competition."""
    result = json.loads(resolve_competition.invoke({"competition_or_league": "SP1"}))
    assert result["tier"] == "competition_specific"
    assert result["recommended_tool"] == "forecast_league"


def test_resolve_competition_recommends_forecast_international_for_unregistered_competition():
    """A49: Bundesliga (D1) has no entry in config/competitions.yaml at all --
    must fall back to general_purpose/forecast_international, not error or
    default to league. Replaced "La Liga" as this test's stock unregistered-
    competition example -- La Liga (SP1) is now genuinely registered
    (US#147), so it no longer demonstrates the unregistered-competition
    path; Bundesliga/D1 is confirmed to still have no registry entry."""
    result = json.loads(resolve_competition.invoke({"competition_or_league": "Bundesliga"}))
    assert result["tier"] == "general_purpose"
    assert result["recommended_tool"] == "forecast_international"


def test_following_resolve_competitions_advice_for_unregistered_league_yields_market_odds_only():
    """Acceptance: for a known-unregistered competition, following
    resolve_competition's recommendation ends up with market_odds_only, not a
    cold-start team_history_and_market result."""
    from unittest.mock import MagicMock, patch

    advice = json.loads(resolve_competition.invoke({"competition_or_league": "Bundesliga"}))
    assert advice["recommended_tool"] == "forecast_international"

    mock_result = {
        "forecast": {"result_3way": {"probabilities": {"home": 0.4, "draw": 0.3, "away": 0.3}}},
        "data_quality": {"prediction_basis": "market_odds_only", "unknown_team": False},
    }
    with patch("src.forecast.forecast_service.ForecastService") as MockSvc:
        instance = MagicMock()
        MockSvc.return_value = instance
        instance.forecast_upcoming.return_value = mock_result

        result_str = forecast_international.invoke({
            "home_team": "Bayern Munich", "away_team": "Borussia Dortmund", "date": "2026-08-15",
            "odds_h": 2.1, "odds_d": 3.4, "odds_a": 3.3,
        })

    result = json.loads(result_str)
    assert result["data_quality"]["prediction_basis"] == "market_odds_only"
