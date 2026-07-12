"""W12: bet-logging request models. The from-recommendation path structurally
cannot accept odds/home_team/away_team/date as separate client-supplied
fields -- they're always derived from the recommendation snapshot itself, so
there's nothing for a client to "edit" except stake. The manual path requires
a resolved match_id, not free-typed team names."""

from __future__ import annotations

from pathlib import Path
import sys

import pytest
from pydantic import ValidationError

sys.path.append(str(Path(__file__).resolve().parents[3]))

from app.backend.bets import BetFromRecommendationRequest, BetManualRequest, resolve_from_recommendation

_RECOMMENDATION = {
    "match": {"home": "Arsenal", "away": "Everton", "date": "2026-08-22", "league": "E0"},
    "overall": "direct_bet",
    "markets": [
        {"market": "result_3way", "selection": "home", "recommendation_type": "direct_bet", "current_odds": 2.1},
        {"market": "btts", "selection": "yes", "recommendation_type": "conditional", "current_odds": 1.9},
    ],
    "explanation": "test",
    "confidence": "medium",
    "limitations": [],
    "prediction_basis": "team_history_and_market",
}


def test_from_recommendation_request_has_no_odds_or_team_fields():
    """Structural proof there's nothing to 'edit' but stake -- these fields
    simply don't exist on the request model."""
    fields = set(BetFromRecommendationRequest.model_fields.keys())
    assert "odds" not in fields
    assert "home_team" not in fields
    assert "away_team" not in fields
    assert "date" not in fields
    assert fields == {"match_id", "recommendation", "market", "selection", "stake"}


def test_resolve_from_recommendation_derives_all_fields_from_the_snapshot():
    request = BetFromRecommendationRequest(
        match_id="m1", recommendation=_RECOMMENDATION, market="result_3way", selection="home", stake=10.0,
    )
    resolved = resolve_from_recommendation(request)
    assert resolved["home_team"] == "Arsenal"
    assert resolved["away_team"] == "Everton"
    assert resolved["date"] == "2026-08-22"
    assert resolved["odds"] == 2.1
    assert resolved["stake"] == 10.0


def test_resolve_from_recommendation_raises_when_market_selection_not_in_snapshot():
    request = BetFromRecommendationRequest(
        match_id="m1", recommendation=_RECOMMENDATION, market="result_3way", selection="away", stake=10.0,
    )
    with pytest.raises(ValueError, match="not found"):
        resolve_from_recommendation(request)


def test_resolve_from_recommendation_raises_when_odds_are_null():
    rec = {**_RECOMMENDATION, "markets": [{"market": "btts", "selection": "yes", "current_odds": None}]}
    request = BetFromRecommendationRequest(
        match_id="m1", recommendation=rec, market="btts", selection="yes", stake=10.0,
    )
    with pytest.raises(ValueError, match="no current_odds"):
        resolve_from_recommendation(request)


def test_manual_request_requires_non_empty_match_id():
    with pytest.raises(ValidationError):
        BetManualRequest(
            match_id="", date="2026-08-22", home_team="Arsenal", away_team="Everton",
            market="btts", selection="yes", odds=1.9, stake=5.0,
        )


def test_manual_request_accepts_a_resolved_fixture():
    request = BetManualRequest(
        match_id="m1", date="2026-08-22", home_team="Arsenal", away_team="Everton",
        market="btts", selection="yes", odds=1.9, stake=5.0,
    )
    assert request.match_id == "m1"
