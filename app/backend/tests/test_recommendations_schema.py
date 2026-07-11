"""W02: app-owned Pydantic validation layer for MatchRecommendation, wholly
independent of the agent's own extract_recommendation (A28) -- graceful
degradation means a malformed market is flagged/omitted, not a crashed
request, regardless of whether the agent's own validation already caught it
upstream."""

from __future__ import annotations

from pathlib import Path
import sys

sys.path.append(str(Path(__file__).resolve().parents[3]))

from app.backend.recommendations import validate_and_degrade

_VALID_MARKET = {
    "market": "result_3way",
    "selection": "home",
    "recommendation_type": "direct_bet",
    "current_odds": 2.1,
    "min_odds": 1.8,
    "ml_probability": 0.55,
    "implied_probability": 0.48,
    "value_edge": 0.07,
}

_VALID_RAW = {
    "match": {"home": "Arsenal", "away": "Everton", "date": "2026-08-22", "league": "E0"},
    "overall": "direct_bet",
    "markets": [_VALID_MARKET],
    "explanation": "Value found on the home win.",
    "confidence": "medium",
    "limitations": [],
    "prediction_basis": "team_history_and_market",
}


def test_valid_recommendation_passes_through_unchanged():
    result = validate_and_degrade(_VALID_RAW)
    assert result.overall == "direct_bet"
    assert len(result.markets) == 1
    assert result.markets[0].value_edge == 0.07
    assert result.invalid_market_count == 0


def test_malformed_market_is_omitted_not_crashed():
    bad_market = {**_VALID_MARKET, "value_edge": "high"}
    good_market = {**_VALID_MARKET, "market": "btts", "selection": "yes"}
    raw = {**_VALID_RAW, "markets": [good_market, bad_market]}

    result = validate_and_degrade(raw)

    assert len(result.markets) == 1
    assert result.markets[0].market == "btts"
    assert result.invalid_market_count == 1
    assert any("1 market" in note for note in result.limitations)


def test_all_markets_malformed_returns_empty_markets_not_an_exception():
    bad_market = {**_VALID_MARKET, "confidence_typo": "oops", "value_edge": "nonsense"}
    raw = {**_VALID_RAW, "markets": [bad_market]}

    result = validate_and_degrade(raw)

    assert result.markets == []
    assert result.invalid_market_count == 1
    assert result.overall == "direct_bet"  # top-level fields untouched


def test_missing_top_level_fields_default_safely_instead_of_raising():
    """Belt-and-suspenders: even a badly malformed top-level payload (e.g.
    from a pre-A28 cached recommendation) must not crash the app layer."""
    result = validate_and_degrade({"markets": []})
    assert result.overall == "insufficient_data"
    assert result.markets == []
