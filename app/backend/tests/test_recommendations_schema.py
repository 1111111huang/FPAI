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
    "cold_start_risk": False,
    "feature_completeness": 0.97,
    "unknown_team": False,
}


def test_valid_recommendation_passes_through_unchanged():
    result = validate_and_degrade(_VALID_RAW, "Arsenal", "Everton")
    assert result.overall == "direct_bet"
    assert len(result.markets) == 1
    assert result.markets[0].value_edge == 0.07
    assert result.invalid_market_count == 0


def test_malformed_market_is_omitted_not_crashed():
    bad_market = {**_VALID_MARKET, "value_edge": "high"}
    good_market = {**_VALID_MARKET, "market": "btts", "selection": "yes"}
    raw = {**_VALID_RAW, "markets": [good_market, bad_market]}

    result = validate_and_degrade(raw, "Arsenal", "Everton")

    assert len(result.markets) == 1
    assert result.markets[0].market == "btts"
    assert result.invalid_market_count == 1
    assert any("1 market" in note for note in result.limitations)


def test_non_canonical_market_name_is_dropped_like_any_other_malformed_market():
    """Observed live in the sandbox cache: the same result_3way market
    rendered as "1X2" for one real fixture. A defense-in-depth backstop for
    a recommendation that reaches this layer without having gone through
    extract_recommendation's own (now equivalent) check -- e.g. an
    already-cached row from before that check shipped."""
    bad_market = {**_VALID_MARKET, "market": "1X2"}
    good_market = {**_VALID_MARKET, "market": "btts", "selection": "yes"}
    raw = {**_VALID_RAW, "markets": [good_market, bad_market]}

    result = validate_and_degrade(raw, "Arsenal", "Everton")

    assert len(result.markets) == 1
    assert result.markets[0].market == "btts"
    assert result.invalid_market_count == 1


def test_all_markets_malformed_returns_empty_markets_not_an_exception():
    bad_market = {**_VALID_MARKET, "confidence_typo": "oops", "value_edge": "nonsense"}
    raw = {**_VALID_RAW, "markets": [bad_market]}

    result = validate_and_degrade(raw, "Arsenal", "Everton")

    assert result.markets == []
    assert result.invalid_market_count == 1
    assert result.overall == "direct_bet"  # top-level fields untouched


def test_missing_top_level_fields_default_safely_instead_of_raising():
    """Belt-and-suspenders: even a badly malformed top-level payload (e.g.
    from a pre-A28 cached recommendation) must not crash the app layer."""
    result = validate_and_degrade({"markets": []}, "Arsenal", "Everton")
    assert result.overall == "insufficient_data"
    assert result.markets == []


def test_cold_start_risk_and_unknown_team_pass_through():
    """W15: these fields must reach the app response, since the UI treats
    cold_start_risk as a first-class trust signal regardless of prediction_basis."""
    raw = {**_VALID_RAW, "cold_start_risk": True, "feature_completeness": 0.41, "unknown_team": True}
    result = validate_and_degrade(raw, "Arsenal", "Everton")
    assert result.cold_start_risk is True
    assert result.feature_completeness == 0.41
    assert result.unknown_team is True


def test_missing_w15_fields_default_safely_for_pre_w15_cached_data():
    """A recommendation cached before W15 shipped won't have these keys at
    all -- must default, not raise."""
    raw = {k: v for k, v in _VALID_RAW.items() if k not in ("cold_start_risk", "feature_completeness", "unknown_team")}
    result = validate_and_degrade(raw, "Arsenal", "Everton")
    assert result.cold_start_risk is False
    assert result.feature_completeness is None
    assert result.unknown_team is False


def test_agent_match_mismatch_is_degraded_to_insufficient_data():
    """BUG-023/024: the agent hallucinated a "Manchester City vs Liverpool"
    analysis for a real Brentford vs Wolverhampton request -- confirmed live
    in a sandbox precompute batch. Must be caught and discarded, not served
    to the frontend as if it were a real analysis of the requested match."""
    raw = {
        **_VALID_RAW,
        "match": {"home_team": "Manchester City", "away_team": "Liverpool"},
    }

    result = validate_and_degrade(raw, "Brentford", "Wolverhampton")

    assert result.overall == "insufficient_data"
    assert result.markets == []
    assert result.invalid_market_count == 1
    assert any("Manchester City v Liverpool" in note for note in result.limitations)
    assert any("Brentford v Wolverhampton" in note for note in result.limitations)


def test_home_away_swap_alone_is_not_a_mismatch():
    """A plain home/away swap (observed live: Sunderland v Brighton reported
    back as Brighton v Sunderland) is not the hallucination bug -- only a
    genuinely different pair of clubs should be discarded."""
    raw = {**_VALID_RAW, "match": {"home": "Everton", "away": "Arsenal"}}

    result = validate_and_degrade(raw, "Arsenal", "Everton")

    assert result.overall == "direct_bet"
    assert len(result.markets) == 1


def test_omitting_home_away_skips_the_mismatch_check_but_still_degrades_malformed_markets():
    """BUG-028: GET /api/recommendations/{match_id} has no ground-truth
    fixture to compare against (only match_id/date), so it calls
    validate_and_degrade(raw) with neither -- the match-mismatch check must
    be skipped (not crash on missing args), while the per-market validation
    that endpoint actually needs still runs."""
    bad_market = {**_VALID_MARKET, "market": "1X2"}
    good_market = {**_VALID_MARKET, "market": "btts", "selection": "yes"}
    raw = {**_VALID_RAW, "match": {"home": "Manchester City", "away": "Liverpool"}, "markets": [good_market, bad_market]}

    result = validate_and_degrade(raw)

    assert result.overall == "direct_bet"  # not degraded to insufficient_data -- no mismatch check ran
    assert len(result.markets) == 1
    assert result.markets[0].market == "btts"
    assert result.invalid_market_count == 1


def test_missing_match_field_is_not_treated_as_a_mismatch():
    """Some raw payloads omit `match` entirely (e.g. pre-existing malformed
    top-level payloads) -- that's a separate, already-handled degradation
    path, not this mismatch check's concern."""
    result = validate_and_degrade({"markets": [_VALID_MARKET]}, "Arsenal", "Everton")
    assert result.overall == "insufficient_data"  # from the existing "no overall" default
    assert len(result.markets) == 1  # markets themselves are untouched by the mismatch check
