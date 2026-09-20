"""W02: app-owned Pydantic validation layer for MatchRecommendation, wholly
independent of the agent's own extract_recommendation (A28) -- graceful
degradation means a malformed market is flagged/omitted, not a crashed
request, regardless of whether the agent's own validation already caught it
upstream."""

from __future__ import annotations

from pathlib import Path
import sys

sys.path.append(str(Path(__file__).resolve().parents[3]))

from app.backend.recommendations import RecommendationPickOut, validate_and_degrade

_VALID_CANDIDATE = {
    "market": "result_3way",
    "selection": "home",
    "recommendation_type": "direct_bet",
    "current_odds": 2.1,
    "min_odds": 1.8,
    "ml_probability": 0.55,
    "implied_probability": 0.48,
    "value_edge": 0.07,
    "composite_score": 0.6,
    "reason": "Clears the edge floor at a realistic price.",
}

_VALID_RAW = {
    "match": {"home": "Arsenal", "away": "Everton", "date": "2026-08-22", "league": "E0"},
    "overall": "direct_bet",
    "candidates": [_VALID_CANDIDATE],
    "recommendation_pick": {"market": "result_3way", "selection": "home"},
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
    assert len(result.candidates) == 1
    assert result.candidates[0].value_edge == 0.07
    assert result.invalid_market_count == 0


def test_malformed_market_is_omitted_not_crashed():
    bad_market = {**_VALID_CANDIDATE, "value_edge": "high"}
    good_market = {**_VALID_CANDIDATE, "market": "btts", "selection": "yes"}
    raw = {**_VALID_RAW, "candidates": [good_market, bad_market]}

    result = validate_and_degrade(raw, "Arsenal", "Everton")

    assert len(result.candidates) == 1
    assert result.candidates[0].market == "btts"
    assert result.invalid_market_count == 1
    assert any("1 market" in note for note in result.limitations)


def test_non_canonical_market_name_is_dropped_like_any_other_malformed_market():
    """Observed live in the sandbox cache: the same result_3way market
    rendered as "1X2" for one real fixture. A defense-in-depth backstop for
    a recommendation that reaches this layer without having gone through
    extract_recommendation's own (now equivalent) check -- e.g. an
    already-cached row from before that check shipped."""
    bad_market = {**_VALID_CANDIDATE, "market": "1X2"}
    good_market = {**_VALID_CANDIDATE, "market": "btts", "selection": "yes"}
    raw = {**_VALID_RAW, "candidates": [good_market, bad_market]}

    result = validate_and_degrade(raw, "Arsenal", "Everton")

    assert len(result.candidates) == 1
    assert result.candidates[0].market == "btts"
    assert result.invalid_market_count == 1


def test_all_markets_malformed_returns_empty_markets_not_an_exception():
    bad_market = {**_VALID_CANDIDATE, "confidence_typo": "oops", "value_edge": "nonsense"}
    raw = {**_VALID_RAW, "candidates": [bad_market]}

    result = validate_and_degrade(raw, "Arsenal", "Everton")

    assert result.candidates == []
    assert result.invalid_market_count == 1
    assert result.overall == "no_bet"  # top-level fields default-capped, no resolvable pick


def test_missing_top_level_fields_default_safely_instead_of_raising():
    """Belt-and-suspenders: even a badly malformed top-level payload (e.g.
    from a pre-A28 cached recommendation) must not crash the app layer."""
    result = validate_and_degrade({"candidates": []}, "Arsenal", "Everton")
    assert result.overall == "insufficient_data"
    assert result.candidates == []


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
    assert result.candidates == []
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
    assert len(result.candidates) == 1


def test_omitting_home_away_skips_the_mismatch_check_but_still_degrades_malformed_markets():
    """BUG-028: GET /api/recommendations/{match_id} has no ground-truth
    fixture to compare against (only match_id/date), so it calls
    validate_and_degrade(raw) with neither -- the match-mismatch check must
    be skipped (not crash on missing args), while the per-market validation
    that endpoint actually needs still runs."""
    bad_market = {**_VALID_CANDIDATE, "market": "1X2"}
    good_market = {**_VALID_CANDIDATE, "market": "btts", "selection": "yes"}
    raw = {
        **_VALID_RAW,
        "match": {"home": "Manchester City", "away": "Liverpool"},
        "candidates": [good_market, bad_market],
        "recommendation_pick": {"market": "btts", "selection": "yes"},
    }

    result = validate_and_degrade(raw)

    assert result.overall == "direct_bet"  # not degraded to insufficient_data -- no mismatch check ran
    assert len(result.candidates) == 1
    assert result.candidates[0].market == "btts"
    assert result.invalid_market_count == 1


def test_missing_match_field_is_not_treated_as_a_mismatch():
    """Some raw payloads omit `match` entirely (e.g. pre-existing malformed
    top-level payloads) -- that's a separate, already-handled degradation
    path, not this mismatch check's concern."""
    result = validate_and_degrade({"candidates": [_VALID_CANDIDATE]}, "Arsenal", "Everton")
    assert result.overall == "insufficient_data"  # from the existing "no overall" default
    assert len(result.candidates) == 1  # candidates themselves are untouched by the mismatch check


def test_target_odds_passes_through_unchanged_w83():
    """W83: the agent-side A52 target_odds field reaches the API response
    on a conditional market exactly as computed, no re-derivation here."""
    market = {**_VALID_CANDIDATE, "recommendation_type": "conditional", "target_odds": 2.35}
    raw = {**_VALID_RAW, "overall": "conditional", "candidates": [market]}

    result = validate_and_degrade(raw, "Arsenal", "Everton")

    assert result.candidates[0].target_odds == 2.35


def test_initial_recommendation_type_passes_through_unchanged_a107():
    """A107: src/agent/schema.py's own initial_recommendation_type
    (the LLM's self-reported type before any downgrade pass ran) must
    reach the API response/cache unchanged, not be silently dropped by
    this app's own independent validation layer."""
    market = {**_VALID_CANDIDATE, "recommendation_type": "no_bet", "initial_recommendation_type": "direct_bet"}
    raw = {**_VALID_RAW, "candidates": [market]}

    result = validate_and_degrade(raw, "Arsenal", "Everton")

    assert result.candidates[0].initial_recommendation_type == "direct_bet"
    assert result.candidates[0].recommendation_type == "no_bet"


def test_missing_initial_recommendation_type_defaults_to_none_for_pre_a107_cached_data():
    """A recommendation cached before A107 shipped won't have this key at
    all -- must default, not raise, same convention as target_odds below."""
    result = validate_and_degrade(_VALID_RAW, "Arsenal", "Everton")
    assert result.candidates[0].initial_recommendation_type is None


def test_missing_target_odds_defaults_to_none_for_pre_a52_cached_data_w83():
    """A recommendation cached before A52 shipped won't have this key at
    all -- must default, not raise, same convention as feature_completeness."""
    result = validate_and_degrade(_VALID_RAW, "Arsenal", "Everton")
    assert result.candidates[0].target_odds is None


def test_missing_min_odds_defaults_instead_of_dropping_the_market_bug032():
    """BUG-032: real DeepSeek output regularly omits min_odds on some
    markets -- must default to 0.0 (matching src/agent/schema.py's own
    default), not drop the market as malformed."""
    market = {k: v for k, v in _VALID_CANDIDATE.items() if k != "min_odds"}
    raw = {**_VALID_RAW, "candidates": [market]}

    result = validate_and_degrade(raw, "Arsenal", "Everton")

    assert len(result.candidates) == 1
    assert result.candidates[0].min_odds == 0.0
    assert result.invalid_market_count == 0


def test_unit_bet_multiplier_passes_through_unchanged():
    raw = {**_VALID_RAW, "unit_bet_multiplier": 3.5}
    result = validate_and_degrade(raw, "Arsenal", "Everton")
    assert result.unit_bet_multiplier == 3.5


def test_missing_unit_bet_multiplier_defaults_to_none_for_pre_a82_cached_data():
    result = validate_and_degrade(_VALID_RAW, "Arsenal", "Everton")
    assert result.unit_bet_multiplier is None


def test_a113_structured_reasoning_passes_through_unchanged():
    raw = {
        **_VALID_RAW,
        "team_evidence": {"home": "Arsenal fact.", "away": "Everton fact."},
        "the_read": "A plain-language judgment.",
    }
    result = validate_and_degrade(raw, "Arsenal", "Everton")
    assert result.team_evidence == {"home": "Arsenal fact.", "away": "Everton fact."}
    assert result.the_read == "A plain-language judgment."


def test_missing_a113_structured_reasoning_defaults_to_none_for_pre_a113_cached_data():
    result = validate_and_degrade(_VALID_RAW, "Arsenal", "Everton")
    assert result.team_evidence is None
    assert result.the_read is None
    assert result.no_bet_read is None


def test_malformed_a113_team_evidence_degrades_to_none_not_a_crash():
    raw = {**_VALID_RAW, "team_evidence": "not a dict"}
    result = validate_and_degrade(raw, "Arsenal", "Everton")
    assert result.team_evidence is None


def test_candidates_and_recommendation_pick_pass_through():
    raw = {**_VALID_RAW}
    result = validate_and_degrade(raw)
    assert len(result.candidates) == 1
    assert result.candidates[0].composite_score == 0.6
    assert result.recommendation_pick == RecommendationPickOut(market="result_3way", selection="home")


def test_dangling_pick_is_dropped_and_overall_capped_to_no_bet():
    """recommendation_pick names a market/selection absent from candidates
    -- the app layer doesn't re-run the agent's own guardrails, so it must
    apply the same downgrade-only cap A90 established: no resolvable pick,
    overall can't stay at a stronger claim than the evidence supports."""
    raw = {**_VALID_RAW, "recommendation_pick": {"market": "btts", "selection": "yes"}}
    result = validate_and_degrade(raw)
    assert result.recommendation_pick is None
    assert result.overall == "no_bet"


def test_home_away_goals_candidate_is_not_dropped_as_malformed():
    """W199 added home_goals/away_goals (over_1.5/under_1.5) to
    src/agent/schema.py's own market/selection Literal on the day it
    shipped, but this file's MarketCandidateOut/RecommendationPickOut kept
    the pre-W199 vocabulary -- so every home/away-goals candidate the agent
    produced was silently dropped here as "malformed data from the agent"
    and could never become the recommendation_pick, even with a real edge."""
    candidate = {**_VALID_CANDIDATE, "market": "home_goals", "selection": "over_1.5"}
    raw = {
        **_VALID_RAW,
        "candidates": [candidate],
        "recommendation_pick": {"market": "home_goals", "selection": "over_1.5"},
    }
    result = validate_and_degrade(raw)
    assert result.invalid_market_count == 0
    assert len(result.candidates) == 1
    assert result.recommendation_pick == RecommendationPickOut(market="home_goals", selection="over_1.5")


def test_pick_pointing_at_a_candidate_that_failed_validation_is_also_dropped():
    """The picked candidate itself is malformed (fails MarketCandidateOut
    validation, gets omitted from `candidates`) -- the pick must not survive
    just because resolve_recommendation_pick() found it in the *raw* list;
    it has to also appear in the *validated* one."""
    bad_candidate = {**_VALID_CANDIDATE, "value_edge": "not-a-number"}
    raw = {**_VALID_RAW, "candidates": [bad_candidate]}
    result = validate_and_degrade(raw)
    assert result.candidates == []
    assert result.recommendation_pick is None
    assert result.overall == "no_bet"


def test_old_shape_row_with_no_candidates_key_degrades_gracefully():
    """A row cached before this migration -- no candidates/recommendation_pick
    key at all, just the old (now-ignored) markets key and a real overall.
    No special detection needed: raw.get("candidates") is naturally [],
    raw.get("recommendation_pick") is naturally None -- the same cap applies."""
    old_shape_raw = {
        "match": {"home": "Arsenal", "away": "Chelsea", "date": "2026-06-15", "league": "E0"},
        "overall": "direct_bet",
        "markets": [{"market": "result_3way", "selection": "home", "recommendation_type": "direct_bet",
                      "current_odds": 2.1, "min_odds": 1.8, "ml_probability": 0.55,
                      "implied_probability": 0.48, "value_edge": 0.07}],
        "explanation": "Value found.", "confidence": "medium", "limitations": [],
        "prediction_basis": "team_history_and_market",
    }
    result = validate_and_degrade(old_shape_raw)
    assert result.candidates == []
    assert result.recommendation_pick is None
    assert result.overall == "no_bet"


def test_no_bet_and_insufficient_data_are_never_touched_by_the_cap():
    """The cap only ever applies when overall claims something stronger --
    an honest no_bet/insufficient_data with no pick is already consistent,
    not something to 'downgrade' further."""
    for value in ("no_bet", "insufficient_data"):
        raw = {**_VALID_RAW, "overall": value, "candidates": [], "recommendation_pick": None}
        result = validate_and_degrade(raw)
        assert result.overall == value


def test_malformed_recommendation_pick_degrades_gracefully_not_a_500():
    """Code-quality review (2026-09-01): resolve_recommendation_pick() is
    called here against RAW, not-yet-validated data -- a recommendation_pick
    missing "selection", or one that isn't a dict at all, previously raised
    an unguarded KeyError/TypeError straight through GET
    /api/recommendations/{match_id}, exactly the crash class BUG-028 exists
    to prevent. Both must now degrade the same as any other unresolvable
    pick: no pick, overall capped."""
    raw = {**_VALID_RAW, "recommendation_pick": {"market": "result_3way"}}  # missing "selection"
    result = validate_and_degrade(raw)
    assert result.recommendation_pick is None
    assert result.overall == "no_bet"

    raw = {**_VALID_RAW, "recommendation_pick": "not-a-dict"}
    result = validate_and_degrade(raw)
    assert result.recommendation_pick is None
    assert result.overall == "no_bet"
