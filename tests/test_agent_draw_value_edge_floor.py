"""Agent-side guardrail (direct user request, 2026-08-29): result_3way's draw
selection has an independently measured reliability problem the ML model
itself can't currently fix -- the "draw-framing fallacy" lesson found only a
23-38% hit rate (28% aggregate across 85 picks, 5 leagues) despite an
apparently-positive value_edge, root-caused to result_3way's training-time
class-balance sample weighting inflating the model's own predicted P(draw)
relative to the market (docs/superpowers/specs/2026-08-20-result-3way-sample-weight-retune-design.md).
Both obvious ML-side fixes (recalibration, dampening the sample weighting
further) were already tried and confirmed live not to fix it. This raises
result_3way/draw's own value_edge bar above the shared min_value_edge floor
every other market/selection uses, same downgrade precedent as A67's
_downgrade_direct_bet_below_value_edge_floor -- see BUG-054/test_agent_value_edge_floor.py
for that one."""
from __future__ import annotations

import json

from src.agent.schema import extract_recommendation

_DRAW_MARKET = {
    "market": "result_3way",
    "selection": "draw",
    "recommendation_type": "direct_bet",
    "current_odds": 5.0,
    "min_odds": 1.8,
    "ml_probability": 0.334,
    "implied_probability": 0.20,
    "value_edge": 0.134,
}

_VALID = {
    "match": {"home": "Liverpool", "away": "Nottingham", "date": "2026-08-27", "league": "E0"},
    "overall": "direct_bet",
    "markets": [_DRAW_MARKET],
    "explanation": "Value found on the draw.",
    "confidence": "medium",
    "limitations": [],
    "prediction_basis": "team_history_and_market",
}


def _wrap_json(data: dict) -> str:
    return f"Some reasoning here.\n\n```json\n{json.dumps(data)}\n```"


def test_draw_floor_off_by_default_preserves_existing_behavior():
    """The default (None) must not change any existing config's behavior --
    only a config that explicitly sets the new threshold is affected."""
    rec = extract_recommendation(_wrap_json(_VALID))

    assert rec["markets"][0]["recommendation_type"] == "direct_bet"
    assert rec["limitations"] == []


def test_draw_below_the_dedicated_floor_is_downgraded_even_though_it_clears_the_general_floor():
    """0.134 clears the general min_value_edge floor (0.05) but not a
    draw-specific 0.15 floor -- the exact real production case that
    surfaced this (Liverpool v Nottingham, BUG-054/055)."""
    rec = extract_recommendation(_wrap_json(_VALID), min_value_edge_result_3way_draw=0.15)

    assert rec["markets"][0]["recommendation_type"] == "no_bet"
    assert any("result_3way" in note and "draw" in note and "0.15" in note for note in rec["limitations"])
    assert rec["overall"] == "no_bet"  # A65 reconciliation


def test_draw_above_the_dedicated_floor_is_untouched():
    market = {**_DRAW_MARKET, "value_edge": 0.22, "ml_probability": 0.42}
    data = {**_VALID, "markets": [market]}

    rec = extract_recommendation(_wrap_json(data), min_value_edge_result_3way_draw=0.15)

    assert rec["markets"][0]["recommendation_type"] == "direct_bet"
    assert rec["limitations"] == []


def test_draw_at_exact_dedicated_floor_is_accepted():
    market = {**_DRAW_MARKET, "value_edge": 0.15}
    data = {**_VALID, "markets": [market]}

    rec = extract_recommendation(_wrap_json(data), min_value_edge_result_3way_draw=0.15)

    assert rec["markets"][0]["recommendation_type"] == "direct_bet"


def test_home_and_away_result_3way_selections_are_not_affected():
    """The measured reliability problem, and the ML miscalibration behind
    it, is specifically about draw -- home/away don't share the same
    documented failure mode, so the dedicated floor must not touch them."""
    for selection in ("home", "away"):
        market = {**_DRAW_MARKET, "selection": selection, "value_edge": 0.08}
        data = {**_VALID, "markets": [market]}

        rec = extract_recommendation(_wrap_json(data), min_value_edge_result_3way_draw=0.15)

        assert rec["markets"][0]["recommendation_type"] == "direct_bet", selection
        assert rec["limitations"] == [], selection


def test_other_markets_draw_named_selection_not_affected():
    """Scoped to market == 'result_3way' specifically -- a coincidentally
    'draw'-named selection on a different market (shouldn't happen given
    the schema's own Literal constraints, but this pass must not key off
    selection alone)."""
    market = {**_DRAW_MARKET, "market": "btts", "selection": "yes", "value_edge": 0.08}
    data = {**_VALID, "markets": [market]}

    rec = extract_recommendation(_wrap_json(data), min_value_edge_result_3way_draw=0.15)

    assert rec["markets"][0]["recommendation_type"] == "direct_bet"


def test_no_bet_draw_market_is_not_touched_by_this_pass():
    """This pass is scoped to 'direct_bet' only, same precedent as A67's
    own floor -- an already-no_bet market gets no new limitations note."""
    market = {**_DRAW_MARKET, "recommendation_type": "no_bet", "value_edge": -0.3}
    data = {**_VALID, "overall": "no_bet", "markets": [market]}

    rec = extract_recommendation(_wrap_json(data), min_value_edge_result_3way_draw=0.15)

    assert rec["markets"][0]["recommendation_type"] == "no_bet"
    assert rec["limitations"] == []


def test_conditional_draw_market_is_downgraded_by_a54_not_by_this_pass():
    """Draw/result_3way can never actually be 'conditional' -- A54
    (_restrict_conditional_to_eligible_markets) already force-downgrades it
    to no_bet regardless of value_edge, before this pass even matters."""
    market = {**_DRAW_MARKET, "recommendation_type": "conditional", "value_edge": -0.3}
    data = {**_VALID, "overall": "conditional", "markets": [market]}

    rec = extract_recommendation(_wrap_json(data), min_value_edge_result_3way_draw=0.15)

    assert rec["markets"][0]["recommendation_type"] == "no_bet"
    assert any("only applies to over/yes-type markets" in note for note in rec["limitations"])


def test_runs_after_the_general_floor_so_a_negative_edge_draw_cites_the_general_reason():
    """A67's general floor (min_value_edge, default 0.05) already downgrades
    any negative-edge direct_bet regardless of market -- the draw-specific
    floor only needs to add anything for the gap between the two floors."""
    market = {**_DRAW_MARKET, "value_edge": -0.1}
    data = {**_VALID, "markets": [market]}

    rec = extract_recommendation(_wrap_json(data), min_value_edge_result_3way_draw=0.15)

    assert rec["markets"][0]["recommendation_type"] == "no_bet"
    assert any("-0.1" in note for note in rec["limitations"])
