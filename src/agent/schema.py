from __future__ import annotations

import json
from pathlib import Path
import re
from typing import Literal, TypedDict

import json_repair
from pydantic import BaseModel, ValidationError

from src.agent.market_resolution import resolve_recommendation_pick
from src.agent.staking import kelly_fraction
from src.ingestion.common.team_mapping import TeamNameMapper

_TEAM_MAPPING_PATH = Path(__file__).resolve().parent.parent.parent / "config" / "team_mapping.json"


class MarketCandidate(TypedDict):
    market: Literal["result_3way", "btts", "total_goals", "home_corners", "away_corners", "total_corners"]
    selection: Literal["home", "draw", "away", "yes", "no", "over_2.5", "under_2.5", "over_9.5", "under_9.5"]
    recommendation_type: Literal["direct_bet", "conditional", "no_bet"]
    current_odds: float
    min_odds: float
    ml_probability: float
    implied_probability: float
    value_edge: float
    # A52: populated by extract_recommendation() itself (_compute_target_odds),
    # never by the LLM -- the price a 'conditional' market would need to reach
    # to clear min_value_edge, or None when not applicable/computable.
    target_odds: float | None
    # A88 (2026-08-31 design): the LLM's own self-reported balance of
    # value_edge against ml_probability -- see MarketCandidateModel below.
    composite_score: float
    # A88: one line -- why this candidate won or lost.
    reason: str
    # A107: the LLM's own self-reported recommendation_type, captured
    # before any of the downgrade passes below can touch it -- so a
    # guardrail-caused change (BUG-054's whole investigation needed a full
    # reasoning-trace read to see "direct_bet -> conditional -> no_bet")
    # is a plain initial_recommendation_type != recommendation_type
    # comparison instead. Populated by extract_recommendation() itself,
    # never by the LLM -- same convention as target_odds above.
    initial_recommendation_type: Literal["direct_bet", "conditional", "no_bet"]


class RecommendationPick(TypedDict):
    market: Literal["result_3way", "btts", "total_goals", "home_corners", "away_corners", "total_corners"]
    selection: Literal["home", "draw", "away", "yes", "no", "over_2.5", "under_2.5", "over_9.5", "under_9.5"]


class MatchRecommendation(TypedDict):
    match: dict
    overall: Literal["direct_bet", "conditional", "no_bet", "insufficient_data"]
    candidates: list[MarketCandidate]
    recommendation_pick: RecommendationPick | None
    # One bullet per aspect (value edge, team news, form, market caveats,
    # ...) instead of one narrative paragraph -- direct user request. A plain
    # string (a pre-this-change cached row, or a model that ignores the
    # updated prompt) is still accepted and normalized to a single-item list,
    # see normalize_explanation().
    explanation: list[str]
    confidence: Literal["low", "medium", "high"]
    limitations: list[str]
    prediction_basis: str
    # A82: Kelly-derived stake-sizing suggestion for the recommendation's
    # actual pick, as a multiple of an abstract "Unit Bet" -- not a dollar
    # figure. Computed here (like target_odds/A52), never by the LLM. None
    # when there's no priced pick (no_bet/insufficient_data, or missing
    # odds); 0.0 is a real, distinct value -- a priced 'conditional' market
    # whose edge doesn't clear the bar yet.
    unit_bet_multiplier: float | None
    # W15: not populated by extract_recommendation() itself -- graph.py's
    # _build_recommendation() adds these afterward, read deterministically
    # from the forecast tool's own diagnostics rather than the LLM's JSON.
    cold_start_risk: bool
    feature_completeness: float | None
    unknown_team: bool


_REQUIRED_KEYS = {"match", "overall", "candidates", "explanation", "confidence", "limitations", "prediction_basis"}
_VALID_OVERALL = {"direct_bet", "conditional", "no_bet", "insufficient_data"}


class MarketCandidateModel(BaseModel):
    """A88 (2026-08-31 design): replaces MarketRecommendationModel. Every
    market with a real matched current price gets one entry here -- the LLM
    is asked to list candidates it's rejecting too, not just the one it
    picks (see RecommendationPickModel below), so this comparison survives for
    settlement/frontend transparency the same way the old `markets` array
    did.

    `market`/`selection` were plain `str` (any value accepted) until this
    codebase's prompt (config/prompts/agent_v1.txt) already specified this
    exact fixed vocabulary for both -- nothing enforced it. Confirmed live in
    the sandbox cache: the same result_3way market rendered as "1X2" for one
    fixture; other real generations invented markets/selections entirely
    outside this schema ("Asian Handicap", "IF Brommapojkarna to win",
    team names used as a result_3way selection instead of home/draw/away).
    A market naming a real but different betting line (e.g. a 1.5-goal line
    reported as market="Over 1.5 goals") can't be safely renamed to a
    canonical name without misrepresenting which line it actually was --
    rejecting it (same as any other malformed market) is the safe choice for
    a betting app, not silently relabeling it."""

    market: Literal["result_3way", "btts", "total_goals", "home_corners", "away_corners", "total_corners"]
    selection: Literal["home", "draw", "away", "yes", "no", "over_2.5", "under_2.5", "over_9.5", "under_9.5"]
    recommendation_type: Literal["direct_bet", "conditional", "no_bet"]
    current_odds: float | None
    # BUG-032: defaulted, not required -- confirmed live, DeepSeek output
    # regularly omits this field on some markets within an otherwise-valid
    # recommendation. min_odds is also effectively vestigial now that A52's
    # target_odds is the verified, code-computed replacement the UI
    # actually shows (W84/W87).
    min_odds: float = 0.0
    ml_probability: float
    implied_probability: float
    value_edge: float
    # A52: optional/defaulted so a pre-A52 candidate dict (the LLM never
    # writes this field itself) still validates -- _compute_target_odds()
    # populates the real value after this structural pass runs.
    target_odds: float | None = None
    # A88 (2026-08-31 design): the LLM's own self-reported balance of
    # value_edge against ml_probability (the "hit probability") -- not a
    # code-computed formula, since the whole point is capturing the model's
    # own judgment about the tradeoff, not restating value_edge under a new
    # name. Only ever used by A91's self-consistency guardrail below (does
    # the picked candidate's own score beat every other candidate's) --
    # never trusted as a betting decision on its own, the same "guidance,
    # not a rule code blindly follows" posture as every LLM-self-reported
    # number in this file.
    composite_score: float
    # A88: one line -- why this candidate won or lost, required so the
    # comparison is genuinely legible later (settlement/frontend/lessons),
    # not just a bare number.
    reason: str


class RecommendationPickModel(BaseModel):
    """A88 (2026-08-31 design): which candidate is the actual pick --
    deliberately just the two Literal fields that identify it, not a
    duplicate copy of its numeric fields. resolve_recommendation_pick()
    (src/agent/market_resolution.py) looks the real candidate up in
    `candidates` by matching both fields -- this makes it structurally
    impossible for "the pick" and "its own listed numbers" to quietly
    disagree, since there's only ever one copy of the data."""

    market: Literal["result_3way", "btts", "total_goals", "home_corners", "away_corners", "total_corners"]
    selection: Literal["home", "draw", "away", "yes", "no", "over_2.5", "under_2.5", "over_9.5", "under_9.5"]


class MatchRecommendationModel(BaseModel):
    """A28: adds type/enum validation for confidence and every market field,
    beyond the pre-existing key-presence/overall-enum checks.

    A37: also used directly as the schema passed to
    llm.with_structured_output() for the final-answer synthesis call --
    public (no leading underscore) since it's now imported cross-module by
    src/agent/graph.py, not just used internally by extract_recommendation().

    A88 (2026-08-31 design): `markets` replaced by `candidates` +
    `recommendation_pick` -- see MarketCandidateModel/RecommendationPickModel
    above. `recommendation_pick` defaults to None (not in _REQUIRED_KEYS)
    since a genuine no_bet/insufficient_data response may omit it entirely
    rather than write a literal null."""

    match: dict
    overall: Literal["direct_bet", "conditional", "no_bet", "insufficient_data"]
    candidates: list[MarketCandidateModel]
    recommendation_pick: RecommendationPickModel | None = None
    explanation: list[str]
    confidence: Literal["low", "medium", "high"]
    limitations: list[str]
    prediction_basis: str


def normalize_explanation(value: object) -> list[str]:
    """Bullet-point explanation, one item per aspect -- direct user request,
    replacing the old single-paragraph string. Accepts a plain string too
    (a pre-this-change cached row, or a model that ignores the updated
    prompt) and degrades it to a single-item list rather than failing
    validation over a formatting difference; empty/blank items are dropped."""
    if isinstance(value, list):
        return [str(item).strip() for item in value if str(item).strip()]
    if isinstance(value, str) and value.strip():
        return [value.strip()]
    return []


def _downgrade_direct_bet_below_value_edge_floor(data: dict, min_value_edge: float) -> dict:
    """A67: recommendation_type='direct_bet' requires the market's own
    self-reported value_edge to actually clear min_value_edge -- confirmed
    live: a market reported 'direct_bet' with value_edge=-0.138, a
    fundamentally incoherent combination (a direct bet, by definition,
    claims the model's probability clears the value bar *right now*)
    nothing before this checked, only current_odds bounds/eligibility/the
    conditional floor (A29/A54/A66) -- all price-*realism* checks, not
    value-*coherence*. Runs first, before any of those: a negative-edge
    market has no genuine underlying value to even wait for, so it must
    never reach A29's "reclassify as conditional" path in the first place.
    Downgrades to 'no_bet' -- same BUG-013 precedent (no coherent
    actionable state left). Deliberately scoped to 'direct_bet' only --
    'conditional' explicitly tolerates value_edge below the floor right
    now, that's the entire premise of "wait for a better price to clear
    it later" (A52's target_odds computation)."""
    limitations = list(data.get("limitations") or [])
    for candidate in data.get("candidates", []):
        if candidate["recommendation_type"] != "direct_bet":
            continue
        if candidate["value_edge"] >= min_value_edge:
            continue
        candidate["recommendation_type"] = "no_bet"
        limitations.append(
            f"Downgraded {candidate['market']!r}/{candidate['selection']!r} from direct_bet to no_bet: "
            f"value_edge {candidate['value_edge']} is below the {min_value_edge} floor -- not a coherent "
            "direct bet without a real edge."
        )
    data["limitations"] = limitations
    return data


def _downgrade_direct_bet_below_draw_value_edge_floor(data: dict, min_value_edge_result_3way_draw: float | None) -> dict:
    """Agent-side guardrail (direct user request, 2026-08-29): result_3way's
    draw selection has an independently measured reliability problem the ML
    model itself can't currently fix -- the "draw-framing fallacy" lesson
    (documents/agent_techspec.md) found only a 23-38% hit rate (28%
    aggregate across 85 picks, 5 leagues) despite an apparently-positive
    value_edge. Root-caused to result_3way's training-time class-balance
    sample weighting inflating the model's own predicted P(draw) relative to
    the market (docs/superpowers/specs/2026-08-20-result-3way-sample-weight-retune-design.md)
    -- both obvious ML-side fixes (loading the existing calibration sidecar
    at serving time, dampening the sample weighting further) were already
    tried and confirmed live not to fix it, so this raises the value_edge
    bar for this one already-proven-unreliable market/selection instead,
    same downgrade precedent as A67's own
    _downgrade_direct_bet_below_value_edge_floor just above.

    None (the default) leaves every existing config's behavior unchanged --
    opt-in per config, not a silent global behavior change. Deliberately
    scoped to 'direct_bet' + market == 'result_3way' + selection == 'draw'
    only: home/away don't share this documented failure mode (if anything
    the model under-predicts their probability), and 'conditional' can
    never apply to result_3way at all (A54 already forces it to no_bet).

    ponytail: 0.15 (config/agent_config.yaml) is a first-pass heuristic
    picked from the observed inflation magnitude, not a backtested-optimal
    value -- validate/retune via agent-backtest once enough post-fix data
    exists."""
    if min_value_edge_result_3way_draw is None:
        return data
    limitations = list(data.get("limitations") or [])
    for candidate in data.get("candidates", []):
        if candidate["recommendation_type"] != "direct_bet":
            continue
        if candidate["market"] != "result_3way" or candidate["selection"] != "draw":
            continue
        if candidate["value_edge"] >= min_value_edge_result_3way_draw:
            continue
        candidate["recommendation_type"] = "no_bet"
        limitations.append(
            f"Downgraded 'result_3way'/'draw' from direct_bet to no_bet: value_edge "
            f"{candidate['value_edge']} is below the draw-specific {min_value_edge_result_3way_draw} "
            "floor -- result_3way draw picks have an independently measured reliability problem "
            "(see documents/agent_techspec.md's draw-framing fallacy lesson)."
        )
    data["limitations"] = limitations
    return data


def _downgrade_direct_bet_with_null_odds(data: dict) -> dict:
    """BUG-013: recommendation_type='direct_bet' requires a non-null
    current_odds -- downgrade to 'no_bet' (the only other value valid for this
    market-level field) instead of passing the incoherent combination through."""
    limitations = list(data.get("limitations") or [])
    for candidate in data.get("candidates", []):
        if candidate["recommendation_type"] == "direct_bet" and candidate["current_odds"] is None:
            candidate["recommendation_type"] = "no_bet"
            limitations.append(
                f"Downgraded {candidate['market']!r} from direct_bet to no_bet: current_odds was null."
            )
    data["limitations"] = limitations
    return data


def _downgrade_direct_bet_outside_odds_bounds(
    data: dict, min_odds_threshold: float, max_odds_threshold: float
) -> dict:
    """A29: recommendation_type='direct_bet' requires current_odds within
    [min_odds_threshold, max_odds_threshold] (inclusive) -- code-enforced,
    not left as a prompt-only suggestion. Downgrades to 'conditional' (not
    'no_bet'), matching the pre-existing prompt convention that a market with
    a real price outside the comfort zone is a conditional opportunity, not a
    non-bet. A null current_odds is out of scope here -- BUG-013's rule
    (above) already downgraded that case to 'no_bet' before this runs."""
    limitations = list(data.get("limitations") or [])
    for candidate in data.get("candidates", []):
        if candidate["recommendation_type"] != "direct_bet":
            continue
        odds = candidate["current_odds"]
        if odds is None:
            continue
        if odds < min_odds_threshold or odds > max_odds_threshold:
            candidate["recommendation_type"] = "conditional"
            limitations.append(
                f"Downgraded {candidate['market']!r} from direct_bet to conditional: "
                f"current_odds {odds} outside [{min_odds_threshold}, {max_odds_threshold}]."
            )
    data["limitations"] = limitations
    return data


_CONDITIONAL_ELIGIBLE_MARKETS = frozenset({
    ("total_goals", "over_2.5"),
    ("home_corners", "over_2.5"),
    ("away_corners", "over_2.5"),
    ("btts", "yes"),
    ("total_corners", "over_9.5"),  # A101
})


def _restrict_conditional_to_eligible_markets(data: dict) -> dict:
    """A54: 'conditional' is only a coherent recommendation for "over"/"yes"
    -type markets (total_goals over, corners over, BTTS yes) -- these are the
    markets where waiting for a better price is a real, directional strategy,
    not a coin flip on which way the market moves. The complement markets
    (under, BTTS no) and outcome markets (result_3way) have no such reliable
    one-directional drift, so labeling them 'conditional' -- a state whose
    entire premise is "waiting predictably helps" -- is structurally
    misleading regardless of how correctly A52 computes a target_odds for
    it. Downgrades to 'no_bet', not 'conditional' -- same BUG-013 precedent
    (no coherent actionable state left -> the safe non-bet default). Run
    after A29's own downgrade pass, so both its algorithmic conditional
    calls and the LLM's own organic ones are covered by one check; before
    A52's target_odds computation, so an ineligible market never gets one
    (it's no longer 'conditional' by the time that pass runs)."""
    limitations = list(data.get("limitations") or [])
    for candidate in data.get("candidates", []):
        if candidate["recommendation_type"] != "conditional":
            continue
        if (candidate["market"], candidate["selection"]) in _CONDITIONAL_ELIGIBLE_MARKETS:
            continue
        candidate["recommendation_type"] = "no_bet"
        limitations.append(
            f"Downgraded {candidate['market']!r}/{candidate['selection']!r} from conditional to no_bet: "
            "'conditional' only applies to over/yes-type markets (total_goals over, corners over, "
            "btts yes), where waiting for a better price is a directional strategy, not a coin flip."
        )
    data["limitations"] = limitations
    return data


def _downgrade_conditional_below_floor(data: dict, min_conditional_odds_threshold: float) -> dict:
    """A66: 'conditional' current_odds below this floor is downgraded to
    'no_bet' -- either the price is already so short that "wait for it to
    improve" isn't a realistic strategy (a market wouldn't plausibly move
    from e.g. 1.13 to a value-clearing price), or current_odds isn't a
    real price at all (confirmed live: current_odds=0.0 on a corners
    market with no real bookmaker feed to ground it -- decimal odds are
    mathematically never below 1.0, so any value under a sane floor is
    already degenerate). One check covers both: null current_odds is
    already out of scope here (never triggers this comparison), same
    precedent as A29's own bounds check leaving null to BUG-013's rule.
    Run after A29's/A54's own downgrade passes, so this catches both an
    algorithmically-downgraded direct_bet and the LLM's own organic
    'conditional' call; before A52's target_odds computation, so a
    downgraded market never gets one (it's no longer 'conditional' by the
    time that pass runs)."""
    limitations = list(data.get("limitations") or [])
    for candidate in data.get("candidates", []):
        if candidate["recommendation_type"] != "conditional":
            continue
        odds = candidate["current_odds"]
        if odds is None or odds >= min_conditional_odds_threshold:
            continue
        candidate["recommendation_type"] = "no_bet"
        limitations.append(
            f"Downgraded {candidate['market']!r}/{candidate['selection']!r} from conditional to no_bet: "
            f"current_odds {odds} is below the {min_conditional_odds_threshold} floor -- too short "
            "a price for 'wait for it to improve' to be a realistic strategy."
        )
    data["limitations"] = limitations
    return data


def _downgrade_conditional_above_ceiling(data: dict, max_conditional_odds_threshold: float) -> dict:
    """Direct user request (2026-08-28): a hard ceiling on top of A29's own
    direct_bet ceiling -- without this, a direct_bet priced above
    max_odds_threshold downgrades to 'conditional' (A29's own rule) and
    would otherwise sail through with no upper bound at all, since this
    codebase's own conditional handling previously only had a floor (A66),
    never a ceiling. Same downgrade target/precedent as A66's floor: 'no_bet',
    not left as 'conditional' -- a price this long isn't a "wait for a
    better number" situation, it's simply outside the range the user wants
    recommended at all. Default is unbounded (float('inf')) so every config
    that doesn't explicitly set max_conditional_odds_threshold keeps
    today's real, pre-existing no-ceiling behavior unchanged."""
    limitations = list(data.get("limitations") or [])
    for candidate in data.get("candidates", []):
        if candidate["recommendation_type"] != "conditional":
            continue
        odds = candidate["current_odds"]
        if odds is None or odds <= max_conditional_odds_threshold:
            continue
        candidate["recommendation_type"] = "no_bet"
        limitations.append(
            f"Downgraded {candidate['market']!r}/{candidate['selection']!r} from conditional to no_bet: "
            f"current_odds {odds} is above the {max_conditional_odds_threshold} ceiling."
        )
    data["limitations"] = limitations
    return data


def _promote_favorite_to_conditional_for_live_wait(
    data: dict, live_wait_min_odds: float | None, live_wait_target_odds: float, min_value_edge: float,
) -> dict:
    """A112, direct user strategy (2026-09-17): a favorite priced at
    live_wait_min_odds or better (not shorter) is worth recommending as
    'conditional' rather than not betting at all right now. The strategy:
    waiting into the match for an early non-event (classic example: btts
    priced around -150, wait ~20min, if neither side has scored yet the
    live price drifts out toward +100 with the true probability barely
    changed) reliably gets a better number on a pick that's still likely
    to win.

    A candidate qualifies when ALL of:
      1. recommendation_type == 'no_bet' -- scoped to 'no_bet' only. Found
         live (2026-09-17), a real card (Espanyol v Elche btts, edge 9.4%
         at 1.73/-137, the only qualifying candidate on the match): the
         original version of this rule ALSO overrode an existing
         'direct_bet' to 'conditional' whenever it met the same price/
         probability checks -- but that throws away a CERTAIN edge in hand
         (9.4%, already clearing the bar right now) for an UNCERTAIN future
         one, with no comparison of which is actually better, and no
         weighing of the real risk that the price doesn't drift as
         expected (an early goal moves btts:yes the WRONG way -- shorter,
         not toward +100). The LLM's own explanation text is also written
         BEFORE this override runs, so it stayed direct_bet-flavored
         ("the bet can be taken at the current price") while the badge
         said "Conditional" -- a live, user-reported contradiction.
         Direct user clarification, same investigation: overriding an
         *already-qualifying* direct_bet isn't a same-candidate decision
         at all -- it's a cross-candidate one (comparing a genuine
         direct_bet against a DIFFERENT, higher-probability candidate that
         only qualifies once its price improves), handled by
         _prefer_higher_probability_conditional_pick() below, which runs
         later in the pipeline. This function's only job is filling the
         'no_bet' gap the value-edge check alone wouldn't (a candidate
         whose CURRENT price/edge doesn't clear the bar, but whose
         expected target price would).
      2. market/selection is in _CONDITIONAL_ELIGIBLE_MARKETS (A54's own
         restriction -- waiting has to be a real directional strategy).
      3. current_odds >= live_wait_min_odds -- not too short a price to
         realistically wait on (e.g. -150 or better).
      4. current_odds < live_wait_target_odds -- there's an actual price
         improvement to wait FOR (a 'no_bet' candidate already priced
         longer than the target would have cleared edge at the target too,
         by the same monotonic-implied-probability reasoning as check 5 --
         this is mostly a defensive/structural guard here, not the load-
         bearing check it was when this rule could also touch direct_bet).
      5. ml_probability - implied(live_wait_target_odds) >= min_value_edge
         -- direct user refinement (2026-09-17): don't recommend waiting
         on a coin-flip-ish favorite that wouldn't even clear the edge bar
         at the assumed target price.

    live_wait_min_odds=None (default) disables this rule entirely -- only a
    config that explicitly opts in (config/agent_config.yaml) sees this
    behavior; every other caller/test is unaffected by construction.

    Must run AFTER A66's own floor check and the ceiling check: those exist
    for the ORIGINAL 'conditional' reasoning (waiting for a general
    pre-match price drift), and their floor (currently 1.71, -140) is
    actually tighter than this strategy's own explicit floor (e.g. 1.6667,
    -150) -- if this ran before them, a newly-promoted -150 pick would be
    immediately undone by a floor built for a different premise. Must run
    BEFORE A52's target_odds computation, so these newly-conditional
    candidates get a real target price too, same as any other conditional
    pick."""
    if live_wait_min_odds is None:
        return data
    implied_at_target = 1 / live_wait_target_odds
    limitations = list(data.get("limitations") or [])
    for candidate in data.get("candidates", []):
        if candidate["recommendation_type"] != "no_bet":
            continue
        if (candidate["market"], candidate["selection"]) not in _CONDITIONAL_ELIGIBLE_MARKETS:
            continue
        odds = candidate["current_odds"]
        prob = candidate["ml_probability"]
        if odds is None or prob is None:
            continue
        if odds < live_wait_min_odds or odds >= live_wait_target_odds:
            continue
        edge_at_target = prob - implied_at_target
        if edge_at_target >= min_value_edge:
            candidate["recommendation_type"] = "conditional"
            limitations.append(
                f"Promoted {candidate['market']!r}/{candidate['selection']!r} from 'no_bet' to "
                f"conditional: current_odds {odds} is between the {live_wait_min_odds} floor and the "
                f"{live_wait_target_odds} live-wait target, and ml_probability {prob} clears "
                f"min_value_edge ({edge_at_target:.4f} >= {min_value_edge}) at that target price -- "
                "a likely winner still worth waiting on for a better price rather than not betting at all."
            )
    data["limitations"] = limitations
    return data


def _prefer_higher_probability_conditional_pick(data: dict) -> dict:
    """A112 refinement, direct user clarification (2026-09-17), replacing
    this rule's original same-candidate direct_bet override (see the
    docstring above): the live-wait trade-off is a CROSS-candidate
    decision, not a same-candidate one. Concrete example given: a match
    has a 'draw' direct_bet (10% edge, 30% ml_probability) and a 'btts'
    candidate that only qualifies as 'conditional' once its price hits the
    live-wait target (4% edge now, would be 5% at +100, 55% ml_probability)
    -- the btts conditional pick is the BETTER recommendation, because its
    win probability is far higher, even though its edge is smaller and it
    isn't bettable at the current price at all.

    Runs near the end of the pipeline: only reconsiders `recommendation_pick`
    when it currently resolves to a 'direct_bet' candidate (the LLM's own
    choice, already past every other guardrail including A91's self-
    consistency check) -- finds every OTHER candidate that is 'conditional'
    and in _CONDITIONAL_ELIGIBLE_MARKETS (A54's restriction still applies:
    waiting only makes sense for those markets), and switches the pick to
    whichever one has the single HIGHEST ml_probability, but only if that
    beats the current direct_bet's own ml_probability. A no-op when the
    resolved pick isn't 'direct_bet', when no eligible 'conditional'
    candidate exists, or when none beats the current pick's probability --
    the certain, already-qualifying direct_bet stands in every other case.

    Must run AFTER _downgrade_recommendation_below_top_composite_score
    (A91): that check validates the LLM's own self-consistency against its
    OWN pick, which would be the wrong thing to re-run against a pick this
    function itself chose. Must run BEFORE _resolve_recommendation_pick,
    which syncs `overall` to whatever `recommendation_pick` names -- so a
    switch made here is reflected in `overall` (and therefore the frontend
    badge) the same way any other pick resolution already is, with no
    separate sync logic needed."""
    pick = data.get("recommendation_pick")
    candidates = data.get("candidates") or []
    current = resolve_recommendation_pick(candidates, pick)
    if current is None or current["recommendation_type"] != "direct_bet":
        return data

    eligible_conditionals = [
        c for c in candidates
        if c is not current
        and c["recommendation_type"] == "conditional"
        and (c["market"], c["selection"]) in _CONDITIONAL_ELIGIBLE_MARKETS
    ]
    if not eligible_conditionals:
        return data

    best = max(eligible_conditionals, key=lambda c: c["ml_probability"])
    if best["ml_probability"] <= current["ml_probability"]:
        return data

    data["recommendation_pick"] = {"market": best["market"], "selection": best["selection"]}
    limitations = list(data.get("limitations") or [])
    limitations.append(
        f"Switched the recommendation from {current['market']!r}/{current['selection']!r} (direct_bet, "
        f"ml_probability {current['ml_probability']}) to {best['market']!r}/{best['selection']!r} "
        f"(conditional, ml_probability {best['ml_probability']}) -- a much more likely winner worth "
        "waiting on, even though it isn't bettable at the current price."
    )
    data["limitations"] = limitations
    return data


def _compute_target_odds(data: dict, min_value_edge: float) -> dict:
    """A52: for each 'conditional' market with real current_odds, compute the
    price it would need to reach to actually clear the value-edge bar --
    code-computed since ml_probability/current_odds are deterministic inputs
    and min_value_edge is config, unlike the LLM's own min_odds field (never
    verified against anything). Run last, after both downgrade passes above,
    since a market's final recommendation_type (in particular A29's
    direct_bet -> conditional bounds downgrade) must already be settled
    before this decides which markets it applies to.

    needed_prob = ml_probability - min_value_edge is the ML-probability floor
    this market would still need at whatever price we solve for; its
    break-even price is target_price = 1 / needed_prob. That's only a genuine
    forward target when it's strictly above current_odds -- target_price <=
    current_odds means the current price already clears the bar (nothing to
    wait for) or sits on the wrong side of it entirely (A29's ceiling-
    downgrade case: current_odds already too high, so 'wait for it to rise'
    would be backwards). Both degrade to None, same as needed_prob <= 0
    (no price fixes an ml_probability that's already below the edge floor)."""
    for candidate in data.get("candidates", []):
        if candidate["recommendation_type"] != "conditional" or candidate["current_odds"] is None:
            candidate["target_odds"] = None
            continue
        needed_prob = candidate["ml_probability"] - min_value_edge
        if needed_prob <= 0:
            candidate["target_odds"] = None
            continue
        target_price = 1 / needed_prob
        candidate["target_odds"] = target_price if target_price > candidate["current_odds"] else None
    return data


_RANK_TO_OVERALL = ["insufficient_data", "no_bet", "conditional", "direct_bet"]
_OVERALL_RANK = {name: rank for rank, name in enumerate(_RANK_TO_OVERALL)}


def _resolve_recommendation_pick(data: dict) -> dict:
    """Replaces A65's _reconcile_overall_with_markets now that there's at
    most one real pick instead of an array to reconcile against. Runs last
    among the downgrade passes (after Task 3's seven per-candidate checks
    and A91's self-consistency check below have already mutated whichever
    candidate recommendation_pick names) -- looks that candidate up via
    resolve_recommendation_pick() and syncs `overall`/`recommendation_pick`
    to its state.

    IMPORTANT, not the same direction as A65: when a pick DOES resolve, the
    main branch below (`else: data["overall"] = resolved["recommendation_type"]`)
    is a direct, bidirectional sync, not a downgrade-only cap -- it can also
    RAISE `overall` above whatever the LLM originally self-reported, if the
    resolved pick's own (guardrail-validated) type outranks it (e.g. the LLM
    said 'conditional' but its own picked candidate is a clean, still-valid
    'direct_bet'). This is deliberate: A65's downgrade-only rule existed
    because a self-reported `overall` could legitimately outrank every
    market in an *array* with no single one of them being "the" ground
    truth to sync to. Now there's exactly one resolved, validated candidate
    left once every guardrail above has run -- syncing to it in either
    direction is more honest than leaving `overall` artificially capped at
    a stale, understated self-report.

    Only the OTHER branch below (no resolved pick at all -- null, or
    dangling: recommendation_pick names a market/selection absent from
    candidates, the LLM pointed at something it never actually listed) is
    still downgrade-only, same direction A65 already established: no real
    recommendation, overall capped at 'no_bet' -- never claims a stronger
    state than the candidates actually support. A dangling pick additionally
    gets its own limitations note, distinguishing "the model pointed at
    nothing real" from an ordinary no_bet."""
    pick = data.get("recommendation_pick")
    candidates = data.get("candidates") or []
    resolved = resolve_recommendation_pick(candidates, pick)

    if resolved is None:
        if pick is not None:
            limitations = list(data.get("limitations") or [])
            limitations.append(
                "recommendation_pick named a market/selection not present in candidates -- "
                "treated as no recommendation."
            )
            data["limitations"] = limitations
        data["recommendation_pick"] = None
        if _OVERALL_RANK[data["overall"]] > _OVERALL_RANK["no_bet"]:
            data["overall"] = "no_bet"
        return data

    if resolved["recommendation_type"] == "no_bet":
        data["recommendation_pick"] = None
        data["overall"] = "no_bet"
    else:
        data["overall"] = resolved["recommendation_type"]
    return data


# A82, direct user definition: "the stake you'd risk for a 50/50 confidence
# bet" -- deliberately NOT a Kelly-derived quantity (Kelly at a genuine
# zero-edge 50/50 proposition, p=0.5 against fair 2.0 odds, is 0 -- there's
# no "stake" to derive from that). This is a fixed, flat reference amount
# instead: your standard bet size for a coin-flip proposition, independent
# of any specific bet's odds/edge. unit_bet_multiplier (below) expresses
# each recommendation's actual Kelly-sized stake as a multiple of this one
# fixed baseline -- 1.0 means "bet your standard 50/50 amount," 3.0 means
# "bet 3x that." 1% of bankroll is a conventional flat-unit size in sports
# betting bankroll management (typically 1-2%); happens to equal
# simulate_flat_stake's own default stake_pct (src/agent/staking.py), but
# that's a shared, sensible convention, not a dependency -- this constant
# is not read from or coupled to that function's own default parameter.
UNIT_BET_BASELINE_FRACTION = 0.01


def _attach_unit_bet_multiplier(data: dict) -> dict:
    """A82: deterministic stake-sizing suggestion for the recommendation's
    actual pick, expressed as a multiple of a standard "Unit Bet" (UB) --
    an abstract betting unit, not a dollar figure (bet 2 UB at odds 3.0,
    get 6 UB back). UB itself (UNIT_BET_BASELINE_FRACTION, above) is a
    fixed reference stake, not Kelly-derived; the multiplier is
    A80's kelly_fraction (the actual Kelly-optimal stake for this specific
    pick) expressed as a multiple of that fixed reference. kelly_fraction's
    own max_fraction=0.10 default caps the result at 10.0 automatically, no
    separate clamping needed here.

    Run last, after _resolve_recommendation_pick: by that point
    recommendation_pick is either null (nothing to size) or names a
    candidate whose recommendation_type genuinely survived every guardrail
    above -- A88 (2026-08-31 design) replaces A81's pick_recommended_market
    reduction with a direct pointer lookup, since there's only one real
    candidate left to resolve."""
    picked = resolve_recommendation_pick(data.get("candidates") or [], data.get("recommendation_pick"))
    if picked is None or picked.get("current_odds") is None or picked.get("recommendation_type") == "no_bet":
        data["unit_bet_multiplier"] = None
    else:
        fraction = kelly_fraction(picked.get("value_edge") or 0.0, picked["current_odds"])
        data["unit_bet_multiplier"] = fraction / UNIT_BET_BASELINE_FRACTION
    return data


def _downgrade_recommendation_below_top_composite_score(data: dict) -> dict:
    """A91 (2026-08-31 design): the LLM self-reports composite_score per
    candidate to balance value_edge against ml_probability (the "hit
    probability") -- unlike the checks in Task 3, there's no fixed formula
    for code to verify this number against, so this can't validate the
    *number* itself, only self-consistency: did the model's own stated pick
    actually have the best score among its own listed candidates?

    A rejected candidate self-reporting a strictly higher composite_score
    than the one actually picked is the model contradicting its own
    numbers -- same class of self-contradiction BUG-027/BUG-019 already
    found this model prone to elsewhere. Downgrades straight to 'no_bet',
    same downgrade-only direction as every guardrail in this file -- never
    auto-substitutes the higher-scoring candidate instead (that candidate
    was never itself vetted as the pick, and might fail one of Task 3's
    checks for all this function knows).

    Only compares against candidates whose recommendation_type still
    survives (not 'no_bet') at this point in the pipeline -- an already-
    disqualified candidate was never a real alternative, so out-scoring it
    isn't a contradiction. Runs after every Task 3 guardrail (judging final,
    validated recommendation_type, not a stale pre-downgrade one) and
    before _resolve_recommendation_pick (which reacts to this function's
    own downgrade the same way it reacts to any other)."""
    pick = data.get("recommendation_pick")
    candidates = data.get("candidates") or []
    resolved = resolve_recommendation_pick(candidates, pick)
    if resolved is None or resolved["recommendation_type"] == "no_bet":
        return data

    own_score = resolved["composite_score"]
    better = [
        c for c in candidates
        if c is not resolved and c["recommendation_type"] != "no_bet" and c["composite_score"] > own_score
    ]
    if not better:
        return data

    top = max(better, key=lambda c: c["composite_score"])
    original_type = resolved["recommendation_type"]
    resolved["recommendation_type"] = "no_bet"
    limitations = list(data.get("limitations") or [])
    limitations.append(
        f"Downgraded {resolved['market']!r}/{resolved['selection']!r} from {original_type!r} to "
        f"no_bet: self-reported composite_score {own_score} is lower than {top['market']!r}/"
        f"{top['selection']!r}'s own {top['composite_score']} -- the pick contradicts its own "
        "listed candidates."
    )
    data["limitations"] = limitations
    return data


def reported_teams(match_field: dict) -> tuple[str, str] | None:
    """The two team names the agent's own `match` field claims this
    recommendation is about, tolerating the `home`/`away` key spelling the
    LLM sometimes uses instead of `home_team`/`away_team`. None if either
    side is missing -- some raw payloads omit `match` entirely, which is
    not itself a mismatch. Public (no leading underscore): reused by
    app/backend/recommendations.py's own defensive check on cached/historical
    recommendations (BUG-023/024), so there's one canonical implementation."""
    home = match_field.get("home_team") or match_field.get("home")
    away = match_field.get("away_team") or match_field.get("away")
    return (home, away) if home and away else None


def teams_match(requested: tuple[str, str], reported: tuple[str, str]) -> bool:
    """Canonical, order-independent comparison via TeamNameMapper/
    config/team_mapping.json (the same pattern BUG-015 established for
    odds-to-fixture matching). Order-independent so a plain home/away swap
    isn't flagged as a mismatch, only a genuinely different pair of clubs."""
    mapper = TeamNameMapper(mapping_path=str(_TEAM_MAPPING_PATH))
    return {mapper.map_team(t) for t in requested} == {mapper.map_team(t) for t in reported}


class RecommendationParseError(Exception):
    def __init__(self, raw_text: str, reason: str = ""):
        self.raw_text = raw_text
        msg = f"Failed to parse MatchRecommendation from agent output"
        if reason:
            msg += f": {reason}"
        super().__init__(msg)


def extract_recommendation(
    text: str,
    min_odds_threshold: float = 1.2,
    max_odds_threshold: float = 11.0,
    min_conditional_odds_threshold: float = 1.5,
    max_conditional_odds_threshold: float = float("inf"),
    min_value_edge: float = 0.05,
    min_value_edge_result_3way_draw: float | None = None,
    live_wait_min_odds: float | None = None,
    live_wait_target_odds: float = 2.0,
    home_team: str | None = None,
    away_team: str | None = None,
) -> MatchRecommendation:
    """Extract and validate a MatchRecommendation JSON block from agent output text.

    Tries all fenced ```json blocks last-to-first (the final block is the recommendation),
    then falls back to the outermost bare JSON object.

    BUG-023/024: the agent's LLM call has been observed hallucinating a
    completely unrelated match's analysis (most often "Manchester City vs
    Liverpool", confirmed on 5/10 fixtures in one live sandbox batch) instead
    of grounding itself in the real requested fixture. When `home_team`/
    `away_team` are supplied (the real fixture this call was for), a
    candidate whose own `match` field names a different pair of clubs is
    rejected the same way any other malformed candidate is -- tried against
    the next candidate JSON block if one exists, otherwise surfaced as the
    existing RecommendationParseError degrade path (graph.py's
    `_build_recommendation` already turns that into a safe
    `insufficient_data` placeholder keyed on the *real* match_info, not the
    hallucinated one). Optional/backward compatible: omitting them (as every
    pre-existing caller/test in this file does) skips the check entirely.
    """
    candidates: list[str] = []

    # Collect all fenced ```json blocks, reversed so we try the last one first
    fenced_blocks = re.findall(r"```json\s*(.*?)\s*```", text, re.DOTALL)
    candidates.extend(reversed(fenced_blocks))

    # Fall back to outermost { ... } object
    bare = re.search(r"\{.*\}", text, re.DOTALL)
    if bare:
        candidates.append(bare.group(0))

    if not candidates:
        raise RecommendationParseError(text, "no JSON object found")

    _decoder = json.JSONDecoder()
    last_error = ""
    for json_str in candidates:
        try:
            # raw_decode tolerates trailing characters (e.g. duplicate '}' from weak models)
            data, _ = _decoder.raw_decode(json_str.lstrip())
        except json.JSONDecodeError as exc:
            # Fall back to a tolerant repair pass for structurally broken JSON
            # (e.g. weak models splitting one array into several bracket groups)
            try:
                data = json_repair.loads(json_str)
            except Exception:
                data = None
            if not isinstance(data, dict):
                last_error = f"invalid JSON: {exc}"
                continue

        missing = _REQUIRED_KEYS - data.keys()
        if missing:
            last_error = f"missing fields: {sorted(missing)}"
            continue

        # Normalize before the structural (list[str]) validation below, not
        # after -- a plain-string explanation would otherwise fail
        # MatchRecommendationModel.model_validate() before this ever runs.
        data["explanation"] = normalize_explanation(data.get("explanation"))

        # BUG-020: `not in` on a set requires hashing the LHS -- a malformed
        # response with a dict/list `overall` (observed live from
        # qwen2.5-coder:7b) raised an unhandled TypeError here instead of
        # being treated as an invalid value. isinstance-check first so any
        # non-string overall falls through to the same graceful path.
        if not isinstance(data["overall"], str) or data["overall"] not in _VALID_OVERALL:
            last_error = f"invalid overall value: {data['overall']!r}"
            continue

        # A28: type/enum validation for every market field and top-level
        # confidence, beyond the key-presence/overall-enum checks above.
        try:
            MatchRecommendationModel.model_validate(data)
        except ValidationError as exc:
            last_error = f"field validation failed: {exc}"
            continue

        # BUG-023/024: reject a candidate whose self-reported match is a
        # different pair of clubs than what was actually requested.
        if home_team and away_team:
            reported = reported_teams(data.get("match") or {})
            if reported is not None and not teams_match((home_team, away_team), reported):
                last_error = (
                    f"match mismatch: agent reported {reported[0]!r} v {reported[1]!r}, "
                    f"requested {home_team!r} v {away_team!r}"
                )
                continue

        # A107: snapshot each candidate's own self-reported recommendation_type
        # before any downgrade pass below can overwrite it in place.
        for candidate in data.get("candidates", []):
            candidate["initial_recommendation_type"] = candidate["recommendation_type"]

        data = _downgrade_direct_bet_below_value_edge_floor(data, min_value_edge)
        data = _downgrade_direct_bet_below_draw_value_edge_floor(data, min_value_edge_result_3way_draw)
        data = _downgrade_direct_bet_with_null_odds(data)
        data = _downgrade_direct_bet_outside_odds_bounds(data, min_odds_threshold, max_odds_threshold)
        data = _restrict_conditional_to_eligible_markets(data)
        data = _downgrade_conditional_below_floor(data, min_conditional_odds_threshold)
        data = _downgrade_conditional_above_ceiling(data, max_conditional_odds_threshold)
        data = _promote_favorite_to_conditional_for_live_wait(
            data, live_wait_min_odds, live_wait_target_odds, min_value_edge,
        )
        data = _compute_target_odds(data, min_value_edge)
        data = _downgrade_recommendation_below_top_composite_score(data)
        data = _prefer_higher_probability_conditional_pick(data)
        data = _resolve_recommendation_pick(data)
        data = _attach_unit_bet_multiplier(data)
        return data  # type: ignore[return-value]

    raise RecommendationParseError(text, f"no valid MatchRecommendation found ({last_error})")
