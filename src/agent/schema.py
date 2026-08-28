from __future__ import annotations

import json
from pathlib import Path
import re
from typing import Literal, TypedDict

import json_repair
from pydantic import BaseModel, ValidationError

from src.agent.market_resolution import pick_recommended_market
from src.agent.staking import kelly_fraction
from src.ingestion.common.team_mapping import TeamNameMapper

_TEAM_MAPPING_PATH = Path(__file__).resolve().parent.parent.parent / "config" / "team_mapping.json"


class MarketRecommendation(TypedDict):
    market: Literal["result_3way", "btts", "total_goals", "home_corners", "away_corners"]
    selection: Literal["home", "draw", "away", "yes", "no", "over_2.5", "under_2.5"]
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


class MatchRecommendation(TypedDict):
    match: dict
    overall: Literal["direct_bet", "conditional", "no_bet", "insufficient_data"]
    markets: list[MarketRecommendation]
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
    # actual pick (pick_recommended_market), as a multiple of an abstract
    # "Unit Bet" -- not a dollar figure. Computed here (like target_odds/
    # A52), never by the LLM. None when there's no priced pick (no_bet/
    # insufficient_data, or missing odds); 0.0 is a real, distinct value --
    # a priced 'conditional' market whose edge doesn't clear the bar yet.
    unit_bet_multiplier: float | None
    # W15: not populated by extract_recommendation() itself -- graph.py's
    # _build_recommendation() adds these afterward, read deterministically
    # from the forecast tool's own diagnostics rather than the LLM's JSON.
    cold_start_risk: bool
    feature_completeness: float | None
    unknown_team: bool


_REQUIRED_KEYS = {"match", "overall", "markets", "explanation", "confidence", "limitations", "prediction_basis"}
_VALID_OVERALL = {"direct_bet", "conditional", "no_bet", "insufficient_data"}


class MarketRecommendationModel(BaseModel):
    """A28: type/enum validation for every market-level field. current_odds is
    nullable -- that's a legitimate state (odds simply weren't found for this
    market) -- the direct_bet + null-odds combination (BUG-013) is a separate
    semantic rule applied after this structural validation passes.

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

    market: Literal["result_3way", "btts", "total_goals", "home_corners", "away_corners"]
    selection: Literal["home", "draw", "away", "yes", "no", "over_2.5", "under_2.5"]
    recommendation_type: Literal["direct_bet", "conditional", "no_bet"]
    current_odds: float | None
    # BUG-032: defaulted, not required -- confirmed live, DeepSeek output
    # regularly omits this field on some markets within an otherwise-valid
    # recommendation (real example: a btts market missing min_odds entirely).
    # Before this default, that single missing field failed
    # MatchRecommendationModel validation for the *whole* candidate,
    # discarding every other market's real data along with it. min_odds is
    # also effectively vestigial now that A52's target_odds is the verified,
    # code-computed replacement the UI actually shows (W84/W87) -- there's
    # no remaining reason a missing value here should sink an entire
    # recommendation.
    min_odds: float = 0.0
    ml_probability: float
    implied_probability: float
    value_edge: float
    # A52: optional/defaulted so a pre-A52 candidate dict (the LLM never
    # writes this field itself) still validates -- _compute_target_odds()
    # populates the real value after this structural pass runs.
    target_odds: float | None = None


class MatchRecommendationModel(BaseModel):
    """A28: adds type/enum validation for confidence and every market field,
    beyond the pre-existing key-presence/overall-enum checks.

    A37: also used directly as the schema passed to
    llm.with_structured_output() for the final-answer synthesis call --
    public (no leading underscore) since it's now imported cross-module by
    src/agent/graph.py, not just used internally by extract_recommendation()."""

    match: dict
    overall: Literal["direct_bet", "conditional", "no_bet", "insufficient_data"]
    markets: list[MarketRecommendationModel]
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
    for market in data.get("markets", []):
        if market["recommendation_type"] != "direct_bet":
            continue
        if market["value_edge"] >= min_value_edge:
            continue
        market["recommendation_type"] = "no_bet"
        limitations.append(
            f"Downgraded {market['market']!r}/{market['selection']!r} from direct_bet to no_bet: "
            f"value_edge {market['value_edge']} is below the {min_value_edge} floor -- not a coherent "
            "direct bet without a real edge."
        )
    data["limitations"] = limitations
    return data


def _downgrade_direct_bet_with_null_odds(data: dict) -> dict:
    """BUG-013: recommendation_type='direct_bet' requires a non-null
    current_odds -- downgrade to 'no_bet' (the only other value valid for this
    market-level field) instead of passing the incoherent combination through."""
    limitations = list(data.get("limitations") or [])
    for market in data.get("markets", []):
        if market["recommendation_type"] == "direct_bet" and market["current_odds"] is None:
            market["recommendation_type"] = "no_bet"
            limitations.append(
                f"Downgraded {market['market']!r} from direct_bet to no_bet: current_odds was null."
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
    for market in data.get("markets", []):
        if market["recommendation_type"] != "direct_bet":
            continue
        odds = market["current_odds"]
        if odds is None:
            continue
        if odds < min_odds_threshold or odds > max_odds_threshold:
            market["recommendation_type"] = "conditional"
            limitations.append(
                f"Downgraded {market['market']!r} from direct_bet to conditional: "
                f"current_odds {odds} outside [{min_odds_threshold}, {max_odds_threshold}]."
            )
    data["limitations"] = limitations
    return data


_CONDITIONAL_ELIGIBLE_MARKETS = frozenset({
    ("total_goals", "over_2.5"),
    ("home_corners", "over_2.5"),
    ("away_corners", "over_2.5"),
    ("btts", "yes"),
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
    for market in data.get("markets", []):
        if market["recommendation_type"] != "conditional":
            continue
        if (market["market"], market["selection"]) in _CONDITIONAL_ELIGIBLE_MARKETS:
            continue
        market["recommendation_type"] = "no_bet"
        limitations.append(
            f"Downgraded {market['market']!r}/{market['selection']!r} from conditional to no_bet: "
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
    for market in data.get("markets", []):
        if market["recommendation_type"] != "conditional":
            continue
        odds = market["current_odds"]
        if odds is None or odds >= min_conditional_odds_threshold:
            continue
        market["recommendation_type"] = "no_bet"
        limitations.append(
            f"Downgraded {market['market']!r}/{market['selection']!r} from conditional to no_bet: "
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
    for market in data.get("markets", []):
        if market["recommendation_type"] != "conditional":
            continue
        odds = market["current_odds"]
        if odds is None or odds <= max_conditional_odds_threshold:
            continue
        market["recommendation_type"] = "no_bet"
        limitations.append(
            f"Downgraded {market['market']!r}/{market['selection']!r} from conditional to no_bet: "
            f"current_odds {odds} is above the {max_conditional_odds_threshold} ceiling."
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
    break-even price is candidate = 1 / needed_prob. That's only a genuine
    forward target when it's strictly above current_odds -- candidate <=
    current_odds means the current price already clears the bar (nothing to
    wait for) or sits on the wrong side of it entirely (A29's ceiling-
    downgrade case: current_odds already too high, so 'wait for it to rise'
    would be backwards). Both degrade to None, same as needed_prob <= 0
    (no price fixes an ml_probability that's already below the edge floor)."""
    for market in data.get("markets", []):
        if market["recommendation_type"] != "conditional" or market["current_odds"] is None:
            market["target_odds"] = None
            continue
        needed_prob = market["ml_probability"] - min_value_edge
        if needed_prob <= 0:
            market["target_odds"] = None
            continue
        candidate = 1 / needed_prob
        market["target_odds"] = candidate if candidate > market["current_odds"] else None
    return data


_RANK_TO_OVERALL = ["insufficient_data", "no_bet", "conditional", "direct_bet"]
_OVERALL_RANK = {name: rank for rank, name in enumerate(_RANK_TO_OVERALL)}


def _reconcile_overall_with_markets(data: dict) -> dict:
    """A65: 'overall' is the LLM's own top-level self-report, written
    before the code-enforced downgrade passes above ever run -- A29's
    odds-bounds check (direct_bet -> conditional) and A54's eligible-
    markets check (conditional -> no_bet) mutate individual markets'
    recommendation_type but never touched this field. Confirmed live: a
    Dashboard card showed a 'Direct Bet' badge (from 'overall') over a
    market whose own recommendation_type had already been downgraded to
    'conditional' by A29's odds-bounds check -- the badge and the market
    it was describing told two different stories. Caps 'overall' at the
    strongest recommendation_type any market still actually has, same
    downgrade-only direction as every pass above -- never claims a
    stronger state than the LLM itself originally reported, only a
    weaker, more honest one when the markets no longer support what it
    said. A no-op when `markets` is empty (nothing to reconcile against;
    that shape is reserved for the graph's own no-forecast short-circuit,
    which never reaches this function at all)."""
    markets = data.get("markets") or []
    if not markets:
        return data
    strongest = max(_OVERALL_RANK[m["recommendation_type"]] for m in markets)
    if _OVERALL_RANK[data["overall"]] > strongest:
        old_overall = data["overall"]
        data["overall"] = _RANK_TO_OVERALL[strongest]
        limitations = list(data.get("limitations") or [])
        limitations.append(
            f"Downgraded overall from {old_overall!r} to {data['overall']!r}: no market actually "
            f"supports {old_overall!r} after the downgrade passes above."
        )
        data["limitations"] = limitations
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

    Run last, after every downgrade pass and _reconcile_overall_with_markets:
    needs each market's FINAL recommendation_type to pick the right one
    (A81's pick_recommended_market). pick_recommended_market falls back to
    ranking every market (no_bet included) when nothing in the recommendation
    is actionable at all -- that fallback pick still carries a price, but
    "no_bet" means there's nothing to size, so it's excluded here too, not
    just a missing price."""
    picked = pick_recommended_market(data.get("markets") or [])
    if picked is None or picked.get("current_odds") is None or picked.get("recommendation_type") == "no_bet":
        data["unit_bet_multiplier"] = None
    else:
        fraction = kelly_fraction(picked.get("value_edge") or 0.0, picked["current_odds"])
        data["unit_bet_multiplier"] = fraction / UNIT_BET_BASELINE_FRACTION
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

        data = _downgrade_direct_bet_below_value_edge_floor(data, min_value_edge)
        data = _downgrade_direct_bet_with_null_odds(data)
        data = _downgrade_direct_bet_outside_odds_bounds(data, min_odds_threshold, max_odds_threshold)
        data = _restrict_conditional_to_eligible_markets(data)
        data = _downgrade_conditional_below_floor(data, min_conditional_odds_threshold)
        data = _downgrade_conditional_above_ceiling(data, max_conditional_odds_threshold)
        data = _compute_target_odds(data, min_value_edge)
        data = _reconcile_overall_with_markets(data)
        data = _attach_unit_bet_multiplier(data)
        return data  # type: ignore[return-value]

    raise RecommendationParseError(text, f"no valid MatchRecommendation found ({last_error})")
