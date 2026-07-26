from __future__ import annotations

import json
import re
from typing import Literal, TypedDict

import json_repair
from pydantic import BaseModel, ValidationError


class MarketRecommendation(TypedDict):
    market: str
    selection: str
    recommendation_type: Literal["direct_bet", "conditional", "no_bet"]
    current_odds: float
    min_odds: float
    ml_probability: float
    implied_probability: float
    value_edge: float


class MatchRecommendation(TypedDict):
    match: dict
    overall: Literal["direct_bet", "conditional", "no_bet", "insufficient_data"]
    markets: list[MarketRecommendation]
    explanation: str
    confidence: Literal["low", "medium", "high"]
    limitations: list[str]
    prediction_basis: str
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
    semantic rule applied after this structural validation passes."""

    market: str
    selection: str
    recommendation_type: Literal["direct_bet", "conditional", "no_bet"]
    current_odds: float | None
    min_odds: float
    ml_probability: float
    implied_probability: float
    value_edge: float


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
    explanation: str
    confidence: Literal["low", "medium", "high"]
    limitations: list[str]
    prediction_basis: str


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


class RecommendationParseError(Exception):
    def __init__(self, raw_text: str, reason: str = ""):
        self.raw_text = raw_text
        msg = f"Failed to parse MatchRecommendation from agent output"
        if reason:
            msg += f": {reason}"
        super().__init__(msg)


def extract_recommendation(
    text: str, min_odds_threshold: float = 1.2, max_odds_threshold: float = 11.0
) -> MatchRecommendation:
    """Extract and validate a MatchRecommendation JSON block from agent output text.

    Tries all fenced ```json blocks last-to-first (the final block is the recommendation),
    then falls back to the outermost bare JSON object.
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

        data = _downgrade_direct_bet_with_null_odds(data)
        data = _downgrade_direct_bet_outside_odds_bounds(data, min_odds_threshold, max_odds_threshold)
        return data  # type: ignore[return-value]

    raise RecommendationParseError(text, f"no valid MatchRecommendation found ({last_error})")
