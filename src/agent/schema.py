from __future__ import annotations

import json
from pathlib import Path
import re
from typing import Literal, TypedDict

import json_repair
from pydantic import BaseModel, ValidationError

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

        data = _downgrade_direct_bet_with_null_odds(data)
        data = _downgrade_direct_bet_outside_odds_bounds(data, min_odds_threshold, max_odds_threshold)
        return data  # type: ignore[return-value]

    raise RecommendationParseError(text, f"no valid MatchRecommendation found ({last_error})")
