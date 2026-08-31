# Single-Market Recommendation Decision Mechanism Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the agent's independently-scored `markets` array + post-hoc `pick_recommended_market()`/`bestMarket()` reduction with a schema where the LLM weighs all evaluated markets against each other and commits to one `recommendation_pick`, in the same single `run_agent()` call, validated by code-enforced guardrails that operate on that one resolved pick.

**Architecture:** `MatchRecommendationModel` gains two fields, `candidates: list[MarketCandidateModel]` (every market with real matched odds, richer than today's `MarketRecommendationModel` — adds `composite_score`/`reason`) and `recommendation_pick: RecommendationPick | None` (a `market`+`selection` pointer, not a duplicate copy of the candidate's numbers). A new shared resolver (`resolve_recommendation_pick()`, `src/agent/market_resolution.py`) looks the pick up in `candidates` by equality match. The six existing per-market guardrail functions in `src/agent/schema.py` need almost no logic change — they already loop over every market downgrading whichever ones fail; they just loop over `candidates` instead of `markets` now. What replaces `_reconcile_overall_with_markets` (A65) is a new, much simpler function that resolves the pick against its (now guardrail-validated) candidate and syncs `overall` to match — no reconciliation across an array needed, since there's only one pick to check. A new self-consistency guardrail catches the LLM naming a pick whose own listed `composite_score` isn't actually the best among its own candidates.

**Tech Stack:** Python 3, Pydantic (schema/validation), pytest (TDD) — no new dependencies.

**Scope:** Covers `src/agent/schema.py`, `src/agent/market_resolution.py`, and all 4 prompt files only (A88–A91, Phase 29, `documents/agent_user_stories.md`) — exactly the design spec's `docs/superpowers/specs/2026-08-31-single-market-recommendation-design.md` scope. `app/backend/recommendation_outcomes.py` (settlement), `app/frontend/components/MatchUI.tsx`, and the `agent-backtest`/`agent-train` harness are **not touched by this plan** — `pick_recommended_market()` and `bestMarket()` keep their current callers and definitions untouched, since deleting either now would break those still-on-the-old-schema call sites. That's a follow-up phase's job, per the design spec's own explicit scoping.

**A note on test state between tasks:** `extract_recommendation()`'s Pydantic validation is all-or-nothing — once Task 1 changes `MatchRecommendationModel` to require `candidates` instead of `markets`, every *other* existing test file that still builds a `markets: [...]` fixture (the six guardrail test files, reworked in Task 3) will fail until its own task reworks it. This is expected mid-plan, not a regression to chase — each task's own "run tests" step is scoped to the file(s) that task just touched; the full-suite green check is Task 7's job, at the end.

---

### Task 1: New schema — `MarketCandidateModel`, `RecommendationPick`, updated `MatchRecommendationModel`

**Files:**
- Modify: `src/agent/schema.py:1-119` (imports, `MatchRecommendation` TypedDict, `_REQUIRED_KEYS`, `MarketRecommendationModel` → `MarketCandidateModel`, new `RecommendationPick`, `MatchRecommendationModel`)
- Modify: `tests/test_agent_schema_validation.py`

- [ ] **Step 1: Write the failing tests**

Replace the top of `tests/test_agent_schema_validation.py` (imports through the module-level fixtures) and its first three tests:

```python
"""Regression tests for A28: extract_recommendation must validate field types/
enums beyond key presence, and close BUG-013's root cause (a market marked
direct_bet with a null current_odds) at extraction time rather than passing
it through to crash downstream. Covers the three specific gaps documented in
agent_techspec.md Section 17 (value_edge as a string, confidence as an empty
string, an arbitrary recommendation_type string) plus BUG-013's null-odds
case.

A88 (2026-08-31): reworked for the single-recommendation schema -- `markets`
is now `candidates` (richer: adds composite_score/reason) plus a
`recommendation_pick` pointer naming which candidate is the actual pick."""

from __future__ import annotations

import json

import pytest

from src.agent.schema import RecommendationParseError, extract_recommendation

_VALID_CANDIDATE = {
    "market": "result_3way",
    "selection": "home",
    "recommendation_type": "direct_bet",
    "current_odds": 2.1,
    "min_odds": 1.8,
    "ml_probability": 0.55,
    "implied_probability": 0.48,
    "value_edge": 0.07,
    "composite_score": 0.62,
    "reason": "Clears the edge floor with a well-supported home win probability.",
}

_VALID_PICK = {"market": "result_3way", "selection": "home"}

_VALID = {
    "match": {"home": "Arsenal", "away": "Chelsea", "date": "2026-06-15", "league": "E0"},
    "overall": "direct_bet",
    "candidates": [_VALID_CANDIDATE],
    "recommendation_pick": _VALID_PICK,
    "explanation": "Value found on the home win.",
    "confidence": "medium",
    "limitations": [],
    "prediction_basis": "team_history_and_market",
}


def _wrap_json(data: dict) -> str:
    return f"Some reasoning here.\n\n```json\n{json.dumps(data)}\n```"


def test_fully_valid_output_with_a_real_candidate_still_parses_unchanged():
    """Regression: a valid single-recommendation output (candidates +
    recommendation_pick) must parse cleanly with no downgrades."""
    rec = extract_recommendation(_wrap_json(_VALID))
    assert rec["overall"] == "direct_bet"
    assert rec["candidates"][0]["recommendation_type"] == "direct_bet"
    assert rec["candidates"][0]["current_odds"] == 2.1
    assert rec["candidates"][0]["composite_score"] == 0.62
    assert rec["recommendation_pick"] == _VALID_PICK
    assert rec["limitations"] == []


def test_missing_composite_score_raises():
    """composite_score/reason are new, required MarketCandidateModel fields
    -- a candidate missing either fails validation the same way a missing
    ml_probability already does, not silently defaulted (unlike min_odds,
    BUG-032 -- composite_score is new and load-bearing for A91's
    self-consistency guardrail, not vestigial)."""
    bad_candidate = {k: v for k, v in _VALID_CANDIDATE.items() if k != "composite_score"}
    bad = {**_VALID, "candidates": [bad_candidate]}
    with pytest.raises(RecommendationParseError, match="composite_score"):
        extract_recommendation(_wrap_json(bad))


def test_recommendation_pick_missing_selection_raises():
    bad_pick = {"market": "result_3way"}
    bad = {**_VALID, "recommendation_pick": bad_pick}
    with pytest.raises(RecommendationParseError, match="selection"):
        extract_recommendation(_wrap_json(bad))


def test_recommendation_pick_can_be_null():
    """A no_bet call with nothing actionable: recommendation_pick is null,
    candidates can still list what was considered (or be empty)."""
    data = {**_VALID, "overall": "no_bet", "recommendation_pick": None,
             "candidates": [{**_VALID_CANDIDATE, "recommendation_type": "no_bet"}]}
    rec = extract_recommendation(_wrap_json(data))
    assert rec["recommendation_pick"] is None
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_agent_schema_validation.py -v`
Expected: FAIL — `extract_recommendation` still requires `markets`, not `candidates`; these tests get `RecommendationParseError: missing fields: ['markets']` or similar. (The file's *other*, not-yet-updated tests below this point will also start failing once Step 3 lands — that's expected, see the plan-level note above. Leave them broken; they get fixed in this same task's Step 3 continuation below.)

- [ ] **Step 3: Update the remaining tests in the same file to the new fixture shape, and implement the schema change**

The rest of `tests/test_agent_schema_validation.py` (every test after the three above) follows the exact same two substitutions throughout the file — apply both to every remaining test:

1. `{**_VALID_MARKET, ...}` → `{**_VALID_CANDIDATE, ...}`, and `{**_VALID, "markets": [...]}` → `{**_VALID, "candidates": [...]}` (add `"recommendation_pick": _VALID_PICK` alongside when the test's own candidate keeps `market`/`selection` matching `_VALID_PICK`'s `result_3way`/`home`; when a test constructs a candidate with different market/selection, its own `recommendation_pick` must name that same market/selection instead).
2. `rec["markets"][0][...]` → `rec["candidates"][0][...]`.

Now implement the schema itself. In `src/agent/schema.py`, locate `MarketRecommendationModel` (currently `~line 65`) and replace it, `MatchRecommendationModel` (currently `~line 108`), the `MatchRecommendation` TypedDict's `markets` field (currently `~line 36`), and `_REQUIRED_KEYS` (currently `~line 61`):

```python
class MatchRecommendation(TypedDict):
    match: dict
    overall: Literal["direct_bet", "conditional", "no_bet", "insufficient_data"]
    candidates: list[MarketCandidate]
    recommendation_pick: MarketPick | None
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
    picks (see RecommendationPick below), so this comparison survives for
    settlement/frontend transparency the same way the old `markets` array
    did.

    market`/`selection` were plain `str` (any value accepted) until this
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


class RecommendationPick(BaseModel):
    """A88 (2026-08-31 design): which candidate is the actual pick --
    deliberately just the two Literal fields that identify it, not a
    duplicate copy of its numeric fields. resolve_recommendation_pick()
    (src/agent/market_resolution.py) looks the real candidate up in
    `candidates` by matching both fields -- this makes it structurally
    impossible for "the pick" and "its own listed numbers" to quietly
    disagree, since there's only ever one copy of the data."""

    market: Literal["result_3way", "btts", "total_goals", "home_corners", "away_corners"]
    selection: Literal["home", "draw", "away", "yes", "no", "over_2.5", "under_2.5"]


class MatchRecommendationModel(BaseModel):
    """A28: adds type/enum validation for confidence and every market field,
    beyond the pre-existing key-presence/overall-enum checks.

    A37: also used directly as the schema passed to
    llm.with_structured_output() for the final-answer synthesis call --
    public (no leading underscore) since it's now imported cross-module by
    src/agent/graph.py, not just used internally by extract_recommendation().

    A88 (2026-08-31 design): `markets` replaced by `candidates` +
    `recommendation_pick` -- see MarketCandidateModel/RecommendationPick
    above. `recommendation_pick` defaults to None (not in _REQUIRED_KEYS)
    since a genuine no_bet/insufficient_data response may omit it entirely
    rather than write a literal null."""

    match: dict
    overall: Literal["direct_bet", "conditional", "no_bet", "insufficient_data"]
    candidates: list[MarketCandidateModel]
    recommendation_pick: RecommendationPick | None = None
    explanation: list[str]
    confidence: Literal["low", "medium", "high"]
    limitations: list[str]
    prediction_basis: str
```

`MarketCandidate` and `MarketPick` in the `MatchRecommendation` TypedDict above are new plain-dict type aliases — add them near the top of the file, right above the `MatchRecommendation` TypedDict definition (they mirror `MarketRecommendation`'s existing TypedDict pattern, whatever that currently looks like a few lines above `MatchRecommendation` — same fields as `MarketCandidateModel`/`RecommendationPick`, as a `TypedDict`, not a `BaseModel`, matching how `MarketRecommendation` already relates to `MarketRecommendationModel`).

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_agent_schema_validation.py -v`
Expected: PASS, all tests in this file.

Run: `python -m pytest tests/ -k "agent_odds_bounds or agent_conditional or agent_draw_value_edge or agent_value_edge_floor" -v`
Expected: FAIL (still on the old `markets` fixture shape) — confirms the plan-level note above; Task 3 fixes these.

- [ ] **Step 5: Commit**

```bash
git add src/agent/schema.py tests/test_agent_schema_validation.py
git commit -m "feat(agent): A88 -- MarketCandidateModel/RecommendationPick schema

New candidates+recommendation_pick shape replaces the markets array.
Other guardrail test files are expected red until Task 3 reworks them.

Co-Authored-By: Claude Sonnet 5 <noreply@anthropic.com>"
```

---

### Task 2: `resolve_recommendation_pick()` — the shared lookup

**Files:**
- Modify: `src/agent/market_resolution.py`
- Modify: `tests/test_market_resolution.py`

- [ ] **Step 1: Write the failing test**

Append to `tests/test_market_resolution.py`:

```python
from src.agent.market_resolution import resolve_recommendation_pick


def test_resolve_recommendation_pick_finds_the_matching_candidate():
    candidates = [
        {"market": "result_3way", "selection": "home", "value_edge": 0.02},
        {"market": "btts", "selection": "no", "value_edge": 0.08},
    ]
    pick = {"market": "btts", "selection": "no"}
    assert resolve_recommendation_pick(candidates, pick) == candidates[1]


def test_resolve_recommendation_pick_returns_none_for_null_pick():
    candidates = [{"market": "result_3way", "selection": "home", "value_edge": 0.02}]
    assert resolve_recommendation_pick(candidates, None) is None


def test_resolve_recommendation_pick_returns_none_when_pick_not_in_candidates():
    """The LLM pointed recommendation_pick at a market/selection it never
    actually listed in candidates -- a real, new failure mode this schema
    makes detectable for the first time."""
    candidates = [{"market": "result_3way", "selection": "home", "value_edge": 0.02}]
    pick = {"market": "btts", "selection": "yes"}
    assert resolve_recommendation_pick(candidates, pick) is None
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_market_resolution.py -k resolve_recommendation_pick -v`
Expected: FAIL with `ImportError: cannot import name 'resolve_recommendation_pick'`

- [ ] **Step 3: Implement**

Add to `src/agent/market_resolution.py`, after the existing `pick_recommended_market()`:

```python
def resolve_recommendation_pick(
    candidates: list[dict[str, Any]], pick: dict[str, Any] | None
) -> dict[str, Any] | None:
    """A88 (2026-08-31 design): the new single-recommendation schema's
    lookup -- `pick` is the LLM's own stated choice (RecommendationPick,
    src/agent/schema.py: market+selection only, no duplicated numeric
    fields), and this finds the matching full entry in `candidates`. Unlike
    pick_recommended_market() above (a max(value_edge) reduction, kept
    unchanged -- still used by app/backend/recommendation_outcomes.py's
    settlement path until that migrates in a follow-up phase), this is a
    plain equality lookup: there is nothing to rank, `pick` already names
    the one candidate that matters.

    Returns None both when `pick` is None (no recommendation offered) and
    when `pick` names a market/selection absent from `candidates` (the LLM
    pointed at something it never actually listed) -- both mean "nothing to
    recommend" to every caller, deliberately collapsed into one return
    value rather than distinguished, since the caller's reaction is
    identical either way (src/agent/schema.py's
    _resolve_recommendation_pick adds a distinguishing limitations note for
    the second case, but treats both as no-pick)."""
    if pick is None:
        return None
    for candidate in candidates:
        if candidate["market"] == pick["market"] and candidate["selection"] == pick["selection"]:
            return candidate
    return None
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_market_resolution.py -v`
Expected: PASS, all tests in this file.

- [ ] **Step 5: Commit**

```bash
git add src/agent/market_resolution.py tests/test_market_resolution.py
git commit -m "feat(agent): A88 -- resolve_recommendation_pick() lookup

Co-Authored-By: Claude Sonnet 5 <noreply@anthropic.com>"
```

---

### Task 3: Adapt the six existing guardrail functions + `_compute_target_odds` to `candidates`

**Plan correction (found during Task 1):** Task 1's implementer, of necessity, already mechanically renamed every `data.get("markets", ...)` lookup inside the seven functions below to `data.get("candidates", ...)` — required for `test_agent_schema_validation.py`'s own BUG-013 downgrade test to pass. This task's own field-rename work may already be done; verify first (`grep -n '"markets"' src/agent/schema.py`), and if so, this task's real remaining work is: (a) the `market` → `candidate` loop-variable rename for clarity (cosmetic, still worth doing — it's what this task's own diff will look like in git blame), and (b) reworking the seven test files below, which are still on the old fixture shape regardless of what Task 1 did to the source. Also: the original plan missed `tests/test_agent_target_odds.py` (covers `_compute_target_odds`, in scope here) — added below.

**Files:**
- Modify: `src/agent/schema.py` (six `_downgrade_*`/`_restrict_conditional_to_eligible_markets` functions + `_compute_target_odds`, currently `~lines 139-390`)
- Modify: `tests/test_agent_odds_bounds.py`, `tests/test_agent_value_edge_floor.py`, `tests/test_agent_draw_value_edge_floor.py`, `tests/test_agent_conditional_market_eligibility.py`, `tests/test_agent_conditional_odds_floor.py`, `tests/test_agent_conditional_odds_ceiling.py`, `tests/test_agent_target_odds.py`

Every one of these seven functions has the identical shape today: `for market in data.get("markets", []): ... market["recommendation_type"] = ...`. The only change needed in every single one is `data.get("markets", [])` → `data.get("candidates", [])` (rename the loop variable `market` → `candidate` for clarity while touching the line; every `market[...]` reference inside the loop body becomes `candidate[...]`, and every f-string `market['market']!r`/`market['selection']!r` becomes `candidate['market']!r`/`candidate['selection']!r`). No other logic changes — same thresholds, same conditions, same downgrade targets.

- [ ] **Step 1: Update the seven functions in `src/agent/schema.py`**

First run `grep -n '"markets"' src/agent/schema.py`. If it returns nothing inside these seven functions (Task 1 already did the field rename out of necessity), skip straight to the `market` → `candidate` loop-variable rename below (still do it — it's this task's own attribution in git blame, and makes the diff match what's described here). If `"markets"` is still present, do both the field rename and the variable rename together.

Apply the rename below to: `_downgrade_direct_bet_below_value_edge_floor`, `_downgrade_direct_bet_below_draw_value_edge_floor`, `_downgrade_direct_bet_with_null_odds`, `_downgrade_direct_bet_outside_odds_bounds`, `_restrict_conditional_to_eligible_markets`, `_downgrade_conditional_below_floor`, `_downgrade_conditional_above_ceiling`, `_compute_target_odds`. For example, `_downgrade_direct_bet_with_null_odds` (the shortest, fully shown as the worked example — apply the identical pattern to the other six):

```python
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
```

Every docstring's own prose can stay as-is (they describe *why* the rule exists, not the `markets`/`candidates` variable name) — only the code body changes.

- [ ] **Step 2: Rework the seven test files to the new fixture shape**

Each of the seven files listed above (six guardrail files plus `tests/test_agent_target_odds.py`) follows the exact `_VALID_MARKET`/`_VALID` (or `_DRAW_MARKET`/`_VALID` in `test_agent_draw_value_edge_floor.py`) pattern Task 1 already reworked in `tests/test_agent_schema_validation.py`. Apply the identical two substitutions to each file:

1. Rename the module-level candidate fixture (`_VALID_MARKET` or `_DRAW_MARKET`) by adding two new required keys, `"composite_score": 0.6` and `"reason": "<short reason matching the test's own scenario>"`, and rename `"markets": [...]` → `"candidates": [...]` everywhere it's built into a payload dict, adding `"recommendation_pick": {"market": <candidate's own market>, "selection": <candidate's own selection>}` alongside every such payload (matching whichever candidate that specific test is exercising — most tests build one candidate and pick it; a test asserting on an *ineligible* candidate should still set `recommendation_pick` to name it, since the whole point of these tests is checking that a picked-but-ineligible candidate gets downgraded).
2. Rename every assertion `rec["markets"][0][...]` → `rec["candidates"][0][...]`.

Fully worked example — `tests/test_agent_odds_bounds.py`'s existing `_VALID_MARKET`/`_VALID` block and its first test become:

```python
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

_VALID_PICK = {"market": "result_3way", "selection": "home"}

_VALID = {
    "match": {"home": "Arsenal", "away": "Chelsea", "date": "2026-06-15", "league": "E0"},
    "overall": "direct_bet",
    "candidates": [_VALID_CANDIDATE],
    "recommendation_pick": _VALID_PICK,
    "explanation": "Value found on the home win.",
    "confidence": "medium",
    "limitations": [],
    "prediction_basis": "team_history_and_market",
}
```

with every downstream test in that file changed by the same two substitution rules (e.g. a test building `bad_market = {**_VALID_MARKET, "current_odds": 0.9}` becomes `bad_candidate = {**_VALID_CANDIDATE, "current_odds": 0.9}`, and `{**_VALID, "markets": [bad_market]}` becomes `{**_VALID, "candidates": [bad_candidate], "recommendation_pick": _VALID_PICK}`; its assertion `rec["markets"][0]["recommendation_type"] == "conditional"` becomes `rec["candidates"][0]["recommendation_type"] == "conditional"`).

Apply this same rule to the remaining six files (including `tests/test_agent_target_odds.py`). `tests/test_agent_draw_value_edge_floor.py`'s fixture is named `_DRAW_MARKET` (not `_VALID_MARKET`) — rename it to `_DRAW_CANDIDATE` instead of `_VALID_CANDIDATE`, same two new keys, same substitution rule otherwise.

- [ ] **Step 3: Run tests to verify they pass**

Run: `python -m pytest tests/test_agent_odds_bounds.py tests/test_agent_value_edge_floor.py tests/test_agent_draw_value_edge_floor.py tests/test_agent_conditional_market_eligibility.py tests/test_agent_conditional_odds_floor.py tests/test_agent_conditional_odds_ceiling.py tests/test_agent_target_odds.py -v`
Expected: PASS, every test across all seven files.

- [ ] **Step 4: Commit**

```bash
git add src/agent/schema.py tests/test_agent_odds_bounds.py tests/test_agent_value_edge_floor.py tests/test_agent_draw_value_edge_floor.py tests/test_agent_conditional_market_eligibility.py tests/test_agent_conditional_odds_floor.py tests/test_agent_conditional_odds_ceiling.py tests/test_agent_target_odds.py
git commit -m "feat(agent): A90 -- adapt the six guardrails + target_odds to candidates

Same thresholds, same logic -- loops over candidates instead of markets.

Co-Authored-By: Claude Sonnet 5 <noreply@anthropic.com>"
```

---

### Task 4: Replace `_reconcile_overall_with_markets` with `_resolve_recommendation_pick`; update `_attach_unit_bet_multiplier`; rewire the pipeline

**Plan correction (found during Task 1):** the original plan missed `tests/test_agent_unit_bet_multiplier.py` (covers `_attach_unit_bet_multiplier`, in scope here) — added below. It follows the same `_VALID_MARKET`/`_VALID` pattern as every other file in this project; rework it with the identical two substitutions Task 3 used (`_VALID_MARKET`→`_VALID_CANDIDATE`, `markets`→`candidates`+`recommendation_pick`, `rec["markets"][0]`→`rec["candidates"][0]`).

**Files:**
- Modify: `src/agent/schema.py` (`_reconcile_overall_with_markets` → deleted, `_attach_unit_bet_multiplier` at `~line 446`, `extract_recommendation`'s pipeline at `~lines 603-613`)
- Modify: `tests/test_agent_schema.py` (or wherever `_reconcile_overall_with_markets`'s own regression tests live — grep first: `grep -rn "_reconcile_overall_with_markets\|reconcile.*overall" tests/*.py`)
- Modify: `tests/test_agent_unit_bet_multiplier.py`

- [ ] **Step 1: Write the failing tests**

Add to `tests/test_agent_schema_validation.py` (reusing `_VALID`/`_VALID_CANDIDATE`/`_wrap_json` from Task 1):

```python
def test_overall_syncs_to_the_resolved_picks_actual_type():
    """A90 (2026-08-31 design): replaces A65's _reconcile_overall_with_markets.
    The LLM claims overall='direct_bet' but the picked candidate's own price
    is outside the configured odds bounds -- Task 3's already-adapted
    _downgrade_direct_bet_outside_odds_bounds downgrades the candidate to
    'conditional'; overall must follow it, not stay stuck at the LLM's
    original, now-stale self-report."""
    candidate = {**_VALID_CANDIDATE, "current_odds": 15.0}
    data = {**_VALID, "candidates": [candidate], "recommendation_pick": _VALID_PICK}
    rec = extract_recommendation(_wrap_json(data), min_odds_threshold=1.2, max_odds_threshold=11.0)
    assert rec["overall"] == "conditional"
    assert rec["recommendation_pick"] == _VALID_PICK  # still a real pick, just re-typed


def test_pick_downgraded_to_no_bet_is_nulled_and_overall_follows():
    candidate = {**_VALID_CANDIDATE, "value_edge": -0.02}
    data = {**_VALID, "candidates": [candidate], "recommendation_pick": _VALID_PICK}
    rec = extract_recommendation(_wrap_json(data), min_value_edge=0.05)
    assert rec["overall"] == "no_bet"
    assert rec["recommendation_pick"] is None


def test_pick_naming_a_candidate_absent_from_candidates_is_treated_as_no_pick():
    dangling_pick = {"market": "btts", "selection": "yes"}
    data = {**_VALID, "recommendation_pick": dangling_pick}
    rec = extract_recommendation(_wrap_json(data))
    assert rec["recommendation_pick"] is None
    assert any("not present in candidates" in note for note in rec["limitations"])


def test_never_upgrades_insufficient_data():
    """Downgrade-only, same direction A65 already established -- a
    legitimately empty candidates list (the graph's own no-forecast
    short-circuit) with overall already 'insufficient_data' must stay that
    way, not get bumped to 'no_bet' just because there's no resolvable pick."""
    data = {**_VALID, "overall": "insufficient_data", "candidates": [], "recommendation_pick": None}
    rec = extract_recommendation(_wrap_json(data))
    assert rec["overall"] == "insufficient_data"
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_agent_schema_validation.py -k "syncs_to_the_resolved or downgraded_to_no_bet_is_nulled or absent_from_candidates or never_upgrades" -v`
Expected: FAIL — `_reconcile_overall_with_markets` still keys off `markets` (already empty/absent post-Task-1, so it's a silent no-op today), `overall` never actually syncs.

- [ ] **Step 3: Implement**

First, in `src/agent/schema.py:11`, replace the now-about-to-be-unused top-level import — `pick_recommended_market` has no remaining call site in this file after this task (`_attach_unit_bet_multiplier` below switches to the new resolver):

```python
from src.agent.market_resolution import resolve_recommendation_pick
```

In `src/agent/schema.py`, delete `_reconcile_overall_with_markets` entirely (currently `~lines 397-427`, including the `_RANK_TO_OVERALL`/`_OVERALL_RANK` module-level constants just above it — keep those two, `_resolve_recommendation_pick` below still needs them) and replace it with:

```python
def _resolve_recommendation_pick(data: dict) -> dict:
    """Replaces A65's _reconcile_overall_with_markets now that there's at
    most one real pick instead of an array to reconcile against. Runs last
    among the downgrade passes (after Task 3's seven per-candidate checks
    and A91's self-consistency check below have already mutated whichever
    candidate recommendation_pick names) -- looks that candidate up via
    resolve_recommendation_pick() and syncs `overall`/`recommendation_pick`
    to its now-possibly-downgraded state.

    A pick that no longer resolves at all (recommendation_pick is null, or
    names a market/selection absent from candidates -- the LLM pointed at
    something it never actually listed) is treated identically to a pick
    downgraded to 'no_bet': no real recommendation, overall capped at
    'no_bet' -- never claims a stronger state than the candidates actually
    support, same downgrade-only direction A65 already established. A
    dangling pick additionally gets its own limitations note, distinguishing
    "the model pointed at nothing real" from an ordinary no_bet."""
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
```

Update `_attach_unit_bet_multiplier` (currently `~lines 446-470`) to resolve via the new pointer instead of the old reduction:

```python
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
```

Update `extract_recommendation`'s pipeline (currently `~lines 603-612`) — replace the two final lines (`_reconcile_overall_with_markets`/`_attach_unit_bet_multiplier`) and insert the new resolve step after `_compute_target_odds`:

```python
        data = _downgrade_direct_bet_below_value_edge_floor(data, min_value_edge)
        data = _downgrade_direct_bet_below_draw_value_edge_floor(data, min_value_edge_result_3way_draw)
        data = _downgrade_direct_bet_with_null_odds(data)
        data = _downgrade_direct_bet_outside_odds_bounds(data, min_odds_threshold, max_odds_threshold)
        data = _restrict_conditional_to_eligible_markets(data)
        data = _downgrade_conditional_below_floor(data, min_conditional_odds_threshold)
        data = _downgrade_conditional_above_ceiling(data, max_conditional_odds_threshold)
        data = _compute_target_odds(data, min_value_edge)
        data = _downgrade_recommendation_below_top_composite_score(data)
        data = _resolve_recommendation_pick(data)
        data = _attach_unit_bet_multiplier(data)
        return data  # type: ignore[return-value]
```

(`_downgrade_recommendation_below_top_composite_score` doesn't exist yet — Task 5 adds it. Its call is wired in now so Task 5 only needs to add the function itself, not touch the pipeline again.) For this task, temporarily stub it at the bottom of the file so the suite runs:

```python
def _downgrade_recommendation_below_top_composite_score(data: dict) -> dict:
    """A91 -- implemented in Task 5. Stub: no-op until then."""
    return data
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_agent_schema_validation.py -v`
Expected: PASS, all tests in this file.

Then check for `_reconcile_overall_with_markets`'s own dedicated tests: run `grep -rln "_reconcile_overall_with_markets\|reconcile_overall" tests/*.py`. If any file other than `test_agent_schema_validation.py` references it directly (by name, e.g. importing the private function), delete those specific test functions — the behavior they covered is superseded by this task's four new tests above, and the function itself no longer exists.

Rework `tests/test_agent_unit_bet_multiplier.py` to the new fixture shape now too (same two substitutions as Task 3: `_VALID_MARKET`→`_VALID_CANDIDATE` with `composite_score`/`reason` added, `markets`→`candidates`+`recommendation_pick`, `rec["markets"][0]`→`rec["candidates"][0]`).

Run: `python -m pytest tests/ -k "agent_schema or agent_odds_bounds or agent_value_edge or agent_conditional or agent_draw or agent_target_odds or agent_unit_bet_multiplier" -v`
Expected: PASS, every test across every file this and Task 3 touched.

- [ ] **Step 5: Commit**

```bash
git add src/agent/schema.py tests/test_agent_schema_validation.py tests/test_agent_unit_bet_multiplier.py
git commit -m "feat(agent): A90 -- _resolve_recommendation_pick replaces A65's reconcile

overall now syncs directly from the resolved pick's own state instead of
scanning an array for the strongest surviving market.

Co-Authored-By: Claude Sonnet 5 <noreply@anthropic.com>"
```

---

### Task 5: Self-consistency guardrail — `_downgrade_recommendation_below_top_composite_score`

**Files:**
- Create: `tests/test_agent_self_consistency_guardrail.py`
- Modify: `src/agent/schema.py` (replace Task 4's stub)

- [ ] **Step 1: Write the failing test**

```python
"""A91 (2026-08-31 design): the LLM self-reports a composite_score per
candidate meant to balance value_edge against ml_probability. Code can't
verify the *number* itself -- there's no fixed formula -- but it can catch
self-contradiction: did the model's own stated pick actually have the best
score among its own listed candidates?"""

from __future__ import annotations

import json

from src.agent.schema import extract_recommendation

_HOME = {
    "market": "result_3way", "selection": "home", "recommendation_type": "direct_bet",
    "current_odds": 2.1, "min_odds": 1.8, "ml_probability": 0.55, "implied_probability": 0.48,
    "value_edge": 0.07, "composite_score": 0.4, "reason": "Modest edge, high uncertainty.",
}
_BTTS = {
    "market": "btts", "selection": "no", "recommendation_type": "direct_bet",
    "current_odds": 2.2, "min_odds": 1.8, "ml_probability": 0.6, "implied_probability": 0.45,
    "value_edge": 0.15, "composite_score": 0.8, "reason": "Large edge with strong hit probability.",
}

_BASE = {
    "match": {"home": "Arsenal", "away": "Chelsea", "date": "2026-06-15", "league": "E0"},
    "overall": "direct_bet",
    "explanation": "Value found.",
    "confidence": "medium",
    "limitations": [],
    "prediction_basis": "team_history_and_market",
}


def _wrap_json(data: dict) -> str:
    return f"Some reasoning here.\n\n```json\n{json.dumps(data)}\n```"


def test_pick_with_the_top_composite_score_is_unaffected():
    data = {**_BASE, "candidates": [_HOME, _BTTS], "recommendation_pick": {"market": "btts", "selection": "no"}}
    rec = extract_recommendation(_wrap_json(data))
    assert rec["recommendation_pick"] == {"market": "btts", "selection": "no"}
    assert rec["overall"] == "direct_bet"


def test_pick_with_a_lower_composite_score_than_a_rejected_candidate_is_downgraded():
    data = {**_BASE, "candidates": [_HOME, _BTTS], "recommendation_pick": {"market": "result_3way", "selection": "home"}}
    rec = extract_recommendation(_wrap_json(data))
    assert rec["recommendation_pick"] is None
    assert rec["overall"] == "no_bet"
    assert any("composite_score" in note for note in rec["limitations"])


def test_a_no_bet_rejected_candidates_own_higher_score_does_not_count():
    """A no_bet candidate was already disqualified by an earlier guardrail --
    it was never a real alternative, so out-scoring it isn't a
    contradiction."""
    disqualified = {**_BTTS, "recommendation_type": "no_bet", "composite_score": 0.9}
    data = {**_BASE, "candidates": [_HOME, disqualified], "recommendation_pick": {"market": "result_3way", "selection": "home"}}
    rec = extract_recommendation(_wrap_json(data))
    assert rec["recommendation_pick"] == {"market": "result_3way", "selection": "home"}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_agent_self_consistency_guardrail.py -v`
Expected: FAIL — `_downgrade_recommendation_below_top_composite_score` is still Task 4's no-op stub, so the second test's pick is never downgraded.

- [ ] **Step 3: Implement**

Replace the stub at the bottom of `src/agent/schema.py`:

```python
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
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_agent_self_consistency_guardrail.py -v`
Expected: PASS, all three tests.

Run: `python -m pytest tests/test_agent_schema_validation.py tests/test_market_resolution.py -v`
Expected: PASS — confirms Task 4's wiring plus this task's real implementation didn't regress anything from Tasks 1-4.

- [ ] **Step 5: Commit**

```bash
git add src/agent/schema.py tests/test_agent_self_consistency_guardrail.py
git commit -m "feat(agent): A91 -- self-consistency guardrail on composite_score

Downgrades to no_bet when a rejected candidate self-reports a higher
composite_score than the actual pick.

Co-Authored-By: Claude Sonnet 5 <noreply@anthropic.com>"
```

---

### Task 6: Prompt changes — all 4 files

**Files:**
- Modify: `config/prompts/agent_v1.txt`, `config/prompts/agent_v1_conservative.txt`, `config/prompts/agent_v1_balanced.txt`, `config/prompts/agent_v1_aggressive.txt`
- Modify: `tests/test_agent_prompt_thresholds.py`

All six guardrail thresholds are **already** templated into every prompt file (`{{MIN_VALUE_EDGE}}`, `{{MIN_ODDS_THRESHOLD}}`, `{{MAX_ODDS_THRESHOLD}}`, `{{MIN_CONDITIONAL_ODDS_THRESHOLD}}`, `{{MAX_CONDITIONAL_ODDS_CLAUSE}}`, `{{DRAW_VALUE_EDGE_CLAUSE}}` — confirmed live in `src/agent/graph.py:88-131` and covered by `tests/test_agent_prompt_thresholds.py`'s existing five tests). **This task does not need to add threshold templating** — the design spec's claim that it does is corrected in Task 7. What's actually needed is new instruction text and the JSON output shape.

- [ ] **Step 1: Write the failing tests**

Add to `tests/test_agent_prompt_thresholds.py` (reusing whatever config-building helper the existing five tests already use — check the top of the file for the pattern, e.g. a `_config(**overrides)` helper):

```python
def test_prompt_output_format_uses_candidates_and_recommendation_pick():
    text = _load_system_prompt(_config())
    assert '"candidates"' in text
    assert '"recommendation_pick"' in text
    assert '"markets"' not in text


def test_prompt_mentions_composite_score_balance_instruction():
    text = _load_system_prompt(_config())
    assert "composite_score" in text
    assert "hit probability" in text or "ml_probability" in text.lower()
```

(If `_load_system_prompt`/`_config` aren't already imported/defined at the top of this test file under those exact names, use whichever names the file's existing five tests actually call — read the file first and match its real helper names.)

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_agent_prompt_thresholds.py -k "candidates_and_recommendation_pick or composite_score_balance" -v`
Expected: FAIL — the prompt files still say `"markets"` and have no `composite_score` mention.

- [ ] **Step 3: Update the prompt files**

In `config/prompts/agent_v1.txt`, replace the `## Value Calculation` section's final two bullets (currently ending at `... a "conditional" call outside this range will be downgraded to "no_bet" automatically.`) by appending two new bullets:

```
- After scoring every market with a real current price, weigh them against each other before picking one: a smaller value_edge backed by a materially higher ml_probability can be the stronger pick over a larger, noisier edge — don't default to whichever number is biggest. Record this balance as each candidate's composite_score.
- recommendation_pick must name a candidate whose recommendation_type/value_edge/odds already clear every rule above — the code enforces this, a pick that doesn't will be downgraded to no_bet automatically. A candidate you already know is ineligible can still appear in candidates for transparency, just not as the pick. If nothing clears every rule, leave recommendation_pick null and set overall to "no_bet".
```

Replace the `## Output Format` section's intro paragraph (the `explanation` guidance, currently starting `` `explanation` is an array of short bullet strings...``) — append one sentence to its existing text: `If you're choosing between two candidates that both cleared the bar, say why the one you picked won (its balance of edge and hit probability) — don't just restate its number.`

Replace the JSON block:

```json
{
  "match": {
    "home": "<team name>",
    "away": "<team name>",
    "date": "YYYY-MM-DD",
    "league": "<league code or 'international'>"
  },
  "overall": "<direct_bet | conditional | no_bet | insufficient_data>",
  "candidates": [
    {
      "market": "<result_3way | btts | total_goals | home_corners | away_corners>",
      "selection": "<home | draw | away | yes | no | over_2.5 | under_2.5>",
      "recommendation_type": "<direct_bet | conditional | no_bet>",
      "current_odds": 0.0,
      "min_odds": 0.0,
      "ml_probability": 0.0,
      "implied_probability": 0.0,
      "value_edge": 0.0,
      "composite_score": 0.0,
      "reason": "<one line: why this candidate won or lost>"
    }
  ],
  "recommendation_pick": {"market": "<...>", "selection": "<...>"},
  "explanation": ["<one aspect of your reasoning per item, see below>"],
  "confidence": "<low | medium | high>",
  "limitations": ["<what could not be assessed>"],
  "prediction_basis": "<team_history_and_market | market_odds_only | partial>"
}
```

Add one line directly above the JSON block's own fenced-code-block opening: `List one candidates entry for every market with a real current price you evaluated, including ones you're rejecting — not just the one you pick. Set recommendation_pick to null (and overall to "no_bet") if nothing clears every rule above.`

Apply the identical three edits (two new Value Calculation bullets, the explanation-guidance sentence, the JSON block replacement, the line above it) to `agent_v1_conservative.txt`, `agent_v1_balanced.txt`, and `agent_v1_aggressive.txt` — each file is byte-identical to `agent_v1.txt` in every section this task touches (confirmed: the only difference between the four files is one extra bullet each already has after the existing Value Calculation section, untouched by this task).

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_agent_prompt_thresholds.py -v`
Expected: PASS, every test in the file (the five pre-existing ones plus the two new ones).

- [ ] **Step 5: Commit**

```bash
git add config/prompts/agent_v1.txt config/prompts/agent_v1_conservative.txt config/prompts/agent_v1_balanced.txt config/prompts/agent_v1_aggressive.txt tests/test_agent_prompt_thresholds.py
git commit -m "feat(agent): A89 -- prompt output shape + balance instruction

candidates/recommendation_pick replace markets in all 4 prompt files, plus
the edge-vs-hit-probability balance guidance and pre-commit eligibility
instruction. Guardrail thresholds were already templated -- no change
needed there.

Co-Authored-By: Claude Sonnet 5 <noreply@anthropic.com>"
```

---

### Task 7: Full regression, spec correction, user-story completion notes

**Files:**
- Modify: `docs/superpowers/specs/2026-08-31-single-market-recommendation-design.md`
- Modify: `documents/agent_user_stories.md`

- [ ] **Step 1: Run the full suite**

Run: `python -m pytest tests/ -v 2>&1 | tail -60`
Expected: every test passes except the pre-existing, unrelated `app/backend/tests/test_fixtures_endpoint.py` failures already tracked across this whole session's other stories (not touched by this plan). If anything else fails, stop and fix it before continuing — do not proceed to Step 2 with red tests.

Run: `python -m pytest app/backend/tests/ -v 2>&1 | tail -30`
Expected: same — clean except the known pre-existing `test_fixtures_endpoint.py` failures. This plan never touches `app/backend/`, so nothing here should have moved.

- [ ] **Step 2: Correct the design spec's inaccurate templating claim**

The spec's "Prompt changes" section currently says templating "Extends the pattern A84/A85 already established... to all six." Task 6 found all six were already templated before this project started. In `docs/superpowers/specs/2026-08-31-single-market-recommendation-design.md`, replace the "Prompt changes" section's first bullet:

```
1. **Every guardrail threshold, templated in as a real number.** Extends the pattern A84/A85 already established (`{{DRAW_VALUE_EDGE_CLAUSE}}`) to all six: min/max odds bounds (A29), conditional floor/ceiling (A66/A84), the base value-edge floor, and the draw-specific floor (A85). New instruction alongside them: a candidate that would fail one of these checks may still appear in `candidates` for transparency, but must not be the `recommendation_pick` — pick a different eligible one, or if none clear every check, set `overall` to `no_bet` and leave `recommendation_pick` null.
```

with:

```
1. **Every guardrail threshold, already a real number.** Corrected during implementation (2026-09-01): all six thresholds (`{{MIN_VALUE_EDGE}}`, `{{MIN_ODDS_THRESHOLD}}`, `{{MAX_ODDS_THRESHOLD}}`, `{{MIN_CONDITIONAL_ODDS_THRESHOLD}}`, `{{MAX_CONDITIONAL_ODDS_CLAUSE}}`, `{{DRAW_VALUE_EDGE_CLAUSE}}`) were already templated into every prompt file before this project started (`src/agent/graph.py:_load_system_prompt`) — this spec's original claim that only two were templated, extending A84/A85's pattern to the rest, was inaccurate. What's actually new: an explicit pre-commit eligibility instruction alongside the existing thresholds — a candidate that would fail one of these checks may still appear in `candidates` for transparency, but must not be the `recommendation_pick`; pick a different eligible one, or if none clear every check, set `overall` to `no_bet` and leave `recommendation_pick` null.
```

- [ ] **Step 3: Update Phase 29's stories in `documents/agent_user_stories.md`**

Change `A88`/`A89`/`A90`/`A91`'s `future` status to `completed`, and append a completion note to each `Comments` cell (after the existing `Size ... · Depends on: ...` text), summarizing what actually shipped. For `A88`:

```
**Completion notes (2026-09-01):** `MarketCandidateModel`/`RecommendationPick`/updated `MatchRecommendationModel` in `src/agent/schema.py`, exactly as designed. TDD: `tests/test_agent_schema_validation.py` extended with the new fixture shape plus 4 new tests (composite_score required, recommendation_pick validation, null pick). Full suite: [paste the real passing count from Step 1] passed, zero regressions.
```

Write equivalent, accurate completion notes for `A89` (prompt changes — note the corrected templating scope from Step 2), `A90` (the seven adapted guardrails + `_resolve_recommendation_pick`), and `A91` (the self-consistency guardrail) — each should name its own real test file(s) and reference the real passing counts from Step 1, matching this codebase's own completion-note convention throughout `documents/agent_user_stories.md`.

- [ ] **Step 4: Commit**

```bash
git add docs/superpowers/specs/2026-08-31-single-market-recommendation-design.md documents/agent_user_stories.md
git commit -m "docs(agent): A88-A91 completion notes + spec correction

Corrects the design spec's inaccurate claim that guardrail thresholds
needed new prompt templating -- all six were already there. Marks Phase
29's four stories completed with real test/suite evidence.

Co-Authored-By: Claude Sonnet 5 <noreply@anthropic.com>"
```
