# LLM-Only Recommendations for UCL/UEL Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Extend FPAI's recommendations beyond ML-modeled leagues by adding a new `llm_only` competition tier (no trained model; the agent reasons from market odds + research) and onboarding UEFA Champions League (UCL) and Europa League (UEL) under it.

**Architecture:** A third tier alongside `general_purpose`/`competition_specific` in the existing competition registry, routed to a new no-model forecast tool and a separate tier-specific prompt/`AgentConfig`, reusing the existing agent graph, backtest harness, and lesson-approval pipeline throughout. Live fixtures come from fotmob (already unfiltered for cup competitions); odds (live and historical) come from OddsPapi, newly extended to cover UCL/UEL tournament IDs.

**Tech Stack:** Python (FastAPI backend, LangGraph agent, DuckDB), TypeScript/React frontend. No new dependencies.

**Design doc:** `docs/superpowers/specs/2026-10-06-llm-only-cup-recommendations-design.md`

---

## Phase 1: Registry & Tiering

### Task 1: Add `llm_only` to `VALID_TIERS`

**Files:**
- Modify: `src/logic/competition_registry.py:39`
- Test: `tests/test_competition_registry.py`

- [ ] **Step 1: Write the failing test**

Add to `tests/test_competition_registry.py`:

```python
def test_valid_tiers_includes_llm_only() -> None:
    from src.logic.competition_registry import VALID_TIERS
    assert "llm_only" in VALID_TIERS


def test_resolve_feature_subset_for_tier_rejects_llm_only() -> None:
    """llm_only has no feature subset -- no model is ever trained for it.
    Training code must fail loudly if ever pointed at this tier, not
    silently train a model nobody will use."""
    with pytest.raises(ValueError, match="Unknown tier"):
        resolve_feature_subset_for_tier("llm_only")
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_competition_registry.py -k "llm_only" -v`
Expected: FAIL (`"llm_only" in VALID_TIERS` is `False`)

- [ ] **Step 3: Implement**

In `src/logic/competition_registry.py`, change line 39:

```python
VALID_TIERS = ("general_purpose", "competition_specific", "llm_only")
```

No change needed to `resolve_feature_subset_for_tier` (`competition_registry.py:190-196`) — it already raises `ValueError` for any tier it doesn't explicitly handle, which now correctly includes `"llm_only"` by omission. Add one line to its docstring/comment noting this is deliberate:

```python
def resolve_feature_subset_for_tier(tier: str) -> list[str] | None:
    """Return the feature subset for a tier, or None to use the full schema.yaml list.

    llm_only has no feature subset by design -- no model is ever trained
    or loaded for that tier, so this deliberately falls through to the
    final ValueError rather than gaining its own branch."""
    if tier == "general_purpose":
        return list(GENERAL_PURPOSE_FEATURES)
    if tier == "competition_specific":
        return None
    raise ValueError(f"Unknown tier '{tier}'. Must be one of {VALID_TIERS}.")
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_competition_registry.py -k "llm_only" -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/logic/competition_registry.py tests/test_competition_registry.py
git commit -m "feat(registry): add llm_only to VALID_TIERS"
```

### Task 2: Register UCL and UEL in `config/competitions.yaml`

**Files:**
- Modify: `config/competitions.yaml`
- Test: `tests/test_competition_registry.py`

- [ ] **Step 1: Write the failing test**

```python
def test_ucl_uel_registered_as_llm_only() -> None:
    for code in ("UCL", "UEL"):
        definition = get_competition_definition(code)
        assert definition.tier == "llm_only"
        assert definition.display_enabled is False  # not shown until launch gate passes


def test_ucl_uel_excluded_from_context_keys() -> None:
    """llm_only competitions have no model_selection.yaml bucket -- they
    must never appear in the contexts list training/promotion code reads."""
    from src.logic.competition_registry import list_context_keys
    contexts = list_context_keys()
    assert "UCL" not in contexts
    assert "UEL" not in contexts
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_competition_registry.py -k "ucl_uel" -v`
Expected: FAIL (`Unknown competition 'UCL'`)

- [ ] **Step 3: Implement**

Append to `config/competitions.yaml`, after the `international` entry:

```yaml
  UCL:
    competition_id: UCL
    tier: llm_only
    league_code: UCL
    # Not shown to users until the Phase 9 backtest-ROI launch gate passes
    # (docs/superpowers/specs/2026-10-06-llm-only-cup-recommendations-design.md
    # Section 8). No ML model is ever trained or loaded for this tier --
    # see resolve_feature_subset_for_tier's llm_only branch (none exists by
    # design) and src/agent/tools.py's forecast_llm_only.
    display_enabled: false
    enabled_feature_groups: []
    player_data_sources: []
  UEL:
    competition_id: UEL
    tier: llm_only
    league_code: UEL
    display_enabled: false
    enabled_feature_groups: []
    player_data_sources: []
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_competition_registry.py -k "ucl_uel" -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add config/competitions.yaml tests/test_competition_registry.py
git commit -m "feat(registry): register UCL/UEL as llm_only competitions"
```

---

## Phase 2: Schema Changes (`src/agent/schema.py`)

### Task 3: Make `ml_probability`/`value_edge`/`composite_score` nullable for a new llm_only validation model

**Why a separate model, not widening the existing one:** `MarketCandidateModel.ml_probability`/`value_edge`/`composite_score` are currently required `float` fields (`schema.py:124,126,140`). Widening them to `float | None` globally would let a malformed response for the *existing* 6 leagues silently pass validation with `None` in these fields, then crash with an unhandled `TypeError` inside `_downgrade_direct_bet_below_value_edge_floor` (`candidate["value_edge"] >= min_value_edge` — `None >= 0.05` raises). A sibling model keeps the existing tiers' validation behavior byte-for-byte unchanged.

**Files:**
- Modify: `src/agent/schema.py`
- Test: `tests/test_agent_schema.py`

- [ ] **Step 1: Write the failing test**

Add to `tests/test_agent_schema.py`:

```python
_VALID_LLM_ONLY = {
    "match": {"home": "Real Madrid", "away": "Bayern Munich", "date": "2026-10-20", "league": "UCL"},
    "overall": "no_bet",
    "candidates": [],
    "recommendation_pick": None,
    "explanation": "No value found.",
    "confidence": "medium",
    "limitations": [],
    "prediction_basis": "llm_only_no_model",
}


def test_llm_only_candidate_allows_null_ml_probability_and_value_edge():
    data = {
        **_VALID_LLM_ONLY,
        "candidates": [{
            "market": "result_3way", "selection": "home", "recommendation_type": "no_bet",
            "current_odds": 1.8, "min_odds": 0.0,
            "ml_probability": None, "implied_probability": 0.55, "value_edge": None,
            "composite_score": None, "reason": "Thin evidence on both sides tonight.",
        }],
    }
    rec = extract_recommendation(_wrap_json(data), tier="llm_only")
    assert rec["candidates"][0]["ml_probability"] is None
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_agent_schema.py -k "llm_only_candidate" -v`
Expected: FAIL (`extract_recommendation() got an unexpected keyword argument 'tier'`)

- [ ] **Step 3: Implement — add the sibling Pydantic model**

In `src/agent/schema.py`, immediately after `MarketCandidateModel` (ends at line 144):

```python
class LLMOnlyMarketCandidateModel(BaseModel):
    """Sibling of MarketCandidateModel for tier='llm_only': ml_probability,
    value_edge, and composite_score are genuinely absent (no ML model backs
    this tier), not just sometimes-missing -- Optional here, not just
    defaulted, so the LLM's own JSON literally writing `null` validates.
    See this file's module-level note on why this is a SEPARATE model
    rather than widening MarketCandidateModel itself: that would let a
    malformed existing-tier response silently pass with None in these
    fields and then crash the value_edge-floor downgrade pass, which
    genuinely requires them to be real floats."""

    market: Literal["result_3way", "btts", "total_goals", "home_goals", "away_goals"]
    selection: Literal["home", "draw", "away", "yes", "no", "over_2.5", "under_2.5", "over_9.5", "under_9.5", "over_1.5", "under_1.5"]
    recommendation_type: Literal["direct_bet", "conditional", "no_bet"]
    current_odds: float | None
    min_odds: float = 0.0
    ml_probability: float | None = None
    implied_probability: float
    value_edge: float | None = None
    target_odds: float | None = None
    composite_score: float | None = None
    reason: str


class LLMOnlyMatchRecommendationModel(BaseModel):
    """Sibling of MatchRecommendationModel for tier='llm_only' -- same
    shape, candidates validated against LLMOnlyMarketCandidateModel
    instead."""

    match: dict
    overall: Literal["direct_bet", "conditional", "no_bet", "insufficient_data"]
    candidates: list[LLMOnlyMarketCandidateModel]
    recommendation_pick: RecommendationPickModel | None = None
    explanation: list[str]
    confidence: Literal["low", "medium", "high"]
    limitations: list[str]
    prediction_basis: str
```

- [ ] **Step 4: Implement — thread `tier` through `extract_recommendation` and branch validation**

In `src/agent/schema.py`, change the signature at line 985:

```python
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
    tier: str = "competition_specific",
) -> MatchRecommendation:
```

Add one line to the docstring: `tier: "llm_only" validates candidates against LLMOnlyMarketCandidateModel (nullable edge fields) and runs the llm_only-specific downgrade chain (Task 4) instead of the value-edge-based one. Any other value (the default) is today's unchanged behavior.`

Change the validation call at line 1070-1074 from:

```python
        try:
            MatchRecommendationModel.model_validate(data)
        except ValidationError as exc:
            last_error = f"field validation failed: {exc}"
            continue
```

to:

```python
        try:
            if tier == "llm_only":
                LLMOnlyMatchRecommendationModel.model_validate(data)
            else:
                MatchRecommendationModel.model_validate(data)
        except ValidationError as exc:
            last_error = f"field validation failed: {exc}"
            continue
```

- [ ] **Step 5: Run test to verify it passes**

Run: `python -m pytest tests/test_agent_schema.py -k "llm_only_candidate" -v`
Expected: PASS

- [ ] **Step 6: Run the full schema test suite to confirm no regression**

Run: `python -m pytest tests/test_agent_schema.py tests/test_agent_schema_validation.py -v`
Expected: all pre-existing tests still PASS (the default `tier="competition_specific"` preserves today's validation path exactly)

- [ ] **Step 7: Commit**

```bash
git add src/agent/schema.py tests/test_agent_schema.py
git commit -m "feat(schema): add llm_only validation model with nullable edge fields"
```

### Task 4: New confidence-gated downgrade chain for `llm_only`

**Files:**
- Modify: `src/agent/schema.py`
- Test: `tests/test_agent_schema_validation.py`

- [ ] **Step 1: Write the failing tests**

Add to `tests/test_agent_schema_validation.py` (import `extract_recommendation` and the `_wrap_json`/`_VALID_LLM_ONLY` helpers from `tests/test_agent_schema.py`, or redefine locally matching that file's pattern):

```python
def _llm_only_candidate(recommendation_type: str, market="result_3way", selection="home", odds=1.8) -> dict:
    return {
        "market": market, "selection": selection, "recommendation_type": recommendation_type,
        "current_odds": odds, "min_odds": 0.0,
        "ml_probability": None, "implied_probability": 0.55, "value_edge": None,
        "composite_score": None, "reason": "test",
    }


def test_llm_only_direct_bet_downgraded_below_high_confidence():
    data = {
        **_VALID_LLM_ONLY, "overall": "direct_bet", "confidence": "medium",
        "candidates": [_llm_only_candidate("direct_bet")],
        "recommendation_pick": {"market": "result_3way", "selection": "home"},
    }
    rec = extract_recommendation(_wrap_json(data), tier="llm_only")
    assert rec["candidates"][0]["recommendation_type"] == "no_bet"
    assert rec["overall"] == "no_bet"


def test_llm_only_direct_bet_survives_at_high_confidence():
    data = {
        **_VALID_LLM_ONLY, "overall": "direct_bet", "confidence": "high",
        "candidates": [_llm_only_candidate("direct_bet")],
        "recommendation_pick": {"market": "result_3way", "selection": "home"},
    }
    rec = extract_recommendation(_wrap_json(data), tier="llm_only")
    assert rec["candidates"][0]["recommendation_type"] == "direct_bet"


def test_llm_only_conditional_downgraded_at_low_confidence():
    data = {
        **_VALID_LLM_ONLY, "overall": "conditional", "confidence": "low",
        "candidates": [_llm_only_candidate("conditional", market="btts", selection="yes", odds=2.0)],
        "recommendation_pick": {"market": "btts", "selection": "yes"},
    }
    rec = extract_recommendation(_wrap_json(data), tier="llm_only")
    assert rec["candidates"][0]["recommendation_type"] == "no_bet"


def test_llm_only_unit_bet_multiplier_is_confidence_tier_based():
    data = {
        **_VALID_LLM_ONLY, "overall": "direct_bet", "confidence": "high",
        "candidates": [_llm_only_candidate("direct_bet")],
        "recommendation_pick": {"market": "result_3way", "selection": "home"},
    }
    rec = extract_recommendation(_wrap_json(data), tier="llm_only")
    assert rec["unit_bet_multiplier"] == 1.0  # high confidence -> full flat unit, not Kelly-on-null-edge
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_agent_schema_validation.py -k "llm_only" -v`
Expected: FAIL (direct_bet at medium confidence is not downgraded yet — the existing value-edge chain runs unconditionally regardless of `tier`)

- [ ] **Step 3: Implement — two new functions**

In `src/agent/schema.py`, after `_downgrade_conditional_above_ceiling` (ends at line ~490, right before `_promote_favorite_to_conditional_for_live_wait`):

```python
def _downgrade_llm_only_below_confidence_floor(data: dict) -> dict:
    """llm_only tier's replacement for the value-edge floor: there is no
    ML probability to measure an edge against, so confidence (the LLM's
    own self-reported evidence-quality tier) is the gate instead. direct_bet
    requires 'high'; conditional requires at least 'medium'. Mirrors
    _downgrade_direct_bet_below_value_edge_floor's downgrade-reason-logging
    convention."""
    confidence = data.get("confidence")
    limitations = list(data.get("limitations") or [])
    for candidate in data.get("candidates", []):
        rec_type = candidate["recommendation_type"]
        if rec_type == "direct_bet" and confidence != "high":
            candidate["recommendation_type"] = "no_bet"
            limitations.append(
                f"Downgraded {candidate['market']!r} from direct_bet to no_bet: confidence "
                f"{confidence!r} is below the 'high' bar required with no ML model backing this tier."
            )
        elif rec_type == "conditional" and confidence == "low":
            candidate["recommendation_type"] = "no_bet"
            limitations.append(
                f"Downgraded {candidate['market']!r} from conditional to no_bet: confidence 'low' "
                "is below the 'medium' bar required with no ML model backing this tier."
            )
    data["limitations"] = limitations
    return data


_LLM_ONLY_CONFIDENCE_STAKE = {"high": 1.0, "medium": 0.5, "low": None}


def _attach_llm_only_stake_tier(data: dict) -> dict:
    """llm_only replacement for _attach_unit_bet_multiplier: there is no
    value_edge to Kelly-size from, so stake is a flat multiple of the same
    UNIT_BET_BASELINE_FRACTION reference, keyed directly off the
    self-reported confidence tier that already gated recommendation_type
    above -- not a fabricated edge-derived number."""
    picked = resolve_recommendation_pick(data.get("candidates") or [], data.get("recommendation_pick"))
    if picked is None or picked.get("recommendation_type") == "no_bet":
        data["unit_bet_multiplier"] = None
    else:
        data["unit_bet_multiplier"] = _LLM_ONLY_CONFIDENCE_STAKE.get(data.get("confidence"))
    return data
```

- [ ] **Step 4: Implement — branch the downgrade chain**

In `src/agent/schema.py`, change the block at lines 1089-1108 from:

```python
        for candidate in data.get("candidates", []):
            candidate["initial_recommendation_type"] = candidate["recommendation_type"]

        data = _downgrade_direct_bet_below_value_edge_floor(data, min_value_edge)
        data = _downgrade_direct_bet_below_draw_value_edge_floor(data, min_value_edge_result_3way_draw)
        data = _downgrade_direct_bet_btts_no(data)
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
        data = _downgrade_pick_dominated_by_another_candidate(data)
        data = _prefer_higher_probability_conditional_pick(data)
        data = _resolve_recommendation_pick(data)
        data = _attach_unit_bet_multiplier(data)
        return data  # type: ignore[return-value]
```

to:

```python
        for candidate in data.get("candidates", []):
            candidate["initial_recommendation_type"] = candidate["recommendation_type"]

        if tier == "llm_only":
            # No ML model backs this tier -- skip every value_edge/
            # ml_probability/composite_score-dependent pass (they'd operate
            # on None and either no-op or crash) and the btts/no model-bias
            # suppression (US#210, specific to this project's trained
            # classifier's own calibration quirk -- meaningless with no
            # model). Odds-bounds/market-eligibility checks are pure
            # trading-risk constraints, independent of edge source, and
            # still apply.
            data = _downgrade_direct_bet_with_null_odds(data)
            data = _downgrade_direct_bet_outside_odds_bounds(data, min_odds_threshold, max_odds_threshold)
            data = _restrict_conditional_to_eligible_markets(data)
            data = _downgrade_conditional_below_floor(data, min_conditional_odds_threshold)
            data = _downgrade_conditional_above_ceiling(data, max_conditional_odds_threshold)
            data = _downgrade_llm_only_below_confidence_floor(data)
            data = _resolve_recommendation_pick(data)
            data = _attach_llm_only_stake_tier(data)
            return data  # type: ignore[return-value]

        data = _downgrade_direct_bet_below_value_edge_floor(data, min_value_edge)
        data = _downgrade_direct_bet_below_draw_value_edge_floor(data, min_value_edge_result_3way_draw)
        data = _downgrade_direct_bet_btts_no(data)
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
        data = _downgrade_pick_dominated_by_another_candidate(data)
        data = _prefer_higher_probability_conditional_pick(data)
        data = _resolve_recommendation_pick(data)
        data = _attach_unit_bet_multiplier(data)
        return data  # type: ignore[return-value]
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `python -m pytest tests/test_agent_schema_validation.py -k "llm_only" -v`
Expected: PASS

- [ ] **Step 6: Run the full schema + schema_validation suites to confirm no regression**

Run: `python -m pytest tests/test_agent_schema.py tests/test_agent_schema_validation.py -v`
Expected: all pass, same count as before this task plus the new llm_only tests

- [ ] **Step 7: Commit**

```bash
git add src/agent/schema.py tests/test_agent_schema_validation.py
git commit -m "feat(schema): confidence-gated downgrade chain for llm_only tier"
```

---

## Phase 3: New Forecast Tool & Routing

### Task 5: `forecast_llm_only` tool (`src/agent/tools.py`)

**Files:**
- Modify: `src/agent/tools.py`
- Test: `tests/test_agent_tools_snapshot.py`

- [ ] **Step 1: Write the failing test**

Add to `tests/test_agent_tools_snapshot.py` (match that file's existing style of calling the `_impl` function directly, bypassing the `@tool`/snapshot wrapper):

```python
def test_forecast_llm_only_never_loads_a_model():
    """llm_only has no model_selection.yaml context -- this must not even
    attempt ForecastService.forecast_upcoming, which would raise
    FileNotFoundError for an unregistered context."""
    from src.agent.tools import _forecast_llm_only_impl
    raw = _forecast_llm_only_impl(
        home_team="Real Madrid", away_team="Bayern Munich", date="2026-10-20",
        league="UCL", odds_h=2.1, odds_d=3.4, odds_a=3.3,
    )
    result = json.loads(raw)
    assert "error" not in result
    assert result["data_quality"]["prediction_basis"] == "llm_only_no_model"
    assert result["data_quality"]["tier"] == "llm_only"
    assert 0.0 < result["market_odds"]["implied_home"] < 1.0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_agent_tools_snapshot.py -k "forecast_llm_only" -v`
Expected: FAIL (`ImportError: cannot import name '_forecast_llm_only_impl'`)

- [ ] **Step 3: Implement**

In `src/agent/tools.py`, after `_forecast_international_impl` (ends at line 378) and before the `@tool def web_search` region:

```python
def _forecast_llm_only_impl(
    home_team: str,
    away_team: str,
    date: str,
    league: str,
    odds_h: float,
    odds_d: float,
    odds_a: float,
) -> str:
    """No ML model is ever consulted for this tier (src/logic/
    competition_registry.py's llm_only tier) -- unlike forecast_league,
    there is no FileNotFoundError retry chain here, because there is
    nothing to retry into; this tool IS the fallback destination, not
    another link in it. Reuses ForecastService's existing, already
    model-independent odds-to-market-feature math (the same Poisson-based
    lambda/BTTS-prior computation the general_purpose tier's models
    consume as ML input) directly as LLM-facing context instead."""
    try:
        from src.forecast.forecast_service import ForecastService
        mkt_row = ForecastService._compute_mkt_features_from_odds(odds_h, odds_d, odds_a)
        mkt = mkt_row.iloc[0].to_dict()
        result = {
            "match": {"home": home_team, "away": away_team, "date": date, "league": league},
            "market_odds": {
                "implied_home": mkt["MKT_IMPLIED_HOME"],
                "implied_draw": mkt["MKT_IMPLIED_DRAW"],
                "implied_away": mkt["MKT_IMPLIED_AWAY"],
                "overround": mkt["MKT_OVERROUND"],
                "implied_lambda_total_goals": mkt["MKT_LAMBDA_TOTAL"],
                "implied_lambda_home_goals": mkt["MKT_LAMBDA_HOME"],
                "implied_lambda_away_goals": mkt["MKT_LAMBDA_AWAY"],
                "implied_btts_probability": mkt["MKT_POISSON_BTTS_PROB"],
            },
            "data_quality": {"prediction_basis": "llm_only_no_model", "tier": "llm_only"},
        }
        return json.dumps(result, default=str)
    except Exception as exc:
        return json.dumps({"error": str(exc), "status": "tool_error"})


@tool
def forecast_llm_only(
    home_team: str,
    away_team: str,
    date: str,
    league: str,
    odds_h: float,
    odds_d: float,
    odds_a: float,
) -> str:
    """Get market-implied context for a match with no trained ML model
    (e.g. UEFA Champions League, Europa League). Returns odds-derived
    implied probabilities and Poisson-based goal/BTTS priors -- NOT an ML
    probability. Use when resolve_competition's tier is 'llm_only'.

    Args:
        home_team: Home team name.
        away_team: Away team name.
        date: Match date in YYYY-MM-DD format.
        league: Competition code, e.g. 'UCL', 'UEL'.
        odds_h: Home win decimal odds from bookmaker.
        odds_d: Draw decimal odds.
        odds_a: Away win decimal odds.

    Returns JSON with market_odds (implied probabilities and Poisson priors)
    and data_quality.prediction_basis='llm_only_no_model'."""
    return _snapshot_store.wrap("forecast_llm_only", _forecast_llm_only_impl)(
        home_team=home_team, away_team=away_team, date=date, league=league,
        odds_h=odds_h, odds_d=odds_d, odds_a=odds_a,
    )
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_agent_tools_snapshot.py -k "forecast_llm_only" -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/agent/tools.py tests/test_agent_tools_snapshot.py
git commit -m "feat(agent): add forecast_llm_only tool, no ML model lookup"
```

### Task 6: Three-way routing in `resolve_competition` and `forecast_node`

**Files:**
- Modify: `src/agent/tools.py:226-252`
- Modify: `src/agent/pipeline.py:198-249`
- Test: `tests/test_agent_tools_snapshot.py`, `tests/test_agent_pipeline.py`

- [ ] **Step 1: Write the failing tests**

Add to `tests/test_agent_tools_snapshot.py`:

```python
def test_resolve_competition_recommends_forecast_llm_only_for_ucl():
    from src.agent.tools import _resolve_competition_impl
    result = json.loads(_resolve_competition_impl("UCL"))
    assert result["tier"] == "llm_only"
    assert result["recommended_tool"] == "forecast_llm_only"
```

Add to `tests/test_agent_pipeline.py` (match its existing style of constructing a minimal state dict and calling `forecast_node` directly):

```python
def test_forecast_node_routes_llm_only_tier_to_new_tool(monkeypatch):
    calls = []
    def fake_llm_only_impl(**kwargs):
        calls.append(kwargs)
        return json.dumps({
            "match": {"home": kwargs["home_team"], "away": kwargs["away_team"]},
            "market_odds": {"implied_home": 0.4},
            "data_quality": {"prediction_basis": "llm_only_no_model", "tier": "llm_only"},
        })
    monkeypatch.setattr("src.agent.tools._forecast_llm_only_impl", fake_llm_only_impl)

    state = {
        "match_info": {
            "home_team": "Real Madrid", "away_team": "Bayern Munich", "date": "2026-10-20",
            "league": "UCL", "odds": {"home": 2.1, "draw": 3.4, "away": 3.3},
        },
        "competition_resolution": {"competition": "UCL", "tier": "llm_only", "recommended_tool": "forecast_llm_only"},
    }
    result = forecast_node(state)
    assert len(calls) == 1
    assert result["forecast_payload"]["data_quality"]["tier"] == "llm_only"
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_agent_tools_snapshot.py tests/test_agent_pipeline.py -k "llm_only" -v`
Expected: FAIL (`recommended_tool` is `"forecast_international"`; `forecast_node` calls `_forecast_international_impl` instead)

- [ ] **Step 3: Implement — `_resolve_competition_impl`**

In `src/agent/tools.py`, change line 247 from:

```python
    recommended_tool = "forecast_league" if tier == "competition_specific" else "forecast_international"
```

to:

```python
    if tier == "competition_specific":
        recommended_tool = "forecast_league"
    elif tier == "llm_only":
        recommended_tool = "forecast_llm_only"
    else:
        recommended_tool = "forecast_international"
```

Update `resolve_competition`'s docstring (lines 257-269) to mention the third tier/tool: change `'tier' ('competition_specific' or 'general_purpose')` to `'tier' ('competition_specific', 'general_purpose', or 'llm_only')`, and `'recommended_tool' ('forecast_league' or 'forecast_international')` to `'recommended_tool' ('forecast_league', 'forecast_international', or 'forecast_llm_only')`.

- [ ] **Step 4: Implement — `forecast_node`**

In `src/agent/pipeline.py`, change lines 226-240 from:

```python
    from src.agent.tools import _forecast_international_impl, _forecast_league_impl, get_snapshot_store

    store = get_snapshot_store()
    if recommended_tool == "forecast_league":
        raw = store.wrap("forecast_league", _forecast_league_impl)(
            home_team=match_info["home_team"], away_team=match_info["away_team"],
            date=match_info["date"], league=match_info.get("league", ""),
            odds_h=odds["home"], odds_d=odds["draw"], odds_a=odds["away"],
        )
    else:
        raw = store.wrap("forecast_international", _forecast_international_impl)(
            home_team=match_info["home_team"], away_team=match_info["away_team"],
            date=match_info["date"],
            odds_h=odds["home"], odds_d=odds["draw"], odds_a=odds["away"],
        )
```

to:

```python
    from src.agent.tools import (
        _forecast_international_impl, _forecast_league_impl, _forecast_llm_only_impl, get_snapshot_store,
    )

    store = get_snapshot_store()
    if recommended_tool == "forecast_league":
        raw = store.wrap("forecast_league", _forecast_league_impl)(
            home_team=match_info["home_team"], away_team=match_info["away_team"],
            date=match_info["date"], league=match_info.get("league", ""),
            odds_h=odds["home"], odds_d=odds["draw"], odds_a=odds["away"],
        )
    elif recommended_tool == "forecast_llm_only":
        raw = store.wrap("forecast_llm_only", _forecast_llm_only_impl)(
            home_team=match_info["home_team"], away_team=match_info["away_team"],
            date=match_info["date"], league=match_info.get("league", ""),
            odds_h=odds["home"], odds_d=odds["draw"], odds_a=odds["away"],
        )
    else:
        raw = store.wrap("forecast_international", _forecast_international_impl)(
            home_team=match_info["home_team"], away_team=match_info["away_team"],
            date=match_info["date"],
            odds_h=odds["home"], odds_d=odds["draw"], odds_a=odds["away"],
        )
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `python -m pytest tests/test_agent_tools_snapshot.py tests/test_agent_pipeline.py -k "llm_only" -v`
Expected: PASS

- [ ] **Step 6: Run full pipeline/tools suites to confirm no regression**

Run: `python -m pytest tests/test_agent_pipeline.py tests/test_agent_tools_snapshot.py -v`
Expected: all pass

- [ ] **Step 7: Commit**

```bash
git add src/agent/tools.py src/agent/pipeline.py tests/test_agent_tools_snapshot.py tests/test_agent_pipeline.py
git commit -m "feat(agent): route llm_only tier through forecast_llm_only"
```

### Task 7: Thread `tier` from `output_node` into `extract_recommendation`

**Files:**
- Modify: `src/agent/graph.py:318-356,407-460`
- Test: `tests/test_agent_graph.py`

- [ ] **Step 1: Write the failing test**

Add to `tests/test_agent_graph.py` (matching its existing pattern for testing `_build_recommendation`/`output_node`):

```python
def test_build_recommendation_passes_llm_only_tier_through():
    text = _wrap_json({
        **_VALID_LLM_ONLY, "overall": "direct_bet", "confidence": "medium",
        "candidates": [_llm_only_candidate("direct_bet")],
        "recommendation_pick": {"market": "result_3way", "selection": "home"},
    })
    rec = _build_recommendation(
        text, match_info={"home_team": "Real Madrid", "away_team": "Bayern Munich"},
        forecast_payload={"data_quality": {"tier": "llm_only"}}, research_evidence=None,
        config=AgentConfig.default(),
    )
    # medium confidence must be downgraded below llm_only's 'high' bar --
    # proves tier="llm_only" actually reached extract_recommendation, not
    # the default competition_specific value-edge chain (which would have
    # downgraded this for a totally different reason: null value_edge).
    assert rec["candidates"][0]["recommendation_type"] == "no_bet"
```

(Reuse `_VALID_LLM_ONLY`/`_llm_only_candidate`/`_wrap_json` from Tasks 3-4's test files, imported or redefined locally matching `test_agent_graph.py`'s existing import style.)

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_agent_graph.py -k "llm_only_tier_through" -v`
Expected: FAIL (`_build_recommendation() got an unexpected keyword argument 'tier'` — not yet accepted)

- [ ] **Step 3: Implement**

In `src/agent/graph.py`, change `_build_recommendation`'s signature and call (lines 318-342):

```python
def _build_recommendation(
    text: str,
    match_info: dict,
    forecast_payload: dict | None,
    research_evidence: dict | None,
    config: AgentConfig,
    tier: str = "competition_specific",
) -> dict:
    """Extract the LLM's MatchRecommendation JSON (or fall back to an
    insufficient_data placeholder on parse failure), then enrich/normalize it
    against the deterministic pipeline's own evidence -- never the LLM's
    prose (A30/A31/A32).

    tier: forwarded to extract_recommendation() to select the right
    validation model and downgrade chain (Task 3/4 in the llm_only plan) --
    "llm_only" for UCL/UEL-style competitions with no trained model,
    otherwise today's unchanged default."""
    try:
        recommendation = extract_recommendation(
            text,
            min_odds_threshold=config.min_odds_threshold,
            max_odds_threshold=config.max_odds_threshold,
            min_conditional_odds_threshold=config.min_conditional_odds_threshold,
            max_conditional_odds_threshold=config.max_conditional_odds_threshold,
            min_value_edge=config.min_value_edge,
            min_value_edge_result_3way_draw=config.min_value_edge_result_3way_draw,
            live_wait_min_odds=config.live_wait_min_odds,
            live_wait_target_odds=config.live_wait_target_odds,
            home_team=match_info.get("home_team"),
            away_team=match_info.get("away_team"),
            tier=tier,
        )
```

Change `output_node` (`graph.py:457-459`) to resolve tier from state and pass it through:

```python
        recommendation = _build_recommendation(
            text, match_info, forecast_payload, state.get("research_evidence"), config,
            tier=(state.get("competition_resolution") or {}).get("tier", "competition_specific"),
        )
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_agent_graph.py -k "llm_only_tier_through" -v`
Expected: PASS

- [ ] **Step 5: Run full graph suite to confirm no regression**

Run: `python -m pytest tests/test_agent_graph.py -v`
Expected: all pass

- [ ] **Step 6: Commit**

```bash
git add src/agent/graph.py tests/test_agent_graph.py
git commit -m "feat(agent): thread competition tier into recommendation extraction"
```

---

## Phase 4: Prompt & AgentConfig for `llm_only`

### Task 8: Write `config/prompts/agent_llm_only_v1.txt`

**Files:**
- Create: `config/prompts/agent_llm_only_v1.txt`

- [ ] **Step 1: Write the prompt file**

Base it on `config/prompts/agent_v1.txt`, keeping unchanged: the per-market evidence-priority ordering (result_3way/total_goals/btts/home_goals/away_goals), the injury-vs-benched distinction, the stale-vs-recent injury distinction, the FORM_SEARCH_RESULT competition-mixing warning, the `ACT, NEVER JUST PLAN` rule, and the output JSON field list (`explanation`/`team_evidence`/`the_read`/`no_bet_read`/`limitations`). Per the design doc Section 5, make these changes:

- Remove the "Evidence Already Gathered" framing around an ML forecast; replace with: "No ML forecast exists for this match — the data below is market-odds-derived context plus research. Market odds and your own research are the only evidence."
- Remove the per-target `metrics`/trust-calibration paragraph (there is no `target_versions` field in this tier's forecast payload).
- Replace "Value Calculation" with a rule stating market odds are the prior: deviating from the market-implied read requires a cited, concrete piece of evidence (a named source from a tool call this turn) the market likely hasn't priced in.
- Replace the value-edge-based direct_bet/conditional eligibility rules with: `direct_bet` requires `confidence: "high"`; `conditional` requires `confidence` at least `"medium"`; odds sanity bounds (`{{MIN_ODDS_THRESHOLD}}`/`{{MAX_ODDS_THRESHOLD}}`/`{{MIN_CONDITIONAL_ODDS_THRESHOLD}}`) still apply unchanged.
- Rewrite "Confidence Guidelines": high = clear, recent, multi-source confirmation for both sides; medium = partial coverage on one side; low = thin or conflicting sources.
- Add a **grounding requirement**: every claim in `team_evidence`/`the_read` must trace to an actual tool result from this turn — no claim without a cited source.
- Raise the tool budget: change "Most matches already have enough evidence — skip straight to step 3 unless there's a specific gap" to mandatory form/injury/stakes searches for both sides (there is no pre-gathered evidence besides odds in this tier); raise "After 2 tool calls total" to "After 5 tool calls total" (covering: home-side form, away-side form, both sides' injury/availability, and one stakes/rotation check).
- Add a new **"Cup/European Context" evidence-priority section**, parallel to the existing per-market list: squad rotation risk (fixture congestion, proximity to the next domestic match), dead-rubber group games, two-leg aggregate situations and the away-goals-rule context, knockout-stage stakes.
- In the output JSON example's `candidates` block, set `ml_probability`, `value_edge`, `composite_score` to `null` explicitly (not omitted) to match `LLMOnlyMarketCandidateModel`'s expectations; add a line stating `prediction_basis` must always be the literal string `"llm_only_no_model"`.
- Replace the reference-data preamble's forecast payload line (`FORECAST_PAYLOAD: ...`) with `MARKET_ODDS_CONTEXT: ...` describing the implied probabilities/Poisson priors `forecast_llm_only` returns.

- [ ] **Step 2: Sanity-check the file parses as plain text with the expected template variables**

Run: `grep -o "{{[A-Z_]*}}" config/prompts/agent_llm_only_v1.txt | sort -u`
Expected output includes exactly: `{{MIN_ODDS_THRESHOLD}}`, `{{MAX_ODDS_THRESHOLD}}`, `{{MIN_CONDITIONAL_ODDS_THRESHOLD}}` (no `{{MIN_VALUE_EDGE}}` or draw-specific clauses — those don't apply to this tier)

- [ ] **Step 3: Commit**

```bash
git add config/prompts/agent_llm_only_v1.txt
git commit -m "feat(agent): write llm_only tier prompt (no ML forecast, odds as prior)"
```

### Task 9: `config/agent_config_llm_only.yaml`

**Files:**
- Create: `config/agent_config_llm_only.yaml`
- Read for reference: `config/agent_config.yaml` (the live default, to mirror its required fields)

- [ ] **Step 1: Read the existing default config to confirm required fields**

Run: `cat config/agent_config.yaml`

- [ ] **Step 2: Write the new config**

Create `config/agent_config_llm_only.yaml`, copying every field from `config/agent_config.yaml` except:

```yaml
system_prompt_version: llm_only_v1
# min_value_edge is still required by AgentConfig._REQUIRED, but is never
# consulted on this tier's downgrade path (schema.py's llm_only branch
# skips every value_edge-dependent function) -- kept at the same value as
# the default config purely to satisfy the dataclass's required-field
# check, not because it does anything here.
```

Keep `markets` as the full existing list (full market-set decision, design doc Section 2) and every odds-bound threshold identical to the default config (design doc Section 5: these are reused unchanged, not redefined per tier).

- [ ] **Step 3: Verify it loads**

Run:
```bash
python -c "from src.agent.agent_config import AgentConfig; c = AgentConfig.from_yaml('config/agent_config_llm_only.yaml'); print(c.system_prompt_version)"
```
Expected output: `llm_only_v1`

- [ ] **Step 4: Commit**

```bash
git add config/agent_config_llm_only.yaml
git commit -m "feat(agent): add AgentConfig preset for llm_only tier"
```

### Task 10: Wire tier-aware config selection into live serving

**Why this is needed:** `_load_system_prompt(config)` (`src/agent/graph.py:88-102`) loads the prompt file named by `config.system_prompt_version` once per `AgentConfig`, at graph-build time — it is not re-selected per match inside one compiled graph. Since `AgentConfig` is already a plain pass-through parameter to `recommendations.run_agent` (`app/backend/recommendations.py:214`), the fix is to resolve tier and pick the right config *before* calling `run_agent`, at each live call site, not inside the graph itself.

**Files:**
- Modify: `app/backend/recommendations.py`
- Modify: `app/backend/main.py:393,422,1200`
- Modify: `app/backend/t30_refresh.py` (the call site around line 144's caller)
- Modify: `app/backend/eod_batch.py` (the call site around line 457's caller)
- Test: `app/backend/tests/test_recommendations.py` (or wherever this module's existing tests live — confirm exact filename with `find app/backend/tests -iname "*recommendation*"`)

- [ ] **Step 1: Write the failing test**

```python
def test_resolve_agent_config_for_league_picks_llm_only_preset():
    from app.backend.recommendations import resolve_agent_config_for_league
    config = resolve_agent_config_for_league("UCL")
    assert config.system_prompt_version == "llm_only_v1"


def test_resolve_agent_config_for_league_picks_default_for_e0():
    from app.backend.recommendations import resolve_agent_config_for_league
    config = resolve_agent_config_for_league("E0")
    assert config.system_prompt_version != "llm_only_v1"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest app/backend/tests/ -k "resolve_agent_config_for_league" -v`
Expected: FAIL (`ImportError`)

- [ ] **Step 3: Implement the helper**

In `app/backend/recommendations.py`, near the top (after existing imports, before `run_agent`):

```python
_LLM_ONLY_AGENT_CONFIG_PATH = Path(__file__).parent.parent.parent / "config" / "agent_config_llm_only.yaml"


def resolve_agent_config_for_league(league: str) -> AgentConfig:
    """Picks the llm_only AgentConfig preset (different system_prompt_version,
    see config/agent_config_llm_only.yaml) for a competition registered under
    that tier, the live default otherwise. Every call site that previously
    called AgentConfig.default() unconditionally -- main.py's pregenerate
    loop and live endpoint, t30_refresh.py, eod_batch.py -- should resolve
    through here instead, since AgentConfig is already a plain pass-through
    parameter to run_agent (not picked once per process)."""
    from src.logic.competition_registry import get_competition_definition
    try:
        tier = get_competition_definition(league).tier
    except ValueError:
        tier = "general_purpose"
    if tier == "llm_only":
        return AgentConfig.from_yaml(_LLM_ONLY_AGENT_CONFIG_PATH)
    return AgentConfig.default()
```

(Add `from src.agent.agent_config import AgentConfig` and `from pathlib import Path` to this file's imports if not already present — check first with `grep -n "^from\|^import" app/backend/recommendations.py`.)

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest app/backend/tests/ -k "resolve_agent_config_for_league" -v`
Expected: PASS

- [ ] **Step 5: Wire the 4 call sites**

In `app/backend/main.py`:
- Line 393 (inside the per-league pregenerate loop, lines 395-412): move `config = AgentConfig.default()` from line 393 to inside the `for league, league_fixtures in fixtures_by_league.items():` loop (after line 395), changed to `config = recommendations.resolve_agent_config_for_league(league)`.
- Line 1200: change `agent_result = await run_in_threadpool(recommendations.run_agent, match_info)` to first resolve `config = recommendations.resolve_agent_config_for_league(match_info.get("league", ""))`, then `agent_result = await run_in_threadpool(recommendations.run_agent, match_info, config)`.
- Line 422 (startup `lifespan`'s reachability check) is a one-time generic LLM-reachability probe at boot, not tied to any specific league — leave as `AgentConfig.default()` (confirm by reading the surrounding function before changing; if it loops over leagues, apply the same per-league resolution instead).

In `app/backend/t30_refresh.py`: at the call site around line 144, confirm `league` is in scope in the enclosing function (grep its signature/body above line 120), and change the `config` passed into `recommendations.run_agent(match_info=match_info, config=config)` to `recommendations.resolve_agent_config_for_league(league)` resolved just before this call (replacing however `config` currently reaches this function — read the full enclosing function first to avoid duplicating an already-correct per-call resolution).

In `app/backend/eod_batch.py`: same pattern at the call site around line 457 — `config` is already a per-league parameter to `run_eod_batch` per main.py's loop (Task 10 Step 5's first bullet already makes it per-league at the source), so once main.py resolves it per-league before calling `run_eod_batch`, this call site needs no further change — confirm by re-reading `run_eod_batch`'s signature to ensure `config` flows straight through without being re-defaulted internally.

- [ ] **Step 6: Run the backend test suite to confirm no regression**

Run: `python -m pytest app/backend/tests/ -v`
Expected: all pass

- [ ] **Step 7: Commit**

```bash
git add app/backend/recommendations.py app/backend/main.py app/backend/t30_refresh.py app/backend/eod_batch.py app/backend/tests/
git commit -m "feat(agent): resolve llm_only AgentConfig per-league at every live call site"
```

---

## Phase 5: Lesson-Loop Tier Awareness

### Task 11: Thread `tier` into `judge_lesson_candidate`/`classify_lesson_sensitivity`, short-circuit for `llm_only`

**Files:**
- Modify: `src/agent/lessons.py:470-569`
- Test: `tests/test_agent_lessons.py`

- [ ] **Step 1: Write the failing tests**

```python
def test_classify_lesson_sensitivity_llm_only_always_survives_model_change():
    """No model exists for this tier -- the question is vacuous, and must
    not cost an extra LLM round-trip to answer."""
    calls = []
    def fake_invoke(prompt):
        calls.append(prompt)
        return '{"survives_model_change": false, "reasoning": "should never be called"}'
    result = classify_lesson_sensitivity("some lesson text", "UCL", "llm_only", fake_invoke)
    assert result is True
    assert calls == []  # never actually invoked the LLM


def test_judge_lesson_candidate_llm_only_skips_model_change_question(monkeypatch):
    captured_prompt = {}
    def fake_invoke(prompt):
        captured_prompt["text"] = prompt
        return '{"approve": true, "scope": "tier", "reasoning": "clear pattern"}'
    decision = judge_lesson_candidate("some lesson text", "UCL", "llm_only", fake_invoke)
    assert decision.survives_model_change is True
    assert "retrained or replaced" not in captured_prompt["text"]
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_agent_lessons.py -k "llm_only" -v`
Expected: FAIL (both functions currently always ask the model-change question and parse its answer)

- [ ] **Step 3: Implement**

In `src/agent/lessons.py`, modify `classify_lesson_sensitivity` (lines 535-569) — add the short-circuit as the first statement in the function body:

```python
def classify_lesson_sensitivity(
    lesson_text: str, competition_id: str | None, tier: str, llm_invoke: Callable[[str], str],
) -> bool:
    """...(existing docstring)...

    tier == "llm_only": always returns True without invoking the LLM --
    no model exists for this tier, so "does this survive a model swap" is
    vacuously true and asking it would be a wasted round-trip."""
    if tier == "llm_only":
        return True
    prompt = (
        # ...unchanged...
```

Modify `judge_lesson_candidate` (lines 470-532) similarly — after the successful parse, override `survives_model_change` when `tier == "llm_only"`, and simplify the prompt text to drop the model-change question for this tier:

```python
def judge_lesson_candidate(
    lesson_text: str, competition_id: str | None, tier: str, llm_invoke: Callable[[str], str],
) -> LessonDecision:
    """...(existing docstring)...

    tier == "llm_only": the prompt omits the model-retrain question
    entirely (there is no model to retrain) and survives_model_change is
    hardcoded True on approval, not parsed from the response."""
    model_change_question = (
        "Also decide (A127): would this rule still hold even if the "
        "underlying ML forecasting model were retrained or replaced (a general reasoning/prompt-behavior "
        "insight), or is it tied to this specific model's current calibration/output quirk and likely to "
        "stop holding once that model changes?\n\n"
    ) if tier != "llm_only" else ""
    survives_field = (
        ', "survives_model_change": true|false' if tier != "llm_only" else ""
    )
    prompt = (
        f"You are deciding whether to promote a batch of live betting-recommendation results into a "
        f"standing rule for an automated agent's future recommendations in this competition "
        f"(competition_id={competition_id!r}, tier={tier!r}).\n\n"
        f"{lesson_text}\n\n"
        "Only approve if the pattern is clearly systematic, not noise from a small sample -- when in "
        "doubt, reject. If you approve, also decide scope: \"competition\" if the pattern is specific to "
        "this one competition, \"tier\" if it reflects something general enough to apply to every "
        f"competition of this tier.\n\n{model_change_question}"
        "Respond with exactly one JSON object, nothing else, with \"approve\" and \"scope\" "
        f'as JSON literals: {{"approve": true|false, "scope": "competition"|"tier"|null, '
        f'"reasoning": "one or two sentences"{survives_field}}}'
    )
    try:
        parsed = _parse_judge_json(llm_invoke(prompt))
        approve = parsed["approve"] is True
        scope = parsed.get("scope") if approve else None
        reasoning = str(parsed.get("reasoning") or "").strip() or "(no reasoning given)"
        survives_model_change = True if tier == "llm_only" else parsed.get("survives_model_change") is True
        if approve and scope not in _VALID_SCOPES:
            return LessonDecision(
                approve=False, scope=None,
                reasoning=f"invalid scope {scope!r} returned -- defaulting to reject",
                survives_model_change=False,
            )
        return LessonDecision(approve=approve, scope=scope, reasoning=reasoning, survives_model_change=survives_model_change)
    except Exception as exc:
        return LessonDecision(
            approve=False, scope=None, reasoning=f"judge call failed ({exc!r}) -- defaulting to reject",
            survives_model_change=False,
        )
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_agent_lessons.py -k "llm_only" -v`
Expected: PASS

- [ ] **Step 5: Run full lessons suite to confirm no regression**

Run: `python -m pytest tests/test_agent_lessons.py -v`
Expected: all pass

- [ ] **Step 6: Commit**

```bash
git add src/agent/lessons.py tests/test_agent_lessons.py
git commit -m "feat(agent): llm_only lessons always survive model-change check"
```

### Task 12: Evidence-gap-biased reflection prompts for `llm_only`

**Files:**
- Modify: `src/agent/lessons.py:618-855` (`generate_match_reflection`, `generate_batch_match_comparisons`, `generate_rule_from_lesson`)
- Test: `tests/test_agent_lessons.py`

- [ ] **Step 1: Write the failing tests**

```python
def test_generate_match_reflection_llm_only_biases_toward_evidence_gap():
    captured = {}
    def fake_invoke(prompt):
        captured["text"] = prompt
        return "reflection text"
    record = _make_fake_record()  # reuse this file's existing test record builder -- grep
                                   # tests/test_agent_lessons.py for the current helper name/shape
    generate_match_reflection(record, [{"role": "assistant", "content": "reasoning"}], fake_invoke, tier="llm_only")
    assert "assume" in captured["text"].lower() and "evidence gap" in captured["text"].lower()
    assert "specific search query" in captured["text"].lower() or "specific tool" in captured["text"].lower()


def test_generate_rule_from_lesson_llm_only_biases_toward_search_rules():
    captured = {}
    def fake_invoke(prompt):
        captured["text"] = prompt
        return "IF a cup tie is a dead rubber THEN check for confirmed squad rotation."
    generate_rule_from_lesson("some reflection text", fake_invoke, tier="llm_only")
    assert "check/search" in captured["text"].lower() or "search" in captured["text"].lower()
```

(First run `grep -n "_make_fake_record\|def _record\|class.*Record" tests/test_agent_lessons.py` to find this file's actual existing record-construction helper and reuse its real name/shape — do not invent a new one.)

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_agent_lessons.py -k "llm_only_biases" -v`
Expected: FAIL (`generate_match_reflection() got an unexpected keyword argument 'tier'`)

- [ ] **Step 3: Implement — `generate_match_reflection`**

In `src/agent/lessons.py`, change the signature (line 618) and the prompt-building block (lines 656-679):

```python
def generate_match_reflection(
    record: Any,
    reasoning_trace: list[dict[str, Any]] | None,
    llm_invoke: Callable[[str], str] | None,
    match_stats: dict[str, Any] | None = None,
    tier: str = "competition_specific",
) -> str:
    """...(existing docstring, plus:)...

    tier == "llm_only": defaults to assuming an EVIDENCE gap, not a
    reasoning gap -- the opposite of this function's normal neutral
    framing. Rationale: with no ML model floor under the agent and a
    larger tool-call budget for this tier (design doc Section 5), the
    dominant fixable failure mode is "didn't go look for X," not
    "misweighed X." Only call it a reasoning/interpretation gap when the
    trace clearly shows the relevant fact was already gathered this turn
    but misweighed."""
    if not reasoning_trace or llm_invoke is None:
        return generate_lesson_text(record)

    markets_summary, _ = _market_and_limitations_summary(record)
    trace_text = "\n".join(f"[{m.get('role')}] {m.get('content')}" for m in reasoning_trace)
    stats_line = (
        f" Post-match stats: {', '.join(f'{k}={v}' for k, v in match_stats.items())}."
        if match_stats else ""
    )
    actual_match_ask = (
        "what the match stats above (shots, shots on target, cards) show actually happened on the "
        "pitch, not just the final result"
        if match_stats else
        "what the final result and market outcome show actually happened"
    )

    if tier == "llm_only":
        gap_instruction = (
            "(3) assume by default this is an EVIDENCE gap (something the agent should have searched "
            "for or checked but didn't, given it has a larger tool-call budget for this tier) rather than "
            "a reasoning gap (evidence already gathered but misweighed) -- only call it a reasoning gap if "
            "the trace above clearly shows the relevant fact was already gathered this turn; (4) name ONE "
            "specific, callable fix: a specific search query, a specific tool (e.g. get_player_rating), or "
            "a specific data source to check next time -- never a vaguer 'weigh form more heavily' or "
            "'more data would help'."
        )
    else:
        gap_instruction = (
            "(3) identify the gap between the two, if any, and whether it traces to "
            "a reasoning gap (the evidence was already there but wasn't weighed correctly) or an "
            "evidence gap (something relevant was missing entirely); (4) name ONE specific, concrete "
            "piece of additional information the agent should have gathered pre-match that would have "
            "closed that gap -- a specific search query or a specific stat/data source, not a generic "
            "'more data would help'."
        )

    prompt = (
        f"You are reviewing a betting recommendation an automated agent made for a "
        f"{record.league or 'an unlabeled competition'} match, now that the actual result is known.\n\n"
        f"Recommendation: '{record.recommendation.get('overall', 'unknown')}' "
        f"(confidence={record.recommendation.get('confidence', 'unknown')}). "
        f"Markets: {markets_summary}. Actual result: {record.actual.get('result')}.{stats_line}\n\n"
        f"The agent's own reasoning and tool calls at the time:\n{trace_text}\n\n"
        "Write a reflective lesson (4-6 sentences) structured as: (1) one sentence on how the "
        "agent's own reasoning expected this match to play out pre-match -- which side it favored, "
        "by how much, and on what evidence; (2) one sentence on "
        f"{actual_match_ask}; {gap_instruction} Do not invent facts not present above."
    )
    try:
        reflection = llm_invoke(prompt)
    except Exception as exc:
        LOGGER.warning("generate_match_reflection: llm_invoke failed (%s), falling back to template.", exc)
        return generate_lesson_text(record)
    return reflection.strip() or generate_lesson_text(record)
```

- [ ] **Step 4: Implement — `generate_batch_match_comparisons`**

In `src/agent/lessons.py`, change the signature (line 816) and the prompt (lines 839-850) to add the same tier-conditional instruction:

```python
def generate_batch_match_comparisons(
    records: list[Any], llm_invoke: Callable[[str], str], tier: str = "competition_specific",
) -> str | None:
    """...(existing docstring, plus:)...

    tier == "llm_only": per A129's finding (documents/agent_user_stories.md
    PHASE 64) that the batch path produces nearly all real lesson volume
    but has never asked for the evidence-gap/reasoning-gap classification
    generate_match_reflection's single-match path does -- this tier's batch
    prompt asks for it explicitly, biased toward evidence-gap by default,
    same rationale as generate_match_reflection above."""
    blocks = "\n\n".join(
        f"--- Match {i + 1} ---\n{_describe_match_for_comparison(r)}" for i, r in enumerate(records)
    )
    if tier == "llm_only":
        per_match_ask = (
            "For EACH match above, in one short sentence, compare what the agent's reasoning expected "
            "pre-match to what actually happened, and classify any gap as an EVIDENCE gap (something "
            "specific the agent should have searched for or checked, given its larger tool-call budget "
            "for this tier) by default -- only call it a reasoning gap if the trace clearly shows the "
            "relevant fact was already gathered but misweighed."
        )
        final_ask = (
            "Then write one final paragraph (3-5 sentences) naming the SINGLE most important, recurring "
            "evidence gap across these matches and one concrete, callable fix -- a specific search query, "
            "a specific tool, or a specific data source to check next time, never a vaguer 'weigh X more "
            "heavily' or 'more data would help'."
        )
    else:
        per_match_ask = (
            "For EACH match above, in one short sentence, compare what the agent's reasoning expected "
            "pre-match to what the match stats (or, if unavailable, the final result and market outcome) "
            "show actually happened, and name the gap if any."
        )
        final_ask = (
            "Then write one final paragraph (3-5 sentences) naming the SINGLE most important, recurring "
            "pattern across these matches and one concrete, actionable adjustment for future "
            "recommendations in this competition."
        )
    prompt = (
        f"You are reviewing a batch of {len(records)} betting recommendations an automated agent made for "
        "historical matches, now that the actual results are known. Below is each match's recommendation, "
        "actual result, match stats (when available), and the agent's own pre-match reasoning.\n\n"
        f"{blocks}\n\n"
        f"{per_match_ask} {final_ask} Do not invent facts not present above, and do not use "
        "generic hedging language like 'more data would help'."
    )
    try:
        comparison = llm_invoke(prompt)
    except Exception:
        return None
    return comparison.strip() or None
```

- [ ] **Step 5: Implement — `generate_rule_from_lesson`**

In `src/agent/lessons.py`, change the signature (line 229) and prompt text (lines 245-254):

```python
def generate_rule_from_lesson(
    lesson_text: str, llm_invoke: Callable[[str], str], tier: str = "competition_specific",
) -> str | None:
    """...(existing docstring, plus:)...

    tier == "llm_only": biases distillation toward an "IF <situation> THEN
    check/search <specific evidence source>" rule rather than a "NEVER
    weigh X over Y" rule -- without hard-banning the latter, since a
    genuine misweighting lesson should still be nameable when the
    reflection actually shows one."""
    style_instruction = (
        "Prefer the form 'IF <match situation> THEN check/search <specific evidence source>' when the "
        "analysis points at something the agent should have gathered -- only use a 'NEVER weigh X over "
        "Y'-style rule if the analysis clearly shows existing evidence was misweighed, not missing."
        if tier == "llm_only" else
        "Output exactly one clean sentence starting with 'NEVER...' or 'IF...'."
    )
    prompt = (
        "Below is a reviewed post-mortem analysis of a batch of an automated betting agent's historical "
        "recommendations, including deterministic statistics and a reflective narrative.\n\n"
        f"{lesson_text}\n\n"
        "Extract ONLY the single most programmatic, highly action-oriented rule from this analysis. "
        f"Strip out all historical team names, match dates, and batch statistics. {style_instruction} "
        "Output nothing else -- no preamble, no explanation, just the single sentence."
    )
    try:
        rule = llm_invoke(prompt)
    except Exception:
        return None
    rule = rule.strip()
    return rule or None
```

- [ ] **Step 6: Run tests to verify they pass**

Run: `python -m pytest tests/test_agent_lessons.py -k "llm_only_biases" -v`
Expected: PASS

- [ ] **Step 7: Run full lessons suite to confirm no regression**

Run: `python -m pytest tests/test_agent_lessons.py -v`
Expected: all pass, same pre-existing count plus new tests

- [ ] **Step 8: Commit**

```bash
git add src/agent/lessons.py tests/test_agent_lessons.py
git commit -m "feat(agent): bias llm_only lesson reflection toward evidence gaps"
```

---

## Phase 6: Data Ingestion

### Task 13: Discover OddsPapi tournament IDs for UCL/UEL

**Files:**
- Modify: `app/backend/oddspapi_client.py` (`LEAGUE_TOURNAMENT_IDS`, line ~71-77)
- Modify: `scripts/pull_oddspapi_btts_corners.py` (`LEAGUE_TOURNAMENT_IDS`, line 48-54) — or a new sibling script, see Task 14

- [ ] **Step 1: Query the tournaments-list endpoint**

Run (substituting a real key from `.env`'s `ODDSPAPI_API_KEY`):

```bash
python3 -c "
import os, requests
from dotenv import load_dotenv
load_dotenv()
key = os.environ['ODDSPAPI_API_KEY']
resp = requests.get('https://api.oddspapi.io/v4/tournaments', params={'apiKey': key, 'sportId': 10})
for t in resp.json():
    name = t.get('name', '')
    if 'champion' in name.lower() or 'europa' in name.lower():
        print(t.get('id'), name)
"
```

Expected: two rows printed, one naming the UEFA Champions League, one naming the UEFA Europa League. Record both numeric IDs.

- [ ] **Step 2: Cross-verify each ID**

Run (substituting the discovered UCL id):

```bash
python3 -c "
import os, requests
from dotenv import load_dotenv
load_dotenv()
key = os.environ['ODDSPAPI_API_KEY']
resp = requests.get('https://api.oddspapi.io/v4/fixtures', params={'apiKey': key, 'tournamentId': '<UCL_ID>', 'statusId': 2})
data = resp.json()
print(len(data), 'finished fixtures found')
print(data[0] if data else 'none')
"
```
Expected: a non-empty list of real, recognizable UCL fixtures (known club names). Repeat for the UEL id.

- [ ] **Step 3: Record the IDs**

In `app/backend/oddspapi_client.py`, extend `LEAGUE_TOURNAMENT_IDS` (mirroring the existing comment convention):

```python
LEAGUE_TOURNAMENT_IDS = {
    "E0": 17,   # Premier League, England
    "SP1": 8,   # LaLiga, Spain
    "I1": 23,   # Serie A, Italy
    "F1": 34,   # Ligue 1, France
    "D1": 35,   # Bundesliga, Germany
    "UCL": <discovered id>,  # UEFA Champions League -- confirmed live via /v4/tournaments?sportId=10, 2026-10-XX
    "UEL": <discovered id>,  # UEFA Europa League -- confirmed live via /v4/tournaments?sportId=10, 2026-10-XX
}
```

- [ ] **Step 4: Commit**

```bash
git add app/backend/oddspapi_client.py
git commit -m "feat(ingestion): register OddsPapi tournament IDs for UCL/UEL"
```

### Task 14: Extract 1X2 odds (not just BTTS/corners) from OddsPapi's markets payload

**Why:** `OddsPapiClient` currently only parses the `total_corners` market out of the full markets payload each fixture call returns (per Phase-6 research). `forecast_llm_only` needs `odds_h`/`odds_d`/`odds_a` (1X2), which the API returns in the same payload under a different market id — not yet extracted anywhere.

**Files:**
- Modify: `app/backend/oddspapi_client.py`
- Test: `app/backend/tests/test_oddspapi_client.py`

- [ ] **Step 1: Find the 1X2 market id**

Run, using one fixtureId discovered in Task 13 Step 2:

```bash
python3 -c "
import os, requests, json
from dotenv import load_dotenv
load_dotenv()
key = os.environ['ODDSPAPI_API_KEY']
resp = requests.get('https://api.oddspapi.io/v4/historical-odds', params={'apiKey': key, 'fixtureId': '<a real UCL fixtureId>', 'bookmakers': 'pinnacle'})
data = resp.json()
markets = data.get('bookmakers', {}).get('pinnacle', {}).get('markets', {})
print(list(markets.keys()))
print(markets.get('1'))  # 1X2/match-winner is conventionally market id '1' -- confirm against the actual keys printed above
"
```
Expected: the printed key list includes a market id whose structure has three outcomes (home/draw/away) with decimal odds — confirm which numeric id this is (likely `"1"`, matching the BTTS precedent of `"104"` already confirmed in `scripts/pull_oddspapi_btts_corners.py:114`).

- [ ] **Step 2: Write the failing test**

Add to `app/backend/tests/test_oddspapi_client.py` (matching its existing style of feeding a fixture JSON payload into whatever parsing function already extracts the corners market):

```python
def test_extract_1x2_odds_from_markets_payload():
    from app.backend.oddspapi_client import extract_1x2_odds  # new function, Step 3
    payload = {"bookmakers": {"pinnacle": {"markets": {
        "<discovered 1x2 market id>": {"outcomes": [
            {"name": "Home", "price": 2.1}, {"name": "Draw", "price": 3.4}, {"name": "Away", "price": 3.3},
        ]},
    }}}}
    odds = extract_1x2_odds(payload)
    assert odds == {"home": 2.1, "draw": 3.4, "away": 3.3}
```

(Adjust the mock payload's exact outcome-name/field spelling to match whatever Step 1 actually printed — the test must mirror the real response shape, not a guess.)

- [ ] **Step 3: Run test to verify it fails**

Run: `python -m pytest app/backend/tests/test_oddspapi_client.py -k "1x2" -v`
Expected: FAIL (`ImportError`)

- [ ] **Step 4: Implement**

In `app/backend/oddspapi_client.py`, add `extract_1x2_odds`, mirroring whatever existing function extracts the BTTS/corners market (read that function first and match its exact parsing pattern/field names rather than guessing).

- [ ] **Step 5: Run test to verify it passes**

Run: `python -m pytest app/backend/tests/test_oddspapi_client.py -k "1x2" -v`
Expected: PASS

- [ ] **Step 6: Run full oddspapi client suite to confirm no regression**

Run: `python -m pytest app/backend/tests/test_oddspapi_client.py -v`
Expected: all pass

- [ ] **Step 7: Commit**

```bash
git add app/backend/oddspapi_client.py app/backend/tests/test_oddspapi_client.py
git commit -m "feat(ingestion): extract 1X2 odds from OddsPapi markets payload"
```

### Task 15: Historical UCL/UEL backfill script

**Files:**
- Create: `scripts/pull_oddspapi_ucl_uel_1x2.py` (adapted from `scripts/pull_oddspapi_btts_corners.py`)

- [ ] **Step 1: Write the script**

Copy `scripts/pull_oddspapi_btts_corners.py`'s exact structure (manifest-based resumability, `CUTOFF_DATE = "2026-01-01"`, quota-exhaustion detection, `COOLDOWN_SECONDS`), with these changes:
- `LEAGUE_TOURNAMENT_IDS` = just the two UCL/UEL entries from Task 13.
- Use `extract_1x2_odds` (Task 14) instead of the BTTS/corners market-key check, and additionally record `home`/`draw`/`away` odds directly in the manifest entry (not just a raw JSON dump) so Task 16's loader can read them without re-parsing every file.
- A finished fixture also needs a FINAL SCORE for backtest truth, which neither this endpoint nor the existing script's fixture payload has been confirmed to carry — add one check-and-record step: print one sample fixture's full dict from `/v4/fixtures?statusId=2` and inspect it for a score field (e.g. `homeScore`/`awayScore`) before writing the rest of the script; if absent, the manifest records `fixtureId`/team names/date only, and Task 16 cross-references the final score from fotmob's finished-matches payload (`fetch_all_matches`, Task 17) by matching team names + date, the same cross-source join Task 17 builds anyway.

- [ ] **Step 2: Run it once against the free-tier quota**

Run: `python scripts/pull_oddspapi_ucl_uel_1x2.py`
Expected: a `data/oddspapi_ucl_uel_snapshots/manifest.json` with at least a handful of matches from 2026-01-01 onward, honoring quota exhaustion the same way the existing script does (stop cleanly, report a summary, resumable on next run).

- [ ] **Step 3: Commit**

```bash
git add scripts/pull_oddspapi_ucl_uel_1x2.py data/oddspapi_ucl_uel_snapshots/manifest.json
git commit -m "feat(ingestion): historical UCL/UEL 1X2 odds backfill script"
```

### Task 16: Extend fotmob's finished-match parser to extract final score

**Files:**
- Modify: `src/ingestion/fotmob/fetcher.py:70-94` (`_parse_finished_matches`)
- Test: `tests/test_fotmob_fetcher.py`

- [ ] **Step 1: Inspect one real payload for the score field**

Run:
```bash
python3 -c "
import requests
resp = requests.get('https://www.fotmob.com/api/data/matches', params={'date': '20260301'}, headers={'User-Agent': 'Mozilla/5.0'})
leagues = resp.json().get('leagues', [])
for l in leagues:
    if 'champion' in l.get('name', '').lower():
        print(l['matches'][0])
        break
"
```
Expected: one match dict printed — find the score field's exact key path (commonly `status.scoreStr` or a nested `home`/`away` score object).

- [ ] **Step 2: Write the failing test**

Add to `tests/test_fotmob_fetcher.py`, using the real field shape found in Step 1:

```python
def test_parse_finished_matches_extracts_score():
    payload = [{
        "id": 123, "home": {"name": "Real Madrid"}, "away": {"name": "Bayern Munich"},
        "status": {"finished": True, "scoreStr": "2 - 1"},  # adjust to match Step 1's real shape
    }]
    parsed = _parse_finished_matches(payload)
    assert parsed[0]["home_score"] == 2
    assert parsed[0]["away_score"] == 1
```

- [ ] **Step 3: Run test to verify it fails**

Run: `python -m pytest tests/test_fotmob_fetcher.py -k "extracts_score" -v`
Expected: FAIL (`KeyError: 'home_score'`)

- [ ] **Step 4: Implement**

In `src/ingestion/fotmob/fetcher.py`, extend `_parse_finished_matches` to additionally extract and parse the score field found in Step 1 into `home_score`/`away_score` integer columns, alongside the existing `fotmob_match_id`/`match_date`/`home_team`/`away_team`.

- [ ] **Step 5: Run test to verify it passes**

Run: `python -m pytest tests/test_fotmob_fetcher.py -k "extracts_score" -v`
Expected: PASS

- [ ] **Step 6: Run full fetcher suite to confirm no regression**

Run: `python -m pytest tests/test_fotmob_fetcher.py -v`
Expected: all pass

- [ ] **Step 7: Commit**

```bash
git add src/ingestion/fotmob/fetcher.py tests/test_fotmob_fetcher.py
git commit -m "feat(ingestion): extract final score from fotmob finished-match payload"
```

### Task 17: Join fotmob results + OddsPapi odds into `raw_matches` for UCL/UEL

**Files:**
- Create: `scripts/build_ucl_uel_raw_matches.py`
- Test: manual verification query (no new unit test — this is a one-off data-assembly script in the same spirit as `scripts/pull_oddspapi_btts_corners.py`, which also has none; the demo/self-check requirement below covers it)

- [ ] **Step 1: Write the script**

For each entry in Task 15's manifest: look up the matching fotmob finished match (Task 16's extended parser, called via `fetch_all_matches` for that date) by team-name + date match (reuse `TeamNameMapper`/`standardize_team_name`, the same normalization `forecast_service.py` already uses), join its `home_score`/`away_score` onto the manifest row's `home`/`draw`/`away` odds, and upsert one row per match into `raw_matches` with `league="UCL"` or `"UEL"` as appropriate (match the existing `raw_matches` schema's column names — check `src/ingestion/football_data/loader.py`'s insert statement for the exact columns a row needs at minimum: league, date, home_team, away_team, fthg/ftag or equivalent, odds_h/odds_d/odds_a).

- [ ] **Step 2: Self-check**

Add a `if __name__ == "__main__":` block that, after the join, prints a count of successfully-joined vs. unmatched rows (fotmob match found but no odds, or vice versa) — this is the "one runnable check" for this non-trivial join logic, matching the project's `demo()`/`__main__`-self-check convention for a script with no pytest suite of its own.

Run: `python scripts/build_ucl_uel_raw_matches.py`
Expected: a nonzero joined-row count printed, with any unmatched rows named explicitly (team names + date) rather than silently dropped.

- [ ] **Step 3: Verify the rows landed correctly**

Run:
```bash
python3 -c "
import duckdb
conn = duckdb.connect('data/fpai_core.db', read_only=True)
print(conn.execute(\"SELECT league, count(*) FROM raw_matches WHERE league IN ('UCL','UEL') GROUP BY league\").fetchall())
"
```
Expected: nonzero row counts for at least one of UCL/UEL.

- [ ] **Step 4: Commit**

```bash
git add scripts/build_ucl_uel_raw_matches.py
git commit -m "feat(ingestion): join fotmob results + OddsPapi odds into raw_matches for UCL/UEL"
```

### Task 18: Wire UCL/UEL into the live fixtures endpoint

**Files:**
- Modify: `app/backend/main.py` (the `get_fixtures` endpoint, ~lines 894-1035)

- [ ] **Step 1: Add the per-competition block**

Following the exact pattern of the 6 existing `if "<CODE>" in enabled:` blocks (`main.py:894-1035`), add two new blocks for `"UCL"`/`"UEL"`: past-range results from `raw_matches` (same pattern SWE uses, `historical_results_from_raw_matches`, since there's no dedicated historical results client for these competitions), future-range fixtures from fotmob's `fetch_all_matches` (Task 16) filtered to the Champions League/Europa League entries, tagged via the existing `_tag(matches, "<CODE>")` helper.

- [ ] **Step 2: Manual verification**

With `display_enabled: false` still set (Task 2), confirm the new blocks don't fire: `list_display_enabled_competition_ids()` excludes UCL/UEL, so `main.py`'s `enabled` set check naturally skips them — no risk of surfacing to users yet. Verify this by temporarily flipping `display_enabled: true` in a local/test config copy (not committed) and confirming `GET /api/fixtures` includes UCL/UEL fixtures with the expected tag, then revert.

- [ ] **Step 3: Commit**

```bash
git add app/backend/main.py
git commit -m "feat(ingestion): wire UCL/UEL into the live fixtures endpoint (still display_enabled=false)"
```

---

## Phase 7: Frontend Tier Label

### Task 19: Add the `llm_only` tier tag

**Files:**
- Modify: `app/frontend/components/MatchUI.tsx:65,462-465,471-474`

- [ ] **Step 1: Extend the `Tier` type**

Change line 65:

```typescript
export type Tier = "competition_specific" | "general_purpose" | "llm_only";
```

- [ ] **Step 2: Add label and explanation**

Change `TIER_LABEL` (lines 462-465):

```typescript
const TIER_LABEL: Record<Tier, string> = {
  competition_specific: "Modeled",
  general_purpose: "General",
  llm_only: "AI Judgment",
};
```

Change `TIER_EXPLANATION` (lines 471-474):

```typescript
const TIER_EXPLANATION: Record<Tier, string> = {
  competition_specific: "This competition has its own trained model, built on real historical team data.",
  general_purpose: "No dedicated model for this competition yet -- a general-purpose fallback model instead.",
  llm_only: "No model backs this competition at all -- a pure AI judgment call from market odds and research, not benchmarked against a trained model.",
};
```

- [ ] **Step 3: Type-check**

Run: `cd app/frontend && npx tsc --noEmit`
Expected: no new type errors (every `Record<Tier, ...>` map in the file must now have all three keys — `tsc` will flag any that don't).

- [ ] **Step 4: Commit**

```bash
git add app/frontend/components/MatchUI.tsx
git commit -m "feat(frontend): add AI Judgment tier tag for llm_only competitions"
```

---

## Phase 8: Documentation

### Task 20: Update `documents/FRAI_TECHSPEC.md`

**Files:**
- Modify: `documents/FRAI_TECHSPEC.md`

- [ ] **Step 1: Add a new section after Section 29**

Following the existing style (motivation subsection, verified-against-file:line claims, Tests subsection), add:

```markdown
## 30. Phase 65: LLM-Only Recommendation Tier — UCL/UEL (US#213+)

### 30.1 Motivation

[Summarize: frontend only ever showed ML-modeled leagues; this extends coverage to competitions with no trained model, starting with UEFA Champions League/Europa League, via direct LLM reasoning over market odds + research instead of a model-vs-market edge. See docs/superpowers/specs/2026-10-06-llm-only-cup-recommendations-design.md for the full design rationale.]

### 30.2 The `llm_only` Tier

[Document VALID_TIERS extension, competitions.yaml UCL/UEL entries, the three-way routing in resolve_competition/forecast_node, forecast_llm_only's reuse of ForecastService._compute_mkt_features_from_odds with no model lookup.]

### 30.3 Schema Changes

[Document LLMOnlyMarketCandidateModel/LLMOnlyMatchRecommendationModel, the confidence-gated downgrade chain replacing value-edge floors, the confidence-tier stake mapping replacing Kelly sizing.]

### 30.4 Lesson-Loop Tier Awareness

[Document the evidence-gap-biased reflection prompts, the model-change-question short-circuit, and the A129 (PHASE 64) connection — this tier's design directly addresses two gaps that investigation flagged: the batch path never asking for evidence/reasoning-gap classification, and no tool-call budget for lesson-triggered research.]

### 30.5 Data Sourcing and the Backtest-ROI Launch Gate

[Document the OddsPapi tournament-ID discovery, the 2026-01-01 historical-odds cutoff constraint, the fotmob score-extraction addition, and the accepted-thin-sample decision for the launch gate.]

**Tests:** [list every test file touched in Phases 1-7, matching the existing section's "re-run live" convention — run the full suite once everything above is merged and report the actual pass count, the same way Section 29.2/29.3 do.]
```

- [ ] **Step 2: Commit**

```bash
git add documents/FRAI_TECHSPEC.md
git commit -m "docs(techspec): document the llm_only recommendation tier (Phase 65)"
```

### Task 21: Append user stories

**Files:**
- Modify: `documents/user_stories.md`
- Modify: `documents/agent_user_stories.md`

- [ ] **Step 1: `documents/user_stories.md`**

Append, matching the existing `| US#NNN | status | description | comments |` table convention, starting at `US#213`:

- US#213: Add `llm_only` to the competition registry's `VALID_TIERS`, register UCL/UEL.
- US#214: `forecast_llm_only` tool + three-way routing in `resolve_competition`/`forecast_node`.
- US#215: Nullable-edge-field validation model + confidence-gated downgrade chain for `llm_only` recommendations.
- US#216: Data ingestion — OddsPapi tournament IDs, 1X2 odds extraction, historical backfill, fotmob score extraction, `raw_matches` join for UCL/UEL.
- US#217: Live fixtures endpoint + frontend tier label for `llm_only` competitions.

Mark each `active` until its corresponding phase's tasks are committed and tested, then flip to `completed` per `CLAUDE.md`'s workflow instruction.

- [ ] **Step 2: `documents/agent_user_stories.md`**

Append a new `## PHASE 65: LLM-Only Recommendation Tier — UCL/UEL (2026-10-06)` section, in the same style as PHASE 64 (motivation paragraph, `| ID | Status | Description | Comments |` table). Give it IDs continuing from the highest existing (`A129`/`W243`) — e.g. `A130` for the prompt/schema/lesson-loop work, since it's agent-side design work in the same vein as A106-A129, not a `W`-prefixed one-off. Explicitly cross-reference A129/PHASE 64 in the motivation paragraph: this tier's lesson design directly implements two of A129's candidate next steps (extending evidence/reasoning-gap classification to the batch path; raising tool-call budget to actually afford lesson-triggered research) — in a new tier, not a retrofit to the existing one, so A129 itself stays open/unaddressed for the 6 ML leagues.

- [ ] **Step 3: Commit**

```bash
git add documents/user_stories.md documents/agent_user_stories.md
git commit -m "docs(stories): add US#213-217 and PHASE 65 for the llm_only tier"
```

---

## Phase 9: Lesson Mining, Backtest, Launch Gate

### Task 22: Mine initial `scope='tier'` lessons before the ROI gate

**Files:** none (operational — uses existing `agent-train-experiment` tooling, Task 13-17's ingested corpus)

- [ ] **Step 1: Run agent-train-experiment over the UCL/UEL sample**

Follow the `agent-train-experiment` skill's existing procedure (cost-aware DeepSeek balance checks, stratified sampling), scoped to `league IN ('UCL', 'UEL')` once Task 17's `raw_matches` rows exist.

- [ ] **Step 2: Review and approve lesson candidates**

Use `agent-lessons approve --scope tier` for patterns that generalize across both competitions (per design doc Section 7's default), `--scope competition` only for a genuine UCL-vs-UEL divergence (e.g. squad rotation intensity).

- [ ] **Step 3: Confirm lessons load live**

Run a single `forecast_llm_only`-tier match through `lessons_node` (or re-run `tests/test_agent_lessons.py`'s `load_approved_lessons` tests scoped to `tier="llm_only"`) to confirm approved rules are actually injected.

### Task 23: Backtest-ROI launch gate

**Files:** none (operational)

- [ ] **Step 1: Run agent-backtest-run over the UCL/UEL historical sample**

Per the `agent-backtest-run` skill's existing procedure.

- [ ] **Step 2: Compare ROI against the two baselines**

Flat-stake-every-match, and always-bet-market-favorite (design doc Section 8). Per the user's explicit 2026-10-06 decision, treat a small/inconclusive sample the same way the existing E0 lesson A/B (n=6-13 bets) was treated — inconclusive-not-negative, not a blocker — rather than withholding launch indefinitely on sample-size grounds alone. A **clearly negative** result (not just inconclusive) still blocks launch.

- [ ] **Step 3: Flip `display_enabled: true`**

In `config/competitions.yaml`, for whichever of UCL/UEL passed (independently — one could launch before the other).

```bash
git add config/competitions.yaml
git commit -m "feat(agent): enable llm_only UCL/UEL recommendations for users"
```

- [ ] **Step 4: Mark US#213-217 / PHASE 65 complete**

Per `CLAUDE.md`'s workflow instruction, flip the corresponding rows in `documents/user_stories.md`/`documents/agent_user_stories.md` to `completed` with a completion note (actual test counts, actual ROI numbers, actual sample size) — matching the existing convention throughout both documents.

```bash
git add documents/user_stories.md documents/agent_user_stories.md
git commit -m "docs(stories): mark llm_only UCL/UEL tier complete"
```

---

## Self-Review Notes

- **Spec coverage:** every numbered section of the design doc (Sections 3-9) maps to a phase above (3→Phase1, 4→Phase3, 5→Phase4, 6→Phase2, 7→Phase5, 8→Phase6/9, frontend bullet of 8→Phase7, 10→tests embedded throughout, 11→Phase8).
- **Open/unresolved-until-executed values** (OddsPapi tournament IDs, the 1X2 market id, fotmob's score-field key path) are each resolved by a concrete discovery step with exact commands before being used — never left as a bare TBD. This mirrors the codebase's own existing convention (`fetcher.py`'s `LEAGUE_IDS` comment: "live-verified 2026-08-15 against /api/data/matches").
- **Type consistency:** `tier` is threaded with the exact same string values (`"llm_only"`, default `"competition_specific"`) through `extract_recommendation` → `_build_recommendation` → `output_node`, and separately through `generate_match_reflection`/`generate_batch_match_comparisons`/`generate_rule_from_lesson`/`judge_lesson_candidate`/`classify_lesson_sensitivity` — no renaming across tasks.
- **Known risk flagged, not hidden:** Phase 9's backtest sample will be thin for months (2026-01-01 historical-odds cutoff); the user explicitly accepted this tradeoff on 2026-10-06 rather than deferring the gate.
