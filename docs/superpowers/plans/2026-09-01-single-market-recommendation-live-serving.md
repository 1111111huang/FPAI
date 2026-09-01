# Single-Market Recommendation — Live Serving Wiring Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Migrate every remaining consumer of the agent's old `markets` array — the app-side schema mirror, settlement, bet-logging, and the frontend — onto the `candidates`/`recommendation_pick` shape Phase 29 (A88–A91) already built, and delete the two now-dead reduction functions (`pick_recommended_market()`, `bestMarket()`/`marketDirections()`) once every real caller has migrated.

**Architecture:** `app/backend/recommendations.py` gets the same `MarketCandidateOut`/`RecommendationPickOut` split Task 1 of the prior plan gave the agent side (kept deliberately looser-typed, matching this file's own existing convention). `validate_and_degrade()` resolves `recommendation_pick` by calling the already-existing `resolve_recommendation_pick()` (`src/agent/market_resolution.py`) directly against the raw candidate dicts, then cross-checks the match/selection against the *validated* candidate list before trusting it — closing a narrow but real inconsistency window a naive port would leave open. Settlement and bet-logging get one-line swaps to the same shared resolver. The frontend gets a TS port, `resolveRecommendation()`, with the identical three-case contract. Old cached rows need no migration — empty `candidates`/null `recommendation_pick` naturally result from a missing key, and one small downgrade-only rule (ported from A90) keeps `overall` honest.

**Tech Stack:** Python 3 (Pydantic), TypeScript/React, pytest + Jest (TDD) — no new dependencies.

**Scope:** `app/backend/recommendations.py`, `app/backend/recommendation_outcomes.py`, `app/backend/bets.py`, `app/frontend/lib/types.ts`, `app/frontend/components/MatchUI.tsx`, `app/frontend/lib/dashboardMetrics.ts`, and `src/agent/market_resolution.py` (deletion only) — exactly `docs/superpowers/specs/2026-09-01-single-market-recommendation-live-serving-design.md`'s scope, `documents/app_user_stories.md` Phase 47 (W193–W197). The backtest/train harness (`BacktestRecord`, `evaluation.py`, `staking.py`, `src/agent/backtest.py`) is **not touched by this plan** — sub-project #3, separately deferred.

**A note on test state between tasks:** same discipline as the prior plan — `MatchRecommendationOut` becoming strict about `candidates` (Task 1) will not itself break other files (Python's `.get("markets")`/`.get("candidates")` on a plain dict never raises), but each task's own test file is reworked in the same task that touches its source file. No file is left mid-migration for another task to silently rely on.

---

### Task 1: App-side schema + `validate_and_degrade()` (`app/backend/recommendations.py`)

**Files:**
- Modify: `app/backend/recommendations.py` (`MarketRecommendationOut`/`MatchRecommendationOut` at `~lines 297-346`, `validate_and_degrade` at `~lines 349-427`)
- Modify: `app/backend/tests/test_recommendations_schema.py`

- [ ] **Step 1: Rework the file's own fixtures, then write the failing tests**

Do the fixture rework FIRST — the new tests below reference the reworked names directly, and adding them before the rework would break the whole file at collection time (an undefined `_VALID_CANDIDATE`), not just show the new tests failing.

At the top of `app/backend/tests/test_recommendations_schema.py`, the existing `_VALID_MARKET`/`_VALID_RAW` module-level dicts get the same two substitutions every other file in this codebase's schema migration has used: rename `_VALID_MARKET` → `_VALID_CANDIDATE` with two new keys added (`"composite_score": 0.6`, `"reason": "Clears the edge floor at a realistic price."`), and `_VALID_RAW`'s `"markets": [_VALID_MARKET]` → `"candidates": [_VALID_CANDIDATE], "recommendation_pick": {"market": "result_3way", "selection": "home"}`. Add the import `from app.backend.recommendations import RecommendationPickOut` alongside the file's existing `validate_and_degrade` import. Apply the identical substitution to every other EXISTING test in the file (`{**_VALID_MARKET, ...}` → `{**_VALID_CANDIDATE, ...}`, `{**_VALID_RAW, "markets": [...]}` → `{**_VALID_RAW, "candidates": [...]}`, `result.markets` → `result.candidates` in every assertion).

Now add these new tests, which exercise the *target* behavior (still failing at this point, since `app/backend/recommendations.py` itself hasn't changed yet — Step 3 does that):

```python
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
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest app/backend/tests/test_recommendations_schema.py -k "candidates_and_recommendation_pick or dangling_pick or failed_validation or old_shape_row or never_touched_by_the_cap" -v`
Expected: FAIL — `MatchRecommendationOut` still has `markets`, not `candidates`; `RecommendationPickOut` doesn't exist yet (the fixture rework from Step 1 already landed, so this fails on the real target behavior, not a missing name).

- [ ] **Step 3: Implement**

In `app/backend/recommendations.py`, add the import (near the existing `from src.agent.schema import ...` line):

```python
from src.agent.market_resolution import resolve_recommendation_pick
```

Replace `MarketRecommendationOut`/`MatchRecommendationOut` (currently `~lines 297-346`):

```python
class MarketCandidateOut(BaseModel):
    # Constrained to the same vocabulary config/prompts/agent_v1.txt already
    # specifies to the LLM (result_3way/btts/total_goals/home_corners/
    # away_corners, home/draw/away/yes/no/over_2.5/under_2.5) -- previously
    # plain `str`, so a non-canonical name the agent invented (observed live:
    # "1X2" for what should be result_3way, "Asian Handicap", team names used
    # as a selection) passed validation instead of being dropped like any
    # other malformed market.
    market: Literal["result_3way", "btts", "total_goals", "home_corners", "away_corners"]
    selection: Literal["home", "draw", "away", "yes", "no", "over_2.5", "under_2.5"]
    recommendation_type: str
    current_odds: float | None
    # BUG-032: defaulted for the same reason as src/agent/schema.py's
    # MarketCandidateModel -- a cached/replayed row missing this key
    # (e.g. any generation predating this fix) must still validate here,
    # not just at the agent layer.
    min_odds: float = 0.0
    ml_probability: float
    implied_probability: float
    value_edge: float
    # W83: agent-side A52 computes this (src/agent/schema.py) for a
    # 'conditional' market -- the price it'd need to reach to clear
    # min_value_edge, or None when not applicable/computable. Defaulted so a
    # pre-A52 cached row (no such key at all) still validates, same
    # W15-established convention as feature_completeness below.
    target_odds: float | None = None
    # A88/W193: the agent's own self-reported edge/hit-probability balance --
    # defaulted so a pre-this-change cached row (no such key at all) still
    # validates, same convention as target_odds above.
    composite_score: float = 0.0
    reason: str = ""


class RecommendationPickOut(BaseModel):
    market: Literal["result_3way", "btts", "total_goals", "home_corners", "away_corners"]
    selection: Literal["home", "draw", "away", "yes", "no", "over_2.5", "under_2.5"]


class MatchRecommendationOut(BaseModel):
    match: dict
    overall: str
    candidates: list[MarketCandidateOut]
    recommendation_pick: RecommendationPickOut | None = None
    # One bullet per aspect, mirroring src/agent/schema.py's MatchRecommendationModel.
    explanation: list[str]
    confidence: str
    limitations: list[str]
    prediction_basis: str
    invalid_market_count: int = 0
    # W15: surfaced so the UI can treat cold_start_risk as a first-class
    # trust signal regardless of what prediction_basis claims. Default safely
    # for recommendations cached before W15 shipped (no such keys at all).
    cold_start_risk: bool = False
    feature_completeness: float | None = None
    unknown_team: bool = False
    # A82 (agent_user_stories.md): Kelly-derived stake-sizing suggestion for
    # this recommendation's actual pick, as a multiple of an abstract "Unit
    # Bet" -- not a dollar figure. None when there's no priced pick, or
    # absent entirely on a pre-A82 cached row, same convention as
    # target_odds/feature_completeness above.
    unit_bet_multiplier: float | None = None
```

Replace `validate_and_degrade` (currently `~lines 349-427`):

```python
def validate_and_degrade(
    raw: dict, home_team: str | None = None, away_team: str | None = None
) -> MatchRecommendationOut:
    """Validate a raw MatchRecommendation dict (from run_agent, a cache, or
    anywhere else), dropping any candidate that fails validation rather than
    raising for the whole request. Top-level fields default safely too, so
    even a badly malformed payload can't crash the endpoint.

    BUG-023/024: the agent's LLM call has been observed hallucinating a
    completely unrelated match's analysis (most often "Manchester City vs
    Liverpool", confirmed on 5/10 fixtures in one live sandbox precompute
    batch) instead of grounding itself in the real requested fixture -- with
    nothing else in the pipeline catching it before this recommendation is
    cached and served. `home_team`/`away_team` are the fixture actually
    requested, so a mismatch against the agent's own self-reported `match`
    field can be caught and degraded here, the one layer already responsible
    for never trusting the agent's output blindly. Optional (mirroring
    extract_recommendation's own home_team/away_team params, src/agent/schema.py):
    `GET /api/recommendations/{match_id}` (main.py) only has match_id/date, not
    a ground-truth fixture to compare against, so it calls this with neither --
    the match-mismatch check is skipped, but the per-candidate validation below
    still runs, which is what that endpoint actually needs (BUG-028: it used
    to call MatchRecommendationOut.model_validate() directly instead of this
    function at all, so any pre-existing cached row with a market/selection
    that predates the Literal constraints below -- extremely common, since the
    local-model hallucinations this file's other bugs document routinely wrote
    non-canonical values -- raised an uncaught ValidationError, a 500 for a
    plain cache read, not the graceful degrade every other caller already got.)

    W193 (2026-09-01 design): `markets`/`pick_recommended_market()`'s
    max(value_edge) reduction replaced by `candidates`/`recommendation_pick`
    -- resolved via the same resolve_recommendation_pick() the agent side
    uses (src/agent/market_resolution.py), against the *raw* candidate dicts
    first (so a dangling pointer -- the LLM named something never listed --
    is caught the same way as any other malformed pick), then cross-checked
    against the *validated* candidate list (so a pick whose own candidate
    failed structural validation doesn't survive just because it was found
    in the raw list). Either failure mode collapses to the same outcome: no
    resolvable pick, `overall` capped at "no_bet" if it claimed anything
    stronger -- the app-layer mirror of A90's own downgrade-only rule, since
    this layer doesn't re-run the agent's guardrails itself. An old-shape
    cached row (no `candidates`/`recommendation_pick` key at all) needs no
    separate detection: raw.get("candidates") is naturally [],
    raw.get("recommendation_pick") is naturally None, and the same cap
    applies."""
    if home_team and away_team:
        reported = reported_teams(raw.get("match") or {})
    else:
        reported = None
    if reported is not None and not teams_match((home_team, away_team), reported):
        _LOG.warning(
            "agent_match_mismatch | requested=%s v %s | agent_reported=%s v %s",
            home_team, away_team, reported[0], reported[1],
        )
        return MatchRecommendationOut(
            match={"home_team": home_team, "away_team": away_team},
            overall="insufficient_data",
            candidates=[],
            recommendation_pick=None,
            explanation=["The agent's analysis referenced a different match than requested and was discarded."],
            confidence="low",
            limitations=[
                f"Agent output was for {reported[0]} v {reported[1]}, not the requested "
                f"{home_team} v {away_team} -- discarded as a mismatch."
            ],
            prediction_basis="unknown",
            invalid_market_count=len(raw.get("candidates") or []),
        )

    valid_candidates: list[MarketCandidateOut] = []
    invalid_count = 0
    for candidate in raw.get("candidates") or []:
        try:
            valid_candidates.append(MarketCandidateOut.model_validate(candidate))
        except ValidationError:
            invalid_count += 1

    limitations = list(raw.get("limitations") or [])
    if invalid_count:
        limitations.append(f"{invalid_count} market(s) omitted: malformed data from the agent.")

    overall = raw.get("overall") or "insufficient_data"
    picked_raw = resolve_recommendation_pick(raw.get("candidates") or [], raw.get("recommendation_pick"))
    resolved_pick: RecommendationPickOut | None = None
    if picked_raw is not None:
        still_valid = any(
            c.market == picked_raw.get("market") and c.selection == picked_raw.get("selection")
            for c in valid_candidates
        )
        if still_valid:
            resolved_pick = RecommendationPickOut(market=picked_raw["market"], selection=picked_raw["selection"])
    if resolved_pick is None and overall not in ("no_bet", "insufficient_data"):
        limitations.append("No resolvable recommendation_pick -- overall capped at no_bet.")
        overall = "no_bet"

    return MatchRecommendationOut(
        match=raw.get("match") or {},
        overall=overall,
        candidates=valid_candidates,
        recommendation_pick=resolved_pick,
        explanation=normalize_explanation(raw.get("explanation")),
        confidence=raw.get("confidence") or "low",
        limitations=limitations,
        prediction_basis=raw.get("prediction_basis") or "unknown",
        invalid_market_count=invalid_count,
        cold_start_risk=bool(raw.get("cold_start_risk", False)),
        feature_completeness=raw.get("feature_completeness"),
        unknown_team=bool(raw.get("unknown_team", False)),
        unit_bet_multiplier=raw.get("unit_bet_multiplier"),
    )
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest app/backend/tests/test_recommendations_schema.py -v`
Expected: PASS, every test in the file.

- [ ] **Step 5: Commit**

```bash
git add app/backend/recommendations.py app/backend/tests/test_recommendations_schema.py
git commit -m "feat(app): W193 -- MarketCandidateOut/RecommendationPickOut schema

validate_and_degrade() resolves recommendation_pick via the agent's own
resolve_recommendation_pick(), cross-checked against the validated
candidate list. Old-shape cached rows degrade gracefully via the same
downgrade-only cap A90 established -- no migration, no detection code.

Co-Authored-By: Claude Sonnet 5 <noreply@anthropic.com>"
```

---

### Task 2: Settlement swap (`app/backend/recommendation_outcomes.py`)

**Files:**
- Modify: `app/backend/recommendation_outcomes.py` (import at `~line 27`, `resolve_pending_recommendations` at `~line 277`)
- Modify: `app/backend/tests/test_recommendation_outcomes.py`
- Modify: `app/backend/tests/test_recommendation_outcomes_endpoints.py`

- [ ] **Step 1: Write the failing test**

Add to `app/backend/tests/test_recommendation_outcomes.py`:

```python
def test_resolves_using_the_new_candidates_and_recommendation_pick_shape(tmp_path: Path) -> None:
    cache = RecommendationCache(db_path=tmp_path / "cache.db")
    store = RecommendationOutcomeStore(db_path=tmp_path / "outcomes.db")
    rec = {
        "match": {"home": "Arsenal", "away": "Everton", "date": "2026-08-22", "league": "E0"},
        "overall": "direct_bet",
        "candidates": [{
            "market": "result_3way", "selection": "home", "recommendation_type": "direct_bet",
            "current_odds": 2.0, "value_edge": 0.1, "composite_score": 0.6, "reason": "Clears the floor.",
        }],
        "recommendation_pick": {"market": "result_3way", "selection": "home"},
        "confidence": "medium", "explanation": [], "limitations": [], "prediction_basis": "team_history_and_market",
    }
    cache.record_generation("m1", "2026-08-22", "hash1", {}, rec, "scheduled")
    client = MagicMock()
    client.get_results.return_value = [_match("m1", 2, 1)]

    resolved = resolve_pending_recommendations(cache, store, client)

    assert len(resolved) == 1
    assert resolved[0].correct is True
    assert resolved[0].market == "result_3way"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest app/backend/tests/test_recommendation_outcomes.py -k new_candidates_and_recommendation_pick_shape -v`
Expected: FAIL — `resolve_pending_recommendations` still calls `pick_recommended_market(rec.get("markets") or [])`, which finds nothing in a `candidates`-shaped row.

- [ ] **Step 3: Implement, then rework the file's own fixtures**

In `app/backend/recommendation_outcomes.py`, change the import at `~line 27`:

```python
from src.agent.market_resolution import RESOLVABLE_MARKETS, build_actual_outcome, market_correct, resolve_recommendation_pick
```

And the one call site at `~line 277`:

```python
picked = resolve_recommendation_pick(rec.get("candidates") or [], rec.get("recommendation_pick"))
```

(Everything else in `resolve_pending_recommendations` — the `if picked is None or picked.get("market") not in RESOLVABLE_MARKETS: unresolvable_market_count += 1; continue` line right after it, and everything downstream — is unchanged.)

In `app/backend/tests/test_recommendation_outcomes.py`, the shared `_rec()` helper (currently `~lines 26-36`) is the single place every test in this file builds its fixture through — update it once:

```python
def _rec(overall: str, market: str, selection: str, recommendation_type: str, current_odds, value_edge=0.1, league="E0", confidence="medium") -> dict:
    return {
        "match": {"home": "Arsenal", "away": "Everton", "date": "2026-08-22", "league": league},
        "overall": overall,
        "candidates": [{
            "market": market, "selection": selection, "recommendation_type": recommendation_type,
            "current_odds": current_odds, "value_edge": value_edge,
            "composite_score": 0.6, "reason": "Test fixture.",
        }],
        "recommendation_pick": {"market": market, "selection": selection},
        "confidence": confidence,
        "explanation": [], "limitations": [], "prediction_basis": "team_history_and_market",
    }
```

In `app/backend/tests/test_recommendation_outcomes_endpoints.py`, the module-level `_REC` dict (currently `~lines 45-49`) gets the same two-field treatment:

```python
_REC = {
    "match": {"home": "Arsenal", "away": "Everton", "date": "2026-08-22", "league": "E0"},
    "overall": "direct_bet",
    "candidates": [{"market": "result_3way", "selection": "home", "recommendation_type": "direct_bet",
                     "current_odds": 2.0, "value_edge": 0.1, "composite_score": 0.6, "reason": "Test fixture."}],
    "recommendation_pick": {"market": "result_3way", "selection": "home"},
    "confidence": "medium", "explanation": [], "limitations": [], "prediction_basis": "team_history_and_market",
}
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest app/backend/tests/test_recommendation_outcomes.py app/backend/tests/test_recommendation_outcomes_endpoints.py -v`
Expected: PASS, every test in both files.

- [ ] **Step 5: Commit**

```bash
git add app/backend/recommendation_outcomes.py app/backend/tests/test_recommendation_outcomes.py app/backend/tests/test_recommendation_outcomes_endpoints.py
git commit -m "feat(app): W194 -- settlement reads candidates/recommendation_pick

One-line swap: pick_recommended_market() -> resolve_recommendation_pick(),
reusing the agent's own resolver. Downstream unresolvable-market handling
unchanged.

Co-Authored-By: Claude Sonnet 5 <noreply@anthropic.com>"
```

---

### Task 3: Bets swap (`app/backend/bets.py`)

**Files:**
- Modify: `app/backend/bets.py` (imports, `resolve_from_recommendation` at `~lines 62-95`)
- Modify: `app/backend/tests/test_bets_schema.py`

- [ ] **Step 1: Write the failing test**

Add to `app/backend/tests/test_bets_schema.py`:

```python
def test_resolve_from_recommendation_reads_candidates_shape():
    rec = {
        "match": {"home": "Arsenal", "away": "Everton", "date": "2026-08-22", "league": "E0"},
        "overall": "direct_bet",
        "candidates": [
            {"market": "result_3way", "selection": "home", "recommendation_type": "direct_bet", "current_odds": 2.1},
            {"market": "btts", "selection": "yes", "recommendation_type": "conditional", "current_odds": 1.9},
        ],
        "recommendation_pick": {"market": "result_3way", "selection": "home"},
        "explanation": "test", "confidence": "medium", "limitations": [], "prediction_basis": "team_history_and_market",
    }
    request = BetFromRecommendationRequest(match_id="m1", recommendation=rec, market="btts", selection="yes", stake=10.0)

    resolved = resolve_from_recommendation(request)

    assert resolved["odds"] == 1.9
```

Note this deliberately bets on `btts`/`yes` — a candidate present in `candidates` but NOT the `recommendation_pick` — proving `resolve_from_recommendation` still permits betting on any listed candidate, matching the design's explicit choice to preserve that permissiveness.

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest app/backend/tests/test_bets_schema.py -k candidates_shape -v`
Expected: FAIL — `resolve_from_recommendation` still searches `recommendation.get("markets")`, empty for a `candidates`-shaped payload, raises `ValueError`.

- [ ] **Step 3: Implement, then rework the file's own fixture**

In `app/backend/bets.py`, add the import:

```python
from src.agent.market_resolution import resolve_recommendation_pick
```

Replace the market-lookup block inside `resolve_from_recommendation` (currently `~lines 71-84`):

```python
    picked = resolve_recommendation_pick(
        request.recommendation.get("candidates") or [],
        {"market": request.market, "selection": request.selection},
    )
    if picked is None:
        raise ValueError(
            f"Market {request.market!r}/selection {request.selection!r} not found in the given recommendation."
        )
    odds = picked.get("current_odds")
    if odds is None:
        raise ValueError(f"Market {request.market!r}/selection {request.selection!r} has no current_odds to bet against.")
```

In `app/backend/tests/test_bets_schema.py`, the module-level `_RECOMMENDATION` dict (currently `~lines 20-30`) becomes:

```python
_RECOMMENDATION = {
    "match": {"home": "Arsenal", "away": "Everton", "date": "2026-08-22", "league": "E0"},
    "overall": "direct_bet",
    "candidates": [
        {"market": "result_3way", "selection": "home", "recommendation_type": "direct_bet", "current_odds": 2.1},
        {"market": "btts", "selection": "yes", "recommendation_type": "conditional", "current_odds": 1.9},
    ],
    "recommendation_pick": {"market": "result_3way", "selection": "home"},
    "explanation": "test",
    "confidence": "medium",
    "limitations": [],
    "prediction_basis": "team_history_and_market",
}
```

And the one other inline `"markets"` usage in this file (currently `~line 65`, `test_resolve_from_recommendation_raises_when_odds_are_null`): `{**_RECOMMENDATION, "markets": [{"market": "btts", ...}]}` becomes `{**_RECOMMENDATION, "candidates": [{"market": "btts", "selection": "yes", "current_odds": None}]}`.

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest app/backend/tests/test_bets_schema.py -v`
Expected: PASS, every test in the file.

- [ ] **Step 5: Commit**

```bash
git add app/backend/bets.py app/backend/tests/test_bets_schema.py
git commit -m "feat(app): W195 -- bet-logging reads candidates/recommendation_pick

resolve_from_recommendation() builds its own market+selection pointer from
the request (not the recommendation's own recommendation_pick), preserving
today's permissive any-listed-candidate-is-loggable behavior.

Co-Authored-By: Claude Sonnet 5 <noreply@anthropic.com>"
```

---

### Task 4: Frontend wire types + `MatchUI.tsx`

**Files:**
- Modify: `app/frontend/lib/types.ts` (`MarketRecommendationOut`/`MatchRecommendationOut` at `~lines 21-56`)
- Modify: `app/frontend/components/MatchUI.tsx` (`Match`/`MarketRec` types at `~lines 60-108`, `fixtureToMatch` at `~line 166`, `applyRecommendation` at `~lines 177-203`, `bestMarket`/`hasPositiveEdge`/`marketDirections` at `~lines 467-501`, call sites at `~lines 488, 828, 1885, 1994`, render site at `~line 2069`, second `Match`-literal at `~line 1952`)
- Modify: `app/frontend/components/__tests__/MatchUI.test.tsx`, `app/frontend/components/__tests__/MatchUI.hitMiss.test.tsx`

**A note on why this task isn't split into a strict red/green TDD cycle like the others:** TypeScript's `Match` type is atomic across the whole file — the compiler won't let `bestMarket()`'s own body keep reading `match.markets` once `Match.candidates` replaces it, so the type change and every one of its consumers (types.ts, `fixtureToMatch`, `applyRecommendation`, `bestMarket`→`resolveRecommendation`, both test files' fixtures) have to land together or nothing in the file compiles. The TDD discipline that survives intact: the new `resolveRecommendation`-specific tests are written first (Step 1, conceptually red — they exercise a function that doesn't exist yet), the full implementation follows in one step (Step 2), and Step 3 is the one and only run, expected green across the board.

- [ ] **Step 1: Write the new `resolveRecommendation` tests**

Add to `app/frontend/components/__tests__/MatchUI.test.tsx` (adjust the import line to add `resolveRecommendation` alongside whatever this file already imports from `@/components/MatchUI` — it will not resolve until Step 2 lands, that's expected):

```typescript
import { resolveRecommendation, type Match, type MarketRec } from "@/components/MatchUI";

const CANDIDATE_HOME: MarketRec = {
  market: "result_3way", selection: "home", recommendationType: "direct_bet",
  currentOdds: 2.1, minOdds: 1.8, mlProbability: 0.55, impliedProbability: 0.48, valueEdge: 0.07,
};
const CANDIDATE_BTTS: MarketRec = {
  market: "btts", selection: "no", recommendationType: "direct_bet",
  currentOdds: 2.2, minOdds: 1.8, mlProbability: 0.6, impliedProbability: 0.45, valueEdge: 0.15,
};

function _matchWith(candidates: MarketRec[], recommendationPick: { market: string; selection: string } | null): Match {
  return {
    id: "m1", league: "E0", tier: "competition_specific", kickoffIso: "2026-06-15T15:00:00Z",
    home: "Arsenal", away: "Chelsea", status: "upcoming", hasRecommendation: true,
    overall: "direct_bet", confidence: "medium", candidates, recommendationPick,
    explanation: [], limitations: [], predictionBasis: "team_history_and_market",
    coldStartRisk: false, featureCompleteness: null, unknownTeam: false, invalidMarketCount: 0,
  };
}

describe("resolveRecommendation", () => {
  test("finds the matching candidate", () => {
    const match = _matchWith([CANDIDATE_HOME, CANDIDATE_BTTS], { market: "btts", selection: "no" });
    expect(resolveRecommendation(match)).toEqual(CANDIDATE_BTTS);
  });

  test("returns undefined for a null pick", () => {
    const match = _matchWith([CANDIDATE_HOME], null);
    expect(resolveRecommendation(match)).toBeUndefined();
  });

  test("returns undefined when the pick names a candidate not in the list", () => {
    const match = _matchWith([CANDIDATE_HOME], { market: "btts", selection: "yes" });
    expect(resolveRecommendation(match)).toBeUndefined();
  });
});
```

- [ ] **Step 2: Implement everything -- types, behavior, and both test files' own fixtures, together**

In `app/frontend/lib/types.ts`, replace `MarketRecommendationOut`/`MatchRecommendationOut` (currently `~lines 21-56`):

```typescript
export type MarketCandidateOut = {
  market: string;
  selection: string;
  recommendation_type: "direct_bet" | "conditional" | "no_bet";
  current_odds: number | null;
  min_odds: number;
  ml_probability: number;
  implied_probability: number;
  value_edge: number;
  // W84 (app_user_stories.md), agent-side A52 (agent_user_stories.md): the
  // price a "conditional" market would need to reach to clear
  // min_value_edge, code-computed server-side -- null when not applicable
  // (not conditional, no current_odds, or no such target exists) or absent
  // entirely on a pre-A52 cached row, so optional rather than required.
  target_odds?: number | null;
  // A88/W193: the agent's own self-reported edge/hit-probability balance,
  // and the one-line reason this candidate won or lost -- both optional so
  // a pre-this-change cached row (no such keys at all) still type-checks.
  composite_score?: number;
  reason?: string;
};

export type RecommendationPickOut = {
  market: string;
  selection: string;
};

export type MatchRecommendationOut = {
  match: Record<string, unknown>;
  overall: "direct_bet" | "conditional" | "no_bet" | "insufficient_data";
  candidates: MarketCandidateOut[];
  recommendation_pick: RecommendationPickOut | null;
  // One bullet per aspect (value edge, team news, form, market caveats, ...)
  // instead of one narrative paragraph -- direct user request.
  explanation: string[];
  confidence: "low" | "medium" | "high" | string;
  limitations: string[];
  prediction_basis: string;
  invalid_market_count: number;
  // W15: cold_start_risk/unknown_team are first-class trust signals --
  // treat them as authoritative even when prediction_basis itself claims
  // team_history_and_market (see agent_techspec.md / US#108).
  cold_start_risk: boolean;
  feature_completeness: number | null;
  unknown_team: boolean;
  // A82 (agent_user_stories.md): Kelly-derived stake-sizing suggestion for
```

(Leave the remainder of the type — `unit_bet_multiplier` and its closing brace — untouched; only the `markets`/`MarketRecommendationOut` lines above it change.)

In `app/frontend/components/MatchUI.tsx`, replace the `Match`/`MarketRec` types (currently `~lines 60-108`) — `MarketRec` itself is unchanged, only `Match`:

```typescript
export type MarketRec = {
  market: string;
  selection: string;
  recommendationType: RecommendationType;
  currentOdds: number | null;
  minOdds: number;
  mlProbability: number;
  impliedProbability: number;
  valueEdge: number;
  targetOdds?: number | null;
};

export type RecommendationPick = { market: string; selection: string };

export type Match = {
  id: string;
  league: string;
  tier: Tier;
  kickoffIso: string;
  home: string;
  away: string;
  status: "upcoming" | "live" | "completed";
  result?: { home: number; away: number };
  hasRecommendation: boolean;
  overall: Overall;
  confidence: Confidence;
  candidates: MarketRec[];
  recommendationPick: RecommendationPick | null;
  explanation: string[];
  limitations: string[];
  predictionBasis: string;
  coldStartRisk: boolean;
  featureCompleteness: number | null;
  unknownTeam: boolean;
  invalidMarketCount: number;
  unitBetMultiplier?: number | null;
};
```

`fixtureToMatch` (currently `~line 166`): `markets: [],` becomes `candidates: [], recommendationPick: null,`.

`applyRecommendation` (currently `~lines 177-203`):

```typescript
function applyRecommendation(match: Match, rec: MatchRecommendationOut): Match {
  return {
    ...match,
    hasRecommendation: true,
    overall: rec.overall,
    confidence: rec.confidence,
    predictionBasis: rec.prediction_basis,
    explanation: rec.explanation,
    limitations: rec.limitations,
    coldStartRisk: rec.cold_start_risk,
    featureCompleteness: rec.feature_completeness,
    unknownTeam: rec.unknown_team,
    unitBetMultiplier: rec.unit_bet_multiplier ?? null,
    invalidMarketCount: rec.invalid_market_count,
    recommendationPick: rec.recommendation_pick,
    candidates: rec.candidates.map((c) => ({
      market: c.market,
      selection: c.selection,
      recommendationType: c.recommendation_type,
      currentOdds: c.current_odds,
      minOdds: c.min_odds,
      mlProbability: c.ml_probability,
      impliedProbability: c.implied_probability,
      valueEdge: c.value_edge,
      targetOdds: c.target_odds ?? null,
    })),
  };
}
```

Replace `bestMarket`/`hasPositiveEdge`/`marketDirections` (currently `~lines 467-501`) — `hasPositiveEdge` is kept (still a real, used export), `marketDirections` is deleted (zero real callers, confirmed live: only its own definition matched a repo-wide grep):

```typescript
/** W193 (2026-09-01 design): TS port of resolve_recommendation_pick()
 * (src/agent/market_resolution.py) -- same three-case contract: the
 * matching candidate, or undefined when recommendationPick is null OR
 * names a market/selection absent from candidates (a dangling pointer).
 * Replaces bestMarket()'s own max(valueEdge) reduction now that the
 * backend already resolved which candidate is the pick -- there is
 * nothing left to rank client-side. */
export function resolveRecommendation(match: Match): MarketRec | undefined {
  if (!match.recommendationPick) return undefined;
  return match.candidates.find(
    (c) => c.market === match.recommendationPick!.market && c.selection === match.recommendationPick!.selection
  );
}

/** Mockup point 3: backs Daily Edges' "N with positive edge" summary line.
 * Same predicate as the "Positive Edge" tag/green edge coloring on
 * MatchCard itself (recommendationType !== "no_bet" && valueEdge >= 0) --
 * kept as one shared function rather than a third inline copy of that
 * condition. */
export function hasPositiveEdge(match: Match): boolean {
  const m = resolveRecommendation(match);
  return !!m && m.currentOdds != null && m.recommendationType !== "no_bet" && m.valueEdge >= 0;
}
```

Update the 4 call sites: `~line 488` (inside `hasPositiveEdge`, already shown above), `~line 828`, `~line 1885`, `~line 1994` — each `bestMarket(match)`/`bestMarket(m)` becomes `resolveRecommendation(match)`/`resolveRecommendation(m)` (identical call shape, no other change at any of these sites).

The render site at `~line 2069`: `match.markets.map((m, i) => (` becomes `match.candidates.map((m, i) => (`.

The second `Match`-literal construction at `~line 1952` (inside a component's inline placeholder object, mirroring `fixtureToMatch`'s own shape): `markets: [],` becomes `candidates: [], recommendationPick: null,`.

Finally, in the same step: both `MatchUI.test.tsx` and `MatchUI.hitMiss.test.tsx` build `Match` object literals directly elsewhere in each file (no shared `_VALID`-style helper — grep each file for `markets:` to find every one, beyond the new tests Step 1 already added). Apply the same two substitutions to every one of those pre-existing literals: `markets: [...]` → `candidates: [...], recommendationPick: <pointer to whichever candidate that test's own assertions are about, or null for an empty/no-bet case>`; any direct `bestMarket(...)` call in test code itself becomes `resolveRecommendation(...)`. This has to land in this same step, not a later one -- the file won't compile with `Match.markets` gone until every literal is updated.

- [ ] **Step 3: Run tests to verify they pass**

Run: `npx jest MatchUI.test.tsx MatchUI.hitMiss.test.tsx -v`
Expected: PASS, every test in both files -- this is the one and only test run for this task, per the note at the top explaining why.

- [ ] **Step 4: Commit**

```bash
git add app/frontend/lib/types.ts app/frontend/components/MatchUI.tsx app/frontend/components/__tests__/MatchUI.test.tsx app/frontend/components/__tests__/MatchUI.hitMiss.test.tsx
git commit -m "feat(app): W196 -- MatchUI.tsx reads candidates/recommendationPick

resolveRecommendation() replaces bestMarket()'s reduction with a direct
lookup, mirroring resolve_recommendation_pick(). marketDirections()
deleted (confirmed zero real callers). Model Probabilities table and both
headline/verdict call sites switched over -- no visual change.

Co-Authored-By: Claude Sonnet 5 <noreply@anthropic.com>"
```

---

### Task 5: `dashboardMetrics.ts`

**Files:**
- Modify: `app/frontend/lib/dashboardMetrics.ts` (`pricedEdge` and the sort comparator inside `sortMatches`, both calling `bestMarket`)
- Modify: `app/frontend/lib/dashboardMetrics.test.ts`

- [ ] **Step 1: Write the failing test**

Add to `app/frontend/lib/dashboardMetrics.test.ts` (adjust the import to add whatever helper the file already uses to build a `Match`, or build one inline matching this file's own existing test style):

```typescript
test("rankTopEdges reads the resolved recommendation, not a value-maximizing reduction", () => {
  const match = {
    ...baseMatch, // reuse whatever base-Match builder this test file already has
    overall: "direct_bet" as const,
    candidates: [
      { market: "result_3way", selection: "home", recommendationType: "direct_bet" as const, currentOdds: 2.1, minOdds: 1.8, mlProbability: 0.55, impliedProbability: 0.48, valueEdge: 0.07 },
      { market: "btts", selection: "no", recommendationType: "direct_bet" as const, currentOdds: 2.2, minOdds: 1.8, mlProbability: 0.6, impliedProbability: 0.45, valueEdge: 0.15 },
    ],
    recommendationPick: { market: "result_3way", selection: "home" },
  };
  const [top] = rankTopEdges([match], 1);
  // The pick is result_3way/home (edge 0.07), even though btts/no has a
  // higher raw edge (0.15) -- proves this reads the resolved pick, not
  // bestMarket()'s old max(valueEdge) behavior.
  expect(top.edge).toBeCloseTo(0.07);
});
```

(If this test file has no existing `baseMatch`-shaped helper, build the object inline with every required `Match` field, matching this plan's Task 4 `_matchWith` helper shape.)

- [ ] **Step 2: Run test to verify it fails**

Run: `npx jest dashboardMetrics.test.ts -t "reads the resolved recommendation"`
Expected: FAIL — `pricedEdge`/`bestMarket` still reduces by max `valueEdge`, so `top.edge` would be `0.15`, not `0.07`.

- [ ] **Step 3: Implement, then rework the file's own fixtures**

Change the import at the top of `app/frontend/lib/dashboardMetrics.ts`:

```typescript
import { resolveRecommendation, dayDiff, type Match, type Overall } from "@/components/MatchUI";
```

`pricedEdge` (wherever `bestMarket(m)` currently appears in this function): `const shown = bestMarket(m);` becomes `const shown = resolveRecommendation(m);`.

The sort comparator inside `sortMatches` (the two calls in the tie-break branch): `bestMarket(a)?.valueEdge ?? -Infinity` / `bestMarket(b)?.valueEdge ?? -Infinity` become `resolveRecommendation(a)?.valueEdge ?? -Infinity` / `resolveRecommendation(b)?.valueEdge ?? -Infinity`.

Rework `dashboardMetrics.test.ts`'s own existing `Match` fixtures the same way Task 4 did (grep for `markets:` in this file, apply the same two substitutions).

- [ ] **Step 4: Run tests to verify they pass**

Run: `npx jest dashboardMetrics.test.ts -v`
Expected: PASS, every test in the file.

- [ ] **Step 5: Commit**

```bash
git add app/frontend/lib/dashboardMetrics.ts app/frontend/lib/dashboardMetrics.test.ts
git commit -m "feat(app): W196 -- dashboardMetrics.ts reads resolveRecommendation

Same swap as MatchUI.tsx's own call sites -- pricedEdge and the sort
comparator's tie-break both read the resolved pick directly.

Co-Authored-By: Claude Sonnet 5 <noreply@anthropic.com>"
```

---

### Task 6: Delete `pick_recommended_market()` and `bestMarket()`/`marketDirections()`

**Files:**
- Modify: `src/agent/market_resolution.py` (delete `pick_recommended_market`)
- Modify: `tests/test_market_resolution.py` (delete its dedicated tests)
- Confirm: `bestMarket()`/`marketDirections()` are already gone from `MatchUI.tsx` (Task 4 deleted them directly, not left for this task — this task just verifies)

- [ ] **Step 1: Confirm zero remaining callers**

Run: `grep -rn "pick_recommended_market" --include="*.py" . | grep -v "\.pyc"`
Expected: no matches outside `src/agent/market_resolution.py`'s own (about-to-be-deleted) definition. If anything else still calls it, STOP -- a caller was missed in an earlier task; do not delete until every real caller is confirmed migrated.

Run: `grep -rn "bestMarket\|marketDirections" --include="*.tsx" --include="*.ts" app/frontend`
Expected: no matches at all (both already deleted from `MatchUI.tsx` in Task 4; `marketDirections` never had real callers to migrate).

- [ ] **Step 2: Delete**

In `src/agent/market_resolution.py`, delete `pick_recommended_market()` in full (the function this codebase's own A81/W151/eod_batch.py comments have referenced throughout — those comments are historical, referring to what the function *did*, not asserting it still exists; leave them as-is, they're accurate history).

In `tests/test_market_resolution.py`, delete the tests exercising `pick_recommended_market` specifically (grep the file for `pick_recommended_market` to find them) and remove it from the file's own import line.

- [ ] **Step 3: Run tests to verify nothing broke**

Run: `python -m pytest tests/test_market_resolution.py -v`
Expected: PASS, every remaining test (the `resolve_recommendation_pick`/`market_correct`/`build_actual_outcome`/`RESOLVABLE_MARKETS` ones, all untouched).

Run: `python -m pytest tests/ app/backend/tests/ -q`
Expected: PASS across both, matching whatever clean baseline Task 1-5 already established (see Task 7 for the exact expected numbers).

- [ ] **Step 4: Commit**

```bash
git add src/agent/market_resolution.py tests/test_market_resolution.py
git commit -m "chore(agent): W197 -- delete pick_recommended_market()

Zero remaining callers after W193-W196 migrated every real consumer
(settlement, bets, MatchUI.tsx, dashboardMetrics.ts) onto
resolve_recommendation_pick(). bestMarket()/marketDirections() were
already deleted directly in W196 (Task 4) once their own call sites
migrated -- this closes the Python side of the same cleanup.

Co-Authored-By: Claude Sonnet 5 <noreply@anthropic.com>"
```

---

### Task 7: Full regression + user-story completion notes

**Files:**
- Modify: `documents/app_user_stories.md` (Phase 47's W193–W197 rows)

- [ ] **Step 1: Run the full suite, both languages**

Run: `python -m pytest tests/ app/backend/tests/ -q 2>&1 | tail -30`
Expected: clean except the session's own already-documented pre-existing `app/backend/tests/test_fixtures_endpoint.py` failures (5, confirmed unrelated across every story this session). If anything else is red, stop and fix it before continuing.

Run: `npx jest 2>&1 | tail -40` (or whatever this project's real frontend test command is — check `package.json`'s `scripts.test` first if unsure)
Expected: clean.

- [ ] **Step 2: Update Phase 47's stories in `documents/app_user_stories.md`**

Change `W193`/`W194`/`W195`/`W196`/`W197`'s `future` status to `completed`, and append a completion note to each `Comments` cell (after the existing `Size ... · Depends on: ...` text), summarizing what actually shipped, naming real test files and the real passing counts from Step 1 -- matching this codebase's own completion-note convention throughout `documents/app_user_stories.md`. `W196` covers both Task 4 and Task 5 (frontend, 3 files) in one story per the original plan -- write one completion note spanning both.

- [ ] **Step 3: Commit**

```bash
git add documents/app_user_stories.md
git commit -m "docs(app): W193-W197 completion notes

Marks Phase 47's five stories completed with real test/suite evidence.

Co-Authored-By: Claude Sonnet 5 <noreply@anthropic.com>"
```
