# Single-Market Recommendation — Live Serving Wiring — Design

**Date:** 2026-09-01
**Status:** approved, pending implementation plan
**Origin:** Direct continuation of `docs/superpowers/specs/2026-08-31-single-market-recommendation-design.md` ("sub-project #2", explicitly deferred there). That spec moved the agent's own decision mechanism (`src/agent/schema.py`, `src/agent/market_resolution.py`, the prompt files) from an independently-scored `markets` array + post-hoc `pick_recommended_market()`/`bestMarket()` reduction to a single `recommendation_pick` resolved against a richer `candidates` list. This spec covers everything downstream of that: the app-layer schema mirror, settlement, bet-logging, and the frontend — the four call sites the final review of sub-project #1 named as a concrete blocker list, confirmed unsafe to leave unmigrated once the app resumes live traffic.

## The problem

`run_agent()` now returns a `candidates`/`recommendation_pick`-shaped dict. Four places still read the old `markets` field and would silently see "no markets, no pick" for every recommendation generated from here on:

- `app/backend/recommendations.py` — its own full parallel schema (`MarketRecommendationOut`/`MatchRecommendationOut`, deliberately looser-typed than the agent-side models) and `validate_and_degrade()`'s per-market validation loop.
- `app/backend/recommendation_outcomes.py` — settlement's `resolve_pending_recommendations()`, via `pick_recommended_market(rec.get("markets") or [])`.
- `app/backend/bets.py` — `resolve_from_recommendation()`'s manual market/selection search.
- The frontend — `MatchUI.tsx`'s `bestMarket()` (the dashboard card headline and the detail page's big verdict) and a third consumer found during this brainstorm, `app/frontend/lib/dashboardMetrics.ts` (the dashboard's own edge-based sort/display, not caught by the original blocker list).

## Old-row compatibility: graceful degrade, no adapter

Already-cached recommendations written before this ships sit in `recommendation_cache.db` under the old `markets` shape. Decision (confirmed): **no migration, no read-time adapter reconstructing `candidates` from the old array.** `validate_and_degrade()` needs no old-row *detection* at all — `raw.get("candidates") or []` and `raw.get("recommendation_pick")` are naturally empty/`None` for an old-shape row, since those keys never existed on it. The one thing that needs adding: since the app layer doesn't re-run the agent's own guardrails, it needs the identical one-line downgrade-only rule A90 already established on the agent side — **if `recommendation_pick` is `None`, cap `overall` at `"no_bet"`** — so an old row can't come back claiming `overall="direct_bet"` with zero candidates behind it. Old cards read as uncommitted/`no_bet` until the scheduler's next real pass (EOD/T-30, within ~24h of the app resuming) regenerates them in the new shape. Matches this codebase's own precedent elsewhere (W176: "an accepted small one-time gap rather than a migration script") — zero new code to remove later, no shim with an unclear expiry.

## App-side schema (`app/backend/recommendations.py`)

Mirrors `MarketCandidateModel`/`RecommendationPickModel`/`MatchRecommendationModel` (`src/agent/schema.py`), but keeps this file's own existing, deliberate looseness (`recommendation_type: str`/`overall: str`, not `Literal`s — a slightly-off cached value degrades instead of the whole API request crashing):

```python
class MarketCandidateOut(BaseModel):
    market: Literal["result_3way", "btts", "total_goals", "home_corners", "away_corners"]
    selection: Literal["home", "draw", "away", "yes", "no", "over_2.5", "under_2.5"]
    recommendation_type: str
    current_odds: float | None
    min_odds: float = 0.0
    ml_probability: float
    implied_probability: float
    value_edge: float
    target_odds: float | None = None
    composite_score: float = 0.0   # defaulted -- a pre-this-change cached row has none
    reason: str = ""                # defaulted, same reason


class RecommendationPickOut(BaseModel):
    market: Literal["result_3way", "btts", "total_goals", "home_corners", "away_corners"]
    selection: Literal["home", "draw", "away", "yes", "no", "over_2.5", "under_2.5"]


class MatchRecommendationOut(BaseModel):
    match: dict
    overall: str
    candidates: list[MarketCandidateOut]
    recommendation_pick: RecommendationPickOut | None = None
    explanation: list[str]
    confidence: str
    limitations: list[str]
    prediction_basis: str
    invalid_market_count: int = 0
    cold_start_risk: bool = False
    feature_completeness: float | None = None
    unknown_team: bool = False
    unit_bet_multiplier: float | None = None
```

`validate_and_degrade()`: the per-market validation loop (`for market in raw.get("markets") or []: ...`) becomes `for candidate in raw.get("candidates") or []: ...`, unchanged otherwise. After building `valid_markets`/`invalid_count`, resolve `recommendation_pick` against the *validated* list the same way the agent side does (`resolve_recommendation_pick()`, reused directly — see below), and apply the one-line cap: `overall = raw.get("overall") or "insufficient_data"`, then if the resolved pick is `None` and `overall` outranks `"no_bet"`, drop it to `"no_bet"`. The BUG-023/024 match-mismatch degenerate return (`markets=[]`) becomes `candidates=[]`, `recommendation_pick=None` — the same one-line fix already applied three times in `graph.py`.

## Settlement (`app/backend/recommendation_outcomes.py`)

One-line swap, reusing the agent-side resolver directly (this file already imports across the `app/backend` → `src/agent` boundary):

```python
# before
picked = pick_recommended_market(rec.get("markets") or [])
# after
picked = resolve_recommendation_pick(rec.get("candidates") or [], rec.get("recommendation_pick"))
```

Everything downstream (`if picked is None or picked.get("market") not in RESOLVABLE_MARKETS: unresolvable_market_count += 1; continue`) is unchanged. An old-shape row naturally resolves to `None` here too and gets skipped via the existing counter — already-correct behavior, no special-casing.

## Bets (`app/backend/bets.py`)

`resolve_from_recommendation()`'s manual `next(m for m in recommendation.get("markets") ...)` search becomes:

```python
picked = resolve_recommendation_pick(
    request.recommendation.get("candidates") or [],
    {"market": request.market, "selection": request.selection},
)
if picked is None:
    raise ValueError(f"Market {request.market!r}/selection {request.selection!r} not found in the given recommendation.")
```

Deliberately constructs a synthetic pointer from the request's own `market`/`selection` rather than trusting `recommendation.get("recommendation_pick")` — preserves today's permissive behavior (a user can log a bet on any listed candidate, not only the actual `recommendation_pick`) rather than silently narrowing what's loggable. Not this sub-project's call to tighten.

## Frontend — three files

- **`MatchUI.tsx`**: `Match.markets: MarketRec[]` → `Match.candidates: MarketRec[]` + `Match.recommendationPick: { market; selection } | null`. New `resolveRecommendation(match)` — a TS port of `resolve_recommendation_pick()`, same three-case contract (found / null pick / dangling pick returns `undefined`). `bestMarket()` and the already-dead `marketDirections()` deleted; both call sites (`MatchCard`'s headline, the detail page's big verdict) call `resolveRecommendation()` instead. The "Model Probabilities" table (`match.markets.map(...)`) becomes `match.candidates.map(...)` — same rendering, no visual change.
- **`dashboardMetrics.ts`**: its two `bestMarket(...)` calls (display edge, sort comparator) become `resolveRecommendation(...)?.valueEdge`, same semantics — found during this brainstorm, not in the original blocker list.

Keeping `resolveRecommendation()`/`resolve_recommendation_pick()` as two hand-ported implementations (Python + TS) rather than eliminating cross-language duplication entirely is accepted: the frontend has no way to avoid needing *some* lookup (a not-yet-settled match has no server-side resolution to source it from), so the realistic fix is keeping both ports honest and simple (a five-line equality scan, not a value-maximizing reduction), not eliminating the need for a frontend-side lookup altogether.

## Deletion

`pick_recommended_market()` (`src/agent/market_resolution.py`) and `bestMarket()`/`marketDirections()` (`MatchUI.tsx`) deleted once the call sites above no longer reference them — closes the "two hand-maintained implementations that could silently drift" gap identified earlier this session for good.

## Testing

TDD, matching this codebase's convention:

- **App schema**: `MatchRecommendationOut`/`MarketCandidateOut` field tests; `validate_and_degrade()` covered for the new-shape happy path, the old-row degrade case (empty `candidates`, capped `overall`), and the match-mismatch degenerate path's new field names.
- **Settlement/bets**: existing test files reworked to the `candidates`/`recommendation_pick` fixture shape, same substitution pattern Tasks 3/4 already used on the agent side.
- **Frontend**: a new test for `resolveRecommendation()` mirroring `resolve_recommendation_pick()`'s three cases; existing `MatchCard`/detail-page/`dashboardMetrics` tests reworked to the new `Match` shape.

## Explicitly out of scope here

- The backtest/train harness (`BacktestRecord`, `evaluation.py`, `staking.py`, `live_lessons.py`'s historical adapter, `agent-backtest`/`agent-train`) — sub-project #3, per the original spec's own scoping.
