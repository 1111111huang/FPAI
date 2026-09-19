# Home/Away Goals Market Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add `home_goals`/`away_goals` (fixed 1.5-goal-line "team totals") as real, recommendable, gradable betting markets — live odds via The Odds API, backtest/agent-train odds via OddsPapi (data already pulled) — matching every existing market's schema/guardrail/grading/frontend treatment.

**Architecture:** Extends the existing `total_goals`/corners plumbing rather than building anything new: `src/agent/schema.py`'s market/selection enums and conditional-eligibility set gain two entries; `app/backend/odds_api_client.py` gains a `team_totals`-parsing branch reusing `get_event_odds()`'s existing per-event-endpoint mechanism with a wider region override; `src/agent/market_resolution.py` (shared by live settlement and backtest) gains two grading branches; `app/frontend/components/MatchUI.tsx` mirrors those in TypeScript. No ML changes: `home_goals`/`away_goals` are already-active forecast targets whose Poisson distributions already reach the LLM's prompt.

**Tech Stack:** Python (FastAPI backend, LangGraph agent), TypeScript/React (Next.js frontend), pytest, Jest.

---

## Before you start

Read the approved design spec in full: `docs/superpowers/specs/2026-09-19-home-away-goals-market-design.md`. This plan implements it section-by-section; task numbers below map to that doc's section numbers in parentheses.

Two tasks in this plan (Task 5 and Task 10) are **live investigation scripts** that hit real, paid third-party APIs (The Odds API, OddsPapi) using real API keys from `.env`. Their exact printed output can't be known until you run them — that's the point. Do not skip running them, and do not guess their output; the tasks after them depend on what they actually print.

---

### Task 1: Schema & guardrails (design §5)

**Files:**
- Modify: `src/agent/schema.py:19-21,47-48,114-115,156-157,355-361`
- Test: `tests/test_agent_schema_validation.py`
- Test: `tests/test_agent_conditional_market_eligibility.py`

- [ ] **Step 1: Write the failing tests**

Add to `tests/test_agent_schema_validation.py` (near `test_total_corners_over_9_5_is_a_valid_market_and_selection`, around line 198):

```python
@pytest.mark.parametrize("market", ["home_goals", "away_goals"])
@pytest.mark.parametrize("selection", ["over_1.5", "under_1.5"])
def test_home_away_goals_is_a_valid_market_and_selection(market, selection):
    """Team-goals-total market (W199): a single team's own goal count,
    fixed at the 1.5 line -- same fixed-line convention as total_goals'
    2.5 and total_corners' 9.5."""
    candidate = {**_VALID_CANDIDATE, "market": market, "selection": selection, "recommendation_type": "no_bet", "current_odds": None}
    good = {**_VALID, "overall": "no_bet", "candidates": [candidate], "recommendation_pick": None}
    rec = extract_recommendation(_wrap_json(good))
    assert rec["candidates"][0]["market"] == market
    assert rec["candidates"][0]["selection"] == selection
```

Add to `tests/test_agent_conditional_market_eligibility.py`: extend the two existing `@pytest.mark.parametrize` lists (do not add new test functions — DRY, these two tests already parametrize exactly this shape):

```python
@pytest.mark.parametrize(
    "market,selection",
    [
        ("total_goals", "over_2.5"),
        ("home_corners", "over_2.5"),
        ("away_corners", "over_2.5"),
        ("btts", "yes"),
        ("home_goals", "over_1.5"),
        ("away_goals", "over_1.5"),
    ],
)
def test_eligible_market_stays_conditional_after_a29_ceiling_downgrade(market, selection):
    ...  # body unchanged
```

```python
@pytest.mark.parametrize(
    "market,selection",
    [
        ("result_3way", "home"),
        ("result_3way", "draw"),
        ("result_3way", "away"),
        ("total_goals", "under_2.5"),
        ("home_corners", "under_2.5"),
        ("away_corners", "under_2.5"),
        ("btts", "no"),
        ("home_goals", "under_1.5"),
        ("away_goals", "under_1.5"),
    ],
)
def test_ineligible_market_downgraded_to_no_bet_after_a29_downgrade(market, selection):
    ...  # body unchanged
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_agent_schema_validation.py::test_home_away_goals_is_a_valid_market_and_selection tests/test_agent_conditional_market_eligibility.py -v`
Expected: the four new `test_home_away_goals_is_a_valid_market_and_selection` cases FAIL with a Pydantic `ValidationError` (`market`/`selection` not in the current Literal); the two new parametrized cases in `test_agent_conditional_market_eligibility.py` also FAIL the same way.

- [ ] **Step 3: Implement the schema change**

In `src/agent/schema.py`, update all four occurrences of the `market`/`selection` `Literal`s (lines 19-20, 47-48, 114-115, 156-157 — `MarketCandidate`, `RecommendationPick`, `MarketCandidateModel`, `RecommendationPickModel`):

```python
    market: Literal["result_3way", "btts", "total_goals", "home_corners", "away_corners", "total_corners", "home_goals", "away_goals"]
    selection: Literal["home", "draw", "away", "yes", "no", "over_2.5", "under_2.5", "over_9.5", "under_9.5", "over_1.5", "under_1.5"]
```

Update `_CONDITIONAL_ELIGIBLE_MARKETS` (lines 355-361):

```python
_CONDITIONAL_ELIGIBLE_MARKETS = frozenset({
    ("total_goals", "over_2.5"),
    ("home_corners", "over_2.5"),
    ("away_corners", "over_2.5"),
    ("btts", "yes"),
    ("total_corners", "over_9.5"),  # A101
    ("home_goals", "over_1.5"),  # W199
    ("away_goals", "over_1.5"),  # W199
})
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_agent_schema_validation.py tests/test_agent_conditional_market_eligibility.py tests/test_agent_schema.py -v`
Expected: all PASS.

- [ ] **Step 5: Commit**

```bash
git add src/agent/schema.py tests/test_agent_schema_validation.py tests/test_agent_conditional_market_eligibility.py
git commit -m "feat(agent): add home_goals/away_goals market to schema and conditional eligibility (W199)"
```

---

### Task 2: Prompt vocabulary (design §5)

**Files:**
- Modify: `config/prompts/agent_v1.txt:18,54,101-102`
- Modify: `config/prompts/agent_v1_aggressive.txt` (same lines, +1 offset)
- Modify: `config/prompts/agent_v1_balanced.txt` (same lines, +1 offset)
- Modify: `config/prompts/agent_v1_conservative.txt` (same lines, +1 offset)

The four prompt files are near-identical copies (confirmed: the market-vocabulary lines below are byte-identical across all four) — apply the same three edits to each file.

- [ ] **Step 1: Edit the Evidence Priority section**

In each of the 4 files, change:

```
- **home_corners / away_corners / total_corners**: wing-play tactics and game-state dynamics (e.g. a trailing team pressing late tends to win more corners).
```

to:

```
- **home_corners / away_corners / total_corners**: wing-play tactics and game-state dynamics (e.g. a trailing team pressing late tends to win more corners).
- **home_goals / away_goals**: that team's own attacking output specifically -- chance-creation/conversion efficiency for that side alone, and defensive/goalkeeper absences on the OTHER side (a weaker opposing defense raises this team's own goal count even if the match overall isn't high-scoring). Same priority ordering rationale as total_goals, applied per side.
```

- [ ] **Step 2: Edit the conditional-eligibility rule**

In each of the 4 files, change:

```
- Only use "conditional" for total_goals/over_2.5, home_corners/over_2.5, away_corners/over_2.5, total_corners/over_9.5, or btts/yes — the markets where waiting for a better price is a real, directional strategy. Never use "conditional" for result_3way, under_2.5, or btts/no — use "no_bet" instead if value exists but the price isn't right. This is code-enforced — a "conditional" call on any other market/selection will be downgraded to "no_bet" automatically.
```

to:

```
- Only use "conditional" for total_goals/over_2.5, home_corners/over_2.5, away_corners/over_2.5, total_corners/over_9.5, home_goals/over_1.5, away_goals/over_1.5, or btts/yes — the markets where waiting for a better price is a real, directional strategy. Never use "conditional" for result_3way, under_2.5, or btts/no — use "no_bet" instead if value exists but the price isn't right. This is code-enforced — a "conditional" call on any other market/selection will be downgraded to "no_bet" automatically.
```

- [ ] **Step 3: Edit the market/selection enum in the output-format JSON block**

In each of the 4 files, change:

```
      "market": "<result_3way | btts | total_goals | home_corners | away_corners | total_corners>",
      "selection": "<home | draw | away | yes | no | over_2.5 | under_2.5 | over_9.5 | under_9.5>",
```

to:

```
      "market": "<result_3way | btts | total_goals | home_corners | away_corners | total_corners | home_goals | away_goals>",
      "selection": "<home | draw | away | yes | no | over_2.5 | under_2.5 | over_9.5 | under_9.5 | over_1.5 | under_1.5>",
```

- [ ] **Step 4: Verify all 4 files were updated identically**

Run: `grep -c "home_goals / away_goals" config/prompts/agent_v1.txt config/prompts/agent_v1_aggressive.txt config/prompts/agent_v1_balanced.txt config/prompts/agent_v1_conservative.txt`
Expected: `1` for each of the 4 files.

Run: `grep -c "home_goals/over_1.5, away_goals/over_1.5" config/prompts/agent_v1.txt config/prompts/agent_v1_aggressive.txt config/prompts/agent_v1_balanced.txt config/prompts/agent_v1_conservative.txt`
Expected: `1` for each of the 4 files.

- [ ] **Step 5: Run the existing prompt-consistency test suite (regression check)**

Run: `python -m pytest tests/test_agent_prompt_thresholds.py -v`
Expected: all PASS (these edits don't touch any `{{PLACEHOLDER}}` token, so nothing here should be affected).

- [ ] **Step 6: Commit**

```bash
git add config/prompts/agent_v1.txt config/prompts/agent_v1_aggressive.txt config/prompts/agent_v1_balanced.txt config/prompts/agent_v1_conservative.txt
git commit -m "feat(agent): document home_goals/away_goals market in all prompt variants (W199)"
```

---

### Task 3: Grading — Python (design §6)

**Files:**
- Modify: `src/agent/market_resolution.py:21,24-47,50-84`
- Test: `tests/test_market_resolution.py`

- [ ] **Step 1: Write the failing tests**

Add to `tests/test_market_resolution.py`:

```python
def test_resolvable_markets_includes_home_and_away_goals():
    assert RESOLVABLE_MARKETS == {
        "result_3way", "btts", "total_goals", "total_corners", "home_goals", "away_goals",
    }


def test_build_actual_outcome_includes_home_and_away_goals_side_unconditionally():
    """Unlike total_corners (optional -- not every settlement source has
    corner counts), home_goals/away_goals_side is unconditional: home_goals
    and away_goals are this function's own required positional params, so
    there's no missing-data case to guard against."""
    actual = build_actual_outcome(2, 1)
    assert actual["home_goals_side"] == "over_1.5"
    assert actual["away_goals_side"] == "under_1.5"


def test_build_actual_outcome_home_goals_side_under_on_exactly_one():
    actual = build_actual_outcome(1, 0)
    assert actual["home_goals_side"] == "under_1.5"
    assert actual["away_goals_side"] == "under_1.5"


def test_market_correct_home_goals():
    actual = build_actual_outcome(2, 1)
    assert market_correct({"market": "home_goals", "selection": "over_1.5"}, actual) is True
    assert market_correct({"market": "home_goals", "selection": "under_1.5"}, actual) is False


def test_market_correct_away_goals():
    actual = build_actual_outcome(2, 1)
    assert market_correct({"market": "away_goals", "selection": "under_1.5"}, actual) is True
    assert market_correct({"market": "away_goals", "selection": "over_1.5"}, actual) is False
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_market_resolution.py -v`
Expected: the 5 new tests FAIL — `RESOLVABLE_MARKETS` mismatch, then `KeyError`/`None`-vs-`True` for the others.

- [ ] **Step 3: Implement**

In `src/agent/market_resolution.py`, update `RESOLVABLE_MARKETS` (line 21):

```python
RESOLVABLE_MARKETS = {"result_3way", "btts", "total_goals", "total_corners", "home_goals", "away_goals"}
```

Update `market_correct()` (add two branches before the final `total_goals` fallback, lines 40-47):

```python
    if market == "result_3way":
        return selection == actual["result"]
    if market == "btts":
        return selection == actual["btts"]
    if market == "total_corners":
        side = actual.get("total_corners_side")
        return None if side is None else selection == side
    if market == "home_goals":
        return selection == actual["home_goals_side"]
    if market == "away_goals":
        return selection == actual["away_goals_side"]
    return selection == actual["total_goals_side"]  # market == "total_goals"
```

Update `build_actual_outcome()` (lines 64-84) — add the two new keys unconditionally, right after `total_goals_side`:

```python
    home_goals, away_goals = int(home_goals), int(away_goals)
    if home_goals > away_goals:
        result = "home"
    elif home_goals < away_goals:
        result = "away"
    else:
        result = "draw"
    total_goals = home_goals + away_goals
    outcome = {
        "fthg": home_goals,
        "ftag": away_goals,
        "result": result,
        "btts": "yes" if (home_goals > 0 and away_goals > 0) else "no",
        "total_goals": total_goals,
        "total_goals_side": "over_2.5" if total_goals > 2 else "under_2.5",
        "home_goals_side": "over_1.5" if home_goals > 1 else "under_1.5",
        "away_goals_side": "over_1.5" if away_goals > 1 else "under_1.5",
    }
    if home_corners is not None and away_corners is not None:
        total_corners = int(home_corners) + int(away_corners)
        outcome["total_corners"] = total_corners
        outcome["total_corners_side"] = "over_9.5" if total_corners > 9 else "under_9.5"
    return outcome
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_market_resolution.py -v`
Expected: all PASS, including every pre-existing test in this file (unaffected).

- [ ] **Step 5: Run the wider test suite that consumes this module (regression check)**

Run: `python -m pytest tests/test_backtest.py tests/test_agent_schema_validation.py -v`
Expected: all PASS (both modules call `build_actual_outcome`/`market_correct` — confirms the new unconditional keys don't break anything downstream).

- [ ] **Step 6: Commit**

```bash
git add src/agent/market_resolution.py tests/test_market_resolution.py
git commit -m "feat(agent): grade home_goals/away_goals in market_resolution (W199)"
```

---

### Task 4: Grading & labels — frontend (design §6)

**Files:**
- Modify: `app/frontend/components/MatchUI.tsx:586,588-610,616-622,2442-2452,2596-2603`
- Test: `app/frontend/components/__tests__/MatchUI.hitMiss.test.tsx`

- [ ] **Step 1: Write the failing tests**

Add to `app/frontend/components/__tests__/MatchUI.hitMiss.test.tsx`, inside the existing `describe("buildActualOutcome ...")` block:

```typescript
  it("W199: includes homeGoalsSide/awayGoalsSide unconditionally (unlike total_corners, home/away goals are always known)", () => {
    const actual = buildActualOutcome(2, 1);
    expect(actual.homeGoalsSide).toBe("over_1.5");
    expect(actual.awayGoalsSide).toBe("under_1.5");
  });

  it("W199: under_1.5 on exactly one goal", () => {
    const actual = buildActualOutcome(1, 0);
    expect(actual.homeGoalsSide).toBe("under_1.5");
    expect(actual.awayGoalsSide).toBe("under_1.5");
  });
```

And inside the existing `describe("marketCorrect ...")` block:

```typescript
  it("W199: home_goals resolves against the actual side", () => {
    const actual = buildActualOutcome(2, 1);
    expect(marketCorrect("home_goals", "over_1.5", actual)).toBe(true);
    expect(marketCorrect("home_goals", "under_1.5", actual)).toBe(false);
  });

  it("W199: away_goals resolves against the actual side", () => {
    const actual = buildActualOutcome(2, 1);
    expect(marketCorrect("away_goals", "under_1.5", actual)).toBe(true);
    expect(marketCorrect("away_goals", "over_1.5", actual)).toBe(false);
  });
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd app/frontend && npx vitest run MatchUI.hitMiss.test.tsx`
Expected: the 4 new tests FAIL (`homeGoalsSide`/`awayGoalsSide` undefined; `marketCorrect` returns `null`).

- [ ] **Step 3: Implement the TS mirror**

In `app/frontend/components/MatchUI.tsx`, update `RESOLVABLE_MARKETS` (line 586):

```typescript
const RESOLVABLE_MARKETS = new Set(["result_3way", "btts", "total_goals", "total_corners", "home_goals", "away_goals"]);
```

Update the `ActualOutcome` type (lines 588-594):

```typescript
export type ActualOutcome = {
  result: "home" | "away" | "draw";
  btts: "yes" | "no";
  totalGoalsSide: "over_2.5" | "under_2.5";
  homeGoalsSide: "over_1.5" | "under_1.5";
  awayGoalsSide: "over_1.5" | "under_1.5";
  totalCorners?: number;
  totalCornersSide?: "over_9.5" | "under_9.5";
};
```

Update `buildActualOutcome()` (lines 596-610):

```typescript
export function buildActualOutcome(home: number, away: number, homeCorners?: number, awayCorners?: number): ActualOutcome {
  const result = home > away ? "home" : home < away ? "away" : "draw";
  const totalGoals = home + away;
  const outcome: ActualOutcome = {
    result,
    btts: home > 0 && away > 0 ? "yes" : "no",
    totalGoalsSide: totalGoals > 2 ? "over_2.5" : "under_2.5",
    homeGoalsSide: home > 1 ? "over_1.5" : "under_1.5",
    awayGoalsSide: away > 1 ? "over_1.5" : "under_1.5",
  };
  if (homeCorners !== undefined && awayCorners !== undefined) {
    const totalCorners = homeCorners + awayCorners;
    outcome.totalCorners = totalCorners;
    outcome.totalCornersSide = totalCorners > 9 ? "over_9.5" : "under_9.5";
  }
  return outcome;
}
```

Update `marketCorrect()` (lines 616-622):

```typescript
export function marketCorrect(market: string, selection: string, actual: ActualOutcome): boolean | null {
  if (!RESOLVABLE_MARKETS.has(market)) return null;
  if (market === "result_3way") return selection === actual.result;
  if (market === "btts") return selection === actual.btts;
  if (market === "total_corners") return actual.totalCornersSide === undefined ? null : selection === actual.totalCornersSide;
  if (market === "home_goals") return selection === actual.homeGoalsSide;
  if (market === "away_goals") return selection === actual.awayGoalsSide;
  return selection === actual.totalGoalsSide; // market === "total_goals"
}
```

Update `MARKET_LABEL` (lines 2596-2603):

```typescript
const MARKET_LABEL: Record<string, { label: string; subtitle: string }> = {
  result_3way: { label: "3-Way Result", subtitle: "Full Time" },
  total_goals: { label: "Over/Under", subtitle: "Full Time" },
  btts: { label: "Both Teams to Score", subtitle: "Full Time" },
  home_corners: { label: "Home Corners", subtitle: "Full Time" },
  away_corners: { label: "Away Corners", subtitle: "Full Time" },
  total_corners: { label: "Total Corners", subtitle: "Full Time" }, // A101
  home_goals: { label: "Home Goals", subtitle: "Full Time" }, // W199
  away_goals: { label: "Away Goals", subtitle: "Full Time" }, // W199
};
```

Update `_MARKET_SELECTION_TITLE` (lines 2442-2452):

```typescript
const _MARKET_SELECTION_TITLE: Record<string, string> = {
  "result_3way:home": "Home Win",
  "result_3way:draw": "Draw",
  "result_3way:away": "Away Win",
  "btts:yes": "BTTS Yes",
  "btts:no": "BTTS No",
  "total_goals:over_2.5": "Over 2.5",
  "total_goals:under_2.5": "Under 2.5",
  "total_corners:over_9.5": "Corners Over 9.5",
  "total_corners:under_9.5": "Corners Under 9.5",
  "home_goals:over_1.5": "Over 1.5",
  "home_goals:under_1.5": "Under 1.5",
  "away_goals:over_1.5": "Over 1.5",
  "away_goals:under_1.5": "Under 1.5",
};
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd app/frontend && npx vitest run MatchUI.hitMiss.test.tsx`
Expected: all PASS.

- [ ] **Step 5: Run the full MatchUI test suite (regression check)**

Run: `cd app/frontend && npx vitest run MatchUI`
Expected: all PASS (the new `ActualOutcome` fields are non-optional additions consumed only by the new branches above — every existing call site of `buildActualOutcome`/`marketCorrect` is unaffected).

- [ ] **Step 6: Commit**

```bash
git add app/frontend/components/MatchUI.tsx app/frontend/components/__tests__/MatchUI.hitMiss.test.tsx
git commit -m "feat(app): grade and label home_goals/away_goals in MatchUI (W199)"
```

---

### Task 5: Live verification of The Odds API's `team_totals` shape (design §2)

**This is a real, one-off, network-calling investigation script — not unit-testable.** Its purpose is to capture the true JSON shape before Task 6 writes a parser against it, matching this codebase's established "confirmed live, not assumed" practice (see `app/backend/odds_api_client.py`'s own W164 module docstring for precedent).

**Files:**
- Create: `scripts/verify_odds_api_team_totals.py`

- [ ] **Step 1: Write the script**

```python
"""One-off investigation script (W199): capture a real team_totals response
from The Odds API before writing a parser against it -- this codebase's
established practice (see app/backend/odds_api_client.py's W164 docstring)
is to never ship a new odds-JSON parser against an assumed shape.

Usage: python scripts/verify_odds_api_team_totals.py
Requires ODDS_API_KEY in .env (this makes one real, billed API call plus
one bulk-odds call to find a live event_id -- ~4-7 credits total against
the free-tier 500/month budget)."""
from __future__ import annotations

import json
import os
import sys

import requests
from dotenv import load_dotenv

load_dotenv()

API_KEY = os.environ.get("ODDS_API_KEY")
if not API_KEY:
    sys.exit("No API key: set ODDS_API_KEY in .env")

BASE_URL = "https://api.the-odds-api.com/v4"
SPORT_KEY = "soccer_epl"


def main() -> None:
    bulk_resp = requests.get(
        f"{BASE_URL}/sports/{SPORT_KEY}/odds",
        params={"apiKey": API_KEY, "regions": "uk", "markets": "h2h", "oddsFormat": "decimal"},
        timeout=10,
    )
    bulk_resp.raise_for_status()
    events = bulk_resp.json()
    if not events:
        sys.exit("No upcoming EPL events found -- try again closer to a matchday.")

    event = events[0]
    print(f"Using event: {event['home_team']} v {event['away_team']} ({event['id']})")

    event_resp = requests.get(
        f"{BASE_URL}/sports/{SPORT_KEY}/events/{event['id']}/odds",
        params={"apiKey": API_KEY, "regions": "uk,us,us2", "markets": "team_totals", "oddsFormat": "decimal"},
        timeout=10,
    )
    event_resp.raise_for_status()
    payload = event_resp.json()

    print("\n=== Full response ===")
    print(json.dumps(payload, indent=2))

    print("\n=== team_totals markets found, by bookmaker ===")
    for bookmaker in payload.get("bookmakers", []):
        for market in bookmaker.get("markets", []):
            if market.get("key") == "team_totals":
                print(f"\nBookmaker: {bookmaker['key']}")
                for outcome in market.get("outcomes", []):
                    print(f"  {outcome}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Run it and record the real outcome shape**

Run: `python scripts/verify_odds_api_team_totals.py`

Expected: prints a real `team_totals` market with outcomes. **Read the printed outcome objects carefully** — each one has a `"name"` ("Over"/"Under"), a `"point"` (the line), a `"price"`, and (per The Odds API's documented convention for other per-entity markets) very likely a `"description"` field naming which team the line belongs to. Confirm:
1. The exact key name for the team identifier (assumed `"description"` below — if the real output uses a different key, e.g. `"participant"`, use that key instead everywhere in Task 6).
2. Whether team names in that field match `home_team`/`away_team` from the same payload's top level exactly, or need the same `TeamNameMapper` normalization already used elsewhere.
3. Whether a 1.5 line is actually offered (if not, note the closest available line — the design's fixed-1.5 convention still applies, this only affects real-world coverage, not the code).

If no UK/US/US2 bookmaker prices `team_totals` on this particular fixture, re-run against a different sport_key (e.g. `soccer_spain_la_liga`) or try again closer to kickoff — bookmakers often only post secondary markets within a day or two of the match.

This step has no fixed pass/fail assertion — it is complete once you have a real captured JSON sample and have confirmed the team-identifier field name for Task 6.

- [ ] **Step 3: Commit the script (not its output — it makes live paid API calls, so it's not run in CI)**

```bash
git add scripts/verify_odds_api_team_totals.py
git commit -m "chore: add one-off investigation script for The Odds API's team_totals shape (W199)"
```

---

### Task 6: The Odds API client — `team_totals` parsing (design §3)

**Depends on Task 5's findings.** The code below assumes outcomes carry `{"name": "Over"/"Under", "description": "<team name>", "point": 1.5, "price": ...}` — The Odds API's documented convention for other per-entity markets (e.g. player props). **If Task 5's real capture used a different field name for the team identifier, substitute that field name everywhere `description` appears below, in both the implementation and the test fixture.**

**Files:**
- Modify: `app/backend/odds_api_client.py:48-61,107-141,210-296`
- Test: `app/backend/tests/test_odds_api_client.py`

- [ ] **Step 1: Write the failing tests**

Add to `app/backend/tests/test_odds_api_client.py`:

```python
def test_normalize_secondary_reads_team_totals_at_the_1_5_line() -> None:
    """W199: team_totals outcomes are tagged per-team via a `description`
    field (The Odds API's documented convention for other per-entity
    markets) -- confirmed live in scripts/verify_odds_api_team_totals.py
    before this was written. Matched against home_team/away_team the same
    way _normalize()'s own h2h parsing matches by name."""
    payload = {
        "home_team": "Arsenal",
        "away_team": "Everton",
        "bookmakers": [
            {"key": "draftkings", "markets": [
                {"key": "team_totals", "outcomes": [
                    {"name": "Over", "description": "Arsenal", "price": 1.83, "point": 1.5},
                    {"name": "Under", "description": "Arsenal", "price": 1.95, "point": 1.5},
                    {"name": "Over", "description": "Everton", "price": 2.20, "point": 1.5},
                    {"name": "Under", "description": "Everton", "price": 1.65, "point": 1.5},
                ]},
            ]},
        ],
    }

    result = _normalize_secondary(payload, home_team="Arsenal", away_team="Everton")

    assert result.home_goals == {"over_1.5": 1.83, "under_1.5": 1.95}
    assert result.away_goals == {"over_1.5": 2.20, "under_1.5": 1.65}


def test_normalize_secondary_ignores_non_1_5_team_totals_lines() -> None:
    payload = {
        "home_team": "Arsenal", "away_team": "Everton",
        "bookmakers": [{"markets": [{"key": "team_totals", "outcomes": [
            {"name": "Over", "description": "Arsenal", "price": 1.4, "point": 2.5},
            {"name": "Under", "description": "Arsenal", "price": 3.0, "point": 2.5},
        ]}]}],
    }

    result = _normalize_secondary(payload, home_team="Arsenal", away_team="Everton")

    assert result.home_goals is None


def test_normalize_secondary_home_goals_none_when_home_team_not_supplied() -> None:
    """team_totals parsing is skipped entirely (not a crash) when the
    caller doesn't pass home_team/away_team -- existing totals/btts-only
    callers pass neither."""
    payload = {"bookmakers": [{"markets": [{"key": "team_totals", "outcomes": [
        {"name": "Over", "description": "Arsenal", "price": 1.4, "point": 1.5},
    ]}]}]}

    result = _normalize_secondary(payload)

    assert result.home_goals is None
    assert result.away_goals is None


def test_get_event_odds_regions_override_falls_back_to_client_default() -> None:
    session = _mock_event_odds_session({"bookmakers": []})
    counter = CreditCounter()
    client = OddsAPIClient(api_key="my-key", credit_counter=counter, session=session, regions=("uk",))

    client.get_event_odds(sport_key="soccer_epl", event_id="evt123")

    assert session.get.call_args.kwargs["params"]["regions"] == "uk"


def test_get_event_odds_regions_override_used_when_supplied() -> None:
    """W199: team_totals needs a wider region set than totals/btts -- an
    explicit override lets one client instance serve both without paying
    the wider cost on every call."""
    session = _mock_event_odds_session({"bookmakers": []})
    counter = CreditCounter()
    client = OddsAPIClient(api_key="my-key", credit_counter=counter, session=session, regions=("uk",))

    client.get_event_odds(
        sport_key="soccer_epl", event_id="evt123", markets=("team_totals",), regions=("uk", "us", "us2"),
    )

    assert session.get.call_args.kwargs["params"]["regions"] == "uk,us,us2"
    assert session.get.call_args.kwargs["params"]["markets"] == "team_totals"


def test_get_event_odds_costs_by_the_override_regions_not_the_client_default() -> None:
    session = _mock_event_odds_session({"bookmakers": []})
    counter = CreditCounter(now_fn=lambda: datetime(2026, 7, 11, tzinfo=timezone.utc))
    client = OddsAPIClient(api_key="fake-key", credit_counter=counter, session=session, regions=("uk",))

    client.get_event_odds(sport_key="soccer_epl", event_id="evt123", markets=("team_totals",), regions=("uk", "us", "us2"))

    assert counter.credits_used == 3  # 1 market x 3 regions, not 1 x 1


def test_get_event_odds_passes_home_and_away_team_through_to_normalize_secondary() -> None:
    payload = {
        "home_team": "Arsenal", "away_team": "Everton",
        "bookmakers": [{"markets": [{"key": "team_totals", "outcomes": [
            {"name": "Over", "description": "Arsenal", "price": 1.83, "point": 1.5},
            {"name": "Under", "description": "Arsenal", "price": 1.95, "point": 1.5},
        ]}]}],
    }
    session = _mock_event_odds_session(payload)
    counter = CreditCounter()
    client = OddsAPIClient(api_key="my-key", credit_counter=counter, session=session)

    result = client.get_event_odds(
        sport_key="soccer_epl", event_id="evt123", markets=("team_totals",),
        regions=("uk", "us", "us2"), home_team="Arsenal", away_team="Everton",
    )

    assert result.home_goals == {"over_1.5": 1.83, "under_1.5": 1.95}
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd app/backend && python -m pytest tests/test_odds_api_client.py -v`
Expected: the 6 new tests FAIL — `NormalizedSecondaryOdds` has no `home_goals`/`away_goals` fields, `_normalize_secondary()` doesn't accept `home_team`/`away_team` kwargs, `get_event_odds()` doesn't accept a `regions` override or `home_team`/`away_team`.

- [ ] **Step 3: Implement**

In `app/backend/odds_api_client.py`, update `NormalizedSecondaryOdds` (lines 107-114):

```python
_TEAM_GOALS_LINE = 1.5


@dataclass(frozen=True)
class NormalizedSecondaryOdds:
    """totals/btts/team_totals odds for one fixture, fetched via the
    per-event endpoint (W164/W199) -- The Odds API doesn't serve these on
    the bulk /odds endpoint (confirmed live: 422 "Markets not supported by
    this endpoint"). None per field when no bookmaker priced that market
    for this fixture."""
    total_goals: dict[str, float] | None  # {"over_2.5": .., "under_2.5": ..}
    btts: dict[str, float] | None  # {"yes": .., "no": ..}
    home_goals: dict[str, float] | None = None  # {"over_1.5": .., "under_1.5": ..}
    away_goals: dict[str, float] | None = None
```

Update `_normalize_secondary()` (lines 117-140) to accept `home_team`/`away_team` and parse `team_totals`:

```python
def _normalize_secondary(
    payload: dict, home_team: str | None = None, away_team: str | None = None,
) -> NormalizedSecondaryOdds:
    bookmakers = payload.get("bookmakers") or []

    total_goals = None
    totals_outcomes = _first_priced_outcomes(bookmakers, "totals")
    for outcome in totals_outcomes or []:
        if outcome.get("point") != _TOTAL_GOALS_LINE:
            continue
        name, price = outcome.get("name"), outcome.get("price")
        if name == "Over":
            total_goals = {**(total_goals or {}), "over_2.5": price}
        elif name == "Under":
            total_goals = {**(total_goals or {}), "under_2.5": price}

    btts = None
    btts_outcomes = _first_priced_outcomes(bookmakers, "btts")
    for outcome in btts_outcomes or []:
        name, price = outcome.get("name"), outcome.get("price")
        if name == "Yes":
            btts = {**(btts or {}), "yes": price}
        elif name == "No":
            btts = {**(btts or {}), "no": price}

    home_goals = away_goals = None
    if home_team and away_team:
        team_totals_outcomes = _first_priced_outcomes(bookmakers, "team_totals")
        for outcome in team_totals_outcomes or []:
            if outcome.get("point") != _TEAM_GOALS_LINE:
                continue
            name, price, team = outcome.get("name"), outcome.get("price"), outcome.get("description")
            key = "over_1.5" if name == "Over" else "under_1.5" if name == "Under" else None
            if key is None:
                continue
            if team == home_team:
                home_goals = {**(home_goals or {}), key: price}
            elif team == away_team:
                away_goals = {**(away_goals or {}), key: price}

    return NormalizedSecondaryOdds(total_goals=total_goals, btts=btts, home_goals=home_goals, away_goals=away_goals)
```

Update `get_event_odds()` (lines 270-296) to accept a `regions` override and thread `home_team`/`away_team` through:

```python
    def get_event_odds(
        self, sport_key: str, event_id: str, markets: tuple[str, ...] = ("totals", "btts"),
        regions: tuple[str, ...] | None = None, home_team: str | None = None, away_team: str | None = None,
    ) -> NormalizedSecondaryOdds | None:
        effective_regions = regions if regions is not None else self._regions
        cost = len(markets) * len(effective_regions)

        if self._credit_counter.would_exceed(cost, self._credit_limit, self._safety_margin):
            LOGGER.warning(
                "OddsAPIClient.get_event_odds: skipping event_id=%s, would cross safety margin "
                "(used=%d cost=%d limit=%d safety_margin=%d).",
                event_id, self._credit_counter.credits_used, cost, self._credit_limit, self._safety_margin,
            )
            return None

        response = self._session.get(
            f"{BASE_URL}/sports/{sport_key}/events/{event_id}/odds",
            params={
                "apiKey": self._api_key,
                "regions": ",".join(effective_regions),
                "markets": ",".join(markets),
                "oddsFormat": "decimal",
            },
            timeout=10,
        )
        response.raise_for_status()
        self._credit_counter.record_usage(cost)

        return _normalize_secondary(response.json(), home_team=home_team, away_team=away_team)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd app/backend && python -m pytest tests/test_odds_api_client.py -v`
Expected: all PASS (32 tests: 26 pre-existing + 6 new).

- [ ] **Step 5: Commit**

```bash
git add app/backend/odds_api_client.py app/backend/tests/test_odds_api_client.py
git commit -m "feat(app): parse team_totals (home_goals/away_goals) from The Odds API, with a region override (W199)"
```

---

### Task 7: Wire live `team_totals` fetch into `eod_batch.py` (design §4)

**Files:**
- Modify: `app/backend/eod_batch.py:157-211`
- Test: `app/backend/tests/test_eod_batch.py`

- [ ] **Step 1: Update the 3 existing tests whose `get_event_odds` call-count assertions this change affects**

In `app/backend/tests/test_eod_batch.py`, `test_secondary_odds_reused_from_cache_not_refetched_when_h2h_unchanged` (around line 316): extend the pre-seeded cache row's `odds` dict to also carry `home_goals`/`away_goals` keys (so the new team_totals cache-reuse branch also finds "already checked", keeping the `assert_not_called()` true):

```python
    cache.record_generation(
        match_id="m1", date=_future_date(1), agent_config_hash=agent_config_hash,
        odds={
            "home": 1.8, "draw": 3.6, "away": 4.5,
            "total_goals": {"over_2.5": 1.9, "under_2.5": 1.95}, "btts": {"yes": 1.7, "no": 2.1},
            "home_goals": {"over_1.5": 1.6, "under_1.5": 2.2}, "away_goals": {"over_1.5": 2.5, "under_1.5": 1.5},
        },
        recommendation=_RECOMMENDATION, triggered_by="scheduled",
    )
```

In `test_secondary_odds_refetched_when_h2h_odds_moved` (around line 480) and `test_secondary_odds_backfilled_once_for_a_cache_row_that_predates_the_feature` (around line 516), replace:

```python
    odds_client.get_event_odds.assert_called_once_with(sport_key="soccer_epl", event_id="evt1")
```

with:

```python
    assert odds_client.get_event_odds.call_count == 2
    odds_client.get_event_odds.assert_any_call(sport_key="soccer_epl", event_id="evt1")
```

- [ ] **Step 2: Write the new failing test**

Add to `app/backend/tests/test_eod_batch.py`:

```python
def test_team_totals_odds_fetched_with_wider_regions_and_threaded_into_match_info(tmp_path: Path) -> None:
    """W199: team_totals needs a wider region set than totals/btts (The
    Odds API confirmed live: UK bookmakers essentially never price it) --
    a second get_event_odds() call, distinct from the totals/btts one,
    with regions=(uk,us,us2) and the matched event's own team-name
    spelling (not the fixture's football-data.org spelling)."""
    fixtures_client = MagicMock()
    fixtures_client.get_fixtures.return_value = [_fixture("m1", "Arsenal", "Everton")]
    odds_client = MagicMock()
    odds_client.get_odds.return_value = [
        NormalizedOdds(
            home_team="Arsenal", away_team="Everton", commence_time="2026-08-22T15:00:00Z",
            home_odds=1.8, draw_odds=3.6, away_odds=4.5, event_id="evt1",
        ),
    ]
    odds_client.get_event_odds.side_effect = [
        NormalizedSecondaryOdds(total_goals={"over_2.5": 1.9, "under_2.5": 1.95}, btts={"yes": 1.7, "no": 2.1}),
        NormalizedSecondaryOdds(
            total_goals=None, btts=None,
            home_goals={"over_1.5": 1.6, "under_1.5": 2.2}, away_goals={"over_1.5": 2.5, "under_1.5": 1.5},
        ),
    ]
    cache = RecommendationCache(db_path=tmp_path / "cache.db")
    config = AgentConfig.default()
    captured_match_info = {}

    def _capture(match_info, config):
        captured_match_info.update(match_info)
        return _RECOMMENDATION

    with patch("app.backend.recommendations.run_agent", side_effect=_capture):
        asyncio.run(
            run_eod_batch(
                fixtures_client=fixtures_client, odds_client=odds_client, cache=cache, config=config,
                schedule_t30=lambda f: None, date_str=_future_date(1),
            )
        )

    odds_client.get_event_odds.assert_any_call(
        sport_key="soccer_epl", event_id="evt1", markets=("team_totals",),
        regions=("uk", "us", "us2"), home_team="Arsenal", away_team="Everton",
    )
    assert captured_match_info["home_goals_odds"] == {"over_1.5": 1.6, "under_1.5": 2.2}
    assert captured_match_info["away_goals_odds"] == {"over_1.5": 2.5, "under_1.5": 1.5}

    agent_config_hash = compute_agent_config_hash(config)
    cached = cache.get_latest("m1", _future_date(1), agent_config_hash)
    assert cached.odds["home_goals"] == {"over_1.5": 1.6, "under_1.5": 2.2}
    assert cached.odds["away_goals"] == {"over_1.5": 2.5, "under_1.5": 1.5}
```

- [ ] **Step 3: Run tests to verify they fail**

Run: `cd app/backend && python -m pytest tests/test_eod_batch.py -v -k "secondary_odds or team_totals"`
Expected: the new test FAILS (no `home_goals_odds`/`away_goals_odds` in `match_info` yet); the 2 updated tests FAIL on the call-count assertion (still only called once); the cache-reuse test still PASSES (no source change yet, but its own fixture update in Step 1 is a no-op until Step 4's source change exists — confirm it doesn't regress).

- [ ] **Step 4: Implement**

In `app/backend/eod_batch.py`, add a module-level constant near the top (after `_TEAM_MAPPING_PATH`, line 57):

```python
# W199: The Odds API confirmed live (2026-09) that UK bookmakers essentially
# never price team_totals, unlike totals/btts (which get fine uk-only
# coverage) -- us/us2-licensed books (DraftKings/FanDuel/BetMGM-type) do.
# Scoped to just this one fetch so totals/btts/h2h keep their existing,
# cheaper uk-only cost.
_TEAM_TOTALS_REGIONS = ("uk", "us", "us2")
```

Update `add_secondary_odds()` (lines 157-211) — add the team_totals fetch/cache-reuse block, mirroring the existing totals/btts one:

```python
def add_secondary_odds(
    match_info: dict,
    odds: dict,
    odds_client: OddsAPIClient | None,
    cache: RecommendationCache,
    fixture: NormalizedMatch,
    fixture_date: str,
    agent_config_hash: str,
    sport_key: str,
    odds_by_teams: dict[tuple[str, str], NormalizedOdds],
) -> None:
    """W164/W164a/W199, shared by run_eod_batch (below) and t30_refresh.py's
    refresh_match_at_t30. Fetches totals/btts AND (W199) team_totals
    (home_goals/away_goals), folding both into `match_info` and `odds`,
    each with its own independent cache-reuse tracking -- a cache row that
    predates W199 has total_goals/btts keys but no home_goals/away_goals
    key, so team_totals gets its own one-time backfill fetch even when
    totals/btts are reused unchanged from cache."""
    cached_entry = cache.get_latest(fixture.match_id, fixture_date, agent_config_hash)
    h2h_unchanged = cached_entry is not None and {
        k: cached_entry.odds.get(k) for k in ("home", "draw", "away")
    } == odds
    already_checked_secondary = cached_entry is not None and (
        "total_goals" in cached_entry.odds or "btts" in cached_entry.odds
    )
    already_checked_team_totals = cached_entry is not None and (
        "home_goals" in cached_entry.odds or "away_goals" in cached_entry.odds
    )
    odds_event = matched_odds_event(fixture, odds_by_teams)
    get_event_odds = getattr(odds_client, "get_event_odds", None)

    if h2h_unchanged and already_checked_secondary:
        if cached_entry.odds.get("total_goals"):
            match_info["total_goals_odds"] = cached_entry.odds["total_goals"]
        if cached_entry.odds.get("btts"):
            match_info["btts_odds"] = cached_entry.odds["btts"]
        odds["total_goals"] = cached_entry.odds.get("total_goals")
        odds["btts"] = cached_entry.odds.get("btts")
    else:
        if get_event_odds is not None and odds_event is not None and odds_event.event_id:
            secondary = get_event_odds(sport_key=sport_key, event_id=odds_event.event_id)
            if secondary is not None:
                if secondary.total_goals:
                    match_info["total_goals_odds"] = secondary.total_goals
                if secondary.btts:
                    match_info["btts_odds"] = secondary.btts
                odds["total_goals"] = secondary.total_goals
                odds["btts"] = secondary.btts

    if h2h_unchanged and already_checked_team_totals:
        if cached_entry.odds.get("home_goals"):
            match_info["home_goals_odds"] = cached_entry.odds["home_goals"]
        if cached_entry.odds.get("away_goals"):
            match_info["away_goals_odds"] = cached_entry.odds["away_goals"]
        odds["home_goals"] = cached_entry.odds.get("home_goals")
        odds["away_goals"] = cached_entry.odds.get("away_goals")
    else:
        if get_event_odds is not None and odds_event is not None and odds_event.event_id:
            team_totals = get_event_odds(
                sport_key=sport_key, event_id=odds_event.event_id, markets=("team_totals",),
                regions=_TEAM_TOTALS_REGIONS, home_team=odds_event.home_team, away_team=odds_event.away_team,
            )
            if team_totals is not None:
                if team_totals.home_goals:
                    match_info["home_goals_odds"] = team_totals.home_goals
                if team_totals.away_goals:
                    match_info["away_goals_odds"] = team_totals.away_goals
                odds["home_goals"] = team_totals.home_goals
                odds["away_goals"] = team_totals.away_goals
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `cd app/backend && python -m pytest tests/test_eod_batch.py -v`
Expected: all PASS.

- [ ] **Step 6: Commit**

```bash
git add app/backend/eod_batch.py app/backend/tests/test_eod_batch.py
git commit -m "feat(app): fetch team_totals with wider regions in the EOD batch, cached independently (W199)"
```

---

### Task 8: Update `t30_refresh.py` tests (shares `add_secondary_odds`, no source change)

**Files:**
- Test: `app/backend/tests/test_t30_refresh.py`

`refresh_match_at_t30` calls the same `add_secondary_odds()` Task 7 already changed — no source edit needed here, only test updates for the same call-count/cache-shape reasons as Task 7.

- [ ] **Step 1: Update the affected tests**

In `test_secondary_odds_fetched_and_threaded_into_match_info_and_odds_dedup_key` (around line 338), replace:

```python
    odds_client.get_event_odds.assert_called_once_with(sport_key="soccer_epl", event_id="evt1")
```

with:

```python
    assert odds_client.get_event_odds.call_count == 2
    odds_client.get_event_odds.assert_any_call(sport_key="soccer_epl", event_id="evt1")
```

In `test_secondary_odds_reused_from_cache_not_refetched_when_h2h_unchanged` (around line 357), extend the pre-seeded cache row's `odds` dict, same fix as Task 7 Step 1:

```python
    cache.record_generation(
        match_id="m1", date=_future_date(1), agent_config_hash=agent_config_hash,
        odds={
            "home": 1.8, "draw": 3.6, "away": 4.5,
            "total_goals": {"over_2.5": 1.9, "under_2.5": 1.95}, "btts": {"yes": 1.7, "no": 2.1},
            "home_goals": {"over_1.5": 1.6, "under_1.5": 2.2}, "away_goals": {"over_1.5": 2.5, "under_1.5": 1.5},
        },
        recommendation=_RECOMMENDATION, triggered_by="scheduled",
    )
```

- [ ] **Step 2: Run tests to verify they now pass**

Run: `cd app/backend && python -m pytest tests/test_t30_refresh.py -v`
Expected: all PASS (these two were failing before this step, since Task 7 already changed the shared `add_secondary_odds` behavior these tests exercise).

- [ ] **Step 3: Commit**

```bash
git add app/backend/tests/test_t30_refresh.py
git commit -m "test(app): update t30_refresh tests for add_secondary_odds' new team_totals call (W199)"
```

---

### Task 9: Surface `home_goals_odds`/`away_goals_odds` in the LLM prompt (design §4)

**Files:**
- Modify: `src/agent/graph.py:572-579`
- Test: `tests/test_agent_graph.py`

- [ ] **Step 1: Write the failing test**

Add to `tests/test_agent_graph.py`, right after `test_run_agent_includes_corners_odds_in_prompt_when_present` (around line 521):

```python
def test_run_agent_includes_home_and_away_goals_odds_in_prompt_when_present():
    """W199: home_goals_odds/away_goals_odds (from eod_batch.py's team_totals
    fetch, or backtest.py's OddsPapi lookup) must reach the LLM's prompt the
    same way total_goals_odds/btts_odds/corners_odds already do."""
    from unittest.mock import MagicMock, patch
    from langchain_core.messages import AIMessage, HumanMessage
    from src.agent.graph import run_agent
    from src.agent import tools as agent_tools

    agent_tools._snapshot_store.set_mode("live")
    llm_json = json.dumps({
        "match": {"home": "Man City", "away": "Arsenal", "date": "2026-06-21", "league": "E0"},
        "overall": "no_bet", "candidates": [], "recommendation_pick": None, "explanation": "Balanced match.",
        "confidence": "medium", "limitations": [], "prediction_basis": "team_history_and_market",
    })
    fake_forecast_result = {"result_3way": {"probabilities": {"home": 0.4}}, "data_quality": {"prediction_basis": "team_history_and_market"}}

    with patch("src.agent.graph._build_llm") as mock_build_llm, \
         patch("src.agent.graph._load_system_prompt", return_value="stub prompt"), \
         patch("src.agent.tools._dated_web_search", return_value="No results found."), \
         patch("src.forecast.forecast_service.ForecastService") as MockSvc, \
         patch("src.agent.lessons.load_approved_lessons", return_value=[]), \
         patch("src.utils.db_manager.DuckDBManager") as MockDB:
        MockDB.return_value.connection.return_value.__enter__.return_value = MagicMock()
        instance = MagicMock()
        MockSvc.return_value = instance
        instance.forecast_upcoming.return_value = fake_forecast_result

        mock_llm = MagicMock()
        mock_llm.bind_tools.return_value.invoke.return_value = AIMessage(content=llm_json)
        mock_build_llm.return_value = mock_llm

        cfg = _make_config()
        run_agent(
            match_info={
                "home_team": "Man City", "away_team": "Arsenal", "date": "2026-06-21", "league": "E0",
                "odds": {"home": 2.0, "draw": 3.4, "away": 3.6},
                "home_goals_odds": {"over_1.5": 1.6, "under_1.5": 2.2},
                "away_goals_odds": {"over_1.5": 2.5, "under_1.5": 1.5},
            },
            config=cfg,
            tools=[],
        )

    messages = mock_llm.bind_tools.return_value.invoke.call_args[0][0]
    prompt_message = next(m for m in messages if isinstance(m, HumanMessage) and "Analyse the upcoming match" in m.content)
    assert "over_1.5=1.6" in prompt_message.content
    assert "under_1.5=2.2" in prompt_message.content
    assert "over_1.5=2.5" in prompt_message.content
    assert "under_1.5=1.5" in prompt_message.content
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_agent_graph.py::test_run_agent_includes_home_and_away_goals_odds_in_prompt_when_present -v`
Expected: FAIL — `AssertionError`, the prompt text never mentions these odds.

- [ ] **Step 3: Implement**

In `src/agent/graph.py`, add a new block right after the existing `corners_odds` block (lines 572-579):

```python
    corners_odds = match_info.get("corners_odds")
    if corners_odds:
        over = corners_odds.get("over_9.5")
        under = corners_odds.get("under_9.5")
        prompt += f" Bookmaker odds for total corners (over/under 9.5): over_9.5={over}, under_9.5={under}."
    # W199: home_goals_odds/away_goals_odds threaded from eod_batch.py's
    # team_totals fetch (live) or backtest.py's OddsPapi lookup (backtest/
    # agent-train) -- same "populating match_info alone is a no-op if the
    # model never sees it" precedent as total_goals_odds/btts_odds/corners_odds above.
    home_goals_odds = match_info.get("home_goals_odds")
    if home_goals_odds:
        over = home_goals_odds.get("over_1.5")
        under = home_goals_odds.get("under_1.5")
        prompt += (
            f" Bookmaker odds for {match_info['home_team']}'s own goals (over/under 1.5): "
            f"over_1.5={over}, under_1.5={under}."
        )
    away_goals_odds = match_info.get("away_goals_odds")
    if away_goals_odds:
        over = away_goals_odds.get("over_1.5")
        under = away_goals_odds.get("under_1.5")
        prompt += (
            f" Bookmaker odds for {match_info['away_team']}'s own goals (over/under 1.5): "
            f"over_1.5={over}, under_1.5={under}."
        )
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_agent_graph.py -v -k "goals_odds or corners_odds or btts_odds"`
Expected: all PASS, including the pre-existing `total_goals_odds`/`btts_odds`/`corners_odds` tests (unaffected).

- [ ] **Step 5: Commit**

```bash
git add src/agent/graph.py tests/test_agent_graph.py
git commit -m "feat(agent): surface home_goals_odds/away_goals_odds in the LLM prompt (W199)"
```

---

### Task 10: Pin OddsPapi's team-goals market IDs (design §7, step 1)

**Real, one-off, network-calling investigation script** — same nature as Task 5, but against OddsPapi's free metadata endpoint (not counted against the 250/month historical-odds quota).

**Files:**
- Create: `scripts/find_oddspapi_team_goals_market_ids.py`

- [ ] **Step 1: Write the script**

```python
"""One-off investigation script (W199): pin which of OddsPapi's team-goals
market IDs (10224-10236 and 10240-10250, already confirmed present in
data/oddspapi_snapshots/ -- 98.3% of the 1,005 already-pulled matches carry
at least one) map to home vs. away and to the 1.5 line specifically. Mirrors
how data/oddspapi_snapshots/corners_line_map.json was built for corners'
own 9.5 line.

Usage: python scripts/find_oddspapi_team_goals_market_ids.py
Requires ODDSPAPI_API_KEY in .env. /v4/markets is a metadata/reference
endpoint -- confirmed in the W199 investigation notes (documents/
app_user_stories.md) not to count against the 250/month historical-odds
quota."""
from __future__ import annotations

import json
import os
import sys

import requests
from dotenv import load_dotenv

load_dotenv()

API_KEY = os.environ.get("ODDSPAPI_API_KEY")
if not API_KEY:
    sys.exit("No API key: set ODDSPAPI_API_KEY in .env")

BASE_URL = "https://api.oddspapi.io"
_CANDIDATE_IDS = list(range(10224, 10237, 2)) + list(range(10240, 10251, 2))


def main() -> None:
    resp = requests.get(f"{BASE_URL}/v4/markets", params={"apiKey": API_KEY, "sportId": 10}, timeout=30)
    resp.raise_for_status()
    markets = resp.json()

    by_id = {str(m.get("id")): m for m in markets if isinstance(m, dict)}

    print(f"Fetched {len(markets)} total market definitions.\n")
    print("=== Candidate team-goals market IDs (from already-pulled snapshot data) ===")
    for market_id in _CANDIDATE_IDS:
        entry = by_id.get(str(market_id))
        if entry is None:
            print(f"{market_id}: NOT FOUND in /v4/markets response")
            continue
        print(f"{market_id}: {json.dumps(entry)}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Run it and build the line map**

Run: `python scripts/find_oddspapi_team_goals_market_ids.py`

Expected: prints a name/description for each of the 13 candidate market IDs — something like "Team 1 Over/Under 0.5", "Team 1 Over/Under 1.5", ..., "Team 2 Over/Under 0.5", etc. (naming convention confirmed by the earlier W199 investigation: "Over Under Team 1"/"Over Under Team 2"). From the printed output:
1. Identify which "Team N" corresponds to the **home** team and which to **away** (OddsPapi's fixture data typically labels `participant1`/`participant2` as home/away in fixture order — cross-check against `data/oddspapi_snapshots/manifest.json` or a fixture lookup if the market names alone don't say).
2. Identify the exact market ID for each side's **1.5 line** specifically.

Write the result to a new file, mirroring `corners_line_map.json`'s exact shape:

```bash
python3 -c "
import json
# Replace these two with what Step 2 actually printed:
result = {'home_goals_1_5_market_id': '<ID>', 'away_goals_1_5_market_id': '<ID>'}
with open('data/oddspapi_snapshots/team_goals_line_map.json', 'w') as f:
    json.dump(result, f, indent=2)
print(json.dumps(result, indent=2))
"
```

This step has no fixed pass/fail assertion — it is complete once `data/oddspapi_snapshots/team_goals_line_map.json` contains the two real, confirmed market IDs.

- [ ] **Step 3: Commit the script and the resulting line map**

```bash
git add scripts/find_oddspapi_team_goals_market_ids.py data/oddspapi_snapshots/team_goals_line_map.json
git commit -m "chore: pin OddsPapi's home/away 1.5-goal-line market IDs (W199)"
```

---

### Task 11: Extract `home_goals`/`away_goals` odds from already-pulled OddsPapi data (design §7, step 2)

**Depends on Task 10's `team_goals_line_map.json`.** Substitute the two real market IDs it contains for `<HOME_GOALS_MARKET_ID>`/`<AWAY_GOALS_MARKET_ID>` below.

**Files:**
- Modify: `scripts/extract_oddspapi_odds_lookup.py`

- [ ] **Step 1: Implement**

In `scripts/extract_oddspapi_odds_lookup.py`, add the new market-ID constants (after `CORNERS_OUTCOME_UNDER`, line 31) — replace the two placeholders with the real IDs from `data/oddspapi_snapshots/team_goals_line_map.json`:

```python
HOME_GOALS_LINE = 1.5
HOME_GOALS_MARKET_ID = "<HOME_GOALS_MARKET_ID>"  # from data/oddspapi_snapshots/team_goals_line_map.json
HOME_GOALS_OUTCOME_OVER = "<HOME_GOALS_MARKET_ID>"
HOME_GOALS_OUTCOME_UNDER = str(int("<HOME_GOALS_MARKET_ID>") + 1)

AWAY_GOALS_LINE = 1.5
AWAY_GOALS_MARKET_ID = "<AWAY_GOALS_MARKET_ID>"
AWAY_GOALS_OUTCOME_OVER = "<AWAY_GOALS_MARKET_ID>"
AWAY_GOALS_OUTCOME_UNDER = str(int("<AWAY_GOALS_MARKET_ID>") + 1)
```

Update the `main()` loop's per-fixture extraction (lines 56-73):

```python
        data = json.loads(path.read_text())
        outcomes_by_market = data.get("bookmakers", {}).get("pinnacle", {}).get("markets", {})

        btts_outcomes = outcomes_by_market.get(BTTS_MARKET_ID, {}).get("outcomes", {})
        btts_yes = _last_price(btts_outcomes, BTTS_OUTCOME_YES)
        btts_no = _last_price(btts_outcomes, BTTS_OUTCOME_NO)

        corners_outcomes = outcomes_by_market.get(CORNERS_MARKET_ID, {}).get("outcomes", {})
        corners_over = _last_price(corners_outcomes, CORNERS_OUTCOME_OVER)
        corners_under = _last_price(corners_outcomes, CORNERS_OUTCOME_UNDER)

        home_goals_outcomes = outcomes_by_market.get(HOME_GOALS_MARKET_ID, {}).get("outcomes", {})
        home_goals_over = _last_price(home_goals_outcomes, HOME_GOALS_OUTCOME_OVER)
        home_goals_under = _last_price(home_goals_outcomes, HOME_GOALS_OUTCOME_UNDER)

        away_goals_outcomes = outcomes_by_market.get(AWAY_GOALS_MARKET_ID, {}).get("outcomes", {})
        away_goals_over = _last_price(away_goals_outcomes, AWAY_GOALS_OUTCOME_OVER)
        away_goals_under = _last_price(away_goals_outcomes, AWAY_GOALS_OUTCOME_UNDER)

        entry: dict = {}
        if btts_yes is not None and btts_no is not None:
            entry["btts_odds"] = {"yes": btts_yes, "no": btts_no}
        if corners_over is not None and corners_under is not None:
            entry[f"corners_{CORNERS_LINE}_odds"] = {"over": corners_over, "under": corners_under}
        if home_goals_over is not None and home_goals_under is not None:
            entry[f"home_goals_{HOME_GOALS_LINE}_odds"] = {"over": home_goals_over, "under": home_goals_under}
        if away_goals_over is not None and away_goals_under is not None:
            entry[f"away_goals_{AWAY_GOALS_LINE}_odds"] = {"over": away_goals_over, "under": away_goals_under}
        if entry:
            lookup[match_id] = entry
```

Update the summary print block (lines 75-82):

```python
    OUT_PATH.write_text(json.dumps(lookup, indent=2))
    with_btts = sum(1 for v in lookup.values() if "btts_odds" in v)
    with_corners = sum(1 for v in lookup.values() if f"corners_{CORNERS_LINE}_odds" in v)
    with_home_goals = sum(1 for v in lookup.values() if f"home_goals_{HOME_GOALS_LINE}_odds" in v)
    with_away_goals = sum(1 for v in lookup.values() if f"away_goals_{AWAY_GOALS_LINE}_odds" in v)
    print(f"Matches with any odds: {len(lookup)}")
    print(f"  with btts_odds: {with_btts}")
    print(f"  with corners_{CORNERS_LINE}_odds: {with_corners}")
    print(f"  with home_goals_{HOME_GOALS_LINE}_odds: {with_home_goals}")
    print(f"  with away_goals_{AWAY_GOALS_LINE}_odds: {with_away_goals}")
    print(f"Skipped (no saved file): {skipped_no_file}")
    print(f"Wrote {OUT_PATH}")
```

- [ ] **Step 2: Re-run against the already-downloaded snapshot files (no new API calls)**

Run: `python scripts/extract_oddspapi_odds_lookup.py`
Expected: prints `Matches with any odds: <N>` including non-zero `with home_goals_1.5_odds:`/`with away_goals_1.5_odds:` counts (expect roughly 900+ of 1,005, consistent with the 98.3% coverage already confirmed for this market ID range) and rewrites `data/oddspapi_btts_corners_odds.json`.

- [ ] **Step 3: Spot-check the output**

Run: `python3 -c "
import json
data = json.load(open('data/oddspapi_btts_corners_odds.json'))
sample = next(v for v in data.values() if 'home_goals_1.5_odds' in v)
print(sample)
"`
Expected: prints a dict containing `home_goals_1.5_odds: {"over": <float>, "under": <float>}`.

- [ ] **Step 4: Commit**

```bash
git add scripts/extract_oddspapi_odds_lookup.py data/oddspapi_btts_corners_odds.json
git commit -m "feat: extract home_goals/away_goals odds from already-pulled OddsPapi data (W199)"
```

---

### Task 12: Thread OddsPapi `home_goals_odds`/`away_goals_odds` into backtest/agent-train (design §7, step 3-4)

**Files:**
- Modify: `src/agent/backtest.py:183-194`
- Test: `tests/test_backtest.py`

- [ ] **Step 1: Write the failing tests**

Add to `tests/test_backtest.py`, near `test_build_match_info_includes_only_whichever_market_the_lookup_actually_has` (around line 171):

```python
def test_build_match_info_includes_home_and_away_goals_odds_when_present():
    with patch(
        "src.agent.backtest._load_oddspapi_odds_lookup",
        return_value={"m1": {
            "home_goals_1.5_odds": {"over": 1.6, "under": 2.2},
            "away_goals_1.5_odds": {"over": 2.5, "under": 1.5},
        }},
    ):
        info = _build_match_info(_row())
    assert info["home_goals_odds"] == {"over_1.5": 1.6, "under_1.5": 2.2}
    assert info["away_goals_odds"] == {"over_1.5": 2.5, "under_1.5": 1.5}


def test_build_match_info_omits_home_and_away_goals_odds_when_match_id_not_in_lookup():
    with patch("src.agent.backtest._load_oddspapi_odds_lookup", return_value={}):
        info = _build_match_info(_row())
    assert "home_goals_odds" not in info
    assert "away_goals_odds" not in info


def test_build_match_info_includes_home_goals_odds_independent_of_away_goals_odds():
    """Same independent-per-market coverage precedent as btts vs. corners
    (A99.5) -- a match missing one team's line must not lose the other."""
    with patch(
        "src.agent.backtest._load_oddspapi_odds_lookup",
        return_value={"m1": {"home_goals_1.5_odds": {"over": 1.6, "under": 2.2}}},
    ):
        info = _build_match_info(_row())
    assert info["home_goals_odds"] == {"over_1.5": 1.6, "under_1.5": 2.2}
    assert "away_goals_odds" not in info
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_backtest.py -v -k home_goals_odds`
Expected: all 3 FAIL — `_build_match_info()` doesn't set these keys yet.

- [ ] **Step 3: Implement**

In `src/agent/backtest.py`, update `_build_match_info()` (lines 183-194):

```python
    # A100: OddsPapi lookup, keyed by our own match_id -- covers a subset of
    # matches (2026-01-01 onward only, confirmed vendor cutoff) and each
    # market independently (btts ~95% real-tick coverage, corners ~99.9% at
    # the 9.5 line within that window; home/away goals ~98.3% at the 1.5
    # line, W199), so each is threaded in only when actually present rather
    # than assumed to travel together.
    oddspapi_odds = _load_oddspapi_odds_lookup().get(row["match_id"], {})
    if "btts_odds" in oddspapi_odds:
        match_info["btts_odds"] = oddspapi_odds["btts_odds"]
    if "corners_9.5_odds" in oddspapi_odds:
        corners = oddspapi_odds["corners_9.5_odds"]
        match_info["corners_odds"] = {"over_9.5": corners["over"], "under_9.5": corners["under"]}
    if "home_goals_1.5_odds" in oddspapi_odds:
        home_goals = oddspapi_odds["home_goals_1.5_odds"]
        match_info["home_goals_odds"] = {"over_1.5": home_goals["over"], "under_1.5": home_goals["under"]}
    if "away_goals_1.5_odds" in oddspapi_odds:
        away_goals = oddspapi_odds["away_goals_1.5_odds"]
        match_info["away_goals_odds"] = {"over_1.5": away_goals["over"], "under_1.5": away_goals["under"]}
    return match_info
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_backtest.py -v`
Expected: all PASS, including every pre-existing test in this file (unaffected — `agent-train` shares this same function per the module's own docstring, so no separate agent-train test/code path is needed).

- [ ] **Step 5: Run the full backend + agent test suites (final regression check)**

Run: `python -m pytest tests/ app/backend/tests/ -v`
Expected: all PASS.

Run: `cd app/frontend && npx vitest run`
Expected: all PASS.

- [ ] **Step 6: Commit**

```bash
git add src/agent/backtest.py tests/test_backtest.py
git commit -m "feat(agent): thread home_goals_odds/away_goals_odds from OddsPapi into backtest/agent-train (W199)"
```

---

## Plan self-review notes

- **Spec coverage**: §1 (scope) → Tasks 1-12 collectively. §2 (live verification) → Task 5. §3 (odds fetching) → Task 6. §4 (wiring into live recs) → Tasks 7-9. §5 (schema/guardrails) → Tasks 1-2. §6 (grading) → Tasks 3-4. §7 (backtest/train) → Tasks 10-12. §8 (testing) → a dedicated test step in every task. §9 (out of scope) → deliberately untouched (no region change to totals/btts/h2h; no configurable line; no home_corners/away_corners odds source work).
- **Sequencing**: Tasks 1, 3, 4 (schema/grading) have no dependency on live/backtest wiring and can run first or in parallel. Task 6 depends on Task 5's real-world capture. Tasks 7-9 depend on Task 6. Task 11 depends on Task 10's real-world capture. Task 12 depends on Task 11.
- **Type consistency check**: `home_goals_odds`/`away_goals_odds` (match_info keys), `home_goals`/`away_goals` (odds-dict/cache/NormalizedSecondaryOdds field names), `over_1.5`/`under_1.5` (selection strings), `homeGoalsSide`/`awayGoalsSide` (TS field names) are used consistently across every task above.
