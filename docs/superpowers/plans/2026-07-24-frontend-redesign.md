# Frontend Redesign: Sidebar Shell & Sketch-Inspired Dashboard Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Rebuild `app/frontend`'s chrome around a left-sidebar + top-search-bar shell inspired by the user's sketch, adding a real-data Dashboard analytics rail (Edge Distribution donut, Top Edges) and honest league-grouped sections — while dropping every sketch element with no real functionality behind it.

**Architecture:** One new `AppShell` component replaces the existing `DraftNav`+`<main>` wrapper each page renders internally; a new `DashboardRail` component (Edge Distribution + Top Edges) renders only on the Dashboard; a new `lib/dashboardMetrics.ts` holds pure, unit-tested derivations (grouping/sorting/counting) the UI layer consumes. One small backend addition — a `competition` field on `NormalizedMatch`/`Fixture` — is required first, since `/api/fixtures` today merges Premier League and Allsvenskan fixtures with no field distinguishing them, which blocks honest league grouping.

**Tech Stack:** Next.js 14 (App Router) + React 18 + TypeScript, Tailwind CSS (existing dark-locked token set in `app/globals.css`), Vitest + React Testing Library (frontend), pytest + FastAPI `TestClient` (backend). No new dependencies.

**Design doc:** `docs/superpowers/specs/2026-07-23-frontend-redesign-design.md` (approved, revised 2026-07-24).

---

## Before you start

Read these once, in this order — every task below assumes this context:
- `app/frontend/components/MatchUI.tsx` — the file most tasks touch. `Match`/`MarketRec`/`Overall`/`Tier` types, `STATUS_META`, `TeamBadge`, `MatchCard`, `DashboardPage`, `MatchExplorerPage`, `MatchAnalysisPage` all live here today.
- `app/frontend/components/BetTracker.tsx` — `BetTrackerPage`.
- `app/frontend/lib/types.ts` — wire types mirroring the backend exactly.
- `app/frontend/app/globals.css` — the locked dark palette. `--status-good #0ca30c`, `--status-warning #fab219`, `--status-serious #ec835a`, `--text-muted #898781` are the four colors this plan reuses for the Edge Distribution donut (no new colors).
- `app/backend/main.py`'s `get_fixtures()` (~line 289) — the endpoint Task 1 changes.

**Two non-obvious constraints, load-bearing for several tasks below:**
1. `BetTracker.fixtureError.test.tsx` asserts `getFixtures` was called exactly 1 or 2 times against `ManualBetForm`'s own search. `AppShell`'s top-bar search must fetch **lazily on the input's first focus**, never on mount — otherwise it silently adds calls to that same shared mock and breaks those exact-count assertions.
2. `BetTracker.fixtureError.test.tsx`/`BetTracker.race.test.tsx` already query `getByPlaceholderText("Search a real fixture by team name…")`. `AppShell`'s search input must use a different placeholder string so both inputs stay independently queryable once a page renders both.

---

### Task 1: Backend — tag each fixture with its source competition

**Files:**
- Modify: `app/backend/football_data_client.py:20-27` (`NormalizedMatch` dataclass)
- Modify: `app/backend/main.py:288-330` (`get_fixtures()`)
- Test: `app/backend/tests/test_fixtures_endpoint.py`

- [ ] **Step 1: Write the failing test**

Add to `app/backend/tests/test_fixtures_endpoint.py`, right after `test_fixtures_endpoint_merges_sweden_fixtures_alongside_epl` (after line 391):

```python
def test_fixtures_endpoint_tags_each_fixture_with_its_source_competition(sweden_client_mock):
    """The frontend has no way to distinguish an EPL fixture from an
    Allsvenskan one unless the endpoint tags it -- both currently come back
    as the same NormalizedMatch shape with no competition field. Tagging
    must happen at the merge point in main.py (not just inside each
    client's own normalize function), since this test -- like the existing
    W57 tests above -- mocks the clients directly and bypasses their
    internal normalize functions entirely."""
    sweden_client_mock.get_fixtures.return_value = [_SWEDISH_FIXTURE]
    with patch("app.backend.main.get_fixtures_client") as mock_get_client:
        mock_get_client.return_value.get_fixtures.return_value = [_REAL_FIXTURE]
        with TestClient(app) as client:
            response = client.get(
                "/api/fixtures", params={"date_from": "2026-08-21", "date_to": "2026-08-28"}
            )

    assert response.status_code == 200
    body = response.json()
    by_team = {m["home_team"]: m["competition"] for m in body}
    assert by_team == {"Chelsea": "E0", "Malmo FF": "SWE"}


def test_fixtures_endpoint_tags_epl_results_e0(sweden_client_mock):
    """Mirrors the fixtures-side test above for the get_results() (past
    date range) path -- a separate code path in get_fixtures() with its own
    merge call, so needs its own coverage."""
    with patch("app.backend.main._current_real_date", return_value=date(2025, 3, 10)):
        with patch("app.backend.main.get_fixtures_client") as mock_get_client:
            mock_get_client.return_value.get_results.return_value = [_REAL_RESULT]
            with TestClient(app) as client:
                response = client.get(
                    "/api/fixtures", params={"date_from": "2025-03-08", "date_to": "2025-03-08"}
                )

    body = response.json()
    assert body[0]["competition"] == "E0"


def test_fixtures_endpoint_tags_sweden_results_swe(sweden_client_mock):
    sweden_client_mock.get_results.return_value = [_SWEDISH_RESULT]
    with patch("app.backend.main._current_real_date", return_value=date(2025, 3, 10)):
        with patch("app.backend.main.get_fixtures_client") as mock_get_client:
            mock_get_client.return_value.get_results.return_value = []
            with TestClient(app) as client:
                response = client.get(
                    "/api/fixtures", params={"date_from": "2025-03-08", "date_to": "2025-03-08"}
                )

    body = response.json()
    assert body[0]["competition"] == "SWE"
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `cd /Users/tianqihuang/Documents/GitHub/FPAI && python -m pytest app/backend/tests/test_fixtures_endpoint.py -k "tags" -v`
Expected: 3 failures — `KeyError: 'competition'` (the field doesn't exist yet in the response body).

- [ ] **Step 3: Add the `competition` field with a safe default**

In `app/backend/football_data_client.py`, change the `NormalizedMatch` dataclass:

```python
@dataclass(frozen=True)
class NormalizedMatch:
    match_id: str
    utc_date: str
    status: str
    home_team: str
    away_team: str
    home_goals: int | None
    away_goals: int | None
    # W64: which competition this fixture belongs to -- "E0" or "SWE" today.
    # Defaults to "E0" so every existing construction site across the
    # codebase (tests included) keeps working unchanged; only
    # get_fixtures()'s merge logic in main.py sets it explicitly per source.
    competition: str = "E0"
```

- [ ] **Step 4: Tag matches at the merge point in `get_fixtures()`**

In `app/backend/main.py`, add the import (near the other stdlib imports, ~line 9):

```python
import dataclasses
```

Then replace the merge block (lines ~308-330):

```python
    def _tag(matches: list[NormalizedMatch], competition: str) -> list[NormalizedMatch]:
        # W64: explicit tagging here (not inside each client's own
        # normalize function) is deliberate -- this is the one place every
        # return path through this endpoint passes through, real or
        # test-mocked, so it's the only place a tag is guaranteed to stick.
        return [dataclasses.replace(m, competition=competition) for m in matches]

    matches: list[NormalizedMatch] = []
    if results_range is not None:
        past_from, past_to = results_range
        matches += _tag(
            await _cached_fixture_call(
                ("results", past_from, past_to), client.get_results, date_from=past_from, date_to=past_to
            ),
            "E0",
        )
        # W57: Sweden (Allsvenskan) merged in alongside EPL, sourced from The
        # Odds API instead of football-data.org (W55 -- football-data.org's
        # free tier has no Allsvenskan coverage at all). Cache-keyed
        # separately ("results_swe" vs "results") so it can never collide
        # with football-data.org's entry for the identical date range.
        matches += _tag(
            await _cached_fixture_call(
                ("results_swe", past_from, past_to), sweden_client.get_results, date_from=past_from, date_to=past_to
            ),
            "SWE",
        )
    if fixtures_range is not None:
        future_from, future_to = fixtures_range
        matches += _tag(
            await _cached_fixture_call(
                ("fixtures", future_from, future_to), client.get_fixtures, date_from=future_from, date_to=future_to
            ),
            "E0",
        )
        matches += _tag(
            await _cached_fixture_call(
                ("fixtures_swe", future_from, future_to), sweden_client.get_fixtures, date_from=future_from, date_to=future_to
            ),
            "SWE",
        )
    return matches
```

- [ ] **Step 5: Run the tests to verify they pass**

Run: `cd /Users/tianqihuang/Documents/GitHub/FPAI && python -m pytest app/backend/tests/test_fixtures_endpoint.py -v`
Expected: all pass, including the 3 new ones.

- [ ] **Step 6: Run the full backend suite to confirm zero regressions**

Run: `cd /Users/tianqihuang/Documents/GitHub/FPAI && python -m pytest tests/ app/backend/tests/ -q`
Expected: same pass count as before this change, plus 3.

- [ ] **Step 7: Commit**

```bash
git add app/backend/football_data_client.py app/backend/main.py app/backend/tests/test_fixtures_endpoint.py
git commit -m "feat(app): tag /api/fixtures matches with their source competition (E0/SWE)

Needed for the frontend redesign's league-grouped Dashboard sections --
the endpoint previously merged EPL and Allsvenskan into one flat list
with no field distinguishing them."
```

---

### Task 2: Frontend types — add `competition` to `Fixture`

**Files:**
- Modify: `app/frontend/lib/types.ts:5-13`

- [ ] **Step 1: Add the field as optional**

In `app/frontend/lib/types.ts`, update `Fixture`:

```typescript
export type Fixture = {
  match_id: string;
  utc_date: string;
  status: string;
  home_team: string;
  away_team: string;
  home_goals: number | null;
  away_goals: number | null;
  // W64: "E0" or "SWE". Optional (not `competition: string`) so every
  // existing hand-built Fixture literal across the test suite -- there are
  // several, none of which set this field -- keeps type-checking without
  // modification; fixtureToMatch() (Task 3) defaults a missing value to
  // "E0", mirroring the backend's own default.
  competition?: string;
};
```

- [ ] **Step 2: Type-check**

Run: `cd /Users/tianqihuang/Documents/GitHub/FPAI/app/frontend && npx tsc --noEmit`
Expected: no new errors.

- [ ] **Step 3: Commit**

```bash
git add app/frontend/lib/types.ts
git commit -m "feat(app): add optional competition field to the Fixture type"
```

---

### Task 3: Frontend — use the real competition instead of a hardcoded "E0"

This closes a real (pre-existing, independent of this redesign) correctness gap: `fixtureToMatch()` today hardcodes `league: "E0"` for every fixture, meaning a Sweden fixture's card already silently requests a recommendation with the wrong league. Task 2's field makes the fix trivial and it's directly required for grouping anyway.

**Files:**
- Modify: `app/frontend/components/MatchUI.tsx:107-136` (`fixtureToMatch`), `:499-645` (`MatchCard`'s "Full analysis" link)
- Test: `app/frontend/components/__tests__/MatchUI.test.tsx`

- [ ] **Step 1: Write the failing test**

Add to `app/frontend/components/__tests__/MatchUI.test.tsx`, inside the existing `describe("MatchCard -- cache-first expand (W47)", ...)` block (after the test ending at line 157):

```typescript
  it("W64: requests a recommendation with the fixture's real competition, not a hardcoded E0", async () => {
    vi.mocked(getCachedRecommendation).mockResolvedValue(null);
    vi.mocked(generateRecommendation).mockResolvedValue(makeRecommendation());
    const user = userEvent.setup();
    const match = baseMatch({ hasRecommendation: false, league: "SWE" });

    render(<MatchCard match={match} onUpdate={vi.fn()} />);
    await user.click(screen.getByText("Not yet generated"));

    await waitFor(() => expect(generateRecommendation).toHaveBeenCalled());
    expect(generateRecommendation).toHaveBeenCalledWith(expect.objectContaining({ league: "SWE" }));
  });
```

This reuses `makeRecommendation()`, defined earlier in the same file (line 118) for the adjacent W47 tests — no new helper needed.

- [ ] **Step 2: Run it to verify it fails**

Run: `cd /Users/tianqihuang/Documents/GitHub/FPAI/app/frontend && npx vitest run components/__tests__/MatchUI.test.tsx -t "W64"`
Expected: FAIL — `generateRecommendation` was called with `league: "E0"`, not `"SWE"` (it already passes `match.league`, but every `Match` in this codebase was always constructed with a hardcoded `"E0"`, so this is the first test to exercise a non-E0 value at all).

Actually — check this carefully before writing Step 3: `MatchCard` already does `league: match.league` (line 539), so this test's failure mode depends on whether `baseMatch()`'s `league: "SWE"` override flows through correctly. If it already passes, that confirms `MatchCard` was never the bug — `fixtureToMatch` was. Either way, proceed to Step 3, which is required regardless.

- [ ] **Step 3: Fix `fixtureToMatch`**

In `app/frontend/components/MatchUI.tsx`, change line 109 inside `fixtureToMatch`:

```typescript
  return {
    id: fixture.match_id,
    league: fixture.competition ?? "E0",
    tier: "competition_specific",
```

- [ ] **Step 4: Thread `league` through the "Full analysis" link**

`MatchAnalysisPage` (Task 10) needs to know which competition it's analyzing — the link `MatchCard` renders today has no way to carry that. In `app/frontend/components/MatchUI.tsx`, update the `Link` inside `MatchCard` (~line 630):

```typescript
                <Link
                  href={`/matches/${match.id}?home=${encodeURIComponent(match.home)}&away=${encodeURIComponent(
                    match.away
                  )}&date=${match.kickoffIso.slice(0, 10)}&league=${encodeURIComponent(match.league)}`}
                  className="mt-3 inline-flex items-center gap-1 text-sm font-medium text-accent"
                >
                  Full analysis <CaretRight size={12} />
                </Link>
```

- [ ] **Step 5: Run the test to verify it passes**

Run: `cd /Users/tianqihuang/Documents/GitHub/FPAI/app/frontend && npx vitest run components/__tests__/MatchUI.test.tsx -t "W64"`
Expected: PASS.

- [ ] **Step 6: Run the full frontend suite**

Run: `cd /Users/tianqihuang/Documents/GitHub/FPAI/app/frontend && npm test`
Expected: same pass count as before, plus 1. (This step also validates Task 2's optional-field choice — every existing hand-built `Fixture` in other test files should still type-check and pass unchanged.)

- [ ] **Step 7: Commit**

```bash
git add app/frontend/components/MatchUI.tsx app/frontend/components/__tests__/MatchUI.test.tsx
git commit -m "fix(app): use each fixture's real competition instead of a hardcoded E0

fixtureToMatch() previously hardcoded league: \"E0\" for every fixture,
meaning a Sweden match's card already silently requested its
recommendation with the wrong league. Now reads the real value backend
Task W64 added. Also threads league through MatchCard's Full analysis
link so Match Analysis (W64 Task 10) gets it too."
```

---

### Task 4: Frontend — pure Dashboard derivation helpers

**Files:**
- Create: `app/frontend/lib/dashboardMetrics.ts`
- Test: `app/frontend/lib/dashboardMetrics.test.ts`

- [ ] **Step 1: Write the failing tests**

Create `app/frontend/lib/dashboardMetrics.test.ts`:

```typescript
import { describe, expect, it } from "vitest";
import { countByOverall, groupByLeague, rankTopEdges, sortMatches } from "./dashboardMetrics";
import type { Match } from "@/components/MatchUI";

function match(overrides: Partial<Match> = {}): Match {
  return {
    id: "m1",
    league: "E0",
    tier: "competition_specific",
    kickoffIso: "2026-08-22T15:00:00Z",
    home: "Arsenal",
    away: "Everton",
    status: "upcoming",
    hasRecommendation: true,
    overall: "direct_bet",
    confidence: "medium",
    markets: [],
    explanation: "",
    limitations: [],
    predictionBasis: "team_history_and_market",
    coldStartRisk: false,
    featureCompleteness: 0.9,
    unknownTeam: false,
    invalidMarketCount: 0,
    ...overrides,
  };
}

describe("countByOverall", () => {
  it("counts only matches with a generated recommendation, grouped by overall", () => {
    const matches = [
      match({ id: "1", overall: "direct_bet" }),
      match({ id: "2", overall: "direct_bet" }),
      match({ id: "3", overall: "conditional" }),
      match({ id: "4", hasRecommendation: false }),
    ];
    expect(countByOverall(matches)).toEqual({
      direct_bet: 2,
      conditional: 1,
      no_bet: 0,
      insufficient_data: 0,
    });
  });

  it("returns all-zero counts for an empty list", () => {
    expect(countByOverall([])).toEqual({
      direct_bet: 0,
      conditional: 0,
      no_bet: 0,
      insufficient_data: 0,
    });
  });
});

describe("rankTopEdges", () => {
  it("ranks by value_edge descending, limited to N, excluding matches with no priced market", () => {
    const matches = [
      match({ id: "low", markets: [{ market: "result_3way", selection: "home", recommendationType: "direct_bet", currentOdds: 2.0, minOdds: 1.5, mlProbability: 0.5, impliedProbability: 0.5, valueEdge: 0.02 }] }),
      match({ id: "high", markets: [{ market: "result_3way", selection: "home", recommendationType: "direct_bet", currentOdds: 2.0, minOdds: 1.5, mlProbability: 0.5, impliedProbability: 0.5, valueEdge: 0.09 }] }),
      match({ id: "no-odds", markets: [{ market: "result_3way", selection: "home", recommendationType: "direct_bet", currentOdds: null, minOdds: 1.5, mlProbability: 0.5, impliedProbability: 0.5, valueEdge: 0.5 }] }),
      match({ id: "no-rec", hasRecommendation: false }),
    ];
    const ranked = rankTopEdges(matches, 5);
    expect(ranked.map((r) => r.match.id)).toEqual(["high", "low"]);
    expect(ranked[0].edge).toBeCloseTo(0.09);
  });

  it("respects the limit", () => {
    const matches = Array.from({ length: 10 }, (_, i) =>
      match({
        id: `m${i}`,
        markets: [{ market: "result_3way", selection: "home", recommendationType: "direct_bet", currentOdds: 2.0, minOdds: 1.5, mlProbability: 0.5, impliedProbability: 0.5, valueEdge: i / 100 }],
      })
    );
    expect(rankTopEdges(matches, 3)).toHaveLength(3);
  });
});

describe("groupByLeague", () => {
  it("groups matches by league, preserving first-seen order, with real labels", () => {
    const matches = [
      match({ id: "1", league: "E0" }),
      match({ id: "2", league: "SWE" }),
      match({ id: "3", league: "E0" }),
    ];
    const groups = groupByLeague(matches);
    expect(groups.map((g) => g.league)).toEqual(["E0", "SWE"]);
    expect(groups[0].label).toBe("Premier League");
    expect(groups[1].label).toBe("Allsvenskan");
    expect(groups[0].matches.map((m) => m.id)).toEqual(["1", "3"]);
  });

  it("falls back to the raw league code for an unrecognized value", () => {
    const groups = groupByLeague([match({ league: "XYZ" })]);
    expect(groups[0].label).toBe("XYZ");
  });
});

describe("sortMatches", () => {
  it("sorts by kickoff time ascending", () => {
    const matches = [
      match({ id: "later", kickoffIso: "2026-08-22T18:00:00Z" }),
      match({ id: "earlier", kickoffIso: "2026-08-22T11:00:00Z" }),
    ];
    expect(sortMatches(matches, "kickoff").map((m) => m.id)).toEqual(["earlier", "later"]);
  });

  it("sorts by edge descending, treating no-priced-market matches as lowest", () => {
    const matches = [
      match({ id: "no-market", markets: [] }),
      match({ id: "priced", markets: [{ market: "result_3way", selection: "home", recommendationType: "direct_bet", currentOdds: 2.0, minOdds: 1.5, mlProbability: 0.5, impliedProbability: 0.5, valueEdge: 0.05 }] }),
    ];
    expect(sortMatches(matches, "edge").map((m) => m.id)).toEqual(["priced", "no-market"]);
  });

  it("does not mutate the input array", () => {
    const matches = [match({ id: "a", kickoffIso: "2026-08-22T18:00:00Z" }), match({ id: "b", kickoffIso: "2026-08-22T11:00:00Z" })];
    const original = [...matches];
    sortMatches(matches, "kickoff");
    expect(matches).toEqual(original);
  });
});
```

- [ ] **Step 2: Run to verify it fails**

Run: `cd /Users/tianqihuang/Documents/GitHub/FPAI/app/frontend && npx vitest run lib/dashboardMetrics.test.ts`
Expected: FAIL — `Cannot find module './dashboardMetrics'`.

- [ ] **Step 3: Export `bestMarket` from `MatchUI.tsx`**

`dashboardMetrics.ts` needs it. In `app/frontend/components/MatchUI.tsx`, change line 288:

```typescript
export function bestMarket(match: Match): MarketRec | undefined {
  return [...match.markets].sort((a, b) => b.valueEdge - a.valueEdge)[0];
}
```

- [ ] **Step 4: Implement `dashboardMetrics.ts`**

Create `app/frontend/lib/dashboardMetrics.ts`:

```typescript
import { bestMarket, type Match, type Overall } from "@/components/MatchUI";

export type OverallCounts = Record<Overall, number>;

/** Counts only matches with a generated recommendation -- a match still
 * showing "Not yet generated" has no overall worth counting. */
export function countByOverall(matches: Match[]): OverallCounts {
  const counts: OverallCounts = { direct_bet: 0, conditional: 0, no_bet: 0, insufficient_data: 0 };
  for (const m of matches) {
    if (!m.hasRecommendation) continue;
    counts[m.overall] += 1;
  }
  return counts;
}

export type TopEdge = { match: Match; edge: number };

/** Ranks by the best-priced market's value_edge, descending. A match with
 * no recommendation, or whose best market has no live odds (current_odds
 * null -- an unpriceable edge, not a real one), is excluded rather than
 * ranked with a fabricated value. */
export function rankTopEdges(matches: Match[], limit: number): TopEdge[] {
  const priced: TopEdge[] = [];
  for (const m of matches) {
    if (!m.hasRecommendation) continue;
    const shown = bestMarket(m);
    if (!shown || shown.currentOdds === null) continue;
    priced.push({ match: m, edge: shown.valueEdge });
  }
  return priced.sort((a, b) => b.edge - a.edge).slice(0, limit);
}

const LEAGUE_LABEL: Record<string, string> = { E0: "Premier League", SWE: "Allsvenskan" };

export type LeagueGroup = { league: string; label: string; matches: Match[] };

/** Groups by league in first-seen order (not alphabetical) so the section
 * order tracks whatever order the fixtures actually arrived in. */
export function groupByLeague(matches: Match[]): LeagueGroup[] {
  const order: string[] = [];
  const groups = new Map<string, Match[]>();
  for (const m of matches) {
    if (!groups.has(m.league)) {
      groups.set(m.league, []);
      order.push(m.league);
    }
    groups.get(m.league)!.push(m);
  }
  return order.map((league) => ({
    league,
    label: LEAGUE_LABEL[league] ?? league,
    matches: groups.get(league)!,
  }));
}

export type MatchSort = "kickoff" | "edge";

/** Returns a new array -- never mutates `matches` -- since callers hold
 * this list in React state and an in-place sort would be a silent mutation
 * bug (stale closures/memoization comparing the same array reference). */
export function sortMatches(matches: Match[], sort: MatchSort): Match[] {
  if (sort === "kickoff") {
    return [...matches].sort((a, b) => a.kickoffIso.localeCompare(b.kickoffIso));
  }
  return [...matches].sort((a, b) => {
    const edgeA = bestMarket(a)?.valueEdge ?? -Infinity;
    const edgeB = bestMarket(b)?.valueEdge ?? -Infinity;
    return edgeB - edgeA;
  });
}
```

- [ ] **Step 5: Run to verify it passes**

Run: `cd /Users/tianqihuang/Documents/GitHub/FPAI/app/frontend && npx vitest run lib/dashboardMetrics.test.ts`
Expected: all pass.

- [ ] **Step 6: Full suite + typecheck**

Run: `cd /Users/tianqihuang/Documents/GitHub/FPAI/app/frontend && npm test && npx tsc --noEmit`
Expected: all pass, no new type errors.

- [ ] **Step 7: Commit**

```bash
git add app/frontend/lib/dashboardMetrics.ts app/frontend/lib/dashboardMetrics.test.ts app/frontend/components/MatchUI.tsx
git commit -m "feat(app): add pure Dashboard derivation helpers (grouping/sorting/counting)"
```

---

### Task 5: Frontend — `DashboardRail` (Edge Distribution + Top Edges)

**Files:**
- Create: `app/frontend/components/DashboardRail.tsx`
- Test: `app/frontend/components/__tests__/DashboardRail.test.tsx`
- Modify: `app/frontend/components/MatchUI.tsx:284-287` (export `formatEdge`)

- [ ] **Step 1: Export `formatEdge`**

In `app/frontend/components/MatchUI.tsx`, change line 284:

```typescript
export function formatEdge(v: number) {
  const pct = (v * 100).toFixed(1);
  return v >= 0 ? `+${pct}%` : `${pct}%`;
}
```

- [ ] **Step 2: Write the failing test**

Create `app/frontend/components/__tests__/DashboardRail.test.tsx`:

```typescript
import { render, screen } from "@testing-library/react";
import { describe, expect, it } from "vitest";
import { DashboardRail } from "../DashboardRail";
import type { Match } from "../MatchUI";

function match(overrides: Partial<Match> = {}): Match {
  return {
    id: "m1",
    league: "E0",
    tier: "competition_specific",
    kickoffIso: "2026-08-22T15:00:00Z",
    home: "Arsenal",
    away: "Everton",
    status: "upcoming",
    hasRecommendation: true,
    overall: "direct_bet",
    confidence: "medium",
    markets: [{ market: "result_3way", selection: "home", recommendationType: "direct_bet", currentOdds: 2.0, minOdds: 1.5, mlProbability: 0.5, impliedProbability: 0.5, valueEdge: 0.05 }],
    explanation: "",
    limitations: [],
    predictionBasis: "team_history_and_market",
    coldStartRisk: false,
    featureCompleteness: 0.9,
    unknownTeam: false,
    invalidMarketCount: 0,
    ...overrides,
  };
}

describe("DashboardRail", () => {
  it("shows an empty state for no matches", () => {
    render(<DashboardRail matches={[]} />);
    expect(screen.getByText("No matches loaded yet.")).toBeInTheDocument();
    expect(screen.getByText("No priced edges yet.")).toBeInTheDocument();
  });

  it("renders Edge Distribution counts per status", () => {
    const matches = [
      match({ id: "1", overall: "direct_bet" }),
      match({ id: "2", overall: "direct_bet" }),
      match({ id: "3", overall: "conditional" }),
    ];
    render(<DashboardRail matches={matches} />);
    expect(screen.getByText("Direct Bet")).toBeInTheDocument();
    expect(screen.getByText("Conditional")).toBeInTheDocument();
    expect(screen.getByText("2")).toBeInTheDocument();
    expect(screen.getByText("3")).toBeInTheDocument(); // total in the donut center
  });

  it("renders Top Edges ranked by value_edge descending, as links to Match Analysis", () => {
    const matches = [
      match({ id: "low", home: "LowEdgeTeam", markets: [{ ...match().markets[0], valueEdge: 0.01 }] }),
      match({ id: "high", home: "HighEdgeTeam", markets: [{ ...match().markets[0], valueEdge: 0.09 }] }),
    ];
    render(<DashboardRail matches={matches} />);
    const links = screen.getAllByRole("link");
    expect(links[0]).toHaveTextContent("HighEdgeTeam");
    expect(links[0]).toHaveAttribute("href", expect.stringContaining("/matches/high"));
    expect(screen.getByText("+9.0%")).toBeInTheDocument();
  });
});
```

- [ ] **Step 3: Run to verify it fails**

Run: `cd /Users/tianqihuang/Documents/GitHub/FPAI/app/frontend && npx vitest run components/__tests__/DashboardRail.test.tsx`
Expected: FAIL — `Cannot find module '../DashboardRail'`.

- [ ] **Step 4: Implement `DashboardRail.tsx`**

Create `app/frontend/components/DashboardRail.tsx`:

```typescript
"use client";

import Link from "next/link";
import { useMemo } from "react";

import { countByOverall, rankTopEdges } from "@/lib/dashboardMetrics";
import { formatEdge, type Match, type Overall } from "./MatchUI";

// W64: fixed order + colors reused verbatim from STATUS_META's existing,
// already-locked status palette (app/globals.css D6) -- not a new color
// pick. Inline styles (not Tailwind classes) since these are chosen
// programmatically per status key, same pattern TeamBadge already uses for
// per-team colors.
const DONUT_ORDER: Overall[] = ["direct_bet", "conditional", "no_bet", "insufficient_data"];
const DONUT_COLOR: Record<Overall, string> = {
  direct_bet: "var(--status-good)",
  conditional: "var(--status-warning)",
  no_bet: "var(--text-muted)",
  insufficient_data: "var(--status-serious)",
};
const DONUT_LABEL: Record<Overall, string> = {
  direct_bet: "Direct Bet",
  conditional: "Conditional",
  no_bet: "No Edge",
  insufficient_data: "No Data",
};

const RADIUS = 40;
const CIRCUMFERENCE = 2 * Math.PI * RADIUS;
const SEGMENT_GAP = 3; // dataviz mark spec: a visible surface gap between adjacent segments

function matchHref(m: Match) {
  return `/matches/${m.id}?home=${encodeURIComponent(m.home)}&away=${encodeURIComponent(
    m.away
  )}&date=${m.kickoffIso.slice(0, 10)}&league=${encodeURIComponent(m.league)}`;
}

export function DashboardRail({ matches }: { matches: Match[] }) {
  const counts = useMemo(() => countByOverall(matches), [matches]);
  const topEdges = useMemo(() => rankTopEdges(matches, 5), [matches]);
  const total = DONUT_ORDER.reduce((sum, key) => sum + counts[key], 0);

  let cumulative = 0;
  const arcs = DONUT_ORDER.filter((key) => counts[key] > 0).map((key) => {
    const frac = counts[key] / total;
    const rawDash = frac * CIRCUMFERENCE;
    const arc = { key, dash: Math.max(rawDash - SEGMENT_GAP, 0), offset: cumulative };
    cumulative += rawDash;
    return arc;
  });

  return (
    <aside className="flex w-full flex-col gap-6 lg:w-72">
      <section className="rounded-lg border border-border p-4">
        <h2 className="text-xs font-semibold uppercase tracking-wide text-muted">Edge Distribution</h2>
        {total === 0 ? (
          <p className="mt-3 text-sm text-ink-secondary">No matches loaded yet.</p>
        ) : (
          <div className="mt-3 flex items-center gap-4">
            <svg viewBox="0 0 100 100" width={88} height={88} className="shrink-0 -rotate-90">
              <circle cx="50" cy="50" r={RADIUS} fill="none" stroke="var(--gridline)" strokeWidth={14} />
              {arcs.map((arc) => (
                <circle
                  key={arc.key}
                  cx="50"
                  cy="50"
                  r={RADIUS}
                  fill="none"
                  stroke={DONUT_COLOR[arc.key]}
                  strokeWidth={14}
                  strokeDasharray={`${arc.dash} ${CIRCUMFERENCE - arc.dash}`}
                  strokeDashoffset={-arc.offset}
                />
              ))}
              <text
                x="50"
                y="50"
                textAnchor="middle"
                dominantBaseline="central"
                className="fill-ink text-[22px] font-semibold"
                style={{ transform: "rotate(90deg)", transformOrigin: "50px 50px" }}
              >
                {total}
              </text>
            </svg>
            <ul className="flex flex-1 flex-col gap-1.5 text-xs">
              {DONUT_ORDER.filter((key) => counts[key] > 0).map((key) => (
                <li key={key} className="flex items-center justify-between gap-2">
                  <span className="flex items-center gap-1.5 text-ink-secondary">
                    <span className="h-2 w-2 shrink-0 rounded-full" style={{ background: DONUT_COLOR[key] }} />
                    {DONUT_LABEL[key]}
                  </span>
                  <span className="font-mono text-ink">{counts[key]}</span>
                </li>
              ))}
            </ul>
          </div>
        )}
      </section>

      <section className="rounded-lg border border-border p-4">
        <h2 className="text-xs font-semibold uppercase tracking-wide text-muted">Top Edges</h2>
        {topEdges.length === 0 ? (
          <p className="mt-3 text-sm text-ink-secondary">No priced edges yet.</p>
        ) : (
          <ul className="mt-3 flex flex-col gap-2.5">
            {topEdges.map(({ match, edge }) => (
              <li key={match.id}>
                <Link href={matchHref(match)} className="flex items-center justify-between gap-2 text-sm text-ink-secondary hover:text-ink">
                  <span className="truncate">
                    {match.home} v {match.away}
                  </span>
                  <span className="shrink-0 font-mono text-good">{formatEdge(edge)}</span>
                </Link>
              </li>
            ))}
          </ul>
        )}
      </section>
    </aside>
  );
}
```

- [ ] **Step 5: Run to verify it passes**

Run: `cd /Users/tianqihuang/Documents/GitHub/FPAI/app/frontend && npx vitest run components/__tests__/DashboardRail.test.tsx`
Expected: all pass.

If the "renders Edge Distribution counts" test's `getByText("3")` is ambiguous (e.g. collides with another "3" on the page), scope it with `within()` against the first `section` instead — check the actual DOM before assuming; this is diagnostic, not a rewrite.

- [ ] **Step 6: Commit**

```bash
git add app/frontend/components/DashboardRail.tsx app/frontend/components/__tests__/DashboardRail.test.tsx app/frontend/components/MatchUI.tsx
git commit -m "feat(app): add DashboardRail (Edge Distribution + Top Edges), reusing the existing status palette"
```

---

### Task 6: Frontend — `AppShell`

**Files:**
- Create: `app/frontend/components/AppShell.tsx`
- Test: `app/frontend/components/__tests__/AppShell.test.tsx`

- [ ] **Step 1: Write the failing test**

Create `app/frontend/components/__tests__/AppShell.test.tsx`:

```typescript
import { render, screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { describe, expect, it, vi, beforeEach } from "vitest";
import { AppShell } from "../AppShell";

vi.mock("@/lib/api", () => ({
  getStatus: vi.fn(),
  getFixtures: vi.fn(),
  getSandboxStatus: vi.fn(),
}));

import { getFixtures, getSandboxStatus, getStatus } from "@/lib/api";

describe("AppShell", () => {
  beforeEach(() => {
    vi.mocked(getStatus).mockReset();
    vi.mocked(getFixtures).mockReset();
    vi.mocked(getSandboxStatus).mockReset();
    vi.mocked(getSandboxStatus).mockResolvedValue({ sandbox_mode: false, as_of: null });
  });

  it("renders nav links and page content", () => {
    vi.mocked(getStatus).mockRejectedValue(new Error("no backend"));
    render(
      <AppShell active="dashboard">
        <p>page content</p>
      </AppShell>
    );
    expect(screen.getByText("Dashboard")).toBeInTheDocument();
    expect(screen.getByText("All Matches")).toBeInTheDocument();
    expect(screen.getByText("Bets")).toBeInTheDocument();
    expect(screen.getByText("page content")).toBeInTheDocument();
  });

  it("does not crash when the status fetch fails, and shows a placeholder instead of a real value", async () => {
    vi.mocked(getStatus).mockRejectedValue(new Error("no backend"));
    render(
      <AppShell active="dashboard">
        <p>content</p>
      </AppShell>
    );
    await waitFor(() => expect(getStatus).toHaveBeenCalled());
    expect(screen.getByText("—")).toBeInTheDocument();
  });

  it("renders real model status and last-updated once the status fetch resolves", async () => {
    vi.mocked(getStatus).mockResolvedValue({
      data_freshness: { latest_match_date: "2026-07-20", days_since_update: 3, match_count: 100, is_stale: false },
      model_status: { league: { result_3way: { model_type: "x", primary_metric_value: 0.6, metric_name: "m", selected_at: "now" } }, international: {} },
    });
    render(
      <AppShell active="dashboard">
        <p>content</p>
      </AppShell>
    );
    expect(await screen.findByText(/league 1/)).toBeInTheDocument();
    expect(screen.getByText(/2026-07-20/)).toBeInTheDocument();
  });

  it("shows Active Edges only when the prop is provided", () => {
    vi.mocked(getStatus).mockRejectedValue(new Error("no backend"));
    const { rerender } = render(
      <AppShell active="dashboard">
        <p>content</p>
      </AppShell>
    );
    expect(screen.queryByText("Active Edges")).not.toBeInTheDocument();
    rerender(
      <AppShell active="dashboard" activeEdgesCount={4}>
        <p>content</p>
      </AppShell>
    );
    expect(screen.getByText("Active Edges")).toBeInTheDocument();
    expect(screen.getByText("4")).toBeInTheDocument();
  });

  it("does not fetch fixtures on mount -- only once the search input is focused", async () => {
    vi.mocked(getStatus).mockRejectedValue(new Error("no backend"));
    render(
      <AppShell active="dashboard">
        <p>content</p>
      </AppShell>
    );
    await waitFor(() => expect(getStatus).toHaveBeenCalled());
    expect(getFixtures).not.toHaveBeenCalled();

    const user = userEvent.setup();
    await user.click(screen.getByPlaceholderText("Search fixtures, teams…"));
    await waitFor(() => expect(getFixtures).toHaveBeenCalledTimes(1));
  });

  it("filters and links search results to Match Analysis, carrying league", async () => {
    vi.mocked(getStatus).mockRejectedValue(new Error("no backend"));
    vi.mocked(getFixtures).mockResolvedValue([
      { match_id: "1", utc_date: "2026-08-22T15:00:00Z", status: "SCHEDULED", home_team: "Arsenal", away_team: "Everton", home_goals: null, away_goals: null, competition: "E0" },
      { match_id: "2", utc_date: "2026-08-23T15:00:00Z", status: "SCHEDULED", home_team: "Malmo FF", away_team: "AIK", home_goals: null, away_goals: null, competition: "SWE" },
    ]);
    const user = userEvent.setup();
    render(
      <AppShell active="dashboard">
        <p>content</p>
      </AppShell>
    );
    const input = screen.getByPlaceholderText("Search fixtures, teams…");
    await user.click(input);
    await user.type(input, "malmo");

    const link = await screen.findByText(/Malmo FF/);
    expect(link.closest("a")).toHaveAttribute("href", expect.stringContaining("league=SWE"));
    expect(screen.queryByText(/Arsenal/)).not.toBeInTheDocument();
  });
});
```

- [ ] **Step 2: Run to verify it fails**

Run: `cd /Users/tianqihuang/Documents/GitHub/FPAI/app/frontend && npx vitest run components/__tests__/AppShell.test.tsx`
Expected: FAIL — `Cannot find module '../AppShell'`.

- [ ] **Step 3: Implement `AppShell.tsx`**

Create `app/frontend/components/AppShell.tsx`:

```typescript
"use client";

import Link from "next/link";
import { useRef, useState, useEffect } from "react";
import { MagnifyingGlass } from "@phosphor-icons/react";

import { getFixtures, getStatus } from "@/lib/api";
import type { Fixture, StatusResponse } from "@/lib/types";
import { useSandboxAsOf } from "@/lib/useSandboxAsOf";

function matchAnalysisHref(f: Fixture) {
  return `/matches/${f.match_id}?home=${encodeURIComponent(f.home_team)}&away=${encodeURIComponent(
    f.away_team
  )}&date=${f.utc_date.slice(0, 10)}&league=${encodeURIComponent(f.competition ?? "E0")}`;
}

const NAV_ITEMS: { href: string; label: string; key: "dashboard" | "matches" | "bets" }[] = [
  { href: "/", label: "Dashboard", key: "dashboard" },
  { href: "/matches", label: "All Matches", key: "matches" },
  { href: "/bets", label: "Bets", key: "bets" },
];

export function AppShell({
  active,
  activeEdgesCount,
  children,
}: {
  active: "dashboard" | "matches" | "bets";
  activeEdgesCount?: number;
  children: React.ReactNode;
}) {
  const { asOf } = useSandboxAsOf();
  const [status, setStatus] = useState<StatusResponse | null>(null);
  const [query, setQuery] = useState("");
  const [searchFixtures, setSearchFixtures] = useState<Fixture[]>([]);
  const fetchedSearchRef = useRef(false);

  useEffect(() => {
    let cancelled = false;
    (async () => {
      try {
        const s = await getStatus();
        if (!cancelled) setStatus(s);
      } catch {
        // W17's StatusFooter precedent: a passive display, not worth an
        // error state of its own -- the sidebar just shows "--" instead.
        if (!cancelled) setStatus(null);
      }
    })();
    return () => {
      cancelled = true;
    };
  }, []);

  function ensureSearchFixturesLoaded() {
    // Lazy on purpose -- see this plan's "Before you start" note. Eagerly
    // fetching on mount would add a call to the same shared `getFixtures`
    // mock BetTracker.fixtureError.test.tsx asserts an exact count against.
    if (fetchedSearchRef.current) return;
    fetchedSearchRef.current = true;
    const from = new Date(asOf);
    const to = new Date(asOf);
    to.setUTCDate(to.getUTCDate() + 90);
    (async () => {
      try {
        const fixtures = await getFixtures(from.toISOString().slice(0, 10), to.toISOString().slice(0, 10));
        setSearchFixtures(fixtures ?? []);
      } catch {
        setSearchFixtures([]);
      }
    })();
  }

  const q = query.trim().toLowerCase();
  const results =
    q.length === 0
      ? []
      : searchFixtures
          .filter((f) => f.home_team.toLowerCase().includes(q) || f.away_team.toLowerCase().includes(q))
          .slice(0, 8);

  const dataFreshness = status?.data_freshness;
  const leagueModelCount = Object.keys(status?.model_status.league ?? {}).length;
  const internationalModelCount = Object.keys(status?.model_status.international ?? {}).length;

  return (
    <div className="min-h-screen lg:flex">
      <aside className="flex shrink-0 flex-col justify-between border-b border-border px-4 py-5 lg:h-screen lg:w-56 lg:border-b-0 lg:border-r lg:px-5 lg:py-6">
        <div>
          <span className="text-sm font-semibold tracking-tight text-ink">FPAI</span>
          <nav className="mt-6 flex flex-col gap-1 text-sm">
            {NAV_ITEMS.map((item) => (
              <Link
                key={item.key}
                href={item.href}
                className={`rounded-md px-2 py-1.5 transition-colors duration-150 ${
                  active === item.key ? "bg-surface text-ink" : "text-ink-secondary hover:text-ink"
                }`}
              >
                {item.label}
              </Link>
            ))}
          </nav>
        </div>

        <div className="mt-6 flex flex-col gap-3 border-t border-border pt-4 text-xs lg:mt-0">
          {activeEdgesCount !== undefined && (
            <div>
              <div className="text-muted uppercase tracking-wide">Active Edges</div>
              <div className="mt-0.5 font-mono text-base text-ink">{activeEdgesCount}</div>
            </div>
          )}
          <div>
            <div className="text-muted uppercase tracking-wide">Model Status</div>
            <div className="mt-0.5 text-ink-secondary">
              {status ? `league ${leagueModelCount} · international ${internationalModelCount}` : "—"}
            </div>
          </div>
          <div>
            <div className="text-muted uppercase tracking-wide">Last Updated</div>
            <div className={`mt-0.5 ${dataFreshness?.is_stale ? "text-warning" : "text-ink-secondary"}`}>
              {dataFreshness
                ? `${dataFreshness.latest_match_date ?? "unknown"}${
                    dataFreshness.days_since_update !== null ? ` (${dataFreshness.days_since_update}d ago)` : ""
                  }`
                : "—"}
            </div>
          </div>
        </div>
      </aside>

      <div className="flex min-w-0 flex-1 flex-col">
        <div className="border-b border-border px-4 py-3 sm:px-6">
          <div className="relative max-w-md">
            <MagnifyingGlass size={16} className="pointer-events-none absolute left-3 top-1/2 -translate-y-1/2 text-muted" />
            <input
              value={query}
              onChange={(e) => setQuery(e.target.value)}
              onFocus={ensureSearchFixturesLoaded}
              placeholder="Search fixtures, teams…"
              className="w-full rounded-lg border border-border bg-surface py-2 pl-9 pr-3 text-sm text-ink outline-none placeholder:text-muted focus:border-accent"
            />
            {results.length > 0 && (
              <div className="absolute left-0 right-0 top-full z-10 mt-1 flex flex-col gap-1 rounded-lg border border-border bg-page p-1.5 shadow-lg">
                {results.map((f) => (
                  <Link
                    key={f.match_id}
                    href={matchAnalysisHref(f)}
                    onClick={() => setQuery("")}
                    className="rounded-md px-2 py-1.5 text-sm text-ink hover:bg-surface"
                  >
                    {f.home_team} v {f.away_team}
                    <span className="ml-2 text-xs text-ink-secondary">{f.utc_date.slice(0, 10)}</span>
                  </Link>
                ))}
              </div>
            )}
          </div>
        </div>

        <main className="flex-1 px-4 py-8 sm:px-6">{children}</main>
      </div>
    </div>
  );
}
```

- [ ] **Step 4: Run to verify it passes**

Run: `cd /Users/tianqihuang/Documents/GitHub/FPAI/app/frontend && npx vitest run components/__tests__/AppShell.test.tsx`
Expected: all pass.

- [ ] **Step 5: Typecheck**

Run: `cd /Users/tianqihuang/Documents/GitHub/FPAI/app/frontend && npx tsc --noEmit`

- [ ] **Step 6: Commit**

```bash
git add app/frontend/components/AppShell.tsx app/frontend/components/__tests__/AppShell.test.tsx
git commit -m "feat(app): add AppShell (sidebar nav, footer stats, lazy top-bar search)"
```

---

### Task 7: Wire `AppShell` into `DashboardPage`, add league grouping + sort

**Files:**
- Modify: `app/frontend/components/MatchUI.tsx:238-241` (`TIER_LABEL`), `:651-804` (`DashboardPage`)
- Existing tests (verify only, no edits expected): `MatchUI.dateboundary.test.tsx`, `MatchUI.emptyFallback.test.tsx`, `MatchUI.precompute.test.tsx`, `MatchUI.race.test.tsx`

- [ ] **Step 1: Fix the misleading tier label**

`TierTag` currently shows "EPL" for every `competition_specific` match, including Sweden ones (a match's `tier` reflects model-data-quality, not literally EPL — this only read correctly before Sweden existed). In `app/frontend/components/MatchUI.tsx`, change line 239:

```typescript
const TIER_LABEL: Record<Tier, string> = {
  competition_specific: "Modeled",
  general_purpose: "General",
};
```

- [ ] **Step 2: Add the imports `DashboardPage` needs**

In `app/frontend/components/MatchUI.tsx`, near the top (after the existing `@/lib/useSandboxAsOf` import, ~line 39):

```typescript
import { groupByLeague, sortMatches, type MatchSort } from "@/lib/dashboardMetrics";
import { AppShell } from "./AppShell";
import { DashboardRail } from "./DashboardRail";
```

- [ ] **Step 3: Replace `DashboardPage`'s body**

Replace the entire `DashboardPage` function (`app/frontend/components/MatchUI.tsx:651-804`) — keep the existing `useState`/`useEffect`/`updateMatch`/`updateNextMatch` exactly as they are today (lines 652-752 are unchanged), only replacing the `return` statement and adding one new piece of state:

```typescript
export function DashboardPage() {
  const { asOf, sandboxMode } = useSandboxAsOf();
  const [matches, setMatches] = useState<Match[] | null>(null);
  const [nextMatches, setNextMatches] = useState<Match[] | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [retryTick, setRetryTick] = useState(0);
  const [sort, setSort] = useState<MatchSort>("kickoff");

  // ... existing useEffect (lines 666-744), updateMatch (746-748),
  // updateNextMatch (750-752) unchanged ...

  const usingFallback = matches !== null && matches.length === 0 && nextMatches !== null && nextMatches.length > 0;
  const shownMatches = usingFallback ? nextMatches! : matches ?? [];
  const updateShown = usingFallback ? updateNextMatch : updateMatch;
  const activeEdgesCount = shownMatches.filter(
    (m) => m.hasRecommendation && (m.overall === "direct_bet" || m.overall === "conditional")
  ).length;
  const leagueGroups = groupByLeague(sortMatches(shownMatches, sort));

  return (
    <AppShell active="dashboard" activeEdgesCount={matches !== null ? activeEdgesCount : undefined}>
      <div className="lg:flex lg:items-start lg:gap-8">
        <div className="min-w-0 flex-1">
          <div className="flex flex-wrap items-center justify-between gap-4">
            <div>
              <h1 className="text-xl font-semibold tracking-tight text-ink">Today&apos;s Edges</h1>
              <p className="mt-1 text-sm text-ink-secondary">
                Real E0 &amp; Allsvenskan fixtures for today. Expand a card to generate its recommendation.
              </p>
            </div>
            {shownMatches.length > 0 && (
              <SegmentedControl
                options={[
                  { value: "kickoff", label: "Kickoff" },
                  { value: "edge", label: "Edge %" },
                ]}
                value={sort}
                onChange={setSort}
              />
            )}
          </div>

          <div className="mt-6">
            {error && <ErrorState message={error} onRetry={() => setRetryTick((t) => t + 1)} />}
            {!error && matches === null && <LoadingRows />}
            {!error && matches !== null && matches.length === 0 && nextMatches === null && (
              <>
                <p className="py-4 text-center text-sm text-ink-secondary">No fixtures today.</p>
                <LoadingRows />
              </>
            )}
            {!error && matches !== null && matches.length === 0 && nextMatches !== null && nextMatches.length === 0 && (
              <p className="py-8 text-center text-sm text-ink-secondary">No fixtures today.</p>
            )}
            {!error && shownMatches.length > 0 && (
              <div className="flex flex-col gap-6">
                {usingFallback && (
                  <p className="text-sm text-ink-secondary">No fixtures today — next matches:</p>
                )}
                {leagueGroups.map((group) => (
                  <div key={group.league}>
                    <div className="mb-2 flex items-center gap-2">
                      <h2 className="text-xs font-semibold uppercase tracking-wide text-muted">{group.label}</h2>
                      <span className="rounded-full border border-border px-1.5 py-0.5 text-[10px] text-ink-secondary">
                        {group.matches.length} match{group.matches.length === 1 ? "" : "es"}
                      </span>
                    </div>
                    <div className="flex flex-col gap-2.5">
                      {group.matches.map((m) => (
                        <MatchCard key={m.id} match={m} onUpdate={updateShown} asOf={asOf} sandboxMode={sandboxMode} />
                      ))}
                    </div>
                  </div>
                ))}
              </div>
            )}
          </div>
        </div>

        {shownMatches.length > 0 && (
          <div className="mt-8 lg:mt-0">
            <DashboardRail matches={shownMatches} />
          </div>
        )}
      </div>
    </AppShell>
  );
}
```

- [ ] **Step 4: MatchCard — a small, concrete density pass**

`MatchCard` never had an xG-trend row to begin with (design doc's non-goal, already satisfied — no code change needed for that part). What the sketch's cards still read denser on is corner radius and hover treatment. In `app/frontend/components/MatchUI.tsx`, change `MatchCard`'s outer container (~line 553) from:

```typescript
    <div className="rounded-lg border border-border transition-transform duration-150 hover:-translate-y-px">
```

to:

```typescript
    <div className="rounded-xl border border-border transition-all duration-150 hover:-translate-y-px hover:border-border-strong">
```

This is deliberately scoped to the container only — the internal padding/type-scale (`p-3.5`, `text-sm`, `text-xs`) already matches the sketch's density; a broader rewrite isn't needed to satisfy the design doc's "denser card look" direction.

- [ ] **Step 5: Run the existing Dashboard-touching tests**

Run: `cd /Users/tianqihuang/Documents/GitHub/FPAI/app/frontend && npx vitest run components/__tests__/MatchUI.dateboundary.test.tsx components/__tests__/MatchUI.emptyFallback.test.tsx components/__tests__/MatchUI.precompute.test.tsx components/__tests__/MatchUI.race.test.tsx`

Expected: all pass unchanged. If anything fails, read the failure carefully before changing test files — per this plan's "Before you start" notes, the two known risk areas (`getFixtures` call-count, placeholder collision) don't apply to these four files (they don't render `BetTrackerPage`), so a failure here means something in Step 3 broke real Dashboard behavior, not a known/expected test-compatibility tradeoff. Fix the component, not the test.

- [ ] **Step 6: Full frontend suite + typecheck**

Run: `cd /Users/tianqihuang/Documents/GitHub/FPAI/app/frontend && npm test && npx tsc --noEmit`

- [ ] **Step 7: Commit**

```bash
git add app/frontend/components/MatchUI.tsx
git commit -m "feat(app): wire AppShell into DashboardPage, add league grouping + sort + rail"
```

---

### Task 8: Wire `AppShell` into `MatchExplorerPage`

**Files:**
- Modify: `app/frontend/components/MatchUI.tsx:810-913`

- [ ] **Step 1: Replace the wrapper**

In `app/frontend/components/MatchUI.tsx`, `MatchExplorerPage`'s `return` currently starts with `<main className="mx-auto max-w-4xl px-4 py-8 sm:px-6"><DraftNav active="matches" />`. Replace lines 880-882:

```typescript
  return (
    <AppShell active="matches">
```

And the closing tag — replace the final `</main>` (line 912) with `</AppShell>`. Everything between (`<h1>Match Explorer</h1>` through the search input and match list, lines 884-910) stays exactly as-is — this task only swaps the wrapper, no other change. Match Explorer keeps its own in-page team-name filter unchanged (it's a *list filter* over a 90-day window; `AppShell`'s search is a *global quick-jump*, a different job — see the design doc's Architecture section).

- [ ] **Step 2: Run the Match Explorer tests**

Run: `cd /Users/tianqihuang/Documents/GitHub/FPAI/app/frontend && npx vitest run -t "MatchExplorer" components/__tests__/`

Also re-run the four files from Task 7 Step 4 (several exercise `MatchExplorerPage` directly): `npx vitest run components/__tests__/MatchUI.dateboundary.test.tsx components/__tests__/MatchUI.precompute.test.tsx components/__tests__/MatchUI.race.test.tsx`

Expected: all pass unchanged.

- [ ] **Step 3: Commit**

```bash
git add app/frontend/components/MatchUI.tsx
git commit -m "feat(app): wire AppShell into MatchExplorerPage"
```

---

### Task 9: Wire `AppShell` into `MatchAnalysisPage`, thread `league` through

**Files:**
- Modify: `app/frontend/components/MatchUI.tsx:1034-1227` (`MatchAnalysisPage`)
- Modify: `app/frontend/app/matches/[id]/page.tsx`

- [ ] **Step 1: Add an optional `league` prop, defaulting to `"E0"`**

In `app/frontend/components/MatchUI.tsx`, change `MatchAnalysisPage`'s signature (line 1034):

```typescript
export function MatchAnalysisPage({
  id,
  home,
  away,
  date,
  league = "E0",
}: {
  id: string;
  home: string;
  away: string;
  date: string;
  league?: string;
}) {
```

Default value (not a required prop) keeps every existing test call site (`<MatchAnalysisPage id="m1" home="Arsenal" away="Everton" date="2026-08-22" />`, no `league`) valid unchanged.

- [ ] **Step 2: Use it instead of the two hardcoded `"E0"` literals**

Still in `MatchAnalysisPage`, replace line 1065:

```typescript
        rec = await generateRecommendation({ home_team: home, away_team: away, date, league, match_id: id });
```

And in the initial `Match` object passed to `applyRecommendation` (line 1072), replace `league: "E0",` with `league,`.

Also update `useEffect`'s dependency array (line 1103) to include it:

```typescript
  }, [id, home, away, date, league]);
```

- [ ] **Step 3: Replace both `<main><DraftNav.../>` wrappers with `AppShell`**

Two places in `MatchAnalysisPage`: the early-return "Missing match details" branch (lines 1105-1117) and the main render (lines 1119-1226).

Early-return branch:

```typescript
  if (!home || !away || !date) {
    return (
      <AppShell active="matches">
        <p className="text-sm text-ink-secondary">
          Missing match details.{" "}
          <Link href="/matches" className="text-accent">
            Back to Match Explorer
          </Link>
        </p>
      </AppShell>
    );
  }
```

Main render — replace the opening (lines 1119-1121):

```typescript
  return (
    <AppShell active="matches">
```

and the closing `</main>` (last line of the function, previously line 1226) with `</AppShell>`. Everything in between (the back link, header, Model Probabilities/Squad Intelligence/Agent Reasoning sections) is unchanged — this is a wrapper swap plus the `league` threading from Steps 1-2, nothing else.

Also update the header's hardcoded `<span>E0</span>` (line 1133) to show the real league:

```typescript
            <span>{league}</span>
```

- [ ] **Step 4: Update `app/matches/[id]/page.tsx` to read and pass `league`**

`app/frontend/app/matches/[id]/page.tsx`:

```typescript
import { MatchAnalysisPage } from "@/components/MatchUI";

export default function Page({
  params,
  searchParams,
}: {
  params: { id: string };
  searchParams: { home?: string; away?: string; date?: string; league?: string };
}) {
  return (
    <MatchAnalysisPage
      id={params.id}
      home={searchParams.home ?? ""}
      away={searchParams.away ?? ""}
      date={searchParams.date ?? ""}
      league={searchParams.league ?? "E0"}
    />
  );
}
```

- [ ] **Step 5: Add a regression test for the league threading**

Add to `app/frontend/components/__tests__/MatchUI.test.tsx`, inside `describe("MatchAnalysisPage -- cache-first load (W47)", ...)`:

```typescript
  it("W64: requests a recommendation with the passed league, not a hardcoded E0", async () => {
    vi.mocked(getCachedRecommendation).mockResolvedValue(null);
    vi.mocked(generateRecommendation).mockResolvedValue(makeRecommendation());

    render(<MatchAnalysisPage id="m1" home="Malmo FF" away="AIK" date="2026-08-22" league="SWE" />);

    await waitFor(() => expect(generateRecommendation).toHaveBeenCalled());
    expect(generateRecommendation).toHaveBeenCalledWith(expect.objectContaining({ league: "SWE" }));
  });
```

- [ ] **Step 6: Run the tests**

Run: `cd /Users/tianqihuang/Documents/GitHub/FPAI/app/frontend && npx vitest run components/__tests__/MatchUI.test.tsx`
Expected: all pass, including the new one.

- [ ] **Step 7: Full suite + typecheck**

Run: `cd /Users/tianqihuang/Documents/GitHub/FPAI/app/frontend && npm test && npx tsc --noEmit`

- [ ] **Step 8: Commit**

```bash
git add app/frontend/components/MatchUI.tsx app/frontend/app/matches/[id]/page.tsx app/frontend/components/__tests__/MatchUI.test.tsx
git commit -m "feat(app): wire AppShell into MatchAnalysisPage; thread real league through instead of hardcoded E0"
```

---

### Task 10: Wire `AppShell` into `BetTrackerPage`

**Files:**
- Modify: `app/frontend/components/BetTracker.tsx:1-14, 289-333`

- [ ] **Step 1: Swap the import**

In `app/frontend/components/BetTracker.tsx`, change line 14:

```typescript
import { AppShell } from "./AppShell";
import { ErrorState, TeamBadge } from "./MatchUI";
```

(splitting the previous single `import { DraftNav, ErrorState, TeamBadge } from "./MatchUI";` into two lines — `DraftNav` is dropped, `ErrorState`/`TeamBadge` stay.)

- [ ] **Step 2: Replace the wrapper in `BetTrackerPage`**

Replace lines 290-292:

```typescript
    <AppShell active="bets">
```

And the final `</main>` (line 333) with `</AppShell>`. Everything in between (`<h1>`, `StatsBar`, `ManualBetForm`, the logged-bets list) is unchanged.

- [ ] **Step 3: Run the BetTracker tests — this is the critical compatibility check**

Run: `cd /Users/tianqihuang/Documents/GitHub/FPAI/app/frontend && npx vitest run components/__tests__/BetTracker.fixtureError.test.tsx components/__tests__/BetTracker.race.test.tsx`

Expected: all pass unchanged. This is the exact scenario Task 6's lazy-fetch design and distinct placeholder exist for — if `getFixtures.toHaveBeenCalledTimes(1)`/`(2)` fails here, `AppShell`'s search is fetching eagerly (re-check `ensureSearchFixturesLoaded` is only called from `onFocus`, never from a `useEffect`); if `getByPlaceholderText("Search a real fixture by team name…")` throws a multiple-elements error, `AppShell`'s placeholder collides (re-check it's exactly `"Search fixtures, teams…"`, not the same string).

- [ ] **Step 4: Full frontend suite + typecheck**

Run: `cd /Users/tianqihuang/Documents/GitHub/FPAI/app/frontend && npm test && npx tsc --noEmit`

- [ ] **Step 5: Commit**

```bash
git add app/frontend/components/BetTracker.tsx
git commit -m "feat(app): wire AppShell into BetTrackerPage"
```

---

### Task 11: Retire `StatusFooter` and the dead `DraftNav` export

`AppShell`'s sidebar footer (Task 6) now covers everything `StatusFooter` showed (data freshness + model counts). `DraftNav` has no remaining callers after Tasks 7-10.

**Files:**
- Delete: `app/frontend/components/StatusFooter.tsx`
- Delete: `app/frontend/components/__tests__/StatusFooter.test.tsx`
- Modify: `app/frontend/app/layout.tsx`
- Modify: `app/frontend/components/MatchUI.tsx:434-468` (remove `DraftNav`)

- [ ] **Step 1: Confirm no remaining references**

Run: `cd /Users/tianqihuang/Documents/GitHub/FPAI/app/frontend && grep -rn "StatusFooter\|DraftNav" app components --include="*.tsx" --include="*.ts"`

Expected output: only `components/MatchUI.tsx`'s own `DraftNav` definition and `components/StatusFooter.tsx`'s own definition — no callers. (If Tasks 7-10 were all completed, this should be the case; if something still references either, finish wiring it into `AppShell` first rather than deleting out from under it.)

- [ ] **Step 2: Remove `<StatusFooter />` from the root layout**

In `app/frontend/app/layout.tsx`:

```typescript
import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "FPAI",
  description: "FPAI betting agent web app.",
};

export default function RootLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return (
    <html lang="en">
      <body>{children}</body>
    </html>
  );
}
```

- [ ] **Step 3: Delete the files**

```bash
rm app/frontend/components/StatusFooter.tsx app/frontend/components/__tests__/StatusFooter.test.tsx
```

- [ ] **Step 4: Remove the `DraftNav` function from `MatchUI.tsx`**

Delete lines 434-468 of `app/frontend/components/MatchUI.tsx` (the entire `export function DraftNav({ active }: ...) { ... }` block).

- [ ] **Step 5: Full suite + typecheck**

Run: `cd /Users/tianqihuang/Documents/GitHub/FPAI/app/frontend && npm test && npx tsc --noEmit`
Expected: all pass, no unresolved-import errors.

- [ ] **Step 6: Commit**

```bash
git add -A app/frontend
git commit -m "chore(app): retire StatusFooter and DraftNav, superseded by AppShell"
```

---

### Task 12: Live verification in a real browser

Per this repo's established practice (every prior W-story that touched the frontend was verified live, not just via automated tests), and per this session's own working rules for UI changes.

- [ ] **Step 1: Start both dev servers**

Run (background):
```bash
cd /Users/tianqihuang/Documents/GitHub/FPAI/app/backend && uvicorn main:app --reload --port 8000
```
```bash
cd /Users/tianqihuang/Documents/GitHub/FPAI/app/frontend && npm run dev
```

- [ ] **Step 2: Drive it with Playwright (already a project dependency)**

Check whether sandbox mode is available for a deterministic session (per `documents/sandbox_testing_runbook.md`) — prefer it over depending on real off-season fixture availability. Confirm the following, screenshotting each:
- Dashboard: sidebar renders with Dashboard/All Matches/Bets, footer stats show real (or honest placeholder `—`) values, top-bar search returns real results and navigates correctly with `league=` in the URL, league-grouped sections show real section labels when more than one competition is present in the loaded window, sort control toggles order, Edge Distribution donut and Top Edges render (or show their empty states) without console errors.
- All Matches: same shell, existing in-page filter still works.
- Match Analysis: reachable both via a card's "Full analysis" link and via top-bar search; shows the correct league.
- Bets: shell renders, manual bet logging still works end-to-end (search a real fixture, log a bet), stats bar renders.
- Zero console errors on every page.

- [ ] **Step 3: Stop the dev servers**

- [ ] **Step 4: Note results** — carry findings into Task 13's story completion notes (this repo's established convention — see any completed `W##` entry in `documents/app_user_stories.md` for the expected level of detail).

---

### Task 13: Document the new stories and mark them complete

**Files:**
- Modify: `documents/app_user_stories.md`

- [ ] **Step 1: Append the new stories**

Add a new row block to the `## Stories` table in `documents/app_user_stories.md`, continuing from `W63`. Each row's `Comments` column gets real completion notes once Tasks 1-12 are actually done and verified — write them the way every other completed row in this table does (what was built, what was found along the way, what was verified live, full-suite pass counts). At minimum, the story set should be:

- **W64** — Backend: tag `/api/fixtures` matches with their source competition (Task 1). Note this was found mid-implementation, not planned upfront, and that it also fixed a real pre-existing bug (Task 3 — recommendations for Sweden fixtures were silently requested with `league: "E0"`).
- **W65** — `AppShell`: sidebar nav, footer stats (Active Edges / Model Status / Last Updated), lazy top-bar search — replaces `DraftNav` and `StatusFooter` (Tasks 6, 11).
- **W66** — Dashboard: league-grouped sections + sort control (Task 4, 7).
- **W67** — Dashboard: `DashboardRail` — Edge Distribution + Top Edges, reusing the existing status palette (Task 5, 7).
- **W68** — Match Analysis: real `league` threaded end-to-end instead of a hardcoded `"E0"` (Task 9).
- **W69** — All Matches / Bets: wired into `AppShell` (Tasks 8, 10).

Follow the existing table's exact format (`| ID | Status | Description | Comments |`) and the existing rows' level of acceptance-criteria detail — read 3-4 nearby completed rows (e.g. W61-W63) immediately before writing these, so tone/detail match.

- [ ] **Step 2: Mark each `completed`**, with real completion notes reflecting Tasks 1-12's actual outcomes (test counts, what Task 12's live check found) — per `CLAUDE.md`: "When tasks are complete, mark the user story as completed."

- [ ] **Step 3: Commit**

```bash
git add documents/app_user_stories.md
git commit -m "docs(app): add and complete W64-W69 (sidebar-shell frontend redesign)"
```

---

## Summary of files touched

**Backend:** `app/backend/football_data_client.py`, `app/backend/main.py`, `app/backend/tests/test_fixtures_endpoint.py`

**Frontend — new:** `components/AppShell.tsx`, `components/DashboardRail.tsx`, `lib/dashboardMetrics.ts`, plus their test files

**Frontend — modified:** `lib/types.ts`, `components/MatchUI.tsx`, `components/BetTracker.tsx`, `app/layout.tsx`, `app/matches/[id]/page.tsx`, `components/__tests__/MatchUI.test.tsx`

**Frontend — deleted:** `components/StatusFooter.tsx`, `components/__tests__/StatusFooter.test.tsx`

**Docs:** `documents/app_user_stories.md`
