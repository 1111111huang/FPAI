/**
 * W53: the Dashboard/Match Explorer's initial fixture list must reflect the
 * precomputed recommendation cache (W50/W51) up front, not only after a user
 * clicks a card. fixtureToMatch() unconditionally set hasRecommendation:
 * false, and the only two call sites of getCachedRecommendation() were both
 * lazy (MatchCard.handleExpand on click, MatchAnalysisPage.load on
 * navigation) -- so even a fully-precomputed cache never visually manifested
 * until every card was clicked individually.
 *
 * This file proves: (1) a cache hit for a fixture in the initial list
 * renders its recommendation with zero clicks, (2) a cache miss is
 * unaffected -- still "Not yet generated" initially, and the existing W47
 * click-through lazy fallback (cache-check -> generateRecommendation) still
 * works end-to-end afterward, and (3) the bulk cache check fires
 * concurrently (all per-match calls start before any of them resolve), not
 * sequentially, matching the story's explicit "run concurrently ... given
 * the list is capped at 10" requirement.
 */
import { describe, expect, it, vi, beforeEach } from "vitest";
import { render, screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { DashboardPage, MatchExplorerPage } from "../MatchUI";
import { generateRecommendation, getCachedRecommendation, getFixtures, getSandboxStatus } from "@/lib/api";
import type { Fixture, MatchRecommendationOut } from "@/lib/types";

vi.mock("@/lib/api");

function fixture(id: string, utcDate: string, home = id, away = "Away"): Fixture {
  return {
    match_id: id,
    utc_date: utcDate,
    status: "SCHEDULED",
    home_team: home,
    away_team: away,
    home_goals: null,
    away_goals: null,
  };
}

function makeRecommendation(overrides: Partial<MatchRecommendationOut> = {}): MatchRecommendationOut {
  return {
    match: { home: "Arsenal", away: "Everton", date: "2026-08-22", league: "E0" },
    overall: "direct_bet",
    markets: [],
    explanation: "test explanation",
    confidence: "medium",
    limitations: [],
    prediction_basis: "team_history_and_market",
    invalid_market_count: 0,
    cold_start_risk: false,
    feature_completeness: 0.9,
    unknown_team: false,
    ...overrides,
  };
}

describe("Dashboard initial-list precompute visibility (W53)", () => {
  beforeEach(() => {
    vi.mocked(getFixtures).mockReset();
    vi.mocked(getCachedRecommendation).mockReset();
    vi.mocked(generateRecommendation).mockReset();
    vi.mocked(getSandboxStatus).mockReset();
    // Pin asOf to the real-clock Date the hook starts with (sandbox_mode:
    // false skips its setState entirely) so these tests exercise a single
    // non-racing effect run, same convention as MatchUI.emptyFallback.test.tsx.
    vi.mocked(getSandboxStatus).mockResolvedValue({ sandbox_mode: false, as_of: "" });
  });

  it("a precomputed (cache-hit) fixture in the initial list renders its recommendation with no click", async () => {
    const today = new Date().toISOString().slice(0, 10);
    const f = fixture("precomputed-match", `${today}T15:00:00Z`, "Arsenal", "Everton");
    vi.mocked(getFixtures).mockImplementation(async (from, to) => (from === today && to === today ? [f] : []));
    const cachedRec = makeRecommendation({ explanation: "precomputed explanation" });
    vi.mocked(getCachedRecommendation).mockResolvedValue(cachedRec);

    render(<DashboardPage />);

    // Proven purely from the initial fetch -- no userEvent.click anywhere in
    // this test.
    expect(await screen.findByText("Direct Bet")).toBeInTheDocument();
    expect(screen.queryByText("Not yet generated")).not.toBeInTheDocument();
    expect(getCachedRecommendation).toHaveBeenCalledWith("precomputed-match", today);
    expect(generateRecommendation).not.toHaveBeenCalled();
  });

  it("a cache-miss fixture in the initial list still renders 'Not yet generated', and clicking it still triggers the existing generateRecommendation fallback", async () => {
    const today = new Date().toISOString().slice(0, 10);
    const f = fixture("uncached-match", `${today}T15:00:00Z`, "Arsenal", "Everton");
    vi.mocked(getFixtures).mockImplementation(async (from, to) => (from === today && to === today ? [f] : []));
    vi.mocked(getCachedRecommendation).mockResolvedValue(null);
    const liveRec = makeRecommendation({ explanation: "live explanation" });
    vi.mocked(generateRecommendation).mockResolvedValue(liveRec);

    render(<DashboardPage />);

    // Unaffected initial render: still shows the miss state, not a click-free
    // recommendation.
    expect(await screen.findByText("Not yet generated")).toBeInTheDocument();

    // The bulk cache check must still have run (and found a miss) during the
    // initial load -- exactly one call before any click.
    await waitFor(() => expect(getCachedRecommendation).toHaveBeenCalledWith("uncached-match", today));
    await waitFor(() => expect(getCachedRecommendation).toHaveBeenCalledTimes(1));

    // The existing W47 lazy fallback must still work end-to-end after the
    // bulk check finds a miss -- its own (separate) cache-check call brings
    // the total to 2.
    const user = userEvent.setup();
    await user.click(screen.getByText("Not yet generated"));

    await waitFor(() => expect(getCachedRecommendation).toHaveBeenCalledTimes(2));
    await waitFor(() => expect(generateRecommendation).toHaveBeenCalledWith({
      home_team: "Arsenal",
      away_team: "Everton",
      date: today,
      league: "E0",
      match_id: "uncached-match",
    }));
    expect(await screen.findByText("Direct Bet")).toBeInTheDocument();
  });

  it("resolves the cache check concurrently across the whole initial list, not one match at a time", async () => {
    const today = new Date().toISOString().slice(0, 10);
    const fixtures = [
      fixture("match-a", `${today}T15:00:00Z`, "Arsenal", "Everton"),
      fixture("match-b", `${today}T17:00:00Z`, "Chelsea", "Fulham"),
    ];
    vi.mocked(getFixtures).mockImplementation(async (from, to) => (from === today && to === today ? fixtures : []));

    const resolvers: Array<() => void> = [];
    vi.mocked(getCachedRecommendation).mockImplementation(
      () =>
        new Promise((resolve) => {
          resolvers.push(() => resolve(null));
        })
    );

    render(<DashboardPage />);

    // Both calls must have been *started* before either has resolved --
    // proof this is Promise.all-style concurrent dispatch, not a sequential
    // await-one-then-the-next loop (which would only ever have 1 pending
    // call at this point).
    await waitFor(() => expect(getCachedRecommendation).toHaveBeenCalledTimes(2));
    expect(resolvers).toHaveLength(2);

    resolvers.forEach((r) => r());
    await waitFor(() => expect(screen.getAllByText("Not yet generated")).toHaveLength(2));
  });
});

// ---------------------------------------------------------------------------
// W53 follow-up (code review): unlike Dashboard's two call sites (each
// capped at 10), MatchExplorerPage's 90-day search window can realistically
// return 50-100+ fixtures in-season. Blocking first paint on every one of
// those fixtures' cache checks resolving would queue behind the browser's
// per-origin connection cap, making this specific page *slower* to first
// paint than before this story. MatchExplorerPage must render its fixture
// list immediately (unblocked, exactly like pre-W53 behavior) and then patch
// precomputed results in via a follow-up state update once the bulk check
// resolves in the background -- not gate the first render on it.
// ---------------------------------------------------------------------------

describe("Match Explorer initial render is not blocked by the bulk cache check (W53 follow-up)", () => {
  beforeEach(() => {
    vi.mocked(getFixtures).mockReset();
    vi.mocked(getCachedRecommendation).mockReset();
    vi.mocked(generateRecommendation).mockReset();
    vi.mocked(getSandboxStatus).mockReset();
    vi.mocked(getSandboxStatus).mockResolvedValue({ sandbox_mode: false, as_of: "" });
  });

  it("renders the fixture list immediately (before any cache-check promise resolves), then patches a precomputed recommendation in once it resolves", async () => {
    const today = new Date().toISOString().slice(0, 10);
    const f = fixture("explorer-match", `${today}T15:00:00Z`, "Arsenal", "Everton");
    vi.mocked(getFixtures).mockResolvedValue([f]);

    // A cache-check promise under this test's own control -- never resolved
    // until asserted otherwise, so the initial render cannot be depending on
    // it having settled.
    let resolveCache!: (rec: MatchRecommendationOut | null) => void;
    vi.mocked(getCachedRecommendation).mockImplementation(
      () =>
        new Promise((resolve) => {
          resolveCache = resolve;
        })
    );

    render(<MatchExplorerPage />);

    // The fixture must appear -- as "Not yet generated" -- while its
    // cache-check promise is still outstanding. If the fix regresses back to
    // awaiting the bulk check before the first setMatches, this would still
    // be showing LoadingRows here and neither assertion below would find
    // anything (a timeout, not a false pass).
    expect(await screen.findByText("Not yet generated")).toBeInTheDocument();
    expect(getCachedRecommendation).toHaveBeenCalledWith("explorer-match", today);

    // Now resolve the cache hit -- the recommendation must be patched into
    // the already-rendered list, not require a click.
    const cachedRec = makeRecommendation({ explanation: "patched in after first paint" });
    resolveCache(cachedRec);

    expect(await screen.findByText("Direct Bet")).toBeInTheDocument();
    expect(screen.queryByText("Not yet generated")).not.toBeInTheDocument();
    expect(generateRecommendation).not.toHaveBeenCalled();
  });
});
