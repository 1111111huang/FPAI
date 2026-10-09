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
import { render, screen, waitFor, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { DashboardPage, MatchExplorerPage, dateString, __resetDashboardMatchesCacheForTests } from "../MatchUI";
import {
  generateRecommendation,
  getCachedRecommendation,
  getCachedRecommendationsBulk,
  getFixtures,
  getSandboxStatus,
} from "@/lib/api";
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
    candidates: [],
    recommendation_pick: null,
    explanation: ["test explanation"],
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
    __resetDashboardMatchesCacheForTests();
    vi.mocked(getFixtures).mockReset();
    vi.mocked(getCachedRecommendation).mockReset();
    vi.mocked(getCachedRecommendationsBulk).mockReset();
    vi.mocked(generateRecommendation).mockReset();
    vi.mocked(getSandboxStatus).mockReset();
    // Pin asOf to the real-clock Date the hook starts with (sandbox_mode:
    // false skips its setState entirely) so these tests exercise a single
    // non-racing effect run, same convention as MatchUI.emptyFallback.test.tsx.
    vi.mocked(getSandboxStatus).mockResolvedValue({ sandbox_mode: false, as_of: "" });
  });

  it("a precomputed (cache-hit) fixture in the initial list renders its recommendation with no click", async () => {
    // W40: America/New_York's calendar day (dateString, the canonical
    // helper MatchUI.tsx's own components call), not the test runner's own
    // ambient timezone.
    const now = new Date();
    const today = dateString(now, false);
    const f = fixture("precomputed-match", `${today}T15:00:00Z`, "Arsenal", "Everton");
    vi.mocked(getFixtures).mockImplementation(async (from, to) => (from === today ? [f] : []));
    const cachedRec = makeRecommendation({ explanation: ["precomputed explanation"] });
    vi.mocked(getCachedRecommendationsBulk).mockResolvedValue({ "precomputed-match": cachedRec });

    render(<DashboardPage />);

    // Proven purely from the initial fetch -- no userEvent.click anywhere in
    // this test. Scoped to the card's own <button> (not just screen-wide)
    // since DashboardRail's legend independently renders "Direct Bet" too --
    // this keeps the assertion specific to the card's own status badge.
    const card = (await screen.findByText("Arsenal")).closest('[role="button"]') as HTMLElement | null;
    expect(card).not.toBeNull();
    expect(within(card!).getByText("Direct Bet")).toBeInTheDocument();
    expect(screen.queryByText("Not yet generated")).not.toBeInTheDocument();
    expect(getCachedRecommendationsBulk).toHaveBeenCalledWith([{ matchId: "precomputed-match", date: today }]);
    expect(generateRecommendation).not.toHaveBeenCalled();
  });

  it("a cache-miss fixture in the initial list still renders 'Not yet generated', and clicking it still triggers the existing generateRecommendation fallback", async () => {
    // W40: America/New_York's calendar day (dateString, the canonical
    // helper MatchUI.tsx's own components call), not the test runner's own
    // ambient timezone.
    const now = new Date();
    const today = dateString(now, false);
    const f = fixture("uncached-match", `${today}T15:00:00Z`, "Arsenal", "Everton");
    vi.mocked(getFixtures).mockImplementation(async (from, to) => (from === today ? [f] : []));
    vi.mocked(getCachedRecommendationsBulk).mockResolvedValue({ "uncached-match": null });
    const liveRec = makeRecommendation({ explanation: ["live explanation"] });
    vi.mocked(generateRecommendation).mockResolvedValue(liveRec);

    render(<DashboardPage />);

    // Unaffected initial render: still shows the miss state, not a click-free
    // recommendation.
    expect(await screen.findByText("Not yet generated")).toBeInTheDocument();

    // The bulk cache check must still have run (and found a miss) during the
    // initial load -- exactly one call before any click.
    await waitFor(() =>
      expect(getCachedRecommendationsBulk).toHaveBeenCalledWith([{ matchId: "uncached-match", date: today }])
    );
    await waitFor(() => expect(getCachedRecommendationsBulk).toHaveBeenCalledTimes(1));

    // The existing W47 lazy fallback must still work end-to-end after the
    // bulk check finds a miss -- a click goes through the card's own
    // (separate, singular) getCachedRecommendation cache-check call.
    const user = userEvent.setup();
    vi.mocked(getCachedRecommendation).mockResolvedValue(null);
    await user.click(screen.getByText("Not yet generated"));

    await waitFor(() => expect(getCachedRecommendation).toHaveBeenCalledWith("uncached-match", today));
    await waitFor(() => expect(generateRecommendation).toHaveBeenCalledWith({
      home_team: "Arsenal",
      away_team: "Everton",
      date: today,
      league: "E0",
      match_id: "uncached-match",
    }));
    // Scoped to the card's own <button>, not screen-wide -- see the previous
    // test's comment on why (DashboardRail's legend also renders this text).
    const card = (await screen.findByText("Arsenal")).closest('[role="button"]') as HTMLElement | null;
    expect(card).not.toBeNull();
    expect(within(card!).getByText("Direct Bet")).toBeInTheDocument();
  });

  it("resolves the whole initial list's cache check in a single bulk call, not one request per match", async () => {
    // W40: America/New_York's calendar day (dateString, the canonical
    // helper MatchUI.tsx's own components call), not the test runner's own
    // ambient timezone.
    const now = new Date();
    const today = dateString(now, false);
    const fixtures = [
      fixture("match-a", `${today}T15:00:00Z`, "Arsenal", "Everton"),
      fixture("match-b", `${today}T17:00:00Z`, "Chelsea", "Fulham"),
    ];
    vi.mocked(getFixtures).mockImplementation(async (from, to) => (from === today ? fixtures : []));

    let resolveBulk!: (value: Record<string, MatchRecommendationOut | null>) => void;
    vi.mocked(getCachedRecommendationsBulk).mockImplementation(
      () => new Promise((resolve) => { resolveBulk = resolve; })
    );

    render(<DashboardPage />);

    // One request covering both matches -- not two separate per-match
    // requests the way the pre-bulk-endpoint code made (direct user report
    // that switching pages was still slow with N round trips per list).
    await waitFor(() => expect(getCachedRecommendationsBulk).toHaveBeenCalledTimes(1));
    expect(getCachedRecommendationsBulk).toHaveBeenCalledWith([
      { matchId: "match-a", date: today },
      { matchId: "match-b", date: today },
    ]);

    resolveBulk({ "match-a": null, "match-b": null });
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
    __resetDashboardMatchesCacheForTests();
    vi.mocked(getFixtures).mockReset();
    vi.mocked(getCachedRecommendationsBulk).mockReset();
    vi.mocked(generateRecommendation).mockReset();
    vi.mocked(getSandboxStatus).mockReset();
    vi.mocked(getSandboxStatus).mockResolvedValue({ sandbox_mode: false, as_of: "" });
  });

  it("renders the fixture list immediately (before the bulk cache-check call resolves), then patches a precomputed recommendation in once it resolves", async () => {
    // W40: America/New_York's calendar day (dateString, the canonical
    // helper MatchUI.tsx's own components call), not the test runner's own
    // ambient timezone.
    const now = new Date();
    const today = dateString(now, false);
    const f = fixture("explorer-match", `${today}T15:00:00Z`, "Arsenal", "Everton");
    vi.mocked(getFixtures).mockResolvedValue([f]);

    // The bulk cache-check call, under this test's own control -- never
    // resolved until asserted otherwise, so the initial render cannot be
    // depending on it having settled.
    let resolveBulk!: (value: Record<string, MatchRecommendationOut | null>) => void;
    vi.mocked(getCachedRecommendationsBulk).mockImplementation(
      () => new Promise((resolve) => { resolveBulk = resolve; })
    );

    render(<MatchExplorerPage />);

    // The fixture must appear -- as "Not yet generated" -- while its
    // cache-check call is still outstanding. If the fix regresses back to
    // awaiting the bulk check before the first setMatches, this would still
    // be showing LoadingRows here and neither assertion below would find
    // anything (a timeout, not a false pass).
    expect(await screen.findByText("Not yet generated")).toBeInTheDocument();
    expect(getCachedRecommendationsBulk).toHaveBeenCalledWith([{ matchId: "explorer-match", date: today }]);

    // Now resolve the cache hit -- the recommendation must be patched into
    // the already-rendered list, not require a click.
    const cachedRec = makeRecommendation({ explanation: ["patched in after first paint"] });
    resolveBulk({ "explorer-match": cachedRec });

    expect(await screen.findByText("Direct Bet")).toBeInTheDocument();
    expect(screen.queryByText("Not yet generated")).not.toBeInTheDocument();
    expect(generateRecommendation).not.toHaveBeenCalled();
  });
});

// Direct user follow-up: does switching back and forth between Daily Edges
// and Match Explorer actually benefit from caching? DashboardPage already
// had its own page-level cache (W233); MatchExplorerPage didn't, so leaving
// and returning to it always blanked to the loading skeleton and re-ran the
// full fixtures + bulk-recommendations round trip, no matter how recently
// you'd just been there. MatchExplorerPage now shares DashboardPage's same
// cache map, keyed by its own fetch window.
describe("MatchExplorerPage reuses a cached match list across remounts (navigating back and forth)", () => {
  beforeEach(() => {
    __resetDashboardMatchesCacheForTests();
    vi.mocked(getFixtures).mockReset();
    vi.mocked(getCachedRecommendationsBulk).mockReset();
    vi.mocked(getSandboxStatus).mockReset();
    vi.mocked(getSandboxStatus).mockResolvedValue({ sandbox_mode: false, as_of: "" });
  });

  it("a remount within the cache's TTL renders from the cache -- no new getFixtures/bulk call, no loading flash", async () => {
    const now = new Date();
    const today = dateString(now, false);
    const f = fixture("cached-explorer-match", `${today}T15:00:00Z`, "Arsenal", "Everton");
    vi.mocked(getFixtures).mockResolvedValue([f]);
    const cachedRec = makeRecommendation({ explanation: ["cached across remounts"] });
    vi.mocked(getCachedRecommendationsBulk).mockResolvedValue({ "cached-explorer-match": cachedRec });

    const first = render(<MatchExplorerPage />);
    expect(await screen.findByText("Arsenal")).toBeInTheDocument();
    await waitFor(() => expect(getFixtures).toHaveBeenCalledTimes(1));
    await waitFor(() => expect(getCachedRecommendationsBulk).toHaveBeenCalledTimes(1));
    first.unmount();

    render(<MatchExplorerPage />);

    // Renders immediately from the cache -- already-resolved recommendation
    // included, no intermediate "Not yet generated" flash.
    expect(await screen.findByText("Arsenal")).toBeInTheDocument();
    expect(screen.getByText("Direct Bet")).toBeInTheDocument();
    expect(getFixtures).toHaveBeenCalledTimes(1);
    expect(getCachedRecommendationsBulk).toHaveBeenCalledTimes(1);
  });
});

describe("Daily Edges UB explainer line (W169)", () => {
  beforeEach(() => {
    __resetDashboardMatchesCacheForTests();
    vi.mocked(getFixtures).mockReset();
    vi.mocked(getCachedRecommendation).mockReset();
    vi.mocked(getSandboxStatus).mockReset();
    vi.mocked(getSandboxStatus).mockResolvedValue({ sandbox_mode: false, as_of: "" });
    vi.mocked(getFixtures).mockResolvedValue([]);
  });

  it("shows a static explanation of the UB (Unit Bet) convention under the header", async () => {
    render(<DashboardPage />);
    expect(
      await screen.findByText(/UB = Unit Bet, your standard bet amount — the money you'd put on a 50\/50 match bet\./)
    ).toBeInTheDocument();
  });
});
