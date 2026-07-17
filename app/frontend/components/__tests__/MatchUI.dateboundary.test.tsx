import { describe, expect, it, vi, beforeEach } from "vitest";
import { render, screen, waitFor } from "@testing-library/react";
import { DashboardPage, MatchExplorerPage, MatchCard, type Match } from "../MatchUI";
import { getFixtures, getSandboxStatus } from "@/lib/api";

vi.mock("@/lib/api");

describe("date-boundary correctness via the sandbox clock (W38)", () => {
  beforeEach(() => {
    // Reset call history (not just resolved values) between tests. Without
    // this, getFixtures/getSandboxStatus.mock.calls accumulate across the
    // `it` blocks below (vi.mock("@/lib/api") without a factory shares one
    // mock instance for the whole file), which can mask a genuinely failing
    // assertion behind a stale matching call left over from an earlier test.
    vi.mocked(getFixtures).mockReset();
    vi.mocked(getSandboxStatus).mockReset();
    vi.mocked(getFixtures).mockResolvedValue([]);
  });

  it("Dashboard queries fixtures for the sandbox as_of date, not the real browser date", async () => {
    vi.mocked(getSandboxStatus).mockResolvedValue({ sandbox_mode: true, as_of: "2026-03-01" });

    render(<DashboardPage />);

    await waitFor(() => expect(getFixtures).toHaveBeenCalledWith("2026-03-01", "2026-03-01"));
  });

  it("Dashboard's fixture query shifts to the next simulated day at midnight", async () => {
    // useSandboxAsOf() fetches once per mount (empty effect deps, W30) --
    // a same-instance rerender() never re-fires it, so unmount+remount (a
    // page reload) is how a new simulated day is actually observed.
    vi.mocked(getSandboxStatus).mockResolvedValueOnce({ sandbox_mode: true, as_of: "2026-03-01" });
    const { unmount } = render(<DashboardPage />);
    await waitFor(() => expect(getFixtures).toHaveBeenCalledWith("2026-03-01", "2026-03-01"));
    unmount();

    vi.mocked(getSandboxStatus).mockResolvedValueOnce({ sandbox_mode: true, as_of: "2026-03-02" });
    render(<DashboardPage />);

    await waitFor(() => expect(getFixtures).toHaveBeenCalledWith("2026-03-02", "2026-03-02"));
  });

  it("Match Explorer's 90-day window is anchored to the sandbox as_of date, not the real browser date", async () => {
    vi.mocked(getSandboxStatus).mockResolvedValue({ sandbox_mode: true, as_of: "2026-03-01" });

    render(<MatchExplorerPage />);

    // 2026-03-01 + 90 days = 2026-05-30
    await waitFor(() => expect(getFixtures).toHaveBeenCalledWith("2026-03-01", "2026-05-30"));
  });
});

describe("MatchCard's relative-day label respects the sandbox clock", () => {
  function matchWithKickoff(kickoffIso: string): Match {
    return {
      id: "m1", league: "E0", tier: "competition_specific", kickoffIso,
      home: "Arsenal", away: "Everton", status: "upcoming",
      hasRecommendation: true, overall: "no_bet", confidence: "medium",
      markets: [], explanation: "", limitations: [],
      predictionBasis: "team_history_and_market", coldStartRisk: false,
      featureCompleteness: 0.9, unknownTeam: false, invalidMarketCount: 0,
    };
  }

  // Kickoffs are deliberately set at UTC noon, not an arbitrary hour: the
  // fixture's own calendar day (dOnly, pre-existing/unchanged/out of scope
  // for this fix) is computed via *local* Date getters, same as it always
  // was for real-clock display -- so it legitimately rolls to the next
  // local day for a late-UTC kickoff in far-positive-offset timezones. UTC
  // noon keeps the same local calendar day across every timezone this
  // project actually verifies against (UTC/Asia-Tokyo/America-Los_Angeles);
  // it is not testing that pre-existing local-rollover behavior, only that
  // asOf (not the real browser clock) now drives the comparison.

  it("labels a fixture kicking off on the sandbox as_of date as 'today', not a real-clock-relative day", () => {
    // Real wall-clock "now" is whatever the test runs at (in CI, months away
    // from 2026-03-01) -- without asOf/sandboxMode wired through, this
    // fixture would show something like "138 days ago" instead of "today"
    // (the bug this test guards against, found in the final whole-branch
    // review). sandboxMode={true} matches how Dashboard/Match Explorer
    // actually pass it -- omitting it here would (correctly) fall back to
    // local getters, the real-clock behavior this test isn't exercising.
    const match = matchWithKickoff("2026-03-01T12:00:00Z");

    render(
      <MatchCard match={match} onUpdate={vi.fn()} asOf={new Date("2026-03-01T00:00:00Z")} sandboxMode={true} />
    );

    expect(screen.getByText("today")).toBeInTheDocument();
  });

  it("labels a fixture kicking off the day after the sandbox as_of date as 'tomorrow'", () => {
    const match = matchWithKickoff("2026-03-02T12:00:00Z");

    render(
      <MatchCard match={match} onUpdate={vi.fn()} asOf={new Date("2026-03-01T00:00:00Z")} sandboxMode={true} />
    );

    expect(screen.getByText("tomorrow")).toBeInTheDocument();
  });

  it("uses the real local clock (not UTC getters) when sandbox mode is off, even with an explicit asOf", () => {
    // Regression guard for the bug the prior fix introduced: reading a real
    // new Date() via UTC getters mislabels "today" as "yesterday"/"tomorrow"
    // for roughly half of every day in any non-UTC timezone. A kickoff at
    // the same real instant as asOf must always read as "today" when
    // sandboxMode is false (the default), regardless of the local timezone
    // the test runs under.
    const now = new Date();
    const match = matchWithKickoff(now.toISOString());

    render(<MatchCard match={match} onUpdate={vi.fn()} asOf={now} />);

    expect(screen.getByText("today")).toBeInTheDocument();
  });
});
