/**
 * Dashboard always shows the next 10 matches going forward from asOf, in a
 * single 90-day-forward window query -- regardless of whether asOf's own
 * date has any fixtures. There is no separate "today empty -> fall back to
 * next matches" two-fetch path anymore (that was W46's original behavior,
 * superseded by this always-widened query); matches are grouped into date
 * rows (MatchUI.dateGroups.test.tsx-adjacent behavior lives in
 * dashboardMetrics.test.ts's groupByDate tests), with asOf's own date
 * labeled "Today".
 */
import { describe, expect, it, vi, beforeEach } from "vitest";
import { render, screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { DashboardPage } from "../MatchUI";
import { getCachedRecommendation, getFixtures, getSandboxStatus } from "@/lib/api";
import type { Fixture, MatchRecommendationOut } from "@/lib/types";

vi.mock("@/lib/api");

function fixture(id: string, utcDate: string): Fixture {
  return {
    match_id: id,
    utc_date: utcDate,
    status: "SCHEDULED",
    home_team: id,
    away_team: "Away",
    home_goals: null,
    away_goals: null,
  };
}

describe("Dashboard always shows the next 10 matches (date-grouped, not today-only)", () => {
  beforeEach(() => {
    vi.mocked(getFixtures).mockReset();
    vi.mocked(getSandboxStatus).mockReset();
    // Keep useSandboxAsOf's asOf pinned to the real-clock Date it starts
    // with (sandbox_mode: false means the hook's setState call is skipped
    // entirely) so these tests exercise a single, non-racing effect run.
    vi.mocked(getSandboxStatus).mockResolvedValue({ sandbox_mode: false, as_of: "" });
  });

  it("queries a single 90-day-forward window (not a same-day-only query)", async () => {
    const today = new Date().toISOString().slice(0, 10);
    vi.mocked(getFixtures).mockResolvedValue([]);

    render(<DashboardPage />);

    await waitFor(() => expect(getFixtures).toHaveBeenCalledTimes(1));
    const [from, to] = vi.mocked(getFixtures).mock.calls[0];
    expect(from).toBe(today);
    expect(to).toMatch(/^\d{4}-\d{2}-\d{2}$/);
    expect(to).not.toBe(today);
  });

  it("shows up to 10 matches sorted nearest-first, even when today itself has none", async () => {
    const today = new Date().toISOString().slice(0, 10);

    // 12 fixtures, deliberately out of order and on many different days, so
    // this must both trim to 10 and actually sort by kickoff.
    const unordered = Array.from({ length: 12 }, (_, i) => i).reverse();
    const fixtures = unordered.map((i) => fixture(`match-${i}`, `2026-09-${String(i + 1).padStart(2, "0")}T15:00:00Z`));
    vi.mocked(getFixtures).mockResolvedValue(fixtures);

    render(<DashboardPage />);

    await waitFor(() => expect(screen.getByText("match-0")).toBeInTheDocument());
    expect(screen.getByText("match-9")).toBeInTheDocument();
    expect(screen.queryByText("match-10")).not.toBeInTheDocument();
    expect(screen.queryByText("match-11")).not.toBeInTheDocument();
  });

  it("labels asOf's own date 'Today' and shows a real calendar date for every other date row", async () => {
    // Non-sandbox mode -- dayDiff compares via *local* Date getters (see its
    // own comment in MatchUI.tsx), not UTC. A fixture built from
    // `now.toISOString().slice(0, 10)` (always a UTC date) can land on the
    // wrong side of local midnight whenever local time and UTC disagree on
    // the calendar day -- true for several hours of every real day in any
    // non-UTC-zero timezone, not just a rare edge case. Using the exact same
    // instant (and +24h for "tomorrow") sidesteps the UTC/local distinction
    // entirely: a fixture at the identical wall-clock moment as `asOf` is
    // always "today" in whichever getters compare them.
    const now = new Date();
    const tomorrow = new Date(now.getTime() + 24 * 60 * 60 * 1000);

    vi.mocked(getFixtures).mockResolvedValue([
      fixture("today-match", now.toISOString()),
      fixture("tomorrow-match", tomorrow.toISOString()),
    ]);

    render(<DashboardPage />);

    expect(await screen.findByText("Today")).toBeInTheDocument();
    expect(screen.getByText("today-match")).toBeInTheDocument();
    expect(screen.getByText("tomorrow-match")).toBeInTheDocument();
    // The non-today row is a real calendar date, not another relative word.
    expect(screen.queryByText("Tomorrow")).not.toBeInTheDocument();
  });

  it("excludes already-decided (FINISHED) matches from the shown list -- the Dashboard is pre-match only", async () => {
    // A same-day FINISHED fixture renders "completed" (with a real score)
    // from fixtureToMatch's own W48 same-day carve-out, in both live and
    // sandbox mode -- correct for MatchExplorerPage/BetTracker, wrong for
    // the Dashboard, which must never show a match's real outcome on a
    // recommendation card.
    const today = new Date().toISOString().slice(0, 10);
    const decided: Fixture = {
      match_id: "decided-match",
      utc_date: `${today}T12:00:00Z`,
      status: "FINISHED",
      home_team: "Decided",
      away_team: "Away",
      home_goals: 2,
      away_goals: 1,
    };
    vi.mocked(getFixtures).mockResolvedValue([decided, fixture("upcoming-match", `${today}T18:00:00Z`)]);

    render(<DashboardPage />);

    expect(await screen.findByText("upcoming-match")).toBeInTheDocument();
    expect(screen.queryByText("Decided")).not.toBeInTheDocument();
    expect(screen.queryByText("2-1")).not.toBeInTheDocument();
  });

  it("renders a sensible, non-crashing empty state when the whole 90-day window is empty", async () => {
    vi.mocked(getFixtures).mockResolvedValue([]);

    render(<DashboardPage />);

    expect(await screen.findByText("No upcoming fixtures.")).toBeInTheDocument();
  });

  it("degrades to a top-level error state when the fetch itself fails", async () => {
    vi.mocked(getFixtures).mockRejectedValue(new Error("network blip"));

    render(<DashboardPage />);

    expect(await screen.findByText(/could not load/i)).toBeInTheDocument();
  });

  it("W108: 'Actionable only' hides non-actionable matches, and unchecking restores them", async () => {
    vi.mocked(getFixtures).mockResolvedValue([
      fixture("actionable-match", "2026-09-01T15:00:00Z"),
      fixture("not-yet-generated-match", "2026-09-01T18:00:00Z"),
    ]);
    const rec: MatchRecommendationOut = {
      match: {}, overall: "direct_bet", markets: [], explanation: [], confidence: "high",
      limitations: [], prediction_basis: "team_history_and_market", invalid_market_count: 0,
      cold_start_risk: false, feature_completeness: 0.9, unknown_team: false,
    };
    vi.mocked(getCachedRecommendation).mockImplementation(async (matchId: string) =>
      matchId === "actionable-match" ? rec : null
    );

    const user = userEvent.setup();
    render(<DashboardPage />);

    await screen.findByText("actionable-match");
    expect(screen.getByText("not-yet-generated-match")).toBeInTheDocument();

    await user.click(screen.getByText("Actionable only"));
    expect(screen.getByText("actionable-match")).toBeInTheDocument();
    expect(screen.queryByText("not-yet-generated-match")).not.toBeInTheDocument();

    await user.click(screen.getByText("Actionable only"));
    expect(screen.getByText("not-yet-generated-match")).toBeInTheDocument();
  });
});
