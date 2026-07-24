/**
 * W52: ManualBetForm's fixture-search fetch previously did
 * `.catch(() => setFixtures([]))` -- any fetch failure (network error, the
 * football-data.org 429 rate-limit -> backend 503, or anything else)
 * silently degraded to an empty fixture list with no error shown at all.
 * A user searching for a real fixture to log a bet against would see
 * nothing and have no idea whether that's a genuine "no matches" or a
 * broken fetch. This mirrors MatchExplorerPage's established pattern
 * (ApiError message vs a generic fallback, rendered visibly) -- see
 * MatchUI.tsx's ErrorState usage.
 */
import { describe, expect, it, vi, beforeEach } from "vitest";
import { render, screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { BetTrackerPage } from "../BetTracker";

// W52: vi.mock("@/lib/api") without a factory (BetTracker.race.test.tsx's
// approach) automocks ApiError into a mock constructor that doesn't behave
// like a real Error subclass -- `err instanceof ApiError` then fails inside
// the component. MatchUI.test.tsx already established the fix (a real
// `class ApiError extends Error {}` in the factory) -- reused here since
// this test needs `err instanceof ApiError` to genuinely work.
vi.mock("@/lib/api", () => ({
  getBets: vi.fn(),
  getBetStats: vi.fn(),
  getFixtures: vi.fn(),
  logBetManual: vi.fn(),
  settleOpenBets: vi.fn(),
  getSandboxStatus: vi.fn(),
  // AppShell (wired into BetTrackerPage as of this task) calls getStatus()
  // on mount for its top-bar status indicator -- without this the mock
  // module has no such export and AppShell's mount throws. AppShell
  // degrades its status display to "--" whether this resolves or rejects,
  // so an unresolved vi.fn() is sufficient here (same fix as Tasks 7/9).
  getStatus: vi.fn(),
  ApiError: class ApiError extends Error {
    status?: number;
    constructor(message: string, status?: number) {
      super(message);
      this.name = "ApiError";
      this.status = status;
    }
  },
}));

import { ApiError, getBets, getBetStats, getFixtures, getSandboxStatus } from "@/lib/api";

describe("ManualBetForm surfaces a visible error when the fixture fetch fails (W52)", () => {
  beforeEach(() => {
    vi.mocked(getFixtures).mockReset();
    vi.mocked(getSandboxStatus).mockReset();
    vi.mocked(getBets).mockReset();
    vi.mocked(getBetStats).mockReset();
    vi.mocked(getBets).mockResolvedValue([]);
    vi.mocked(getBetStats).mockResolvedValue({
      bets_settled: 0, bets_open: 0, bets_won: 0, roi: 0, hit_rate: 0,
      total_staked: 0, total_profit: 0, max_drawdown: 0,
      starting_bankroll: 0, current_bankroll: 0,
    });
    vi.mocked(getSandboxStatus).mockResolvedValue({ sandbox_mode: false, as_of: null });
  });

  it("shows a visible error message when the fixture fetch rejects, not just a silent empty search", async () => {
    vi.mocked(getFixtures).mockRejectedValue(
      new ApiError("Fixture data is temporarily unavailable (the upstream provider is rate-limited or unreachable).", 503)
    );

    const user = userEvent.setup();
    render(<BetTrackerPage />);

    await waitFor(() => expect(getFixtures).toHaveBeenCalled());

    await user.type(
      screen.getByPlaceholderText("Search a real fixture by team name…"),
      "Arsenal"
    );

    await waitFor(() =>
      expect(
        screen.getByText(/Fixture data is temporarily unavailable/i)
      ).toBeInTheDocument()
    );
  });

  it("clicking Retry on the fixture error re-runs the fetch and clears the error once it succeeds (W52 code review follow-up)", async () => {
    vi.mocked(getFixtures)
      .mockRejectedValueOnce(new ApiError("Fixture data is temporarily unavailable.", 503))
      .mockResolvedValueOnce([]);

    const user = userEvent.setup();
    render(<BetTrackerPage />);

    await waitFor(() =>
      expect(screen.getByText(/Fixture data is temporarily unavailable/i)).toBeInTheDocument()
    );
    expect(getFixtures).toHaveBeenCalledTimes(1);

    await user.click(screen.getByRole("button", { name: /retry/i }));

    await waitFor(() => expect(getFixtures).toHaveBeenCalledTimes(2));
    await waitFor(() =>
      expect(screen.queryByText(/Fixture data is temporarily unavailable/i)).not.toBeInTheDocument()
    );
  });
});
