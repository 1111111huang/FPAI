import { render, screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { describe, expect, it, vi, beforeEach } from "vitest";

import { AgentPerformancePage } from "../AgentPerformanceDashboard";
import { getAgentPerformanceDashboard, getStatus, getSandboxStatus } from "@/lib/api";
import type { AgentPerformanceDashboard as DashboardData, SegmentMetrics } from "@/lib/types";

vi.mock("@/lib/api", () => ({
  getAgentPerformanceDashboard: vi.fn(),
  getStatus: vi.fn(),
  getSandboxStatus: vi.fn(),
  ApiError: class ApiError extends Error {
    status?: number;
  },
}));

function segmentMetrics(overrides: Partial<SegmentMetrics> = {}): SegmentMetrics {
  return {
    matches_evaluated: 10, bets_placed: 5, bets_won: 2, roi: 0.1, hit_rate: 0.4,
    bet_frequency: 0.5, max_drawdown: 0.1, insufficient_data_rate: 0.0,
    starting_bankroll: 1000, ending_bankroll: 1100, total_staked: 50, total_profit: 5,
    ...overrides,
  };
}

function dashboardData(overrides: Partial<DashboardData> = {}): DashboardData {
  return {
    overall: { sample_size: 10, correct: 5, hit_rate: 0.5 },
    by_market: {},
    by_competition: {},
    by_confidence: {},
    // Deliberately distinct bets_placed from the segment tables below (each
    // default to 5 via segmentMetrics()'s own default) -- keeps
    // getByText("7") in the KPI-row test unambiguous instead of colliding
    // with every segment table's own "5" in its Bets column.
    kelly_roi_simulation: segmentMetrics({ bets_placed: 7 }),
    by_market_metrics: { result_3way: segmentMetrics() },
    by_market_selection_metrics: { "result_3way:home": segmentMetrics() },
    by_league_metrics: { E0: segmentMetrics({ bets_placed: 5 }) },
    staked_bets: [
      { match_id: "m1", market: "result_3way", selection: "home", odds: 2.1, stake: 3.0, won: true, payout: 3.3 },
    ],
    top_winners: [
      { match_id: "m1", market: "result_3way", selection: "home", odds: 2.1, stake: 3.0, won: true, payout: 3.3, date: "2026-08-22", competition: "E0", home_team: "Arsenal", away_team: "Everton" },
    ],
    top_losers: [],
    ...overrides,
  };
}

describe("AgentPerformancePage", () => {
  beforeEach(() => {
    vi.mocked(getAgentPerformanceDashboard).mockReset();
    vi.mocked(getStatus).mockReset();
    vi.mocked(getSandboxStatus).mockReset();
    vi.mocked(getSandboxStatus).mockResolvedValue({ sandbox_mode: false, as_of: null });
    vi.mocked(getStatus).mockResolvedValue({} as never);
  });

  it("renders the Main Metrics KPI row from kelly_roi_simulation", async () => {
    vi.mocked(getAgentPerformanceDashboard).mockResolvedValue(dashboardData());
    render(<AgentPerformancePage />);

    expect(await screen.findByText("ROI")).toBeInTheDocument();
    expect(screen.getByText("Total Stake")).toBeInTheDocument();
    expect(screen.getByText("Money Won")).toBeInTheDocument();
    expect(screen.getByText("Bets Placed")).toBeInTheDocument();
    expect(screen.getByText("Hit %")).toBeInTheDocument();
    expect(screen.getByText("7")).toBeInTheDocument(); // kelly_roi_simulation.bets_placed
  });

  it("renders the three breakdown tables", async () => {
    vi.mocked(getAgentPerformanceDashboard).mockResolvedValue(dashboardData());
    render(<AgentPerformancePage />);

    expect(await screen.findByText("By Market")).toBeInTheDocument();
    expect(screen.getByText("By Market + Direction")).toBeInTheDocument();
    expect(screen.getByText("By League")).toBeInTheDocument();
    expect(screen.getByText("3-Way Result")).toBeInTheDocument(); // marketLabel("result_3way")
  });

  it("renders top winners and top losers tables, including an empty-losers state", async () => {
    vi.mocked(getAgentPerformanceDashboard).mockResolvedValue(dashboardData());
    render(<AgentPerformancePage />);

    expect(await screen.findByText("Arsenal v Everton")).toBeInTheDocument();
    expect(screen.getByText("Top 5 Winners")).toBeInTheDocument();
    expect(screen.getByText("Top 5 Losers")).toBeInTheDocument();
    expect(screen.getByText("None yet.")).toBeInTheDocument();
  });

  it("falls back to match_id when a top bet has no team names (cache miss)", async () => {
    vi.mocked(getAgentPerformanceDashboard).mockResolvedValue(
      dashboardData({
        top_winners: [
          { match_id: "m404", market: "btts", selection: "yes", odds: 1.9, stake: 2.0, won: true, payout: 1.8, date: "2026-08-22", competition: "E0", home_team: null, away_team: null },
        ],
      })
    );
    render(<AgentPerformancePage />);

    expect(await screen.findByText("m404")).toBeInTheDocument();
  });

  it("re-fetches with a new days value when a time-range pill is clicked", async () => {
    vi.mocked(getAgentPerformanceDashboard).mockResolvedValue(dashboardData());
    const user = userEvent.setup();
    render(<AgentPerformancePage />);

    await screen.findByText("ROI");
    await user.click(screen.getByText("Last 30 days"));

    await waitFor(() => expect(getAgentPerformanceDashboard).toHaveBeenLastCalledWith(30));
  });

  it("shows an error state when the fetch fails", async () => {
    const { ApiError } = await import("@/lib/api");
    vi.mocked(getAgentPerformanceDashboard).mockRejectedValue(new ApiError("boom", 500));
    render(<AgentPerformancePage />);

    expect(await screen.findByText("boom")).toBeInTheDocument();
  });
});
