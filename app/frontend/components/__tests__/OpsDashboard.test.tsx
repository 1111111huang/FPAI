import { render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { describe, expect, it, vi } from "vitest";

import { OpsDashboard } from "../OpsDashboard";

vi.mock("@/lib/api", () => ({
  getOpsFailures: vi.fn().mockResolvedValue({
    failures: [],
    circuit_breakers: { the_odds_api: { state: "closed" }, oddspapi: { state: "closed" } },
  }),
  getOpsOddsAvailability: vi.fn().mockResolvedValue({ fixtures: [] }),
  getOpsFeatureHealth: vi.fn().mockResolvedValue({ by_league: {} }),
  getAgentPerformanceDashboard: vi.fn().mockResolvedValue({
    kelly_roi_simulation: {
      matches_evaluated: 0, bets_placed: 0, bets_won: 0, roi: 0, hit_rate: 0,
      bet_frequency: 0, max_drawdown: 0, insufficient_data_rate: 0,
      starting_bankroll: 0, ending_bankroll: 0, total_staked: 0, total_profit: 0,
    },
    by_market_metrics: {},
    by_market_selection_metrics: {},
    by_league_metrics: {},
    staked_bets: [],
    top_winners: [],
    top_losers: [],
  }),
  getStatus: vi.fn().mockResolvedValue({}),
  getSandboxStatus: vi.fn().mockResolvedValue({ sandbox_mode: false, as_of: null }),
  ApiError: class ApiError extends Error {
    status?: number;
  },
}));

describe("OpsDashboard", () => {
  it("renders all four tab labels and defaults to the failures tab", async () => {
    render(<OpsDashboard />);
    expect(await screen.findByRole("tab", { name: /failures/i })).toHaveAttribute("aria-selected", "true");
    expect(screen.getByRole("tab", { name: /odds availability/i })).toBeInTheDocument();
    expect(screen.getByRole("tab", { name: /feature health/i })).toBeInTheDocument();
    expect(screen.getByRole("tab", { name: /performance/i })).toBeInTheDocument();
  });

  it("switches tabs on click", async () => {
    const user = userEvent.setup();
    render(<OpsDashboard />);
    await screen.findByRole("tab", { name: /failures/i });
    await user.click(screen.getByRole("tab", { name: /odds availability/i }));
    expect(screen.getByRole("tab", { name: /odds availability/i })).toHaveAttribute("aria-selected", "true");
  });

  it("renders a league filter that defaults to all leagues", async () => {
    render(<OpsDashboard />);
    await screen.findByRole("tab", { name: /failures/i });
    expect(screen.getByLabelText(/league/i)).toHaveValue("");
  });
});
