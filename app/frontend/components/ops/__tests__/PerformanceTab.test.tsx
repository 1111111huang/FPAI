import { render, screen, fireEvent } from "@testing-library/react";
import { describe, expect, it, vi } from "vitest";

import { PerformanceTab } from "../PerformanceTab";
import { getOpsPerformanceTrend } from "@/lib/api";

vi.mock("@/lib/api", () => ({
  getOpsPerformanceTrend: vi.fn(),
}));

describe("PerformanceTab", () => {
  it("renders a bar per week and a week-over-week delta", async () => {
    vi.mocked(getOpsPerformanceTrend).mockResolvedValue({
      weeks: [
        { week_start: "2026-10-29", hit_rate: 0.58, roi: 0.1, bets_placed: 10, total_profit: 50, ending_bankroll: 1050 },
        { week_start: "2026-11-05", hit_rate: 0.71, roi: 0.2, bets_placed: 8, total_profit: 80, ending_bankroll: 1080 },
      ],
      pick_scope: "one_per_match",
      include_conditional: false,
    });
    render(<PerformanceTab weeks={8} />);
    expect(await screen.findByText(/71%/)).toBeInTheDocument();
    expect(screen.getByText(/\+13pp/)).toBeInTheDocument();
  });

  it("re-fetches with the roi metric mode when selected", async () => {
    vi.mocked(getOpsPerformanceTrend).mockResolvedValue({
      weeks: [{ week_start: "2026-11-05", hit_rate: 0.5, roi: 0.25, bets_placed: 4, total_profit: 20, ending_bankroll: 1020 }],
      pick_scope: "one_per_match",
      include_conditional: false,
    });
    render(<PerformanceTab weeks={8} />);
    await screen.findByText(/50%/);
    fireEvent.click(screen.getByRole("button", { name: "ROI" }));
    expect(await screen.findByText(/25%/)).toBeInTheDocument();
  });

  it("re-fetches with the chosen strategy toggles", async () => {
    vi.mocked(getOpsPerformanceTrend).mockResolvedValue({
      weeks: [{ week_start: "2026-11-05", hit_rate: 0.5, roi: 0.1, bets_placed: 4, total_profit: 20, ending_bankroll: 1020 }],
      pick_scope: "all_direct",
      include_conditional: true,
    });
    render(<PerformanceTab weeks={8} />);
    await screen.findByText(/50%/);
    fireEvent.click(screen.getByRole("button", { name: /all direct/i }));
    fireEvent.click(screen.getByRole("button", { name: /include conditional/i }));
    expect(getOpsPerformanceTrend).toHaveBeenLastCalledWith(
      expect.objectContaining({ pickScope: "all_direct", includeConditional: true }),
    );
  });

  it("shows an error state when the fetch fails", async () => {
    vi.mocked(getOpsPerformanceTrend).mockRejectedValue(new Error("boom"));
    render(<PerformanceTab weeks={8} />);
    expect(await screen.findByText("boom")).toBeInTheDocument();
  });
});
