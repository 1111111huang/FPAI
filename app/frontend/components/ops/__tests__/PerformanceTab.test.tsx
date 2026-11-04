import { act, render, screen, fireEvent } from "@testing-library/react";
import { beforeEach, describe, expect, it, vi } from "vitest";

import { PerformanceTab } from "../PerformanceTab";
import { getOpsPerformanceTrend } from "@/lib/api";

vi.mock("@/lib/api", () => ({
  getOpsPerformanceTrend: vi.fn(),
}));

describe("PerformanceTab", () => {
  // Mock call history is otherwise cumulative across tests in this file --
  // needed so "switches metric display without re-fetching" can assert an
  // exact call count without counting prior tests' renders.
  beforeEach(() => {
    vi.mocked(getOpsPerformanceTrend).mockClear();
  });


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

  it("switches metric display without re-fetching", async () => {
    vi.mocked(getOpsPerformanceTrend).mockResolvedValue({
      weeks: [{ week_start: "2026-11-05", hit_rate: 0.5, roi: 0.25, bets_placed: 4, total_profit: 20, ending_bankroll: 1020 }],
      pick_scope: "one_per_match",
      include_conditional: false,
    });
    render(<PerformanceTab weeks={8} />);
    await screen.findByText(/50%/);
    fireEvent.click(screen.getByRole("button", { name: "ROI" }));
    expect(await screen.findByText(/25%/)).toBeInTheDocument();
    // metric is deliberately not a useEffect dependency (it's a display
    // choice over data already fetched) -- switching it must never trigger
    // a second network call.
    expect(getOpsPerformanceTrend).toHaveBeenCalledTimes(1);
  });

  it("re-fetches with the chosen strategy toggles", async () => {
    vi.mocked(getOpsPerformanceTrend).mockResolvedValue({
      weeks: [{ week_start: "2026-11-05", hit_rate: 0.5, roi: 0.1, bets_placed: 4, total_profit: 20, ending_bankroll: 1020 }],
      pick_scope: "all_direct",
      include_conditional: true,
    });
    render(<PerformanceTab weeks={8} />);
    await screen.findByText(/50%/);
    await act(async () => {
      fireEvent.click(screen.getByRole("button", { name: /all direct/i }));
      fireEvent.click(screen.getByRole("button", { name: /include conditional/i }));
    });
    expect(getOpsPerformanceTrend).toHaveBeenLastCalledWith(
      expect.objectContaining({ pickScope: "all_direct", includeConditional: true }),
    );
  });

  it("re-fetches when the weeks prop changes", async () => {
    vi.mocked(getOpsPerformanceTrend).mockResolvedValue({
      weeks: [{ week_start: "2026-11-05", hit_rate: 0.5, roi: 0.1, bets_placed: 4, total_profit: 20, ending_bankroll: 1020 }],
      pick_scope: "one_per_match",
      include_conditional: false,
    });
    const { rerender } = render(<PerformanceTab weeks={8} />);
    await screen.findByText(/50%/);
    await act(async () => {
      rerender(<PerformanceTab weeks={12} />);
    });
    expect(getOpsPerformanceTrend).toHaveBeenLastCalledWith(expect.objectContaining({ weeks: 12 }));
  });

  it("renders a losing week's negative total_profit with a serious (not good) delta", async () => {
    vi.mocked(getOpsPerformanceTrend).mockResolvedValue({
      weeks: [
        { week_start: "2026-10-29", hit_rate: 0.6, roi: 0.1, bets_placed: 5, total_profit: 40, ending_bankroll: 1040 },
        { week_start: "2026-11-05", hit_rate: 0.5, roi: -0.05, bets_placed: 3, total_profit: -20, ending_bankroll: 1020 },
      ],
      pick_scope: "one_per_match",
      include_conditional: false,
    });
    render(<PerformanceTab weeks={8} />);
    await screen.findByText(/50%/);
    fireEvent.click(screen.getByRole("button", { name: "Total Profit" }));
    const delta = await screen.findByText(/-60\.0/);
    expect(delta).toHaveClass("text-serious");
    expect(screen.getByText("-20.0")).toBeInTheDocument();
  });

  it("shows an error state when the fetch fails", async () => {
    vi.mocked(getOpsPerformanceTrend).mockRejectedValue(new Error("boom"));
    render(<PerformanceTab weeks={8} />);
    expect(await screen.findByText("boom")).toBeInTheDocument();
  });
});
