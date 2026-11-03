import { render, screen } from "@testing-library/react";
import { describe, expect, it, vi } from "vitest";

import { OddsAvailabilityTab } from "../OddsAvailabilityTab";
import { getOpsOddsAvailabilityTrend } from "@/lib/api";
import type { OpsOddsAvailabilityTrendResponse } from "@/lib/types";

vi.mock("@/lib/api", () => ({
  getOpsOddsAvailabilityTrend: vi.fn(),
}));

describe("OddsAvailabilityTab", () => {
  it("ranks markets by % of fixtures missing them across the window", async () => {
    const response: OpsOddsAvailabilityTrendResponse = {
      weeks: [
        {
          week_start: "2026-10-08",
          total_fixtures: 50,
          by_market: { total_corners: { missing: 40, circuit_open: 30, not_posted: 10 } },
        },
        {
          week_start: "2026-10-15",
          total_fixtures: 50,
          by_market: {
            total_corners: { missing: 36, circuit_open: 20, not_posted: 16 },
            asian_handicap: { missing: 5, circuit_open: 0, not_posted: 5 },
          },
        },
      ],
    };
    vi.mocked(getOpsOddsAvailabilityTrend).mockResolvedValue(response);

    render(<OddsAvailabilityTab weeks={8} />);

    expect(await screen.findByText("total_corners")).toBeInTheDocument();
    // (40+36) missing / (50+50) total = 76%
    expect(screen.getByText("76%")).toBeInTheDocument();
    // 5 missing / 100 total = 5%
    expect(screen.getByText("asian_handicap")).toBeInTheDocument();
    expect(screen.getByText("5%")).toBeInTheDocument();
  });

  it("renders an actual bar (not a plain table) for each ranked market", async () => {
    vi.mocked(getOpsOddsAvailabilityTrend).mockResolvedValue({
      weeks: [
        {
          week_start: "2026-10-08",
          total_fixtures: 50,
          by_market: { total_corners: { missing: 40, circuit_open: 30, not_posted: 10 } },
        },
      ],
    });

    render(<OddsAvailabilityTab weeks={8} />);

    const row = (await screen.findByText("total_corners")).parentElement;
    // The bar track (bg-inset) must be a sibling of the market label, with a
    // severity-colored fill inside it -- a regression back to a plain table
    // would fail to produce either, so this is what would catch it.
    const track = row?.querySelector(".bg-inset");
    expect(track).toBeTruthy();
    expect(track?.querySelector(".bg-serious, .bg-warning, .bg-muted")).toBeTruthy();
  });

  it("renders an empty state with no missing markets anywhere in the window", async () => {
    vi.mocked(getOpsOddsAvailabilityTrend).mockResolvedValue({
      weeks: [{ week_start: "2026-10-08", total_fixtures: 10, by_market: {} }],
    });

    render(<OddsAvailabilityTab weeks={8} />);

    expect(await screen.findByText(/no missing markets/i)).toBeInTheDocument();
  });

  it("shows an error state when the fetch fails", async () => {
    vi.mocked(getOpsOddsAvailabilityTrend).mockRejectedValue(new Error("boom"));

    render(<OddsAvailabilityTab weeks={8} />);

    expect(await screen.findByText("boom")).toBeInTheDocument();
  });
});
