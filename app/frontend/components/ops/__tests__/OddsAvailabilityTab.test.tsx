import { render, screen } from "@testing-library/react";
import { describe, expect, it, vi } from "vitest";

import { OddsAvailabilityTab } from "../OddsAvailabilityTab";
import { getOpsOddsAvailability } from "@/lib/api";
import type { OpsOddsAvailabilityResponse } from "@/lib/types";

vi.mock("@/lib/api", () => ({
  getOpsOddsAvailability: vi.fn(),
}));

describe("OddsAvailabilityTab", () => {
  it("renders fixtures with missing markets and likely cause", async () => {
    const response: OpsOddsAvailabilityResponse = {
      fixtures: [
        {
          match_id: "558542",
          date: "2026-11-02",
          league: "I1",
          home_team: "Udinese",
          away_team: "Roma",
          missing_markets: ["over/under 2.5"],
          likely_cause: "circuit_open",
        },
      ],
    };
    vi.mocked(getOpsOddsAvailability).mockResolvedValue(response);

    render(<OddsAvailabilityTab dateFrom="2026-11-01" dateTo="2026-11-02" />);

    expect(await screen.findByText(/Udinese/)).toBeInTheDocument();
    expect(screen.getByText(/Roma/)).toBeInTheDocument();
    expect(screen.getByText(/over\/under 2.5/)).toBeInTheDocument();
    expect(screen.getByText(/circuit_open/)).toBeInTheDocument();
  });

  it("renders an empty state with no missing markets", async () => {
    vi.mocked(getOpsOddsAvailability).mockResolvedValue({ fixtures: [] });

    render(<OddsAvailabilityTab dateFrom="2026-11-01" dateTo="2026-11-02" />);

    expect(await screen.findByText(/no missing markets/i)).toBeInTheDocument();
  });

  it("shows an error state when the fetch fails", async () => {
    vi.mocked(getOpsOddsAvailability).mockRejectedValue(new Error("boom"));

    render(<OddsAvailabilityTab dateFrom="2026-11-01" dateTo="2026-11-02" />);

    expect(await screen.findByText("boom")).toBeInTheDocument();
  });
});
