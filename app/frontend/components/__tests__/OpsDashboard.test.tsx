import { render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { describe, expect, it, vi } from "vitest";

import { OpsDashboard } from "../OpsDashboard";

vi.mock("@/lib/api", () => ({
  getOpsFailures: vi.fn().mockResolvedValue({
    failures: [],
    circuit_breakers: { the_odds_api: { state: "closed" }, oddspapi: { state: "closed" } },
  }),
  getOpsOddsAvailabilityTrend: vi.fn().mockResolvedValue({ weeks: [] }),
  getOpsFeatureHealthTrend: vi.fn().mockResolvedValue({ week_starts: [], tracked_feature_count: 0, flagged_features: [] }),
  getOpsPerformanceTrend: vi.fn().mockResolvedValue({ weeks: [], pick_scope: "one_per_match", include_conditional: false }),
  // AppShell's useSandboxAsOf() hook calls this on every mount, independent
  // of which /ops tab is active -- still needed even though the Performance
  // tab itself no longer does.
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

  it("defaults the Window selector to 8w", async () => {
    render(<OpsDashboard />);
    await screen.findByRole("tab", { name: /failures/i });
    expect(screen.getByRole("button", { name: "8w" })).toHaveAttribute("aria-pressed", "true");
    expect(screen.getByRole("button", { name: "4w" })).toHaveAttribute("aria-pressed", "false");
  });

  it("selects a different window chip on click", async () => {
    const user = userEvent.setup();
    render(<OpsDashboard />);
    await screen.findByRole("tab", { name: /failures/i });
    await user.click(screen.getByRole("button", { name: "26w" }));
    expect(screen.getByRole("button", { name: "26w" })).toHaveAttribute("aria-pressed", "true");
    expect(screen.getByRole("button", { name: "8w" })).toHaveAttribute("aria-pressed", "false");
  });
});
