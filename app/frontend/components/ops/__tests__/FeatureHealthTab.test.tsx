import { render, screen } from "@testing-library/react";
import { describe, expect, it, vi } from "vitest";

import { FeatureHealthTab } from "../FeatureHealthTab";
import { getOpsFeatureHealth } from "@/lib/api";
import type { OpsFeatureHealthResponse } from "@/lib/types";

vi.mock("@/lib/api", () => ({
  getOpsFeatureHealth: vi.fn(),
}));

describe("FeatureHealthTab", () => {
  it("renders per-league feature health stats", async () => {
    const response: OpsFeatureHealthResponse = {
      by_league: {
        I1: { sample_size: 12, cold_start_risk_rate: 0.25, unknown_team_rate: 0.0, avg_feature_completeness: 0.82 },
      },
    };
    vi.mocked(getOpsFeatureHealth).mockResolvedValue(response);

    render(<FeatureHealthTab dateFrom="2026-11-01" dateTo="2026-11-02" />);

    expect(await screen.findByText("I1")).toBeInTheDocument();
    expect(screen.getByText("12")).toBeInTheDocument();
    expect(screen.getByText(/82/)).toBeInTheDocument();
    expect(screen.getByText(/25/)).toBeInTheDocument();
  });

  it("renders a null avg_feature_completeness as a dash, not a crash", async () => {
    vi.mocked(getOpsFeatureHealth).mockResolvedValue({
      by_league: {
        SP1: { sample_size: 0, cold_start_risk_rate: 0, unknown_team_rate: 0, avg_feature_completeness: null },
      },
    });

    render(<FeatureHealthTab dateFrom="2026-11-01" dateTo="2026-11-02" />);

    expect(await screen.findByText("SP1")).toBeInTheDocument();
    expect(screen.getByText("—")).toBeInTheDocument();
  });

  it("flags a high cold-start risk rate as serious, not the same as a low rate", async () => {
    vi.mocked(getOpsFeatureHealth).mockResolvedValue({
      by_league: {
        I1: { sample_size: 12, cold_start_risk_rate: 0.4, unknown_team_rate: 0.02, avg_feature_completeness: 0.95 },
      },
    });

    render(<FeatureHealthTab dateFrom="2026-11-01" dateTo="2026-11-02" />);

    const risky = await screen.findByText(/40/);
    expect(risky).toHaveClass("text-serious");
    const safe = screen.getByText(/2/, { selector: ".text-good" });
    expect(safe).toHaveClass("text-good");
  });

  it("flags a low avg_feature_completeness as serious, not the same as a high one", async () => {
    vi.mocked(getOpsFeatureHealth).mockResolvedValue({
      by_league: {
        SP1: { sample_size: 10, cold_start_risk_rate: 0, unknown_team_rate: 0, avg_feature_completeness: 0.65 },
        I1: { sample_size: 10, cold_start_risk_rate: 0, unknown_team_rate: 0, avg_feature_completeness: 0.95 },
      },
    });

    render(<FeatureHealthTab dateFrom="2026-11-01" dateTo="2026-11-02" />);

    const low = await screen.findByText(/65/);
    expect(low).toHaveClass("text-serious");
    const high = screen.getByText(/95/);
    expect(high).toHaveClass("text-good");
  });

  it("renders an empty state with no leagues", async () => {
    vi.mocked(getOpsFeatureHealth).mockResolvedValue({ by_league: {} });

    render(<FeatureHealthTab dateFrom="2026-11-01" dateTo="2026-11-02" />);

    expect(await screen.findByText(/no recommendations/i)).toBeInTheDocument();
  });

  it("shows an error state when the fetch fails", async () => {
    vi.mocked(getOpsFeatureHealth).mockRejectedValue(new Error("boom"));

    render(<FeatureHealthTab dateFrom="2026-11-01" dateTo="2026-11-02" />);

    expect(await screen.findByText("boom")).toBeInTheDocument();
  });
});
