import { render, screen, fireEvent } from "@testing-library/react";
import { describe, expect, it, vi } from "vitest";

import { FeatureHealthTab } from "../FeatureHealthTab";
import { getOpsFeatureHealthTrend } from "@/lib/api";
import type { FlaggedFeature } from "@/lib/types";

vi.mock("@/lib/api", () => ({
  getOpsFeatureHealthTrend: vi.fn(),
}));

function flaggedFeature(overrides: Partial<FlaggedFeature> = {}): FlaggedFeature {
  return {
    name: "XOC_HOME", category: "offense", flags: ["drift"],
    baseline_mean: 1.8, recent_mean: 12.7, baseline_null_rate: 0.0, recent_null_rate: 0.0,
    magnitude: 7.1, weekly: [{ week_start: "2026-10-08", mean: 1.8, null_rate: 0.0 }],
    ...overrides,
  };
}

describe("FeatureHealthTab", () => {
  it("renders the scorecard and the flagged-feature list, sorted by the API's own order", async () => {
    vi.mocked(getOpsFeatureHealthTrend).mockResolvedValue({
      week_starts: ["2026-10-08"], tracked_feature_count: 52,
      flagged_features: [flaggedFeature()],
    });
    render(<FeatureHealthTab weeks={8} />);
    expect(await screen.findByText("XOC_HOME")).toBeInTheDocument();
    expect(screen.getByText("52")).toBeInTheDocument(); // tracked_feature_count tile
  });

  it("caps the list and reveals the rest on click", async () => {
    const many = Array.from({ length: 12 }, (_, i) => flaggedFeature({ name: `FEAT_${i}` }));
    vi.mocked(getOpsFeatureHealthTrend).mockResolvedValue({ week_starts: ["2026-10-08"], tracked_feature_count: 52, flagged_features: many });
    render(<FeatureHealthTab weeks={8} />);
    await screen.findByText("FEAT_0");
    expect(screen.queryByText("FEAT_11")).not.toBeInTheDocument();
    fireEvent.click(screen.getByText(/show .* more/i));
    expect(screen.getByText("FEAT_11")).toBeInTheDocument();
  });

  it("filters by name via the search box", async () => {
    vi.mocked(getOpsFeatureHealthTrend).mockResolvedValue({
      week_starts: ["2026-10-08"], tracked_feature_count: 2,
      flagged_features: [flaggedFeature(), flaggedFeature({ name: "DEF_ANCHOR_AWAY" })],
    });
    render(<FeatureHealthTab weeks={8} />);
    await screen.findByText("XOC_HOME");
    fireEvent.change(screen.getByPlaceholderText(/filter by name/i), { target: { value: "DEF" } });
    expect(screen.queryByText("XOC_HOME")).not.toBeInTheDocument();
    expect(screen.getByText("DEF_ANCHOR_AWAY")).toBeInTheDocument();
  });

  it("renders a clean empty state when nothing is flagged", async () => {
    vi.mocked(getOpsFeatureHealthTrend).mockResolvedValue({ week_starts: ["2026-10-08"], tracked_feature_count: 52, flagged_features: [] });
    render(<FeatureHealthTab weeks={8} />);
    expect(await screen.findByText(/no.*flagged/i)).toBeInTheDocument();
  });

  it("shows an error state when the fetch fails", async () => {
    vi.mocked(getOpsFeatureHealthTrend).mockRejectedValue(new Error("boom"));
    render(<FeatureHealthTab weeks={8} />);
    expect(await screen.findByText("boom")).toBeInTheDocument();
  });
});
