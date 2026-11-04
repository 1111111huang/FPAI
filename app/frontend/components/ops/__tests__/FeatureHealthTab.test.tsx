import { act, render, screen, fireEvent } from "@testing-library/react";
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
  it("renders the scorecard and the flagged-feature list", async () => {
    vi.mocked(getOpsFeatureHealthTrend).mockResolvedValue({
      week_starts: ["2026-10-08"], tracked_feature_count: 52,
      flagged_features: [flaggedFeature()],
    });
    render(<FeatureHealthTab weeks={8} />);
    expect(await screen.findByText("XOC_HOME")).toBeInTheDocument();
    expect(screen.getByText("52")).toBeInTheDocument(); // tracked_feature_count tile
  });

  it("renders a per-feature sparkline with real min/max axis labels", async () => {
    vi.mocked(getOpsFeatureHealthTrend).mockResolvedValue({
      week_starts: ["2026-10-08", "2026-10-15", "2026-10-22"], tracked_feature_count: 52,
      flagged_features: [
        flaggedFeature({
          weekly: [
            { week_start: "2026-10-08", mean: 1.8, null_rate: 0.0 },
            { week_start: "2026-10-15", mean: 6.2, null_rate: 0.0 },
            { week_start: "2026-10-22", mean: 12.7, null_rate: 0.0 },
          ],
        }),
      ],
    });
    render(<FeatureHealthTab weeks={8} />);
    await screen.findByText("XOC_HOME");
    expect(screen.getByText("1.8")).toBeInTheDocument(); // min
    expect(screen.getByText("12.7")).toBeInTheDocument(); // max
  });

  it("renders flagged features sorted by magnitude, highest first", async () => {
    vi.mocked(getOpsFeatureHealthTrend).mockResolvedValue({
      week_starts: ["2026-10-08"], tracked_feature_count: 52,
      flagged_features: [
        flaggedFeature({ name: "LOW_MAGNITUDE", magnitude: 2.0 }),
        flaggedFeature({ name: "HIGH_MAGNITUDE", magnitude: 9.0 }),
      ],
    });
    render(<FeatureHealthTab weeks={8} />);
    await screen.findByText("HIGH_MAGNITUDE");
    const names = screen.getAllByText(/_MAGNITUDE$/).map((el) => el.textContent);
    expect(names).toEqual(["HIGH_MAGNITUDE", "LOW_MAGNITUDE"]);
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

  it("re-fetches when the weeks prop changes", async () => {
    vi.mocked(getOpsFeatureHealthTrend).mockResolvedValue({
      week_starts: ["2026-10-08"], tracked_feature_count: 52,
      flagged_features: [flaggedFeature()],
    });
    const { rerender } = render(<FeatureHealthTab weeks={8} />);
    await screen.findByText("XOC_HOME");
    await act(async () => {
      rerender(<FeatureHealthTab weeks={12} />);
    });
    expect(getOpsFeatureHealthTrend).toHaveBeenLastCalledWith(expect.objectContaining({ weeks: 12 }));
  });
});
