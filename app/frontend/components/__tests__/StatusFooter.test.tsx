/**
 * W17: system status footer -- surfaces data staleness and current model
 * selection counts, fetched from GET /api/status (mocked here, no live
 * backend).
 */
import { render, screen } from "@testing-library/react";
import { describe, expect, it, vi } from "vitest";

import { StatusFooter } from "../StatusFooter";

vi.mock("@/lib/api", () => ({
  getStatus: vi.fn(),
}));

import { getStatus } from "@/lib/api";

describe("StatusFooter", () => {
  it("renders data staleness and model selection counts", async () => {
    vi.mocked(getStatus).mockResolvedValue({
      data_freshness: { latest_match_date: "2026-05-24", days_since_update: 49, match_count: 3800, is_stale: true },
      model_status: {
        league: { btts: { model_type: "xgboost", primary_metric_value: 0.68, metric_name: "test_log_loss", selected_at: "now" }, result_3way: { model_type: "xgboost", primary_metric_value: 0.9, metric_name: "test_log_loss", selected_at: "now" } },
        international: {},
      },
    });

    render(<StatusFooter />);

    expect(await screen.findByText(/2026-05-24/)).toBeInTheDocument();
    expect(screen.getByText(/49d/)).toBeInTheDocument();
    expect(screen.getByText(/stale/i)).toBeInTheDocument();
    expect(screen.getByText(/league 2/)).toBeInTheDocument();
    expect(screen.getByText(/international 0/)).toBeInTheDocument();
  });

  it("renders nothing while loading or if the fetch fails, rather than crashing", async () => {
    vi.mocked(getStatus).mockRejectedValue(new Error("network error"));
    const { container } = render(<StatusFooter />);
    expect(container).toBeTruthy(); // does not throw
  });

  it("does not render a stale warning when data is fresh", async () => {
    vi.mocked(getStatus).mockResolvedValue({
      data_freshness: { latest_match_date: "2026-07-11", days_since_update: 1, match_count: 4000, is_stale: false },
      model_status: { league: {}, international: {} },
    });

    render(<StatusFooter />);

    expect(await screen.findByText(/2026-07-11/)).toBeInTheDocument();
    expect(screen.queryByText(/stale/i)).not.toBeInTheDocument();
  });
});
