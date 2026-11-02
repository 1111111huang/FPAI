import { render, screen } from "@testing-library/react";
import { describe, expect, it, vi } from "vitest";

import { FailuresTab } from "../FailuresTab";
import { getOpsFailures } from "@/lib/api";
import type { OpsFailuresResponse } from "@/lib/types";

vi.mock("@/lib/api", () => ({
  getOpsFailures: vi.fn(),
}));

describe("FailuresTab", () => {
  it("renders failure rows and circuit breaker state", async () => {
    const response: OpsFailuresResponse = {
      failures: [
        {
          id: 1,
          ts: "2026-10-31T11:07:54Z",
          subsystem: "pregenerate",
          unit_id: "league=I1",
          league: "I1",
          exc_type: "JSONDecodeError",
          message: "Expecting value: line 1 column 1 (char 0)",
          traceback: "...",
          retryable: null,
          alerted: 1,
        },
      ],
      circuit_breakers: { the_odds_api: { state: "open" }, oddspapi: { state: "closed" } },
    };
    vi.mocked(getOpsFailures).mockResolvedValue(response);

    render(<FailuresTab />);

    expect(await screen.findByText(/JSONDecodeError/)).toBeInTheDocument();
    expect(screen.getByText(/league=I1/)).toBeInTheDocument();
    expect(screen.getByText(/the_odds_api/i)).toBeInTheDocument();
    expect(screen.getByText(/open/i)).toBeInTheDocument();
  });

  it("renders an empty state with no failures", async () => {
    vi.mocked(getOpsFailures).mockResolvedValue({
      failures: [],
      circuit_breakers: { the_odds_api: { state: "closed" }, oddspapi: { state: "closed" } },
    });

    render(<FailuresTab />);

    expect(await screen.findByText(/no failures/i)).toBeInTheDocument();
  });

  it("shows an error state when the fetch fails", async () => {
    vi.mocked(getOpsFailures).mockRejectedValue(new Error("boom"));

    render(<FailuresTab />);

    expect(await screen.findByText("boom")).toBeInTheDocument();
  });
});
