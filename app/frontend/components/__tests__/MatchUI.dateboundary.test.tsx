import { describe, expect, it, vi, beforeEach } from "vitest";
import { render, waitFor } from "@testing-library/react";
import { DashboardPage, MatchExplorerPage } from "../MatchUI";
import { getFixtures, getSandboxStatus } from "@/lib/api";

vi.mock("@/lib/api");

describe("date-boundary correctness via the sandbox clock (W38)", () => {
  beforeEach(() => {
    // Reset call history (not just resolved values) between tests. Without
    // this, getFixtures/getSandboxStatus.mock.calls accumulate across the
    // `it` blocks below (vi.mock("@/lib/api") without a factory shares one
    // mock instance for the whole file), which can mask a genuinely failing
    // assertion behind a stale matching call left over from an earlier test.
    vi.mocked(getFixtures).mockReset();
    vi.mocked(getSandboxStatus).mockReset();
    vi.mocked(getFixtures).mockResolvedValue([]);
  });

  it("Dashboard queries fixtures for the sandbox as_of date, not the real browser date", async () => {
    vi.mocked(getSandboxStatus).mockResolvedValue({ sandbox_mode: true, as_of: "2026-03-01" });

    render(<DashboardPage />);

    await waitFor(() => expect(getFixtures).toHaveBeenCalledWith("2026-03-01", "2026-03-01"));
  });

  it("Dashboard's fixture query shifts to the next simulated day at midnight", async () => {
    // useSandboxAsOf() fetches once per mount (empty effect deps, W30) --
    // a same-instance rerender() never re-fires it, so unmount+remount (a
    // page reload) is how a new simulated day is actually observed.
    vi.mocked(getSandboxStatus).mockResolvedValueOnce({ sandbox_mode: true, as_of: "2026-03-01" });
    const { unmount } = render(<DashboardPage />);
    await waitFor(() => expect(getFixtures).toHaveBeenCalledWith("2026-03-01", "2026-03-01"));
    unmount();

    vi.mocked(getSandboxStatus).mockResolvedValueOnce({ sandbox_mode: true, as_of: "2026-03-02" });
    render(<DashboardPage />);

    await waitFor(() => expect(getFixtures).toHaveBeenCalledWith("2026-03-02", "2026-03-02"));
  });

  it("Match Explorer's 90-day window is anchored to the sandbox as_of date, not the real browser date", async () => {
    vi.mocked(getSandboxStatus).mockResolvedValue({ sandbox_mode: true, as_of: "2026-03-01" });

    render(<MatchExplorerPage />);

    // 2026-03-01 + 90 days = 2026-05-30
    await waitFor(() => expect(getFixtures).toHaveBeenCalledWith("2026-03-01", "2026-05-30"));
  });
});
