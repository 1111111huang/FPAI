import { render, screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { describe, expect, it, vi, beforeEach } from "vitest";
import { AppShell } from "../AppShell";

vi.mock("@/lib/api", () => ({
  getStatus: vi.fn(),
  getFixtures: vi.fn(),
  getSandboxStatus: vi.fn(),
}));

import { getFixtures, getSandboxStatus, getStatus } from "@/lib/api";

/** A promise whose resolution is controlled from outside, so the test can
 * force a specific out-of-order resolution sequence -- same helper as
 * MatchUI.race.test.tsx's W42 regression tests. */
function deferred<T>() {
  let resolve!: (value: T) => void;
  const promise = new Promise<T>((res) => {
    resolve = res;
  });
  return { promise, resolve };
}

describe("AppShell", () => {
  beforeEach(() => {
    vi.mocked(getStatus).mockReset();
    vi.mocked(getFixtures).mockReset();
    vi.mocked(getSandboxStatus).mockReset();
    vi.mocked(getSandboxStatus).mockResolvedValue({ sandbox_mode: false, as_of: null });
  });

  it("renders nav links and page content", () => {
    vi.mocked(getStatus).mockRejectedValue(new Error("no backend"));
    render(
      <AppShell active="dashboard">
        <p>page content</p>
      </AppShell>
    );
    expect(screen.getByText("Dashboard")).toBeInTheDocument();
    expect(screen.getByText("All Matches")).toBeInTheDocument();
    expect(screen.getByText("Bets")).toBeInTheDocument();
    expect(screen.getByText("page content")).toBeInTheDocument();
  });

  it("does not crash when the status fetch fails, and shows a placeholder instead of a real value", async () => {
    vi.mocked(getStatus).mockRejectedValue(new Error("no backend"));
    render(
      <AppShell active="dashboard">
        <p>content</p>
      </AppShell>
    );
    await waitFor(() => expect(getStatus).toHaveBeenCalled());
    expect(screen.getAllByText("—")).toHaveLength(2);
  });

  it("renders real model status and last-updated once the status fetch resolves", async () => {
    vi.mocked(getStatus).mockResolvedValue({
      data_freshness: { latest_match_date: "2026-07-20", days_since_update: 3, match_count: 100, is_stale: false },
      model_status: { league: { result_3way: { model_type: "x", primary_metric_value: 0.6, metric_name: "m", selected_at: "now" } }, international: {} },
    });
    render(
      <AppShell active="dashboard">
        <p>content</p>
      </AppShell>
    );
    expect(await screen.findByText(/league 1/)).toBeInTheDocument();
    expect(screen.getByText(/2026-07-20/)).toBeInTheDocument();
  });

  it("shows Active Edges only when the prop is provided", () => {
    vi.mocked(getStatus).mockRejectedValue(new Error("no backend"));
    const { rerender } = render(
      <AppShell active="dashboard">
        <p>content</p>
      </AppShell>
    );
    expect(screen.queryByText("Active Edges")).not.toBeInTheDocument();
    rerender(
      <AppShell active="dashboard" activeEdgesCount={4}>
        <p>content</p>
      </AppShell>
    );
    expect(screen.getByText("Active Edges")).toBeInTheDocument();
    expect(screen.getByText("4")).toBeInTheDocument();
  });

  it("does not fetch fixtures on mount -- only once the search input is focused", async () => {
    vi.mocked(getStatus).mockRejectedValue(new Error("no backend"));
    render(
      <AppShell active="dashboard">
        <p>content</p>
      </AppShell>
    );
    await waitFor(() => expect(getStatus).toHaveBeenCalled());
    expect(getFixtures).not.toHaveBeenCalled();

    const user = userEvent.setup();
    await user.click(screen.getByPlaceholderText("Search fixtures, teams…"));
    await waitFor(() => expect(getFixtures).toHaveBeenCalledTimes(1));
  });

  it("filters and links search results to Match Analysis, carrying league", async () => {
    vi.mocked(getStatus).mockRejectedValue(new Error("no backend"));
    vi.mocked(getFixtures).mockResolvedValue([
      { match_id: "1", utc_date: "2026-08-22T15:00:00Z", status: "SCHEDULED", home_team: "Arsenal", away_team: "Everton", home_goals: null, away_goals: null, competition: "E0" },
      { match_id: "2", utc_date: "2026-08-23T15:00:00Z", status: "SCHEDULED", home_team: "Malmo FF", away_team: "AIK", home_goals: null, away_goals: null, competition: "SWE" },
    ]);
    const user = userEvent.setup();
    render(
      <AppShell active="dashboard">
        <p>content</p>
      </AppShell>
    );
    const input = screen.getByPlaceholderText("Search fixtures, teams…");
    await user.click(input);
    await user.type(input, "malmo");

    const link = await screen.findByText(/Malmo FF/);
    expect(link.closest("a")).toHaveAttribute("href", expect.stringContaining("league=SWE"));
    expect(screen.queryByText(/Arsenal/)).not.toBeInTheDocument();
  });

  it("re-fetches search fixtures with the corrected sandbox date once asOf resolves after an early focus (W42-style race)", async () => {
    vi.mocked(getStatus).mockRejectedValue(new Error("no backend"));
    const today = new Date().toISOString().slice(0, 10);
    const sandboxDate = "2025-03-05";
    const sandboxWindowEnd = "2025-06-03"; // 2025-03-05 + 90 days

    const sandboxStatus = deferred<{ sandbox_mode: boolean; as_of: string | null }>();
    vi.mocked(getSandboxStatus).mockReturnValue(sandboxStatus.promise);
    vi.mocked(getFixtures).mockResolvedValue([]);

    const user = userEvent.setup();
    render(
      <AppShell active="dashboard">
        <p>content</p>
      </AppShell>
    );

    // Focus lands before the sandbox status corrects `asOf` -- fires the
    // first (stale, real-clock) fetch.
    await user.click(screen.getByPlaceholderText("Search fixtures, teams…"));
    await waitFor(() => expect(getFixtures).toHaveBeenCalledWith(today, expect.any(String)));

    // Now the sandbox status resolves, correcting `asOf` -- this must
    // trigger a second, corrected fetch rather than leaving the search box
    // permanently locked into the stale real-clock window.
    sandboxStatus.resolve({ sandbox_mode: true, as_of: sandboxDate });
    await waitFor(() => expect(getFixtures).toHaveBeenCalledWith(sandboxDate, sandboxWindowEnd));
  });
});
