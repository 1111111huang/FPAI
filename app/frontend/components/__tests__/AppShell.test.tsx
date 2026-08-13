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
    // W114: each nav label now renders twice by design -- once in the
    // desktop sidebar, once in the mobile bottom tab bar (CSS decides which
    // is visible; jsdom renders both regardless of viewport).
    expect(screen.getAllByText("Dashboard")).toHaveLength(2);
    expect(screen.getAllByText("All Matches")).toHaveLength(2);
    expect(screen.getByText("page content")).toBeInTheDocument();
  });

  it("does not render a Bets nav link -- feature not ready yet (hidden 2026-08-13)", () => {
    vi.mocked(getStatus).mockRejectedValue(new Error("no backend"));
    render(
      <AppShell active="dashboard">
        <p>page content</p>
      </AppShell>
    );
    expect(screen.queryByText("Bets")).not.toBeInTheDocument();
  });

  it("W114: renders a bottom tab bar with the same nav items, for small screens", () => {
    vi.mocked(getStatus).mockRejectedValue(new Error("no backend"));
    const { container } = render(
      <AppShell active="dashboard">
        <p>page content</p>
      </AppShell>
    );
    const bottomNav = container.querySelector("nav.fixed");
    expect(bottomNav).not.toBeNull();
    expect(bottomNav?.querySelectorAll("a")).toHaveLength(2);
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
      model_status: { E0: { result_3way: { model_type: "x", primary_metric_value: 0.6, metric_name: "m", selected_at: "now" } }, international: {} },
    });
    render(
      <AppShell active="dashboard">
        <p>content</p>
      </AppShell>
    );
    expect(await screen.findByText(/league 1/)).toBeInTheDocument();
    expect(screen.getByText(/2026-07-20/)).toBeInTheDocument();
    expect(screen.queryByText(/-- stale/)).not.toBeInTheDocument();
  });

  // US#110: model_status has no fixed "league" key -- it's keyed dynamically
  // per competition_id, so a second competition-specific league (SWE) must
  // still add to the combined league model count.
  it("sums model counts across every non-international context key", async () => {
    vi.mocked(getStatus).mockResolvedValue({
      data_freshness: { latest_match_date: "2026-07-20", days_since_update: 3, match_count: 100, is_stale: false },
      model_status: {
        E0: { result_3way: { model_type: "x", primary_metric_value: 0.6, metric_name: "m", selected_at: "now" } },
        SWE: { result_3way: { model_type: "x", primary_metric_value: 0.6, metric_name: "m", selected_at: "now" } },
        international: { result_3way: { model_type: "x", primary_metric_value: 0.5, metric_name: "m", selected_at: "now" } },
      },
    });
    render(
      <AppShell active="dashboard">
        <p>content</p>
      </AppShell>
    );
    expect(await screen.findByText(/league 2 · international 1/)).toBeInTheDocument();
  });

  it("shows a textual '-- stale' suffix (not just a color change) when data_freshness.is_stale is true", async () => {
    vi.mocked(getStatus).mockResolvedValue({
      data_freshness: { latest_match_date: "2026-05-24", days_since_update: 49, match_count: 3800, is_stale: true },
      model_status: { league: {}, international: {} },
    });
    render(
      <AppShell active="dashboard">
        <p>content</p>
      </AppShell>
    );
    expect(await screen.findByText(/2026-05-24/)).toBeInTheDocument();
    expect(screen.getByText(/-- stale/)).toBeInTheDocument();
  });

  it("does not show a per-league breakdown when by_league has 0 or 1 entries (single-competition, unchanged)", async () => {
    vi.mocked(getStatus).mockResolvedValue({
      data_freshness: {
        latest_match_date: "2026-07-20",
        days_since_update: 3,
        match_count: 100,
        is_stale: false,
        by_league: { E0: { latest_match_date: "2026-07-20", days_since_update: 3, match_count: 100, is_stale: false } },
      },
      model_status: { league: {}, international: {} },
    });
    render(
      <AppShell active="dashboard">
        <p>content</p>
      </AppShell>
    );
    await screen.findByText(/2026-07-20/);
    expect(screen.queryByText("E0")).not.toBeInTheDocument();
  });

  // W74: US#136 (engine-side) added a by_league breakdown to
  // get_data_freshness() precisely because a blended is_stale can mask one
  // competition going stale behind another staying fresh. This proves that
  // masking no longer happens in the UI once a second competition exists.
  it("shows a per-league staleness breakdown when by_league has more than one entry, even though the blended figure reads fresh", async () => {
    vi.mocked(getStatus).mockResolvedValue({
      data_freshness: {
        latest_match_date: "2026-07-20",
        days_since_update: 1,
        match_count: 3824,
        is_stale: false,
        by_league: {
          E0: { latest_match_date: "2026-05-24", days_since_update: 58, match_count: 3800, is_stale: true },
          SWE: { latest_match_date: "2026-07-20", days_since_update: 1, match_count: 24, is_stale: false },
        },
      },
      model_status: { league: {}, international: {} },
    });
    render(
      <AppShell active="dashboard">
        <p>content</p>
      </AppShell>
    );
    expect(await screen.findByText("E0")).toBeInTheDocument();
    expect(screen.getByText("SWE")).toBeInTheDocument();
    // The blended top line reads fresh, but E0's own row must still say stale.
    const e0Row = screen.getByText("E0").closest("div");
    expect(e0Row?.textContent).toMatch(/58d ago -- stale/);
    const sweRow = screen.getByText("SWE").closest("div");
    expect(sweRow?.textContent).not.toMatch(/stale/);
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
