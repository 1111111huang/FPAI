import { render, screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { describe, expect, it, vi, beforeEach } from "vitest";
import { AppShell } from "../AppShell";

vi.mock("@/lib/api", () => ({
  getFixtures: vi.fn(),
  getSandboxStatus: vi.fn(),
}));

import { getFixtures, getSandboxStatus } from "@/lib/api";

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
    vi.mocked(getFixtures).mockReset();
    vi.mocked(getSandboxStatus).mockReset();
    vi.mocked(getSandboxStatus).mockResolvedValue({ sandbox_mode: false, as_of: null });
  });

  it("renders nav links and page content", () => {
    render(
      <AppShell active="dashboard">
        <p>page content</p>
      </AppShell>
    );
    // W114: each nav label now renders twice by design -- once in the
    // desktop sidebar, once in the mobile bottom tab bar (CSS decides which
    // is visible; jsdom renders both regardless of viewport).
    expect(screen.getAllByText("Daily Edges")).toHaveLength(2);
    expect(screen.getAllByText("All Matches")).toHaveLength(2);
    expect(screen.getByText("page content")).toBeInTheDocument();
  });

  it("BUG-046: keeps top padding off <main> itself -- it's the lg:overflow-y-auto scroll container, so padding there sits inside the scrollport, letting scrolled content slide up through the gap above a page's sticky header instead of being hidden behind it", () => {
    render(
      <AppShell active="dashboard">
        <p>page content</p>
      </AppShell>
    );
    const main = screen.getByText("page content").closest("main");
    expect(main).not.toHaveClass("pt-8");
    // The padding must still exist somewhere between <main> and the content,
    // just on a non-scrolling wrapper instead.
    expect(main?.firstElementChild).toHaveClass("pt-8");
  });

  it("renders an optional railTrigger next to the search bar, mobile-only", () => {
    const { rerender } = render(
      <AppShell active="dashboard">
        <p>page content</p>
      </AppShell>
    );
    expect(screen.queryByText("trigger")).not.toBeInTheDocument();

    rerender(
      <AppShell active="dashboard" railTrigger={<span>trigger</span>}>
        <p>page content</p>
      </AppShell>
    );
    const trigger = screen.getByText("trigger");
    expect(trigger).toBeInTheDocument();
    // lg:hidden -- only meant for the mobile top bar, not shown alongside
    // the always-visible desktop sidebar/rail.
    expect(trigger.closest(".lg\\:hidden")).not.toBeNull();
  });

  it("rebrand: shows 'Oddsey' (not 'FPAI') in the sidebar", () => {
    render(
      <AppShell active="dashboard">
        <p>page content</p>
      </AppShell>
    );
    // "Oddsey" now renders twice by design -- once in the desktop sidebar,
    // once in the mobile collapsed header bar (CSS decides which is
    // visible; jsdom renders both regardless of viewport).
    expect(screen.getAllByText("Oddsey")).toHaveLength(2);
    expect(screen.queryByText("FPAI")).not.toBeInTheDocument();
  });

  it("W133: no longer shows the 'Edge Engine' tagline under the logo -- direct feedback that it wasn't needed", () => {
    render(
      <AppShell active="dashboard">
        <p>page content</p>
      </AppShell>
    );
    expect(screen.queryByText("Edge Engine")).not.toBeInTheDocument();
  });

  it("collapses the whole left panel (menu/footer) behind a mobile drawer, opening as an overlay", async () => {
    // The brand block ("Oddsey") is shared between the desktop <aside> and
    // the mobile drawer -- both are always in the DOM (CSS-hidden on the
    // wrong viewport, jsdom doesn't evaluate that), so opening the drawer
    // adds a third occurrence on top of the two the collapsed state already
    // has (desktop sidebar + mobile compact header).
    const user = userEvent.setup();
    render(
      <AppShell active="dashboard">
        <p>page content</p>
      </AppShell>
    );

    // Collapsed by default -- drawer isn't rendered at all.
    expect(screen.queryByRole("button", { name: "Close menu" })).not.toBeInTheDocument();
    expect(screen.getAllByText("Oddsey")).toHaveLength(2);

    await user.click(screen.getByRole("button", { name: "Open menu" }));
    expect(screen.getByRole("button", { name: "Close menu" })).toBeInTheDocument();
    expect(screen.getAllByText("Oddsey")).toHaveLength(3);
    // Overlay, not a replacement -- page content stays mounted underneath.
    expect(screen.getByText("page content")).toBeInTheDocument();

    await user.click(screen.getByRole("button", { name: "Close menu" }));
    expect(screen.queryByRole("button", { name: "Close menu" })).not.toBeInTheDocument();
    expect(screen.getAllByText("Oddsey")).toHaveLength(2);
  });

  it("mockup point 2: each desktop nav link renders its own icon", () => {
    const { container } = render(
      <AppShell active="dashboard">
        <p>page content</p>
      </AppShell>
    );
    // Desktop sidebar's <nav> lives inside the (now hidden lg:flex) <aside>
    // -- vs. the mobile bottom-tab-bar's <nav>, which already had icons
    // before this change and isn't inside an <aside> -- scope the query so
    // this doesn't pass by accident.
    const desktopNav = container.querySelector("aside nav");
    expect(desktopNav?.querySelectorAll("svg").length).toBe(2);
  });

  it("does not render a Bets nav link -- feature not ready yet (hidden 2026-08-13)", () => {
    render(
      <AppShell active="dashboard">
        <p>page content</p>
      </AppShell>
    );
    expect(screen.queryByText("Bets")).not.toBeInTheDocument();
  });

  it("W114: renders a bottom tab bar with the same nav items, for small screens", () => {
    const { container } = render(
      <AppShell active="dashboard">
        <p>page content</p>
      </AppShell>
    );
    const bottomNav = container.querySelector("nav.fixed");
    expect(bottomNav).not.toBeNull();
    expect(bottomNav?.querySelectorAll("a")).toHaveLength(2);
  });

  it("does not fetch fixtures on mount -- only once the search input is focused", async () => {
    render(
      <AppShell active="dashboard">
        <p>content</p>
      </AppShell>
    );
    expect(getFixtures).not.toHaveBeenCalled();

    const user = userEvent.setup();
    await user.click(screen.getByPlaceholderText("Search fixtures, teams…"));
    await waitFor(() => expect(getFixtures).toHaveBeenCalledTimes(1));
  });

  it("filters and links search results to Match Analysis, carrying league", async () => {
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
