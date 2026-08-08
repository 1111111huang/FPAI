/**
 * W42: regression test for a real fixture-fetch race condition, found by
 * actually launching the app with SANDBOX_DATE=2025-03-05.
 *
 * useSandboxAsOf() resolves `asOf` asynchronously -- the real browser
 * new Date() first, then the corrected sandbox date once GET
 * /api/sandbox/status returns -- so DashboardPage/MatchExplorerPage's
 * `useEffect(() => { load(); }, [asOf])` fires load() twice in quick
 * succession: once for the wrong real-clock date, once for the correct
 * sandbox date. If the *first* (stale, real-clock) request's response
 * lands *after* the second (correct) one -- ordinary network jitter, not
 * something these tests need StrictMode to reproduce -- the stale response
 * silently overwrites the correct state.
 *
 * W38's date-boundary tests (MatchUI.dateboundary.test.tsx) don't catch
 * this: their mocked getFixtures always resolves deterministically/in
 * order, so they never exercise out-of-order resolution. This file
 * deliberately resolves the earlier request's promise *after* the later
 * one to prove the race is closed.
 */
import { describe, expect, it, vi, beforeEach } from "vitest";
import { render, screen, waitFor } from "@testing-library/react";
import { DashboardPage, MatchExplorerPage } from "../MatchUI";
import { getFixtures, getSandboxStatus } from "@/lib/api";
import type { Fixture } from "@/lib/types";

vi.mock("@/lib/api");

/** A promise whose resolution is controlled from outside, so the test can
 * force a specific out-of-order resolution sequence between two in-flight
 * fetches. */
function deferred<T>() {
  let resolve!: (value: T) => void;
  const promise = new Promise<T>((res) => {
    resolve = res;
  });
  return { promise, resolve };
}

function fixture(id: string): Fixture {
  return {
    match_id: id,
    utc_date: "2026-03-05T15:00:00Z",
    status: "SCHEDULED",
    home_team: id,
    away_team: "Away",
    home_goals: null,
    away_goals: null,
  };
}

describe("fixture-fetch race guard (W42)", () => {
  beforeEach(() => {
    vi.mocked(getFixtures).mockReset();
    vi.mocked(getSandboxStatus).mockReset();
  });

  it("Dashboard: a stale real-clock response landing after the correct sandbox response does not overwrite it", async () => {
    const today = new Date().toISOString().slice(0, 10);
    const sandboxDate = "2025-03-05";

    const staleFixtures = [fixture("real-clock-fixture")];
    const correctFixtures: Fixture[] = [];

    const stale = deferred<Fixture[]>();
    const correct = deferred<Fixture[]>();

    vi.mocked(getFixtures).mockImplementation((from) => {
      if (from === today) return stale.promise;
      if (from === sandboxDate) return correct.promise;
      return Promise.resolve([]);
    });
    vi.mocked(getSandboxStatus).mockResolvedValue({ sandbox_mode: true, as_of: sandboxDate });

    render(<DashboardPage />);

    // Both the stale (real-clock) and correct (sandbox) requests must have
    // been fired -- this is the double-fetch this bug depends on. Dashboard
    // now always queries a 90-day-forward window (not just the exact day),
    // so match on `from` only -- the exact `to` isn't this test's concern.
    await waitFor(() => expect(getFixtures).toHaveBeenCalledWith(today, expect.any(String)));
    await waitFor(() => expect(getFixtures).toHaveBeenCalledWith(sandboxDate, expect.any(String)));

    // Resolve out of order: the correct (later-fired) request resolves
    // FIRST, then the stale (earlier-fired) request resolves AFTER it --
    // exactly the ordering captured live via Playwright.
    correct.resolve(correctFixtures);
    await waitFor(() => expect(screen.getByText("No upcoming fixtures.")).toBeInTheDocument());

    stale.resolve(staleFixtures);

    // Give the stale resolution a chance to (wrongly) clobber state if the
    // race guard is missing, then assert it didn't.
    await new Promise((r) => setTimeout(r, 10));
    expect(screen.queryByText("real-clock-fixture")).not.toBeInTheDocument();
    expect(screen.getByText("No upcoming fixtures.")).toBeInTheDocument();
  });

  it("Match Explorer: a stale real-clock response landing after the correct sandbox response does not overwrite it", async () => {
    const today = new Date().toISOString().slice(0, 10);
    const sandboxDate = "2025-03-05";
    const sandboxWindowEnd = "2025-06-03"; // 2025-03-05 + 90 days

    const staleFixtures = [fixture("real-clock-fixture")];
    const correctFixtures: Fixture[] = [];

    const stale = deferred<Fixture[]>();
    const correct = deferred<Fixture[]>();

    vi.mocked(getFixtures).mockImplementation((from) => {
      if (from === today) return stale.promise;
      if (from === sandboxDate) return correct.promise;
      return Promise.resolve([]);
    });
    vi.mocked(getSandboxStatus).mockResolvedValue({ sandbox_mode: true, as_of: sandboxDate });

    render(<MatchExplorerPage />);

    await waitFor(() => expect(getFixtures).toHaveBeenCalledWith(today, expect.any(String)));
    await waitFor(() => expect(getFixtures).toHaveBeenCalledWith(sandboxDate, sandboxWindowEnd));

    correct.resolve(correctFixtures);
    await waitFor(() => expect(screen.getByText("No matches found.")).toBeInTheDocument());

    stale.resolve(staleFixtures);

    await new Promise((r) => setTimeout(r, 10));
    expect(screen.queryByText("real-clock-fixture")).not.toBeInTheDocument();
    expect(screen.getByText("No matches found.")).toBeInTheDocument();
  });
});
