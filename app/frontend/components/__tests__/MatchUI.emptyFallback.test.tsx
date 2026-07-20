/**
 * W46: empty-window fallback -- when DashboardPage's same-day fixture query
 * comes back empty (real off-season, or a past sandbox date), the Dashboard
 * should fall back to fetching and showing up to 10 of the nearest matches
 * going forward, clearly labeled as "next matches" rather than presented as
 * if they were today's. See documents/app_user_stories.md (W46) and the
 * unchanged non-empty case + existing cancellation-guard pattern in
 * MatchUI.race.test.tsx (W42), which the new fallback fetch must also honor.
 */
import { describe, expect, it, vi, beforeEach } from "vitest";
import { render, screen, waitFor } from "@testing-library/react";
import { DashboardPage } from "../MatchUI";
import { getFixtures, getSandboxStatus } from "@/lib/api";
import type { Fixture } from "@/lib/types";

vi.mock("@/lib/api");

/** A promise whose resolution is controlled from outside, so a test can
 * force a specific resolution order between in-flight fetches (same helper
 * as MatchUI.race.test.tsx). */
function deferred<T>() {
  let resolve!: (value: T) => void;
  const promise = new Promise<T>((res) => {
    resolve = res;
  });
  return { promise, resolve };
}

function fixture(id: string, utcDate: string): Fixture {
  return {
    match_id: id,
    utc_date: utcDate,
    status: "SCHEDULED",
    home_team: id,
    away_team: "Away",
    home_goals: null,
    away_goals: null,
  };
}

describe("Dashboard empty-window fallback (W46)", () => {
  beforeEach(() => {
    vi.mocked(getFixtures).mockReset();
    vi.mocked(getSandboxStatus).mockReset();
    // Keep useSandboxAsOf's asOf pinned to the real-clock Date it starts
    // with (sandbox_mode: false means the hook's setState call is skipped
    // entirely) so these tests exercise a single, non-racing effect run
    // unless a test deliberately sets up the race scenario itself.
    vi.mocked(getSandboxStatus).mockResolvedValue({ sandbox_mode: false, as_of: "" });
  });

  it("falls back to up to 10 nearest matches, sorted and clearly labeled, when today's query is empty", async () => {
    const today = new Date().toISOString().slice(0, 10);

    // 12 fixtures, deliberately out of order, so the fallback must both
    // trim to 10 and actually sort by kickoff to prove "nearest" is nearest.
    const unordered = Array.from({ length: 12 }, (_, i) => i).reverse();
    const fallbackFixtures = unordered.map((i) =>
      fixture(`match-${i}`, `2026-09-${String(i + 1).padStart(2, "0")}T15:00:00Z`)
    );

    vi.mocked(getFixtures).mockImplementation(async (from, to) => {
      if (from === today && to === today) return [];
      if (from === today) return fallbackFixtures;
      return [];
    });

    render(<DashboardPage />);

    // The fallback query must start from the requested date and go forward
    // (a distinct window from the same-day primary query) -- exactly 2
    // calls, and the second's `to` must be strictly after `today` (not
    // just any YYYY-MM-DD string, which "today" itself would also match).
    await waitFor(() => expect(getFixtures).toHaveBeenCalledWith(today, today));
    await waitFor(() => expect(getFixtures).toHaveBeenCalledTimes(2));
    const secondCall = vi.mocked(getFixtures).mock.calls[1];
    expect(secondCall[0]).toBe(today);
    expect(secondCall[1]).toMatch(/^\d{4}-\d{2}-\d{2}$/);
    expect(secondCall[1]).not.toBe(today);

    // Clearly labeled as "next matches", not today's.
    expect(await screen.findByText(/No E0 fixtures today.*next matches/)).toBeInTheDocument();

    // Trimmed to 10 (of the 12 available) and rendered nearest-first: the
    // earliest kickoffs (match-0..match-9) win over the two latest
    // (match-10, match-11).
    await waitFor(() => expect(screen.getByText("match-0")).toBeInTheDocument());
    expect(screen.getByText("match-9")).toBeInTheDocument();
    expect(screen.queryByText("match-10")).not.toBeInTheDocument();
    expect(screen.queryByText("match-11")).not.toBeInTheDocument();
  });

  it("does not fetch a fallback window at all when today's query already has fixtures", async () => {
    const today = new Date().toISOString().slice(0, 10);
    const todaysFixtures = [fixture("todays-match", `${today}T15:00:00Z`)];

    vi.mocked(getFixtures).mockImplementation(async (from, to) => {
      if (from === today && to === today) return todaysFixtures;
      return [];
    });

    render(<DashboardPage />);

    expect(await screen.findByText("todays-match")).toBeInTheDocument();

    // Give any (incorrect) fallback fetch a chance to fire, then confirm it
    // never did -- exactly one call, the primary same-day query.
    await new Promise((r) => setTimeout(r, 10));
    expect(getFixtures).toHaveBeenCalledTimes(1);
    expect(getFixtures).toHaveBeenCalledWith(today, today);

    // Unchanged rendering: no "next matches" language anywhere.
    expect(screen.queryByText(/next matches/i)).not.toBeInTheDocument();
  });

  it("renders a sensible, non-crashing empty state when the fallback window is itself empty", async () => {
    const today = new Date().toISOString().slice(0, 10);

    vi.mocked(getFixtures).mockResolvedValue([]);

    render(<DashboardPage />);

    await waitFor(() => expect(getFixtures).toHaveBeenCalledWith(today, today));
    await waitFor(() => expect(getFixtures).toHaveBeenCalledTimes(2));
    const secondCall = vi.mocked(getFixtures).mock.calls[1];
    expect(secondCall[0]).toBe(today);
    expect(secondCall[1]).not.toBe(today);

    expect(await screen.findByText("No E0 fixtures today.")).toBeInTheDocument();
    expect(screen.queryByText(/next matches/i)).not.toBeInTheDocument();
  });

  it("degrades to the plain empty state, not a misleading error, when the fallback fetch itself fails", async () => {
    // Code review finding: today's query already succeeded (0 results,
    // correctly recorded) by the time the fallback fetch runs -- a failure
    // in that second, independent fetch must not replace the already-
    // correct "no fixtures today" state with a top-level error implying
    // the whole page failed to load.
    const today = new Date().toISOString().slice(0, 10);

    vi.mocked(getFixtures)
      .mockResolvedValueOnce([]) // primary: today, genuinely empty
      .mockRejectedValueOnce(new Error("network blip")); // fallback: fails

    render(<DashboardPage />);

    await waitFor(() => expect(getFixtures).toHaveBeenCalledWith(today, today));
    await waitFor(() => expect(getFixtures).toHaveBeenCalledTimes(2));

    expect(await screen.findByText("No E0 fixtures today.")).toBeInTheDocument();
    expect(screen.queryByText(/could not load/i)).not.toBeInTheDocument();
    expect(screen.queryByText(/next matches/i)).not.toBeInTheDocument();
  });

  it("a stale fallback response from a superseded (cancelled) run does not overwrite the correct one", async () => {
    // Mirrors MatchUI.race.test.tsx (W42): useSandboxAsOf resolves
    // asynchronously, so DashboardPage's effect can fire twice in quick
    // succession (once for the real-clock date, once for the corrected
    // sandbox date). This test proves the *fallback* fetch -- itself a
    // second async call within a single effect run -- is guarded by the
    // same `cancelled` flag as the primary fetch, not just the primary.
    const todayReal = new Date().toISOString().slice(0, 10);
    const sandboxDate = "2025-03-05";

    const realPrimary = deferred<Fixture[]>();
    const realFallback = deferred<Fixture[]>();
    const sandboxPrimary = deferred<Fixture[]>();
    const sandboxFallback = deferred<Fixture[]>();
    const sandboxStatus = deferred<{ sandbox_mode: boolean; as_of: string }>();

    vi.mocked(getSandboxStatus).mockImplementation(() => sandboxStatus.promise);
    vi.mocked(getFixtures).mockImplementation((from, to) => {
      if (from === todayReal && to === todayReal) return realPrimary.promise;
      if (from === todayReal) return realFallback.promise;
      if (from === sandboxDate && to === sandboxDate) return sandboxPrimary.promise;
      if (from === sandboxDate) return sandboxFallback.promise;
      return Promise.resolve([]);
    });

    render(<DashboardPage />);

    // Run #1 (real-clock date): primary resolves empty, triggering its
    // fallback fetch.
    await waitFor(() => expect(getFixtures).toHaveBeenCalledWith(todayReal, todayReal));
    realPrimary.resolve([]);
    await waitFor(() => expect(getFixtures).toHaveBeenCalledWith(todayReal, expect.any(String)));

    // Now the sandbox status resolves -- asOf changes, cancelling run #1
    // (its fallback fetch is still in flight) and starting run #2 for the
    // sandbox date.
    sandboxStatus.resolve({ sandbox_mode: true, as_of: sandboxDate });
    await waitFor(() => expect(getFixtures).toHaveBeenCalledWith(sandboxDate, sandboxDate));
    sandboxPrimary.resolve([]);
    await waitFor(() => expect(getFixtures).toHaveBeenCalledWith(sandboxDate, expect.any(String)));

    // Run #2's fallback resolves with the correct matches first.
    sandboxFallback.resolve([fixture("correct-next-match", "2025-03-10T15:00:00Z")]);
    expect(await screen.findByText("correct-next-match")).toBeInTheDocument();

    // The stale run #1 fallback resolves *after* the correct one -- exactly
    // the out-of-order ordering W42 guards against, now one level deeper.
    realFallback.resolve([fixture("stale-next-match", "2026-09-01T15:00:00Z")]);
    await new Promise((r) => setTimeout(r, 10));

    expect(screen.queryByText("stale-next-match")).not.toBeInTheDocument();
    expect(screen.getByText("correct-next-match")).toBeInTheDocument();
  });
});
