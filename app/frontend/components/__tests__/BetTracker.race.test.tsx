/**
 * W42: regression test for ManualBetForm's fixture-search race, the same
 * class of bug as MatchUI.race.test.tsx (see that file's header comment
 * for the full background). useSandboxAsOf() resolves `asOf` asynchronously
 * -- real browser clock first, then the corrected sandbox date -- so
 * ManualBetForm's `useEffect(() => { getFixtures(...).then(setFixtures) },
 * [asOf])` fires twice, and a stale (real-clock) response landing after the
 * correct (sandbox) one must not silently overwrite the fixture list a user
 * searches against.
 */
import { describe, expect, it, vi, beforeEach } from "vitest";
import { act, render, screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { BetTrackerPage } from "../BetTracker";
import { getBets, getBetStats, getFixtures, getSandboxStatus } from "@/lib/api";
import type { Fixture } from "@/lib/types";

vi.mock("@/lib/api");

function deferred<T>() {
  let resolve!: (value: T) => void;
  const promise = new Promise<T>((res) => {
    resolve = res;
  });
  return { promise, resolve };
}

function fixture(id: string, homeTeam: string): Fixture {
  return {
    match_id: id,
    utc_date: "2026-03-05T15:00:00Z",
    status: "SCHEDULED",
    home_team: homeTeam,
    away_team: "Away",
    home_goals: null,
    away_goals: null,
  };
}

describe("ManualBetForm fixture-search race guard (W42)", () => {
  beforeEach(() => {
    vi.mocked(getFixtures).mockReset();
    vi.mocked(getSandboxStatus).mockReset();
    vi.mocked(getBets).mockReset();
    vi.mocked(getBetStats).mockReset();
    vi.mocked(getBets).mockResolvedValue([]);
    vi.mocked(getBetStats).mockResolvedValue({
      bets_settled: 0, bets_open: 0, bets_won: 0, roi: 0, hit_rate: 0,
      total_staked: 0, total_profit: 0, max_drawdown: 0,
      starting_bankroll: 0, current_bankroll: 0,
    });
  });

  it("a stale real-clock fixture list landing after the correct sandbox one is not searchable", async () => {
    const today = new Date().toISOString().slice(0, 10);
    const sandboxDate = "2025-03-05";

    const staleFixtures = [fixture("real-1", "StaleUnitedFC")];
    const correctFixtures: Fixture[] = [];

    const stale = deferred<Fixture[]>();
    const correct = deferred<Fixture[]>();

    vi.mocked(getFixtures).mockImplementation((from) => {
      if (from === today) return stale.promise;
      if (from === sandboxDate) return correct.promise;
      return Promise.resolve([]);
    });
    vi.mocked(getSandboxStatus).mockResolvedValue({ sandbox_mode: true, as_of: sandboxDate });

    const user = userEvent.setup();
    render(<BetTrackerPage />);

    await waitFor(() => expect(getFixtures).toHaveBeenCalledWith(today, expect.any(String)));
    await waitFor(() => expect(getFixtures).toHaveBeenCalledWith(sandboxDate, expect.any(String)));

    // Resolve out of order: correct (sandbox, fired second) resolves first,
    // stale (real-clock, fired first) resolves after it.
    await act(async () => {
      correct.resolve(correctFixtures);
      await new Promise((r) => setTimeout(r, 10));
    });
    await act(async () => {
      stale.resolve(staleFixtures);
      await new Promise((r) => setTimeout(r, 10));
    });

    await user.type(
      screen.getByPlaceholderText("Search a real fixture by team name…"),
      "Stale"
    );

    expect(screen.queryByText(/StaleUnitedFC/)).not.toBeInTheDocument();
  });
});
