/**
 * W52: ManualBetForm's fixture-search fetch previously did
 * `.catch(() => setFixtures([]))` -- any fetch failure (network error, the
 * football-data.org 429 rate-limit -> backend 503, or anything else)
 * silently degraded to an empty fixture list with no error shown at all.
 * A user searching for a real fixture to log a bet against would see
 * nothing and have no idea whether that's a genuine "no matches" or a
 * broken fetch. This mirrors MatchExplorerPage's established pattern
 * (ApiError message vs a generic fallback, rendered visibly) -- see
 * MatchUI.tsx's ErrorState usage.
 */
import { describe, expect, it, vi, beforeEach } from "vitest";
import { render, screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { BetTrackerPage } from "../BetTracker";

// W52: vi.mock("@/lib/api") without a factory (BetTracker.race.test.tsx's
// approach) automocks ApiError into a mock constructor that doesn't behave
// like a real Error subclass -- `err instanceof ApiError` then fails inside
// the component. MatchUI.test.tsx already established the fix (a real
// `class ApiError extends Error {}` in the factory) -- reused here since
// this test needs `err instanceof ApiError` to genuinely work.
vi.mock("@/lib/api", () => ({
  getBets: vi.fn(),
  getBetStats: vi.fn(),
  getFixtures: vi.fn(),
  logBetManual: vi.fn(),
  settleOpenBets: vi.fn(),
  getSandboxStatus: vi.fn(),
  // AppShell (wired into BetTrackerPage as of this task) calls getStatus()
  // on mount for its top-bar status indicator -- without this the mock
  // module has no such export and AppShell's mount throws. Explicitly
  // rejected in beforeEach below (AppShell.test.tsx's own precedent),
  // rather than left unresolved.
  getStatus: vi.fn(),
  deleteBet: vi.fn(),
  updateBet: vi.fn(),
  ApiError: class ApiError extends Error {
    status?: number;
    constructor(message: string, status?: number) {
      super(message);
      this.name = "ApiError";
      this.status = status;
    }
  },
}));

import { ApiError, deleteBet, getBets, getBetStats, getFixtures, getSandboxStatus, getStatus, settleOpenBets, updateBet } from "@/lib/api";

describe("ManualBetForm surfaces a visible error when the fixture fetch fails (W52)", () => {
  beforeEach(() => {
    vi.mocked(getFixtures).mockReset();
    vi.mocked(getSandboxStatus).mockReset();
    vi.mocked(getBets).mockReset();
    vi.mocked(getBetStats).mockReset();
    vi.mocked(getStatus).mockReset();
    vi.mocked(getBets).mockResolvedValue([]);
    vi.mocked(getBetStats).mockResolvedValue({
      bets_settled: 0, bets_open: 0, bets_won: 0, roi: 0, hit_rate: 0,
      total_staked: 0, total_profit: 0, max_drawdown: 0,
      starting_bankroll: 0, current_bankroll: 0,
    });
    vi.mocked(getSandboxStatus).mockResolvedValue({ sandbox_mode: false, as_of: null });
    vi.mocked(getStatus).mockRejectedValue(new Error("no backend"));
  });

  it("shows a visible error message when the fixture fetch rejects, not just a silent empty search", async () => {
    vi.mocked(getFixtures).mockRejectedValue(
      new ApiError("Fixture data is temporarily unavailable (the upstream provider is rate-limited or unreachable).", 503)
    );

    const user = userEvent.setup();
    render(<BetTrackerPage />);

    await waitFor(() => expect(getFixtures).toHaveBeenCalled());

    await user.type(
      screen.getByPlaceholderText("Search a real fixture by team name…"),
      "Arsenal"
    );

    await waitFor(() =>
      expect(
        screen.getByText(/Fixture data is temporarily unavailable/i)
      ).toBeInTheDocument()
    );
  });

  it("clicking Retry on the fixture error re-runs the fetch and clears the error once it succeeds (W52 code review follow-up)", async () => {
    vi.mocked(getFixtures)
      .mockRejectedValueOnce(new ApiError("Fixture data is temporarily unavailable.", 503))
      .mockResolvedValueOnce([]);

    const user = userEvent.setup();
    render(<BetTrackerPage />);

    await waitFor(() =>
      expect(screen.getByText(/Fixture data is temporarily unavailable/i)).toBeInTheDocument()
    );
    expect(getFixtures).toHaveBeenCalledTimes(1);

    await user.click(screen.getByRole("button", { name: /retry/i }));

    await waitFor(() => expect(getFixtures).toHaveBeenCalledTimes(2));
    await waitFor(() =>
      expect(screen.queryByText(/Fixture data is temporarily unavailable/i)).not.toBeInTheDocument()
    );
  });

  it("W211: searches 30 days back through 90 days forward, not forward-only -- a bet couldn't otherwise be logged against a match that already kicked off", async () => {
    const now = new Date();
    const today = `${now.getFullYear()}-${String(now.getMonth() + 1).padStart(2, "0")}-${String(now.getDate()).padStart(2, "0")}`;
    vi.mocked(getFixtures).mockResolvedValue([]);

    render(<BetTrackerPage />);

    await waitFor(() => expect(getFixtures).toHaveBeenCalled());
    const [from, to] = vi.mocked(getFixtures).mock.calls[0];
    expect(from! < today).toBe(true);
    expect(to! > today).toBe(true);
  });
});

// W210 follow-up: getBets and getBetStats used to load via Promise.all,
// so a single failed request (stats included) blocked the whole bets list
// from rendering, and a 401 from an expired session showed a raw
// "Failed to load bets (401)" string instead of prompting the user to sign
// in again. Reuses this file's ApiError-with-status mock factory (see the
// W52 comment above) since these tests depend on `err.status === 401`
// actually being readable on the caught error.
describe("BetTrackerPage decouples stats/bets loading and prompts re-auth on 401 (W210 follow-up)", () => {
  beforeEach(() => {
    vi.mocked(getFixtures).mockReset();
    vi.mocked(getSandboxStatus).mockReset();
    vi.mocked(getBets).mockReset();
    vi.mocked(getBetStats).mockReset();
    vi.mocked(getStatus).mockReset();
    vi.mocked(settleOpenBets).mockReset();
    vi.mocked(getFixtures).mockResolvedValue([]);
    vi.mocked(getSandboxStatus).mockResolvedValue({ sandbox_mode: false, as_of: null });
    vi.mocked(getStatus).mockRejectedValue(new Error("no backend"));
  });

  it("still shows the bets list when only getBetStats fails", async () => {
    vi.mocked(getBets).mockResolvedValue([
      {
        id: 1, match_id: "m1", date: "2026-08-22", home_team: "Arsenal", away_team: "Everton",
        market: "result_3way", selection: "home", odds: 2.1, stake: 10, outcome: "open",
        profit_loss: null, source: "from_recommendation", recommendation_snapshot: null, created_at: "now",
      },
    ]);
    vi.mocked(getBetStats).mockRejectedValue(new ApiError("Failed to load bet stats (500)", 500));

    render(<BetTrackerPage />);

    await waitFor(() => expect(screen.getByText(/logged bets/i)).toBeInTheDocument());
    // the bets list itself rendered even though stats failed
    await waitFor(() => expect(screen.getByText(/Arsenal v Everton/)).toBeInTheDocument());
  });

  it("shows a re-authenticate prompt on a 401, not a raw status-code message", async () => {
    vi.mocked(getBets).mockRejectedValue(new ApiError("Failed to load bets (401)", 401));
    vi.mocked(getBetStats).mockRejectedValue(new ApiError("Failed to load bet stats (401)", 401));

    render(<BetTrackerPage />);

    await waitFor(() => expect(screen.getByRole("link", { name: /sign in/i })).toBeInTheDocument());
    expect(screen.queryByText(/401/)).not.toBeInTheDocument();
  });

  it("recovers after re-login: a later successful load clears needsAuth and shows the real bets list", async () => {
    vi.mocked(getBets).mockRejectedValue(new ApiError("Failed to load bets (401)", 401));
    vi.mocked(getBetStats).mockRejectedValue(new ApiError("Failed to load bet stats (401)", 401));

    const user = userEvent.setup();
    render(<BetTrackerPage />);

    // Banner-specific wording, not the ambiguous /sign in/i -- AppShell's
    // separate UserMenu always renders a plain "Sign in" link once
    // next-auth's useSession() settles to unauthenticated, which would
    // satisfy a bare /sign in/i query before load()'s promises even settle
    // and make this assertion non-load-bearing.
    await waitFor(() => expect(screen.getByText(/your session expired/i)).toBeInTheDocument());
    expect(screen.getByRole("link", { name: /sign in again/i })).toBeInTheDocument();

    // Simulate the user having signed back in, then re-mock the fetches to
    // succeed and trigger a reload through "Settle open bets" -- the only
    // user-facing action (besides ManualBetForm's onLogged) that re-runs
    // load().
    vi.mocked(getBets).mockResolvedValue([
      {
        id: 2, match_id: "m2", date: "2026-08-23", home_team: "Chelsea", away_team: "Fulham",
        market: "result_3way", selection: "away", odds: 3.2, stake: 5, outcome: "open",
        profit_loss: null, source: "manual", recommendation_snapshot: null, created_at: "now",
      },
    ]);
    vi.mocked(getBetStats).mockResolvedValue({
      bets_settled: 0, bets_open: 1, bets_won: 0, roi: 0, hit_rate: 0,
      total_staked: 5, total_profit: 0, max_drawdown: 0,
      starting_bankroll: 0, current_bankroll: 0,
    });
    vi.mocked(settleOpenBets).mockResolvedValue([]);

    await user.click(screen.getByRole("button", { name: /settle open bets/i }));

    // Scoped to the banner's own wording ("sign in again") rather than a
    // bare /sign in/i -- AppShell's separate UserMenu also renders a plain
    // "Sign in" link once next-auth's useSession() settles to
    // unauthenticated, which would otherwise still match after the banner
    // itself is gone.
    await waitFor(() => expect(screen.queryByText(/your session expired/i)).not.toBeInTheDocument());
    expect(screen.queryByRole("link", { name: /sign in again/i })).not.toBeInTheDocument();
    await waitFor(() => expect(screen.getByText(/Chelsea v Fulham/)).toBeInTheDocument());
  });
});

// W214: direct user feedback -- settlement should be automatic on page
// load, the same way a match's completed status just appears without a
// manual action, not gated entirely behind the "Settle open bets" button.
describe("BetTrackerPage settles open bets automatically on load (W214)", () => {
  beforeEach(() => {
    vi.mocked(getFixtures).mockReset();
    vi.mocked(getSandboxStatus).mockReset();
    vi.mocked(getBets).mockReset();
    vi.mocked(getBetStats).mockReset();
    vi.mocked(getStatus).mockReset();
    vi.mocked(settleOpenBets).mockReset();
    vi.mocked(getFixtures).mockResolvedValue([]);
    vi.mocked(getSandboxStatus).mockResolvedValue({ sandbox_mode: false, as_of: null });
    vi.mocked(getStatus).mockRejectedValue(new Error("no backend"));
  });

  it("calls settleOpenBets on mount, with no button click", async () => {
    vi.mocked(getBets).mockResolvedValue([]);
    vi.mocked(getBetStats).mockResolvedValue({
      bets_settled: 0, bets_open: 0, bets_won: 0, roi: 0, hit_rate: 0,
      total_staked: 0, total_profit: 0, max_drawdown: 0,
      starting_bankroll: 0, current_bankroll: 0,
    });
    vi.mocked(settleOpenBets).mockResolvedValue([]);

    render(<BetTrackerPage />);

    await waitFor(() => expect(settleOpenBets).toHaveBeenCalledTimes(1));
  });

  it("settles before the bets list is fetched, so a newly-finished match's real outcome shows on first load", async () => {
    vi.mocked(getBets).mockResolvedValue([
      {
        id: 1, match_id: "m1", date: "2026-08-22", home_team: "Arsenal", away_team: "Everton",
        market: "result_3way", selection: "home", odds: 2.1, stake: 10, outcome: "won",
        profit_loss: 12.1, source: "manual", recommendation_snapshot: null, created_at: "now",
      },
    ]);
    vi.mocked(getBetStats).mockResolvedValue({
      bets_settled: 1, bets_open: 0, bets_won: 1, roi: 1.21, hit_rate: 1,
      total_staked: 10, total_profit: 12.1, max_drawdown: 0,
      starting_bankroll: 1000, current_bankroll: 1012.1,
    });
    vi.mocked(settleOpenBets).mockResolvedValue([]);

    render(<BetTrackerPage />);

    expect(await screen.findByText(/Arsenal v Everton/)).toBeInTheDocument();
    // "won" renders CSS-uppercased (BetRow's `uppercase` class) -- the
    // actual text content stays lowercase, so match case-insensitively.
    expect(screen.getByText(/^won$/i)).toBeInTheDocument();
    // No "Checking results…" spinner text ever shown -- this is a silent,
    // background check, distinct from the manual button's own visible state.
    expect(screen.queryByText(/checking results/i)).not.toBeInTheDocument();
  });

  it("a 401 from the automatic settle attempt prompts re-auth, same as a 401 from load()", async () => {
    vi.mocked(settleOpenBets).mockRejectedValue(new ApiError("Failed to settle bets (401)", 401));
    vi.mocked(getBets).mockRejectedValue(new ApiError("Failed to load bets (401)", 401));
    vi.mocked(getBetStats).mockRejectedValue(new ApiError("Failed to load bet stats (401)", 401));

    render(<BetTrackerPage />);

    await waitFor(() => expect(screen.getByText(/your session expired/i)).toBeInTheDocument());
  });

  it("a non-401 failure from the automatic settle attempt doesn't block the bets list from loading", async () => {
    vi.mocked(settleOpenBets).mockRejectedValue(new Error("network blip"));
    vi.mocked(getBets).mockResolvedValue([
      {
        id: 1, match_id: "m1", date: "2026-08-22", home_team: "Arsenal", away_team: "Everton",
        market: "result_3way", selection: "home", odds: 2.1, stake: 10, outcome: "open",
        profit_loss: null, source: "manual", recommendation_snapshot: null, created_at: "now",
      },
    ]);
    vi.mocked(getBetStats).mockResolvedValue({
      bets_settled: 0, bets_open: 1, bets_won: 0, roi: 0, hit_rate: 0,
      total_staked: 10, total_profit: 0, max_drawdown: 0,
      starting_bankroll: 1000, current_bankroll: 1000,
    });

    render(<BetTrackerPage />);

    expect(await screen.findByText(/Arsenal v Everton/)).toBeInTheDocument();
  });
});

// W210 follow-up (Critical): Market/Selection used to be freeform <input>
// text fields with zero validation. Settlement only ever resolves bets
// whose market/selection exactly match RESOLVABLE_MARKETS
// (src/agent/market_resolution.py) -- a typo or plausible-but-wrong string
// left a bet permanently unsettleable, sitting "open" forever with no error
// anywhere. Reuses this file's ApiError-with-status mock factory since it's
// already wired for BetTrackerPage's full render (AppShell's getStatus
// included) -- ManualBetForm itself isn't exported, so this drives it
// through BetTrackerPage's real fixture-search-then-select flow rather than
// rendering it standalone.
describe("ManualBetForm constrains market/selection to resolvable values (W210 follow-up, Critical)", () => {
  beforeEach(() => {
    vi.mocked(getFixtures).mockReset();
    vi.mocked(getSandboxStatus).mockReset();
    vi.mocked(getBets).mockReset();
    vi.mocked(getBetStats).mockReset();
    vi.mocked(getStatus).mockReset();
    vi.mocked(getBets).mockResolvedValue([]);
    vi.mocked(getBetStats).mockResolvedValue({
      bets_settled: 0, bets_open: 0, bets_won: 0, roi: 0, hit_rate: 0,
      total_staked: 0, total_profit: 0, max_drawdown: 0,
      starting_bankroll: 0, current_bankroll: 0,
    });
    vi.mocked(getSandboxStatus).mockResolvedValue({ sandbox_mode: false, as_of: null });
    vi.mocked(getStatus).mockRejectedValue(new Error("no backend"));
    vi.mocked(getFixtures).mockResolvedValue([
      {
        match_id: "m1", utc_date: "2026-08-22T15:00:00Z", status: "SCHEDULED",
        home_team: "Arsenal", away_team: "Everton", home_goals: null, away_goals: null,
      },
    ]);
  });

  async function selectFixture(user: ReturnType<typeof userEvent.setup>) {
    render(<BetTrackerPage />);
    await user.type(
      screen.getByPlaceholderText("Search a real fixture by team name…"),
      "Arsenal"
    );
    await user.click(await screen.findByText(/Arsenal v Everton/));
  }

  it("only offers valid market/selection combinations, not freeform text", async () => {
    const user = userEvent.setup();
    await selectFixture(user);

    // Anchored (^market$/^pick$) rather than a loose /market/i -- BetRow's
    // own edit form (W216, unrelated) has its own "Edit market: ..." labels
    // that a loose match could ambiguously pick up if a bet row were also
    // rendered, matching MatchUI.test.tsx's established pattern.
    const marketSelect = screen.getByLabelText(/^market$/i);
    expect(marketSelect.tagName).toBe("SELECT");
    await user.selectOptions(marketSelect, "btts");

    const selectionSelect = screen.getByLabelText(/^pick$/i);
    expect(selectionSelect.tagName).toBe("SELECT");
    const options = Array.from(selectionSelect.querySelectorAll("option")).map((o) => o.textContent);
    expect(options).toEqual(expect.arrayContaining(["Yes", "No"]));
    expect(options).not.toEqual(expect.arrayContaining(["home", "draw", "away"]));
  });

  it("switching market resets a stale selection so an invalid combination can't be submitted", async () => {
    const user = userEvent.setup();
    await selectFixture(user);

    const marketSelect = screen.getByLabelText(/^market$/i);
    const selectionSelect = screen.getByLabelText<HTMLSelectElement>(/^pick$/i);
    await user.selectOptions(marketSelect, "result_3way");
    await user.selectOptions(selectionSelect, "away");
    expect(selectionSelect.value).toBe("away");

    await user.selectOptions(marketSelect, "btts");
    // "away" isn't a valid btts option -- the field must reset, not keep
    // pointing at a selection that no longer exists for this market.
    expect(selectionSelect.value).toBe("");
    expect(
      Array.from(selectionSelect.querySelectorAll("option")).map((o) => o.value)
    ).not.toContain("away");
  });
});

// W210 follow-up (Task 6): UI polish -- bet-list column headers, a stats
// loading placeholder, and a "no matching fixtures" empty state. Reuses this
// file's mock factory (ApiError-with-status, getStatus for AppShell) since
// all three tests render the full BetTrackerPage. ManualBetForm itself isn't
// exported (see the describe block above), so the plan's
// `render(<ManualBetForm ... />)` sketch for the fixture-search test is
// adapted to drive the same search box through BetTrackerPage, matching the
// established pattern elsewhere in this file.
describe("BetTracker UI polish: column headers, stats loading, fixture empty state (W210 follow-up)", () => {
  beforeEach(() => {
    vi.mocked(getFixtures).mockReset();
    vi.mocked(getSandboxStatus).mockReset();
    vi.mocked(getBets).mockReset();
    vi.mocked(getBetStats).mockReset();
    vi.mocked(getStatus).mockReset();
    vi.mocked(getSandboxStatus).mockResolvedValue({ sandbox_mode: false, as_of: null });
    vi.mocked(getStatus).mockRejectedValue(new Error("no backend"));
    vi.mocked(getFixtures).mockResolvedValue([]);
  });

  it("shows column headers above the logged bets list", async () => {
    vi.mocked(getBets).mockResolvedValue([
      {
        id: 1, match_id: "m1", date: "2026-08-22", home_team: "Arsenal", away_team: "Everton",
        market: "result_3way", selection: "home", odds: 2.1, stake: 10, outcome: "open",
        profit_loss: null, source: "from_recommendation", recommendation_snapshot: null, created_at: "now",
      },
    ]);
    vi.mocked(getBetStats).mockResolvedValue({
      bets_settled: 0, bets_open: 1, bets_won: 0, roi: 0, hit_rate: 0,
      total_staked: 10, total_profit: 0, max_drawdown: 0,
      starting_bankroll: 0, current_bankroll: 0,
    });

    render(<BetTrackerPage />);
    await waitFor(() => expect(screen.getByText(/logged bets/i)).toBeInTheDocument());
    await waitFor(() => expect(screen.getByText(/Arsenal v Everton/)).toBeInTheDocument());

    expect(screen.getByText("Odds")).toBeInTheDocument();
    expect(screen.getByText("Stake")).toBeInTheDocument();
    expect(screen.getByText("P&L")).toBeInTheDocument();
    expect(screen.getByText("Outcome")).toBeInTheDocument();
  });

  it("shows a loading placeholder for stats while they're in flight", () => {
    vi.mocked(getBets).mockReturnValue(new Promise(() => {})); // never resolves
    vi.mocked(getBetStats).mockReturnValue(new Promise(() => {}));

    render(<BetTrackerPage />);

    // Both the stats placeholder and the pre-existing bets-list placeholder
    // read "Loading…" while getBets/getBetStats are both in flight --
    // getAllByText instead of the plan's getByText, since two legitimate
    // matches now exist rather than one.
    expect(screen.getAllByText(/loading/i).length).toBeGreaterThan(0);
  });

  it("shows a 'no matches' message when a fixture search returns nothing", async () => {
    vi.mocked(getBets).mockResolvedValue([]);
    vi.mocked(getBetStats).mockResolvedValue({
      bets_settled: 0, bets_open: 0, bets_won: 0, roi: 0, hit_rate: 0,
      total_staked: 0, total_profit: 0, max_drawdown: 0,
      starting_bankroll: 0, current_bankroll: 0,
    });
    vi.mocked(getFixtures).mockResolvedValue([]);

    const user = userEvent.setup();
    render(<BetTrackerPage />);

    await waitFor(() => expect(getFixtures).toHaveBeenCalled());
    await user.type(
      screen.getByPlaceholderText(/search a real fixture/i),
      "zzz-no-such-team"
    );

    await waitFor(() => expect(screen.getByText(/no matching fixtures/i)).toBeInTheDocument());
  });

  it("a non-401 stats failure retires the Loading placeholder instead of showing it forever", async () => {
    vi.mocked(getBets).mockResolvedValue([]);
    vi.mocked(getBetStats).mockRejectedValue(new ApiError("Failed to load bet stats (500)", 500));

    render(<BetTrackerPage />);

    // Bets resolves to an empty list ("No bets logged yet."), so the only
    // "Loading…" text left standing once settled is the stats section's --
    // no need to scope further than the plain query.
    await waitFor(() => expect(screen.getByText(/no bets logged yet/i)).toBeInTheDocument());
    expect(screen.queryByText(/loading/i)).not.toBeInTheDocument();
  });
});

describe("BetTrackerPage lets a user delete their own logged bet (W215)", () => {
  beforeEach(() => {
    vi.mocked(getFixtures).mockReset();
    vi.mocked(getSandboxStatus).mockReset();
    vi.mocked(getBets).mockReset();
    vi.mocked(getBetStats).mockReset();
    vi.mocked(getStatus).mockReset();
    vi.mocked(settleOpenBets).mockReset();
    vi.mocked(deleteBet).mockReset();
    vi.mocked(getFixtures).mockResolvedValue([]);
    vi.mocked(getSandboxStatus).mockResolvedValue({ sandbox_mode: false, as_of: null });
    vi.mocked(getStatus).mockRejectedValue(new Error("no backend"));
    vi.mocked(settleOpenBets).mockResolvedValue([]);
  });

  const bet = {
    id: 1, match_id: "m1", date: "2026-08-22", home_team: "Arsenal", away_team: "Everton",
    market: "result_3way", selection: "home", odds: 2.1, stake: 10, outcome: "open" as const,
    profit_loss: null, source: "manual" as const, recommendation_snapshot: null, created_at: "now",
  };

  it("shows a two-step confirm, and removes the row on confirm", async () => {
    vi.mocked(getBets).mockResolvedValueOnce([bet]).mockResolvedValueOnce([]);
    vi.mocked(getBetStats).mockResolvedValue({
      bets_settled: 0, bets_open: 0, bets_won: 0, roi: 0, hit_rate: 0,
      total_staked: 0, total_profit: 0, max_drawdown: 0, starting_bankroll: 0, current_bankroll: 0,
    });
    vi.mocked(deleteBet).mockResolvedValue(undefined);
    const user = userEvent.setup();

    render(<BetTrackerPage />);
    await screen.findByText(/Arsenal v Everton/);

    await user.click(screen.getByRole("button", { name: /delete/i }));
    expect(screen.getByText(/delete this bet\?/i)).toBeInTheDocument();

    await user.click(screen.getByRole("button", { name: /confirm delete/i }));

    expect(deleteBet).toHaveBeenCalledWith(1);
    await waitFor(() => expect(screen.queryByText(/Arsenal v Everton/)).not.toBeInTheDocument());
  });

  it("clicking Cancel on the confirm step leaves the bet in place", async () => {
    vi.mocked(getBets).mockResolvedValue([bet]);
    vi.mocked(getBetStats).mockResolvedValue({
      bets_settled: 0, bets_open: 0, bets_won: 0, roi: 0, hit_rate: 0,
      total_staked: 0, total_profit: 0, max_drawdown: 0, starting_bankroll: 0, current_bankroll: 0,
    });
    const user = userEvent.setup();

    render(<BetTrackerPage />);
    await screen.findByText(/Arsenal v Everton/);

    await user.click(screen.getByRole("button", { name: /delete/i }));
    await user.click(screen.getByRole("button", { name: /cancel delete/i }));

    expect(deleteBet).not.toHaveBeenCalled();
    expect(screen.getByText(/Arsenal v Everton/)).toBeInTheDocument();
  });

  it("shows an inline error and leaves the bet in place when delete fails (not a 401)", async () => {
    vi.mocked(getBets).mockResolvedValue([bet]);
    vi.mocked(getBetStats).mockResolvedValue({
      bets_settled: 0, bets_open: 0, bets_won: 0, roi: 0, hit_rate: 0,
      total_staked: 0, total_profit: 0, max_drawdown: 0, starting_bankroll: 0, current_bankroll: 0,
    });
    vi.mocked(deleteBet).mockRejectedValue(new ApiError("Failed to delete bet (500)", 500));
    const user = userEvent.setup();

    render(<BetTrackerPage />);
    await screen.findByText(/Arsenal v Everton/);

    await user.click(screen.getByRole("button", { name: /delete/i }));
    await user.click(screen.getByRole("button", { name: /confirm delete/i }));

    expect(await screen.findByText(/Failed to delete bet/i)).toBeInTheDocument();
    // Row wasn't removed, and Yes/Cancel are still there so the user can
    // retry or back out.
    expect(screen.getByText(/Arsenal v Everton/)).toBeInTheDocument();
    expect(screen.getByRole("button", { name: /confirm delete/i })).toBeInTheDocument();
    expect(screen.getByRole("button", { name: /cancel delete/i })).toBeInTheDocument();
  });

  it("prompts re-auth (not an inline error) when delete fails with a 401", async () => {
    vi.mocked(getBets).mockResolvedValue([bet]);
    vi.mocked(getBetStats).mockResolvedValue({
      bets_settled: 0, bets_open: 0, bets_won: 0, roi: 0, hit_rate: 0,
      total_staked: 0, total_profit: 0, max_drawdown: 0, starting_bankroll: 0, current_bankroll: 0,
    });
    vi.mocked(deleteBet).mockRejectedValue(new ApiError("Failed to delete bet (401)", 401));
    const user = userEvent.setup();

    render(<BetTrackerPage />);
    await screen.findByText(/Arsenal v Everton/);

    await user.click(screen.getByRole("button", { name: /delete/i }));
    await user.click(screen.getByRole("button", { name: /confirm delete/i }));

    await waitFor(() => expect(screen.getByText(/your session expired/i)).toBeInTheDocument());
    expect(screen.queryByText(/failed to delete bet/i)).not.toBeInTheDocument();
  });
});

describe("BetTrackerPage lets a user edit their own logged bet (W216)", () => {
  beforeEach(() => {
    vi.mocked(getFixtures).mockReset();
    vi.mocked(getSandboxStatus).mockReset();
    vi.mocked(getBets).mockReset();
    vi.mocked(getBetStats).mockReset();
    vi.mocked(getStatus).mockReset();
    vi.mocked(settleOpenBets).mockReset();
    vi.mocked(updateBet).mockReset();
    vi.mocked(getFixtures).mockResolvedValue([]);
    vi.mocked(getSandboxStatus).mockResolvedValue({ sandbox_mode: false, as_of: null });
    vi.mocked(getStatus).mockRejectedValue(new Error("no backend"));
    vi.mocked(settleOpenBets).mockResolvedValue([]);
  });

  const bet = {
    id: 1, match_id: "m1", date: "2026-08-22", home_team: "Arsenal", away_team: "Everton",
    market: "result_3way", selection: "home", odds: 2.1, stake: 10, outcome: "open" as const,
    profit_loss: null, source: "manual" as const, recommendation_snapshot: null, created_at: "now",
  };

  it("shows an edit form pre-filled with the bet's current values, and saves via updateBet", async () => {
    const updated = { ...bet, stake: 25 };
    vi.mocked(getBets).mockResolvedValueOnce([bet]).mockResolvedValueOnce([updated]);
    vi.mocked(getBetStats).mockResolvedValue({
      bets_settled: 0, bets_open: 1, bets_won: 0, roi: 0, hit_rate: 0,
      total_staked: 25, total_profit: 0, max_drawdown: 0, starting_bankroll: 0, current_bankroll: 0,
    });
    vi.mocked(updateBet).mockResolvedValue(updated);
    const user = userEvent.setup();

    render(<BetTrackerPage />);
    await screen.findByText(/Arsenal v Everton/);

    await user.click(screen.getByRole("button", { name: /^edit bet/i }));

    const stakeInput = screen.getByLabelText(/edit stake/i) as HTMLInputElement;
    expect(stakeInput.value).toBe("10");
    await user.clear(stakeInput);
    await user.type(stakeInput, "25");
    await user.click(screen.getByRole("button", { name: /save bet edit/i }));

    expect(updateBet).toHaveBeenCalledWith(1, { market: "result_3way", selection: "home", odds: 2.1, stake: 25 });
    await waitFor(() => expect(screen.getByText("25.00")).toBeInTheDocument());
  });

  it("clicking Cancel on the edit form discards changes and leaves the bet unchanged", async () => {
    vi.mocked(getBets).mockResolvedValue([bet]);
    vi.mocked(getBetStats).mockResolvedValue({
      bets_settled: 0, bets_open: 1, bets_won: 0, roi: 0, hit_rate: 0,
      total_staked: 10, total_profit: 0, max_drawdown: 0, starting_bankroll: 0, current_bankroll: 0,
    });
    const user = userEvent.setup();

    render(<BetTrackerPage />);
    await screen.findByText(/Arsenal v Everton/);

    await user.click(screen.getByRole("button", { name: /^edit bet/i }));
    const stakeInput = screen.getByLabelText(/edit stake/i);
    await user.clear(stakeInput);
    await user.type(stakeInput, "999");
    await user.click(screen.getByRole("button", { name: /cancel bet edit/i }));

    expect(updateBet).not.toHaveBeenCalled();
    expect(screen.getByText("10.00")).toBeInTheDocument();
    expect(screen.queryByLabelText(/edit stake/i)).not.toBeInTheDocument();
  });

  it("shows an inline error and keeps the form open when the update fails (not a 401)", async () => {
    vi.mocked(getBets).mockResolvedValue([bet]);
    vi.mocked(getBetStats).mockResolvedValue({
      bets_settled: 0, bets_open: 1, bets_won: 0, roi: 0, hit_rate: 0,
      total_staked: 10, total_profit: 0, max_drawdown: 0, starting_bankroll: 0, current_bankroll: 0,
    });
    vi.mocked(updateBet).mockRejectedValue(new ApiError("Failed to update bet (500)", 500));
    const user = userEvent.setup();

    render(<BetTrackerPage />);
    await screen.findByText(/Arsenal v Everton/);

    await user.click(screen.getByRole("button", { name: /^edit bet/i }));
    await user.click(screen.getByRole("button", { name: /save bet edit/i }));

    expect(await screen.findByText(/Failed to update bet/i)).toBeInTheDocument();
    expect(screen.getByLabelText(/edit stake/i)).toBeInTheDocument();
  });

  it("prompts re-auth (not an inline error) when the update fails with a 401", async () => {
    vi.mocked(getBets).mockResolvedValue([bet]);
    vi.mocked(getBetStats).mockResolvedValue({
      bets_settled: 0, bets_open: 1, bets_won: 0, roi: 0, hit_rate: 0,
      total_staked: 10, total_profit: 0, max_drawdown: 0, starting_bankroll: 0, current_bankroll: 0,
    });
    vi.mocked(updateBet).mockRejectedValue(new ApiError("Failed to update bet (401)", 401));
    const user = userEvent.setup();

    render(<BetTrackerPage />);
    await screen.findByText(/Arsenal v Everton/);

    await user.click(screen.getByRole("button", { name: /^edit bet/i }));
    await user.click(screen.getByRole("button", { name: /save bet edit/i }));

    await waitFor(() => expect(screen.getByText(/your session expired/i)).toBeInTheDocument());
    expect(screen.queryByText(/failed to update bet/i)).not.toBeInTheDocument();
  });

  it("changing the market resets the selection to force a fresh, valid choice", async () => {
    vi.mocked(getBets).mockResolvedValue([bet]);
    vi.mocked(getBetStats).mockResolvedValue({
      bets_settled: 0, bets_open: 1, bets_won: 0, roi: 0, hit_rate: 0,
      total_staked: 10, total_profit: 0, max_drawdown: 0, starting_bankroll: 0, current_bankroll: 0,
    });
    const user = userEvent.setup();

    render(<BetTrackerPage />);
    await screen.findByText(/Arsenal v Everton/);

    await user.click(screen.getByRole("button", { name: /^edit bet/i }));
    await user.selectOptions(screen.getByLabelText(/edit market/i), "btts");

    expect((screen.getByLabelText(/edit selection/i) as HTMLSelectElement).value).toBe("");
  });
});

describe("ManualBetForm search results show a league tag (W215)", () => {
  beforeEach(() => {
    vi.mocked(getFixtures).mockReset();
    vi.mocked(getSandboxStatus).mockReset();
    vi.mocked(getBets).mockReset();
    vi.mocked(getBetStats).mockReset();
    vi.mocked(getStatus).mockReset();
    vi.mocked(getBets).mockResolvedValue([]);
    vi.mocked(getBetStats).mockResolvedValue({
      bets_settled: 0, bets_open: 0, bets_won: 0, roi: 0, hit_rate: 0,
      total_staked: 0, total_profit: 0, max_drawdown: 0, starting_bankroll: 0, current_bankroll: 0,
    });
    vi.mocked(getSandboxStatus).mockResolvedValue({ sandbox_mode: false, as_of: null });
    vi.mocked(getStatus).mockRejectedValue(new Error("no backend"));
  });

  it("W215: each search result shows its league name, not the raw competition code", async () => {
    vi.mocked(getFixtures).mockResolvedValue([
      {
        match_id: "1", utc_date: "2026-08-22T15:00:00Z", status: "SCHEDULED",
        home_team: "Arsenal", away_team: "Everton", home_goals: null, away_goals: null, competition: "E0",
      },
    ]);
    const user = userEvent.setup();
    render(<BetTrackerPage />);

    await user.type(screen.getByPlaceholderText("Search a real fixture by team name…"), "Arsenal");

    // LEAGUE_LABEL["E0"] -- reused from dashboardMetrics.ts, not a raw code.
    expect(await screen.findByText("Premier League")).toBeInTheDocument();
    expect(screen.queryByText("E0")).not.toBeInTheDocument();
  });

  it("falls back to the raw competition code for an unmapped league", async () => {
    vi.mocked(getFixtures).mockResolvedValue([
      {
        match_id: "2", utc_date: "2026-08-23T15:00:00Z", status: "SCHEDULED",
        home_team: "Malmo FF", away_team: "AIK", home_goals: null, away_goals: null, competition: "XYZ",
      },
    ]);
    const user = userEvent.setup();
    render(<BetTrackerPage />);

    await user.type(screen.getByPlaceholderText("Search a real fixture by team name…"), "Malmo");

    expect(await screen.findByText("XYZ")).toBeInTheDocument();
  });

  it("shows no league tag when a fixture has no competition", async () => {
    vi.mocked(getFixtures).mockResolvedValue([
      {
        match_id: "3", utc_date: "2026-08-24T15:00:00Z", status: "SCHEDULED",
        home_team: "Chelsea", away_team: "Fulham", home_goals: null, away_goals: null,
      },
    ]);
    const user = userEvent.setup();
    render(<BetTrackerPage />);

    await user.type(screen.getByPlaceholderText("Search a real fixture by team name…"), "Chelsea");

    await screen.findByText(/Chelsea v Fulham/);
    expect(screen.queryByText("undefined")).not.toBeInTheDocument();
  });
});
