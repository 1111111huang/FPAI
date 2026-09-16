import { render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { describe, expect, it, vi, beforeEach } from "vitest";
import { LogBetButton } from "../MatchUI";
import { ApiError, logBetFromRecommendation, logBetManual } from "@/lib/api";

const mockUseSession = vi.fn();
vi.mock("next-auth/react", () => ({
  useSession: () => mockUseSession(),
}));

const mockUsePathname = vi.fn();
const mockUseSearchParams = vi.fn();
vi.mock("next/navigation", () => ({
  usePathname: () => mockUsePathname(),
  useSearchParams: () => mockUseSearchParams(),
}));

vi.mock("@/lib/api", () => ({
  logBetFromRecommendation: vi.fn(),
  logBetManual: vi.fn(),
  ApiError: class ApiError extends Error {
    status?: number;
    constructor(message: string, status?: number) {
      super(message);
      this.name = "ApiError";
      this.status = status;
    }
  },
}));

const recommendation = {
  match: { home: "Arsenal", away: "Everton", date: "2026-08-22", league: "E0" },
  overall: "direct_bet",
  candidates: [],
  recommendation_pick: { market: "result_3way", selection: "home" },
  explanation: [],
  confidence: "medium",
  limitations: [],
  prediction_basis: "team_history_and_market",
} as never;

const commonProps = { homeTeam: "Arsenal", awayTeam: "Everton", statusLabel: "Today · Full Time" };

describe("LogBetButton auth-awareness", () => {
  beforeEach(() => {
    mockUseSession.mockReset();
    mockUsePathname.mockReset();
    mockUseSearchParams.mockReset();
    mockUsePathname.mockReturnValue("/matches/m1");
    mockUseSearchParams.mockReturnValue(new URLSearchParams());
    vi.mocked(logBetFromRecommendation).mockReset();
    vi.mocked(logBetManual).mockReset();
  });

  it("shows a Sign in prompt instead of the log-bet control when unauthenticated", () => {
    mockUseSession.mockReturnValue({ status: "unauthenticated" });
    render(<LogBetButton matchId="m1" recommendation={recommendation} market="result_3way" selection="home" {...commonProps} />);
    expect(screen.getByRole("link", { name: /sign in/i })).toHaveAttribute("href", "/login?callbackUrl=%2Fmatches%2Fm1");
    expect(screen.queryByRole("button", { name: /log bet/i })).not.toBeInTheDocument();
  });

  it("shows the real Log bet control when authenticated", () => {
    mockUseSession.mockReturnValue({ status: "authenticated" });
    render(<LogBetButton matchId="m1" recommendation={recommendation} market="result_3way" selection="home" {...commonProps} />);
    expect(screen.getByRole("button", { name: /log bet/i })).toBeInTheDocument();
  });

  it("preserves the current page's full URL (query params included) in the sign-in callbackUrl", () => {
    mockUseSession.mockReturnValue({ status: "unauthenticated" });
    mockUsePathname.mockReturnValue("/matches/560572");
    mockUseSearchParams.mockReturnValue(
      new URLSearchParams("home=Crystal+Palace&away=Ipswich+Town&date=2026-09-12&league=E0")
    );

    render(<LogBetButton matchId="560572" recommendation={recommendation} market="result_3way" selection="home" {...commonProps} />);

    const link = screen.getByRole("link", { name: /sign in to log this bet/i });
    const callbackUrl = new URL(link.getAttribute("href")!, "http://localhost").searchParams.get("callbackUrl")!;
    expect(callbackUrl).toBe("/matches/560572?home=Crystal+Palace&away=Ipswich+Town&date=2026-09-12&league=E0");
  });

  it("shows the real settled outcome when the bet auto-settles immediately, with a link to Bet Tracker", async () => {
    mockUseSession.mockReturnValue({ status: "authenticated", data: { user: { email: "a@b.com" } } });
    vi.mocked(logBetFromRecommendation).mockResolvedValue({
      id: 1, match_id: "m1", date: "2026-08-22", home_team: "Arsenal", away_team: "Everton",
      market: "result_3way", selection: "home", odds: 2.1, stake: 10, outcome: "won",
      profit_loss: 12.1, source: "from_recommendation", recommendation_snapshot: null, created_at: "now",
    });
    const user = userEvent.setup();
    render(<LogBetButton matchId="m1" recommendation={recommendation} market="result_3way" selection="home" {...commonProps} />);

    await user.click(screen.getByRole("button", { name: "Log bet" }));
    await user.type(screen.getByLabelText(/^stake$/i), "10");
    await user.click(screen.getByRole("button", { name: /confirm bet/i }));

    expect(await screen.findByText(/won/i)).toBeInTheDocument();
    expect(screen.getByRole("link", { name: /view in bet tracker/i })).toHaveAttribute("href", "/bets");
  });

  it("shows a lost outcome in the lost color, not the won color", async () => {
    mockUseSession.mockReturnValue({ status: "authenticated", data: { user: { email: "a@b.com" } } });
    vi.mocked(logBetFromRecommendation).mockResolvedValue({
      id: 3, match_id: "m1", date: "2026-08-22", home_team: "Arsenal", away_team: "Everton",
      market: "result_3way", selection: "home", odds: 2.1, stake: 10, outcome: "lost",
      profit_loss: -10, source: "from_recommendation", recommendation_snapshot: null, created_at: "now",
    });
    const user = userEvent.setup();
    render(<LogBetButton matchId="m1" recommendation={recommendation} market="result_3way" selection="home" {...commonProps} />);

    await user.click(screen.getByRole("button", { name: "Log bet" }));
    await user.type(screen.getByLabelText(/^stake$/i), "10");
    await user.click(screen.getByRole("button", { name: /confirm bet/i }));

    const outcomeText = await screen.findByText(/lost/i);
    expect(outcomeText).toHaveClass("text-serious");
    expect(outcomeText).not.toHaveClass("text-good");
  });

  it("still shows a plain 'Logged' (no outcome yet) when the bet stays open", async () => {
    mockUseSession.mockReturnValue({ status: "authenticated", data: { user: { email: "a@b.com" } } });
    vi.mocked(logBetFromRecommendation).mockResolvedValue({
      id: 2, match_id: "m1", date: "2026-08-22", home_team: "Arsenal", away_team: "Everton",
      market: "result_3way", selection: "home", odds: 2.1, stake: 10, outcome: "open",
      profit_loss: null, source: "from_recommendation", recommendation_snapshot: null, created_at: "now",
    });
    const user = userEvent.setup();
    render(<LogBetButton matchId="m1" recommendation={recommendation} market="result_3way" selection="home" {...commonProps} />);

    await user.click(screen.getByRole("button", { name: "Log bet" }));
    await user.type(screen.getByLabelText(/^stake$/i), "10");
    await user.click(screen.getByRole("button", { name: /confirm bet/i }));

    expect(await screen.findByText("Logged")).toBeInTheDocument();
    expect(screen.getByRole("link", { name: /view in bet tracker/i })).toBeInTheDocument();
  });

  it("shows a Cancel control that collapses back to the plain Log bet link without submitting", async () => {
    mockUseSession.mockReturnValue({ status: "authenticated", data: { user: { email: "a@b.com" } } });
    const user = userEvent.setup();
    render(<LogBetButton matchId="m1" recommendation={recommendation} market="result_3way" selection="home" {...commonProps} />);

    await user.click(screen.getByRole("button", { name: "Log bet" }));
    expect(screen.getByLabelText(/^stake$/i)).toBeInTheDocument();

    await user.click(screen.getByRole("button", { name: "Cancel" }));

    expect(screen.queryByLabelText(/^stake$/i)).not.toBeInTheDocument();
    expect(screen.getByRole("button", { name: "Log bet" })).toBeInTheDocument();
    expect(logBetFromRecommendation).not.toHaveBeenCalled();
  });

  it("Cancel clears a stale error -- reopening after a failed attempt starts from a clean slate", async () => {
    // LogBetButton renders `<LogBetModal open .../>` only while its own
    // `open` state is true -- closing and reopening unmounts/remounts a
    // fresh LogBetModal instance each time, which naturally resets its
    // internal errorMsg state (no explicit reset needed here, unlike the
    // old inline-expand single-component version this replaced).
    mockUseSession.mockReturnValue({ status: "authenticated", data: { user: { email: "a@b.com" } } });
    vi.mocked(logBetFromRecommendation).mockRejectedValue(new ApiError("Could not log bet.", 500));
    const user = userEvent.setup();
    render(<LogBetButton matchId="m1" recommendation={recommendation} market="result_3way" selection="home" {...commonProps} />);

    await user.click(screen.getByRole("button", { name: "Log bet" }));
    await user.type(screen.getByLabelText(/^stake$/i), "10");
    await user.click(screen.getByRole("button", { name: /confirm bet/i }));
    expect(await screen.findByText("Could not log bet.")).toBeInTheDocument();

    await user.click(screen.getByRole("button", { name: "Cancel" }));
    await user.click(screen.getByRole("button", { name: "Log bet" }));

    expect(screen.queryByText("Could not log bet.")).not.toBeInTheDocument();
  });

  it("shows the locked market/pick/odds being logged in the modal", async () => {
    mockUseSession.mockReturnValue({ status: "authenticated", data: { user: { email: "a@b.com" } } });
    const recommendationWithOdds = {
      ...(recommendation as Record<string, unknown>),
      candidates: [{ market: "result_3way", selection: "home", recommendation_type: "direct_bet", current_odds: 2.35 }],
    } as never;
    const user = userEvent.setup();
    render(<LogBetButton matchId="m1" recommendation={recommendationWithOdds} market="result_3way" selection="home" {...commonProps} />);

    await user.click(screen.getByRole("button", { name: "Log bet" }));

    expect(screen.getByText("2.35")).toBeInTheDocument();
    expect(screen.getByText("Home")).toBeInTheDocument();
  });
});

describe("LogBetButton -- persisted done state and 'Log another' (2026-09-15)", () => {
  const persistedBet = {
    id: 9, match_id: "m1", date: "2026-08-22", home_team: "Arsenal", away_team: "Everton",
    market: "result_3way", selection: "home", odds: 2.1, stake: 10, outcome: "lost" as const,
    profit_loss: -10, source: "manual" as const, recommendation_snapshot: null, created_at: "now",
  };

  beforeEach(() => {
    mockUseSession.mockReset();
    mockUsePathname.mockReset();
    mockUseSearchParams.mockReset();
    mockUsePathname.mockReturnValue("/matches/m1");
    mockUseSearchParams.mockReturnValue(new URLSearchParams());
    mockUseSession.mockReturnValue({ status: "authenticated", data: { user: { email: "a@b.com" } } });
    vi.mocked(logBetFromRecommendation).mockReset();
    vi.mocked(logBetManual).mockReset();
  });

  it("shows the Logged state on first render from matchBets alone -- no submit needed in this session", () => {
    render(
      <LogBetButton
        matchId="m1" recommendation={recommendation} market="result_3way" selection="home" variant="pill"
        date="2026-08-22" matchBets={[persistedBet]} {...commonProps}
      />
    );

    expect(screen.getByText("Logged · Lost")).toBeInTheDocument();
    expect(screen.queryByRole("button", { name: "Log Bet" })).not.toBeInTheDocument();
  });

  it("'Log another' only appears for the pill variant with a persisted bet -- not the link variant", () => {
    const { rerender } = render(
      <LogBetButton
        matchId="m1" recommendation={recommendation} market="result_3way" selection="home" variant="pill"
        date="2026-08-22" matchBets={[persistedBet]} {...commonProps}
      />
    );
    expect(screen.getByRole("button", { name: /log another/i })).toBeInTheDocument();

    rerender(
      <LogBetButton
        matchId="m1" recommendation={recommendation} market="result_3way" selection="home" variant="link"
        matchBets={[persistedBet]} {...commonProps}
      />
    );
    expect(screen.queryByRole("button", { name: /log another/i })).not.toBeInTheDocument();
  });

  it("'Log another' opens an unlocked modal and logs via logBetManual, then notifies the parent to refetch", async () => {
    vi.mocked(logBetManual).mockResolvedValue({
      id: 10, match_id: "m1", date: "2026-08-22", home_team: "Arsenal", away_team: "Everton",
      market: "btts", selection: "yes", odds: 1.9, stake: 5, outcome: "open",
      profit_loss: null, source: "manual", recommendation_snapshot: null, created_at: "now",
    });
    const onBetsChanged = vi.fn();
    const user = userEvent.setup();
    render(
      <LogBetButton
        matchId="m1" recommendation={recommendation} market="result_3way" selection="home" variant="pill"
        date="2026-08-22" matchBets={[persistedBet]} onBetsChanged={onBetsChanged} {...commonProps}
      />
    );

    await user.click(screen.getByRole("button", { name: /log another/i }));
    // Unlocked -- a real dropdown, not the locked flow's fixed text.
    expect(screen.getByLabelText(/^market$/i).tagName).toBe("SELECT");
    await user.selectOptions(screen.getByLabelText(/^market$/i), "btts");
    await user.selectOptions(screen.getByLabelText(/^pick$/i), "yes");
    await user.type(screen.getByLabelText(/^odds$/i), "1.9");
    await user.type(screen.getByLabelText(/^stake$/i), "5");
    await user.click(screen.getByRole("button", { name: /confirm bet/i }));

    expect(logBetManual).toHaveBeenCalledWith({
      match_id: "m1", date: "2026-08-22", home_team: "Arsenal", away_team: "Everton",
      market: "btts", selection: "yes", odds: 1.9, stake: 5,
    });
    expect(onBetsChanged).toHaveBeenCalled();
  });
});
