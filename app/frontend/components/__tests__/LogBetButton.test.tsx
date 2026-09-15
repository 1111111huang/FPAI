import { render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { describe, expect, it, vi, beforeEach } from "vitest";
import { LogBetButton } from "../MatchUI";
import { logBetFromRecommendation } from "@/lib/api";

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

describe("LogBetButton auth-awareness", () => {
  beforeEach(() => {
    mockUseSession.mockReset();
    mockUsePathname.mockReset();
    mockUseSearchParams.mockReset();
    mockUsePathname.mockReturnValue("/matches/m1");
    mockUseSearchParams.mockReturnValue(new URLSearchParams());
    vi.mocked(logBetFromRecommendation).mockReset();
  });

  it("shows a Sign in prompt instead of the log-bet control when unauthenticated", () => {
    mockUseSession.mockReturnValue({ status: "unauthenticated" });
    render(<LogBetButton matchId="m1" recommendation={recommendation} market="result_3way" selection="home" />);
    expect(screen.getByRole("link", { name: /sign in/i })).toHaveAttribute("href", "/login?callbackUrl=%2Fmatches%2Fm1");
    expect(screen.queryByRole("button", { name: /log bet/i })).not.toBeInTheDocument();
  });

  it("shows the real Log bet control when authenticated", () => {
    mockUseSession.mockReturnValue({ status: "authenticated" });
    render(<LogBetButton matchId="m1" recommendation={recommendation} market="result_3way" selection="home" />);
    expect(screen.getByRole("button", { name: /log bet/i })).toBeInTheDocument();
  });

  it("preserves the current page's full URL (query params included) in the sign-in callbackUrl", () => {
    mockUseSession.mockReturnValue({ status: "unauthenticated" });
    mockUsePathname.mockReturnValue("/matches/560572");
    mockUseSearchParams.mockReturnValue(
      new URLSearchParams("home=Crystal+Palace&away=Ipswich+Town&date=2026-09-12&league=E0")
    );

    render(<LogBetButton matchId="560572" recommendation={recommendation} market="result_3way" selection="home" />);

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
    render(<LogBetButton matchId="m1" recommendation={recommendation} market="result_3way" selection="home" />);

    await user.click(screen.getByRole("button", { name: "Log bet" }));
    await user.type(screen.getByPlaceholderText("Stake"), "10");
    await user.click(screen.getByRole("button", { name: "Confirm" }));

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
    render(<LogBetButton matchId="m1" recommendation={recommendation} market="result_3way" selection="home" />);

    await user.click(screen.getByRole("button", { name: "Log bet" }));
    await user.type(screen.getByPlaceholderText("Stake"), "10");
    await user.click(screen.getByRole("button", { name: "Confirm" }));

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
    render(<LogBetButton matchId="m1" recommendation={recommendation} market="result_3way" selection="home" />);

    await user.click(screen.getByRole("button", { name: "Log bet" }));
    await user.type(screen.getByPlaceholderText("Stake"), "10");
    await user.click(screen.getByRole("button", { name: "Confirm" }));

    expect(await screen.findByText("Logged")).toBeInTheDocument();
    expect(screen.getByRole("link", { name: /view in bet tracker/i })).toBeInTheDocument();
  });
});
