import { render, screen } from "@testing-library/react";
import { describe, expect, it, vi, beforeEach } from "vitest";
import { LogBetButton } from "../MatchUI";

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
});
