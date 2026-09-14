import { render, screen } from "@testing-library/react";
import { describe, expect, it, vi, beforeEach } from "vitest";
import { LogBetButton } from "../MatchUI";

const mockUseSession = vi.fn();
vi.mock("next-auth/react", () => ({
  useSession: () => mockUseSession(),
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
});
