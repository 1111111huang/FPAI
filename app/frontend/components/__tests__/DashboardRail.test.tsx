import { render, screen } from "@testing-library/react";
import { describe, expect, it } from "vitest";
import { DashboardRail } from "../DashboardRail";
import type { Match } from "../MatchUI";

function match(overrides: Partial<Match> = {}): Match {
  return {
    id: "m1",
    league: "E0",
    tier: "competition_specific",
    kickoffIso: "2026-08-22T15:00:00Z",
    home: "Arsenal",
    away: "Everton",
    status: "upcoming",
    hasRecommendation: true,
    overall: "direct_bet",
    confidence: "medium",
    markets: [{ market: "result_3way", selection: "home", recommendationType: "direct_bet", currentOdds: 2.0, minOdds: 1.5, mlProbability: 0.5, impliedProbability: 0.5, valueEdge: 0.05 }],
    explanation: "",
    limitations: [],
    predictionBasis: "team_history_and_market",
    coldStartRisk: false,
    featureCompleteness: 0.9,
    unknownTeam: false,
    invalidMarketCount: 0,
    ...overrides,
  };
}

describe("DashboardRail", () => {
  it("shows an empty state for no matches", () => {
    render(<DashboardRail matches={[]} />);
    expect(screen.getByText("No matches loaded yet.")).toBeInTheDocument();
    expect(screen.getByText("No priced edges yet.")).toBeInTheDocument();
  });

  it("renders Edge Distribution counts per status", () => {
    const matches = [
      match({ id: "1", overall: "direct_bet" }),
      match({ id: "2", overall: "direct_bet" }),
      match({ id: "3", overall: "conditional" }),
    ];
    render(<DashboardRail matches={matches} />);
    expect(screen.getByText("Direct Bet")).toBeInTheDocument();
    expect(screen.getByText("Conditional")).toBeInTheDocument();
    expect(screen.getByText("2")).toBeInTheDocument();
    expect(screen.getByText("3")).toBeInTheDocument(); // total in the donut center
  });

  it("renders Top Edges ranked by value_edge descending, as links to Match Analysis", () => {
    const matches = [
      match({ id: "low", home: "LowEdgeTeam", markets: [{ ...match().markets[0], valueEdge: 0.01 }] }),
      match({ id: "high", home: "HighEdgeTeam", markets: [{ ...match().markets[0], valueEdge: 0.09 }] }),
    ];
    render(<DashboardRail matches={matches} />);
    const links = screen.getAllByRole("link");
    expect(links[0]).toHaveTextContent("HighEdgeTeam");
    expect(links[0]).toHaveAttribute("href", expect.stringContaining("/matches/high"));
    expect(screen.getByText("+9.0%")).toBeInTheDocument();
  });
});
