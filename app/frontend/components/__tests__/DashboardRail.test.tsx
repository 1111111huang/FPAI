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
    candidates: [{ market: "result_3way", selection: "home", recommendationType: "direct_bet", currentOdds: 2.0, minOdds: 1.5, mlProbability: 0.5, impliedProbability: 0.5, valueEdge: 0.05 }],
    recommendationPick: { market: "result_3way", selection: "home" },
    explanation: [],
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

  it("direct user request: a completed match with a determinable pick shows a 'Completed (Hit)' or 'Completed (Not Hit)' row, not a bare 'Completed' row", () => {
    // Fixture default recommendationPick/candidates resolve to a real
    // direct_bet pick on "home" -- {home: 2, away: 0} means it hit.
    const matches = [
      match({ id: "1", status: "completed", result: { home: 2, away: 0 } }),
      match({ id: "2", status: "completed", result: { home: 0, away: 1 } }), // same pick, misses
      match({ id: "3", overall: "conditional" }), // still upcoming
    ];
    render(<DashboardRail matches={matches} />);
    expect(screen.getByText("Completed (Hit)")).toBeInTheDocument();
    expect(screen.getByText("Completed (Not Hit)")).toBeInTheDocument();
    expect(screen.getByText("Conditional")).toBeInTheDocument();
  });

  it("direct user follow-up request: a completed match with no determinable pick (no_bet, or unresolvable) still shows a bare 'Completed' row", () => {
    const matches = [
      match({ id: "1", status: "completed", result: { home: 2, away: 0 }, recommendationPick: null }),
    ];
    render(<DashboardRail matches={matches} />);
    expect(screen.getByText("Completed")).toBeInTheDocument();
    expect(screen.queryByText("Completed (Hit)")).not.toBeInTheDocument();
  });

  it("omits every 'Completed' row entirely when no match is completed -- same zero-count-hidden convention every other category already uses", () => {
    const matches = [match({ id: "1", overall: "direct_bet" })];
    render(<DashboardRail matches={matches} />);
    expect(screen.queryByText("Completed")).not.toBeInTheDocument();
    expect(screen.queryByText("Completed (Hit)")).not.toBeInTheDocument();
    expect(screen.queryByText("Completed (Not Hit)")).not.toBeInTheDocument();
  });

  it("renders Top Edges ranked by value_edge descending, as links to Match Analysis", () => {
    const matches = [
      match({ id: "low", home: "LowEdgeTeam", candidates: [{ ...match().candidates[0], valueEdge: 0.01 }] }),
      match({ id: "high", home: "HighEdgeTeam", candidates: [{ ...match().candidates[0], valueEdge: 0.09 }] }),
    ];
    render(<DashboardRail matches={matches} />);
    const links = screen.getAllByRole("link");
    expect(links[0]).toHaveTextContent("HighEdgeTeam");
    expect(links[0]).toHaveAttribute("href", expect.stringContaining("/matches/high"));
    expect(screen.getByText("+9.0%")).toBeInTheDocument();
  });

  it("direct user request: a completed Top Edges row greys out and strikes through the team-name line and shows the Hit/Not Hit verdict after the edge %", () => {
    const matches = [
      // Fixture default pick (result_3way/home) hits with {home: 2, away: 0}.
      match({ id: "hit", home: "HitTeam", status: "completed", result: { home: 2, away: 0 } }),
      // Same pick, misses.
      match({ id: "miss", home: "MissTeam", status: "completed", result: { home: 0, away: 1 } }),
    ];
    render(<DashboardRail matches={matches} />);

    const hitLine = screen.getByText("HitTeam v Everton");
    expect(hitLine).toHaveClass("text-muted", "line-through");
    expect(screen.getByText("Hit")).toBeInTheDocument();

    const missLine = screen.getByText("MissTeam v Everton");
    expect(missLine).toHaveClass("text-muted", "line-through");
    expect(screen.getByText("Not Hit")).toBeInTheDocument();
  });

  it("an upcoming Top Edges row keeps the plain (non-grey, non-strikethrough) team-name line and shows no Hit/Not Hit verdict", () => {
    const matches = [match({ id: "1" })];
    render(<DashboardRail matches={matches} />);

    const line = screen.getByText("Arsenal v Everton");
    expect(line).not.toHaveClass("text-muted", "line-through");
    expect(screen.queryByText("Hit")).not.toBeInTheDocument();
    expect(screen.queryByText("Not Hit")).not.toBeInTheDocument();
  });

  it("direct user request: a Staking Summary section shows total staked, total won/lost, and average odds across settled picks", () => {
    const matches = [
      // Fixture default pick (result_3way/home @ 2.0) hits with {home: 2, away: 0} -> +2 UB.
      match({ id: "hit", status: "completed", result: { home: 2, away: 0 }, unitBetMultiplier: 2 }),
      // Same pick @ 3.0, misses -> -3 UB.
      match({
        id: "miss", status: "completed", result: { home: 0, away: 1 }, unitBetMultiplier: 3,
        candidates: [{ ...match().candidates[0], currentOdds: 3.0 }],
      }),
    ];
    render(<DashboardRail matches={matches} />);

    expect(screen.getByText("Staking Summary")).toBeInTheDocument();
    expect(screen.getByText("5.0 UB")).toBeInTheDocument(); // staked: 2 + 3
    expect(screen.getByText("-1.0 UB")).toBeInTheDocument(); // won: +2 - 3
    expect(screen.getByText("2.50")).toBeInTheDocument(); // avg odds: (2.0 + 3.0) / 2
  });

  it("shows an empty state for Staking Summary when no match has a settled, actually-staked pick", () => {
    const matches = [match({ id: "1" })]; // upcoming -- nothing settled yet
    render(<DashboardRail matches={matches} />);
    expect(screen.getByText("No settled picks yet.")).toBeInTheDocument();
  });
});
