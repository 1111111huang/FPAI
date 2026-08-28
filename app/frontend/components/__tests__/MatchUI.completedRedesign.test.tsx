/**
 * Direct user request, design reference provided (mockup): the completed-
 * match card gets a distinct treatment from upcoming/live -- FT + Hit/Missed
 * badge replacing the StatusBadge pill up top (that info moves to the
 * footer instead as "Was a <label> pick"), a struck-through pick + inline
 * Hit/Not Hit text when the market resolved, and the pre-match edge value
 * relabeled "Pre-match edge" (not "Positive Edge", which implies a still-
 * actionable state). Upcoming/live cards are explicitly unchanged by this --
 * covered by regression tests below, not just implied.
 */
import { describe, expect, it, vi } from "vitest";
import { render, screen } from "@testing-library/react";
import { MatchCard, type Match } from "../MatchUI";

function baseMatch(overrides: Partial<Match> = {}): Match {
  return {
    id: "m1", league: "SP1", tier: "competition_specific", kickoffIso: "2026-08-15T15:00:00Z",
    home: "Alaves", away: "Getafe", status: "completed", result: { home: 3, away: 0 },
    hasRecommendation: true, overall: "direct_bet", confidence: "medium",
    markets: [
      { market: "result_3way", selection: "draw", recommendationType: "direct_bet", currentOdds: 3.0, minOdds: 0, mlProbability: 0.4, impliedProbability: 0.33, valueEdge: 0.055 },
    ],
    explanation: [], limitations: [], predictionBasis: "team_history_and_market",
    coldStartRisk: false, featureCompleteness: 0.9, unknownTeam: false, invalidMarketCount: 0,
    ...overrides,
  };
}

describe("MatchCard -- completed match, badge row", () => {
  it("shows FT for a completed match", () => {
    render(<MatchCard match={baseMatch()} onUpdate={vi.fn()} />);
    expect(screen.getByText("FT")).toBeInTheDocument();
  });

  it("does not show the StatusBadge pill ('Direct Bet') for a completed match -- that info moves to the footer instead", () => {
    render(<MatchCard match={baseMatch()} onUpdate={vi.fn()} />);
    expect(screen.queryByText("Direct Bet")).not.toBeInTheDocument();
  });

  it("upcoming match is unchanged -- still shows the StatusBadge pill, no FT", () => {
    const match = baseMatch({ status: "upcoming", result: undefined });
    render(<MatchCard match={match} onUpdate={vi.fn()} />);
    expect(screen.getByText("Direct Bet")).toBeInTheDocument();
    expect(screen.queryByText("FT")).not.toBeInTheDocument();
  });

  it("live match is unchanged -- still shows the StatusBadge pill, no FT", () => {
    const match = baseMatch({ status: "live", result: { home: 1, away: 0 } });
    render(<MatchCard match={match} onUpdate={vi.fn()} />);
    expect(screen.getByText("Direct Bet")).toBeInTheDocument();
    expect(screen.queryByText("FT")).not.toBeInTheDocument();
  });
});

describe("MatchCard -- completed match, pick column", () => {
  it("strikes through the pick when the market missed", () => {
    // recommended "draw", actual result home 3-0 -> missed
    render(<MatchCard match={baseMatch()} onUpdate={vi.fn()} />);
    expect(screen.getByText("Draw")).toHaveClass("line-through");
  });

  it("does not strike through the pick when the market hit", () => {
    const match = baseMatch({ result: { home: 1, away: 1 } }); // draw actually happened
    render(<MatchCard match={match} onUpdate={vi.fn()} />);
    expect(screen.getByText("Draw")).not.toHaveClass("line-through");
  });

  it("does not strike through the pick for an unresolvable market (corners)", () => {
    const match = baseMatch({
      markets: [{ market: "home_corners", selection: "over_2.5", recommendationType: "direct_bet", currentOdds: 1.5, minOdds: 0, mlProbability: 0.6, impliedProbability: 0.5, valueEdge: 0.1 }],
    });
    render(<MatchCard match={match} onUpdate={vi.fn()} />);
    expect(screen.getByText("Over")).not.toHaveClass("line-through");
  });

  it("pick is never struck through for an upcoming match", () => {
    const match = baseMatch({ status: "upcoming", result: undefined });
    render(<MatchCard match={match} onUpdate={vi.fn()} />);
    expect(screen.getByText("Draw")).not.toHaveClass("line-through");
  });
});

describe("MatchCard -- completed match, footer", () => {
  it("shows 'Full Time' instead of a kickoff clock time", () => {
    // "Full Time" also already appears once as the market's own subtitle
    // (result_3way/total_goals/btts/corners all have one) -- the footer
    // adds a second, independent occurrence, not a duplicate of that one.
    render(<MatchCard match={baseMatch()} onUpdate={vi.fn()} />);
    expect(screen.getAllByText("Full Time")).toHaveLength(2);
  });

  it("shows 'Was a <recommendation> pick' in the footer", () => {
    render(<MatchCard match={baseMatch()} onUpdate={vi.fn()} />);
    expect(screen.getByText(/Was a Direct Bet pick/)).toBeInTheDocument();
  });

  it("upcoming match footer is unchanged -- still shows a clock time, no footer 'Full Time'/'Was a...' text", () => {
    // The market-subtitle "Full Time" (unrelated to match status) still
    // shows once, same as always -- just not the footer's second one.
    const match = baseMatch({ status: "upcoming", result: undefined });
    render(<MatchCard match={match} onUpdate={vi.fn()} />);
    expect(screen.getAllByText("Full Time")).toHaveLength(1);
    expect(screen.queryByText(/Was a Direct Bet pick/)).not.toBeInTheDocument();
  });
});

describe("MatchCard -- completed match, edge column", () => {
  it("shows a neutral 'Pre-match edge' qualifier, not the actionable 'Positive Edge' one", () => {
    render(<MatchCard match={baseMatch()} onUpdate={vi.fn()} />);
    expect(screen.getByText("Pre-match edge")).toBeInTheDocument();
    expect(screen.queryByText("Positive Edge")).not.toBeInTheDocument();
  });

  it("shows 'Pre-match edge' even when the original edge was negative -- it's descriptive, not a recommendation to act", () => {
    const match = baseMatch({
      markets: [{ market: "result_3way", selection: "draw", recommendationType: "no_bet", currentOdds: 3.0, minOdds: 0, mlProbability: 0.2, impliedProbability: 0.33, valueEdge: -0.05 }],
    });
    render(<MatchCard match={match} onUpdate={vi.fn()} />);
    expect(screen.getByText("Pre-match edge")).toBeInTheDocument();
  });

  it("upcoming match with positive edge is unchanged -- still shows 'Positive Edge', not 'Pre-match edge'", () => {
    const match = baseMatch({ status: "upcoming", result: undefined });
    render(<MatchCard match={match} onUpdate={vi.fn()} />);
    expect(screen.getByText("Positive Edge")).toBeInTheDocument();
    expect(screen.queryByText("Pre-match edge")).not.toBeInTheDocument();
  });
});

describe("MatchCard -- completed match, odds column becomes money won (direct user request)", () => {
  // Money-won hit/miss/conditional/unresolvable math is covered by
  // MatchUI.hitMiss.test.tsx -- this block only covers the box swap and the
  // final score's new home (next to the team names) that made room for it.
  it("shows 'Money Won', not 'Odds', once a match completes", () => {
    render(<MatchCard match={baseMatch()} onUpdate={vi.fn()} />);
    expect(screen.getByText("Money Won")).toBeInTheDocument();
    expect(screen.queryByText("Odds")).not.toBeInTheDocument();
  });

  it("shows the final score next to the team names, not inside the Market/Pick/Odds/Edge row", () => {
    render(<MatchCard match={baseMatch()} onUpdate={vi.fn()} />); // baseMatch() result: 3-0
    expect(screen.getByText("3")).toBeInTheDocument();
    expect(screen.getByText("0")).toBeInTheDocument();
  });

  it("upcoming match is unchanged -- still shows 'Odds' and the numeric price, no 'Money Won', no score", () => {
    const match = baseMatch({ status: "upcoming", result: undefined });
    render(<MatchCard match={match} onUpdate={vi.fn()} />);
    expect(screen.getByText("Odds")).toBeInTheDocument();
    expect(screen.getByText("3.00")).toBeInTheDocument(); // baseMatch()'s currentOdds
    expect(screen.queryByText("Money Won")).not.toBeInTheDocument();
  });
});
