/**
 * Direct user request: for a completed match, show whether the recommended
 * market/selection actually hit. Resolution logic mirrors
 * src/agent/market_resolution.py's market_correct()/build_actual_outcome()
 * exactly (same RESOLVABLE_MARKETS, same None/null-for-unresolvable
 * contract) -- that module's own docstring exists specifically so backtest
 * scoring and live bet settlement never drift out of sync on this; this is
 * a third, presentation-only consumer of the same rule, ported to TS since
 * the frontend already has both ingredients (match.result, the recommended
 * market's selection) once a completed match's recommendation is loaded --
 * no new backend call needed.
 */
import { describe, expect, it, vi } from "vitest";
import { render, screen } from "@testing-library/react";
import { MatchCard, buildActualOutcome, marketCorrect, type Match } from "../MatchUI";

describe("buildActualOutcome -- mirrors src/agent/market_resolution.py", () => {
  it("a home win", () => {
    expect(buildActualOutcome(2, 1)).toEqual({ result: "home", btts: "yes", totalGoalsSide: "over_2.5" });
  });

  it("an away win with no goals conceded", () => {
    expect(buildActualOutcome(0, 1)).toEqual({ result: "away", btts: "no", totalGoalsSide: "under_2.5" });
  });

  it("a draw, both teams scoring, exactly on the 2.5 boundary", () => {
    expect(buildActualOutcome(1, 1)).toEqual({ result: "draw", btts: "yes", totalGoalsSide: "under_2.5" });
  });

  it("a 0-0 draw", () => {
    expect(buildActualOutcome(0, 0)).toEqual({ result: "draw", btts: "no", totalGoalsSide: "under_2.5" });
  });
});

describe("marketCorrect -- mirrors src/agent/market_resolution.py", () => {
  const homeWin = buildActualOutcome(2, 0);

  it("result_3way: correct selection", () => {
    expect(marketCorrect("result_3way", "home", homeWin)).toBe(true);
  });

  it("result_3way: incorrect selection", () => {
    expect(marketCorrect("result_3way", "away", homeWin)).toBe(false);
  });

  it("btts resolves against the actual btts outcome", () => {
    const bttsYes = buildActualOutcome(1, 1);
    expect(marketCorrect("btts", "yes", bttsYes)).toBe(true);
    expect(marketCorrect("btts", "no", bttsYes)).toBe(false);
  });

  it("total_goals resolves against the actual side", () => {
    const over = buildActualOutcome(2, 1);
    expect(marketCorrect("total_goals", "over_2.5", over)).toBe(true);
    expect(marketCorrect("total_goals", "under_2.5", over)).toBe(false);
  });

  it("an unresolvable market (corners) returns null, not false -- caller must not coerce to a miss", () => {
    expect(marketCorrect("home_corners", "over_2.5", homeWin)).toBeNull();
    expect(marketCorrect("away_corners", "under_2.5", homeWin)).toBeNull();
  });
});

function baseMatch(overrides: Partial<Match> = {}): Match {
  return {
    id: "m1", league: "E0", tier: "competition_specific", kickoffIso: "2026-08-15T15:00:00Z",
    home: "Arsenal", away: "Everton", status: "completed", result: { home: 2, away: 0 },
    hasRecommendation: true, overall: "direct_bet", confidence: "medium",
    markets: [
      { market: "result_3way", selection: "home", recommendationType: "direct_bet", currentOdds: 1.8, minOdds: 0, mlProbability: 0.6, impliedProbability: 0.56, valueEdge: 0.04 },
    ],
    explanation: [], limitations: [], predictionBasis: "team_history_and_market",
    coldStartRisk: false, featureCompleteness: 0.9, unknownTeam: false, invalidMarketCount: 0,
    ...overrides,
  };
}

describe("MatchCard -- hit/miss indicator for a completed match", () => {
  it("shows a HIT badge when the recommended market resolved correctly", () => {
    render(<MatchCard match={baseMatch()} onUpdate={vi.fn()} />);
    expect(screen.getByText("Hit")).toBeInTheDocument();
  });

  it("shows a MISSED badge when the recommended market resolved incorrectly", () => {
    const match = baseMatch({ result: { home: 0, away: 1 } }); // recommended "home", actual "away"
    render(<MatchCard match={match} onUpdate={vi.fn()} />);
    expect(screen.getByText("Missed")).toBeInTheDocument();
  });

  it("shows neither badge for an unresolvable market (corners)", () => {
    const match = baseMatch({
      markets: [{ market: "home_corners", selection: "over_2.5", recommendationType: "direct_bet", currentOdds: 1.5, minOdds: 0, mlProbability: 0.6, impliedProbability: 0.5, valueEdge: 0.1 }],
    });
    render(<MatchCard match={match} onUpdate={vi.fn()} />);
    expect(screen.queryByText("Hit")).not.toBeInTheDocument();
    expect(screen.queryByText("Missed")).not.toBeInTheDocument();
  });

  it("shows neither badge for an upcoming (not yet completed) match", () => {
    const match = baseMatch({ status: "upcoming", result: undefined });
    render(<MatchCard match={match} onUpdate={vi.fn()} />);
    expect(screen.queryByText("Hit")).not.toBeInTheDocument();
    expect(screen.queryByText("Missed")).not.toBeInTheDocument();
  });

  it("shows neither badge when there's no recommendation at all", () => {
    const match = baseMatch({ hasRecommendation: false });
    render(<MatchCard match={match} onUpdate={vi.fn()} />);
    expect(screen.queryByText("Hit")).not.toBeInTheDocument();
    expect(screen.queryByText("Missed")).not.toBeInTheDocument();
  });
});
