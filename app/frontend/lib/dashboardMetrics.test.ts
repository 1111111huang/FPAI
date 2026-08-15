import { describe, expect, it } from "vitest";
import { countByOverall, groupByDate, groupByLeague, rankTopEdges, sortMatches } from "./dashboardMetrics";
import type { Match } from "@/components/MatchUI";

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
    markets: [],
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

describe("countByOverall", () => {
  it("counts only matches with a generated recommendation, grouped by overall", () => {
    const matches = [
      match({ id: "1", overall: "direct_bet" }),
      match({ id: "2", overall: "direct_bet" }),
      match({ id: "3", overall: "conditional" }),
      match({ id: "4", hasRecommendation: false }),
    ];
    expect(countByOverall(matches)).toEqual({
      direct_bet: 2,
      conditional: 1,
      no_bet: 0,
      insufficient_data: 0,
      completed: 0,
    });
  });

  it("returns all-zero counts for an empty list", () => {
    expect(countByOverall([])).toEqual({
      direct_bet: 0,
      conditional: 0,
      no_bet: 0,
      insufficient_data: 0,
      completed: 0,
    });
  });

  it("direct user request: a completed match counts under 'completed', not its original overall bucket", () => {
    const matches = [
      match({ id: "1", status: "completed", overall: "direct_bet" }),
      match({ id: "2", status: "completed", overall: "conditional" }),
      match({ id: "3", overall: "direct_bet" }), // still upcoming -- counts normally
    ];
    expect(countByOverall(matches)).toEqual({
      direct_bet: 1,
      conditional: 0,
      no_bet: 0,
      insufficient_data: 0,
      completed: 2,
    });
  });

  it("a live match still counts under its own overall bucket -- not yet decided, unlike completed", () => {
    const matches = [match({ status: "live", result: { home: 1, away: 0 }, overall: "direct_bet" })];
    expect(countByOverall(matches)).toEqual({
      direct_bet: 1,
      conditional: 0,
      no_bet: 0,
      insufficient_data: 0,
      completed: 0,
    });
  });

  it("a completed match with no recommendation at all is still excluded, same as any other match without one", () => {
    const matches = [match({ status: "completed", hasRecommendation: false })];
    expect(countByOverall(matches)).toEqual({
      direct_bet: 0,
      conditional: 0,
      no_bet: 0,
      insufficient_data: 0,
      completed: 0,
    });
  });
});

describe("rankTopEdges", () => {
  it("ranks by value_edge descending, limited to N, excluding matches with no priced market", () => {
    const matches = [
      match({ id: "low", markets: [{ market: "result_3way", selection: "home", recommendationType: "direct_bet", currentOdds: 2.0, minOdds: 1.5, mlProbability: 0.5, impliedProbability: 0.5, valueEdge: 0.02 }] }),
      match({ id: "high", markets: [{ market: "result_3way", selection: "home", recommendationType: "direct_bet", currentOdds: 2.0, minOdds: 1.5, mlProbability: 0.5, impliedProbability: 0.5, valueEdge: 0.09 }] }),
      match({ id: "no-odds", markets: [{ market: "result_3way", selection: "home", recommendationType: "direct_bet", currentOdds: null, minOdds: 1.5, mlProbability: 0.5, impliedProbability: 0.5, valueEdge: 0.5 }] }),
      match({ id: "no-rec", hasRecommendation: false }),
    ];
    const ranked = rankTopEdges(matches, 5);
    expect(ranked.map((r) => r.match.id)).toEqual(["high", "low"]);
    expect(ranked[0].edge).toBeCloseTo(0.09);
  });

  it("respects the limit", () => {
    const matches = Array.from({ length: 10 }, (_, i) =>
      match({
        id: `m${i}`,
        markets: [{ market: "result_3way", selection: "home", recommendationType: "direct_bet", currentOdds: 2.0, minOdds: 1.5, mlProbability: 0.5, impliedProbability: 0.5, valueEdge: i / 100 }],
      })
    );
    expect(rankTopEdges(matches, 3)).toHaveLength(3);
  });

  // Direct user report (2026-08-08): a "No Bet" match whose only market had
  // a numerically higher (but below-threshold, hence still no_bet) edge
  // than a genuine direct_bet elsewhere was still being ranked -- a match
  // with nothing actionable has no business appearing in "Top Edges" at
  // all, regardless of which market's number is highest.
  it("excludes a match whose only market is no_bet, even with a higher raw edge than a real direct_bet", () => {
    const matches = [
      match({
        id: "no-bet-high-edge",
        overall: "no_bet",
        markets: [{ market: "result_3way", selection: "away", recommendationType: "no_bet", currentOdds: 5.0, minOdds: 0, mlProbability: 0.23, impliedProbability: 0.2, valueEdge: 0.08 }],
      }),
      match({
        id: "real-bet-lower-edge",
        overall: "direct_bet",
        markets: [{ market: "result_3way", selection: "home", recommendationType: "direct_bet", currentOdds: 1.8, minOdds: 0, mlProbability: 0.6, impliedProbability: 0.56, valueEdge: 0.04 }],
      }),
    ];
    const ranked = rankTopEdges(matches, 5);
    expect(ranked.map((r) => r.match.id)).toEqual(["real-bet-lower-edge"]);
  });
});

describe("groupByLeague", () => {
  it("groups matches by league, preserving first-seen order, with real labels", () => {
    const matches = [
      match({ id: "1", league: "E0" }),
      match({ id: "2", league: "SWE" }),
      match({ id: "3", league: "E0" }),
    ];
    const groups = groupByLeague(matches);
    expect(groups.map((g) => g.league)).toEqual(["E0", "SWE"]);
    expect(groups[0].label).toBe("Premier League");
    expect(groups[1].label).toBe("Allsvenskan");
    expect(groups[0].matches.map((m) => m.id)).toEqual(["1", "3"]);
  });

  it("falls back to the raw league code for an unrecognized value", () => {
    const groups = groupByLeague([match({ league: "XYZ" })]);
    expect(groups[0].label).toBe("XYZ");
  });
});

describe("groupByDate", () => {
  const asOf = new Date("2026-03-01T00:00:00Z");

  it("labels asOf's own day 'Today' and groups other days under a real calendar date", () => {
    const matches = [
      match({ id: "today-1", kickoffIso: "2026-03-01T15:00:00Z" }),
      match({ id: "tomorrow-1", kickoffIso: "2026-03-02T15:00:00Z" }),
      match({ id: "today-2", kickoffIso: "2026-03-01T18:00:00Z" }),
    ];
    const groups = groupByDate(matches, asOf, true);
    expect(groups.map((g) => g.label)).toEqual(["Today", "Mon, Mar 2"]);
    expect(groups[0].matches.map((m) => m.id)).toEqual(["today-1", "today-2"]);
    expect(groups[1].matches.map((m) => m.id)).toEqual(["tomorrow-1"]);
  });

  it("preserves first-seen day order, not chronological order, for out-of-order input", () => {
    const matches = [
      match({ id: "later", kickoffIso: "2026-03-05T15:00:00Z" }),
      match({ id: "earlier", kickoffIso: "2026-03-01T15:00:00Z" }),
    ];
    const groups = groupByDate(matches, asOf, true);
    expect(groups.map((g) => g.matches[0].id)).toEqual(["later", "earlier"]);
  });

  it("returns an empty array for an empty match list", () => {
    expect(groupByDate([], asOf, true)).toEqual([]);
  });
});

describe("sortMatches", () => {
  it("sorts by kickoff time ascending", () => {
    const matches = [
      match({ id: "later", kickoffIso: "2026-08-22T18:00:00Z" }),
      match({ id: "earlier", kickoffIso: "2026-08-22T11:00:00Z" }),
    ];
    expect(sortMatches(matches, "kickoff").map((m) => m.id)).toEqual(["earlier", "later"]);
  });

  it("sorts by edge descending, treating no-priced-market matches as lowest", () => {
    const matches = [
      match({ id: "no-market", markets: [] }),
      match({ id: "priced", markets: [{ market: "result_3way", selection: "home", recommendationType: "direct_bet", currentOdds: 2.0, minOdds: 1.5, mlProbability: 0.5, impliedProbability: 0.5, valueEdge: 0.05 }] }),
    ];
    expect(sortMatches(matches, "edge").map((m) => m.id)).toEqual(["priced", "no-market"]);
  });

  it("sorts by edge descending, treating an unpriced best market (currentOdds null) as lowest, same as no markets at all", () => {
    const matches = [
      match({ id: "unpriced", markets: [{ market: "result_3way", selection: "home", recommendationType: "direct_bet", currentOdds: null, minOdds: 1.5, mlProbability: 0.5, impliedProbability: 0.5, valueEdge: 0.5 }] }),
      match({ id: "priced", markets: [{ market: "result_3way", selection: "home", recommendationType: "direct_bet", currentOdds: 2.0, minOdds: 1.5, mlProbability: 0.5, impliedProbability: 0.5, valueEdge: 0.05 }] }),
      match({ id: "no-market", markets: [] }),
    ];
    expect(sortMatches(matches, "edge").map((m) => m.id)).toEqual(["priced", "unpriced", "no-market"]);
  });

  it("does not mutate the input array", () => {
    const matches = [match({ id: "a", kickoffIso: "2026-08-22T18:00:00Z" }), match({ id: "b", kickoffIso: "2026-08-22T11:00:00Z" })];
    const original = [...matches];
    sortMatches(matches, "kickoff");
    expect(matches).toEqual(original);
  });

  // Direct user report (2026-08-08): several No Bet cards showing +5.3%,
  // +5.5%, +4.5% rendered in that literal order under "Edge %" sort --
  // pricedEdge() correctly excludes every no_bet-only match from outranking
  // a real bet (both collapse to the same -Infinity), but that also left
  // them all tied relative to *each other*, falling back to array order
  // instead of the numbers actually shown on their own cards.
  it("sorts multiple no_bet-only matches by their own displayed edge, not left tied in array order", () => {
    const matches = [
      match({ id: "5.3pct", overall: "no_bet", markets: [{ market: "result_3way", selection: "away", recommendationType: "no_bet", currentOdds: 2.2, minOdds: 0, mlProbability: 0.5, impliedProbability: 0.45, valueEdge: 0.053 }] }),
      match({ id: "5.5pct", overall: "no_bet", markets: [{ market: "result_3way", selection: "away", recommendationType: "no_bet", currentOdds: 6.25, minOdds: 0, mlProbability: 0.2, impliedProbability: 0.16, valueEdge: 0.055 }] }),
      match({ id: "4.5pct", overall: "no_bet", markets: [{ market: "result_3way", selection: "draw", recommendationType: "no_bet", currentOdds: 3.0, minOdds: 0, mlProbability: 0.38, impliedProbability: 0.33, valueEdge: 0.045 }] }),
    ];
    expect(sortMatches(matches, "edge").map((m) => m.id)).toEqual(["5.5pct", "5.3pct", "4.5pct"]);
  });

  it("still never ranks a no_bet match above a genuine direct_bet, even with a much higher displayed edge", () => {
    const matches = [
      match({ id: "no-bet-9pct", overall: "no_bet", markets: [{ market: "result_3way", selection: "away", recommendationType: "no_bet", currentOdds: 5.0, minOdds: 0, mlProbability: 0.23, impliedProbability: 0.2, valueEdge: 0.09 }] }),
      match({ id: "direct-bet-2pct", overall: "direct_bet", markets: [{ market: "result_3way", selection: "home", recommendationType: "direct_bet", currentOdds: 1.8, minOdds: 0, mlProbability: 0.58, impliedProbability: 0.56, valueEdge: 0.02 }] }),
    ];
    expect(sortMatches(matches, "edge").map((m) => m.id)).toEqual(["direct-bet-2pct", "no-bet-9pct"]);
  });
});
