import { describe, expect, it } from "vitest";
import { countByOverall, groupByLeague, rankTopEdges, sortMatches } from "./dashboardMetrics";
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
    });
  });

  it("returns all-zero counts for an empty list", () => {
    expect(countByOverall([])).toEqual({
      direct_bet: 0,
      conditional: 0,
      no_bet: 0,
      insufficient_data: 0,
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

  it("does not mutate the input array", () => {
    const matches = [match({ id: "a", kickoffIso: "2026-08-22T18:00:00Z" }), match({ id: "b", kickoffIso: "2026-08-22T11:00:00Z" })];
    const original = [...matches];
    sortMatches(matches, "kickoff");
    expect(matches).toEqual(original);
  });
});
