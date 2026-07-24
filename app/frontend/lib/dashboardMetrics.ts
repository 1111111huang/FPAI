import { bestMarket, type Match, type Overall } from "@/components/MatchUI";

export type OverallCounts = Record<Overall, number>;

/** Counts only matches with a generated recommendation -- a match still
 * showing "Not yet generated" has no overall worth counting. */
export function countByOverall(matches: Match[]): OverallCounts {
  const counts: OverallCounts = { direct_bet: 0, conditional: 0, no_bet: 0, insufficient_data: 0 };
  for (const m of matches) {
    if (!m.hasRecommendation) continue;
    counts[m.overall] += 1;
  }
  return counts;
}

export type TopEdge = { match: Match; edge: number };

/** Ranks by the best-priced market's value_edge, descending. A match with
 * no recommendation, or whose best market has no live odds (current_odds
 * null -- an unpriceable edge, not a real one), is excluded rather than
 * ranked with a fabricated value. */
export function rankTopEdges(matches: Match[], limit: number): TopEdge[] {
  const priced: TopEdge[] = [];
  for (const m of matches) {
    if (!m.hasRecommendation) continue;
    const shown = bestMarket(m);
    if (!shown || shown.currentOdds === null) continue;
    priced.push({ match: m, edge: shown.valueEdge });
  }
  return priced.sort((a, b) => b.edge - a.edge).slice(0, limit);
}

const LEAGUE_LABEL: Record<string, string> = { E0: "Premier League", SWE: "Allsvenskan" };

export type LeagueGroup = { league: string; label: string; matches: Match[] };

/** Groups by league in first-seen order (not alphabetical) so the section
 * order tracks whatever order the fixtures actually arrived in. */
export function groupByLeague(matches: Match[]): LeagueGroup[] {
  const order: string[] = [];
  const groups = new Map<string, Match[]>();
  for (const m of matches) {
    if (!groups.has(m.league)) {
      groups.set(m.league, []);
      order.push(m.league);
    }
    groups.get(m.league)!.push(m);
  }
  return order.map((league) => ({
    league,
    label: LEAGUE_LABEL[league] ?? league,
    matches: groups.get(league)!,
  }));
}

export type MatchSort = "kickoff" | "edge";

/** Returns a new array -- never mutates `matches` -- since callers hold
 * this list in React state and an in-place sort would be a silent mutation
 * bug (stale closures/memoization comparing the same array reference). */
export function sortMatches(matches: Match[], sort: MatchSort): Match[] {
  if (sort === "kickoff") {
    return [...matches].sort((a, b) => a.kickoffIso.localeCompare(b.kickoffIso));
  }
  return [...matches].sort((a, b) => {
    const edgeA = bestMarket(a)?.valueEdge ?? -Infinity;
    const edgeB = bestMarket(b)?.valueEdge ?? -Infinity;
    return edgeB - edgeA;
  });
}
