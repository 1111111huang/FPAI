import { bestMarket, dayDiff, type Match, type Overall } from "@/components/MatchUI";

// Direct user request: "completed" as a 5th, mutually-exclusive category --
// not part of the Overall union (that's a recommendation type, this is a
// match-status fact), so tracked as an extra field rather than folded into
// the Overall-keyed counts. A completed match counts here regardless of its
// original overall -- once decided, "what kind of pick it was" is secondary
// to "it's done" for an Edge Distribution panel about upcoming opportunity.
// A live match still counts under its own overall bucket -- not yet decided.
export type OverallCounts = Record<Overall, number> & { completed: number };

/** Counts only matches with a generated recommendation -- a match still
 * showing "Not yet generated" has no overall worth counting. */
export function countByOverall(matches: Match[]): OverallCounts {
  const counts: OverallCounts = { direct_bet: 0, conditional: 0, no_bet: 0, insufficient_data: 0, completed: 0 };
  for (const m of matches) {
    if (!m.hasRecommendation) continue;
    if (m.status === "completed") {
      counts.completed += 1;
      continue;
    }
    counts[m.overall] += 1;
  }
  return counts;
}

export type TopEdge = { match: Match; edge: number };

/** The best market's value_edge, but only when it's a real, priced,
 * actionable edge -- a match with no markets at all, whose best market has
 * no live odds (current_odds null -- an unpriceable edge, not a real one),
 * or whose only markets are no_bet (nothing actually clears the value
 * threshold), has no priced edge to report. That last case matters here
 * specifically: bestMarket() already prefers an actionable market when one
 * exists, but for an all-no_bet match it still returns the least-bad no_bet
 * market so MatchCard has *something* to show -- "Top Edges" must not treat
 * that fallback value as a real ranked edge, or a "No Bet" match could
 * outrank a genuine direct_bet/conditional opportunity. Shared by
 * rankTopEdges and sortMatches so both treat "no real edge" identically
 * instead of drifting apart. */
function pricedEdge(m: Match): number | null {
  const shown = bestMarket(m);
  if (!shown || shown.currentOdds === null || shown.recommendationType === "no_bet") return null;
  return shown.valueEdge;
}

/** Ranks by the best-priced market's value_edge, descending. A match with
 * no recommendation, or with no priced edge (see pricedEdge), is excluded
 * rather than ranked with a fabricated value. */
export function rankTopEdges(matches: Match[], limit: number): TopEdge[] {
  const priced: TopEdge[] = [];
  for (const m of matches) {
    if (!m.hasRecommendation) continue;
    const edge = pricedEdge(m);
    if (edge === null) continue;
    priced.push({ match: m, edge });
  }
  return priced.sort((a, b) => b.edge - a.edge).slice(0, limit);
}

// W110: exported so MatchAnalysisPage (MatchUI.tsx) can show the same full
// competition name instead of the raw code -- one label map, not two.
export const LEAGUE_LABEL: Record<string, string> = { E0: "Premier League", SWE: "Allsvenskan", SP1: "La Liga" };

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

export type DateGroup = { dateKey: string; label: string; matches: Match[] };

/** Calendar-date label for a date group's header -- the as-of day is always
 * "Today"; every other day gets a real calendar date (e.g. "Sun, Mar 15")
 * rather than a relative phrase, since a 10-match forward window can span
 * many different days at once (MatchCard's own per-card relative "in N
 * days" label still shows separately, unaffected). Formatted with the same
 * UTC-vs-local getter choice as dayDiff (via the `timeZone` option) so a
 * sandbox-mode midnight-UTC kickoff doesn't roll to the wrong weekday for a
 * non-UTC viewer -- the same class of bug W48/W71 already fixed elsewhere. */
function formatDateGroupLabel(iso: string, sandboxMode: boolean): string {
  return new Intl.DateTimeFormat(undefined, {
    weekday: "short",
    month: "short",
    day: "numeric",
    ...(sandboxMode ? { timeZone: "UTC" } : {}),
  }).format(new Date(iso));
}

/** Groups by calendar day (not league) in the order matches are given --
 * callers wanting chronological group order should sort `matches` by
 * kickoff first; an edge-sorted list would scramble which date each group
 * appears under. */
export function groupByDate(matches: Match[], asOf: Date, sandboxMode: boolean): DateGroup[] {
  const order: string[] = [];
  const groups = new Map<string, Match[]>();
  for (const m of matches) {
    const key = String(dayDiff(m.kickoffIso, asOf, sandboxMode));
    if (!groups.has(key)) {
      groups.set(key, []);
      order.push(key);
    }
    groups.get(key)!.push(m);
  }
  return order.map((key) => {
    const groupMatches = groups.get(key)!;
    return {
      dateKey: key,
      label: key === "0" ? "Today" : formatDateGroupLabel(groupMatches[0].kickoffIso, sandboxMode),
      matches: groupMatches,
    };
  });
}

export type MatchSort = "kickoff" | "edge";

/** Returns a new array -- never mutates `matches` -- since callers hold
 * this list in React state and an in-place sort would be a silent mutation
 * bug (stale closures/memoization comparing the same array reference). */
export function sortMatches(matches: Match[], sort: MatchSort): Match[] {
  if (sort === "kickoff") {
    return [...matches].sort((a, b) => a.kickoffIso.localeCompare(b.kickoffIso));
  }
  // Direct user report (2026-08-08): every no_bet-only match collapsed to
  // the same pricedEdge()-is-null/-Infinity tie, so within that tie the
  // comparator fell back to array order (kickoff-ish) instead of the edge
  // % each card actually displays -- e.g. +5.3%, +5.5%, +4.5% shown in that
  // literal, unsorted order. Two-tier sort: pricedEdge (real, actionable
  // edges only, matching bestMarket()'s own no_bet-never-outranks-a-real-
  // bet preference) is the primary key, so a No Bet card can still never
  // rank above a genuine direct_bet/conditional one regardless of its
  // number -- but *within* either tier, break ties by each card's own
  // displayed edge (bestMarket's value_edge, whatever market that is) so
  // the visible order always matches the numbers on screen.
  return [...matches].sort((a, b) => {
    const primaryA = pricedEdge(a);
    const primaryB = pricedEdge(b);
    if (primaryA !== null && primaryB === null) return -1;
    if (primaryA === null && primaryB !== null) return 1;
    if (primaryA !== null && primaryB !== null) return primaryB - primaryA;
    const displayA = bestMarket(a)?.valueEdge ?? -Infinity;
    const displayB = bestMarket(b)?.valueEdge ?? -Infinity;
    return displayB - displayA;
  });
}
