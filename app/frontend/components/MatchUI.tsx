"use client";

/**
 * W04 — real-data port of sandbox/components/sandbox/DraftUI.tsx. Visual
 * design (colors, spacing, components, atoms) is reused as-is per the
 * acceptance criteria; MOCK_MATCHES is replaced with real fetch calls to
 * app/backend's /api/fixtures and /api/recommendations.
 *
 * Player/squad/topFeatures data (DraftUI's "Agent Intelligence" section)
 * isn't returned by the real MatchRecommendationOut response today -- that
 * data lives in ForecastService's explainability payload, never plumbed
 * through W02's endpoint. Left as empty arrays here rather than inventing
 * new backend surface beyond what W04 asks for; the existing "unavailable"
 * messaging already covers this honestly.
 */

import { useEffect, useMemo, useState } from "react";
import Link from "next/link";
import {
  ArrowDown,
  ArrowLeft,
  ArrowUp,
  CalendarBlank,
  CaretDown,
  CaretRight,
  ChartBar,
  CheckCircle,
  Clock,
  MagnifyingGlass,
  MinusCircle,
  Question,
  Trophy,
  WarningCircle,
  X,
  XCircle,
} from "@phosphor-icons/react";

import {
  ApiError,
  generateRecommendation,
  getCachedRecommendation,
  getFixtures,
  logBetFromRecommendation,
} from "@/lib/api";
import type { Fixture, MatchRecommendationOut } from "@/lib/types";
import { useSandboxAsOf } from "@/lib/useSandboxAsOf";
import { groupByDate, groupByLeague, sortMatches, LEAGUE_COUNTRY, LEAGUE_LABEL, type MatchSort } from "@/lib/dashboardMetrics";
import { AppShell } from "./AppShell";
import { DashboardRail } from "./DashboardRail";

// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------

export type Tier = "competition_specific" | "general_purpose";
export type RecommendationType = "direct_bet" | "conditional" | "no_bet";
export type Overall = RecommendationType | "insufficient_data";
export type Confidence = "low" | "medium" | "high" | string;

export type MarketRec = {
  market: string;
  selection: string;
  recommendationType: RecommendationType;
  currentOdds: number | null;
  minOdds: number;
  mlProbability: number;
  impliedProbability: number;
  valueEdge: number;
  // W84/A52: code-computed price this market would need to reach to clear
  // min_value_edge -- null/undefined when not applicable or on a pre-A52
  // cached row. Optional (not required) so every existing hand-built
  // MarketRec literal across the test suite, none of which set this field,
  // keeps type-checking without modification -- same convention as
  // Fixture.competition (lib/types.ts).
  targetOdds?: number | null;
};

export type Match = {
  id: string;
  league: string;
  tier: Tier;
  kickoffIso: string;
  home: string;
  away: string;
  status: "upcoming" | "live" | "completed";
  result?: { home: number; away: number };
  // Recommendation data -- absent until generated (hasRecommendation gates this).
  hasRecommendation: boolean;
  overall: Overall;
  confidence: Confidence;
  markets: MarketRec[];
  // One bullet per aspect, mirroring lib/types.ts's MatchRecommendationOut.
  explanation: string[];
  limitations: string[];
  predictionBasis: string;
  // W15: first-class trust signals, independent of predictionBasis/overall --
  // must read as lower-trust even when predictionBasis claims full coverage.
  coldStartRisk: boolean;
  featureCompleteness: number | null;
  unknownTeam: boolean;
  // W16: markets W02 dropped for failing type validation -- an honest note
  // beats silently showing fewer markets with no explanation.
  invalidMarketCount: number;
  // A82/W169: Kelly-derived suggested stake for this recommendation's
  // actual pick, as a multiple of an abstract Unit Bet -- not a dollar
  // figure. null/undefined when there's no priced pick to suggest for.
  unitBetMultiplier?: number | null;
};

// ---------------------------------------------------------------------------
// Adapters -- real API shapes -> UI Match shape
// ---------------------------------------------------------------------------

export function fixtureToMatch(fixture: Fixture, asOf?: Date, sandboxMode = false): Match {
  // W48: a fixture whose kickoff date is strictly after asOf's date hasn't
  // "happened yet" in the sandbox's own pretend timeline, even when it's
  // already really been played (real FINISHED status + real score) relative
  // to actual wall-clock time -- render it as upcoming, exactly like a
  // genuinely future real fixture, so the Dashboard/Match Explorer don't
  // leak future real-world outcomes through the raw fixture list (the same
  // leakage class agent_techspec.md's own defenses cover for agent
  // web-search results, just via a different surface). Only applies in
  // sandbox mode -- outside it, or for a fixture on/before asOf, real-world
  // status is exactly what should show, unchanged. asOf is optional/
  // sandboxMode defaults false so every existing non-sandbox call site
  // keeps its current behavior even without passing them.
  const isFutureInSandbox = sandboxMode && asOf !== undefined && dayDiff(fixture.utc_date, asOf, sandboxMode) > 0;
  // A match currently being played is neither SCHEDULED/TIMED (kickoff
  // already happened) nor FINISHED (not over yet) -- IN_PLAY/PAUSED (e.g.
  // half-time) both mean "live". "LIVE" too (direct user report, confirmed
  // live: football-data.org's own real API returns this exact literal for
  // a currently-in-progress match, per BUG-050's football_data_client.py
  // comment -- BUG-050 added it to the backend's own status *query* so the
  // fixture is fetched at all, but never to this frontend check, so a
  // fetched status="LIVE" fixture fell through to "upcoming" here: no
  // LiveBadge, no live score, the card looked like the match hadn't
  // started). Same isFutureInSandbox guard as FINISHED below: sandbox
  // mode's own historical data source never actually produces a real
  // in-progress fixture, but if it ever did, it must not leak ahead of the
  // sandbox's own pretend clock either.
  const isLive = (fixture.status === "IN_PLAY" || fixture.status === "PAUSED" || fixture.status === "LIVE") && !isFutureInSandbox;
  const isReallyCompleted = fixture.status === "FINISHED" && !isFutureInSandbox;
  const status: Match["status"] = isReallyCompleted ? "completed" : isLive ? "live" : "upcoming";
  return {
    id: fixture.match_id,
    league: fixture.competition ?? "E0",
    tier: "competition_specific",
    kickoffIso: fixture.utc_date,
    home: fixture.home_team,
    away: fixture.away_team,
    status,
    // Gated on the (sandbox-aware) status above, not just goals-present --
    // a FINISHED-but-future-in-sandbox fixture must not carry a real score
    // on the Match object at all, not merely have it hidden at render time
    // (defense in depth: nothing downstream that later reads match.result
    // without re-checking status can leak it). Live carries a result too --
    // football-data.org updates home_goals/away_goals in real time during
    // play, not just at full-time.
    result:
      (status === "completed" || status === "live") && fixture.home_goals !== null && fixture.away_goals !== null
        ? { home: fixture.home_goals, away: fixture.away_goals }
        : undefined,
    hasRecommendation: false,
    overall: "insufficient_data",
    confidence: "low",
    markets: [],
    explanation: [],
    limitations: [],
    predictionBasis: "",
    coldStartRisk: false,
    featureCompleteness: null,
    unknownTeam: false,
    invalidMarketCount: 0,
  };
}

function applyRecommendation(match: Match, rec: MatchRecommendationOut): Match {
  return {
    ...match,
    hasRecommendation: true,
    overall: rec.overall,
    confidence: rec.confidence,
    predictionBasis: rec.prediction_basis,
    explanation: rec.explanation,
    limitations: rec.limitations,
    coldStartRisk: rec.cold_start_risk,
    featureCompleteness: rec.feature_completeness,
    unknownTeam: rec.unknown_team,
    unitBetMultiplier: rec.unit_bet_multiplier ?? null,
    invalidMarketCount: rec.invalid_market_count,
    markets: rec.markets.map((m) => ({
      market: m.market,
      selection: m.selection,
      recommendationType: m.recommendation_type,
      currentOdds: m.current_odds,
      minOdds: m.min_odds,
      mlProbability: m.ml_probability,
      impliedProbability: m.implied_probability,
      valueEdge: m.value_edge,
      targetOdds: m.target_odds ?? null,
    })),
  };
}

/** W53: bulk-resolve the recommendation cache for an initial fixture list so
 * a precomputed (W50/W51) match shows its real recommendation immediately,
 * with no click required -- fixtureToMatch() alone always leaves
 * hasRecommendation: false, and until this, the only two callers of
 * getCachedRecommendation() were both lazy/per-card (MatchCard.handleExpand
 * on click, MatchAnalysisPage.load on navigation), so a fully-precomputed
 * cache never visually manifested until every card was clicked individually.
 *
 * Runs one getCachedRecommendation() call per match concurrently (Promise.all,
 * not a sequential loop) -- the list is capped at 10 and this hits a local
 * SQLite-backed cache, so N concurrent local calls is the simple, correctly
 * scoped choice (no rate-limit concern like W52's football-data.org calls,
 * and no new backend bulk endpoint needed). A miss (null) or a thrown error
 * is treated identically -- same "degrade to miss" reasoning
 * MatchCard.handleExpand's own cache-check catch already established --
 * leaving the match unchanged (still hasRecommendation: false) so the
 * existing W47 lazy click-through fallback still applies untouched. */
async function resolveCachedRecommendations(matches: Match[]): Promise<Match[]> {
  return Promise.all(
    matches.map(async (m) => {
      try {
        const rec = await getCachedRecommendation(m.id, m.kickoffIso.slice(0, 10));
        return rec ? applyRecommendation(m, rec) : m;
      } catch {
        return m;
      }
    })
  );
}

function formatKickoff(iso: string): string {
  const d = new Date(iso);
  return d.toLocaleTimeString(undefined, { hour: "2-digit", minute: "2-digit" });
}

// Shared date-only (whole-day) diff between an ISO kickoff and asOf, in the
// direction (kickoff day) - (asOf day) -- positive means the kickoff is
// after asOf. asOf's meaning depends on sandboxMode, and the two are not
// interchangeable: in sandbox mode, asOf is UTC midnight of a deliberately
// timezone-agnostic chosen calendar date (W30) -- reading it via local
// getters would misread it by a day in non-UTC-zero timezones (the same bug
// class already fixed in Dashboard/Match Explorer's own asOf consumption),
// so UTC getters are required here. Outside sandbox mode, asOf is a real
// new Date() instant, and the viewer's own local calendar day is what
// "today" means for a human reading this -- reading it via UTC getters
// there would wrongly relabel "today" as "yesterday" for roughly half the
// day, every day, for any non-UTC viewer (the exact frame-mismatch class
// this branch keeps re-deriving; caught by review before this shipped).
// Don't unify these into one getter choice -- the branch is load-bearing,
// not incidental. Extracted (W48) so formatDay's relative-day label and
// fixtureToMatch's sandbox-future-fixture check share one implementation of
// this getter choice instead of two copies that could drift out of sync.
export function dayDiff(iso: string, asOf: Date, sandboxMode: boolean): number {
  const date = new Date(iso);
  // W71: fixture-side day must also use UTC getters in sandbox mode,
  // mirroring the asOf-side branch immediately below -- local getters here
  // silently disagree with UTC for a midnight-UTC fixture (exactly what
  // W71's raw_matches-backed historical SWE source synthesizes, since
  // raw_matches carries no real kickoff time) whenever the viewer is in a
  // negative-UTC-offset timezone, which can flip isFutureInSandbox's
  // result (W48's leak guard) for that fixture. Confirmed via direct
  // reproduction during W71's code review, not a theoretical concern.
  const dOnly = sandboxMode
    ? new Date(date.getUTCFullYear(), date.getUTCMonth(), date.getUTCDate())
    : new Date(date.getFullYear(), date.getMonth(), date.getDate());
  const tOnly = sandboxMode
    ? new Date(asOf.getUTCFullYear(), asOf.getUTCMonth(), asOf.getUTCDate())
    : new Date(asOf.getFullYear(), asOf.getMonth(), asOf.getDate());
  return Math.round((dOnly.getTime() - tOnly.getTime()) / 86_400_000);
}

function formatDay(iso: string, asOf: Date, sandboxMode: boolean): string {
  const diffDays = dayDiff(iso, asOf, sandboxMode);
  if (diffDays === 0) return "today";
  if (diffDays === 1) return "tomorrow";
  if (diffDays === -1) return "yesterday";
  if (diffDays > 1) return `in ${diffDays} days`;
  return `${-diffDays} days ago`;
}

// Direct user report: today's own already-finished match (Atleti v Malaga,
// kicked off 19:00 UTC) was completely missing from the Dashboard -- not
// filtered out by dayDiff (which already gets this branch right), but never
// even fetched. DashboardPage/MatchExplorerPage computed their getFixtures()
// window bounds via `asOf.toISOString().slice(0, 10)` -- always UTC --
// which silently advances "today" to tomorrow for several hours every
// evening in any UTC-negative-offset timezone (US zones included), sending
// a date_from that's one day too late and excluding the real local-today's
// fixtures/results before dayDiff/filtering ever runs on them. The exact
// getter-mismatch class this file already fixed three times elsewhere
// (dayDiff's own sandbox-vs-real branch, W30/W48/W71) -- missed here because
// this call site was never touched by any of those stories. Same asOf/
// sandboxMode contract as dayDiff: local getters (no .toISOString() round
// trip, which mis-renders for positive-offset zones too) outside sandbox
// mode, UTC getters (the pre-existing, already-correct behavior) inside it.
function dateString(d: Date, sandboxMode: boolean): string {
  if (sandboxMode) return d.toISOString().slice(0, 10);
  const y = d.getFullYear();
  const m = String(d.getMonth() + 1).padStart(2, "0");
  const day = String(d.getDate()).padStart(2, "0");
  return `${y}-${m}-${day}`;
}

function addDays(d: Date, days: number, sandboxMode: boolean): Date {
  const copy = new Date(d);
  if (sandboxMode) copy.setUTCDate(copy.getUTCDate() + days);
  else copy.setDate(copy.getDate() + days);
  return copy;
}

// ---------------------------------------------------------------------------
// Display metadata
// ---------------------------------------------------------------------------

const TIER_LABEL: Record<Tier, string> = {
  competition_specific: "Modeled",
  general_purpose: "General",
};

// W110/W107: one-line, hover/tap-discoverable explanation of the
// Modeled/General distinction -- previously only the two-word tag itself,
// meaningless to anyone without prior context on how the engine's per-
// competition models work.
const TIER_EXPLANATION: Record<Tier, string> = {
  competition_specific: "This competition has its own trained model, built on real historical team data.",
  general_purpose: "No dedicated model for this competition yet -- a general-purpose fallback model instead.",
};

// W120 follow-up: a small rotating background wash per date group -- purely
// decorative sequencing (index-based, not tied to any specific calendar
// date), matching the mockup's own purple/teal/green rotation. Tailwind's
// built-in violet/teal/emerald palettes (no custom theme config needed),
// faded via gradient-to-br toward transparent so it reads as a wash behind
// the cards, not a flat color block -- direct feedback ("pay attention to
// the gradient") after the first flat-panel attempt.
const DATE_GROUP_WASHES = [
  "from-violet-500/10 via-violet-500/5 to-transparent",
  "from-teal-500/10 via-teal-500/5 to-transparent",
  "from-emerald-500/10 via-emerald-500/5 to-transparent",
];

// Same rotation, same three hues, as a TierTag tint instead of a
// background wash -- so a card's MODELED tag echoes its own date panel's
// color rather than a fixed neutral gray regardless of which group it's
// in. Index-matched to DATE_GROUP_WASHES (both cycle by the same `i`), not
// merged into one array, since the two need different Tailwind utilities
// (bg-gradient-to-br stops vs a plain border/bg/text triple).
const TIER_TAG_TINTS = [
  "border-violet-400/40 bg-violet-500/15 text-violet-300",
  "border-teal-400/40 bg-teal-500/15 text-teal-300",
  "border-emerald-400/40 bg-emerald-500/15 text-emerald-300",
];

const STATUS_META: Record<
  Overall,
  { text: string; ring: string; fill: string; icon: React.ReactNode; label: string; verdict: string; explain: string }
> = {
  direct_bet: {
    text: "text-good",
    ring: "border-good/40",
    fill: "bg-good/15",
    icon: <CheckCircle weight="fill" size={13} />,
    label: "Direct Bet",
    verdict: "BET",
    // W107: plain-language, hover/tap-discoverable explanation of each
    // verdict -- previously only the label/badge itself, no context for a
    // reader without prior betting vocabulary.
    explain: "The model found a strong enough edge to recommend betting now.",
  },
  conditional: {
    text: "text-warning",
    ring: "border-warning/40",
    fill: "bg-warning/15",
    icon: <Clock weight="fill" size={13} />,
    label: "Conditional",
    verdict: "WAIT",
    explain: "There's a real edge here, but the current price isn't good enough yet -- wait for it to improve.",
  },
  no_bet: {
    text: "text-muted",
    ring: "border-border-strong",
    fill: "bg-surface",
    icon: <MinusCircle weight="fill" size={13} />,
    label: "No Bet",
    verdict: "PASS",
    explain: "No sufficient edge found -- not worth betting on this market.",
  },
  insufficient_data: {
    text: "text-serious",
    ring: "border-serious/40",
    fill: "bg-serious/15",
    icon: <Question weight="fill" size={13} />,
    label: "Insufficient Data",
    verdict: "NO READ",
    explain: "Not enough reliable data to make a confident prediction.",
  },
};

// ---------------------------------------------------------------------------
// Helpers ported from DraftUI.tsx
// ---------------------------------------------------------------------------

function formatPct(v: number) {
  return `${(v * 100).toFixed(0)}%`;
}
export function formatEdge(v: number) {
  const pct = (v * 100).toFixed(1);
  return v >= 0 ? `+${pct}%` : `${pct}%`;
}

// BUG-053 follow-up: same abstract UB unit the Stake column already uses
// (schema.py's UNIT_BET_BASELINE_FRACTION docstring -- "bet 2 UB at odds
// 3.0, get 6 UB back"), just signed like formatEdge above.
export function formatMoneyWon(v: number) {
  const ub = v.toFixed(1);
  return v >= 0 ? `+${ub} UB` : `${ub} UB`;
}

// W107: plain-language explanations for jargon labels, reused everywhere
// each label renders (via a native `title` tooltip -- no tooltip library).
const EDGE_EXPLAIN = "How much better the model's estimate is than the market price. Positive means the price looks better than it should be.";
const MODEL_PROBABILITY_EXPLAIN = "The model's own estimated probability of this outcome, independent of the market's price.";
const CONFIDENCE_EXPLAIN = "How reliable the model considers this particular prediction, based on the strength and consistency of the signal.";
/** W108: a match with a generated, actually-actionable recommendation --
 * the same predicate Dashboard/Match Explorer's own "Active Edges" sidebar
 * count already computed inline in two places (now shared, not duplicated
 * a third time for the new actionable-only filter). */
export function isActionable(match: Match): boolean {
  return match.hasRecommendation && (match.overall === "direct_bet" || match.overall === "conditional");
}

// Mirrors src/agent/market_resolution.py's RESOLVABLE_MARKETS/build_actual_outcome/
// market_correct exactly -- that module's docstring exists specifically so
// backtest scoring and live bet settlement never drift out of sync on which
// markets can be programmatically resolved; this is a third, presentation-
// only consumer of the same rule (a completed match's card, not a backend
// call) -- keep in sync if the Python side ever changes. home_corners/
// away_corners stay unresolvable: MarketRec has no numeric line field for
// them, only current_odds/min_odds, so there's no threshold to check against.
const RESOLVABLE_MARKETS = new Set(["result_3way", "btts", "total_goals"]);

export type ActualOutcome = { result: "home" | "away" | "draw"; btts: "yes" | "no"; totalGoalsSide: "over_2.5" | "under_2.5" };

export function buildActualOutcome(home: number, away: number): ActualOutcome {
  const result = home > away ? "home" : home < away ? "away" : "draw";
  const totalGoals = home + away;
  return {
    result,
    btts: home > 0 && away > 0 ? "yes" : "no",
    totalGoalsSide: totalGoals > 2 ? "over_2.5" : "under_2.5",
  };
}

/** Returns null (not false) for a market with no programmatic resolution --
 * callers MUST treat null as "unknown, skip" and never coerce it to a miss. */
export function marketCorrect(market: string, selection: string, actual: ActualOutcome): boolean | null {
  if (!RESOLVABLE_MARKETS.has(market)) return null;
  if (market === "result_3way") return selection === actual.result;
  if (market === "btts") return selection === actual.btts;
  return selection === actual.totalGoalsSide; // market === "total_goals"
}

export function bestMarket(match: Match): MarketRec | undefined {
  // Prefer an actually-recommended market (direct_bet/conditional) over a
  // no_bet one, even if a no_bet market happens to have a numerically
  // higher value_edge (a real case: a market can have positive edge and
  // still be no_bet if it's below min_value_edge, or an ineligible-for-
  // conditional market A54 downgraded) -- direct user report: a "No Bet"
  // card was showing a prominent positive "+3.2% EDGE" from exactly this
  // situation, reading as a good-looking bet that wasn't actually being
  // recommended. Falls back to ranking among all markets (including
  // no_bet) only when nothing is actionable at all.
  const actionable = match.markets.filter((m) => m.recommendationType !== "no_bet");
  const pool = actionable.length > 0 ? actionable : match.markets;
  return [...pool].sort((a, b) => b.valueEdge - a.valueEdge)[0];
}

/** Mockup point 3: backs Daily Edges' "N with positive edge" summary line.
 * Same predicate as the "Positive Edge" tag/green edge coloring on
 * MatchCard itself (recommendationType !== "no_bet" && valueEdge >= 0) --
 * kept as one shared function rather than a third inline copy of that
 * condition. */
export function hasPositiveEdge(match: Match): boolean {
  const m = bestMarket(match);
  return !!m && m.currentOdds != null && m.recommendationType !== "no_bet" && m.valueEdge >= 0;
}

/** W117: every row sharing bestMarket()'s own `market` name -- e.g. all
 * three of a result_3way's home/draw/away rows -- so MatchCard can show a
 * full "market + odds per direction" board instead of only the single
 * highest-edge selection. Deliberately shows every direction's raw price
 * (transparency), never a per-direction edge -- edge stays reserved for the
 * one actually-recommended selection (Selection + Edge row) so a plain,
 * unactioned price never reads as a second recommendation. */
export function marketDirections(match: Match, marketName: string): MarketRec[] {
  return match.markets.filter((m) => m.market === marketName);
}

const TEAM_COLORS: Record<string, { primary: string; secondary?: string }> = {
  Liverpool: { primary: "#C8102E" },
  Arsenal: { primary: "#EF0107", secondary: "#FFFFFF" },
  Chelsea: { primary: "#034694" },
  Brighton: { primary: "#0057B8", secondary: "#FFFFFF" },
  "Man City": { primary: "#6CABDD" },
  "Manchester City": { primary: "#6CABDD" },
  Fulham: { primary: "#FFFFFF", secondary: "#000000" },
  Tottenham: { primary: "#FFFFFF", secondary: "#132257" },
  "West Ham": { primary: "#7A263A", secondary: "#1BB1E7" },
  Newcastle: { primary: "#241F20", secondary: "#FFFFFF" },
  "Aston Villa": { primary: "#670E36", secondary: "#95BFE5" },
  "Man United": { primary: "#DA291C" },
  "Manchester United": { primary: "#DA291C" },
  // W61: Allsvenskan (Swedish top flight). Keys are the exact spelling The
  // Odds API returns for these fixtures (confirmed live, W55/W59) -- not
  // the ML engine's internal canonical short name (config/team_mapping.json),
  // which is only used for odds-matching and never rendered directly.
  "Malmo FF": { primary: "#6CACE4", secondary: "#FFFFFF" },
  AIK: { primary: "#000000", secondary: "#FFD700" },
  "Djurgardens IF": { primary: "#003D7A", secondary: "#6CACE4" },
  "Hammarby IF": { primary: "#046A38", secondary: "#FFFFFF" },
  "BK Hacken": { primary: "#FFD700", secondary: "#000000" },
  "IFK Goteborg": { primary: "#0057A0", secondary: "#FFFFFF" },
  // W80: La Liga (Spanish top flight). Keys are the exact `shortName`
  // football-data.org returns for these fixtures (confirmed live, W74/W76)
  // -- not the ML engine's internal canonical short name
  // (config/team_mapping.json), which is only used for odds/corpus
  // matching and never rendered directly. Mirrors W61's exact rationale.
  "Real Madrid": { primary: "#FFFFFF", secondary: "#00529F" },
  "Barça": { primary: "#A50044", secondary: "#004D98" },
  Atleti: { primary: "#CB3524", secondary: "#272E61" },
  "Sevilla FC": { primary: "#D00027", secondary: "#FFFFFF" },
};

const BADGE_FALLBACK_COLORS = ["#199e70", "#c98500", "#008300", "#9085e9", "#e66767", "#d55181", "#d95926"];
function badgeColor(name: string) {
  let hash = 0;
  for (let i = 0; i < name.length; i++) hash = (hash * 31 + name.charCodeAt(i)) >>> 0;
  return BADGE_FALLBACK_COLORS[hash % BADGE_FALLBACK_COLORS.length];
}
function teamColor(name: string) {
  return TEAM_COLORS[name] ?? { primary: badgeColor(name) };
}
function textColorFor(hex: string) {
  const c = hex.replace("#", "");
  const r = parseInt(c.substring(0, 2), 16) / 255;
  const g = parseInt(c.substring(2, 4), 16) / 255;
  const b = parseInt(c.substring(4, 6), 16) / 255;
  const luminance = 0.2126 * r + 0.7152 * g + 0.0722 * b;
  return luminance > 0.6 ? "#0b0b0b" : "#ffffff";
}
function initials(name: string) {
  const words = name.split(" ").filter(Boolean);
  if (words.length === 1) return words[0].slice(0, 3).toUpperCase();
  return words
    .slice(0, 3)
    .map((w) => w[0])
    .join("")
    .toUpperCase();
}

// ---------------------------------------------------------------------------
// Atoms
// ---------------------------------------------------------------------------

/** A match currently being played -- distinct from both "upcoming" (hasn't
 * kicked off) and "completed" (final score, betting closed). Sits alongside
 * the existing recommendation badge (StatusBadge/TrustSignal) rather than
 * replacing it -- "what was recommended pre-kickoff" and "this is happening
 * right now" are two different, both-relevant facts. status-critical (red)
 * is otherwise unused in this palette -- a natural fit for something this
 * urgent/real-time. No minute/clock shown -- not data this app has. */
function LiveBadge() {
  return (
    <span className="inline-flex items-center gap-1.5 rounded-md border border-critical/40 bg-critical/15 px-2 py-0.5 text-[11px] font-medium uppercase tracking-wide text-critical">
      <span className="h-1.5 w-1.5 animate-pulse rounded-full bg-critical" />
      LIVE
    </span>
  );
}

/** Whether the recommended market actually hit, once a match is completed.
 * hit === null (unresolvable market, e.g. corners) renders nothing --
 * marketCorrect's own contract: null means "unknown", never a miss.
 * Literal class strings per branch (not template-interpolated) -- Tailwind's
 * JIT scanner needs the exact class text present in source, same reason
 * STATUS_META/TrustSignal above never construct class names dynamically. */
function HitBadge({ hit }: { hit: boolean }) {
  if (hit) {
    return (
      <span className="inline-flex items-center gap-1.5 rounded-md border border-good/40 bg-good/15 px-2 py-0.5 text-[11px] font-medium uppercase tracking-wide text-good">
        <CheckCircle weight="fill" size={13} />
        Hit
      </span>
    );
  }
  return (
    <span className="inline-flex items-center gap-1.5 rounded-md border border-serious/40 bg-serious/15 px-2 py-0.5 text-[11px] font-medium uppercase tracking-wide text-serious">
      <XCircle weight="fill" size={13} />
      Not Hit
    </span>
  );
}

export function StatusBadge({ status, size = "sm" }: { status: Overall; size?: "sm" | "lg" }) {
  const s = STATUS_META[status];
  const pad = size === "lg" ? "px-3 py-1.5 text-sm" : "px-2 py-0.5 text-[11px]";
  return (
    <span
      title={s.explain}
      className={`inline-flex items-center gap-1.5 rounded-md border ${s.ring} ${s.fill} ${s.text} ${pad} font-medium`}
    >
      {s.icon}
      {s.label}
    </span>
  );
}

/** W15: a first-class trust signal, independent of predictionBasis/overall --
 * renders whenever cold_start_risk or unknown_team is true, even if
 * predictionBasis itself claims full team_history_and_market coverage.
 * Label shortened to match the filled-badge redesign -- the fuller
 * "-- no history"/"-- thin history" detail lives in the title tooltip
 * (below) instead of the visible label. */
function TrustSignal({ match, size = "sm" }: { match: Match; size?: "sm" | "lg" }) {
  if (!match.coldStartRisk && !match.unknownTeam) return null;
  const label = match.unknownTeam ? "Unseen team" : "Cold start";
  const pad = size === "lg" ? "px-3 py-1.5 text-sm" : "px-2 py-0.5 text-[11px]";
  return (
    <span
      className={`inline-flex items-center gap-1.5 rounded-md border border-warning/40 bg-warning/15 text-warning ${pad} font-medium`}
      title={
        // W107: plain-language first, raw figure second -- previously just
        // the bare `feature_completeness=0.71` figure with no explanation
        // of what it means.
        match.featureCompleteness !== null
          ? `How much real historical data this prediction is based on (feature_completeness=${match.featureCompleteness.toFixed(
              2
            )}, out of 1.00).`
          : undefined
      }
    >
      <WarningCircle weight="fill" size={size === "lg" ? 15 : 13} />
      {label}
    </span>
  );
}

export function TeamBadge({ name, size = "sm" }: { name: string; size?: "sm" | "lg" }) {
  const { primary, secondary } = teamColor(name);
  const dims = size === "lg" ? "h-9 w-9 text-xs" : "h-6 w-6 text-[9px]";
  return (
    <span
      className={`flex shrink-0 items-center justify-center rounded-full font-bold ${dims}`}
      style={{
        background: primary,
        color: textColorFor(primary),
        border: `1.5px solid ${secondary ?? "var(--border-hairline)"}`,
      }}
      aria-hidden="true"
    >
      {initials(name)}
    </span>
  );
}

// Curated per-league colors, same rationale/precedent as TEAM_COLORS above
// (W61/W80) -- a real club can share a fallback hash color with an unrelated
// entity without anyone noticing, but a league only ever has 6 known values
// (match_info.py's COMPETITION_ALLOWLIST) so there's no reason not to name
// them all explicitly.
const LEAGUE_COLORS: Record<string, { primary: string; secondary?: string }> = {
  E0: { primary: "#3D195B" }, // Premier League purple
  SP1: { primary: "#EE2737" }, // La Liga red
  SWE: { primary: "#006AA7", secondary: "#FECC02" }, // Allsvenskan (Sweden flag)
  I1: { primary: "#008C45" }, // Serie A green
  D1: { primary: "#7D1128" }, // Bundesliga maroon
  F1: { primary: "#00A19C" }, // Ligue 1 teal
};
function leagueColor(code: string) {
  return LEAGUE_COLORS[code] ?? { primary: badgeColor(code) };
}

/** A league's own identifying badge -- `rounded-md` (not TeamBadge's
 * `rounded-full`), so the two never read as the same kind of thing when
 * both appear on a card (the league bar above, team circles below). */
export function LeagueBadge({ code, size = "sm" }: { code: string; size?: "sm" | "lg" }) {
  const { primary, secondary } = leagueColor(code);
  const dims = size === "lg" ? "h-9 w-9 text-xs" : "h-6 w-6 text-[9px]";
  return (
    <span
      className={`flex shrink-0 items-center justify-center rounded-md font-bold ${dims}`}
      style={{
        background: primary,
        color: textColorFor(primary),
        border: `1.5px solid ${secondary ?? "var(--border-hairline)"}`,
      }}
      aria-hidden="true"
    >
      {initials(LEAGUE_LABEL[code] ?? code)}
    </span>
  );
}

function TierTag({ tier, tintIndex }: { tier: Tier; tintIndex?: number }) {
  // tintIndex is only ever passed by DashboardPage's date-grouped cards
  // (matching that group's own DATE_GROUP_WASHES index) -- Match Explorer's
  // ungrouped list omits it entirely, keeping the plain neutral style.
  const tint = tintIndex !== undefined ? TIER_TAG_TINTS[tintIndex % TIER_TAG_TINTS.length] : "border-border text-ink-secondary";
  return (
    <span
      title={TIER_EXPLANATION[tier]}
      className={`rounded border px-1.5 py-0.5 text-[10px] font-medium uppercase tracking-wide ${tint}`}
    >
      {TIER_LABEL[tier]}
    </span>
  );
}

function SegmentedControl<T extends string>({
  options,
  value,
  onChange,
}: {
  options: { value: T; label: string }[];
  value: T;
  onChange: (v: T) => void;
}) {
  return (
    <div className="flex items-center gap-4 border-b border-border">
      {options.map((opt) => (
        <button
          key={opt.value}
          type="button"
          onClick={() => onChange(opt.value)}
          className={`-mb-px border-b-2 px-0.5 py-2 text-sm font-medium transition-colors duration-150 ${
            value === opt.value ? "border-accent text-ink" : "border-transparent text-ink-secondary hover:text-ink"
          }`}
        >
          {opt.label}
        </button>
      ))}
    </div>
  );
}

/** W108 follow-up: a real switch matching the app's own atom conventions
 * (accent/border/surface tokens, 150ms transitions -- same as
 * SegmentedControl above), not a bare browser checkbox. Shared by Dashboard
 * and Match Explorer's "Actionable only" filter rather than duplicated. */
function Toggle({ checked, onChange, label }: { checked: boolean; onChange: (v: boolean) => void; label: string }) {
  return (
    <label className="flex select-none items-center gap-2 text-sm text-ink-secondary">
      <button
        type="button"
        role="switch"
        aria-checked={checked}
        onClick={() => onChange(!checked)}
        className={`relative h-5 w-9 shrink-0 rounded-full transition-colors duration-150 ${
          checked ? "bg-accent" : "border border-border-strong bg-surface"
        }`}
      >
        <span
          className={`absolute top-0.5 left-0.5 h-4 w-4 rounded-full bg-ink transition-transform duration-150 ${
            checked ? "translate-x-4" : "translate-x-0"
          }`}
        />
      </button>
      {label}
    </label>
  );
}

export function ErrorState({ message, onRetry }: { message: string; onRetry?: () => void }) {
  return (
    <div className="flex items-center gap-2 rounded-lg border border-serious/40 p-3.5 text-sm text-serious">
      <WarningCircle size={16} weight="fill" />
      <span className="flex-1">{message}</span>
      {onRetry && (
        <button type="button" onClick={onRetry} className="font-medium underline">
          Retry
        </button>
      )}
    </div>
  );
}

function LoadingRows({ count = 3 }: { count?: number }) {
  return (
    <div className="flex flex-col gap-2.5">
      {Array.from({ length: count }).map((_, i) => (
        <div key={i} className="h-[86px] animate-pulse rounded-lg border border-border bg-surface/50" />
      ))}
    </div>
  );
}

// ---------------------------------------------------------------------------
// MatchCard -- click to expand; expanding lazily triggers a live agent call
// if no recommendation has been generated for this fixture yet.
// ---------------------------------------------------------------------------

export function MatchCard({
  match,
  onUpdate,
  asOf = new Date(),
  sandboxMode = false,
  tintIndex,
}: {
  match: Match;
  onUpdate: (m: Match) => void;
  asOf?: Date;
  sandboxMode?: boolean;
  // Mockup point 5: the MODELED tag echoes its own date panel's wash color
  // rather than a fixed neutral gray -- only DashboardPage's date-grouped
  // cards pass this (matching that group's DATE_GROUP_WASHES index); Match
  // Explorer's ungrouped list omits it, keeping the plain style.
  tintIndex?: number;
}) {
  const [open, setOpen] = useState(false);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const isCompleted = match.status === "completed";
  const isLive = match.status === "live";
  const shown = bestMarket(match);
  // null covers "not completed yet", "no recommendation", and "market
  // unresolvable" (e.g. corners) identically -- HitBadge only renders for a
  // real true/false.
  const hit =
    isCompleted && match.hasRecommendation && shown && match.result
      ? marketCorrect(shown.market, shown.selection, buildActualOutcome(match.result.home, match.result.away))
      : null;
  // BUG-053 follow-up (direct user request): money actually won/lost on the
  // pick, in UB, replacing the Odds/Result box once a match completes.
  // Deliberately narrower than `hit !== null` alone -- a `conditional`
  // market ("wait for a better price") can still carry a non-null
  // unitBetMultiplier (schema.py's _attach_unit_bet_multiplier only
  // excludes no_bet, not conditional) despite never having been an actual
  // bet at currentOdds, so this also requires recommendationType ===
  // "direct_bet" specifically. profit = stake*(odds-1) on a hit, -stake on
  // a miss -- the same formula src/agent/staking.py's simulate_flat_stake/
  // simulate_kelly_stake and app/backend/bet_tracker.py's real settlement
  // all already use.
  const moneyWon =
    isCompleted && hit !== null && shown?.recommendationType === "direct_bet" &&
    shown.currentOdds != null && match.unitBetMultiplier != null
      ? hit
        ? match.unitBetMultiplier * (shown.currentOdds - 1)
        : -match.unitBetMultiplier
      : null;
  // The fallback list spans many different days (W46/W51's 90-day window),
  // so the day label must show on every card, not just ones with no market
  // to display -- previously `shown ? market/selection : day` hid it
  // entirely whenever a card had a recommendation.
  const day = formatDay(match.kickoffIso, asOf, sandboxMode);
  const market = shown ? marketLabel(shown.market) : null;

  async function handleExpand() {
    const next = !open;
    setOpen(next);
    if (next && !match.hasRecommendation && !loading) {
      setLoading(true);
      setError(null);
      const date = match.kickoffIso.slice(0, 10);
      try {
        // W47: check the precomputed cache (D2a) first -- only fall back to
        // the live "regenerate now" call on a real miss. A cache-check
        // failure is treated as a miss (not surfaced as an error) since
        // generateRecommendation below is still a fully valid fallback.
        let rec: MatchRecommendationOut | null = null;
        try {
          rec = await getCachedRecommendation(match.id, date);
        } catch {
          rec = null;
        }
        if (!rec) {
          rec = await generateRecommendation({
            home_team: match.home,
            away_team: match.away,
            date,
            league: match.league,
            match_id: match.id,
          });
        }
        onUpdate(applyRecommendation(match, rec));
      } catch (err) {
        setError(err instanceof ApiError ? err.message : "Could not reach the agent.");
      } finally {
        setLoading(false);
      }
    }
  }

  return (
    // W120 follow-up: bg-page (near-opaque) instead of bg-surface/40 -- needs
    // to read as its own distinct surface against the date panel's colored
    // gradient wash behind it, not blend into it.
    <div className="rounded-xl border border-border bg-page/80 transition-all duration-150 hover:-translate-y-px hover:border-border-strong">
      <button type="button" onClick={handleExpand} className="w-full p-4 text-left">
        {/* Direct user request: identify which league/country a card belongs
            to at a glance -- full-bleed via negative margins (undoing the
            button's own p-4) rather than restructuring around the button, so
            its background reaches the card's true edges. Renders on every
            MatchCard regardless of which page grouping wraps it (Dashboard
            groups by date, Match Explorer by league below) -- one change,
            both pages, since both render this same component. */}
        <div className="-mx-4 -mt-4 mb-3 flex items-center justify-between gap-2 rounded-t-xl border-b border-border bg-white/[0.02] px-4 py-2">
          <div className="flex items-center gap-2">
            <LeagueBadge code={match.league} />
            <span className="text-sm font-semibold text-ink">{LEAGUE_LABEL[match.league] ?? match.league}</span>
          </div>
          {LEAGUE_COUNTRY[match.league] && (
            <span className="text-xs text-ink-secondary">{LEAGUE_COUNTRY[match.league]}</span>
          )}
        </div>

        {/* Status badge(s) -- top-right corner, independent of the team/
            market body below rather than sharing a row with the tier tag
            (previous layout). Filled pills (STATUS_META.fill/TrustSignal's
            own bg-warning/15) match this redesign's visual language. */}
        <div className="flex items-center justify-end gap-1.5">
          {isLive && <LiveBadge />}
          {/* Completed: StatusBadge (the pre-match recommendation type) drops
              out of this row -- that's now stated in the footer instead
              ("Was a <label> pick"), since once the match is over what
              matters up here is FT + whether it actually hit, not what kind
              of pick it originally was. Upcoming/live unchanged: StatusBadge
              still leads there, nothing to resolve yet. */}
          {isCompleted && <span className="text-[11px] font-medium uppercase tracking-wide text-muted">FT</span>}
          {hit !== null && <HitBadge hit={hit} />}
          {match.hasRecommendation ? (
            <>
              <TrustSignal match={match} />
              {/* W153: the shown market's own recommendationType, not
                  match.overall -- this badge sits right next to the one
                  market this card actually displays (below), and must
                  describe *that* market, not a separate match-wide
                  aggregate that can legitimately differ from it (see
                  summarySentence's comment for the concrete scenario).
                  Falls back to match.overall only when bestMarket()
                  found nothing to show at all. */}
              {!isCompleted && <StatusBadge status={shown?.recommendationType ?? match.overall} />}
            </>
          ) : (
            <span className="rounded-md border border-border-strong bg-surface px-2 py-0.5 text-[11px] font-medium uppercase tracking-wide text-muted">
              {isCompleted ? "Settled" : "Not yet generated"}
            </span>
          )}
        </div>

        {/* 1. TEAM -- one horizontal row (direct mockup correction: W120's
            vertical home/"v"/away stack was a misread of the reference). */}
        <div className="mt-2 flex items-center gap-2">
          <TeamBadge name={match.home} size="lg" />
          <span className="truncate text-base font-semibold text-ink">{match.home}</span>
          <span className="shrink-0 text-sm text-ink-secondary">v</span>
          <TeamBadge name={match.away} size="lg" />
          <span className="truncate text-base font-semibold text-ink">{match.away}</span>
        </div>

        {/* Live/final score -- separate from the Market/Pick/Odds/Edge row
            below, deliberately: that row still shows the original
            pre-kickoff recommendation and its odds at generation time
            (unchanged, same as it already does for a completed match), not
            something that updates live -- this app has no in-play odds
            feed, only a live/final score (football-data.org updates
            home_goals/away_goals during and after play). Conflating the
            two in one number would imply the odds are live when they
            aren't. Extended to isCompleted (direct user request) once the
            Odds/Result box below became a money-won figure instead of the
            score -- this is now the only place a completed match's final
            score renders at all. */}
        {(isLive || isCompleted) && match.result && (
          <div className="mt-2 flex items-center justify-center gap-3 font-mono text-2xl font-bold text-ink">
            <span>{match.result.home}</span>
            <span className="text-muted">-</span>
            <span>{match.result.away}</span>
          </div>
        )}

        {/* MODELED tag + MARKET/PICK/ODDS/EDGE as one full-width row below
            the team row (direct mockup correction: W120 boxed the grid as a
            narrower side panel next to the team block instead). */}
        <div className="mt-3 flex flex-wrap items-start justify-between gap-4">
          <TierTag tier={match.tier} tintIndex={tintIndex} />

          <div className="flex min-w-0 flex-1 flex-wrap items-start justify-between gap-x-4 gap-y-3">
            <div className="min-w-0">
              <div className="text-[10px] uppercase tracking-wide text-muted">Market</div>
              <div className="truncate text-sm font-semibold text-ink">{market ? market.label : "—"}</div>
              {market?.subtitle && <div className="text-[10px] text-muted">{market.subtitle}</div>}
            </div>

            <div className="min-w-0">
              <div className="text-[10px] uppercase tracking-wide text-muted">Pick</div>
              <div className="flex items-center gap-1 text-sm font-semibold text-ink">
                {shown ? (
                  <>
                    {pickCaption(shown.selection) &&
                      (shown.selection.startsWith("under") ? (
                        <ArrowDown size={11} weight="bold" className="shrink-0 text-good" />
                      ) : (
                        <ArrowUp size={11} weight="bold" className="shrink-0 text-good" />
                      ))}
                    <span className={`truncate ${hit === false ? "line-through" : ""}`}>
                      {pickLabel(match, shown.selection)}
                    </span>
                    {pickCaption(shown.selection) && (
                      <span className="shrink-0 text-xs font-normal text-ink-secondary">
                        {pickCaption(shown.selection)}
                      </span>
                    )}
                  </>
                ) : (
                  "—"
                )}
              </div>
              {/* Inline echo of the same top-right HitBadge, right under the
                  specific pick it's about -- readable at a glance without
                  looking away from the Pick column. hit === null (no
                  recommendation, or an unresolvable market) renders nothing,
                  same contract as the top-right badge. */}
              {hit !== null && (
                <div className={`flex items-center gap-1 text-xs font-medium ${hit ? "text-good" : "text-serious"}`}>
                  {hit ? <CheckCircle weight="fill" size={11} /> : <XCircle weight="fill" size={11} />}
                  {hit ? "Hit" : "Not Hit"}
                </div>
              )}
            </div>

            <div className="shrink-0 text-right">
              {/* W84/A52: for a conditional market with a real targetOdds
                  (code-computed, src/agent/schema.py _compute_target_odds --
                  the price this market needs to reach to clear
                  min_value_edge), that's the number worth surfacing here,
                  not the current price the card already told the user isn't
                  good enough -- shown in the same warning color as the
                  Conditional badge. null covers "not applicable" and "no
                  such target exists" (e.g. A29's ceiling-downgrade case)
                  identically -- both fall back to the plain current-odds
                  display. */}
              {!isCompleted && shown?.recommendationType === "conditional" && shown.targetOdds != null ? (
                <>
                  <div className="text-[10px] uppercase tracking-wide text-warning">Wait ≥</div>
                  <div className="font-mono text-base font-bold text-warning">{shown.targetOdds.toFixed(2)}</div>
                  {/* Direct user feedback: the target alone doesn't say how
                      far off the current price is -- pairing it with the
                      live current_odds lets a reader gauge roughly how long
                      this might take to clear, the same way ProbabilityRow's
                      Model Probabilities table already shows both side by
                      side (further down this file). `> 0`, not `!= null` --
                      decimal odds are never <= 0 in reality; A66
                      (agent_user_stories.md) now code-enforces that
                      server-side going forward, but this guard also covers
                      an already-cached row from before that fix shipped
                      (confirmed live: a 0.0 current_odds rendered as a
                      literal "now 0.00"). */}
                  {shown.currentOdds != null && shown.currentOdds > 0 && (
                    <div className="font-mono text-[10px] text-ink-secondary">now {shown.currentOdds.toFixed(2)}</div>
                  )}
                </>
              ) : isCompleted ? (
                // Direct user request: the final score moved up next to the
                // team names (isLive already showed it there; this box now
                // shows money won/lost on the pick instead) -- "—" for
                // anything that was never an actual bet (conditional/no_bet)
                // or an unresolvable market (hit === null, e.g. corners),
                // same null-propagation contract HitBadge already uses.
                <>
                  <div className="text-[10px] uppercase tracking-wide text-muted">Money Won</div>
                  <div
                    className={`font-mono text-base font-bold ${
                      moneyWon == null ? "text-muted" : moneyWon > 0 ? "text-good" : moneyWon < 0 ? "text-serious" : "text-ink"
                    }`}
                  >
                    {moneyWon != null ? formatMoneyWon(moneyWon) : "—"}
                  </div>
                </>
              ) : (
                <>
                  <div className="text-[10px] uppercase tracking-wide text-muted">Odds</div>
                  <div className="font-mono text-base font-bold text-ink">
                    {shown?.currentOdds ? shown.currentOdds.toFixed(2) : "—"}
                  </div>
                  {shown?.currentOdds != null && (
                    <span className="mt-1 inline-block rounded border border-border px-1.5 py-0.5 text-[10px] text-ink-secondary">
                      Decimal
                    </span>
                  )}
                </>
              )}
            </div>

            <div className="shrink-0 text-right">
              <div title={EDGE_EXPLAIN} className="text-[10px] uppercase tracking-wide text-muted">Edge</div>
              <div
                className={`font-mono text-base font-bold ${
                  isCompleted
                    ? // Plain, not green -- "positive edge" reads as "this is
                      // still worth acting on", which is nonsensical once
                      // the match is decided. This is a historical fact now.
                      "text-ink"
                    : shown?.currentOdds
                    ? shown.recommendationType !== "no_bet" && shown.valueEdge >= 0
                      ? "text-good"
                      : "text-ink-secondary"
                    : "text-muted"
                }`}
              >
                {shown?.currentOdds ? formatEdge(shown.valueEdge) : "—"}
              </div>
              {isCompleted ? (
                shown?.currentOdds != null && (
                  <span className="mt-1 inline-block rounded-full border border-border-strong bg-surface px-1.5 py-0.5 text-[10px] text-muted">
                    Pre-match edge
                  </span>
                )
              ) : (
                shown?.currentOdds != null && shown.recommendationType !== "no_bet" && shown.valueEdge >= 0 && (
                  <span className="mt-1 inline-block rounded-full border border-good/40 bg-good/10 px-1.5 py-0.5 text-[10px] text-good">
                    Positive Edge
                  </span>
                )
              )}
            </div>

            {/* A82/W169: Kelly-derived suggested stake for the actual pick,
                in UB (an abstract unit -- see the Daily Edges header
                explainer, not a dollar figure). Its own column, same
                weight as Market/Pick/Odds/Edge -- direct user feedback
                that a small aside line under Pick undersold it.
                BUG-053: previously hidden once completed ("nothing left to
                size a stake for") -- kept visible instead, mirroring the
                Edge column's own precedent for the identical case (plain
                value, a "Pre-match stake" badge instead of disappearing). */}
            {match.unitBetMultiplier != null && (
              <div className="shrink-0 text-right">
                <div className="text-[10px] uppercase tracking-wide text-muted">Stake</div>
                <div className="font-mono text-base font-bold text-ink">
                  {match.unitBetMultiplier.toFixed(1)} UB
                </div>
                {isCompleted && (
                  <span className="mt-1 inline-block rounded-full border border-border-strong bg-surface px-1.5 py-0.5 text-[10px] text-muted">
                    Pre-match stake
                  </span>
                )}
              </div>
            )}
          </div>
        </div>

        {/* Closing row: day/time (icon + bullet-separated, mockup point 4 --
            was gap-only spacing, no visible "•") + odds source/chevron. */}
        <div className="mt-3 flex items-center justify-between gap-2 border-t border-border pt-2.5">
          {/* day/time are separate text nodes (not one interpolated string)
              so "today"/"tomorrow" etc. stay independently matchable -- a
              single combined node isn't findable by an exact-text query
              once other text shares the node (RTL matches per-node, not
              substrings). */}
          <div className="flex items-center gap-1.5 text-xs text-ink-secondary">
            <span className="flex items-center gap-1">
              <CalendarBlank size={12} />
              <span>{day}</span>
            </span>
            <span className="text-muted">•</span>
            {isCompleted ? (
              // A clock time reads as "this is when it kicks off" -- wrong
              // tense for a match that's already over.
              <span className="flex items-center gap-1">
                <Clock size={12} />
                <span>Full Time</span>
              </span>
            ) : (
              <span className="flex items-center gap-1">
                <Clock size={12} />
                <span>{formatKickoff(match.kickoffIso)}</span>
              </span>
            )}
          </div>
          <span className="flex items-center gap-2">
            {/* StatusBadge dropped out of the top-right badge row for a
                completed match (replaced by FT/HitBadge there) -- restated
                here instead, since "what kind of pick this was" is still
                worth knowing once the match is over. */}
            {isCompleted && match.hasRecommendation && (
              // W153: same shown-market-not-match.overall reasoning as the
              // top badge above -- "what kind of pick this was" must
              // describe the specific market this card showed pre-match.
              <span className="text-[10px] text-muted">Was a {STATUS_META[shown?.recommendationType ?? match.overall].label} pick</span>
            )}
            {shown?.currentOdds != null && <span className="text-[10px] text-muted">via The Odds API</span>}
            <CaretDown
              size={14}
              className={`text-ink-secondary transition-transform duration-150 ${open ? "rotate-180" : ""}`}
            />
          </span>
        </div>
      </button>

      <div className={`expand-rows ${open ? "is-open" : ""}`}>
        <div>
          <div className="border-t border-border p-3.5 text-sm">
            {loading && <LoadingRows count={1} />}
            {error && <ErrorState message={error} onRetry={handleExpand} />}
            {!loading && !error && match.hasRecommendation && (
              <>
                <ul className="space-y-1 text-ink-secondary">
                  {match.explanation.map((point, i) => (
                    <li key={i} className="flex gap-1.5">
                      <span aria-hidden="true">·</span>
                      <span>{point}</span>
                    </li>
                  ))}
                </ul>
                {match.invalidMarketCount > 0 && (
                  <p className="mt-2 flex items-center gap-1.5 text-xs text-serious">
                    <WarningCircle weight="fill" size={13} />
                    {match.invalidMarketCount} market{match.invalidMarketCount > 1 ? "s" : ""} omitted --
                    malformed data.
                  </p>
                )}
                <Link
                  href={`/matches/${match.id}?home=${encodeURIComponent(match.home)}&away=${encodeURIComponent(
                    match.away
                  )}&date=${match.kickoffIso.slice(0, 10)}&league=${encodeURIComponent(match.league)}`}
                  className="mt-3 inline-flex items-center gap-1 text-sm font-medium text-accent"
                >
                  Full analysis <CaretRight size={12} />
                </Link>
              </>
            )}
          </div>
        </div>
      </div>
    </div>
  );
}

// ---------------------------------------------------------------------------
// Page 1 -- Dashboard ("/"): today's real E0 fixtures.
// ---------------------------------------------------------------------------

export function DashboardPage() {
  // AppShell (below) independently calls this same hook too -- see its own
  // comment. Known duplicate fetch, not shared/cached; accepted for now.
  const { asOf, sandboxMode } = useSandboxAsOf();
  const [matches, setMatches] = useState<Match[] | null>(null);
  const [error, setError] = useState<string | null>(null);
  // W42: bumped by the retry button to force a fresh load() run through the
  // same cancellation guard below, rather than calling load() imperatively
  // from outside the effect (which would have no way to invalidate an
  // in-flight request from a *previous* run if the two race).
  const [retryTick, setRetryTick] = useState(0);
  const [sort, setSort] = useState<MatchSort>("kickoff");
  // W108: hide No Bet / Insufficient Data matches, showing only actionable
  // (Direct Bet / Conditional) ones -- direct feedback that non-actionable
  // rows can't be filtered out today, only reordered.
  const [actionableOnly, setActionableOnly] = useState(false);
  // Whether the mobile-only rail overlay drawer (Edge Distribution/Top
  // Edges) is open -- triggered from AppShell's search-bar row, not an
  // inline section of the page itself. Unaffected at `lg` and up, where
  // the rail is the permanent sticky side-by-side column it always was.
  const [railOpen, setRailOpen] = useState(false);

  useEffect(() => {
    let cancelled = false;

    async function load() {
      setError(null);
      setMatches(null);
      try {
        // Always the next 10 matches going forward from asOf, regardless of
        // how many (if any) fall on asOf's own date -- a 90-day forward
        // window, the same convention MatchExplorerPage's search already
        // uses (this codebase's established precedent for "how far to look
        // for the next real fixtures"). W51: scripts/launch_sandbox.py's
        // fetch_sandbox_fixtures() mirrors this exact window/sort/cap (90
        // days forward, sorted kickoff-ascending, capped at 10) so
        // --precompute actually covers what the Dashboard shows -- if this
        // window, sort, or cap ever changes, update that Python copy too,
        // there is no shared implementation.
        const today = dateString(asOf, sandboxMode);
        const to = addDays(asOf, 90, sandboxMode);
        const fixtures = await getFixtures(today, dateString(to, sandboxMode));
        if (cancelled) return;
        const nearest = fixtures
          .map((f) => fixtureToMatch(f, asOf, sandboxMode))
          // Direct user request: a live match, or one completed earlier
          // today, stays in the same list as upcoming ones -- MatchCard
          // itself renders the difference (LiveBadge/score row, or the
          // final score + Hit/Missed badge for a completed one). Every
          // other day in this forward-only window is still upcoming-only:
          // completed matches from any day but today are excluded --
          // checked explicitly via dayDiff rather than relying on the fetch
          // window's own forward-only shape to imply it (defensive: correct
          // even if that window's start date ever changes). Was previously
          // `m.status === "upcoming"` only -- a strict allowlist that (before
          // "live" existed as a status at all) also silently excluded live
          // matches, not just completed ones.
          .filter((m) => m.status !== "completed" || dayDiff(m.kickoffIso, asOf, sandboxMode) === 0)
          // API ordering isn't guaranteed -- sort so "next 10" is actually
          // nearest-first before trimming. ISO 8601 strings sort correctly
          // as strings.
          .sort((a, b) => a.kickoffIso.localeCompare(b.kickoffIso))
          .slice(0, 10);
        // W53: resolve the precomputed cache for the (already-capped-to-10)
        // list before rendering -- an additional await in this same guarded
        // run, so re-check `cancelled` again before touching state.
        const resolvedMatches = await resolveCachedRecommendations(nearest);
        if (cancelled) return;
        setMatches(resolvedMatches);
      } catch (err) {
        if (!cancelled) setError(err instanceof ApiError ? err.message : "Could not load fixtures.");
      }
    }

    load();
    return () => {
      cancelled = true;
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [asOf, retryTick]);

  function updateMatch(updated: Match) {
    setMatches((prev) => prev?.map((m) => (m.id === updated.id ? updated : m)) ?? null);
  }

  const shownMatches = matches ?? [];
  // Mockup point 3: "N with positive edge" -- same predicate the
  // "Positive Edge" tag on each card itself uses.
  const positiveEdgeCount = shownMatches.filter(hasPositiveEdge).length;
  // W108: the rail (Edge Distribution/Top Edges) stays computed over the
  // full loaded set regardless of this filter -- it's a display concern for
  // the list only, not a re-scoping of what "loaded" means.
  const visibleMatches = actionableOnly ? shownMatches.filter(isActionable) : shownMatches;
  // Date-group order always follows kickoff order, regardless of the
  // Kickoff/Edge % toggle below -- an edge-sorted list would scramble which
  // day each group appears under. The toggle still reorders matches within
  // each date group.
  const dateGroups = groupByDate(sortMatches(visibleMatches, "kickoff"), asOf, sandboxMode).map((group) => ({
    ...group,
    matches: sort === "edge" ? sortMatches(group.matches, "edge") : group.matches,
  }));

  return (
    <>
      <AppShell
        active="dashboard"
        // Direct feedback: on small screens the rail should "squish with
        // the top part with the search bar", not be its own section --
        // this trigger opens the mobile-only overlay drawer below instead
        // of the old bottom "Insights" accordion.
        railTrigger={
          shownMatches.length > 0 && (
            <button
              type="button"
              onClick={() => setRailOpen(true)}
              aria-label="Open insights"
              className="flex h-9 w-9 shrink-0 items-center justify-center rounded-lg border border-border bg-surface text-ink-secondary"
            >
              <ChartBar size={16} />
            </button>
          )
        }
      >
        {/* BUG-047: lg:-mt-8 cancels out <main>'s own inner pt-8 wrapper
            (AppShell.tsx) so this row's natural (unscrolled) position sits
            flush with <main>'s true scrollport top -- otherwise the sticky
            children below only reach their stuck position (top-0) after the
            user has already scrolled past that 32px gap, reading as a
            laggy "settles into place" jump instead of being stationary from
            the first pixel of scroll. Each sticky child gets its own
            lg:pt-8 back, both to restore that visual breathing room and so
            its own opaque background covers the reclaimed space (the
            BUG-046 fix this depends on). */}
        <div className="lg:-mt-8 lg:flex lg:items-start lg:gap-6">
          <div className="min-w-0 flex-1">
            {/* Direct feedback: title/subtitle/toggle stay stationary while
                only the match list below scrolls -- sticky within <main>'s
                own scroll region (AppShell.tsx), not the whole page. */}
            <div className="flex flex-wrap items-center justify-between gap-4 lg:sticky lg:top-0 lg:z-10 lg:bg-page lg:pb-4 lg:pt-8">
              <div>
                <h1 className="text-xl font-semibold tracking-tight text-ink">Daily Edges</h1>
                {/* Mockup point 3: a live stat summary, not the old static
                    subtitle W119 removed -- only once matches have actually
                    loaded (nothing to summarize before then). */}
                {shownMatches.length > 0 && (
                  <p className="mt-0.5 text-sm text-ink-secondary">
                    {shownMatches.length} match{shownMatches.length === 1 ? "" : "es"} · {positiveEdgeCount} with
                    positive edge
                  </p>
                )}
                {/* W169: static, no API call -- UB is an abstract betting
                    unit (A82), not a dollar figure, so there's nothing to
                    fetch here, just an explanation of the convention. */}
                <p className="mt-0.5 text-xs text-ink-secondary">
                  UB = Unit Bet, your standard bet amount — the money you'd put on a 50/50 match bet.
                </p>
              </div>
              {/* Edge % sort hidden (2026-08-13, W118) -- flagged as misleading
                  by direct user feedback. Kickoff is the only sort left, so the
                  toggle itself (nothing left to toggle between) is hidden too,
                  not just the option -- `sort` state and `sortMatches`'s
                  "edge" case (dashboardMetrics.ts) are untouched, so restoring
                  the SegmentedControl below is a one-line revert. */}
              {shownMatches.length > 0 && (
                <Toggle checked={actionableOnly} onChange={setActionableOnly} label="Actionable only" />
              )}
            </div>

            <div className="mt-6">
              {error && <ErrorState message={error} onRetry={() => setRetryTick((t) => t + 1)} />}
              {!error && matches === null && <LoadingRows />}
              {!error && matches !== null && matches.length === 0 && (
                <p className="py-8 text-center text-sm text-ink-secondary">No upcoming fixtures.</p>
              )}
              {!error && shownMatches.length > 0 && visibleMatches.length === 0 && (
                <p className="py-8 text-center text-sm text-ink-secondary">
                  No actionable matches right now -- try turning off "Actionable only".
                </p>
              )}
              {!error && visibleMatches.length > 0 && (
                // W120 follow-up: back to a wrapping panel per date group
                // (superseding the dashed-rule-only treatment), now with a
                // rotating colored gradient wash distinguishing one date from
                // the next, plus its own calendar icon -- direct mockup.
                <div className="flex flex-col gap-6">
                  {dateGroups.map((group, i) => (
                    <div
                      key={group.dateKey}
                      className={`rounded-2xl border border-white/5 bg-gradient-to-br p-4 ${
                        DATE_GROUP_WASHES[i % DATE_GROUP_WASHES.length]
                      }`}
                    >
                      <div className="mb-3 flex items-center justify-between gap-3">
                        <div className="flex items-center gap-3">
                          <h2 className="text-lg font-bold tracking-tight text-ink">{group.label}</h2>
                          <span className="rounded-full border border-border-strong px-2 py-0.5 text-xs text-ink-secondary">
                            {group.matches.length} match{group.matches.length === 1 ? "" : "es"}
                          </span>
                        </div>
                        <CalendarBlank size={16} className="text-muted" aria-hidden="true" />
                      </div>
                      <div className="flex flex-col gap-2.5">
                        {group.matches.map((m) => (
                          <MatchCard
                            key={m.id}
                            match={m}
                            onUpdate={updateMatch}
                            asOf={asOf}
                            sandboxMode={sandboxMode}
                            tintIndex={i}
                          />
                        ))}
                      </div>
                    </div>
                  ))}
                </div>
              )}
            </div>
          </div>

          {/* Desktop-only permanent rail, now sticky (stays put while the
              match list scrolls past) instead of just side-by-side in
              normal flow. Below `lg` it's not rendered at all here -- moved
              entirely into the overlay drawer beneath, opened via
              AppShell's railTrigger slot next to the search bar. */}
          {shownMatches.length > 0 && (
            <div className="hidden lg:sticky lg:top-0 lg:block lg:border-l lg:border-border lg:pl-6 lg:pt-8">
              <DashboardRail matches={shownMatches} />
            </div>
          )}
        </div>
      </AppShell>

      {/* Mobile-only overlay drawer for the rail -- mirrors AppShell's own
          left-side menu drawer (W127), sliding from the right instead. */}
      {railOpen && (
        <>
          <div
            className="fixed inset-0 z-40 bg-page/70 lg:hidden"
            onClick={() => setRailOpen(false)}
            aria-hidden="true"
          />
          <div className="fixed inset-y-0 right-0 z-50 w-72 max-w-[85vw] overflow-y-auto bg-surface p-5 shadow-xl lg:hidden">
            <div className="flex items-center justify-between">
              <h2 className="text-sm font-semibold tracking-tight text-ink">Insights</h2>
              <button
                type="button"
                onClick={() => setRailOpen(false)}
                aria-label="Close insights"
                className="text-ink-secondary"
              >
                <X size={20} />
              </button>
            </div>
            <div className="mt-4">
              <DashboardRail matches={shownMatches} />
            </div>
          </div>
        </>
      )}
    </>
  );
}

// ---------------------------------------------------------------------------
// Page 2 -- Match Explorer ("/matches"): search across a wider fixture window.
// ---------------------------------------------------------------------------

export function MatchExplorerPage() {
  const { asOf, sandboxMode } = useSandboxAsOf();
  const [query, setQuery] = useState("");
  const [matches, setMatches] = useState<Match[] | null>(null);
  const [error, setError] = useState<string | null>(null);
  // W42: bumped by the retry button to force a fresh load() run through the
  // same cancellation guard below, rather than calling load() imperatively
  // from outside the effect (which would have no way to invalidate an
  // in-flight request from a *previous* run if the two race).
  const [retryTick, setRetryTick] = useState(0);
  // W108: same actionable-only filter as Dashboard, same shared predicate.
  const [actionableOnly, setActionableOnly] = useState(false);

  useEffect(() => {
    let cancelled = false;

    async function load() {
      setError(null);
      setMatches(null);
      try {
        // Widened from 30 to 90 days after live verification showed the
        // off-season gap between fixture windows can exceed 30 days (e.g.
        // 2026-07-11 -> next real fixture 2026-08-21, 41 days out).
        // dateString()/addDays() branch on sandboxMode the same way dayDiff
        // already does -- UTC getters when asOf really is UTC midnight
        // (sandbox mode, W30), local getters when it's a real browser
        // instant (live mode). The previous version here always used UTC
        // (.toISOString()/setUTCDate), asserting "asOf is UTC midnight" as
        // if that held unconditionally -- true only in sandbox mode, and
        // wrong for roughly a third of every day for a live non-UTC viewer,
        // which silently excluded today's own fixtures/results from the
        // window (confirmed live: a same-day finished match went missing
        // entirely).
        const from = dateString(asOf, sandboxMode);
        const to = dateString(addDays(asOf, 90, sandboxMode), sandboxMode);
        const fixtures = await getFixtures(from, to);
        if (cancelled) return;
        const initialMatches = fixtures.map((f) => fixtureToMatch(f, asOf, sandboxMode));
        // W53: unlike Dashboard's two call sites (each capped at 10 --
        // "today" is one E0 matchday, and the W46 fallback is explicitly
        // sliced to 10), this 90-day search window can realistically return
        // 50-100+ fixtures in-season. Blocking first paint on every one of
        // those cache checks resolving would queue behind the browser's
        // per-origin concurrent-connection cap (~6 for HTTP/1.1) -- N=50-100
        // becomes ~10-17 sequential batches, making this page *slower* to
        // first paint than before this story, the opposite of its goal.
        // Render the list immediately (unblocked), then patch precomputed
        // results in via a follow-up setMatches once the bulk check
        // resolves in the background -- still behind the same `cancelled`
        // guard so a superseded run can't clobber a later one's state.
        setMatches(initialMatches);
        resolveCachedRecommendations(initialMatches).then((resolvedMatches) => {
          if (!cancelled) setMatches(resolvedMatches);
        });
      } catch (err) {
        if (!cancelled) setError(err instanceof ApiError ? err.message : "Could not load fixtures.");
      }
    }

    load();
    return () => {
      cancelled = true;
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [asOf, retryTick]);

  function updateMatch(updated: Match) {
    setMatches((prev) => prev?.map((m) => (m.id === updated.id ? updated : m)) ?? null);
  }

  const rows = useMemo(() => {
    const q = query.trim().toLowerCase();
    let result = matches;
    if (!result) return null;
    if (q.length > 0) {
      result = result.filter((m) => m.home.toLowerCase().includes(q) || m.away.toLowerCase().includes(q));
    }
    // W108: applied after the team-name search, same "narrow what's shown"
    // relationship the search itself already has to the loaded window.
    if (actionableOnly) {
      result = result.filter(isActionable);
    }
    return result;
  }, [matches, query, actionableOnly]);

  // Direct user request: league section headers, since this page has no
  // grouping at all today -- mirrors DashboardPage's own date-group panel
  // (same DATE_GROUP_WASHES/TIER_TAG_TINTS rotation, wrapping-panel shape),
  // just grouped by league instead of date. groupByLeague() already existed
  // (dashboardMetrics.ts) but had never been wired into a page.
  const leagueGroups = useMemo(() => (rows ? groupByLeague(rows) : null), [rows]);

  return (
    <AppShell active="matches">
      <h1 className="text-xl font-semibold tracking-tight text-ink">Match Explorer</h1>
      <p className="mt-1 text-sm text-ink-secondary">Search real upcoming fixtures (next 90 days).</p>

      <div className="relative mt-5">
        <MagnifyingGlass size={16} className="pointer-events-none absolute left-3 top-1/2 -translate-y-1/2 text-muted" />
        <input
          value={query}
          onChange={(e) => setQuery(e.target.value)}
          placeholder="Search by team name…"
          className="w-full rounded-lg border border-border bg-surface py-2 pl-9 pr-3 text-sm text-ink outline-none placeholder:text-muted focus:border-accent"
        />
      </div>

      <div className="mt-3">
        <Toggle checked={actionableOnly} onChange={setActionableOnly} label="Actionable only" />
      </div>

      <div className="mt-6">
        {error && <ErrorState message={error} onRetry={() => setRetryTick((t) => t + 1)} />}
        {!error && rows === null && <LoadingRows />}
        {!error && rows !== null && rows.length === 0 && (
          <p className="py-8 text-center text-sm text-ink-secondary">
            {actionableOnly ? 'No actionable matches right now -- try turning off "Actionable only".' : "No matches found."}
          </p>
        )}
        {!error && rows && rows.length > 0 && leagueGroups && (
          <div className="flex flex-col gap-6">
            {leagueGroups.map((group, i) => (
              <div
                key={group.league}
                className={`rounded-2xl border border-white/5 bg-gradient-to-br p-4 ${
                  DATE_GROUP_WASHES[i % DATE_GROUP_WASHES.length]
                }`}
              >
                <div className="mb-3 flex items-center justify-between gap-3">
                  <div className="flex items-center gap-3">
                    <h2 className="text-lg font-bold tracking-tight text-ink">{group.label}</h2>
                    <span className="rounded-full border border-border-strong px-2 py-0.5 text-xs text-ink-secondary">
                      {group.matches.length} match{group.matches.length === 1 ? "" : "es"}
                    </span>
                  </div>
                  <Trophy size={16} className="text-muted" aria-hidden="true" />
                </div>
                <div className="flex flex-col gap-2.5">
                  {group.matches.map((m) => (
                    <MatchCard key={m.id} match={m} onUpdate={updateMatch} asOf={asOf} sandboxMode={sandboxMode} tintIndex={i} />
                  ))}
                </div>
              </div>
            ))}
          </div>
        )}
      </div>
    </AppShell>
  );
}

// ---------------------------------------------------------------------------
// Page 3 -- Match Analysis & Agent Intelligence ("/matches/:id")
// ---------------------------------------------------------------------------

/** W16: a direct_bet market with no current_odds is a known agent-output
 * quirk (agent_techspec.md §18.3 / BUG-013) -- A28 downgrades this at
 * extraction time, but the app shouldn't assume that fix holds for every
 * recommendation it ever sees (e.g. one cached before A28 shipped). Render
 * it as an explicit data-issue state, not a normal green "Direct Bet". */
function isAnomalousDirectBet(m: MarketRec): boolean {
  return m.recommendationType === "direct_bet" && m.currentOdds === null;
}

/** W12: from-recommendation bet logging -- every field but stake is locked
 * to the given market/selection within the recommendation snapshot. */
export function LogBetButton({
  matchId,
  recommendation,
  market,
  selection,
}: {
  matchId: string;
  recommendation: MatchRecommendationOut;
  market: string;
  selection: string;
}) {
  const [open, setOpen] = useState(false);
  const [stake, setStake] = useState("");
  const [status, setStatus] = useState<"idle" | "saving" | "done" | "error">("idle");
  const [errorMsg, setErrorMsg] = useState("");

  async function submit() {
    const parsedStake = parseFloat(stake);
    if (!parsedStake || parsedStake <= 0) {
      setStatus("error");
      setErrorMsg("Enter a stake greater than 0.");
      return;
    }
    setStatus("saving");
    try {
      await logBetFromRecommendation({ match_id: matchId, recommendation, market, selection, stake: parsedStake });
      setStatus("done");
    } catch (err) {
      setStatus("error");
      setErrorMsg(err instanceof ApiError ? err.message : "Could not log bet.");
    }
  }

  if (status === "done") return <span className="text-xs text-good">Logged</span>;

  if (!open) {
    return (
      <button type="button" onClick={() => setOpen(true)} className="text-xs font-medium text-accent">
        Log bet
      </button>
    );
  }

  return (
    <span className="flex flex-wrap items-center gap-1.5">
      <input
        value={stake}
        onChange={(e) => setStake(e.target.value)}
        placeholder="Stake"
        inputMode="decimal"
        className="w-16 rounded border border-border bg-surface px-1.5 py-0.5 text-xs text-ink outline-none focus:border-accent"
      />
      <button
        type="button"
        onClick={submit}
        disabled={status === "saving"}
        className="text-xs font-medium text-accent disabled:opacity-50"
      >
        {status === "saving" ? "…" : "Confirm"}
      </button>
      {status === "error" && <span className="text-xs text-serious">{errorMsg}</span>}
    </span>
  );
}

function ProbabilityRow({
  m,
  matchId,
  recommendation,
}: {
  m: MarketRec;
  matchId?: string;
  recommendation?: MatchRecommendationOut;
}) {
  const anomalous = isAnomalousDirectBet(m);
  const s = STATUS_META[m.recommendationType];
  return (
    <div className="grid grid-cols-[1fr_auto_auto_auto_auto] items-center gap-4 border-b border-border py-3 text-sm last:border-b-0">
      <span className="flex flex-col gap-1 truncate text-ink">
        <span className="truncate">
          {m.market} · {m.selection}
        </span>
        {/* W84/A52: targetOdds is code-computed (src/agent/schema.py
            _compute_target_odds) -- the price this market would need to
            reach to clear min_value_edge. null covers both "not applicable"
            (not conditional, no current_odds) and "no such target exists"
            (e.g. A29's ceiling-downgrade case) -- neither has a coherent
            condition to state. Same warning color as the Conditional badge
            itself (STATUS_META.conditional.text), so the two visually read
            as one signal. */}
        {m.recommendationType === "conditional" && m.targetOdds != null && (
          <span className={`font-mono text-xs ${STATUS_META.conditional.text}`}>
            Needs {m.targetOdds.toFixed(2)}+ to clear edge
          </span>
        )}
        {/* Log-bet UI hidden (2026-08-13, W115) -- bet tracking isn't built
            out enough to surface yet, same call as W106 hiding the Bets nav
            tab. LogBetButton itself and its backend path are untouched;
            uncomment below to re-enable once ready.
        {matchId && recommendation && !anomalous && (
          <LogBetButton matchId={matchId} recommendation={recommendation} market={m.market} selection={m.selection} />
        )}
        */}
      </span>
      <span className="text-right font-mono text-ink">{formatPct(m.mlProbability)}</span>
      <span className={`text-right font-mono ${anomalous ? "text-serious" : "text-ink-secondary"}`}>
        {m.currentOdds ? m.currentOdds.toFixed(2) : anomalous ? "missing" : "—"}
      </span>
      <span
        className={`text-right font-mono ${
          m.recommendationType !== "no_bet" && m.valueEdge >= 0 ? "text-good" : "text-ink-secondary"
        }`}
      >
        {formatEdge(m.valueEdge)}
      </span>
      {anomalous ? (
        <span className="justify-self-end text-serious" title="direct_bet with no current_odds -- data issue, not a real recommendation">
          Data issue
        </span>
      ) : (
        <span className={`justify-self-end ${s.text}`}>{s.label}</span>
      )}
    </div>
  );
}

/** W117-adjacent naming convention, W111 itself: "home"/"away"/"draw" alone
 * are readable but naming the actual team is clearer for a reader with no
 * prior betting vocabulary -- the whole point of this sentence. Any other
 * market's selection (e.g. "over_2.5") falls back to its raw string with
 * underscores turned into spaces, rather than a hand-maintained label for
 * every possible market. */
function selectionLabel(match: Match, selection: string): string {
  if (selection === "home") return match.home;
  if (selection === "away") return match.away;
  if (selection === "draw") return "a draw";
  return selection.replace(/_/g, " ");
}

/** MatchCard's Market/Pick/Odds/Edge grid needs a standalone, capitalized
 * label ("Draw", "Over 2.5") rather than selectionLabel()'s sentence-
 * embedded phrasing ("a draw") -- same underlying mapping, different
 * display context, so kept as its own small formatter instead of adding a
 * mode flag to the other one. */
function pickLabel(match: Match, selection: string): string {
  if (selection === "draw") return "Draw";
  const label = selectionLabel(match, selection);
  return label.charAt(0).toUpperCase() + label.slice(1);
}

/** A short direction word shown next to the pick -- "To Win" for a team
 * selection, "Over"/"Under" for a totals line. Draw has no direction to
 * name, so returns null (also gates whether the direction arrow renders,
 * up for Over/To Win, down for Under). */
function pickCaption(selection: string): string | null {
  if (selection === "home" || selection === "away") return "To Win";
  if (selection.startsWith("over")) return "Over";
  if (selection.startsWith("under")) return "Under";
  return null;
}

// W121 follow-up (mockup point 3): human-readable market names, not the raw
// backend string (`shown.market` was previously rendered verbatim -- a
// reader would have seen "result_3way"/"total_goals" literally). Covers
// only the five real markets the agent actually emits (src/agent/schema.py
// MarketRecommendation.market Literal) -- an unrecognized market string
// (future market type) falls back to a generic humanization rather than
// silently mislabeling it as one of these five.
const MARKET_LABEL: Record<string, { label: string; subtitle: string }> = {
  result_3way: { label: "3-Way Result", subtitle: "Full Time" },
  total_goals: { label: "Over/Under", subtitle: "Full Time" },
  btts: { label: "Both Teams to Score", subtitle: "Full Time" },
  home_corners: { label: "Home Corners", subtitle: "Full Time" },
  away_corners: { label: "Away Corners", subtitle: "Full Time" },
};
// W174: exported so AgentPerformanceDashboard.tsx can reuse the same
// human-readable market names ("3-Way Result" instead of "result_3way")
// instead of duplicating MARKET_LABEL.
export function marketLabel(market: string): { label: string; subtitle: string | null } {
  if (MARKET_LABEL[market]) return MARKET_LABEL[market];
  const spaced = market.replace(/_/g, " ");
  return { label: spaced.charAt(0).toUpperCase() + spaced.slice(1), subtitle: null };
}

/** W111: one plain-English sentence, composed entirely from fields already
 * on the recommendation (overall/confidence/bestMarket) -- no new backend
 * field, no LLM call. Sits ahead of the jargon-dense Model Probabilities
 * table so a reader with zero betting vocabulary has something to read
 * before the numbers.
 *
 * W153: keys off the *shown* market's own recommendationType, not
 * match.overall -- match.overall describes the match as a whole (used for
 * the dashboard's aggregate "N with positive edge" count, where it's the
 * right concept: "is ANYTHING on this match actionable"), but this
 * sentence is specifically about the one market bestMarket() picked to
 * display, and those two can genuinely differ (a higher-edge conditional
 * market can outrank a lower-edge direct_bet one for "shown", even though
 * match.overall reports the strongest type across every market). Falls
 * back to match.overall only when there's no shown market at all. */
function summarySentence(match: Match): string {
  const shown = bestMarket(match);
  switch (shown?.recommendationType ?? match.overall) {
    case "direct_bet":
      return shown
        ? `Oddsey recommends betting on ${selectionLabel(match, shown.selection)} (${shown.market}), with ${match.confidence} confidence.`
        : "Oddsey recommends a bet on this match.";
    case "conditional":
      return shown
        ? `Oddsey says wait on ${selectionLabel(match, shown.selection)} (${shown.market}) -- the price isn't good enough yet.`
        : "Oddsey says wait -- no price here clears its bar yet.";
    case "no_bet":
      return "Oddsey does not recommend a bet on this match right now.";
    case "insufficient_data":
    default:
      return "Oddsey doesn't have enough data yet for a confident read on this match.";
  }
}

export function MatchAnalysisPage({
  id,
  home,
  away,
  date,
  league = "E0",
}: {
  id: string;
  home: string;
  away: string;
  date: string;
  league?: string;
}) {
  const [match, setMatch] = useState<Match | null>(null);
  const [rawRecommendation, setRawRecommendation] = useState<MatchRecommendationOut | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  async function load() {
    setLoading(true);
    setError(null);
    try {
      // W47: check the precomputed cache (D2a) first -- only fall back to
      // the live "regenerate now" call on a real miss. A cache-check
      // failure is treated as a miss (not surfaced as an error) since
      // generateRecommendation below is still a fully valid fallback.
      let rec: MatchRecommendationOut | null = null;
      try {
        rec = await getCachedRecommendation(id, date);
      } catch {
        rec = null;
      }
      if (!rec) {
        rec = await generateRecommendation({ home_team: home, away_team: away, date, league, match_id: id });
      }
      setRawRecommendation(rec);
      setMatch(
        applyRecommendation(
          {
            id,
            league,
            tier: "competition_specific",
            kickoffIso: date,
            home,
            away,
            status: "upcoming",
            hasRecommendation: false,
            overall: "insufficient_data",
            confidence: "low",
            markets: [],
            explanation: [],
            limitations: [],
            predictionBasis: "",
            coldStartRisk: false,
            featureCompleteness: null,
            unknownTeam: false,
            invalidMarketCount: 0,
          },
          rec
        )
      );
    } catch (err) {
      setError(err instanceof ApiError ? err.message : "Could not reach the agent.");
    } finally {
      setLoading(false);
    }
  }

  useEffect(() => {
    load();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [id, home, away, date, league]);

  if (!home || !away || !date) {
    return (
      <AppShell active="matches">
        <p className="text-sm text-ink-secondary">
          Missing match details.{" "}
          <Link href="/matches" className="text-accent">
            Back to Match Explorer
          </Link>
        </p>
      </AppShell>
    );
  }

  // W153: the shown market's own recommendationType, not match.overall --
  // this verdict sits right above the same Model Probabilities table that
  // highlights this exact market, and must describe it, not a separate
  // match-wide aggregate that can legitimately differ (see
  // summarySentence's comment for the concrete scenario).
  const shown = match ? bestMarket(match) : undefined;

  return (
    <AppShell active="matches">
      <Link
        href="/matches"
        className="inline-flex items-center gap-1.5 text-sm text-ink-secondary transition-colors duration-150 hover:text-ink"
      >
        <ArrowLeft size={14} /> Back to Matches
      </Link>

      <div className="mt-4 flex items-start justify-between gap-4">
        <div>
          <div className="flex items-center gap-2 text-xs text-ink-secondary">
            {/* W110: full competition name, not the raw football-data.org
                code -- direct feedback that "E0"/"SWE" mean nothing to a
                reader who isn't already familiar with them. */}
            <span>{LEAGUE_LABEL[league] ?? league}</span>
            <TierTag tier="competition_specific" />
            <span>{date}</span>
          </div>
          <h1 className="mt-1 text-2xl font-semibold tracking-tight text-ink">
            {home} <span className="text-ink-secondary">vs</span> {away}
          </h1>
        </div>
        {match && (
          <div className="text-right">
            <div
              title={STATUS_META[shown?.recommendationType ?? match.overall].explain}
              className={`text-2xl font-bold tracking-tight ${STATUS_META[shown?.recommendationType ?? match.overall].text}`}
            >
              {STATUS_META[shown?.recommendationType ?? match.overall].verdict}
            </div>
            <div title={CONFIDENCE_EXPLAIN} className="mt-1 text-xs text-ink-secondary">
              Confidence: <span className="font-medium text-ink">{match.confidence}</span>
            </div>
            <div className="mt-2 flex justify-end">
              <TrustSignal match={match} size="lg" />
            </div>
          </div>
        )}
      </div>

      {loading && (
        <div className="mt-8">
          <LoadingRows count={4} />
        </div>
      )}
      {error && (
        <div className="mt-8">
          <ErrorState message={error} onRetry={load} />
        </div>
      )}

      {!loading && !error && match && (
        <>
          {/* W111: plain-language on-ramp, ahead of the jargon-dense table
              below it -- direct feedback that a reader with no betting
              vocabulary has nothing to read before the numbers today. */}
          <p className="mt-6 text-sm leading-relaxed text-ink">{summarySentence(match)}</p>

          <section className="mt-8">
            <h2 className="text-sm font-semibold uppercase tracking-wide text-muted">Model Probabilities</h2>
            <div className="mt-2 grid grid-cols-[1fr_auto_auto_auto_auto] gap-4 text-[11px] uppercase tracking-wide text-muted">
              <span />
              <span title={MODEL_PROBABILITY_EXPLAIN} className="text-right">Model</span>
              <span className="text-right">Market</span>
              <span title={EDGE_EXPLAIN} className="text-right">Edge</span>
              <span className="justify-self-end">Status</span>
            </div>
            {match.markets.length === 0 ? (
              <p className="mt-2 rounded-lg border border-border bg-surface p-3.5 text-sm text-ink-secondary">
                No markets in this recommendation.
              </p>
            ) : (
              match.markets.map((m, i) => (
                <ProbabilityRow
                  key={`${m.market}-${i}`}
                  m={m}
                  matchId={id}
                  recommendation={rawRecommendation ?? undefined}
                />
              ))
            )}
            {match.invalidMarketCount > 0 && (
              <p className="mt-2 flex items-center gap-1.5 text-xs text-serious">
                <WarningCircle weight="fill" size={13} />
                {match.invalidMarketCount} market{match.invalidMarketCount > 1 ? "s" : ""} omitted -- malformed
                data from the agent.
              </p>
            )}
          </section>

          {/* Squad Intelligence section removed (2026-08-13, W112) -- it
              always read "Not yet exposed by the API for this view" (no
              conditional, ForecastService's squad/player data was never
              plumbed through W02's endpoint), which reads as broken rather
              than as an honest "unavailable" note, unlike this page's other
              data-honesty patterns. Re-add once that data actually exists:
              a permanent stub is worse than no section at all. */}

          <section className="mt-8">
            <h2 className="text-sm font-semibold uppercase tracking-wide text-muted">Agent Reasoning</h2>
            <div className="mt-2 rounded-lg border border-border bg-surface p-4">
              <ul className="space-y-1.5 text-sm leading-relaxed text-ink-secondary">
                {match.explanation.map((point, i) => (
                  <li key={i} className="flex gap-1.5">
                    <span aria-hidden="true">·</span>
                    <span>{point}</span>
                  </li>
                ))}
              </ul>
              {match.limitations.length > 0 && (
                <ul className="mt-3 space-y-1 border-t border-border pt-3">
                  {match.limitations.map((l, i) => (
                    <li key={i} className="text-xs text-ink-secondary">
                      · {l}
                    </li>
                  ))}
                </ul>
              )}
            </div>
          </section>
        </>
      )}
    </AppShell>
  );
}
