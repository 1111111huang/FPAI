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
import { usePathname, useSearchParams } from "next/navigation";
import { useSession } from "next-auth/react";
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
  Lightning,
  MagnifyingGlass,
  MinusCircle,
  Plus,
  Question,
  Target,
  Ticket,
  TrendUp,
  Trophy,
  WarningCircle,
  X,
  XCircle,
} from "@phosphor-icons/react";

import {
  ApiError,
  generateRecommendation,
  getBets,
  getCachedRecommendation,
  getFixtures,
  logBetFromRecommendation,
  logBetManual,
} from "@/lib/api";
import type { Bet, Fixture, MatchRecommendationOut } from "@/lib/types";
import { useSandboxAsOf } from "@/lib/useSandboxAsOf";
import { groupByDate, groupByLeague, sortMatches, LEAGUE_COUNTRY, LEAGUE_LABEL, type MatchSort } from "@/lib/dashboardMetrics";
import { AppShell } from "./AppShell";
import { DashboardRail } from "./DashboardRail";
import { LogBetModal } from "./LogBetModal";

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
  targetOdds?: number | null;
  // See lib/types.ts's MarketCandidateOut.shap_contributions -- already
  // signed toward this candidate's own selection, server-side.
  shapContributions?: { feature: string; shapValue: number; value: number | null }[] | null;
};

export type RecommendationPick = { market: string; selection: string };

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
  candidates: MarketRec[];
  recommendationPick: RecommendationPick | null;
  // W215: the untouched recommendation snapshot -- needed to log a bet
  // directly from a MatchCard/LogBetButton without re-fetching it. Set to
  // `null` by fixtureToMatch()/applyRecommendation() (both cover it
  // explicitly), but kept optional (not just nullable) because at least
  // one inline Match literal -- MatchAnalysisPage's load()'s pending-match
  // object, further down this file -- omits it entirely and relies on
  // applyRecommendation()'s spread to backfill it, same as every other
  // recommendation-derived field there.
  rawRecommendation?: MatchRecommendationOut | null;
  // One bullet per aspect, mirroring lib/types.ts's MatchRecommendationOut.
  explanation: string[];
  limitations: string[];
  // A113: structured "Why This Pick" content -- optional (not just
  // nullable), matching unitBetMultiplier's own precedent below, so every
  // existing hand-built Match literal across the test suite (none of
  // which set these) keeps type-checking unmodified. null/undefined
  // whenever absent (pre-A113 data, non-compliant response, or a post-hoc
  // pick switch); renderers must fall back to explanation/limitations.
  teamEvidence?: { home: string; away: string } | null;
  theRead?: string | null;
  noBetRead?: string | null;
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
    candidates: [],
    recommendationPick: null,
    rawRecommendation: null,
    explanation: [],
    limitations: [],
    teamEvidence: null,
    theRead: null,
    noBetRead: null,
    predictionBasis: "",
    coldStartRisk: false,
    featureCompleteness: null,
    unknownTeam: false,
    invalidMarketCount: 0,
  };
}

export function applyRecommendation(match: Match, rec: MatchRecommendationOut): Match {
  return {
    ...match,
    hasRecommendation: true,
    overall: rec.overall,
    confidence: rec.confidence,
    predictionBasis: rec.prediction_basis,
    explanation: rec.explanation,
    limitations: rec.limitations,
    teamEvidence: rec.team_evidence ?? null,
    theRead: rec.the_read ?? null,
    noBetRead: rec.no_bet_read ?? null,
    coldStartRisk: rec.cold_start_risk,
    featureCompleteness: rec.feature_completeness,
    unknownTeam: rec.unknown_team,
    unitBetMultiplier: rec.unit_bet_multiplier ?? null,
    invalidMarketCount: rec.invalid_market_count,
    recommendationPick: rec.recommendation_pick,
    rawRecommendation: rec,
    candidates: rec.candidates.map((c) => ({
      market: c.market,
      selection: c.selection,
      recommendationType: c.recommendation_type,
      currentOdds: c.current_odds,
      minOdds: c.min_odds,
      mlProbability: c.ml_probability,
      impliedProbability: c.implied_probability,
      valueEdge: c.value_edge,
      targetOdds: c.target_odds ?? null,
      shapContributions: c.shap_contributions?.map((s) => ({
        feature: s.feature,
        shapValue: s.shap_value,
        value: s.value,
      })) ?? null,
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

/** Module-level (outside the component, survives unmount) so navigating to a
 * match's detail page and back doesn't re-run DashboardPage's ~12-call load
 * (1 fixtures + up to 10 concurrent recommendation calls + AppShell's own
 * duplicate sandbox-status fetch) every single time -- direct user report,
 * keyed by the same `today` date string load() already fetches with.
 *
 * TTL is adaptive, not fixed: a live or soon-to-start match means scores/
 * live-wait odds can move, so that entry expires fast; a quiet window with
 * nothing imminent can sit far longer since nothing about it changes on its
 * own. ponytail: a plain module-level Map, not a real cache library --
 * upgrade to something with LRU eviction if this ever grows past a handful
 * of date keys per session (it won't -- one user, one dashboard). */
const DASHBOARD_CACHE_LIVE_TTL_MS = 60_000;
const DASHBOARD_CACHE_IDLE_TTL_MS = 5 * 60_000;
const DASHBOARD_CACHE_IMMINENT_WINDOW_MS = 30 * 60_000;
const DASHBOARD_CACHE_LIVE_MATCH_DURATION_MS = 3 * 60 * 60_000;

const dashboardMatchesCache = new Map<string, { matches: Match[]; fetchedAt: number; ttlMs: number }>();

function isLiveOrImminent(kickoffIso: string, now: number): boolean {
  const kickoff = new Date(kickoffIso).getTime();
  const elapsed = now - kickoff;
  return elapsed >= 0
    ? elapsed < DASHBOARD_CACHE_LIVE_MATCH_DURATION_MS
    : -elapsed <= DASHBOARD_CACHE_IMMINENT_WINDOW_MS;
}

function getDashboardMatchesCache(key: string): Match[] | null {
  const entry = dashboardMatchesCache.get(key);
  if (!entry || Date.now() - entry.fetchedAt > entry.ttlMs) return null;
  return entry.matches;
}

function setDashboardMatchesCache(key: string, matches: Match[]): void {
  const now = Date.now();
  const ttlMs = matches.some((m) => isLiveOrImminent(m.kickoffIso, now))
    ? DASHBOARD_CACHE_LIVE_TTL_MS
    : DASHBOARD_CACHE_IDLE_TTL_MS;
  dashboardMatchesCache.set(key, { matches, fetchedAt: now, ttlMs });
}

/** Test-only escape hatch: this cache is module-level by design (it has to
 * survive DashboardPage unmounting), but that means it also survives between
 * separate `render(<DashboardPage />)` calls in the same test file, where
 * each test expects its own fresh mocked fetch. Real app code never calls
 * this -- the TTL is what bounds it there. */
export function __resetDashboardMatchesCacheForTests(): void {
  dashboardMatchesCache.clear();
}

export function formatKickoff(iso: string): string {
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

export function formatDay(iso: string, asOf: Date, sandboxMode: boolean): string {
  const diffDays = dayDiff(iso, asOf, sandboxMode);
  if (diffDays === 0) return "today";
  if (diffDays === 1) return "tomorrow";
  if (diffDays === -1) return "yesterday";
  if (diffDays > 1) return `in ${diffDays} days`;
  return `${-diffDays} days ago`;
}

// W218: shared "Today · Full Time" / "Today · 3:00 PM" style label for
// LogBetModal's fixture header -- the same day/time convention every list
// page already uses (MatchCard's closing row), reused instead of a fourth
// copy of this exact ternary.
export function matchStatusLabel(kickoffIso: string, isCompleted: boolean, asOf: Date, sandboxMode: boolean): string {
  const day = formatDay(kickoffIso, asOf, sandboxMode);
  return `${day} · ${isCompleted ? "Full Time" : formatKickoff(kickoffIso)}`;
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
// Exported for reuse by BetTracker.tsx's ManualBetForm (W211): its fixture
// search previously did its own date-window arithmetic with unconditional
// UTC setters, which is only correct in sandbox mode -- see this file's own
// MatchExplorerPage comment on why that's wrong for a live (non-sandbox)
// viewer in a positive-UTC-offset timezone.
export function dateString(d: Date, sandboxMode: boolean): string {
  if (sandboxMode) return d.toISOString().slice(0, 10);
  const y = d.getFullYear();
  const m = String(d.getMonth() + 1).padStart(2, "0");
  const day = String(d.getDate()).padStart(2, "0");
  return `${y}-${m}-${day}`;
}

export function addDays(d: Date, days: number, sandboxMode: boolean): Date {
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
    fill: "bg-good-dim",
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
    fill: "bg-warning-dim",
    icon: <Clock weight="fill" size={13} />,
    label: "Conditional",
    verdict: "WAIT",
    explain: "There's a real edge here, but the current price isn't good enough yet -- wait for it to improve.",
  },
  no_bet: {
    text: "text-muted",
    ring: "border-border",
    fill: "bg-surface",
    icon: <MinusCircle weight="fill" size={13} />,
    label: "No Bet",
    verdict: "PASS",
    explain: "No sufficient edge found -- not worth betting on this market.",
  },
  // W229 color standardization: moved from "serious" (red) to "warning"
  // (orange) -- direct user framing for orange's freed-up role after
  // the red/orange merge was literally "confidence low," and this status
  // means exactly that. Not a settled-bet negative outcome (red's own,
  // narrower role now) -- a pre-bet "couldn't get a confident read"
  // state, categorically different from "lost."
  insufficient_data: {
    text: "text-warning",
    ring: "border-warning/40",
    fill: "bg-warning-dim",
    icon: <Question weight="fill" size={13} />,
    label: "Insufficient Data",
    verdict: "NO READ",
    explain: "Not enough reliable data to make a confident prediction.",
  },
};

// W230, direct user reference screenshot: full literal Tailwind class
// strings, not derived via runtime string manipulation on STATUS_META's
// own fields -- Tailwind's JIT content scanner only generates CSS for
// class names it finds as complete literal strings in the source, so a
// `.replace("border-", "border-l-")`-style derived class would silently
// produce no CSS (invisible in dev if that exact string happens to
// already be used elsewhere by chance, broken in a real production
// build). A dedicated map, not STATUS_META's own `ring`/`fill` (tuned for
// a small pill badge, where a 40%-opacity border and a 15%-opacity wash
// read clearly against a small contained shape) -- a flat 15%-opacity
// wash was also too subtle spread across a whole wide row (found live,
// 2026-09-17). W230 follow-up: the reference row reads as a continuous
// dark-tinted band behind the recommended bet, not only a left accent or
// a fade that disappears across the row, so highlighted rows use the
// app's solid `-dim` semantic surfaces. The table row still keeps the
// left-border color and Model-probability color tied to the row's own
// RecommendationType (three values only -- never insufficient_data, a
// table row's own status can't be that).
const HIGHLIGHT_ROW_STYLE: Record<RecommendationType, { border: string; background: string; text: string }> = {
  direct_bet: { border: "border-l-good", background: "bg-gradient-to-r from-good-dim via-good-dim/40 to-transparent", text: "text-good" },
  conditional: { border: "border-l-warning", background: "bg-gradient-to-r from-warning-dim via-warning-dim/40 to-transparent", text: "text-warning" },
  no_bet: { border: "border-l-border", background: "bg-surface", text: "text-ink" },
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
// total_corners (A101) is different: a real, fixed 9.5 line via the
// OddsPapi backtest pull (A100) makes it genuinely resolvable the same way
// total_goals already is -- but no live match-result source in this app
// (Match["result"] itself) supplies corner counts yet, so in practice this
// stays dormant on real live completed-match cards until that changes too.
const RESOLVABLE_MARKETS = new Set(["result_3way", "btts", "total_goals", "total_corners", "home_goals", "away_goals"]);

export type ActualOutcome = {
  result: "home" | "away" | "draw";
  btts: "yes" | "no";
  totalGoalsSide: "over_2.5" | "under_2.5";
  homeGoalsSide: "over_1.5" | "under_1.5";
  awayGoalsSide: "over_1.5" | "under_1.5";
  totalCorners?: number;
  totalCornersSide?: "over_9.5" | "under_9.5";
};

export function buildActualOutcome(home: number, away: number, homeCorners?: number, awayCorners?: number): ActualOutcome {
  const result = home > away ? "home" : home < away ? "away" : "draw";
  const totalGoals = home + away;
  const outcome: ActualOutcome = {
    result,
    btts: home > 0 && away > 0 ? "yes" : "no",
    totalGoalsSide: totalGoals > 2 ? "over_2.5" : "under_2.5",
    homeGoalsSide: home > 1 ? "over_1.5" : "under_1.5",
    awayGoalsSide: away > 1 ? "over_1.5" : "under_1.5",
  };
  if (homeCorners !== undefined && awayCorners !== undefined) {
    const totalCorners = homeCorners + awayCorners;
    outcome.totalCorners = totalCorners;
    outcome.totalCornersSide = totalCorners > 9 ? "over_9.5" : "under_9.5";
  }
  return outcome;
}

/** Returns null (not false) for a market with no programmatic resolution,
 * or a resolvable market whose `actual` happens to lack the needed field
 * (e.g. total_corners when no corner counts were supplied) -- callers MUST
 * treat null as "unknown, skip" and never coerce it to a miss. */
export function marketCorrect(market: string, selection: string, actual: ActualOutcome): boolean | null {
  if (!RESOLVABLE_MARKETS.has(market)) return null;
  if (market === "result_3way") return selection === actual.result;
  if (market === "btts") return selection === actual.btts;
  if (market === "total_corners") return actual.totalCornersSide === undefined ? null : selection === actual.totalCornersSide;
  if (market === "home_goals") return selection === actual.homeGoalsSide;
  if (market === "away_goals") return selection === actual.awayGoalsSide;
  return selection === actual.totalGoalsSide; // market === "total_goals"
}

/** W193 (2026-09-01 design): TS port of resolve_recommendation_pick()
 * (src/agent/market_resolution.py) -- same three-case contract: the
 * matching candidate, or undefined when recommendationPick is null OR
 * names a market/selection absent from candidates (a dangling pointer).
 * Replaces bestMarket()'s own max(valueEdge) reduction now that the
 * backend already resolved which candidate is the pick -- there is
 * nothing left to rank client-side. */
export function resolveRecommendation(match: Match): MarketRec | undefined {
  const pick = match.recommendationPick;
  if (!pick) return undefined;
  return match.candidates.find((c) => c.market === pick.market && c.selection === pick.selection);
}

/** Mockup point 3: backs Daily Edges' "N with positive edge" summary line.
 * Same predicate as the "Positive Edge" tag/green edge coloring on
 * MatchCard itself (recommendationType !== "no_bet" && valueEdge >= 0) --
 * kept as one shared function rather than a third inline copy of that
 * condition. */
export function hasPositiveEdge(match: Match): boolean {
  const m = resolveRecommendation(match);
  return !!m && m.currentOdds != null && m.recommendationType !== "no_bet" && m.valueEdge >= 0;
}

/** Whether a completed match's recommended pick actually hit. null covers
 * "not completed yet", "no recommendation", "no actual pick was ever made"
 * (recommendationType "no_bet" -- see the comment on MatchCard's own `hit`
 * computation this was extracted from for why that's excluded rather than
 * graded), an unresolvable market (e.g. corners), or a missing result --
 * the same null-propagation contract HitBadge already establishes. Shared
 * so DashboardRail's Edge Distribution donut can classify a completed
 * match by hit/miss using the exact rule MatchCard's badge uses, not a
 * second copy that could drift. */
export function computeHit(match: Match): boolean | null {
  if (match.status !== "completed" || !match.hasRecommendation || !match.result) return null;
  const shown = resolveRecommendation(match);
  if (!shown || shown.recommendationType === "no_bet") return null;
  return marketCorrect(shown.market, shown.selection, buildActualOutcome(match.result.home, match.result.away));
}

/** Money actually won/lost on the pick, in UB (an abstract Unit Bet, not a
 * dollar figure) -- BUG-053. Deliberately narrower than `computeHit(match)
 * !== null` alone -- a `conditional` market ("wait for a better price") can
 * still carry a non-null unitBetMultiplier (schema.py's
 * _attach_unit_bet_multiplier only excludes no_bet, not conditional)
 * despite never having been an actual bet at currentOdds, so this also
 * requires recommendationType === "direct_bet" specifically. profit =
 * stake*(odds-1) on a hit, -stake on a miss -- the same formula
 * src/agent/staking.py's simulate_flat_stake/simulate_kelly_stake and
 * app/backend/bet_tracker.py's real settlement all already use. Shared so
 * DashboardRail's staking summary can aggregate the exact same per-match
 * profit MatchCard's own footer already shows, not a second copy. */
export function computeMoneyWon(match: Match): number | null {
  const hit = computeHit(match);
  const shown = resolveRecommendation(match);
  if (hit === null || shown?.recommendationType !== "direct_bet" || shown.currentOdds == null || match.unitBetMultiplier == null) {
    return null;
  }
  return hit ? match.unitBetMultiplier * (shown.currentOdds - 1) : -match.unitBetMultiplier;
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
 * right now" are two different, both-relevant facts. Uses the standard
 * "serious"/red token (W229 color standardization merged the old separate
 * "critical" red into it -- one true red, no distinction between them) --
 * a natural fit for something this urgent/real-time. No minute/clock
 * shown -- not data this app has. */
function LiveBadge() {
  return (
    <span className="inline-flex items-center gap-1.5 rounded-md border border-serious/40 bg-serious/15 px-2 py-0.5 text-[11px] font-medium uppercase tracking-wide text-serious">
      <span className="h-1.5 w-1.5 animate-pulse rounded-full bg-serious" />
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
      <span className="inline-flex items-center gap-1.5 rounded-md border border-good/40 bg-good-dim px-2 py-0.5 text-[11px] font-medium uppercase tracking-wide text-good">
        <CheckCircle weight="fill" size={13} />
        Hit
      </span>
    );
  }
  return (
    <span className="inline-flex items-center gap-1.5 rounded-md border border-serious/40 bg-serious-dim px-2 py-0.5 text-[11px] font-medium uppercase tracking-wide text-serious">
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
      className={`inline-flex items-center gap-1.5 rounded-md border border-warning/40 bg-warning-dim text-warning ${pad} font-medium`}
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
        border: `1.5px solid ${secondary ?? "var(--border-soft)"}`,
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
        border: `1.5px solid ${secondary ?? "var(--border-soft)"}`,
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
          checked ? "bg-accent" : "border border-border bg-surface"
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
  bets = NO_BETS,
  onBetsChanged = () => {},
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
  // Direct user request (2026-09-15): the full signed-in user's bet list
  // (useAllBets(), above) -- filtered to this match below for the header's
  // "N logged" indicator and threaded into LogBetButton so its own
  // "Logged"/"Log another" state survives a reload too, not just this
  // session's own just-submitted bet. Defaults to empty so every existing
  // caller/test that doesn't pass it (an unauthenticated visitor's page,
  // or a test rendering MatchCard standalone) is unaffected.
  bets?: Bet[];
  // Called after "Log another" successfully logs a new bet, so the parent
  // page's useAllBets() can refetch and every card's count/state updates
  // together -- a no-op default for the same reason as `bets` above.
  onBetsChanged?: () => void;
}) {
  const [open, setOpen] = useState(false);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const isCompleted = match.status === "completed";
  const isLive = match.status === "live";
  // Direct user request: "more bets are allowed to be logged" even once
  // one already is -- every bet logged for this match, any market, not
  // just the one the card's own quick-log button tracks.
  const myBets = useMemo(() => bets.filter((b) => b.match_id === match.id), [bets, match.id]);
  const shown = resolveRecommendation(match);
  // null covers "not completed yet", "no recommendation", "no bet was
  // actually recommended", and "market unresolvable" (e.g. corners)
  // identically -- HitBadge only renders for a real true/false. Direct user
  // report: a `no_bet` card still showed a green "Hit" badge (the app's own
  // "least-bad no_bet" fallback market, resolveRecommendation(), happening to land on
  // the correct outcome) -- misleading, since Hit/Not Hit should describe
  // whether an actual recommended pick paid off, not whether an unactioned
  // market's own selection happened to match the result.
  const hit = computeHit(match);
  // BUG-053 follow-up (direct user request): money actually won/lost on the
  // pick, in UB, replacing the Odds/Result box once a match completes.
  const moneyWon = computeMoneyWon(match);
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
    <div className="rounded-xl border border-border bg-page/80 transition-all duration-150 hover:-translate-y-px hover:border-border">
      {/* W217: was a plain <button onClick={handleExpand}> -- direct user
          request put a real, always-visible "Log bet" trigger directly on
          this face (below), and a <button> nested inside another <button>
          is invalid HTML/inaccessible, so the whole-card expand toggle
          moved to a role="button" div with matching keyboard handling
          (Enter/Space) instead. LogBetButton's own onClick handlers call
          stopPropagation() so tapping it doesn't also toggle expand. */}
      <div
        role="button"
        tabIndex={0}
        onClick={handleExpand}
        onKeyDown={(e) => {
          if (e.key === "Enter" || e.key === " ") {
            e.preventDefault();
            handleExpand();
          }
        }}
        className="w-full cursor-pointer p-4 text-left"
      >
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
          {/* Direct user request: a match with a logged bet still allows
              logging more (a different market, or another on the same
              pick) -- this count is the one visible sign of that without
              expanding the card, so it isn't mistaken for "done, nothing
              more to do here" the way the bottom box's own badge alone
              could otherwise read. */}
          {myBets.length > 0 && (
            <span className="flex items-center gap-1 rounded-full border border-accent/40 bg-accent/10 px-2 py-0.5 text-[11px] font-medium text-accent">
              <Ticket size={11} weight="fill" />
              {myBets.length} logged
            </span>
          )}
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
                  Falls back to match.overall only when resolveRecommendation()
                  found nothing to show at all. */}
              {!isCompleted && <StatusBadge status={shown?.recommendationType ?? match.overall} />}
            </>
          ) : (
            <span className="rounded-md border border-border bg-surface px-2 py-0.5 text-[11px] font-medium uppercase tracking-wide text-muted">
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

        {/* W219: direct user mockup -- MODELED tag + MARKET/PICK/ODDS/EDGE
            and the Log Bet/Logged status now share one bordered box
            (previously an unboxed flex row, with LogBetButton mixed in as
            just another column) -- matches ManualBetForm's own boxed
            fixture-header convention (bg-page/60, W217) rather than
            inventing a new nesting style. */}
        <div className="mt-3 rounded-xl border border-border bg-page/60 p-3.5">
          <div className="flex flex-wrap items-start justify-between gap-4">
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
                    <span className="mt-1 inline-block rounded-full border border-border bg-surface px-1.5 py-0.5 text-[10px] text-muted">
                      Pre-match edge
                    </span>
                  )
                ) : (
                  shown?.currentOdds != null && shown.recommendationType !== "no_bet" && shown.valueEdge >= 0 && (
                    <span className="mt-1 inline-block rounded-full border border-good/40 bg-good-dim px-1.5 py-0.5 text-[10px] text-good">
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
                    <span className="mt-1 inline-block rounded-full border border-border bg-surface px-1.5 py-0.5 text-[10px] text-muted">
                      Pre-match stake
                    </span>
                  )}
                </div>
              )}
            </div>
          </div>
          {/* W217/W219: log the card's own resolved pick right here on the
              always-visible face, no expand needed -- direct_bet only, a
              conditional/no_bet pick isn't something the agent is actually
              recommending you act on yet; the full multi-market picker on
              the detail page (ProbabilityRow's own LogBetButton) is
              unaffected and still covers every market/recommendation type.
              Kept even once the match is completed -- logging a bet against
              a match that's already finished is the normal case this whole
              feature area (W211-W215) was built for. Its own row inside the
              box, below a divider, per the mockup -- previously just another
              column mixed into the data grid above. */}
          {shown?.recommendationType === "direct_bet" && match.rawRecommendation && (
            <div className="mt-3 flex justify-end border-t border-border pt-3">
              <LogBetButton
                matchId={match.id}
                recommendation={match.rawRecommendation}
                market={shown.market}
                selection={shown.selection}
                variant="pill"
                homeTeam={match.home}
                awayTeam={match.away}
                date={match.kickoffIso.slice(0, 10)}
                statusLabel={matchStatusLabel(match.kickoffIso, isCompleted, asOf, sandboxMode)}
                matchBets={myBets}
                onBetsChanged={onBetsChanged}
              />
            </div>
          )}
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
      </div>

      <div className={`expand-rows ${open ? "is-open" : ""}`}>
        <div>
          <div className="border-t border-border p-3.5 text-sm">
            {loading && <LoadingRows count={1} />}
            {error && <ErrorState message={error} onRetry={handleExpand} />}
            {!loading && !error && match.hasRecommendation && (
              <>
                <WhyThisPickSection match={match} shown={shown} />
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

const NO_BETS: Bet[] = [];

/** Direct user request (2026-09-15): "N logged"/outcome indicators on
 * MatchCard, persisted across a reload -- not just this session's own
 * just-submitted state (LogBetButton's pre-existing `done`/`loggedBet`
 * only ever reflected that). Fetches the signed-in user's full bet list
 * once for whichever page renders a list of MatchCards (Dashboard, Match
 * Explorer) -- mirrors MatchAnalysisPage's own existing single-match
 * `getBets()` fetch (its `loggedKeys` Set, just scoped to one match id
 * there instead of every match here) -- and `refetch()` lets a card ask
 * for a fresh list right after it logs a new bet, so the header count and
 * the bottom-box "Logged" state update together without a page reload.
 * Best-effort: a fetch failure just means no bets show anywhere, the same
 * "cosmetic, never blocking" precedent that hook already established. */
function useAllBets(): { bets: Bet[]; refetch: () => void } {
  const { status } = useSession();
  const [bets, setBets] = useState<Bet[]>(NO_BETS);
  const [tick, setTick] = useState(0);

  useEffect(() => {
    if (status !== "authenticated") {
      setBets(NO_BETS);
      return;
    }
    let cancelled = false;
    getBets()
      .then((result) => {
        if (!cancelled) setBets(result);
      })
      .catch(() => {
        // Best-effort -- see comment above.
      });
    return () => {
      cancelled = true;
    };
  }, [status, tick]);

  return { bets, refetch: () => setTick((t) => t + 1) };
}

// ---------------------------------------------------------------------------
// Page 1 -- Dashboard ("/"): today's real E0 fixtures.
// ---------------------------------------------------------------------------

export function DashboardPage() {
  // AppShell (below) independently calls this same hook too -- see its own
  // comment. Known duplicate fetch, not shared/cached; accepted for now.
  const { asOf, sandboxMode } = useSandboxAsOf();
  const { bets, refetch: refetchBets } = useAllBets();
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
      // W233: a fresh-enough cached list (see dashboardMatchesCache above)
      // skips the fetch entirely -- no blanking loading state, no network
      // calls -- so returning here from a match's detail page shows
      // instantly instead of re-running the full ~12-call load.
      const cached = getDashboardMatchesCache(today);
      if (cached) {
        setMatches(cached);
        return;
      }
      setMatches(null);
      try {
        const to = addDays(asOf, 90, sandboxMode);
        const fixtures = await getFixtures(today, dateString(to, sandboxMode));
        if (cancelled) return;
        const sorted = fixtures
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
          .sort((a, b) => a.kickoffIso.localeCompare(b.kickoffIso));
        // Direct user request: today's matches are never trimmed, even past
        // 10 -- only matches from later days fill the remaining slots (if
        // any) up to a total cap of 10. The forward-only window means
        // today's matches already sort first, so this is a straight
        // partition-and-concat, not a re-sort.
        const todays = sorted.filter((m) => dayDiff(m.kickoffIso, asOf, sandboxMode) === 0);
        const later = sorted.filter((m) => dayDiff(m.kickoffIso, asOf, sandboxMode) !== 0);
        const nearest = [...todays, ...later.slice(0, Math.max(0, 10 - todays.length))];
        // W53: resolve the precomputed cache for the (already-capped-to-10)
        // list before rendering -- an additional await in this same guarded
        // run, so re-check `cancelled` again before touching state.
        const resolvedMatches = await resolveCachedRecommendations(nearest);
        if (cancelled) return;
        setMatches(resolvedMatches);
        setDashboardMatchesCache(today, resolvedMatches);
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
                          <span className="rounded-full border border-border px-2 py-0.5 text-xs text-ink-secondary">
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
                            bets={bets}
                            onBetsChanged={refetchBets}
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
  const { bets, refetch: refetchBets } = useAllBets();
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
        // W211: direct user feedback -- search was forward-only, so a match
        // that already kicked off (including ones from earlier the same
        // week) couldn't be found here or in ManualBetForm's fixture picker
        // at all. 30 days back is plenty for realistic backfill without
        // reintroducing the same "off-season gap" problem the 90-day
        // forward window was widened to avoid (a gap that only grows
        // looking forward, not back).
        const from = dateString(addDays(asOf, -30, sandboxMode), sandboxMode);
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
                    <span className="rounded-full border border-border px-2 py-0.5 text-xs text-ink-secondary">
                      {group.matches.length} match{group.matches.length === 1 ? "" : "es"}
                    </span>
                  </div>
                  <Trophy size={16} className="text-muted" aria-hidden="true" />
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
                      bets={bets}
                      onBetsChanged={refetchBets}
                    />
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
  variant = "link",
  statusLabel,
  homeTeam,
  awayTeam,
  date,
  matchBets = NO_BETS,
  onBetsChanged = () => {},
}: {
  matchId: string;
  recommendation: MatchRecommendationOut;
  market: string;
  selection: string;
  // W217: "pill" is MatchCard's always-visible, filled-accent trigger
  // (direct user request: log a bet without expanding the card first) --
  // "link" (default) is ProbabilityRow's existing plain-text style,
  // unchanged. Only the closed-state trigger differs; the expanded
  // stake-input/Confirm/Cancel row and the settled/"done" state look the
  // same regardless, since neither was asked to change.
  variant?: "link" | "pill";
  // W218: LogBetModal's fixture header needs these -- pulled from the
  // caller's own already-typed home/away strings, NOT recommendation.match
  // (that field is `Record<string, unknown>`-typed in lib/types.ts, so
  // `.home`/`.away` on it doesn't type-check).
  statusLabel: string;
  homeTeam: string;
  awayTeam: string;
  // Direct user request (2026-09-15): needed only for "Log another"'s
  // logBetManual() call (pill variant only, below) -- YYYY-MM-DD, same
  // shape ManualBetForm's own formatDate(fixture.utc_date) produces.
  // Optional since the "link" variant (ProbabilityRow) never renders that
  // flow and has no equally-cheap date on hand to pass.
  date?: string;
  // Every bet logged for this match, any market -- lets "done" (and
  // "Log another"'s own visibility) survive a reload instead of resetting
  // to "not logged" until this component's own local state is set fresh
  // by a submit in the current session. Optional/defaulted for the same
  // reason as `bets` on MatchCard, above.
  matchBets?: Bet[];
  onBetsChanged?: () => void;
}) {
  const { status } = useSession();
  const pathname = usePathname();
  const searchParams = useSearchParams();
  const [open, setOpen] = useState(false);
  // W218 final-review cleanup: was a 4-state union ("idle"|"saving"|"done"|
  // "error"), but LogBetModal now owns saving/error state itself -- this
  // component only ever needs to know whether a bet was just logged.
  const [done, setDone] = useState(false);
  const [loggedBet, setLoggedBet] = useState<Bet | null>(null);
  // Direct user request (2026-09-15): "Log another" opens a second,
  // separate modal instance in editable mode (a different market/
  // selection than the one this button's own locked flow tracks) -- kept
  // as its own state rather than reusing `open`/the locked modal above,
  // since the two need different LogBetModal props (locked vs. not) and
  // can't both be "the" open modal at once anyway.
  const [loggingAnother, setLoggingAnother] = useState(false);
  // The persisted record for this exact market+selection, if any -- takes
  // over from the local `loggedBet` once a fresh page load hands it back
  // via `matchBets` (that local state only ever covers this session's own
  // just-submitted bet). Prefers the local one when both exist so a
  // fresh submit's fuller detail (e.g. an outcome W212 already settled)
  // shows immediately, without waiting on onBetsChanged()'s refetch.
  const persistedBet = matchBets.find((b) => b.market === market && b.selection === selection) ?? null;
  const effectiveDone = done || persistedBet !== null;
  const effectiveLoggedBet = loggedBet ?? persistedBet;

  if (status === "unauthenticated") {
    // W215: the whole current URL (query params included), not a bare
    // `/matches/${matchId}` -- MatchAnalysisPage requires home/away/date to
    // render at all (see its own "Missing match details" guard), so a
    // truncated callbackUrl stranded a signed-in user on a dead page with
    // no way back to the bet they were trying to log. Both hooks are
    // nullable outside a real router context (confirmed empirically: they
    // don't throw, they return null) -- the `?? `/matches/${matchId}`` /
    // `searchParams?.toString()` guards keep this correct in that case
    // rather than only in the browser.
    const base = pathname ?? `/matches/${matchId}`;
    const query = searchParams?.toString();
    const currentUrl = query ? `${base}?${query}` : base;
    return (
      <Link
        href={`/login?callbackUrl=${encodeURIComponent(currentUrl)}`}
        onClick={(e) => e.stopPropagation()}
        className="text-xs font-medium text-accent"
      >
        Sign in to log this bet
      </Link>
    );
  }
  if (status === "loading") return null;

  if (effectiveDone) {
    // W215: logBetFromRecommendation() already returns the settled outcome
    // (W212 may have auto-settled it immediately) -- show it instead of a
    // flat "Logged" that hides real information already in hand, and give
    // a way to see it in context instead of a dead end.
    const outcome = effectiveLoggedBet?.outcome;
    if (variant === "pill") {
      // W219: direct user mockup -- a bordered pill badge (matching this
      // card's own "Not Hit"/"Positive Edge" badge language, not the plain
      // colored text the "link" variant below still uses) with an icon,
      // "Logged · Won"/"Logged · Lost" title-cased, still-open bets shown
      // as a neutral "Logged" with no icon/color (no outcome to react to
      // yet).
      return (
        <span className="flex items-center gap-2 text-xs" onClick={(e) => e.stopPropagation()}>
          <span
            className={`inline-flex items-center gap-1 rounded-full border px-2 py-1 font-medium ${
              outcome === "won"
                ? "border-good/40 bg-good-dim text-good"
                : outcome === "lost"
                ? "border-serious/40 bg-serious-dim text-serious"
                : "border-border bg-surface text-muted"
            }`}
          >
            {outcome === "won" && <CheckCircle weight="fill" size={12} />}
            {outcome === "lost" && <XCircle weight="fill" size={12} />}
            {outcome && outcome !== "open" ? `Logged · ${outcome === "won" ? "Won" : "Lost"}` : "Logged"}
          </span>
          <Link href="/bets" className="flex items-center gap-0.5 font-medium text-accent">
            View in Bet Tracker
            <CaretRight size={11} />
          </Link>
          {/* Direct user request (2026-09-15): being done with *this*
              market/pick doesn't mean nothing more can be logged on this
              match -- lets a different market get its own bet without
              losing the badge/link above. Editable (not locked) since the
              point is picking something other than what's already logged;
              `date` is only ever missing for the "link" variant, which
              never reaches this branch. */}
          {date && (
            <button
              type="button"
              onClick={(e) => {
                e.stopPropagation();
                setLoggingAnother(true);
              }}
              className="ml-auto flex shrink-0 items-center gap-1 rounded-full border border-border px-3 py-1.5 font-semibold text-ink transition hover:border-accent hover:text-accent"
            >
              <Plus size={13} weight="bold" />
              Log another
            </button>
          )}
          {date && loggingAnother && (
            <LogBetModal
              open
              onClose={() => setLoggingAnother(false)}
              homeTeam={homeTeam}
              awayTeam={awayTeam}
              statusLabel={statusLabel}
              locked={false}
              market="result_3way"
              selection=""
              odds={null}
              onSubmit={async ({ market: newMarket, selection: newSelection, odds, stake }) => {
                await logBetManual({
                  match_id: matchId, date, home_team: homeTeam, away_team: awayTeam,
                  market: newMarket, selection: newSelection, odds, stake,
                });
                setLoggingAnother(false);
                onBetsChanged();
              }}
            />
          )}
        </span>
      );
    }
    return (
      <span className="flex items-center gap-2 text-xs" onClick={(e) => e.stopPropagation()}>
        <span className={outcome === "lost" ? "text-serious" : "text-good"}>
          {outcome && outcome !== "open" ? `Logged -- ${outcome}` : "Logged"}
        </span>
        <Link href="/bets" className="font-medium text-accent">
          View in Bet Tracker
        </Link>
      </span>
    );
  }

  if (!open) {
    return (
      <button
        type="button"
        onClick={(e) => {
          e.stopPropagation();
          setOpen(true);
        }}
        className={
          variant === "pill"
            ? "flex shrink-0 items-center gap-1 rounded-full bg-accent px-3 py-1.5 text-xs font-semibold text-white transition hover:bg-accent/90"
            : "text-xs font-medium text-accent"
        }
      >
        {variant === "pill" && <Plus size={13} weight="bold" />}
        {variant === "pill" ? "Log Bet" : "Log bet"}
      </button>
    );
  }

  const matchedCandidate = recommendation.candidates.find((c) => c.market === market && c.selection === selection);

  return (
    <LogBetModal
      open
      onClose={() => setOpen(false)}
      homeTeam={homeTeam}
      awayTeam={awayTeam}
      statusLabel={statusLabel}
      locked
      market={market}
      selection={selection}
      odds={matchedCandidate?.current_odds ?? null}
      onSubmit={async ({ stake }) => {
        const bet = await logBetFromRecommendation({ match_id: matchId, recommendation, market, selection, stake });
        setLoggedBet(bet);
        setDone(true);
        setOpen(false);
        onBetsChanged();
      }}
    />
  );
}

// W227: direct user mockup -- a boxed banner above the Model Probabilities
// table calling out the actual pick (colored to match its own status),
// replacing the plain summarySentence() paragraph for an actionable match.
// null for a non-actionable one (no_bet/insufficient_data, or no resolved
// pick) -- summarySentence()'s existing paragraph still covers that case
// unchanged, nothing to log or call out with a colored pill for it.
function PickBanner({
  match,
  shown,
  matchId,
  homeTeam,
  awayTeam,
  statusLabel,
  alreadyLogged,
}: {
  match: Match;
  shown: MarketRec | undefined;
  matchId: string;
  homeTeam: string;
  awayTeam: string;
  statusLabel: string;
  alreadyLogged: boolean;
}) {
  if (!shown || !isActionable(match) || !match.rawRecommendation) return null;
  const s = STATUS_META[shown.recommendationType];
  // W111's own lesson (a generic "Home Win"/"Away Win" reads worse than
  // naming the actual team) still applies here -- result_3way uses
  // pickLabel() (team-name-aware), everything else uses the generic
  // marketSelectionTitle() the table rows also use, since "BTTS No"/
  // "Over 2.5" already ARE the natural, market-identifying label.
  const pillText = shown.market === "result_3way" ? pickLabel(match, shown.selection) : marketSelectionTitle(shown.market, shown.selection);
  return (
    <div className="mt-6 flex flex-wrap items-center justify-between gap-4 rounded-xl border border-border bg-surface p-3.5">
      <div className="flex min-w-0 flex-wrap items-center gap-3">
        <span
          className={`inline-flex shrink-0 items-center gap-1.5 rounded-full border px-2.5 py-1 text-xs font-semibold ${s.ring} ${s.fill} ${s.text}`}
        >
          {pillText}
        </span>
        <p className="text-sm text-ink">Oddsey&apos;s pick for this fixture, at {match.confidence} confidence.</p>
      </div>
      <div className="flex shrink-0 items-center gap-2">
        <LogBetButton
          matchId={matchId}
          recommendation={match.rawRecommendation}
          market={shown.market}
          selection={shown.selection}
          variant="pill"
          statusLabel={statusLabel}
          homeTeam={homeTeam}
          awayTeam={awayTeam}
        />
        {alreadyLogged && <span className="text-[11px] text-muted">(already logged)</span>}
      </div>
    </div>
  );
}

function ProbabilityRow({
  m,
  matchId,
  recommendation,
  alreadyLogged = false,
  highlighted = false,
  statusLabel,
  homeTeam,
  awayTeam,
}: {
  m: MarketRec;
  matchId?: string;
  recommendation?: MatchRecommendationOut;
  alreadyLogged?: boolean;
  // W227: this row is the match's own resolved pick -- a left accent
  // border + subtle background tint (colored to the row's own status,
  // same STATUS_META colors every other status signal in the app uses)
  // distinguishes it from the rest of the table at a glance.
  highlighted?: boolean;
  // W218: threaded straight through to LogBetButton's own LogBetModal --
  // see its prop comment for why these aren't pulled from recommendation.
  statusLabel: string;
  homeTeam: string;
  awayTeam: string;
}) {
  const anomalous = isAnomalousDirectBet(m);
  const s = STATUS_META[m.recommendationType];
  const highlight = HIGHLIGHT_ROW_STYLE[m.recommendationType];
  return (
    <div
      className={`grid grid-cols-[1fr_auto_auto_auto_auto_auto] items-center gap-4 border-b border-l-4 py-3 pl-3 text-sm last:border-b-0 ${
        highlighted ? `${highlight.border} ${highlight.background}` : "border-l-transparent"
      } border-border`}
    >
      <span className="flex flex-col gap-0.5 truncate">
        {/* W227: a generic, human-readable title ("Home Win", "BTTS No")
            above the raw market/selection identifier -- previously the raw
            identifier was the only thing shown at all. */}
        <span className="truncate font-medium text-ink">{marketSelectionTitle(m.market, m.selection)}</span>
        <span className="truncate font-mono text-[11px] text-muted">
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
      </span>
      {/* W230: colored to match the row's own status when it's the
          resolved pick (reference screenshot: "68%" reads green on the
          highlighted row, plain white everywhere else) -- odds stays
          plain either way, only Model% and Edge% (already conditionally
          colored below) carry the status color. */}
      <span className={`text-right font-mono ${highlighted ? highlight.text : "text-ink"}`}>
        {formatPct(m.mlProbability)}
      </span>
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
        <span className="justify-self-end">
          <StatusBadge status={m.recommendationType} />
        </span>
      )}
      {/* W210 follow-up (2026-09-14): re-enabled -- see Task 3's
          auth-aware LogBetButton and AppShell's Task 1 sign-in UI. W227:
          moved into its own trailing column, previously embedded under
          the market name in column 1. */}
      <span className="justify-self-end">
        {matchId && recommendation && !anomalous && (
          <span className="flex items-center gap-1.5">
            <LogBetButton
              matchId={matchId}
              recommendation={recommendation}
              market={m.market}
              selection={m.selection}
              statusLabel={statusLabel}
              homeTeam={homeTeam}
              awayTeam={awayTeam}
            />
            {/* W215: a warning, not a block -- a genuinely different
                real-world wager on the same market is still plausible, so
                LogBetButton stays fully enabled either way. */}
            {alreadyLogged && <span className="text-[10px] text-muted">(already logged)</span>}
          </span>
        )}
      </span>
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

// W227: direct user mockup -- the Model Probabilities table and the pick
// banner both need one generic, human-readable title per (market,
// selection) pair ("Home Win", "BTTS No", "Over 2.5") -- distinct from
// pickLabel()/selectionLabel() above, which embed the actual team name for
// a single highlighted pick's own sentence ("Oddsey recommends betting on
// Real Betis..."); this table lists every market side by side, so a
// team-name-specific phrasing wouldn't read sensibly for the away/draw
// rows. A lookup table for the markets this app actually emits (mirrors
// src/agent/schema.py's MarketCandidate market/selection Literals), with a
// humanized fallback for anything unexpected rather than rendering blank.
const _MARKET_SELECTION_TITLE: Record<string, string> = {
  "result_3way:home": "Home Win",
  "result_3way:draw": "Draw",
  "result_3way:away": "Away Win",
  "btts:yes": "BTTS Yes",
  "btts:no": "BTTS No",
  "total_goals:over_2.5": "Over 2.5",
  "total_goals:under_2.5": "Under 2.5",
  "total_corners:over_9.5": "Corners Over 9.5",
  "total_corners:under_9.5": "Corners Under 9.5",
  "home_goals:over_1.5": "Over 1.5",
  "home_goals:under_1.5": "Under 1.5",
  "away_goals:over_1.5": "Over 1.5",
  "away_goals:under_1.5": "Under 1.5",
};
function marketSelectionTitle(market: string, selection: string): string {
  const known = _MARKET_SELECTION_TITLE[`${market}:${selection}`];
  if (known) return known;
  const selectionPart = selection.replace(/_/g, " ");
  return `${marketLabel(market).label} ${selectionPart.charAt(0).toUpperCase()}${selectionPart.slice(1)}`;
}

// Human-readable stems for the feature codes that show up as top SHAP
// contributors. Matched by substring, not exact name, since group prefixes
// aren't uniform-length across families (OFF_HOME_XG_R5, SQUAD_HOME_XG_MEAN_R3,
// OPP_ADJ_HOME_GOALS_SCORED_R5 all carry "XG"/"GOALS_SCORED" at different
// depths) -- deliberately approximate, not exhaustive: covers the stems
// that actually showed up as top contributors investigating real
// production models, with a humanized fallback for anything else, same
// "known map + fallback, never a raw code" convention as
// marketSelectionTitle() above.
const _FEATURE_STEM_LABEL: Record<string, string> = {
  KEY_ATTACKER_MISSING: "missing a key attacker",
  XGA: "expected goals conceded",
  XG: "expected goals",
  XA: "expected assists",
  FTHG: "goals scored",
  FTAG: "goals conceded",
  SHOT_ACCURACY: "shot accuracy",
  HST: "shots on target",
  AST: "shots on target",
  HS: "shots",
  AS: "shots",
  HC: "corners won",
  AC: "corners won",
  HY: "yellow cards",
  AY: "yellow cards",
  HR: "red cards",
  AR: "red cards",
  DISCIPLINE_SCORE: "discipline record",
  SAVE_RATE: "save rate",
  REST_DAYS: "days of rest",
  WIN_STREAK: "current win streak",
  SCORE_STREAK: "current scoring streak",
  CS_STREAK: "current clean-sheet streak",
  CUM_PTS: "points total this season",
  PPG_L10: "points per game (last 10)",
  GOALS_STD: "scoring consistency",
  CONCEDED_STD: "defensive consistency",
  CORNERS_STD: "corner-count consistency",
  RATING_MEAN: "squad rating",
  OVERROUND: "the bookmaker's margin",
  IMPLIED_HOME: "the market's implied home-win chance",
  IMPLIED_DRAW: "the market's implied draw chance",
  IMPLIED_AWAY: "the market's implied away-win chance",
  IMPLIED_OVER25: "the market's implied over-2.5 chance",
  LAMBDA_TOTAL: "the market-implied total goals",
  LAMBDA_HOME: "the market-implied home goals",
  LAMBDA_AWAY: "the market-implied away goals",
  LAMBDA_AH_DIFF: "the market's handicap-implied goal gap",
  POISSON_BTTS_PROB: "the market-implied BTTS chance",
  AH_LINE: "the Asian handicap line",
  AH_HOME_ODDS: "the home handicap price",
  AH_AWAY_ODDS: "the away handicap price",
  H2H_TOTAL_GOALS: "head-to-head scoring history",
  H2H_CORNERS: "head-to-head corner history",
  H2H_HOME_WIN_RATE: "head-to-head home win rate",
};
const _FEATURE_STEM_ENTRIES_BY_LENGTH = Object.entries(_FEATURE_STEM_LABEL).sort(
  ([a], [b]) => b.length - a.length
);

/** Turns an internal feature code ("LINEUP_AWAY_KEY_ATTACKER_MISSING",
 * "SQUAD_HOME_XG_MEAN_R3") into a short reader-facing phrase -- names the
 * actual team when the code carries a HOME_/AWAY_ side, and adds a
 * rolling-window qualifier when one applies. */
export function featureLabel(name: string, match: Match): string {
  const side: "home" | "away" | null = /_HOME_/.test(name) ? "home" : /_AWAY_/.test(name) ? "away" : null;

  const windowMatch = name.match(/_(R3|R5|EMA5|L10)$/);
  const windowSuffix = windowMatch
    ? windowMatch[1] === "EMA5"
      ? " (recent trend)"
      : ` (last ${windowMatch[1].replace(/\D/g, "")})`
    : "";

  // Longest key first -- a short stem code ("AY" -> yellow cards) can be a
  // pure substring accident inside an unrelated, longer one ("REST_DAYS"
  // contains "AY"), so the more specific match has to get first refusal.
  let stem: string | null = null;
  for (const [key, label] of _FEATURE_STEM_ENTRIES_BY_LENGTH) {
    if (name.includes(key)) {
      stem = label;
      break;
    }
  }
  if (stem === null) {
    stem = name
      .replace(/^[A-Z_]+?_(HOME|AWAY)_/, "")
      .replace(/_(R3|R5|EMA5|L10)$/, "")
      .replace(/_/g, " ")
      .toLowerCase();
  }

  if (side) {
    const team = side === "home" ? match.home : match.away;
    return `${team}${stem.startsWith("missing") ? " is" : "'s"} ${stem}${windowSuffix}`;
  }
  return `${stem}${windowSuffix}`;
}

/** W233 (direct user spec, 2026-09-18): a 1-2 sentence line composed
 * entirely from this candidate's own shap_contributions -- code-only, never
 * the LLM's prose. Sits directly above the ProbabilityTapeBar it explains,
 * not in its own section: the model-vs-market bar already IS the model's
 * justification, this sentence just names what's actually driving that
 * number. Returns null (renders nothing) when there's nothing to say -- no
 * contributions attached (composite model, or a result_3way candidate,
 * which never gets shap_contributions -- see _attach_shap_contributions),
 * or every contribution happens to point the other way.
 *
 * Deliberately drops any contribution whose value is null (the feature
 * wasn't available for this match, e.g. no O2.5/AH odds quoted) rather than
 * caveating it inline -- naming an unavailable market as if it were
 * observed is exactly the failure mode this session's own investigation
 * flagged. It can still be the single largest contributor and simply won't
 * be named in the sentence; the raw list (shapContributions) still has it
 * for anyone who wants it. */
export function shapSummarySentence(match: Match, candidate: MarketRec): string | null {
  const available = (candidate.shapContributions ?? []).filter((c) => c.value !== null);
  const supporting = available.filter((c) => c.shapValue > 0).slice(0, 2);
  if (supporting.length === 0) return null;

  const against = available.find((c) => c.shapValue < 0);
  const pick = marketSelectionTitle(candidate.market, candidate.selection);
  const supportPhrase = supporting.map((c) => featureLabel(c.feature, match)).join(" and ");
  const againstPhrase = against ? `, despite ${featureLabel(against.feature, match)} pointing the other way` : "";

  return `${pick} is driven mainly by ${supportPhrase}${againstPhrase}.`;
}

// W121 follow-up (mockup point 3): human-readable market names, not the raw
// backend string (`shown.market` was previously rendered verbatim -- a
// reader would have seen "result_3way"/"total_goals" literally). Covers
// the real markets the agent actually emits (src/agent/schema.py
// MarketRecommendation.market Literal) -- an unrecognized market string
// (future market type) falls back to a generic humanization rather than
// silently mislabeling it as one of these.
const MARKET_LABEL: Record<string, { label: string; subtitle: string }> = {
  result_3way: { label: "3-Way Result", subtitle: "Full Time" },
  total_goals: { label: "Over/Under", subtitle: "Full Time" },
  btts: { label: "Both Teams to Score", subtitle: "Full Time" },
  home_corners: { label: "Home Corners", subtitle: "Full Time" },
  away_corners: { label: "Away Corners", subtitle: "Full Time" },
  total_corners: { label: "Total Corners", subtitle: "Full Time" }, // A101
  home_goals: { label: "Home Goals", subtitle: "Full Time" }, // W199
  away_goals: { label: "Away Goals", subtitle: "Full Time" }, // W199
};
// W174: exported so AgentPerformanceDashboard.tsx can reuse the same
// human-readable market names ("3-Way Result" instead of "result_3way")
// instead of duplicating MARKET_LABEL.
export function marketLabel(market: string): { label: string; subtitle: string | null } {
  if (MARKET_LABEL[market]) return MARKET_LABEL[market];
  const spaced = market.replace(/_/g, " ");
  return { label: spaced.charAt(0).toUpperCase() + spaced.slice(1), subtitle: null };
}

// ---------------------------------------------------------------------------
// A113: "Why This Pick" -- icon-tagged structured reasoning blocks, direct
// user spec (2026-09-17). Ground rules the whole section follows: never
// name a threshold/price range/internal cutoff (a reader can only disagree
// with the read, not with a rule they can't see); show the model's own
// numbers as data (ProbabilityTapeBar) rather than narrating them in prose;
// each content type gets its own icon-tagged block, never folded into one
// flat bullet list. W231: the visual treatment must not depend on the
// optional A113 structured fields being present -- expanded MatchCards all
// use the same row-based system, while richer team-specific rows appear only
// when the backend supplied teamEvidence/theRead/noBetRead.
// ---------------------------------------------------------------------------

/** Split-width "tale of the tape" bar: model probability vs. market
 * probability, both rendered as data (bold numbers on a proportionally-
 * sized fill), not narrated in a sentence. `captionMode="gap"` (the
 * direct_bet/conditional value case) states the gap in percentage points
 * with no claim about why it matters; `captionMode="neutral"` (the no_bet
 * closest-candidate case) never states the margin it missed by. */
function ProbabilityTapeBar({
  heading,
  modelProb,
  marketProb,
  captionMode = "gap",
}: {
  heading: string;
  modelProb: number;
  marketProb: number;
  captionMode?: "gap" | "neutral";
}) {
  const modelPct = modelProb * 100;
  const marketPct = marketProb * 100;
  const total = modelPct + marketPct;
  const modelShare = total > 0 ? (modelPct / total) * 100 : 50;
  const gapPts = modelPct - marketPct;
  const caption =
    captionMode === "neutral"
      ? "Too close a read to put money behind"
      : `Model reads ${gapPts >= 0 ? "+" : ""}${gapPts.toFixed(1)}% ${gapPts >= 0 ? "higher" : "lower"} than the market price`;
  return (
    <div className="mt-2">
      <p className="text-center text-[11px] font-medium uppercase tracking-wide text-muted">{heading}</p>
      {/* Found live, direct user report (2026-09-17): the Market segment
          previously had no background of its own at all -- it inherited
          this wrapper's bg-surface, which is the exact same tone as the
          card the bar sits inside, so it read as "no fill." Model's own
          bg-warning/25 (a faint 25%-opacity wash, and the wrong token --
          "warning" is now the caution color, not brand/model-data) was
          real but too subtle to register as an intentional color either.
          W229 color standardization: Model is bg-gold (the standardized
          "Model probability" role, dark text for contrast against the
          bright gold -- white on bright gold fails contrast), Market is
          bg-slate (promoted from this exact one-off gradient into a real
          token, the standardized "Market probability" role). The wrapper
          itself carries no background of its own -- both segments' fills
          meet at the boundary directly. */}
      <div className="mt-1.5 flex overflow-hidden rounded-lg border border-border">
        <div
          className="min-w-0 bg-gold px-3 py-2.5"
          style={{ flexBasis: `${Math.max(modelShare, 1)}%` }}
        >
          <p className="text-[10px] font-medium uppercase tracking-wide text-page">Model</p>
          <p className="font-mono text-xl font-bold text-page">{modelPct.toFixed(1)}%</p>
        </div>
        <div className="min-w-0 flex-1 bg-slate px-3 py-2.5 text-right">
          <p className="text-[10px] font-medium uppercase tracking-wide text-muted">Market</p>
          <p className="font-mono text-xl font-bold text-ink">{marketPct.toFixed(1)}%</p>
        </div>
      </div>
      <p className="mt-1.5 text-center text-xs text-muted">{caption}</p>
    </div>
  );
}

/** One icon-tagged row: a colored square icon, a bold title, an optional
 * "Auto-checked" tag (system-generated content, not the model's own
 * prose), and free-form children below. */
function WhyPickRow({
  icon,
  iconClass,
  title,
  autoChecked = false,
  children,
}: {
  icon: React.ReactNode;
  iconClass: string;
  title: string;
  autoChecked?: boolean;
  children: React.ReactNode;
}) {
  return (
    <div className="flex gap-3 border-t border-border py-4 first:border-t-0 first:pt-0">
      <span className={`flex h-9 w-9 shrink-0 items-center justify-center rounded-lg ${iconClass}`}>{icon}</span>
      <div className="min-w-0 flex-1">
        <div className="flex items-center gap-2">
          <p className="text-sm font-semibold text-ink">{title}</p>
          {autoChecked && (
            <span className="inline-flex items-center gap-1 rounded-full border border-border px-1.5 py-0.5 text-[10px] text-muted">
              <Lightning size={10} weight="fill" /> Auto-checked
            </span>
          )}
        </div>
        <div className="mt-1 text-sm leading-relaxed text-ink-secondary">{children}</div>
      </div>
    </div>
  );
}

/** Betting-price text, entirely code-generated (never the LLM's own
 * prose) -- always tagged "Auto-checked" per the content spec, since the
 * whole point is keeping the source of a code-side note visible. No
 * internal threshold/range is named; target_odds (when present) is a
 * per-pick computed number, not a fixed internal cutoff. */
function bettingPriceText(candidate: MarketRec): string {
  if (candidate.recommendationType === "direct_bet") {
    return "This price is live and ready to bet on now. No need to wait for it to move.";
  }
  if (candidate.targetOdds) {
    return `This price isn't quite there yet -- worth waiting for it to reach about ${candidate.targetOdds.toFixed(2)}.`;
  }
  return "This price isn't quite there yet. If you're comfortable waiting, it may improve.";
}

function ExplanationPoints({ points }: { points: string[] }) {
  return (
    <ul className="space-y-1">
      {points.map((point, i) => (
        <li key={i} className="flex gap-1.5">
          <span aria-hidden="true">·</span>
          <span>{point}</span>
        </li>
      ))}
    </ul>
  );
}

function WhyThisPickSection({ match, shown }: { match: Match; shown: MarketRec | undefined }) {
  const noBetMode = match.overall === "no_bet" || match.overall === "insufficient_data";
  const explanation =
    match.explanation.length > 0
      ? match.explanation
      : ["The agent did not return written reasoning for this recommendation."];

  if (!noBetMode && shown && match.teamEvidence && match.theRead) {
    const marketMeta = marketLabel(shown.market);
    const pick = pickLabel(match, shown.selection);
    const shapSentence = shapSummarySentence(match, shown);
    return (
      <div>
        {/* W229 color standardization: gold (Model/brand-data), not
            warning/orange (now the caution color) -- this block is about
            the model's own value read, not a warning state. Uses the new
            solid gold-dim pill-background token rather than a translucent
            opacity modifier, matching the standardized dim-variant
            convention. */}
        <WhyPickRow icon={<TrendUp size={18} weight="bold" />} iconClass="bg-gold-dim text-gold" title="Value case">
          <p>
            {marketMeta.label} {pick} reads as the strongest value on this fixture, at odds of{" "}
            <span className="font-semibold text-ink">{shown.currentOdds?.toFixed(2)}</span>.
          </p>
          {shapSentence && <p className="mt-1 text-ink-secondary">{shapSentence}</p>}
          <ProbabilityTapeBar heading="Win probability" modelProb={shown.mlProbability} marketProb={shown.impliedProbability} />
        </WhyPickRow>
        <WhyPickRow icon={<TeamBadge name={match.home} />} iconClass="bg-transparent p-0" title={match.home}>
          {match.teamEvidence.home}
        </WhyPickRow>
        <WhyPickRow icon={<TeamBadge name={match.away} />} iconClass="bg-transparent p-0" title={match.away}>
          {match.teamEvidence.away}
        </WhyPickRow>
        <WhyPickRow icon={<Target size={18} weight="bold" />} iconClass="bg-violet-500/15 text-violet-300" title="The read">
          {match.theRead}
        </WhyPickRow>
        <WhyPickRow icon={<Clock size={18} weight="bold" />} iconClass="bg-good/15 text-good" title="Betting price" autoChecked>
          {bettingPriceText(shown)}
        </WhyPickRow>
      </div>
    );
  }

  if (!noBetMode && shown) {
    const marketMeta = marketLabel(shown.market);
    const pick = pickLabel(match, shown.selection);
    const shapSentence = shapSummarySentence(match, shown);
    return (
      <div>
        <WhyPickRow icon={<TrendUp size={18} weight="bold" />} iconClass="bg-gold-dim text-gold" title="Value case">
          <p>
            {marketMeta.label} {pick} is the current pick
            {shown.currentOdds != null ? (
              <>
                , at odds of <span className="font-semibold text-ink">{shown.currentOdds.toFixed(2)}</span>
              </>
            ) : (
              " for this fixture"
            )}
            .
          </p>
          {shapSentence && <p className="mt-1 text-ink-secondary">{shapSentence}</p>}
          <ProbabilityTapeBar heading="Win probability" modelProb={shown.mlProbability} marketProb={shown.impliedProbability} />
        </WhyPickRow>
        <WhyPickRow icon={<Target size={18} weight="bold" />} iconClass="bg-purple-dim text-purple" title="The read">
          <ExplanationPoints points={explanation} />
        </WhyPickRow>
        <WhyPickRow icon={<Clock size={18} weight="bold" />} iconClass="bg-good-dim text-good" title="Betting price" autoChecked>
          {bettingPriceText(shown)}
        </WhyPickRow>
      </div>
    );
  }

  if (noBetMode) {
    const closest = [...match.candidates].sort((a, b) => b.valueEdge - a.valueEdge)[0];
    const shapSentence = closest ? shapSummarySentence(match, closest) : null;
    return (
      <div>
        <WhyPickRow icon={<MinusCircle size={18} weight="bold" />} iconClass="bg-surface text-ink-secondary" title="No qualifying edge today">
          {match.noBetRead ? <p>{match.noBetRead}</p> : <ExplanationPoints points={explanation} />}
          {shapSentence && <p className="mt-1 text-ink-secondary">{shapSentence}</p>}
          {closest && (
            <ProbabilityTapeBar
              heading={`${marketSelectionTitle(closest.market, closest.selection)} — closest read`}
              modelProb={closest.mlProbability}
              marketProb={closest.impliedProbability}
              captionMode="neutral"
            />
          )}
        </WhyPickRow>
      </div>
    );
  }

  // Final fallback: structurally odd data (for example, direct_bet overall
  // with no resolved pick). Keep it in the same row system rather than
  // reverting expanded cards to the old unstyled bullet block.
  return (
    <WhyPickRow icon={<Question size={18} weight="bold" />} iconClass="bg-surface text-ink-secondary" title="The read">
      <ExplanationPoints points={explanation} />
    </WhyPickRow>
  );
}

/** W111: one plain-English sentence, composed entirely from fields already
 * on the recommendation (overall/confidence/resolveRecommendation) -- no new backend
 * field, no LLM call. Sits ahead of the jargon-dense Model Probabilities
 * table so a reader with zero betting vocabulary has something to read
 * before the numbers.
 *
 * W153: keys off the *shown* market's own recommendationType, not
 * match.overall -- match.overall describes the match as a whole (used for
 * the dashboard's aggregate "N with positive edge" count, where it's the
 * right concept: "is ANYTHING on this match actionable"), but this
 * sentence is specifically about the one market resolveRecommendation() picked to
 * display, and those two can genuinely differ (a higher-edge conditional
 * market can outrank a lower-edge direct_bet one for "shown", even though
 * match.overall reports the strongest type across every market). Falls
 * back to match.overall only when there's no shown market at all. */
function summarySentence(match: Match): string {
  const shown = resolveRecommendation(match);
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
  const { status } = useSession();
  const [loggedKeys, setLoggedKeys] = useState<Set<string>>(new Set());
  // W218: needed to compute the same "Today · Full Time"-style statusLabel
  // MatchCard's own quick-log button already computes, for ProbabilityRow's
  // LogBetButton (this page had no reason to call it before).
  const { asOf, sandboxMode } = useSandboxAsOf();

  // W215: best-effort duplicate-bet warning -- fetch the signed-in user's
  // bets once and flag any market/selection on this match that already has
  // a logged bet. Unauthenticated visitors never see this fetch attempted
  // (matches LogBetButton's own auth-gating), and a failure here just means
  // no "already logged" note shows -- it never blocks the page or logging a
  // new bet.
  useEffect(() => {
    if (status !== "authenticated") {
      setLoggedKeys(new Set());
      return;
    }
    let cancelled = false;
    getBets()
      .then((allBets) => {
        if (cancelled) return;
        const keys = allBets.filter((b) => b.match_id === id).map((b) => `${b.market}::${b.selection}`);
        setLoggedKeys(new Set(keys));
      })
      .catch(() => {
        // Best-effort -- see comment above.
      });
    return () => {
      cancelled = true;
    };
  }, [id, status]);

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
            candidates: [],
            recommendationPick: null,
            explanation: [],
            limitations: [],
            teamEvidence: null,
            theRead: null,
            noBetRead: null,
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
  const shown = match ? resolveRecommendation(match) : undefined;

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
            {/* W227: direct user mockup -- the verdict is now a small pill
                (icon + short verdict word, e.g. "BET"/"WAIT"/"PASS") rather
                than large standalone colored text. Reuses STATUS_META's own
                icon/verdict/color fields directly (not StatusBadge, which
                always renders `.label` -- "Direct Bet" -- not the shorter
                `.verdict` this header wants). */}
            <span
              title={STATUS_META[shown?.recommendationType ?? match.overall].explain}
              className={`inline-flex items-center gap-1.5 rounded-full border px-3 py-1 text-sm font-semibold ${
                STATUS_META[shown?.recommendationType ?? match.overall].ring
              } ${STATUS_META[shown?.recommendationType ?? match.overall].fill} ${
                STATUS_META[shown?.recommendationType ?? match.overall].text
              }`}
            >
              {STATUS_META[shown?.recommendationType ?? match.overall].icon}
              {STATUS_META[shown?.recommendationType ?? match.overall].verdict}
            </span>
            <div title={CONFIDENCE_EXPLAIN} className="mt-1.5 text-xs text-ink-secondary">
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
              vocabulary has nothing to read before the numbers today.
              W227: superseded by PickBanner for an actionable match (direct
              user mockup) -- kept unchanged for no_bet/insufficient_data,
              which PickBanner deliberately renders nothing for. */}
          {isActionable(match) && shown ? (
            <PickBanner
              match={match}
              shown={shown}
              matchId={id}
              homeTeam={home}
              awayTeam={away}
              statusLabel={matchStatusLabel(match.kickoffIso, match.status === "completed", asOf, sandboxMode)}
              alreadyLogged={loggedKeys.has(`${shown.market}::${shown.selection}`)}
            />
          ) : (
            <p className="mt-6 text-sm leading-relaxed text-ink">{summarySentence(match)}</p>
          )}

          <section className="mt-8">
            <h2 className="text-sm font-semibold uppercase tracking-wide text-muted">Model Probabilities</h2>
            <div className="mt-2 grid grid-cols-[1fr_auto_auto_auto_auto_auto] gap-4 pl-4 text-[11px] uppercase tracking-wide text-muted">
              <span>Market</span>
              <span title={MODEL_PROBABILITY_EXPLAIN} className="text-right">Model</span>
              <span className="text-right">Market</span>
              <span title={EDGE_EXPLAIN} className="text-right">Edge</span>
              <span className="justify-self-end">Status</span>
              <span className="justify-self-end">&nbsp;</span>
            </div>
            {match.candidates.length === 0 ? (
              <p className="mt-2 rounded-lg border border-border bg-surface p-3.5 text-sm text-ink-secondary">
                No markets in this recommendation.
              </p>
            ) : (
              match.candidates.map((m, i) => (
                <ProbabilityRow
                  key={`${m.market}-${i}`}
                  m={m}
                  matchId={id}
                  recommendation={rawRecommendation ?? undefined}
                  alreadyLogged={loggedKeys.has(`${m.market}::${m.selection}`)}
                  highlighted={shown?.market === m.market && shown?.selection === m.selection}
                  homeTeam={home}
                  awayTeam={away}
                  statusLabel={matchStatusLabel(match.kickoffIso, match.status === "completed", asOf, sandboxMode)}
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
            <h2 className="flex items-center gap-1.5 text-sm font-semibold uppercase tracking-wide text-muted">
              <Question size={16} weight="bold" /> Why This Pick
            </h2>
            <div className="mt-2 rounded-lg border border-border bg-surface p-4">
              <WhyThisPickSection match={match} shown={shown} />
            </div>
          </section>
        </>
      )}
    </AppShell>
  );
}
