"use client";

/**
 * STANDALONE UI DRAFT — not wired to any real API, backend, or data model.
 * Every match, player stat, and feature value below is hardcoded. This file
 * imports nothing from outside itself except React, Next's <Link>, and
 * @phosphor-icons/react. It is rendered by three thin route wrappers under
 * sandbox/app/ (page.tsx, matches/page.tsx, matches/[id]/page.tsx) purely
 * because Next.js routing is file-based — those wrappers contain no real
 * logic, just an import and a render call.
 *
 * Safe to hack on freely. Nothing here is imported by, or imports from, the
 * real app/ directory. See documents/app_user_stories.md (D6) for context.
 */

import { useMemo, useState } from "react";
import Link from "next/link";
import {
  ArrowLeft,
  CaretDown,
  CaretRight,
  CheckCircle,
  Clock,
  MagnifyingGlass,
  MinusCircle,
  Question,
  TrendDown,
  TrendUp,
} from "@phosphor-icons/react";

// ---------------------------------------------------------------------------
// Types — mirrors the real MatchRecommendation shape (documents/agent_prd.md
// Section 4) plus mocked additions (player metrics, squad stats) for the
// deep-dive page's "Agent Intelligence" section.
// ---------------------------------------------------------------------------

type Tier = "competition_specific" | "general_purpose";
type RecommendationType = "direct_bet" | "conditional" | "no_bet";
type Overall = RecommendationType | "insufficient_data";
type Confidence = "low" | "medium" | "high";
type MarketKey = "result_3way" | "total_goals" | "btts" | "home_corners";
type Day = "today" | "tomorrow" | "in 2 days" | "in 3 days" | "3 days ago" | "5 days ago";

type MarketRec = {
  market: MarketKey;
  selection: string;
  recommendationType: RecommendationType;
  currentOdds: number | null;
  minOdds: number;
  mlProbability: number;
  impliedProbability: number;
  valueEdge: number;
};

type TopFeature = {
  name: string;
  value: string;
  importance: number; // 0-1
};

type PlayerStat = {
  name: string;
  position: "GK" | "DEF" | "MID" | "FWD";
  rating: number;
  xgxa90: number;
  formDelta: number;
};

type SquadStat = {
  label: string;
  home: number;
  away: number;
};

type Match = {
  id: string;
  league: string;
  tier: Tier;
  day: Day;
  kickoff: string;
  home: string;
  away: string;
  overall: Overall;
  confidence: Confidence;
  markets: MarketRec[];
  explanation: string;
  limitations: string[];
  topFeatures: TopFeature[];
  homePlayers: PlayerStat[];
  awayPlayers: PlayerStat[];
  squadStats: SquadStat[];
  status: "upcoming" | "completed";
  result?: { home: number; away: number };
};

// ---------------------------------------------------------------------------
// Display metadata
// ---------------------------------------------------------------------------

const MARKET_LABEL: Record<MarketKey, string> = {
  result_3way: "1X2",
  total_goals: "Over/Under 2.5",
  btts: "Both Teams to Score",
  home_corners: "Home Corners O/U 4.5",
};

const TIER_LABEL: Record<Tier, string> = {
  competition_specific: "EPL",
  general_purpose: "General",
};

const STATUS_META: Record<
  Overall,
  { text: string; ring: string; icon: React.ReactNode; label: string; verdict: string }
> = {
  direct_bet: {
    text: "text-good",
    ring: "border-good/40",
    icon: <CheckCircle weight="fill" size={13} />,
    label: "Direct Bet",
    verdict: "BET",
  },
  conditional: {
    text: "text-warning",
    ring: "border-warning/40",
    icon: <Clock weight="fill" size={13} />,
    label: "Conditional",
    verdict: "WAIT",
  },
  no_bet: {
    text: "text-muted",
    ring: "border-border-strong",
    icon: <MinusCircle weight="fill" size={13} />,
    label: "No Bet",
    verdict: "PASS",
  },
  insufficient_data: {
    text: "text-serious",
    ring: "border-serious/40",
    icon: <Question weight="fill" size={13} />,
    label: "Insufficient Data",
    verdict: "NO READ",
  },
};

// ---------------------------------------------------------------------------
// Mock dataset — 5 EPL (competition_specific) + 3 General (general_purpose)
// fixtures, 2 of which are already completed (for the Explorer's historical
// grid). General-tier matches intentionally have no player/squad data: the
// real product's general_purpose tier is market-odds-only, no team history.
// ---------------------------------------------------------------------------

const MOCK_MATCHES: Match[] = [
  {
    id: "liv-ars",
    league: "Premier League",
    tier: "competition_specific",
    day: "today",
    kickoff: "15:00",
    home: "Liverpool",
    away: "Arsenal",
    overall: "direct_bet",
    confidence: "high",
    markets: [
      {
        market: "result_3way",
        selection: "home",
        recommendationType: "direct_bet",
        currentOdds: 2.15,
        minOdds: 2.0,
        mlProbability: 0.52,
        impliedProbability: 0.44,
        valueEdge: 0.08,
      },
      {
        market: "total_goals",
        selection: "over_2.5",
        recommendationType: "no_bet",
        currentOdds: 1.75,
        minOdds: 2.0,
        mlProbability: 0.55,
        impliedProbability: 0.57,
        valueEdge: -0.02,
      },
      {
        market: "btts",
        selection: "yes",
        recommendationType: "conditional",
        currentOdds: 1.7,
        minOdds: 2.0,
        mlProbability: 0.57,
        impliedProbability: 0.58,
        valueEdge: -0.01,
      },
    ],
    explanation:
      "Liverpool's home xG form over the last 5 matches (2.1/game) significantly outpaces Arsenal's away defensive rating. Market-implied home probability (44%) sits well below the model estimate (52%), and current odds of 2.15 clear the 2.0 minimum with margin.",
    limitations: [],
    topFeatures: [
      { name: "OFF_HOME_XG_R5", value: "1.86", importance: 0.091 },
      { name: "MKT_Home_Prob_Real", value: "0.44", importance: 0.071 },
      { name: "FRDS_HOME", value: "1.12", importance: 0.058 },
    ],
    homePlayers: [
      { name: "Salah", position: "FWD", rating: 7.8, xgxa90: 0.94, formDelta: 0.12 },
      { name: "Núñez", position: "FWD", rating: 7.1, xgxa90: 0.61, formDelta: 0.04 },
      { name: "Szoboszlai", position: "MID", rating: 7.3, xgxa90: 0.48, formDelta: 0.01 },
    ],
    awayPlayers: [
      { name: "Saka", position: "FWD", rating: 7.6, xgxa90: 0.71, formDelta: 0.02 },
      { name: "Ødegaard", position: "MID", rating: 7.4, xgxa90: 0.55, formDelta: -0.03 },
      { name: "Rice", position: "MID", rating: 7.2, xgxa90: 0.31, formDelta: 0.05 },
    ],
    squadStats: [
      { label: "FRDS (Rating Dominance)", home: 1.12, away: 0.98 },
      { label: "XOC (Top-3 Off. Concentration)", home: 2.4, away: 2.1 },
    ],
    status: "upcoming",
  },
  {
    id: "che-bha",
    league: "Premier League",
    tier: "competition_specific",
    day: "today",
    kickoff: "17:30",
    home: "Chelsea",
    away: "Brighton",
    overall: "conditional",
    confidence: "medium",
    markets: [
      {
        market: "total_goals",
        selection: "over_2.5",
        recommendationType: "conditional",
        currentOdds: 1.85,
        minOdds: 2.0,
        mlProbability: 0.56,
        impliedProbability: 0.54,
        valueEdge: 0.02,
      },
      {
        market: "result_3way",
        selection: "home",
        recommendationType: "no_bet",
        currentOdds: 1.95,
        minOdds: 2.0,
        mlProbability: 0.48,
        impliedProbability: 0.51,
        valueEdge: -0.03,
      },
    ],
    explanation:
      "Over 2.5 goals shows a real but modest edge — current odds of 1.85 fall short of the 2.0 minimum this app requires before flagging a direct bet. This becomes a direct bet only if odds drift to 2.0 or higher before kickoff.",
    limitations: ["Odds below minimum threshold for a direct bet."],
    topFeatures: [
      { name: "OFF_AWAY_XG_R5", value: "1.41", importance: 0.066 },
      { name: "CTX_HOME_REST_DAYS", value: "4", importance: 0.033 },
    ],
    homePlayers: [
      { name: "Palmer", position: "MID", rating: 7.9, xgxa90: 0.88, formDelta: 0.15 },
      { name: "Jackson", position: "FWD", rating: 7.0, xgxa90: 0.52, formDelta: -0.02 },
    ],
    awayPlayers: [
      { name: "Mitoma", position: "FWD", rating: 7.3, xgxa90: 0.58, formDelta: 0.03 },
      { name: "Ferguson", position: "FWD", rating: 6.9, xgxa90: 0.44, formDelta: 0.01 },
    ],
    squadStats: [
      { label: "FRDS (Rating Dominance)", home: 1.05, away: 0.94 },
      { label: "XOC (Top-3 Off. Concentration)", home: 2.1, away: 1.8 },
    ],
    status: "upcoming",
  },
  {
    id: "mci-ful",
    league: "Premier League",
    tier: "competition_specific",
    day: "today",
    kickoff: "20:00",
    home: "Man City",
    away: "Fulham",
    overall: "no_bet",
    confidence: "high",
    markets: [
      {
        market: "result_3way",
        selection: "home",
        recommendationType: "no_bet",
        currentOdds: 1.35,
        minOdds: 2.0,
        mlProbability: 0.78,
        impliedProbability: 0.74,
        valueEdge: 0.04,
      },
      {
        market: "total_goals",
        selection: "over_2.5",
        recommendationType: "no_bet",
        currentOdds: 1.6,
        minOdds: 2.0,
        mlProbability: 0.5,
        impliedProbability: 0.625,
        valueEdge: -0.125,
      },
    ],
    explanation:
      "The model agrees City are heavy favorites, but at 1.35 the odds sit far below the 2.0 minimum this app enforces regardless of edge size. Over 2.5 goals is priced well ahead of the model's own estimate. No market clears the bar here.",
    limitations: [],
    topFeatures: [
      { name: "STRENGTH_DIFF_HOME", value: "1.94", importance: 0.088 },
      { name: "MKT_Home_Prob_Real", value: "0.74", importance: 0.052 },
    ],
    homePlayers: [
      { name: "Haaland", position: "FWD", rating: 8.1, xgxa90: 1.21, formDelta: 0.08 },
      { name: "De Bruyne", position: "MID", rating: 7.7, xgxa90: 0.69, formDelta: 0.02 },
    ],
    awayPlayers: [
      { name: "Jiménez", position: "FWD", rating: 6.8, xgxa90: 0.41, formDelta: -0.01 },
      { name: "Iwobi", position: "MID", rating: 6.7, xgxa90: 0.35, formDelta: 0.0 },
    ],
    squadStats: [
      { label: "FRDS (Rating Dominance)", home: 1.34, away: 0.81 },
      { label: "XOC (Top-3 Off. Concentration)", home: 3.1, away: 1.4 },
    ],
    status: "upcoming",
  },
  {
    id: "tot-whu",
    league: "Premier League",
    tier: "competition_specific",
    day: "tomorrow",
    kickoff: "14:00",
    home: "Tottenham",
    away: "West Ham",
    overall: "insufficient_data",
    confidence: "low",
    markets: [
      {
        market: "result_3way",
        selection: "home",
        recommendationType: "no_bet",
        currentOdds: null,
        minOdds: 2.0,
        mlProbability: 0.44,
        impliedProbability: 0,
        valueEdge: 0,
      },
    ],
    explanation:
      "Current bookmaker odds for this fixture could not be found during analysis, so no value edge could be computed. Feature completeness is also below the usual threshold (68%) following a managerial change affecting recent form signals.",
    limitations: [
      "Current odds unavailable at analysis time.",
      "Feature completeness 68% — below the usual threshold.",
    ],
    topFeatures: [],
    homePlayers: [],
    awayPlayers: [],
    squadStats: [],
    status: "upcoming",
  },
  {
    id: "new-avl",
    league: "Premier League",
    tier: "competition_specific",
    day: "tomorrow",
    kickoff: "16:30",
    home: "Newcastle",
    away: "Aston Villa",
    overall: "direct_bet",
    confidence: "medium",
    markets: [
      {
        market: "home_corners",
        selection: "over_4.5",
        recommendationType: "direct_bet",
        currentOdds: 2.3,
        minOdds: 2.0,
        mlProbability: 0.5,
        impliedProbability: 0.43,
        valueEdge: 0.07,
      },
      {
        market: "result_3way",
        selection: "home",
        recommendationType: "no_bet",
        currentOdds: 2.05,
        minOdds: 2.0,
        mlProbability: 0.46,
        impliedProbability: 0.49,
        valueEdge: -0.03,
      },
    ],
    explanation:
      "Newcastle's home corner rate (R5 average 6.8) is well above Aston Villa's away corners-conceded rate, and the over 4.5 line at 2.30 clears the minimum threshold with a healthy edge.",
    limitations: [],
    topFeatures: [
      { name: "OFF_HOME_HC_R5", value: "6.8", importance: 0.061 },
      { name: "DEF_AWAY_ACA_R5", value: "3.9", importance: 0.047 },
    ],
    homePlayers: [
      { name: "Isak", position: "FWD", rating: 7.7, xgxa90: 0.86, formDelta: 0.09 },
      { name: "Gordon", position: "FWD", rating: 7.2, xgxa90: 0.53, formDelta: 0.03 },
    ],
    awayPlayers: [
      { name: "Watkins", position: "FWD", rating: 7.4, xgxa90: 0.68, formDelta: -0.02 },
      { name: "McGinn", position: "MID", rating: 6.9, xgxa90: 0.29, formDelta: 0.01 },
    ],
    squadStats: [
      { label: "FRDS (Rating Dominance)", home: 1.02, away: 0.97 },
      { label: "XOC (Top-3 Off. Concentration)", home: 1.9, away: 1.7 },
    ],
    status: "upcoming",
  },
  {
    id: "rma-bay",
    league: "Champions League",
    tier: "general_purpose",
    day: "today",
    kickoff: "21:00",
    home: "Real Madrid",
    away: "Bayern Munich",
    overall: "conditional",
    confidence: "medium",
    markets: [
      {
        market: "result_3way",
        selection: "home",
        recommendationType: "conditional",
        currentOdds: 1.9,
        minOdds: 2.0,
        mlProbability: 0.53,
        impliedProbability: 0.53,
        valueEdge: 0.0,
      },
      {
        market: "total_goals",
        selection: "over_2.5",
        recommendationType: "no_bet",
        currentOdds: 1.65,
        minOdds: 2.0,
        mlProbability: 0.58,
        impliedProbability: 0.61,
        valueEdge: -0.03,
      },
    ],
    explanation:
      "This fixture uses the general-purpose model tier — market-odds features only, no team-history data for either club in this competition context. Home win is priced in line with the model; no independent edge to act on yet.",
    limitations: ["General-purpose tier: market-odds-only features, no squad-level data."],
    topFeatures: [
      { name: "MKT_Home_Prob_Real", value: "0.53", importance: 0.084 },
      { name: "MKT_Draw_Prob_Real", value: "0.26", importance: 0.031 },
    ],
    homePlayers: [],
    awayPlayers: [],
    squadStats: [],
    status: "upcoming",
  },
  {
    id: "psg-bvb",
    league: "Champions League",
    tier: "general_purpose",
    day: "in 2 days",
    kickoff: "21:00",
    home: "PSG",
    away: "Dortmund",
    overall: "direct_bet",
    confidence: "medium",
    markets: [
      {
        market: "total_goals",
        selection: "over_2.5",
        recommendationType: "direct_bet",
        currentOdds: 2.05,
        minOdds: 2.0,
        mlProbability: 0.6,
        impliedProbability: 0.49,
        valueEdge: 0.11,
      },
    ],
    explanation:
      "General-purpose tier. Market-implied probability for over 2.5 goals (49%) is well below the model's market-derived estimate (60%), and odds of 2.05 clear the minimum threshold.",
    limitations: ["General-purpose tier: market-odds-only features, no squad-level data."],
    topFeatures: [
      { name: "MKT_Home_Prob_Real", value: "0.44", importance: 0.058 },
      { name: "MKT_Draw_Prob_Real", value: "0.24", importance: 0.029 },
    ],
    homePlayers: [],
    awayPlayers: [],
    squadStats: [],
    status: "upcoming",
  },
  {
    id: "bra-arg",
    league: "International Friendly",
    tier: "general_purpose",
    day: "in 3 days",
    kickoff: "19:00",
    home: "Brazil",
    away: "Argentina",
    overall: "insufficient_data",
    confidence: "low",
    markets: [
      {
        market: "result_3way",
        selection: "home",
        recommendationType: "no_bet",
        currentOdds: null,
        minOdds: 2.0,
        mlProbability: 0.4,
        impliedProbability: 0,
        valueEdge: 0,
      },
    ],
    explanation:
      "No current odds could be found for this friendly, and feature completeness is very low (41%) — international friendlies have minimal comparable market history.",
    limitations: [
      "Current odds unavailable at analysis time.",
      "Feature completeness 41% — well below threshold.",
    ],
    topFeatures: [],
    homePlayers: [],
    awayPlayers: [],
    squadStats: [],
    status: "upcoming",
  },
  {
    id: "liv-mun",
    league: "Premier League",
    tier: "competition_specific",
    day: "3 days ago",
    kickoff: "16:30",
    home: "Liverpool",
    away: "Man United",
    overall: "direct_bet",
    confidence: "high",
    markets: [
      {
        market: "result_3way",
        selection: "home",
        recommendationType: "direct_bet",
        currentOdds: 2.05,
        minOdds: 2.0,
        mlProbability: 0.5,
        impliedProbability: 0.45,
        valueEdge: 0.05,
      },
    ],
    explanation:
      "Liverpool's home xG form and Man United's away defensive record pointed to value on the home win at 2.05.",
    limitations: [],
    topFeatures: [{ name: "OFF_HOME_XG_R5", value: "1.79", importance: 0.083 }],
    homePlayers: [{ name: "Salah", position: "FWD", rating: 8.4, xgxa90: 1.1, formDelta: 0.2 }],
    awayPlayers: [{ name: "Rashford", position: "FWD", rating: 6.5, xgxa90: 0.38, formDelta: -0.09 }],
    squadStats: [{ label: "FRDS (Rating Dominance)", home: 1.18, away: 0.89 }],
    status: "completed",
    result: { home: 3, away: 1 },
  },
  {
    id: "ars-che",
    league: "Premier League",
    tier: "competition_specific",
    day: "5 days ago",
    kickoff: "12:30",
    home: "Arsenal",
    away: "Chelsea",
    overall: "conditional",
    confidence: "medium",
    markets: [
      {
        market: "total_goals",
        selection: "over_2.5",
        recommendationType: "conditional",
        currentOdds: 1.8,
        minOdds: 2.0,
        mlProbability: 0.54,
        impliedProbability: 0.56,
        valueEdge: -0.02,
      },
    ],
    explanation:
      "Odds never reached the 2.0 threshold before kickoff, so this stayed a conditional watch rather than converting to a direct bet.",
    limitations: ["Odds below minimum threshold for a direct bet."],
    topFeatures: [{ name: "OFF_AWAY_XG_R5", value: "1.33", importance: 0.052 }],
    homePlayers: [{ name: "Ødegaard", position: "MID", rating: 7.1, xgxa90: 0.49, formDelta: -0.01 }],
    awayPlayers: [{ name: "Palmer", position: "MID", rating: 7.6, xgxa90: 0.81, formDelta: 0.1 }],
    squadStats: [{ label: "FRDS (Rating Dominance)", home: 1.03, away: 1.01 }],
    status: "completed",
    result: { home: 1, away: 1 },
  },
];

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

function formatPct(v: number) {
  return `${(v * 100).toFixed(0)}%`;
}
function formatEdge(v: number) {
  const pct = (v * 100).toFixed(1);
  return v >= 0 ? `+${pct}%` : `${pct}%`;
}
function bestMarket(match: Match): MarketRec {
  return [...match.markets].sort((a, b) => b.valueEdge - a.valueEdge)[0];
}
/** Best market by edge, unless a specific market filter is active — then
 *  that market's own row is shown (falls back to best-edge if the match
 *  doesn't carry that market at all, though filtered lists shouldn't hit
 *  that case). */
function displayMarket(match: Match, marketFilter: "all" | MarketKey): MarketRec {
  if (marketFilter === "all") return bestMarket(match);
  return match.markets.find((mk) => mk.market === marketFilter) ?? bestMarket(match);
}
function findMatch(id: string) {
  return MOCK_MATCHES.find((m) => m.id === id);
}

/** Real club home-kit colors (best-effort approximation, not sourced from
 *  official brand guidelines — fine for an internal draft, worth verifying
 *  before this ever ships). `secondary`, when present, renders as the
 *  badge's ring rather than a second fill, so text-contrast only ever has
 *  to be computed against one solid color. Away-kit colors intentionally
 *  not modeled. Teams not listed fall back to the categorical hash below. */
const TEAM_COLORS: Record<string, { primary: string; secondary?: string }> = {
  Liverpool: { primary: "#C8102E" },
  Arsenal: { primary: "#EF0107", secondary: "#FFFFFF" },
  Chelsea: { primary: "#034694" },
  Brighton: { primary: "#0057B8", secondary: "#FFFFFF" },
  "Man City": { primary: "#6CABDD" },
  Fulham: { primary: "#FFFFFF", secondary: "#000000" },
  Tottenham: { primary: "#FFFFFF", secondary: "#132257" },
  "West Ham": { primary: "#7A263A", secondary: "#1BB1E7" },
  Newcastle: { primary: "#241F20", secondary: "#FFFFFF" },
  "Aston Villa": { primary: "#670E36", secondary: "#95BFE5" },
  "Real Madrid": { primary: "#FFFFFF", secondary: "#FEBE10" },
  "Bayern Munich": { primary: "#DC052D", secondary: "#0066B2" },
  PSG: { primary: "#004170", secondary: "#DA291C" },
  Dortmund: { primary: "#FDE100", secondary: "#000000" },
  Brazil: { primary: "#FFDF00", secondary: "#009739" },
  Argentina: { primary: "#75AADB", secondary: "#FFFFFF" },
  "Man United": { primary: "#DA291C" },
};

/** Fallback for any team not yet added to TEAM_COLORS above — categorical
 *  hash palette (slot 1/blue skipped, reserved for the interactive accent),
 *  so a given team still always gets the same color, just not a brand one. */
const BADGE_FALLBACK_COLORS = ["#199e70", "#c98500", "#008300", "#9085e9", "#e66767", "#d55181", "#d95926"];
function badgeColor(name: string) {
  let hash = 0;
  for (let i = 0; i < name.length; i++) hash = (hash * 31 + name.charCodeAt(i)) >>> 0;
  return BADGE_FALLBACK_COLORS[hash % BADGE_FALLBACK_COLORS.length];
}
function teamColor(name: string) {
  return TEAM_COLORS[name] ?? { primary: badgeColor(name) };
}
/** Simple luminance heuristic (not full WCAG contrast math) to pick legible
 *  text color against a given fill — good enough for a small badge glyph. */
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

function StatusBadge({ status, size = "sm" }: { status: Overall; size?: "sm" | "lg" }) {
  const s = STATUS_META[status];
  const pad = size === "lg" ? "px-3 py-1.5 text-sm" : "px-2 py-0.5 text-[11px]";
  return (
    <span
      className={`inline-flex items-center gap-1.5 rounded-md border ${s.ring} ${s.text} ${pad} font-medium`}
    >
      {s.icon}
      {s.label}
    </span>
  );
}

function TeamBadge({ name }: { name: string }) {
  const { primary, secondary } = teamColor(name);
  return (
    <span
      className="flex h-6 w-6 shrink-0 items-center justify-center rounded-full text-[9px] font-bold"
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

function TierTag({ tier }: { tier: Tier }) {
  return (
    <span className="rounded border border-border px-1.5 py-0.5 text-[10px] font-medium uppercase tracking-wide text-ink-secondary">
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
            value === opt.value
              ? "border-accent text-ink"
              : "border-transparent text-ink-secondary hover:text-ink"
          }`}
        >
          {opt.label}
        </button>
      ))}
    </div>
  );
}

function DraftNav({ active }: { active: "dashboard" | "matches" }) {
  return (
    <div className="mb-8 flex items-center justify-between">
      <div className="flex items-baseline gap-1.5">
        <span className="text-sm font-semibold tracking-tight text-ink">FPAI</span>
        <span className="text-xs text-muted">draft</span>
      </div>
      <nav className="flex items-center gap-5 text-sm">
        <Link
          href="/"
          className={`transition-colors duration-150 ${
            active === "dashboard" ? "text-ink" : "text-ink-secondary hover:text-ink"
          }`}
        >
          Dashboard
        </Link>
        <Link
          href="/matches"
          className={`transition-colors duration-150 ${
            active === "matches" ? "text-ink" : "text-ink-secondary hover:text-ink"
          }`}
        >
          Matches
        </Link>
      </nav>
    </div>
  );
}

// ---------------------------------------------------------------------------
// MatchCard — shared card used by both Dashboard and Match Explorer. Click
// (or the chevron) expands an inline detail panel in place; a link inside
// that panel navigates to the full deep-dive page. `marketFilter` controls
// which market's odds/edge are shown — "all" falls back to best-edge.
// ---------------------------------------------------------------------------

function MatchCard({
  match,
  marketFilter = "all",
}: {
  match: Match;
  marketFilter?: "all" | MarketKey;
}) {
  const [open, setOpen] = useState(false);
  const shown = displayMarket(match, marketFilter);
  const isCompleted = match.status === "completed";

  return (
    <div className="rounded-lg border border-border transition-transform duration-150 hover:-translate-y-px">
      <button type="button" onClick={() => setOpen((v) => !v)} className="w-full p-3.5 text-left">
        <div className="flex items-center justify-between">
          <span className="flex items-center gap-2 font-mono text-xs text-muted">
            {match.kickoff}
            <TierTag tier={match.tier} />
          </span>
          <StatusBadge status={match.overall} />
        </div>

        <div className="mt-2.5 flex items-center gap-2">
          <TeamBadge name={match.home} />
          <span className="truncate text-sm font-medium text-ink">{match.home}</span>
          <span className="text-xs text-ink-secondary">v</span>
          <span className="truncate text-sm font-medium text-ink">{match.away}</span>
          <TeamBadge name={match.away} />
        </div>

        <div className="mt-3 flex items-end justify-between gap-3 border-t border-border pt-2.5">
          <span className="truncate text-xs text-ink-secondary">
            {MARKET_LABEL[shown.market]} · {shown.selection}
          </span>
          <div className="flex shrink-0 items-end gap-4">
            <div className="text-right">
              <div className="font-mono text-sm text-ink">
                {isCompleted
                  ? `${match.result?.home}-${match.result?.away}`
                  : shown.currentOdds
                  ? shown.currentOdds.toFixed(2)
                  : "—"}
              </div>
              <div className="text-[10px] uppercase tracking-wide text-muted">
                {isCompleted ? "Result" : "Odds"}
              </div>
            </div>
            <div className="text-right">
              <div
                className={`font-mono text-sm ${
                  !isCompleted && shown.currentOdds
                    ? shown.valueEdge >= 0
                      ? "text-good"
                      : "text-ink-secondary"
                    : "text-muted"
                }`}
              >
                {isCompleted ? match.day : shown.currentOdds ? formatEdge(shown.valueEdge) : "—"}
              </div>
              <div className="text-[10px] uppercase tracking-wide text-muted">
                {isCompleted ? "When" : "Edge"}
              </div>
            </div>
            <CaretDown
              size={14}
              className={`mb-1 text-ink-secondary transition-transform duration-150 ${
                open ? "rotate-180" : ""
              }`}
            />
          </div>
        </div>
      </button>

      <div className={`expand-rows ${open ? "is-open" : ""}`}>
        <div>
          <div className="border-t border-border p-3.5 text-sm">
            <p className="text-ink-secondary">{match.explanation}</p>
            {match.topFeatures.length > 0 && (
              <div className="mt-2.5 flex flex-wrap gap-1.5">
                {match.topFeatures.slice(0, 3).map((f) => (
                  <span
                    key={f.name}
                    className="rounded border border-border px-1.5 py-0.5 font-mono text-[11px] text-ink-secondary"
                  >
                    {f.name} · {f.value}
                  </span>
                ))}
              </div>
            )}
            <Link
              href={`/matches/${match.id}`}
              className="mt-3 inline-flex items-center gap-1 text-sm font-medium text-accent"
            >
              Full analysis <CaretRight size={12} />
            </Link>
          </div>
        </div>
      </div>
    </div>
  );
}

// ---------------------------------------------------------------------------
// Page 1 — Dashboard ("/"): today's matches, sorted by best edge, filterable
// by tier (EPL/General) and market (1X2/Over-Under).
// ---------------------------------------------------------------------------

export function DashboardDraft() {
  const [tier, setTier] = useState<"all" | Tier>("all");
  const [market, setMarket] = useState<"all" | "result_3way" | "total_goals">("all");

  const rows = useMemo(() => {
    return MOCK_MATCHES.filter((m) => m.day === "today" && m.status === "upcoming")
      .filter((m) => tier === "all" || m.tier === tier)
      .filter((m) => market === "all" || m.markets.some((mk) => mk.market === market))
      .sort((a, b) => bestMarket(b).valueEdge - bestMarket(a).valueEdge);
  }, [tier, market]);

  return (
    <main className="mx-auto max-w-4xl px-4 py-8 sm:px-6">
      <DraftNav active="dashboard" />

      <h1 className="text-xl font-semibold tracking-tight text-ink">Today&apos;s Edges</h1>
      <p className="mt-1 text-sm text-ink-secondary">
        Draft dashboard — mock data only. {rows.length} fixtures today, sorted by model edge.
      </p>

      <div className="mt-5 flex flex-wrap items-center gap-x-8 gap-y-2">
        <SegmentedControl
          value={tier}
          onChange={setTier}
          options={[
            { value: "all", label: "All Leagues" },
            { value: "competition_specific", label: "EPL" },
            { value: "general_purpose", label: "General" },
          ]}
        />
        <SegmentedControl
          value={market}
          onChange={setMarket}
          options={[
            { value: "all", label: "All Markets" },
            { value: "result_3way", label: "1X2" },
            { value: "total_goals", label: "Over/Under" },
          ]}
        />
      </div>

      <div className="mt-6 flex flex-col gap-2.5">
        {rows.length === 0 ? (
          <p className="py-8 text-center text-sm text-ink-secondary">
            No fixtures match these filters today.
          </p>
        ) : (
          rows.map((m) => <MatchCard key={m.id} match={m} marketFilter={market} />)
        )}
      </div>
    </main>
  );
}

// ---------------------------------------------------------------------------
// Page 2 — Match Explorer ("/matches"): search + full grid, upcoming and
// completed, for looking up a fixture that wasn't featured on the dashboard.
// ---------------------------------------------------------------------------

export function MatchExplorerDraft() {
  const [query, setQuery] = useState("");
  const [tier, setTier] = useState<"all" | Tier>("all");
  const [status, setStatus] = useState<"all" | "upcoming" | "completed">("all");

  const rows = useMemo(() => {
    const q = query.trim().toLowerCase();
    return MOCK_MATCHES.filter((m) => tier === "all" || m.tier === tier)
      .filter((m) => status === "all" || m.status === status)
      .filter(
        (m) =>
          q.length === 0 ||
          m.home.toLowerCase().includes(q) ||
          m.away.toLowerCase().includes(q)
      );
  }, [query, tier, status]);

  return (
    <main className="mx-auto max-w-4xl px-4 py-8 sm:px-6">
      <DraftNav active="matches" />

      <h1 className="text-xl font-semibold tracking-tight text-ink">Match Explorer</h1>
      <p className="mt-1 text-sm text-ink-secondary">
        Search any fixture, upcoming or completed. Draft data only.
      </p>

      <div className="relative mt-5">
        <MagnifyingGlass
          size={16}
          className="pointer-events-none absolute left-3 top-1/2 -translate-y-1/2 text-muted"
        />
        <input
          value={query}
          onChange={(e) => setQuery(e.target.value)}
          placeholder="Search by team name…"
          className="w-full rounded-lg border border-border bg-surface py-2 pl-9 pr-3 text-sm text-ink outline-none placeholder:text-muted focus:border-accent"
        />
      </div>

      <div className="mt-4 flex flex-wrap items-center gap-x-8 gap-y-2">
        <SegmentedControl
          value={tier}
          onChange={setTier}
          options={[
            { value: "all", label: "All Leagues" },
            { value: "competition_specific", label: "EPL" },
            { value: "general_purpose", label: "General" },
          ]}
        />
        <SegmentedControl
          value={status}
          onChange={setStatus}
          options={[
            { value: "all", label: "All" },
            { value: "upcoming", label: "Upcoming" },
            { value: "completed", label: "Completed" },
          ]}
        />
      </div>

      <div className="mt-6 flex flex-col gap-2.5">
        {rows.length === 0 ? (
          <p className="py-8 text-center text-sm text-ink-secondary">
            No matches found for &ldquo;{query}&rdquo;.
          </p>
        ) : (
          rows.map((m) => <MatchCard key={m.id} match={m} />)
        )}
      </div>
    </main>
  );
}

// ---------------------------------------------------------------------------
// Page 3 — Match Analysis & Agent Intelligence ("/matches/:id")
// ---------------------------------------------------------------------------

function ProbabilityRow({ m }: { m: MarketRec }) {
  const s = STATUS_META[m.recommendationType];
  return (
    <div className="grid grid-cols-[1fr_auto_auto_auto_auto] items-center gap-4 border-b border-border py-3 text-sm last:border-b-0">
      <div>
        <div className="font-medium text-ink">{MARKET_LABEL[m.market]}</div>
        <div className="text-xs text-ink-secondary">{m.selection}</div>
      </div>
      <div className="text-right font-mono text-ink">{formatPct(m.mlProbability)}</div>
      <div className="text-right font-mono text-ink-secondary">
        {m.currentOdds ? formatPct(m.impliedProbability) : "—"}
      </div>
      <div
        className={`text-right font-mono ${
          m.currentOdds && m.valueEdge >= 0 ? "text-good" : "text-ink-secondary"
        }`}
      >
        {m.currentOdds ? formatEdge(m.valueEdge) : "—"}
      </div>
      <span
        className={`justify-self-end rounded-md border ${s.ring} ${s.text} px-2 py-0.5 text-[11px] font-medium`}
      >
        {s.label}
      </span>
    </div>
  );
}

function PlayerRow({ p }: { p: PlayerStat }) {
  const up = p.formDelta >= 0;
  return (
    <div className="flex items-center justify-between border-b border-border py-2 text-sm last:border-b-0">
      <div className="flex items-center gap-2">
        <span className="w-9 font-mono text-[10px] text-muted">{p.position}</span>
        <span className="text-ink">{p.name}</span>
      </div>
      <div className="flex items-center gap-4 font-mono text-xs">
        <span className="text-ink-secondary">rtg {p.rating.toFixed(1)}</span>
        <span className="text-ink-secondary">{p.xgxa90.toFixed(2)} xG+xA/90</span>
        <span className={`flex items-center gap-0.5 ${up ? "text-good" : "text-serious"}`}>
          {up ? <TrendUp size={12} /> : <TrendDown size={12} />}
          {Math.abs(p.formDelta).toFixed(2)}
        </span>
      </div>
    </div>
  );
}

export function MatchAnalysisDraft({ id }: { id: string }) {
  const match = findMatch(id);

  if (!match) {
    return (
      <main className="mx-auto max-w-4xl px-4 py-8 sm:px-6">
        <DraftNav active="matches" />
        <p className="text-sm text-ink-secondary">
          No draft fixture with id &ldquo;{id}&rdquo;.{" "}
          <Link href="/matches" className="text-accent">
            Back to Match Explorer
          </Link>
        </p>
      </main>
    );
  }

  const s = STATUS_META[match.overall];
  const hasSquadData = match.homePlayers.length > 0 || match.awayPlayers.length > 0;

  return (
    <main className="mx-auto max-w-4xl px-4 py-8 sm:px-6">
      <DraftNav active="matches" />

      <Link
        href="/matches"
        className="inline-flex items-center gap-1.5 text-sm text-ink-secondary transition-colors duration-150 hover:text-ink"
      >
        <ArrowLeft size={14} /> Back to Matches
      </Link>

      <div className="mt-4 flex items-start justify-between gap-4">
        <div>
          <div className="flex items-center gap-2 text-xs text-ink-secondary">
            <span>{match.league}</span>
            <TierTag tier={match.tier} />
            <span>
              {match.day} · {match.kickoff}
            </span>
          </div>
          <h1 className="mt-1 text-2xl font-semibold tracking-tight text-ink">
            {match.home} <span className="text-ink-secondary">vs</span> {match.away}
          </h1>
          {match.status === "completed" && match.result && (
            <p className="mt-1 font-mono text-sm text-ink-secondary">
              Final score {match.result.home}-{match.result.away}
            </p>
          )}
        </div>
        <div className="text-right">
          <div className={`text-2xl font-bold tracking-tight ${s.text}`}>{s.verdict}</div>
          <div className="mt-1 text-xs text-ink-secondary">
            Confidence: <span className="font-medium text-ink">{match.confidence}</span>
          </div>
        </div>
      </div>

      <section className="mt-8">
        <h2 className="text-sm font-semibold uppercase tracking-wide text-muted">
          Model Probabilities
        </h2>
        <div className="mt-2 grid grid-cols-[1fr_auto_auto_auto_auto] gap-4 text-[11px] uppercase tracking-wide text-muted">
          <span />
          <span className="text-right">Model</span>
          <span className="text-right">Market</span>
          <span className="text-right">Edge</span>
          <span className="justify-self-end">Status</span>
        </div>
        {match.markets.map((m) => (
          <ProbabilityRow key={m.market} m={m} />
        ))}
      </section>

      <section className="mt-8">
        <h2 className="text-sm font-semibold uppercase tracking-wide text-muted">
          Squad Intelligence
        </h2>
        {!hasSquadData ? (
          <p className="mt-2 rounded-lg border border-border bg-surface p-3.5 text-sm text-ink-secondary">
            Unavailable for this fixture — {TIER_LABEL[match.tier]} tier uses market-odds-only
            forecasting, no team-history or player data.
          </p>
        ) : (
          <>
            {match.squadStats.length > 0 && (
              <div className="mt-2 grid grid-cols-2 gap-3">
                {match.squadStats.map((stat) => (
                  <div key={stat.label} className="rounded-lg border border-border p-3">
                    <div className="text-xs text-ink-secondary">{stat.label}</div>
                    <div className="mt-1 flex items-baseline gap-2 font-mono text-lg text-ink">
                      {stat.home.toFixed(2)}
                      <span className="text-xs font-sans text-muted">vs</span>
                      {stat.away.toFixed(2)}
                    </div>
                  </div>
                ))}
              </div>
            )}
            <div className="mt-3 grid grid-cols-2 gap-4">
              <div>
                <div className="mb-1 text-xs text-ink-secondary">{match.home}</div>
                {match.homePlayers.map((p) => (
                  <PlayerRow key={p.name} p={p} />
                ))}
              </div>
              <div>
                <div className="mb-1 text-xs text-ink-secondary">{match.away}</div>
                {match.awayPlayers.map((p) => (
                  <PlayerRow key={p.name} p={p} />
                ))}
              </div>
            </div>
          </>
        )}
      </section>

      {match.topFeatures.length > 0 && (
        <section className="mt-8">
          <h2 className="text-sm font-semibold uppercase tracking-wide text-muted">
            Explainability
          </h2>
          <div className="mt-2 flex flex-col gap-2">
            {match.topFeatures.map((f) => (
              <div key={f.name} className="flex items-center gap-3 text-sm">
                <span className="w-44 shrink-0 truncate font-mono text-xs text-ink-secondary">
                  {f.name}
                </span>
                <span className="w-14 shrink-0 text-right font-mono text-ink">{f.value}</span>
                <span className="h-px flex-1 bg-border">
                  <span
                    className="block h-px bg-accent"
                    style={{ width: `${Math.min(f.importance * 100 * 6, 100)}%` }}
                  />
                </span>
              </div>
            ))}
          </div>
        </section>
      )}

      <section className="mt-8">
        <h2 className="text-sm font-semibold uppercase tracking-wide text-muted">
          Agent Reasoning
        </h2>
        <div className="mt-2 rounded-lg border border-border bg-surface p-4">
          <p className="text-sm leading-relaxed text-ink-secondary">{match.explanation}</p>
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
    </main>
  );
}
