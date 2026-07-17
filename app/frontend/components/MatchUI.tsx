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
  ArrowLeft,
  CaretDown,
  CaretRight,
  CheckCircle,
  Clock,
  MagnifyingGlass,
  MinusCircle,
  Question,
  WarningCircle,
} from "@phosphor-icons/react";

import { ApiError, generateRecommendation, getFixtures, logBetFromRecommendation } from "@/lib/api";
import type { Fixture, MatchRecommendationOut } from "@/lib/types";
import { useSandboxAsOf } from "@/lib/useSandboxAsOf";

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
};

export type Match = {
  id: string;
  league: string;
  tier: Tier;
  kickoffIso: string;
  home: string;
  away: string;
  status: "upcoming" | "completed";
  result?: { home: number; away: number };
  // Recommendation data -- absent until generated (hasRecommendation gates this).
  hasRecommendation: boolean;
  overall: Overall;
  confidence: Confidence;
  markets: MarketRec[];
  explanation: string;
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
};

// ---------------------------------------------------------------------------
// Adapters -- real API shapes -> UI Match shape
// ---------------------------------------------------------------------------

function fixtureToMatch(fixture: Fixture): Match {
  return {
    id: fixture.match_id,
    league: "E0",
    tier: "competition_specific",
    kickoffIso: fixture.utc_date,
    home: fixture.home_team,
    away: fixture.away_team,
    status: fixture.status === "FINISHED" ? "completed" : "upcoming",
    result:
      fixture.home_goals !== null && fixture.away_goals !== null
        ? { home: fixture.home_goals, away: fixture.away_goals }
        : undefined,
    hasRecommendation: false,
    overall: "insufficient_data",
    confidence: "low",
    markets: [],
    explanation: "",
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
    })),
  };
}

function formatKickoff(iso: string): string {
  const d = new Date(iso);
  return d.toLocaleTimeString(undefined, { hour: "2-digit", minute: "2-digit" });
}

function formatDay(iso: string): string {
  const date = new Date(iso);
  const today = new Date();
  const dOnly = new Date(date.getFullYear(), date.getMonth(), date.getDate());
  const tOnly = new Date(today.getFullYear(), today.getMonth(), today.getDate());
  const diffDays = Math.round((dOnly.getTime() - tOnly.getTime()) / 86_400_000);
  if (diffDays === 0) return "today";
  if (diffDays === 1) return "tomorrow";
  if (diffDays === -1) return "yesterday";
  if (diffDays > 1) return `in ${diffDays} days`;
  return `${-diffDays} days ago`;
}

// ---------------------------------------------------------------------------
// Display metadata
// ---------------------------------------------------------------------------

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
// Helpers ported from DraftUI.tsx
// ---------------------------------------------------------------------------

function formatPct(v: number) {
  return `${(v * 100).toFixed(0)}%`;
}
function formatEdge(v: number) {
  const pct = (v * 100).toFixed(1);
  return v >= 0 ? `+${pct}%` : `${pct}%`;
}
function bestMarket(match: Match): MarketRec | undefined {
  return [...match.markets].sort((a, b) => b.valueEdge - a.valueEdge)[0];
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

export function StatusBadge({ status, size = "sm" }: { status: Overall; size?: "sm" | "lg" }) {
  const s = STATUS_META[status];
  const pad = size === "lg" ? "px-3 py-1.5 text-sm" : "px-2 py-0.5 text-[11px]";
  return (
    <span className={`inline-flex items-center gap-1.5 rounded-md border ${s.ring} ${s.text} ${pad} font-medium`}>
      {s.icon}
      {s.label}
    </span>
  );
}

/** W15: a first-class trust signal, independent of predictionBasis/overall --
 * renders whenever cold_start_risk or unknown_team is true, even if
 * predictionBasis itself claims full team_history_and_market coverage. */
function TrustSignal({ match, size = "sm" }: { match: Match; size?: "sm" | "lg" }) {
  if (!match.coldStartRisk && !match.unknownTeam) return null;
  const label = match.unknownTeam ? "Unseen team — no history" : "Cold start — thin history";
  const pad = size === "lg" ? "px-3 py-1.5 text-sm" : "px-2 py-0.5 text-[11px]";
  return (
    <span
      className={`inline-flex items-center gap-1.5 rounded-md border border-warning/40 text-warning ${pad} font-medium`}
      title={
        match.featureCompleteness !== null
          ? `feature_completeness=${match.featureCompleteness.toFixed(2)}`
          : undefined
      }
    >
      <WarningCircle weight="fill" size={size === "lg" ? 15 : 13} />
      {label}
    </span>
  );
}

export function TeamBadge({ name }: { name: string }) {
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
            value === opt.value ? "border-accent text-ink" : "border-transparent text-ink-secondary hover:text-ink"
          }`}
        >
          {opt.label}
        </button>
      ))}
    </div>
  );
}

export function DraftNav({ active }: { active: "dashboard" | "matches" | "bets" }) {
  return (
    <div className="mb-8 flex items-center justify-between">
      <div className="flex items-baseline gap-1.5">
        <span className="text-sm font-semibold tracking-tight text-ink">FPAI</span>
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
        <Link
          href="/bets"
          className={`transition-colors duration-150 ${
            active === "bets" ? "text-ink" : "text-ink-secondary hover:text-ink"
          }`}
        >
          Bets
        </Link>
      </nav>
    </div>
  );
}

function ErrorState({ message, onRetry }: { message: string; onRetry?: () => void }) {
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

export function MatchCard({ match, onUpdate }: { match: Match; onUpdate: (m: Match) => void }) {
  const [open, setOpen] = useState(false);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const isCompleted = match.status === "completed";
  const shown = bestMarket(match);

  async function handleExpand() {
    const next = !open;
    setOpen(next);
    if (next && !match.hasRecommendation && !loading) {
      setLoading(true);
      setError(null);
      try {
        const rec = await generateRecommendation({
          home_team: match.home,
          away_team: match.away,
          date: match.kickoffIso.slice(0, 10),
          league: match.league,
          match_id: match.id,
        });
        onUpdate(applyRecommendation(match, rec));
      } catch (err) {
        setError(err instanceof ApiError ? err.message : "Could not reach the agent.");
      } finally {
        setLoading(false);
      }
    }
  }

  return (
    <div className="rounded-lg border border-border transition-transform duration-150 hover:-translate-y-px">
      <button type="button" onClick={handleExpand} className="w-full p-3.5 text-left">
        <div className="flex items-center justify-between">
          <span className="flex items-center gap-2 font-mono text-xs text-muted">
            {formatKickoff(match.kickoffIso)}
            <TierTag tier={match.tier} />
          </span>
          {match.hasRecommendation ? (
            <span className="flex items-center gap-1.5">
              <TrustSignal match={match} />
              <StatusBadge status={match.overall} />
            </span>
          ) : (
            <span className="text-[11px] uppercase tracking-wide text-muted">
              {isCompleted ? "Settled" : "Not yet generated"}
            </span>
          )}
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
            {shown ? `${shown.market} · ${shown.selection}` : formatDay(match.kickoffIso)}
          </span>
          <div className="flex shrink-0 items-end gap-4">
            <div className="text-right">
              <div className="font-mono text-sm text-ink">
                {isCompleted && match.result
                  ? `${match.result.home}-${match.result.away}`
                  : shown?.currentOdds
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
                  shown?.currentOdds ? (shown.valueEdge >= 0 ? "text-good" : "text-ink-secondary") : "text-muted"
                }`}
              >
                {shown?.currentOdds ? formatEdge(shown.valueEdge) : "—"}
              </div>
              <div className="text-[10px] uppercase tracking-wide text-muted">Edge</div>
            </div>
            <CaretDown
              size={14}
              className={`mb-1 text-ink-secondary transition-transform duration-150 ${open ? "rotate-180" : ""}`}
            />
          </div>
        </div>
      </button>

      <div className={`expand-rows ${open ? "is-open" : ""}`}>
        <div>
          <div className="border-t border-border p-3.5 text-sm">
            {loading && <LoadingRows count={1} />}
            {error && <ErrorState message={error} onRetry={handleExpand} />}
            {!loading && !error && match.hasRecommendation && (
              <>
                <p className="text-ink-secondary">{match.explanation}</p>
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
                  )}&date=${match.kickoffIso.slice(0, 10)}`}
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
  const asOf = useSandboxAsOf();
  const [matches, setMatches] = useState<Match[] | null>(null);
  const [error, setError] = useState<string | null>(null);

  async function load() {
    setError(null);
    setMatches(null);
    try {
      const today = asOf.toISOString().slice(0, 10);
      const fixtures = await getFixtures(today, today);
      setMatches(fixtures.map(fixtureToMatch));
    } catch (err) {
      setError(err instanceof ApiError ? err.message : "Could not load today's fixtures.");
    }
  }

  useEffect(() => {
    load();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [asOf]);

  function updateMatch(updated: Match) {
    setMatches((prev) => prev?.map((m) => (m.id === updated.id ? updated : m)) ?? null);
  }

  return (
    <main className="mx-auto max-w-4xl px-4 py-8 sm:px-6">
      <DraftNav active="dashboard" />

      <h1 className="text-xl font-semibold tracking-tight text-ink">Today&apos;s Edges</h1>
      <p className="mt-1 text-sm text-ink-secondary">
        Real E0 fixtures for today. Expand a card to generate its recommendation.
      </p>

      <div className="mt-6">
        {error && <ErrorState message={error} onRetry={load} />}
        {!error && matches === null && <LoadingRows />}
        {!error && matches !== null && matches.length === 0 && (
          <p className="py-8 text-center text-sm text-ink-secondary">No E0 fixtures today.</p>
        )}
        {!error && matches && matches.length > 0 && (
          <div className="flex flex-col gap-2.5">
            {matches.map((m) => (
              <MatchCard key={m.id} match={m} onUpdate={updateMatch} />
            ))}
          </div>
        )}
      </div>
    </main>
  );
}

// ---------------------------------------------------------------------------
// Page 2 -- Match Explorer ("/matches"): search across a wider fixture window.
// ---------------------------------------------------------------------------

export function MatchExplorerPage() {
  const asOf = useSandboxAsOf();
  const [query, setQuery] = useState("");
  const [matches, setMatches] = useState<Match[] | null>(null);
  const [error, setError] = useState<string | null>(null);

  async function load() {
    setError(null);
    setMatches(null);
    try {
      const from = new Date(asOf);
      const to = new Date(asOf);
      // Widened from 30 to 90 days after live verification showed the
      // off-season gap between fixture windows can exceed 30 days (e.g.
      // 2026-07-11 -> next real fixture 2026-08-21, 41 days out).
      // UTC methods, not local getDate/setDate: from/to are read back via
      // toISOString() (always UTC) below, and asOf is UTC midnight (W30) --
      // mixing local date arithmetic with a UTC value shifts the window by
      // a day in negative-UTC-offset timezones.
      to.setUTCDate(to.getUTCDate() + 90);
      const fixtures = await getFixtures(from.toISOString().slice(0, 10), to.toISOString().slice(0, 10));
      setMatches(fixtures.map(fixtureToMatch));
    } catch (err) {
      setError(err instanceof ApiError ? err.message : "Could not load fixtures.");
    }
  }

  useEffect(() => {
    load();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [asOf]);

  function updateMatch(updated: Match) {
    setMatches((prev) => prev?.map((m) => (m.id === updated.id ? updated : m)) ?? null);
  }

  const rows = useMemo(() => {
    const q = query.trim().toLowerCase();
    if (!matches) return null;
    if (q.length === 0) return matches;
    return matches.filter((m) => m.home.toLowerCase().includes(q) || m.away.toLowerCase().includes(q));
  }, [matches, query]);

  return (
    <main className="mx-auto max-w-4xl px-4 py-8 sm:px-6">
      <DraftNav active="matches" />

      <h1 className="text-xl font-semibold tracking-tight text-ink">Match Explorer</h1>
      <p className="mt-1 text-sm text-ink-secondary">Search real upcoming E0 fixtures (next 90 days).</p>

      <div className="relative mt-5">
        <MagnifyingGlass size={16} className="pointer-events-none absolute left-3 top-1/2 -translate-y-1/2 text-muted" />
        <input
          value={query}
          onChange={(e) => setQuery(e.target.value)}
          placeholder="Search by team name…"
          className="w-full rounded-lg border border-border bg-surface py-2 pl-9 pr-3 text-sm text-ink outline-none placeholder:text-muted focus:border-accent"
        />
      </div>

      <div className="mt-6">
        {error && <ErrorState message={error} onRetry={load} />}
        {!error && rows === null && <LoadingRows />}
        {!error && rows !== null && rows.length === 0 && (
          <p className="py-8 text-center text-sm text-ink-secondary">No matches found.</p>
        )}
        {!error && rows && rows.length > 0 && (
          <div className="flex flex-col gap-2.5">
            {rows.map((m) => (
              <MatchCard key={m.id} match={m} onUpdate={updateMatch} />
            ))}
          </div>
        )}
      </div>
    </main>
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
        {matchId && recommendation && !anomalous && (
          <LogBetButton matchId={matchId} recommendation={recommendation} market={m.market} selection={m.selection} />
        )}
      </span>
      <span className="text-right font-mono text-ink">{formatPct(m.mlProbability)}</span>
      <span className={`text-right font-mono ${anomalous ? "text-serious" : "text-ink-secondary"}`}>
        {m.currentOdds ? m.currentOdds.toFixed(2) : anomalous ? "missing" : "—"}
      </span>
      <span className={`text-right font-mono ${m.valueEdge >= 0 ? "text-good" : "text-ink-secondary"}`}>
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

export function MatchAnalysisPage({
  id,
  home,
  away,
  date,
}: {
  id: string;
  home: string;
  away: string;
  date: string;
}) {
  const [match, setMatch] = useState<Match | null>(null);
  const [rawRecommendation, setRawRecommendation] = useState<MatchRecommendationOut | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  async function load() {
    setLoading(true);
    setError(null);
    try {
      const rec = await generateRecommendation({ home_team: home, away_team: away, date, league: "E0", match_id: id });
      setRawRecommendation(rec);
      setMatch(
        applyRecommendation(
          {
            id,
            league: "E0",
            tier: "competition_specific",
            kickoffIso: date,
            home,
            away,
            status: "upcoming",
            hasRecommendation: false,
            overall: "insufficient_data",
            confidence: "low",
            markets: [],
            explanation: "",
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
  }, [id, home, away, date]);

  if (!home || !away || !date) {
    return (
      <main className="mx-auto max-w-4xl px-4 py-8 sm:px-6">
        <DraftNav active="matches" />
        <p className="text-sm text-ink-secondary">
          Missing match details.{" "}
          <Link href="/matches" className="text-accent">
            Back to Match Explorer
          </Link>
        </p>
      </main>
    );
  }

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
            <span>E0</span>
            <TierTag tier="competition_specific" />
            <span>{date}</span>
          </div>
          <h1 className="mt-1 text-2xl font-semibold tracking-tight text-ink">
            {home} <span className="text-ink-secondary">vs</span> {away}
          </h1>
        </div>
        {match && (
          <div className="text-right">
            <div className={`text-2xl font-bold tracking-tight ${STATUS_META[match.overall].text}`}>
              {STATUS_META[match.overall].verdict}
            </div>
            <div className="mt-1 text-xs text-ink-secondary">
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
          <section className="mt-8">
            <h2 className="text-sm font-semibold uppercase tracking-wide text-muted">Model Probabilities</h2>
            <div className="mt-2 grid grid-cols-[1fr_auto_auto_auto_auto] gap-4 text-[11px] uppercase tracking-wide text-muted">
              <span />
              <span className="text-right">Model</span>
              <span className="text-right">Market</span>
              <span className="text-right">Edge</span>
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

          <section className="mt-8">
            <h2 className="text-sm font-semibold uppercase tracking-wide text-muted">Squad Intelligence</h2>
            <p className="mt-2 rounded-lg border border-border bg-surface p-3.5 text-sm text-ink-secondary">
              Not yet exposed by the API for this view.
            </p>
          </section>

          <section className="mt-8">
            <h2 className="text-sm font-semibold uppercase tracking-wide text-muted">Agent Reasoning</h2>
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
        </>
      )}
    </main>
  );
}
