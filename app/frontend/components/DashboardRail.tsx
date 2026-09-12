"use client";

import Link from "next/link";
import { useMemo } from "react";

import { computeStakingSummary, countByOverall, rankTopEdges } from "@/lib/dashboardMetrics";
import { computeHit, formatEdge, formatMoneyWon, type Match, type Overall } from "./MatchUI";

// W64: fixed order + colors reused verbatim from STATUS_META's existing,
// already-locked status palette (app/globals.css D6) -- not a new color
// pick. Inline styles (not Tailwind classes) since these are chosen
// programmatically per status key, same pattern TeamBadge already uses for
// per-team colors.
//
// "completed_*" (direct user request) aren't Overall values -- they're
// match-status facts tracked separately by countByOverall -- so the slice
// key type widens to include them here rather than in the Overall union
// itself. Direct user follow-up request: split the single "completed"
// bucket into hit/not-hit -- reuses direct_bet's green for a hit and
// insufficient_data's red-orange for a miss (same HitBadge convention
// MatchCard's own card view already uses, `text-good`/`text-serious`) so
// the meaning ("good outcome" / "bad outcome") is consistent across the
// app, not a new color pick; completed_unresolved (no determinable hit --
// no actual pick was made, or an unresolvable market) keeps the original
// --accent blue, a "done, no verdict" color still otherwise unused here.
// DONUT_ORDER is arranged so no two color-sharing slices ever sit next to
// each other in the ring (direct_bet green -> ... -> insufficient_data red
// -> completed_hit green -> completed_miss red -> completed_unresolved
// blue -> wraps back to direct_bet green): every adjacent pair differs.
type SliceKey = Overall | "completed_hit" | "completed_miss" | "completed_unresolved";
const DONUT_ORDER: SliceKey[] = [
  "direct_bet", "conditional", "no_bet", "insufficient_data",
  "completed_hit", "completed_miss", "completed_unresolved",
];
const DONUT_COLOR: Record<SliceKey, string> = {
  direct_bet: "var(--status-good)",
  conditional: "var(--status-warning)",
  no_bet: "var(--text-muted)",
  insufficient_data: "var(--status-serious)",
  completed_hit: "var(--status-good)",
  completed_miss: "var(--status-serious)",
  completed_unresolved: "var(--accent)",
};
const DONUT_LABEL: Record<SliceKey, string> = {
  direct_bet: "Direct Bet",
  conditional: "Conditional",
  no_bet: "No Edge",
  insufficient_data: "No Data",
  completed_hit: "Completed (Hit)",
  completed_miss: "Completed (Not Hit)",
  completed_unresolved: "Completed",
};

const RADIUS = 40;
const CIRCUMFERENCE = 2 * Math.PI * RADIUS;
const SEGMENT_GAP = 3; // dataviz mark spec: a visible surface gap between adjacent segments

function matchHref(m: Match) {
  return `/matches/${m.id}?home=${encodeURIComponent(m.home)}&away=${encodeURIComponent(
    m.away
  )}&date=${m.kickoffIso.slice(0, 10)}&league=${encodeURIComponent(m.league)}`;
}

export function DashboardRail({ matches }: { matches: Match[] }) {
  const counts = useMemo(() => countByOverall(matches), [matches]);
  const topEdges = useMemo(() => rankTopEdges(matches, 5), [matches]);
  const staking = useMemo(() => computeStakingSummary(matches), [matches]);
  const total = DONUT_ORDER.reduce((sum, key) => sum + counts[key], 0);

  let cumulative = 0;
  const arcs = DONUT_ORDER.filter((key) => counts[key] > 0).map((key) => {
    const frac = counts[key] / total;
    const rawDash = frac * CIRCUMFERENCE;
    const arc = { key, dash: Math.max(rawDash - SEGMENT_GAP, 0), offset: cumulative };
    cumulative += rawDash;
    return arc;
  });

  return (
    <aside className="flex w-full flex-col gap-6 lg:w-72">
      {/* Mockup correction: the rail's own panels carry a distinct accent
          tint (were plain border-border, no fill) -- direct feedback that
          the left sidebar and this rail shouldn't read as the same color;
          reuses the existing --accent token (no new color introduced) at
          low opacity rather than a flat neutral fill. */}
      <section className="rounded-lg border border-accent/25 bg-accent/10 p-4">
        <h2 className="text-xs font-bold uppercase tracking-wide text-muted">Edge Distribution</h2>
        {total === 0 ? (
          <p className="mt-3 text-sm text-ink-secondary">No matches loaded yet.</p>
        ) : (
          <div className="mt-3 flex items-center gap-4">
            <svg viewBox="0 0 100 100" width={88} height={88} className="shrink-0 -rotate-90" aria-hidden="true">
              <circle cx="50" cy="50" r={RADIUS} fill="none" stroke="var(--gridline)" strokeWidth={14} />
              {arcs.map((arc) => (
                <circle
                  key={arc.key}
                  cx="50"
                  cy="50"
                  r={RADIUS}
                  fill="none"
                  stroke={DONUT_COLOR[arc.key]}
                  strokeWidth={14}
                  strokeDasharray={`${arc.dash} ${CIRCUMFERENCE - arc.dash}`}
                  strokeDashoffset={-arc.offset}
                />
              ))}
              <text
                x="50"
                y="50"
                textAnchor="middle"
                dominantBaseline="central"
                className="fill-ink text-[22px] font-semibold"
                style={{ transform: "rotate(90deg)", transformOrigin: "50px 50px" }}
              >
                {total}
              </text>
            </svg>
            <ul className="flex flex-1 flex-col gap-1.5 text-xs">
              {DONUT_ORDER.filter((key) => counts[key] > 0).map((key) => (
                <li key={key} className="flex items-center justify-between gap-2">
                  <span className="flex items-center gap-1.5 text-ink-secondary">
                    <span className="h-2 w-2 shrink-0 rounded-full" style={{ background: DONUT_COLOR[key] }} />
                    {DONUT_LABEL[key]}
                  </span>
                  <span className="font-mono text-ink">{counts[key]}</span>
                </li>
              ))}
            </ul>
          </div>
        )}
      </section>

      {/* Mockup correction: the rail's own panels carry a distinct accent
          tint (were plain border-border, no fill) -- direct feedback that
          the left sidebar and this rail shouldn't read as the same color;
          reuses the existing --accent token (no new color introduced) at
          low opacity rather than a flat neutral fill. */}
      <section className="rounded-lg border border-accent/25 bg-accent/10 p-4">
        <h2 className="text-xs font-bold uppercase tracking-wide text-muted">Top Edges</h2>
        {topEdges.length === 0 ? (
          <p className="mt-3 text-sm text-ink-secondary">No priced edges yet.</p>
        ) : (
          <ul className="mt-3 flex flex-col gap-2.5">
            {topEdges.map(({ match, edge }) => {
              // Direct user request: a completed match's edge was priced
              // pre-match and is no longer a live opportunity -- grey out
              // and strike through the team-name line (same "this is
              // history" treatment MatchCard's own completed Pick column
              // already uses for a miss, just applied to every completed
              // row here regardless of outcome) and append the actual
              // Hit/Not Hit verdict (computeHit() -- same rule HitBadge
              // uses) right after the edge %. null (no_bet was never
              // reachable here since rankTopEdges already excludes it, but
              // an unresolvable market like corners still can be) renders
              // no verdict, same null-propagation contract as everywhere
              // else this rule is used.
              const isCompleted = match.status === "completed";
              const hit = isCompleted ? computeHit(match) : null;
              return (
                <li key={match.id}>
                  <Link href={matchHref(match)} className="flex items-center justify-between gap-2 text-sm text-ink-secondary hover:text-ink">
                    <span className={`truncate ${isCompleted ? "text-muted line-through" : ""}`}>
                      {match.home} v {match.away}
                    </span>
                    <span className="shrink-0 flex items-center gap-1.5 font-mono">
                      <span className="text-good">{formatEdge(edge)}</span>
                      {hit !== null && <span className={hit ? "text-good" : "text-serious"}>{hit ? "Hit" : "Not Hit"}</span>}
                    </span>
                  </Link>
                </li>
              );
            })}
          </ul>
        )}
      </section>

      {/* Direct user request: a staking recap -- total staked, total
          won/lost, and average odds taken, in UB -- across every
          completed, actually-staked pick currently loaded. Same
          what's-loaded-today scoping every other rail stat already uses;
          not the real-money, all-time StatsBar BetTracker shows (a
          different data source, real logged bets, not model picks). Same
          rail panel treatment as the two sections above. */}
      <section className="rounded-lg border border-accent/25 bg-accent/10 p-4">
        <h2 className="text-xs font-bold uppercase tracking-wide text-muted">Staking Summary</h2>
        {staking.avgOdds === null ? (
          <p className="mt-3 text-sm text-ink-secondary">No settled picks yet.</p>
        ) : (
          <ul className="mt-3 flex flex-col gap-1.5 text-xs">
            <li className="flex items-center justify-between gap-2">
              <span className="text-ink-secondary">Staked</span>
              <span className="font-mono text-ink">{staking.staked.toFixed(1)} UB</span>
            </li>
            <li className="flex items-center justify-between gap-2">
              <span className="text-ink-secondary">Won</span>
              <span className={`font-mono ${staking.won > 0 ? "text-good" : staking.won < 0 ? "text-serious" : "text-ink"}`}>
                {formatMoneyWon(staking.won)}
              </span>
            </li>
            <li className="flex items-center justify-between gap-2">
              <span className="text-ink-secondary">Average odds</span>
              <span className="font-mono text-ink">{staking.avgOdds.toFixed(2)}</span>
            </li>
          </ul>
        )}
      </section>
    </aside>
  );
}
