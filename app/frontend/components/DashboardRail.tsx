"use client";

import Link from "next/link";
import { useMemo } from "react";

import { countByOverall, rankTopEdges } from "@/lib/dashboardMetrics";
import { formatEdge, type Match, type Overall } from "./MatchUI";

// W64: fixed order + colors reused verbatim from STATUS_META's existing,
// already-locked status palette (app/globals.css D6) -- not a new color
// pick. Inline styles (not Tailwind classes) since these are chosen
// programmatically per status key, same pattern TeamBadge already uses for
// per-team colors.
const DONUT_ORDER: Overall[] = ["direct_bet", "conditional", "no_bet", "insufficient_data"];
const DONUT_COLOR: Record<Overall, string> = {
  direct_bet: "var(--status-good)",
  conditional: "var(--status-warning)",
  no_bet: "var(--text-muted)",
  insufficient_data: "var(--status-serious)",
};
const DONUT_LABEL: Record<Overall, string> = {
  direct_bet: "Direct Bet",
  conditional: "Conditional",
  no_bet: "No Edge",
  insufficient_data: "No Data",
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
      <section className="rounded-lg border border-border p-4">
        <h2 className="text-xs font-semibold uppercase tracking-wide text-muted">Edge Distribution</h2>
        {total === 0 ? (
          <p className="mt-3 text-sm text-ink-secondary">No matches loaded yet.</p>
        ) : (
          <div className="mt-3 flex items-center gap-4">
            <svg viewBox="0 0 100 100" width={88} height={88} className="shrink-0 -rotate-90">
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

      <section className="rounded-lg border border-border p-4">
        <h2 className="text-xs font-semibold uppercase tracking-wide text-muted">Top Edges</h2>
        {topEdges.length === 0 ? (
          <p className="mt-3 text-sm text-ink-secondary">No priced edges yet.</p>
        ) : (
          <ul className="mt-3 flex flex-col gap-2.5">
            {topEdges.map(({ match, edge }) => (
              <li key={match.id}>
                <Link href={matchHref(match)} className="flex items-center justify-between gap-2 text-sm text-ink-secondary hover:text-ink">
                  <span className="truncate">
                    {match.home} v {match.away}
                  </span>
                  <span className="shrink-0 font-mono text-good">{formatEdge(edge)}</span>
                </Link>
              </li>
            ))}
          </ul>
        )}
      </section>
    </aside>
  );
}
