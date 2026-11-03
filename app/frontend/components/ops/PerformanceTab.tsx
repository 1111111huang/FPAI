"use client";

import { useEffect, useState } from "react";

import { getOpsPerformanceTrend } from "@/lib/api";
import type { OpsPerformanceTrendResponse, PerformanceTrendWeek, PickScope } from "@/lib/types";
import { ErrorState } from "../MatchUI";

type Metric = "hit_rate" | "roi" | "bets_placed" | "total_profit";

const METRIC_LABELS: { key: Metric; label: string; percent: boolean }[] = [
  { key: "hit_rate", label: "Hit Rate", percent: true },
  { key: "roi", label: "ROI", percent: true },
  { key: "bets_placed", label: "Bet Frequency", percent: false },
  { key: "total_profit", label: "Total Profit", percent: false },
];

// Same 0/25/50/75/100%-of-domain tick set OddsAvailabilityTab uses for its
// axis -- just applied to a domain that's [0, 1] for percent metrics and
// [domainMin, domainMax] (see niceCeil below) for count/dollar ones, instead
// of always being a literal percentage.
const AXIS_FRACTIONS = [0, 0.25, 0.5, 0.75, 1];

function formatMetric(metric: Metric, value: number): string {
  if (metric === "hit_rate" || metric === "roi") return `${Math.round(value * 100)}%`;
  if (metric === "bets_placed") return String(value);
  return value.toFixed(1);
}

function formatDelta(metric: Metric, isPercent: boolean, diff: number): string {
  const sign = diff >= 0 ? "+" : "";
  if (isPercent) return `${sign}${Math.round(diff * 100)}pp`;
  if (metric === "bets_placed") return `${sign}${diff}`;
  return `${sign}${diff.toFixed(1)}`;
}

// Rounds up to a "nice" 1/2/5-times-a-power-of-ten number so the axis never
// ends on an ugly value like 0.71 or 37 -- the few lines a standard chart
// library's default tick algorithm actually takes.
function niceCeil(value: number): number {
  if (value <= 0) return 1;
  const exp = Math.floor(Math.log10(value));
  const base = 10 ** exp;
  const frac = value / base;
  const niceFrac = frac <= 1 ? 1 : frac <= 2 ? 2 : frac <= 5 ? 5 : 10;
  return niceFrac * base;
}

const BAR_SLOT_W = 44;
const AXIS_W = 42;
const CHART_H = 140;
const TOP_PAD = 4;

// The real bar chart: inline SVG with gridlines + numeric y-axis tick labels
// (scaled 0-100%-ish for percent metrics, auto-scaled off the data's own max
// for count/dollar metrics via niceCeil), one <rect> per week. Marked
// aria-hidden -- same split FeatureHealthTab's sparkline and
// OddsAvailabilityTab's tick row use: the chart is the visual, the real
// accessible numbers are the plain-text readout row rendered under it.
function Chart({ weeks, metric, isPercent }: { weeks: PerformanceTrendWeek[]; metric: Metric; isPercent: boolean }) {
  const values = weeks.map((w) => w[metric]);
  const rawMax = Math.max(...values, 0);
  const rawMin = Math.min(...values, 0);
  // total_profit (and, in theory, roi) can go negative in a losing week, so
  // the domain's floor isn't always 0: when the worst value is negative,
  // domainMin is that same niceCeil rounding applied to its magnitude, then
  // flipped negative (e.g. a -18 week floors to -20, not -18, same as the
  // positive side rounds 18 up to 20). hit_rate/roi/bets_placed never go
  // negative in practice, so domainMin is 0 for them.
  // toY maps a domain value to its pixel y-coordinate (0 at the plot's top,
  // CHART_H at its bottom); zeroY is specifically where value=0 lands, i.e.
  // the chart's baseline -- bars for a positive value rise upward from it,
  // bars for a negative value hang downward from it.
  const domainMax = isPercent ? Math.max(1, niceCeil(rawMax)) : niceCeil(Math.max(rawMax, 1));
  const domainMin = rawMin < 0 ? -niceCeil(-rawMin) : 0;
  const span = domainMax - domainMin || 1;
  const toY = (v: number) => TOP_PAD + CHART_H - ((v - domainMin) / span) * CHART_H;
  const zeroY = toY(0);
  const svgWidth = AXIS_W + Math.max(weeks.length, 1) * BAR_SLOT_W;
  const svgHeight = TOP_PAD * 2 + CHART_H;

  return (
    <div className="overflow-x-auto rounded-lg border border-border bg-surface p-2">
      <svg aria-hidden="true" width={svgWidth} height={svgHeight} viewBox={`0 0 ${svgWidth} ${svgHeight}`}>
        {AXIS_FRACTIONS.map((f) => {
          const tickValue = domainMin + f * span;
          const y = toY(tickValue);
          return (
            <g key={f}>
              <line x1={AXIS_W} y1={y} x2={svgWidth} y2={y} className="stroke-border" strokeWidth={1} />
              {/* Bare numbers, no "%" suffix -- the metric toggle above already
                  establishes the unit, and a bare tick never collides with the
                  per-bar readout's own "50%"-style text below (both would
                  otherwise render the identical string whenever a data point
                  lands exactly on a tick, making it ambiguous which node a
                  findByText query matched). */}
              <text x={AXIS_W - 6} y={y} textAnchor="end" dominantBaseline="middle" className="fill-muted text-[9px]">
                {isPercent ? Math.round(tickValue * 100) : tickValue.toFixed(tickValue % 1 === 0 ? 0 : 1)}
              </text>
            </g>
          );
        })}
        {/* All bars share one neutral fill -- fill-good/fill-serious are
            reserved codebase-wide for a value's actual sign (profit vs.
            loss, per AgentPerformanceDashboard.tsx/BetTracker.tsx/MatchUI.tsx),
            so coloring only the most-recent bar green would misreport a
            losing latest week as "good" just because of its position. */}
        {weeks.map((w, i) => {
          const value = w[metric];
          const x = AXIS_W + i * BAR_SLOT_W + 6;
          const barW = BAR_SLOT_W - 12;
          const vY = toY(value);
          const y = Math.min(vY, zeroY);
          const h = Math.max(Math.abs(vY - zeroY), 1);
          return <rect key={w.week_start} x={x} y={y} width={barW} height={h} className="fill-accent" />;
        })}
      </svg>
      <div className="flex" style={{ minWidth: svgWidth }}>
        <div style={{ width: AXIS_W }} />
        {weeks.map((w, i) => (
          <div key={w.week_start} className="flex flex-col items-center gap-0.5" style={{ width: BAR_SLOT_W }}>
            {/* Bold marks "most recent," same neutral text-ink color as
                every other week -- not tied to whether the value is good or
                bad (see the fill-accent comment on the bars above). */}
            <span className={`font-mono text-[10px] text-ink ${i === weeks.length - 1 ? "font-semibold" : ""}`}>
              {formatMetric(metric, w[metric])}
            </span>
            <span className="text-[9px] text-muted">{w.week_start.slice(5)}</span>
          </div>
        ))}
      </div>
    </div>
  );
}

export function PerformanceTab({ weeks, league }: { weeks: number; league?: string }) {
  const [data, setData] = useState<OpsPerformanceTrendResponse | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [retryTick, setRetryTick] = useState(0);
  const [metric, setMetric] = useState<Metric>("hit_rate");
  const [pickScope, setPickScope] = useState<PickScope>("one_per_match");
  const [includeConditional, setIncludeConditional] = useState(false);

  useEffect(() => {
    let cancelled = false;
    setData(null);
    setError(null);
    getOpsPerformanceTrend({ weeks, league, pickScope, includeConditional })
      .then((result) => {
        if (!cancelled) setData(result);
      })
      .catch((err) => {
        if (!cancelled) setError(err instanceof Error ? err.message : "Failed to load performance trend");
      });
    return () => {
      cancelled = true;
    };
    // metric is a client-side display choice over the same fetched
    // response (every metric is already in each week's row) -- it doesn't
    // need its own round trip, so it's deliberately not a dependency here.
  }, [weeks, league, pickScope, includeConditional, retryTick]);

  const activeMetric = METRIC_LABELS.find((m) => m.key === metric)!;
  const last = data?.weeks[data.weeks.length - 1];
  const prev = data?.weeks[data.weeks.length - 2];
  const delta = last && prev ? last[metric] - prev[metric] : null;

  // The toggle row stays mounted through loading/error -- a strategy change
  // is what *causes* the next loading state (it's a dependency of the fetch
  // above), so hiding these controls while that fetch is in flight would
  // make it impossible to fire a second toggle click before the first
  // request resolves, and would otherwise flash the whole tab blank on
  // every single toggle click.
  return (
    <div className="flex flex-col gap-4">
      <div className="flex flex-wrap gap-2">
        {METRIC_LABELS.map((m) => (
          <button
            key={m.key}
            type="button"
            aria-pressed={metric === m.key}
            onClick={() => setMetric(m.key)}
            className={`rounded-full px-3 py-1 text-xs font-semibold ${
              metric === m.key ? "bg-accent text-white" : "border border-border text-muted hover:text-ink"
            }`}
          >
            {m.label}
          </button>
        ))}
      </div>

      <div className="flex flex-wrap items-center gap-2 text-xs">
        <span className="text-muted">Strategy</span>
        <div className="flex gap-1">
          <button
            type="button"
            aria-pressed={pickScope === "one_per_match"}
            onClick={() => setPickScope("one_per_match")}
            className={`rounded-full px-3 py-1 font-semibold ${
              pickScope === "one_per_match" ? "bg-accent text-white" : "border border-border text-muted hover:text-ink"
            }`}
          >
            One pick per match
          </button>
          <button
            type="button"
            aria-pressed={pickScope === "all_direct"}
            onClick={() => setPickScope("all_direct")}
            className={`rounded-full px-3 py-1 font-semibold ${
              pickScope === "all_direct" ? "bg-accent text-white" : "border border-border text-muted hover:text-ink"
            }`}
          >
            All direct bets
          </button>
        </div>
        <button
          type="button"
          aria-pressed={includeConditional}
          onClick={() => setIncludeConditional((v) => !v)}
          className={`rounded-full px-3 py-1 font-semibold ${
            includeConditional ? "bg-accent text-white" : "border border-border text-muted hover:text-ink"
          }`}
        >
          Include conditional
        </button>
      </div>

      {error ? (
        <ErrorState message={error} onRetry={() => setRetryTick((t) => t + 1)} />
      ) : !data ? (
        <div className="text-sm text-muted">Loading…</div>
      ) : data.weeks.length === 0 ? (
        <div className="rounded-lg border border-border bg-surface p-4 text-sm text-muted">
          No performance data in this window.
        </div>
      ) : (
        <Chart weeks={data.weeks} metric={metric} isPercent={activeMetric.percent} />
      )}

      {delta !== null && (
        <span
          className={`inline-flex w-fit items-center gap-1 rounded px-2 py-1 text-xs font-bold ${
            delta >= 0 ? "bg-good-dim text-good" : "bg-serious-dim text-serious"
          }`}
        >
          {formatDelta(metric, activeMetric.percent, delta)} vs previous week
        </span>
      )}
    </div>
  );
}
