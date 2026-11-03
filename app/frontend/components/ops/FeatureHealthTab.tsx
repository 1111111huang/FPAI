"use client";

import { useEffect, useMemo, useState } from "react";

import { getOpsFeatureHealthTrend } from "@/lib/api";
import type { FlaggedFeature, OpsFeatureHealthTrendResponse } from "@/lib/types";
import { ErrorState } from "../MatchUI";

const INITIAL_VISIBLE = 8;
const SPARK_W = 100;
const SPARK_H = 24;

// Per-feature trend sparkline from the `weekly` series already in the
// response -- labeled with its min/max so the shape has a real scale
// reference (design spec section 6: "every chart gets a real, labeled
// axis"; min/max is the minimum that satisfies that for a row this small --
// doesn't need OddsAvailabilityTab's full tick row). Decorative squiggle is
// aria-hidden; the min/max numbers are real visible text, same split as
// OddsAvailabilityTab's tick row vs its bars.
function Sparkline({ weekly }: { weekly: FlaggedFeature["weekly"] }) {
  const points = weekly.filter((w): w is { week_start: string; mean: number; null_rate: number } => w.mean !== null);
  if (points.length < 2) return null;

  const values = points.map((p) => p.mean);
  const min = Math.min(...values);
  const max = Math.max(...values);
  const range = max - min || 1;
  const coords = points
    .map((p, i) => {
      const x = (i / (points.length - 1)) * SPARK_W;
      const y = SPARK_H - ((p.mean - min) / range) * SPARK_H;
      return `${x.toFixed(1)},${y.toFixed(1)}`;
    })
    .join(" ");

  return (
    <div className="flex items-center gap-1.5">
      <span className="font-mono text-[9px] text-muted">{min.toFixed(1)}</span>
      <svg aria-hidden="true" viewBox={`0 0 ${SPARK_W} ${SPARK_H}`} width={SPARK_W} height={SPARK_H} className="shrink-0 text-accent">
        <polyline points={coords} fill="none" stroke="currentColor" strokeWidth={1.5} />
      </svg>
      <span className="font-mono text-[9px] text-muted">{max.toFixed(1)}</span>
    </div>
  );
}

// Fixed-width sparkline slot (matches the SVG's own width + its two
// flanking labels) so a suppressed sparkline (fewer than 2 valid weekly
// points) still reserves its column -- otherwise `justify-between` would
// redistribute the row's slack across one fewer gap and shift the badge
// column relative to rows that do render one. Same fixed-column fix
// OddsAvailabilityTab uses for its bar/label columns.
const SPARK_SLOT_W = 124;

function FeatureRow({ feature }: { feature: FlaggedFeature }) {
  return (
    <div className="flex items-center justify-between gap-3 rounded border border-border bg-surface-raised px-3 py-2">
      <div className="min-w-0">
        <span className="font-mono text-xs font-semibold text-ink">{feature.name}</span>
        <span className="ml-2 text-[10px] uppercase text-muted">{feature.category}</span>
      </div>
      <div className="flex shrink-0 items-center gap-2 text-[10px]">
        {feature.flags.includes("drift") && (
          <span className="rounded-full bg-serious-dim px-2 py-0.5 font-semibold text-serious">DRIFT</span>
        )}
        {feature.flags.includes("nullrate") && (
          <span className="rounded-full bg-warning-dim px-2 py-0.5 font-semibold text-warning">NULL RATE ↑</span>
        )}
      </div>
      <div className="shrink-0" style={{ width: SPARK_SLOT_W }}>
        <Sparkline weekly={feature.weekly} />
      </div>
      <div className="shrink-0 text-right text-xs text-muted">
        {feature.flags.includes("drift") && feature.baseline_mean !== null && feature.recent_mean !== null
          ? `${feature.baseline_mean.toFixed(2)} → ${feature.recent_mean.toFixed(2)}`
          : `${(feature.baseline_null_rate * 100).toFixed(0)}% → ${(feature.recent_null_rate * 100).toFixed(0)}% null`}
      </div>
    </div>
  );
}

export function FeatureHealthTab({ weeks, league }: { weeks: number; league?: string }) {
  const [data, setData] = useState<OpsFeatureHealthTrendResponse | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [retryTick, setRetryTick] = useState(0);
  const [search, setSearch] = useState("");
  const [visibleCount, setVisibleCount] = useState(INITIAL_VISIBLE);

  useEffect(() => {
    let cancelled = false;
    setData(null);
    setError(null);
    setVisibleCount(INITIAL_VISIBLE);
    getOpsFeatureHealthTrend({ weeks, league })
      .then((result) => {
        if (!cancelled) setData(result);
      })
      .catch((err) => {
        if (!cancelled) setError(err instanceof Error ? err.message : "Failed to load feature health trend");
      });
    return () => {
      cancelled = true;
    };
  }, [weeks, league, retryTick]);

  // Severity-sorted: worst magnitude-ratio first (design spec section 6).
  const sorted = useMemo(() => {
    if (!data) return [];
    return [...data.flagged_features].sort((a, b) => b.magnitude - a.magnitude);
  }, [data]);

  const filtered = useMemo(() => {
    const q = search.trim().toLowerCase();
    return q ? sorted.filter((f) => f.name.toLowerCase().includes(q)) : sorted;
  }, [sorted, search]);

  if (error) {
    return <ErrorState message={error} onRetry={() => setRetryTick((t) => t + 1)} />;
  }
  if (!data) {
    return <div className="text-sm text-muted">Loading…</div>;
  }

  const drifting = data.flagged_features.filter((f) => f.flags.includes("drift")).length;
  const nullRate = data.flagged_features.filter((f) => f.flags.includes("nullrate")).length;
  const both = data.flagged_features.filter((f) => f.flags.length > 1).length;

  return (
    <div className="flex flex-col gap-4">
      <div className="grid grid-cols-4 divide-x divide-border rounded-lg border border-border bg-surface text-center">
        <div className="p-3">
          <div className="font-mono text-xl font-bold text-ink">{data.tracked_feature_count}</div>
          <div className="text-[10px] uppercase text-muted">Tracked</div>
        </div>
        <div className="p-3">
          <div className="font-mono text-xl font-bold text-serious">{drifting}</div>
          <div className="text-[10px] uppercase text-muted">Drifting</div>
        </div>
        <div className="p-3">
          <div className="font-mono text-xl font-bold text-warning">{nullRate}</div>
          <div className="text-[10px] uppercase text-muted">Elevated Null Rate</div>
        </div>
        <div className="p-3">
          <div className="font-mono text-xl font-bold text-ink">{both}</div>
          <div className="text-[10px] uppercase text-muted">Both</div>
        </div>
      </div>

      {data.flagged_features.length === 0 ? (
        <div className="rounded-lg border border-border bg-surface p-4 text-sm text-muted">
          No features flagged in this window.
        </div>
      ) : (
        <div className="rounded-lg border border-border bg-surface p-4">
          <input
            placeholder="Filter by name, e.g. XOC"
            aria-label="Filter flagged features by name"
            value={search}
            onChange={(e) => setSearch(e.target.value)}
            className="mb-3 w-56 rounded border border-border bg-page px-2 py-1 font-mono text-xs text-ink outline-none focus:border-accent"
          />
          <div className="flex flex-col gap-2">
            {filtered.slice(0, visibleCount).map((f) => (
              <FeatureRow key={f.name} feature={f} />
            ))}
          </div>
          {filtered.length > visibleCount && (
            <button
              type="button"
              onClick={() => setVisibleCount(filtered.length)}
              className="mt-3 w-full border-t border-dashed border-border pt-2 text-xs font-semibold text-accent"
            >
              Show {filtered.length - visibleCount} more flagged features ↓
            </button>
          )}
        </div>
      )}
    </div>
  );
}
