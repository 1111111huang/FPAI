"use client";

import { useEffect, useState } from "react";

import { getOpsOddsAvailabilityTrend } from "@/lib/api";
import type { OpsOddsAvailabilityTrendResponse } from "@/lib/types";
import { ErrorState } from "../MatchUI";

type RankedMarket = { market: string; missing: number; total: number; pct: number };

const AXIS_TICKS = [0, 0.25, 0.5, 0.75, 1];

// Same high-is-bad threshold scale as FeatureHealthTab's riskRateColor, as a
// bar fill instead of text color.
function missingPctFill(pct: number): string {
  if (pct >= 0.5) return "bg-serious";
  if (pct >= 0.2) return "bg-warning";
  return "bg-muted";
}

// One overall % per market across the whole window (not per-week): sum each
// market's `missing` count and the window's total fixture count across all
// weeks, then divide once.
function aggregateByMarket(data: OpsOddsAvailabilityTrendResponse): RankedMarket[] {
  const missingByMarket: Record<string, number> = {};
  let totalFixtures = 0;
  for (const week of data.weeks) {
    totalFixtures += week.total_fixtures;
    for (const [market, stats] of Object.entries(week.by_market)) {
      missingByMarket[market] = (missingByMarket[market] ?? 0) + stats.missing;
    }
  }
  return Object.entries(missingByMarket)
    .map(([market, missing]) => ({
      market,
      missing,
      total: totalFixtures,
      pct: totalFixtures > 0 ? missing / totalFixtures : 0,
    }))
    .sort((a, b) => b.pct - a.pct);
}

export function OddsAvailabilityTab({ weeks, league }: { weeks: number; league?: string }) {
  const [data, setData] = useState<OpsOddsAvailabilityTrendResponse | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [retryTick, setRetryTick] = useState(0);

  useEffect(() => {
    let cancelled = false;
    setData(null);
    setError(null);
    getOpsOddsAvailabilityTrend({ weeks, league })
      .then((result) => {
        if (!cancelled) setData(result);
      })
      .catch((err) => {
        if (!cancelled) setError(err instanceof Error ? err.message : "Failed to load odds availability trend");
      });
    return () => {
      cancelled = true;
    };
  }, [weeks, league, retryTick]);

  if (error) {
    return <ErrorState message={error} onRetry={() => setRetryTick((t) => t + 1)} />;
  }
  if (!data) {
    return <div className="text-sm text-muted">Loading…</div>;
  }

  const ranked = aggregateByMarket(data);
  if (ranked.length === 0) {
    return (
      <div className="rounded-lg border border-border bg-surface p-4 text-sm text-muted">
        No missing markets in this window.
      </div>
    );
  }

  return (
    <div className="flex flex-col gap-3 rounded-lg border border-border bg-surface p-4">
      {/* Shared axis: same w-40 label / flex-1 bar / w-12 readout columns as
          each row below, so its ticks line up over the bar track only. */}
      <div className="flex items-center gap-3 text-[10px] text-muted">
        <div className="w-40 shrink-0" />
        <div className="flex flex-1 justify-between">
          {AXIS_TICKS.map((t) => (
            <span key={t}>{Math.round(t * 100)}%</span>
          ))}
        </div>
        <div className="w-12 shrink-0" />
      </div>
      {ranked.map((r) => (
        <div key={r.market} className="flex items-center gap-3">
          <div className="w-40 shrink-0 truncate text-xs text-ink">{r.market}</div>
          <div className="h-4 flex-1 rounded bg-inset">
            <div className={`h-full rounded ${missingPctFill(r.pct)}`} style={{ width: `${r.pct * 100}%` }} />
          </div>
          <div className="w-12 shrink-0 text-right text-xs font-mono text-ink">{Math.round(r.pct * 100)}%</div>
        </div>
      ))}
    </div>
  );
}
