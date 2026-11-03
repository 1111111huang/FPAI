"use client";

import { useEffect, useState } from "react";

import { getOpsOddsAvailabilityTrend } from "@/lib/api";
import type { OpsOddsAvailabilityTrendResponse } from "@/lib/types";
import { ErrorState } from "../MatchUI";

type RankedMarket = { market: string; missing: number; total: number; pct: number };

// Same high-is-bad threshold scale as FeatureHealthTab's riskRateColor.
function missingPctColor(pct: number): string {
  if (pct >= 0.5) return "text-serious";
  if (pct >= 0.2) return "text-warning";
  return "text-good";
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
    <div className="overflow-x-auto">
      <table className="w-full text-xs">
        <thead>
          <tr className="text-left text-muted">
            <th className="pb-2 pr-3">Market</th>
            <th className="pb-2 text-right">% missing (window)</th>
          </tr>
        </thead>
        <tbody>
          {ranked.map((r) => (
            <tr key={r.market} className="border-t border-border">
              <td className="py-2 pr-3 text-ink">{r.market}</td>
              <td className={`py-2 text-right font-mono ${missingPctColor(r.pct)}`}>{Math.round(r.pct * 100)}%</td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}
