"use client";

import { useEffect, useState } from "react";

import { getOpsFeatureHealth } from "@/lib/api";
import type { OpsFeatureHealthResponse } from "@/lib/types";
import { ErrorState } from "../MatchUI";

function formatPct(v: number): string {
  return `${(v * 100).toFixed(0)}%`;
}

// Risk rates (cold-start, unknown-team): low is fine, high is a real problem.
// Unlike OddsAvailabilityTab's likely_cause (a flat enum), these are
// continuous -- a bare number invites treating 2% and 40% the same, so
// threshold them into the same good/warning/serious scale used elsewhere
// in this dashboard (see FailuresTab's BREAKER_COLOR).
function riskRateColor(v: number): string {
  if (v >= 0.25) return "text-serious";
  if (v >= 0.1) return "text-warning";
  return "text-good";
}

// Completeness is the inverse: high is fine, low is the problem.
function completenessColor(v: number): string {
  if (v < 0.7) return "text-serious";
  if (v < 0.9) return "text-warning";
  return "text-good";
}

export function FeatureHealthTab({
  dateFrom,
  dateTo,
  league,
}: {
  dateFrom: string;
  dateTo: string;
  league?: string;
}) {
  const [data, setData] = useState<OpsFeatureHealthResponse | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [retryTick, setRetryTick] = useState(0);

  useEffect(() => {
    let cancelled = false;
    setData(null);
    setError(null);
    getOpsFeatureHealth({ dateFrom, dateTo, league })
      .then((result) => {
        if (!cancelled) setData(result);
      })
      .catch((err) => {
        if (!cancelled) setError(err instanceof Error ? err.message : "Failed to load feature health");
      });
    return () => {
      cancelled = true;
    };
  }, [dateFrom, dateTo, league, retryTick]);

  if (error) {
    return <ErrorState message={error} onRetry={() => setRetryTick((t) => t + 1)} />;
  }
  if (!data) {
    return <div className="text-sm text-muted">Loading…</div>;
  }

  const rows = Object.entries(data.by_league);
  if (rows.length === 0) {
    return (
      <div className="rounded-lg border border-border bg-surface p-4 text-sm text-muted">
        No recommendations in this window.
      </div>
    );
  }

  return (
    <div className="overflow-x-auto">
      <table className="w-full text-xs">
        <thead>
          <tr className="text-left text-muted">
            <th className="pb-2 pr-3">League</th>
            <th className="pb-2 pr-3 text-right">Sample</th>
            <th className="pb-2 pr-3 text-right">Avg completeness</th>
            <th className="pb-2 pr-3 text-right">Cold-start rate</th>
            <th className="pb-2 text-right">Unknown-team rate</th>
          </tr>
        </thead>
        <tbody>
          {rows.map(([leagueCode, stats]) => (
            <tr key={leagueCode} className="border-t border-border">
              <td className="py-2 pr-3 font-semibold text-ink">{leagueCode}</td>
              <td className="py-2 pr-3 text-right font-mono text-ink-secondary">{stats.sample_size}</td>
              <td
                className={`py-2 pr-3 text-right font-mono ${
                  stats.avg_feature_completeness !== null ? completenessColor(stats.avg_feature_completeness) : "text-muted"
                }`}
              >
                {stats.avg_feature_completeness !== null ? formatPct(stats.avg_feature_completeness) : "—"}
              </td>
              <td className={`py-2 pr-3 text-right font-mono ${riskRateColor(stats.cold_start_risk_rate)}`}>
                {formatPct(stats.cold_start_risk_rate)}
              </td>
              <td className={`py-2 text-right font-mono ${riskRateColor(stats.unknown_team_rate)}`}>
                {formatPct(stats.unknown_team_rate)}
              </td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}
