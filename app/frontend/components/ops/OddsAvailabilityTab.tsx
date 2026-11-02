"use client";

import { useEffect, useState } from "react";

import { getOpsOddsAvailability } from "@/lib/api";
import type { OpsOddsAvailabilityResponse } from "@/lib/types";
import { ErrorState } from "../MatchUI";

const LIKELY_CAUSE_STYLE: Record<OpsOddsAvailabilityResponse["fixtures"][number]["likely_cause"], string> = {
  circuit_open: "text-serious",
  not_posted: "text-muted",
};

export function OddsAvailabilityTab({
  dateFrom,
  dateTo,
  league,
}: {
  dateFrom: string;
  dateTo: string;
  league?: string;
}) {
  const [data, setData] = useState<OpsOddsAvailabilityResponse | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [retryTick, setRetryTick] = useState(0);

  useEffect(() => {
    let cancelled = false;
    setData(null);
    setError(null);
    getOpsOddsAvailability({ dateFrom, dateTo, league })
      .then((result) => {
        if (!cancelled) setData(result);
      })
      .catch((err) => {
        if (!cancelled) setError(err instanceof Error ? err.message : "Failed to load odds availability");
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

  if (data.fixtures.length === 0) {
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
            <th className="pb-2 pr-3">Match</th>
            <th className="pb-2 pr-3">League</th>
            <th className="pb-2 pr-3">Missing markets</th>
            <th className="pb-2">Likely cause</th>
          </tr>
        </thead>
        <tbody>
          {data.fixtures.map((f) => (
            <tr key={f.match_id} className="border-t border-border">
              <td className="py-2 pr-3 text-ink">
                {f.home_team ?? "—"} v {f.away_team ?? "—"}
              </td>
              <td className="py-2 pr-3 text-ink-secondary">{f.league ?? "—"}</td>
              <td className="py-2 pr-3 text-ink">{f.missing_markets.join(", ")}</td>
              <td className={`py-2 font-mono ${LIKELY_CAUSE_STYLE[f.likely_cause]}`}>{f.likely_cause}</td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}
