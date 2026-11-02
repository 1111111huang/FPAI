"use client";

import { useEffect, useState } from "react";

import { getOpsFailures } from "@/lib/api";
import type { CircuitBreakerStatus, OpsFailuresResponse } from "@/lib/types";

const BREAKER_COLOR: Record<CircuitBreakerStatus["state"], string> = {
  open: "border-serious/40 bg-serious-dim text-serious",
  half_open: "border-warning/40 bg-warning-dim text-warning",
  closed: "border-good/40 bg-good-dim text-good",
};

function BreakerPill({ vendor, state }: { vendor: string; state: CircuitBreakerStatus["state"] }) {
  return (
    <span
      className={`inline-flex items-center gap-1.5 rounded-md border px-2 py-0.5 text-xs font-medium ${BREAKER_COLOR[state]}`}
    >
      {vendor}: {state}
    </span>
  );
}

export function FailuresTab({ league }: { league?: string }) {
  const [data, setData] = useState<OpsFailuresResponse | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    let cancelled = false;
    setData(null);
    setError(null);
    getOpsFailures({ league })
      .then((result) => {
        if (!cancelled) setData(result);
      })
      .catch((err) => {
        if (!cancelled) setError(err instanceof Error ? err.message : "Failed to load failures");
      });
    return () => {
      cancelled = true;
    };
  }, [league]);

  if (error) {
    return <div className="rounded-lg border border-serious/40 p-3.5 text-sm text-serious">{error}</div>;
  }
  if (!data) {
    return <div className="text-sm text-muted">Loading...</div>;
  }

  return (
    <div className="flex flex-col gap-4">
      <div className="flex flex-wrap gap-2">
        <BreakerPill vendor="the_odds_api" state={data.circuit_breakers.the_odds_api.state} />
        <BreakerPill vendor="oddspapi" state={data.circuit_breakers.oddspapi.state} />
      </div>

      {data.failures.length === 0 ? (
        <div className="rounded-lg border border-border bg-surface p-4 text-sm text-muted">
          No failures in this window.
        </div>
      ) : (
        <div className="overflow-x-auto">
          <table className="w-full text-xs">
            <thead>
              <tr className="text-left text-muted">
                <th className="pb-2 pr-3">Time</th>
                <th className="pb-2 pr-3">Subsystem</th>
                <th className="pb-2 pr-3">Unit</th>
                <th className="pb-2 pr-3">League</th>
                <th className="pb-2">Error</th>
              </tr>
            </thead>
            <tbody>
              {data.failures.map((f) => (
                <tr key={f.id} className="border-t border-border">
                  <td className="py-2 pr-3 font-mono text-ink-secondary">{f.ts}</td>
                  <td className="py-2 pr-3 text-ink">{f.subsystem}</td>
                  <td className="py-2 pr-3 font-mono text-ink-secondary">{f.unit_id}</td>
                  <td className="py-2 pr-3 text-ink">{f.league ?? "—"}</td>
                  <td className="py-2 font-mono text-serious">
                    {f.exc_type}: {f.message}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}
    </div>
  );
}
