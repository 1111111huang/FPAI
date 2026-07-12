"use client";

/** W17: small system status footer -- surfaces data staleness
 * (raw_matches last-updated) and current model selection counts per
 * context, reusing GET /api/status (itself a pure pass-through of
 * already-exposed src/tools functions). Renders nothing while loading or
 * on a fetch failure -- this is a passive footer, not worth an error state
 * of its own. */

import { useEffect, useState } from "react";

import { getStatus } from "@/lib/api";
import type { StatusResponse } from "@/lib/types";

export function StatusFooter() {
  const [status, setStatus] = useState<StatusResponse | null>(null);

  useEffect(() => {
    getStatus()
      .then(setStatus)
      .catch(() => setStatus(null));
  }, []);

  if (!status) return null;

  const { data_freshness, model_status } = status;
  const leagueCount = Object.keys(model_status.league ?? {}).length;
  const internationalCount = Object.keys(model_status.international ?? {}).length;

  return (
    <footer className="mx-auto flex max-w-4xl flex-wrap items-center gap-x-4 gap-y-1 px-4 py-3 text-[11px] text-ink-secondary sm:px-6">
      <span className={data_freshness.is_stale ? "text-warning" : undefined}>
        Data: {data_freshness.latest_match_date ?? "unknown"}
        {data_freshness.days_since_update !== null && ` (${data_freshness.days_since_update}d ago)`}
        {data_freshness.is_stale && " -- stale"}
      </span>
      <span>
        Models: league {leagueCount} · international {internationalCount}
      </span>
    </footer>
  );
}
