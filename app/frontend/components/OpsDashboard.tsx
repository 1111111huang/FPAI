"use client";

/** 2026-11-02: /ops -- failures & circuit breakers, odds availability,
 * feature health, and recommendation performance (the existing, previously
 * unlinked agent-performance page, folded in here instead of rebuilt).
 * One shared date-range + league filter bar above all four tabs, so they
 * can't drift out of sync with each other.
 * Design: docs/superpowers/specs/2026-11-02-background-job-safety-net-design.md */

import { useState } from "react";

import { AppShell } from "./AppShell";
import { AgentPerformanceContent } from "./AgentPerformanceDashboard";
import { addDays, dateString } from "./MatchUI";
import { FailuresTab } from "./ops/FailuresTab";
import { OddsAvailabilityTab } from "./ops/OddsAvailabilityTab";
import { FeatureHealthTab } from "./ops/FeatureHealthTab";
import { LEAGUE_LABEL } from "@/lib/dashboardMetrics";
import { useSandboxAsOf } from "@/lib/useSandboxAsOf";

type TabKey = "failures" | "odds" | "features" | "performance";

const TABS: { key: TabKey; label: string }[] = [
  { key: "failures", label: "Failures" },
  { key: "odds", label: "Odds Availability" },
  { key: "features", label: "Feature Health" },
  { key: "performance", label: "Performance" },
];

export function OpsDashboard() {
  const [tab, setTab] = useState<TabKey>("failures");
  const [league, setLeague] = useState<string>("");
  // W30: date-window defaults go through the sandbox-aware clock, not a raw
  // `new Date()` (see lib/useSandboxAsOf.ts, MatchUI.tsx's dateString/
  // addDays) -- a real browser instant otherwise, or the sandbox's as_of
  // once GET /api/sandbox/status resolves. Held as an override so these
  // keep re-deriving from the latest asOf until the user actually edits
  // one of the inputs below, instead of freezing at mount time.
  const { asOf, sandboxMode } = useSandboxAsOf();
  const [dateFromOverride, setDateFromOverride] = useState<string | null>(null);
  const [dateToOverride, setDateToOverride] = useState<string | null>(null);
  const dateFrom = dateFromOverride ?? dateString(addDays(asOf, -7, sandboxMode), sandboxMode);
  const dateTo = dateToOverride ?? dateString(asOf, sandboxMode);

  return (
    <AppShell active="ops">
      <div className="mx-auto max-w-5xl p-4">
        <h1 className="text-xl font-semibold text-ink">Ops Dashboard</h1>

        <div className="mt-4 flex flex-wrap items-center gap-3 rounded-lg border border-border bg-surface p-3 text-sm">
          <label className="flex items-center gap-2">
            <span className="text-muted">From</span>
            <input
              type="date"
              value={dateFrom}
              onChange={(e) => setDateFromOverride(e.target.value)}
              className="rounded border border-border bg-bg px-2 py-1 text-ink"
            />
          </label>
          <label className="flex items-center gap-2">
            <span className="text-muted">To</span>
            <input
              type="date"
              value={dateTo}
              onChange={(e) => setDateToOverride(e.target.value)}
              className="rounded border border-border bg-bg px-2 py-1 text-ink"
            />
          </label>
          <label className="flex items-center gap-2">
            <span className="text-muted">League</span>
            <select
              value={league}
              onChange={(e) => setLeague(e.target.value)}
              className="rounded border border-border bg-bg px-2 py-1 text-ink"
            >
              <option value="">All</option>
              {Object.keys(LEAGUE_LABEL).map((code) => (
                <option key={code} value={code}>
                  {LEAGUE_LABEL[code]}
                </option>
              ))}
            </select>
          </label>
        </div>

        <div role="tablist" className="mt-4 flex gap-1 border-b border-border">
          {TABS.map((t) => (
            <button
              key={t.key}
              type="button"
              role="tab"
              aria-selected={tab === t.key}
              onClick={() => setTab(t.key)}
              className={`px-3 py-2 text-sm ${
                tab === t.key ? "border-b-2 border-ink font-medium text-ink" : "text-muted hover:text-ink"
              }`}
            >
              {t.label}
            </button>
          ))}
        </div>

        <div className="mt-4">
          {tab === "failures" && <FailuresTab league={league || undefined} />}
          {tab === "odds" && <OddsAvailabilityTab dateFrom={dateFrom} dateTo={dateTo} league={league || undefined} />}
          {tab === "features" && <FeatureHealthTab dateFrom={dateFrom} dateTo={dateTo} league={league || undefined} />}
          {tab === "performance" && <AgentPerformanceContent />}
        </div>
      </div>
    </AppShell>
  );
}
