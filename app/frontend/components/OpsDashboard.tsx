"use client";

/** 2026-11-02: /ops -- failures & circuit breakers, odds availability,
 * feature health, and recommendation performance (the existing, previously
 * unlinked agent-performance page, folded in here instead of rebuilt).
 * Shared Window (weeks) + league filter bar: fully wired into the Odds
 * Availability and Feature Health tabs; Failures only honors the league
 * filter (it isn't date-scoped); Performance manages its own date range
 * and ignores this filter bar entirely (unmodified, linked-not-rebuilt
 * page).
 * Design: docs/superpowers/specs/2026-11-03-ops-dashboard-trends-design.md */

import { useState } from "react";

import { AppShell } from "./AppShell";
import { AgentPerformanceContent } from "./AgentPerformanceDashboard";
import { FailuresTab } from "./ops/FailuresTab";
import { OddsAvailabilityTab } from "./ops/OddsAvailabilityTab";
import { FeatureHealthTab } from "./ops/FeatureHealthTab";
import { LEAGUE_LABEL } from "@/lib/dashboardMetrics";

type TabKey = "failures" | "odds" | "features" | "performance";

const TABS: { key: TabKey; label: string }[] = [
  { key: "failures", label: "Failures" },
  { key: "odds", label: "Odds Availability" },
  { key: "features", label: "Feature Health" },
  { key: "performance", label: "Performance" },
];

const WEEK_OPTIONS = [4, 8, 12, 26];

export function OpsDashboard() {
  const [tab, setTab] = useState<TabKey>("failures");
  const [league, setLeague] = useState<string>("");
  const [weeks, setWeeks] = useState<number>(8);

  return (
    <AppShell active="ops">
      <div className="mx-auto max-w-5xl p-4">
        <h1 className="text-xl font-semibold text-ink">Ops Dashboard</h1>

        <div className="mt-4 flex flex-wrap items-center gap-3 rounded-lg border border-border bg-surface p-3 text-sm">
          <div className="flex items-center gap-2">
            <span className="text-muted">Window</span>
            <div className="flex gap-1">
              {WEEK_OPTIONS.map((w) => (
                <button
                  key={w}
                  type="button"
                  aria-pressed={weeks === w}
                  onClick={() => setWeeks(w)}
                  className={`rounded-full px-3 py-1 text-xs font-semibold ${
                    weeks === w ? "bg-accent text-white" : "border border-border text-muted hover:text-ink"
                  }`}
                >
                  {w}w
                </button>
              ))}
            </div>
          </div>
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
          {tab === "odds" && <OddsAvailabilityTab weeks={weeks} league={league || undefined} />}
          {tab === "features" && <FeatureHealthTab weeks={weeks} league={league || undefined} />}
          {tab === "performance" && <AgentPerformanceContent />}
        </div>
      </div>
    </AppShell>
  );
}
