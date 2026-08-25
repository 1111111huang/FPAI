"use client";

/** W174: local-only agent performance dashboard -- not linked from
 * AppShell's nav (see app/agent-performance/page.tsx), reachable only by
 * direct URL. Same "unlinked, not removed" precedent as BetTracker's own
 * /bets route (W106/W115). */

import { useEffect, useState } from "react";

import { ApiError, getAgentPerformanceDashboard } from "@/lib/api";
import { LEAGUE_LABEL } from "@/lib/dashboardMetrics";
import type { AgentPerformanceDashboard, SegmentMetrics, TopBet } from "@/lib/types";
import { AppShell } from "./AppShell";
import { ErrorState, marketLabel } from "./MatchUI";

function formatPct(v: number): string {
  return `${(v * 100).toFixed(1)}%`;
}
function formatUB(v: number): string {
  return `${v.toFixed(1)} UB`;
}
function pnlColor(v: number): string {
  return v > 0 ? "text-good" : v < 0 ? "text-serious" : "text-ink";
}
function formatSelection(selection: string): string {
  return selection.replace(/_/g, " ").replace(/\b\w/g, (c) => c.toUpperCase());
}

function StatTile({ label, value, colorClass }: { label: string; value: string; colorClass?: string }) {
  return (
    <div className="min-w-[140px] flex-1 rounded-lg border border-border bg-surface p-4">
      <div className="text-[10px] uppercase tracking-wide text-muted">{label}</div>
      <div className={`mt-1 font-mono text-2xl font-bold ${colorClass ?? "text-ink"}`}>{value}</div>
    </div>
  );
}

function BreakdownTable({ title, rows }: { title: string; rows: { label: string; metrics: SegmentMetrics }[] }) {
  return (
    <div className="min-w-[260px] flex-1 rounded-lg border border-border bg-surface p-4">
      <div className="mb-2 text-xs font-semibold text-ink">{title}</div>
      <table className="w-full text-xs">
        <thead>
          <tr className="text-muted">
            <th className="pb-1.5 text-left text-[10px] font-normal uppercase">Segment</th>
            {/* "ROI%" not "ROI" -- the Main Metrics StatTile above already
                owns the bare "ROI" text; three of these tables render at
                once, so a matching header collides with findByText("ROI"). */}
            <th className="pb-1.5 text-right text-[10px] font-normal uppercase">ROI%</th>
            <th className="pb-1.5 text-right text-[10px] font-normal uppercase">Stake</th>
            <th className="pb-1.5 text-right text-[10px] font-normal uppercase">Won</th>
            <th className="pb-1.5 text-right text-[10px] font-normal uppercase">Bets</th>
            <th className="pb-1.5 text-right text-[10px] font-normal uppercase">Hit%</th>
          </tr>
        </thead>
        <tbody>
          {rows.map((r) => (
            <tr key={r.label} className="border-t border-hairline">
              <td className="py-1.5 text-ink">{r.label}</td>
              <td className={`py-1.5 text-right font-mono ${pnlColor(r.metrics.roi)}`}>{formatPct(r.metrics.roi)}</td>
              <td className="py-1.5 text-right font-mono text-ink-secondary">{formatUB(r.metrics.total_staked)}</td>
              <td className={`py-1.5 text-right font-mono ${pnlColor(r.metrics.total_profit)}`}>
                {r.metrics.total_profit >= 0 ? "+" : ""}
                {r.metrics.total_profit.toFixed(1)}
              </td>
              <td className="py-1.5 text-right font-mono text-ink-secondary">{r.metrics.bets_placed}</td>
              <td className="py-1.5 text-right font-mono text-ink-secondary">{formatPct(r.metrics.hit_rate)}</td>
            </tr>
          ))}
          {rows.length === 0 && (
            <tr>
              <td colSpan={6} className="py-3 text-center text-ink-secondary">
                No staked bets in this window.
              </td>
            </tr>
          )}
        </tbody>
      </table>
    </div>
  );
}

function bucketize(values: number[], edges: number[], labels: string[]): { label: string; count: number }[] {
  const counts = new Array(labels.length).fill(0);
  for (const v of values) {
    let idx = edges.length - 2;
    for (let i = 0; i < edges.length - 1; i++) {
      if (v >= edges[i] && v < edges[i + 1]) {
        idx = i;
        break;
      }
    }
    counts[idx] += 1;
  }
  return labels.map((label, i) => ({ label, count: counts[i] }));
}

const ODDS_EDGES = [1.0, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0, Infinity];
const ODDS_LABELS = ["1.0", "1.5", "2.0", "2.5", "3.0", "4.0", "5.0+"];
const STAKE_EDGES = [0, 2, 4, 6, 8, 10, Infinity];
const STAKE_LABELS = ["0", "2", "4", "6", "8", "10+"];

function Histogram({ title, buckets }: { title: string; buckets: { label: string; count: number }[] }) {
  const max = Math.max(1, ...buckets.map((b) => b.count));
  return (
    <div className="min-w-[200px] flex-1 rounded-lg border border-border bg-surface p-4">
      <div className="mb-2 text-xs font-semibold text-ink">{title}</div>
      <div className="flex items-end gap-1" style={{ height: 80 }}>
        {buckets.map((b) => (
          <div key={b.label} className="flex flex-1 flex-col items-center justify-end" title={`${b.label}: ${b.count}`}>
            <div
              className="w-full rounded-t bg-accent"
              style={{ height: `${(b.count / max) * 100}%`, minHeight: b.count > 0 ? 2 : 0 }}
            />
          </div>
        ))}
      </div>
      <div className="mt-1 flex justify-between text-[10px] text-muted">
        {buckets.map((b) => (
          <span key={b.label}>{b.label}</span>
        ))}
      </div>
    </div>
  );
}

// Dataviz skill's reference palette, dark-mode categorical steps -- slot 1
// (accent) already used by the rest of this app; slots 2-8 introduced here
// for the first time, same fixed order the palette validates.
const CATEGORICAL_COLORS = ["#3987e5", "#d95926", "#199e70", "#c98500", "#d55181", "#9085e9", "#e66767", "#008300"];

// Color follows the entity, never its rank (dataviz skill) -- `entries` is
// sorted by count for display, so indexing CATEGORICAL_COLORS by post-sort
// position would reassign each league's color whenever the time-range
// filter reorders them. Index by LEAGUE_LABEL's fixed insertion order instead.
function colorForLeague(league: string): string {
  const idx = Object.keys(LEAGUE_LABEL).indexOf(league);
  return CATEGORICAL_COLORS[(idx >= 0 ? idx : 0) % CATEGORICAL_COLORS.length];
}

function LeagueBarChart({ entries }: { entries: { league: string; count: number }[] }) {
  const max = Math.max(1, ...entries.map((e) => e.count));
  return (
    <div className="min-w-[200px] flex-1 rounded-lg border border-border bg-surface p-4">
      <div className="mb-2 text-xs font-semibold text-ink">Bets by League</div>
      <div className="flex flex-col gap-2">
        {entries.map((e) => (
          <div key={e.league} className="flex items-center gap-2 text-xs">
            <span className="w-10 shrink-0 truncate text-ink-secondary">{LEAGUE_LABEL[e.league] ?? e.league}</span>
            <div className="h-3.5 flex-1 rounded bg-white/[0.04]">
              <div
                className="h-full rounded"
                style={{ width: `${(e.count / max) * 100}%`, background: colorForLeague(e.league) }}
              />
            </div>
            <span className="w-6 shrink-0 text-right text-muted">{e.count}</span>
          </div>
        ))}
      </div>
    </div>
  );
}

function BetsExampleTable({ title, bets, colorClass }: { title: string; bets: TopBet[]; colorClass: string }) {
  return (
    <div className="min-w-[280px] flex-1 rounded-lg border border-border bg-surface p-4">
      <div className={`mb-2 text-xs font-semibold ${colorClass}`}>{title}</div>
      <table className="w-full text-xs">
        <thead>
          <tr className="text-muted">
            <th className="pb-1.5 text-left text-[10px] font-normal uppercase">Match</th>
            <th className="pb-1.5 text-left text-[10px] font-normal uppercase">Pick</th>
            <th className="pb-1.5 text-right text-[10px] font-normal uppercase">Odds</th>
            <th className="pb-1.5 text-right text-[10px] font-normal uppercase">Stake</th>
            <th className="pb-1.5 text-right text-[10px] font-normal uppercase">Payout</th>
          </tr>
        </thead>
        <tbody>
          {bets.map((b, i) => (
            <tr key={i} className="border-t border-hairline">
              <td className="py-1.5 text-ink">{b.home_team && b.away_team ? `${b.home_team} v ${b.away_team}` : b.match_id}</td>
              <td className="py-1.5 text-ink-secondary">
                {marketLabel(b.market).label} · {formatSelection(b.selection)}
              </td>
              <td className="py-1.5 text-right font-mono text-ink-secondary">{b.odds.toFixed(2)}</td>
              <td className="py-1.5 text-right font-mono text-ink-secondary">{formatUB(b.stake)}</td>
              <td className={`py-1.5 text-right font-mono ${pnlColor(b.payout)}`}>
                {b.payout >= 0 ? "+" : ""}
                {b.payout.toFixed(1)}
              </td>
            </tr>
          ))}
          {bets.length === 0 && (
            <tr>
              <td colSpan={5} className="py-3 text-center text-ink-secondary">
                None yet.
              </td>
            </tr>
          )}
        </tbody>
      </table>
    </div>
  );
}

const DAY_OPTIONS: { label: string; days: number }[] = [
  { label: "All time", days: 3650 },
  { label: "Last 90 days", days: 90 },
  { label: "Last 30 days", days: 30 },
];

export function AgentPerformancePage() {
  const [days, setDays] = useState(3650);
  const [data, setData] = useState<AgentPerformanceDashboard | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [retryTick, setRetryTick] = useState(0);

  useEffect(() => {
    let cancelled = false;
    setData(null);
    setError(null);
    getAgentPerformanceDashboard(days)
      .then((d) => {
        if (!cancelled) setData(d);
      })
      .catch((err) => {
        if (!cancelled) setError(err instanceof ApiError ? err.message : "Could not load performance data.");
      });
    return () => {
      cancelled = true;
    };
  }, [days, retryTick]);

  const kelly = data?.kelly_roi_simulation;

  return (
    <AppShell active="agent-performance">
      <div className="pt-8">
        <h1 className="text-xl font-semibold tracking-tight text-ink">Agent Performance</h1>
        <p className="mt-1 text-sm text-ink-secondary">
          All resolved live recommendations, Kelly-sized simulated stakes (UB)
        </p>

        <div className="mt-4 flex gap-2">
          {DAY_OPTIONS.map((opt) => (
            <button
              key={opt.label}
              type="button"
              onClick={() => setDays(opt.days)}
              className={`rounded-full border px-3 py-1 text-xs ${
                days === opt.days ? "border-accent bg-accent text-white" : "border-border-strong text-ink-secondary"
              }`}
            >
              {opt.label}
            </button>
          ))}
        </div>

        {error && (
          <div className="mt-6">
            <ErrorState message={error} onRetry={() => setRetryTick((t) => t + 1)} />
          </div>
        )}

        {!error && !data && <div className="mt-6 text-sm text-ink-secondary">Loading…</div>}

        {data && kelly && (
          <>
            <div className="mt-6">
              <div className="mb-2 text-sm font-semibold text-ink">Main Metrics</div>
              <div className="flex flex-wrap gap-3">
                <StatTile label="ROI" value={formatPct(kelly.roi)} colorClass={pnlColor(kelly.roi)} />
                <StatTile label="Total Stake" value={formatUB(kelly.total_staked)} />
                <StatTile
                  label="Money Won"
                  value={`${kelly.total_profit >= 0 ? "+" : ""}${kelly.total_profit.toFixed(1)} UB`}
                  colorClass={pnlColor(kelly.total_profit)}
                />
                <StatTile label="Bets Placed" value={String(kelly.bets_placed)} />
                <StatTile label="Hit %" value={formatPct(kelly.hit_rate)} />
              </div>
            </div>

            <div className="mt-8">
              <div className="mb-1 text-sm font-semibold text-ink">Segmentation</div>
              <p className="mb-3 text-xs text-ink-secondary">
                Same metrics, sliced by Market / Market+Direction / League
              </p>
              <div className="flex flex-wrap gap-4">
                <BreakdownTable
                  title="By Market"
                  rows={Object.entries(data.by_market_metrics).map(([k, v]) => ({ label: marketLabel(k).label, metrics: v }))}
                />
                <BreakdownTable
                  title="By Market + Direction"
                  rows={Object.entries(data.by_market_selection_metrics).map(([k, v]) => {
                    const [market, selection] = k.split(":");
                    return { label: `${marketLabel(market).label} · ${formatSelection(selection)}`, metrics: v };
                  })}
                />
                <BreakdownTable
                  title="By League"
                  rows={Object.entries(data.by_league_metrics).map(([k, v]) => ({ label: LEAGUE_LABEL[k] ?? k, metrics: v }))}
                />
              </div>
            </div>

            <div className="mt-8">
              <div className="mb-3 text-sm font-semibold text-ink">Distributions</div>
              <div className="flex flex-wrap gap-4">
                <Histogram
                  title="Odds Distribution"
                  buckets={bucketize(data.staked_bets.map((b) => b.odds), ODDS_EDGES, ODDS_LABELS)}
                />
                <LeagueBarChart
                  entries={Object.entries(data.by_league_metrics)
                    .map(([league, m]) => ({ league, count: m.bets_placed }))
                    .sort((a, b) => b.count - a.count)}
                />
                <Histogram
                  title="Stake Distribution"
                  buckets={bucketize(data.staked_bets.map((b) => b.stake), STAKE_EDGES, STAKE_LABELS)}
                />
              </div>
            </div>

            <div className="mb-8 mt-8">
              <div className="mb-3 text-sm font-semibold text-ink">Top Winning &amp; Losing Bets</div>
              <div className="flex flex-wrap gap-4">
                <BetsExampleTable title="Top 5 Winners" bets={data.top_winners} colorClass="text-good" />
                <BetsExampleTable title="Top 5 Losers" bets={data.top_losers} colorClass="text-serious" />
              </div>
            </div>
          </>
        )}
      </div>
    </AppShell>
  );
}
