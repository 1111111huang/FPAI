"use client";

import Link from "next/link";
import { useRef, useState, useEffect } from "react";
import { MagnifyingGlass } from "@phosphor-icons/react";

import { getFixtures, getStatus } from "@/lib/api";
import type { Fixture, StatusResponse } from "@/lib/types";
import { useSandboxAsOf } from "@/lib/useSandboxAsOf";

function matchAnalysisHref(f: Fixture) {
  return `/matches/${f.match_id}?home=${encodeURIComponent(f.home_team)}&away=${encodeURIComponent(
    f.away_team
  )}&date=${f.utc_date.slice(0, 10)}&league=${encodeURIComponent(f.competition ?? "E0")}`;
}

const NAV_ITEMS: { href: string; label: string; key: "dashboard" | "matches" | "bets" }[] = [
  { href: "/", label: "Dashboard", key: "dashboard" },
  { href: "/matches", label: "All Matches", key: "matches" },
  { href: "/bets", label: "Bets", key: "bets" },
];

export function AppShell({
  active,
  activeEdgesCount,
  children,
}: {
  active: "dashboard" | "matches" | "bets";
  activeEdgesCount?: number;
  children: React.ReactNode;
}) {
  const { asOf } = useSandboxAsOf();
  const [status, setStatus] = useState<StatusResponse | null>(null);
  const [query, setQuery] = useState("");
  const [searchFixtures, setSearchFixtures] = useState<Fixture[]>([]);
  const fetchedSearchRef = useRef(false);

  useEffect(() => {
    let cancelled = false;
    (async () => {
      try {
        const s = await getStatus();
        if (!cancelled) setStatus(s);
      } catch {
        // W17's StatusFooter precedent: a passive display, not worth an
        // error state of its own -- the sidebar just shows "--" instead.
        if (!cancelled) setStatus(null);
      }
    })();
    return () => {
      cancelled = true;
    };
  }, []);

  function ensureSearchFixturesLoaded() {
    // Lazy on purpose -- see this plan's "Before you start" note. Eagerly
    // fetching on mount would add a call to the same shared `getFixtures`
    // mock BetTracker.fixtureError.test.tsx asserts an exact count against.
    if (fetchedSearchRef.current) return;
    fetchedSearchRef.current = true;
    const from = new Date(asOf);
    const to = new Date(asOf);
    to.setUTCDate(to.getUTCDate() + 90);
    (async () => {
      try {
        const fixtures = await getFixtures(from.toISOString().slice(0, 10), to.toISOString().slice(0, 10));
        setSearchFixtures(fixtures ?? []);
      } catch {
        setSearchFixtures([]);
      }
    })();
  }

  const q = query.trim().toLowerCase();
  const results =
    q.length === 0
      ? []
      : searchFixtures
          .filter((f) => f.home_team.toLowerCase().includes(q) || f.away_team.toLowerCase().includes(q))
          .slice(0, 8);

  const dataFreshness = status?.data_freshness;
  const leagueModelCount = Object.keys(status?.model_status.league ?? {}).length;
  const internationalModelCount = Object.keys(status?.model_status.international ?? {}).length;

  return (
    <div className="min-h-screen lg:flex">
      <aside className="flex shrink-0 flex-col justify-between border-b border-border px-4 py-5 lg:h-screen lg:w-56 lg:border-b-0 lg:border-r lg:px-5 lg:py-6">
        <div>
          <span className="text-sm font-semibold tracking-tight text-ink">FPAI</span>
          <nav className="mt-6 flex flex-col gap-1 text-sm">
            {NAV_ITEMS.map((item) => (
              <Link
                key={item.key}
                href={item.href}
                className={`rounded-md px-2 py-1.5 transition-colors duration-150 ${
                  active === item.key ? "bg-surface text-ink" : "text-ink-secondary hover:text-ink"
                }`}
              >
                {item.label}
              </Link>
            ))}
          </nav>
        </div>

        <div className="mt-6 flex flex-col gap-3 border-t border-border pt-4 text-xs lg:mt-0">
          {activeEdgesCount !== undefined && (
            <div>
              <div className="text-muted uppercase tracking-wide">Active Edges</div>
              <div className="mt-0.5 font-mono text-base text-ink">{activeEdgesCount}</div>
            </div>
          )}
          <div>
            <div className="text-muted uppercase tracking-wide">Model Status</div>
            <div className="mt-0.5 text-ink-secondary">
              {status ? `league ${leagueModelCount} · international ${internationalModelCount}` : "—"}
            </div>
          </div>
          <div>
            <div className="text-muted uppercase tracking-wide">Last Updated</div>
            <div className={`mt-0.5 ${dataFreshness?.is_stale ? "text-warning" : "text-ink-secondary"}`}>
              {dataFreshness
                ? `${dataFreshness.latest_match_date ?? "unknown"}${
                    dataFreshness.days_since_update !== null ? ` (${dataFreshness.days_since_update}d ago)` : ""
                  }`
                : "—"}
            </div>
          </div>
        </div>
      </aside>

      <div className="flex min-w-0 flex-1 flex-col">
        <div className="border-b border-border px-4 py-3 sm:px-6">
          <div className="relative max-w-md">
            <MagnifyingGlass size={16} className="pointer-events-none absolute left-3 top-1/2 -translate-y-1/2 text-muted" />
            <input
              value={query}
              onChange={(e) => setQuery(e.target.value)}
              onFocus={ensureSearchFixturesLoaded}
              placeholder="Search fixtures, teams…"
              className="w-full rounded-lg border border-border bg-surface py-2 pl-9 pr-3 text-sm text-ink outline-none placeholder:text-muted focus:border-accent"
            />
            {results.length > 0 && (
              <div className="absolute left-0 right-0 top-full z-10 mt-1 flex flex-col gap-1 rounded-lg border border-border bg-page p-1.5 shadow-lg">
                {results.map((f) => (
                  <Link
                    key={f.match_id}
                    href={matchAnalysisHref(f)}
                    onClick={() => setQuery("")}
                    className="rounded-md px-2 py-1.5 text-sm text-ink hover:bg-surface"
                  >
                    {f.home_team} v {f.away_team}
                    <span className="ml-2 text-xs text-ink-secondary">{f.utc_date.slice(0, 10)}</span>
                  </Link>
                ))}
              </div>
            )}
          </div>
        </div>

        <main className="flex-1 px-4 py-8 sm:px-6">{children}</main>
      </div>
    </div>
  );
}
