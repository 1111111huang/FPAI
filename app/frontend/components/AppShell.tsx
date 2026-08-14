"use client";

import Link from "next/link";
import { useState, useEffect } from "react";
import { House, List, ListBullets, MagnifyingGlass, X, type Icon } from "@phosphor-icons/react";

import { getFixtures, getStatus } from "@/lib/api";
import type { Fixture, StatusResponse } from "@/lib/types";
import { useSandboxAsOf } from "@/lib/useSandboxAsOf";

function matchAnalysisHref(f: Fixture) {
  return `/matches/${f.match_id}?home=${encodeURIComponent(f.home_team)}&away=${encodeURIComponent(
    f.away_team
  )}&date=${f.utc_date.slice(0, 10)}&league=${encodeURIComponent(f.competition ?? "E0")}`;
}

// Bets tab hidden from nav (2026-08-13) -- feature not ready yet. The route
// (/bets, BetTracker.tsx) and its active="bets" AppShell state are left
// intact, just unlinked -- flip this back to re-surface it, no other change
// needed.
const NAV_ITEMS: {
  href: string;
  label: string;
  key: "dashboard" | "matches" | "bets";
  icon: Icon;
}[] = [
  { href: "/", label: "Daily Edges", key: "dashboard", icon: House },
  { href: "/matches", label: "All Matches", key: "matches", icon: ListBullets },
];

/** Rebrand (2026-08-13): FPAI -> Oddsey. Hand-recreated as inline SVG (not
 * an embedded raster copy of the provided reference image) -- a double
 * gold ring, an upward price-chart zigzag with a gold node at its
 * midpoint and green nodes at each end, and two short arc "scan" brackets
 * -- matching this codebase's existing convention of every icon (Phosphor
 * set included) being an inline SVG component, not a separate asset file. */
function OddseyLogo({ size = 32 }: { size?: number }) {
  return (
    <svg width={size} height={size} viewBox="0 0 200 200" aria-hidden="true">
      <circle cx="100" cy="100" r="96" fill="#0f1f3a" />
      <circle cx="100" cy="100" r="88" fill="none" stroke="#d9a94f" strokeWidth="6" />
      <circle cx="100" cy="100" r="76" fill="none" stroke="#d9a94f" strokeWidth="6" />
      <path
        d="M 45 118 L 62 100 L 78 112 L 100 90 L 118 100 L 132 82 L 152 62"
        fill="none"
        stroke="#d9a94f"
        strokeWidth="7"
        strokeLinecap="round"
        strokeLinejoin="round"
      />
      <circle cx="100" cy="100" r="9" fill="#d9a94f" />
      <circle cx="45" cy="118" r="7" fill="#1c8a4d" />
      <circle cx="152" cy="62" r="7" fill="#1c8a4d" />
      <path d="M 128 78 A 22 22 0 0 1 138 96" fill="none" stroke="#d9a94f" strokeWidth="4" strokeLinecap="round" />
      <path d="M 72 104 A 22 22 0 0 0 62 122" fill="none" stroke="#d9a94f" strokeWidth="4" strokeLinecap="round" />
    </svg>
  );
}

export function AppShell({
  active,
  activeEdgesCount,
  railTrigger,
  children,
}: {
  active: "dashboard" | "matches" | "bets";
  activeEdgesCount?: number;
  // Direct feedback: on small screens the right rail should "squish with
  // the top part with the search bar" instead of being its own section --
  // an optional slot next to the search input (mobile-only, lg:hidden) for
  // a page-specific trigger (DashboardPage's own rail-drawer icon). AppShell
  // doesn't know what DashboardRail is; it just reserves the spot.
  railTrigger?: React.ReactNode;
  children: React.ReactNode;
}) {
  // Independent of whatever page renders us, which may also call this hook
  // itself (e.g. DashboardPage) -- a known, accepted duplicate GET
  // /api/sandbox/status per mount, not shared/cached. Cheap local read;
  // worth revisiting if AppShell ends up wrapping many more pages.
  const { asOf } = useSandboxAsOf();
  const [status, setStatus] = useState<StatusResponse | null>(null);
  const [query, setQuery] = useState("");
  const [searchFixtures, setSearchFixtures] = useState<Fixture[]>([]);
  const [hasFocused, setHasFocused] = useState(false);
  // Direct feedback: the whole left panel (menu/model info/footer) should
  // be collapsible on a phone, and the expanded state should overlay the
  // main content rather than sit inline and push it down.
  const [menuOpen, setMenuOpen] = useState(false);

  useEffect(() => {
    // Refetched on every mount -- AppShell is instantiated per-page like the
    // DraftNav it replaces, not persisted across navigation; the sidebar
    // footer briefly shows placeholders after each route change.
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

  useEffect(() => {
    // Lazy on purpose -- see this plan's "Before you start" note. Eagerly
    // fetching on mount would add a call to the same shared `getFixtures`
    // mock BetTracker.fixtureError.test.tsx asserts an exact count against.
    // Keyed on [hasFocused, asOf] rather than a one-shot ref (W42 pattern,
    // see MatchUI.tsx's DashboardPage/MatchExplorerPage): useSandboxAsOf()
    // resolves asynchronously -- the real browser clock first, then the
    // corrected sandbox date once GET /api/sandbox/status returns -- so a
    // focus that happens before that correction lands must still trigger a
    // second, corrected fetch rather than being locked into the stale
    // real-clock window for the rest of the mount's life.
    if (!hasFocused) return;
    let cancelled = false;
    const from = new Date(asOf);
    const to = new Date(asOf);
    to.setUTCDate(to.getUTCDate() + 90);
    (async () => {
      try {
        const fixtures = await getFixtures(from.toISOString().slice(0, 10), to.toISOString().slice(0, 10));
        if (!cancelled) setSearchFixtures(fixtures ?? []);
      } catch {
        if (!cancelled) setSearchFixtures([]);
      }
    })();
    return () => {
      cancelled = true;
    };
  }, [hasFocused, asOf]);

  const q = query.trim().toLowerCase();
  const results =
    q.length === 0
      ? []
      : searchFixtures
          .filter((f) => f.home_team.toLowerCase().includes(q) || f.away_team.toLowerCase().includes(q))
          .slice(0, 8);

  const dataFreshness = status?.data_freshness;
  // US#110: model_status is keyed dynamically by competition_id (e.g. "E0",
  // "SWE"), not a fixed "league" key -- sum every non-international bucket.
  const leagueModelCount = Object.entries(status?.model_status ?? {})
    .filter(([ctx]) => ctx !== "international")
    .reduce((sum, [, models]) => sum + Object.keys(models).length, 0);
  const internationalModelCount = Object.keys(status?.model_status.international ?? {}).length;
  // W74: only worth a breakdown once a second competition actually exists --
  // a single-entry by_league is by definition identical to the blended line
  // above it, so showing both would just be noise.
  const byLeagueEntries = Object.entries(dataFreshness?.by_league ?? {}).sort(([a], [b]) => a.localeCompare(b));

  // Shared between the desktop <aside> and the mobile overlay drawer below
  // -- same content, two different containers, rather than duplicated JSX.
  const brandBlock = (
    <div className="flex items-center gap-2.5">
      <OddseyLogo size={32} />
      <div className="leading-tight">
        <div className="text-sm font-semibold tracking-tight text-ink">Oddsey</div>
        <div className="text-[10px] uppercase tracking-wide text-muted">Edge Engine</div>
      </div>
    </div>
  );

  const infoBlock = (
    <>
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
              }${dataFreshness.is_stale ? " -- stale" : ""}`
            : "—"}
        </div>
        {byLeagueEntries.length > 1 && (
          <div className="mt-1 flex flex-col gap-0.5">
            {byLeagueEntries.map(([league, freshness]) => (
              <div key={league} className={freshness.is_stale ? "text-warning" : "text-ink-secondary"}>
                <span className="font-medium">{league}</span>:{" "}
                {freshness.days_since_update !== null ? `${freshness.days_since_update}d ago` : "unknown"}
                {freshness.is_stale ? " -- stale" : ""}
              </div>
            ))}
          </div>
        )}
      </div>
    </>
  );

  // Direct feedback: only the middle (page) content should scroll on
  // desktop -- sidebar, search bar, and (per-page) sticky chrome stay put.
  // lg:h-screen + lg:overflow-hidden bounds the shell to the viewport so
  // <main> below (lg:overflow-y-auto) becomes the one scrolling region,
  // instead of the whole document scrolling.
  return (
    <div className="min-h-screen lg:flex lg:h-screen lg:overflow-hidden">
      {/* Desktop sidebar -- unchanged content, now hidden lg:flex instead of
          always visible: below `lg` the whole panel (menu/model info/
          footer) collapses behind the mobile header bar + overlay drawer
          below instead of always rendering inline. Mockup correction: the
          sidebar reads as the *same* plain page background as the main
          content (no fill of its own, just the border-r divider) -- the
          bg-surface tried earlier made it match the rail, when the
          reference actually has them looking different from each other. */}
      <aside className="hidden shrink-0 flex-col justify-between px-5 py-6 lg:flex lg:h-screen lg:w-56 lg:border-r lg:border-border">
        <div>
          {brandBlock}
          {/* W114: desktop-only -- below `lg` this list is replaced by the
              fixed bottom tab bar, not shown alongside it. Mockup point 2:
              each link gets its own icon, matching NAV_ITEMS' icon field
              (previously only used by the mobile bottom tab bar). */}
          <nav className="mt-6 flex flex-col gap-1 text-sm">
            {NAV_ITEMS.map((item) => {
              const Icon = item.icon;
              return (
                <Link
                  key={item.key}
                  href={item.href}
                  className={`flex items-center gap-2 rounded-md px-2 py-1.5 transition-colors duration-150 ${
                    active === item.key ? "bg-surface text-ink" : "text-ink-secondary hover:text-ink"
                  }`}
                >
                  <Icon size={16} weight={active === item.key ? "fill" : "regular"} />
                  {item.label}
                </Link>
              );
            })}
          </nav>
        </div>

        <div className="flex flex-col gap-3 border-t border-border pt-4 text-xs">{infoBlock}</div>
      </aside>

      {/* Mobile-only compact header: hamburger trigger + a small logo,
          replacing the sidebar's content that used to always render inline
          and push the page down. Direct feedback: the whole left panel
          (menu/model info/footer) should collapse behind this, and expand
          as an overlay on top of the main content, not inline above it. */}
      <div className="flex items-center justify-between border-b border-border bg-surface px-4 py-3 lg:hidden">
        <div className="flex items-center gap-2">
          <OddseyLogo size={24} />
          <span className="text-sm font-semibold tracking-tight text-ink">Oddsey</span>
        </div>
        <button type="button" onClick={() => setMenuOpen(true)} aria-label="Open menu" className="text-ink-secondary">
          <List size={22} />
        </button>
      </div>

      {/* Mobile overlay drawer -- backdrop + slide-in panel, above the
          fixed bottom tab bar (z-20) and the search dropdown (z-10). */}
      {menuOpen && (
        <>
          <div
            className="fixed inset-0 z-40 bg-page/70 lg:hidden"
            onClick={() => setMenuOpen(false)}
            aria-hidden="true"
          />
          <div className="fixed inset-y-0 left-0 z-50 w-72 max-w-[85vw] overflow-y-auto bg-surface p-5 shadow-xl lg:hidden">
            <div className="flex items-start justify-between">
              {brandBlock}
              <button
                type="button"
                onClick={() => setMenuOpen(false)}
                aria-label="Close menu"
                className="text-ink-secondary"
              >
                <X size={20} />
              </button>
            </div>
            <div className="mt-6 flex flex-col gap-3 border-t border-border pt-4 text-xs">{infoBlock}</div>
          </div>
        </>
      )}

      <div className="flex min-w-0 flex-1 flex-col lg:min-h-0">
        <div className="flex shrink-0 items-center gap-2 border-b border-border px-4 py-3 sm:px-6">
          <div className="relative min-w-0 max-w-md flex-1">
            <MagnifyingGlass size={16} className="pointer-events-none absolute left-3 top-1/2 -translate-y-1/2 text-muted" />
            <input
              value={query}
              onChange={(e) => setQuery(e.target.value)}
              onFocus={() => setHasFocused(true)}
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
          {railTrigger && <div className="shrink-0 lg:hidden">{railTrigger}</div>}
        </div>

        {/* Direct feedback: <main> is now the one scrolling region on
            desktop (lg:min-h-0 + lg:overflow-y-auto, paired with the shell's
            own lg:h-screen above) -- sidebar and this search bar (shrink-0)
            never move. A page can additionally pin its own chrome (e.g.
            DashboardPage's title/rail) with sticky *within* this region.
            W114: pb-20 clears the fixed bottom tab bar below `lg`; back to
            the original py-8 at `lg` and up, where that bar doesn't render. */}
        {/* pt-8 lives on this inner div, not on <main> itself: <main> is the
            scroll container (lg:overflow-y-auto), and padding on a scroll
            container sits *inside* its scrollport, so scrolled content slides
            up through it -- nothing covers that strip above a sticky child's
            own box. A page's sticky title (e.g. DashboardPage's) then has a
            gap above it that scrolled cards bleed through. Padding on a
            non-scrolling descendant just scrolls away with the content, so a
            sticky child's `top-0` lands flush with the true scrollport edge. */}
        <main className="flex-1 px-4 pb-20 sm:px-6 lg:min-h-0 lg:overflow-y-auto lg:pb-8">
          <div className="pt-8">{children}</div>
        </main>
      </div>

      {/* W114: bottom-tab-bar nav for small screens -- the sidebar-nav
          metaphor above is unfamiliar to a bottom-tab/vertical-feed mental
          model (Instagram/TikTok/X). Hidden at `lg` and up, where the
          desktop sidebar nav (above) is the only nav shown. */}
      <nav className="fixed inset-x-0 bottom-0 z-20 flex items-center justify-around border-t border-border bg-page px-2 py-1.5 lg:hidden">
        {NAV_ITEMS.map((item) => {
          const Icon = item.icon;
          const isActive = active === item.key;
          return (
            <Link
              key={item.key}
              href={item.href}
              className={`flex flex-col items-center gap-0.5 rounded-md px-4 py-1.5 text-[11px] font-medium transition-colors duration-150 ${
                isActive ? "text-ink" : "text-ink-secondary"
              }`}
            >
              <Icon size={20} weight={isActive ? "fill" : "regular"} />
              {item.label}
            </Link>
          );
        })}
      </nav>
    </div>
  );
}
