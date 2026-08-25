"use client";

import Image from "next/image";
import Link from "next/link";
import { Montserrat } from "next/font/google";
import { useState, useEffect } from "react";
import { House, List, ListBullets, MagnifyingGlass, X, type Icon } from "@phosphor-icons/react";

import { getFixtures } from "@/lib/api";
import type { Fixture } from "@/lib/types";
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

/** Rebrand (2026-08-13): FPAI -> Oddsey. Was a hand-recreated inline SVG
 * (the original reference image's bytes weren't extractable through that
 * session's tools); replaced 2026-08-14 with the real provided logo file
 * once available (`public/oddsey-logo.png`), served via next/image for
 * automatic sizing/optimization -- this codebase's one raster asset, the
 * rest of its icons (Phosphor set included) are inline SVG. */
function OddseyLogo({ size = 32 }: { size?: number }) {
  return <Image src="/oddsey-logo.png" width={size} height={size} alt="Oddsey" priority />;
}

// Wordmark font, scoped to just the "Oddsey" text (not a site-wide font
// change) -- next/font/google self-hosts at build time (no runtime request,
// no FOUT), already built into Next.js so no new dependency.
const brandFont = Montserrat({ subsets: ["latin"], weight: ["600"] });

export function AppShell({
  active,
  railTrigger,
  children,
}: {
  // W174: "agent-performance" has no NAV_ITEMS entry (the page itself is
  // unlinked -- direct-URL-only, matching "bets"'s own existing precedent:
  // NAV_ITEMS omits it too, but "bets" stays a valid `active` value here
  // for BetTrackerPage's own unaffected use). Adding a value here never
  // requires a matching NAV_ITEMS entry -- the two are independent.
  active: "dashboard" | "matches" | "bets" | "agent-performance";
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
  const [query, setQuery] = useState("");
  const [searchFixtures, setSearchFixtures] = useState<Fixture[]>([]);
  const [hasFocused, setHasFocused] = useState(false);
  // Direct feedback: the whole left panel (menu/footer) should be
  // collapsible on a phone, and the expanded state should overlay the main
  // content rather than sit inline and push it down.
  const [menuOpen, setMenuOpen] = useState(false);

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

  const brandBlock = (
    <div className="flex items-center gap-2.5">
      <OddseyLogo size={48} />
      <div className={`${brandFont.className} text-xl tracking-tight text-ink`}>Oddsey</div>
    </div>
  );

  // Direct feedback: only the middle (page) content should scroll on
  // desktop -- sidebar, search bar, and (per-page) sticky chrome stay put.
  // lg:h-screen + lg:overflow-hidden bounds the shell to the viewport so
  // <main> below (lg:overflow-y-auto) becomes the one scrolling region,
  // instead of the whole document scrolling.
  return (
    <div className="min-h-screen lg:flex lg:h-screen lg:overflow-hidden">
      {/* Desktop sidebar -- unchanged content, now hidden lg:flex instead of
          always visible: below `lg` the whole panel (menu/footer) collapses
          behind the mobile header bar + overlay drawer below instead of
          always rendering inline. Mockup correction: the
          sidebar reads as the *same* plain page background as the main
          content (no fill of its own, just the border-r divider) -- the
          bg-surface tried earlier made it match the rail, when the
          reference actually has them looking different from each other. */}
      <aside className="hidden shrink-0 flex-col px-5 py-6 lg:flex lg:h-screen lg:w-56 lg:border-r lg:border-border">
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
      </aside>

      {/* Mobile-only compact header: hamburger trigger + a small logo,
          replacing the sidebar's content that used to always render inline
          and push the page down. Direct feedback: the whole left panel
          (menu/footer) should collapse behind this, and expand as an
          overlay on top of the main content, not inline above it. */}
      <div className="flex items-center justify-between border-b border-border bg-surface px-4 py-3 lg:hidden">
        <div className="flex items-center gap-2">
          <OddseyLogo size={36} />
          <span className={`${brandFont.className} text-base tracking-tight text-ink`}>Oddsey</span>
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
