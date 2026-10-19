"use client";

import Image from "next/image";
import Link from "next/link";
import { useRouter } from "next/navigation";
import { useState, useEffect } from "react";
import { House, List, ListBullets, MagnifyingGlass, Ticket, X, type Icon } from "@phosphor-icons/react";

import { getFixtures } from "@/lib/api";
import type { Fixture } from "@/lib/types";
import { useSandboxAsOf } from "@/lib/useSandboxAsOf";
import { InstallAppBanner } from "@/components/InstallAppBanner";
import { UserMenu } from "@/components/UserMenu";

function matchAnalysisHref(f: Fixture) {
  return `/matches/${f.match_id}?home=${encodeURIComponent(f.home_team)}&away=${encodeURIComponent(
    f.away_team
  )}&date=${f.utc_date.slice(0, 10)}&league=${encodeURIComponent(f.competition ?? "E0")}`;
}

// W210 follow-up (2026-09-14): re-enabled -- real per-user auth, a visible
// signed-in/sign-out UI (Task 1), and auth-aware failure states (Tasks 2-6)
// close the gaps that motivated hiding this on 2026-08-13 (W106).
const NAV_ITEMS: {
  href: string;
  label: string;
  key: "dashboard" | "matches" | "bets";
  icon: Icon;
}[] = [
  { href: "/", label: "Daily Edges", key: "dashboard", icon: House },
  { href: "/matches", label: "All Matches", key: "matches", icon: ListBullets },
  { href: "/bets", label: "Bets", key: "bets", icon: Ticket },
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
  const router = useRouter();
  const [query, setQuery] = useState("");
  const [searchFixtures, setSearchFixtures] = useState<Fixture[]>([]);
  const [hasFocused, setHasFocused] = useState(false);
  // Direct feedback: the whole left panel (menu/footer) should be
  // collapsible on a phone, and the expanded state should overlay the main
  // content rather than sit inline and push it down.
  const [menuOpen, setMenuOpen] = useState(false);

  function handleTabClick(event: React.MouseEvent<HTMLAnchorElement>, href: string) {
    navigator.vibrate?.(10);
    if (typeof document === "undefined" || !("startViewTransition" in document)) return;
    if (href === window.location.pathname) return;
    event.preventDefault();
    (document as Document & { startViewTransition: (cb: () => void) => void }).startViewTransition(() => {
      router.push(href);
    });
  }

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
      <div className="text-xl font-semibold tracking-tight text-ink">Oddsey</div>
    </div>
  );

  // Direct feedback: only the middle (page) content should scroll, on
  // mobile and desktop alike -- header/sidebar, search bar, and the bottom
  // tab bar stay put. h-dvh + overflow-hidden bounds the shell to the
  // viewport so <main> below (overflow-y-auto) becomes the one scrolling
  // region, instead of the whole document scrolling.
  // BUG-088 follow-up (found live 2026-10-17): mobile used to scroll via
  // the document body instead, with the header/bottom tab bar pinned via
  // `sticky`/`fixed` + transform-gpu -- on iOS that still visibly detached
  // from the viewport during a fast scroll (confirmed live: transform-gpu
  // alone did not fix it). A dedicated scroll container, the same
  // architecture desktop already used, avoids that whole bug class: the
  // header/tab bar are simply outside the scrolling region, so they never
  // need `fixed`/`sticky` positioning at all.
  // BUG-089 (found live 2026-10-19): `h-screen` (100vh) is the *layout*
  // viewport, which on mobile Safari is sized as if the address-bar
  // toolbar were already collapsed -- bigger than the actually-visible
  // area while the toolbar is still showing (e.g. right after a page
  // load). The bottom tab bar, a plain flex sibling positioned after
  // <main> rather than inside its scrollport, rendered below that
  // currently-visible fold -- invisible until some scroll/touch gesture
  // made Safari collapse the toolbar and grow the real viewport to match.
  // `h-dvh` (100dvh, the *dynamic* viewport height Tailwind 3.4+ ships)
  // tracks the real visible area continuously instead, so the shell never
  // allocates more height than is actually on screen.
  return (
    <div className="flex h-dvh flex-col overflow-hidden lg:flex-row">
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
              bottom tab bar, not shown alongside it. Mockup point 2:
              each link gets its own icon, matching NAV_ITEMS' icon field
              (previously only used by the mobile bottom tab bar). */}
          <nav className="mt-6 flex flex-col gap-1 text-sm">
            {NAV_ITEMS.map((item) => {
              const Icon = item.icon;
              return (
                <Link
                  key={item.key}
                  href={item.href}
                  // Production bug (2026-09-15): Next.js auto-prefetches a
                  // visible Link on mount -- for "/bets" (middleware-gated,
                  // W210) that prefetch can fire before the session cookie
                  // exists (e.g. right after landing on "/" pre-login) and
                  // cache a "redirect to /login" result client-side, which
                  // a later real click then replays even once truly signed
                  // in. Only "/bets" needs this; "/" and "/matches" aren't
                  // middleware-gated, so nothing to poison there.
                  prefetch={item.key === "bets" ? false : undefined}
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
        <div className="mt-auto pt-4">
          <UserMenu />
        </div>
      </aside>

      {/* Mobile-only compact header: hamburger trigger + a small logo,
          replacing the sidebar's content that used to always render inline
          and push the page down. Direct feedback: the whole left panel
          (menu/footer) should collapse behind this, and expand as an
          overlay on top of the main content, not inline above it.
          Direct user request (2026-10-17): pinned in place while scrolling,
          matching the bottom tab bar below. `shrink-0` is all this needs
          now -- it's a plain flex child outside <main>'s own scroll region
          (see this component's own top-level comment, BUG-088 follow-up),
          so it's never in motion at all; no `fixed`/`sticky` positioning
          or spacer/padding compensation required. */}
      <div className="flex shrink-0 items-center justify-between border-b border-border bg-surface px-4 pb-3 pt-[calc(0.75rem+env(safe-area-inset-top))] lg:hidden">
        <div className="flex items-center gap-2">
          <OddseyLogo size={36} />
          <span className="text-base font-semibold tracking-tight text-ink">Oddsey</span>
        </div>
        <button type="button" onClick={() => setMenuOpen(true)} aria-label="Open menu" className="text-ink-secondary">
          <List size={22} />
        </button>
      </div>

      {/* Mobile overlay drawer -- backdrop + slide-in panel, above the
          search dropdown (z-10). The bottom tab bar/header no longer carry
          a z-index at all (BUG-088 follow-up) -- they're plain flex
          siblings now, not overlaid content, so stacking order is already
          correct from DOM order alone. */}
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
            <div className="mt-4">
              <UserMenu />
            </div>
          </div>
        </>
      )}

      <div className="flex min-h-0 min-w-0 flex-1 flex-col">
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

        <InstallAppBanner />

        {/* Direct feedback: <main> is the one scrolling region, on mobile
            and desktop alike (min-h-0 + overflow-y-auto, paired with the
            shell's own h-screen above) -- header/sidebar, this search bar
            (shrink-0), and the bottom tab bar never move. A page can
            additionally pin its own chrome (e.g. DashboardPage's
            title/rail) with sticky *within* this region. */}
        {/* pt-8 lives on this inner div, not on <main> itself: <main> is the
            scroll container (overflow-y-auto), and padding on a scroll
            container sits *inside* its scrollport, so scrolled content slides
            up through it -- nothing covers that strip above a sticky child's
            own box. A page's sticky title (e.g. DashboardPage's) then has a
            gap above it that scrolled cards bleed through. Padding on a
            non-scrolling descendant just scrolls away with the content, so a
            sticky child's `top-0` lands flush with the true scrollport edge. */}
        <main className="min-h-0 flex-1 overflow-y-auto px-4 pb-8 [view-transition-name:main-content] sm:px-6">
          <div className="pt-8">{children}</div>
        </main>
      </div>

      {/* W114: bottom-tab-bar nav for small screens -- the sidebar-nav
          metaphor above is unfamiliar to a bottom-tab/vertical-feed mental
          model (Instagram/TikTok/X). Hidden at `lg` and up, where the
          desktop sidebar nav (above) is the only nav shown. `shrink-0` is
          all this needs now -- see this component's own top-level comment
          (BUG-088 follow-up): it's a plain flex child outside <main>'s own
          scroll region, never in motion, so no `fixed` positioning or
          transform-gpu compositing trick is needed at all. */}
      <nav className="flex shrink-0 items-center justify-around border-t border-border bg-page px-2 pb-[calc(0.375rem+env(safe-area-inset-bottom))] pt-1.5 lg:hidden">
        {NAV_ITEMS.map((item) => {
          const Icon = item.icon;
          const isActive = active === item.key;
          return (
            <Link
              key={item.key}
              href={item.href}
              // See the sidebar nav's own Link above -- same pre-login
              // stale-prefetch-redirect issue applies to this mobile tab.
              prefetch={item.key === "bets" ? false : undefined}
              onClick={(event) => handleTabClick(event, item.href)}
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
