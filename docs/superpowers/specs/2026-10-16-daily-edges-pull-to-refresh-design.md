# Daily Edges Pull-to-Refresh

## 1. Problem

The PWA (`docs/superpowers/specs/2026-10-15-pwa-mobile-app-design.md`) has no browser chrome in standalone/installed mode, so the native pull-to-refresh gesture most mobile browsers provide isn't available. The Daily Edges page (`DashboardPage`, `app/frontend/components/MatchUI.tsx`) currently has no way to force a reload except:

- An `ErrorState`'s `onRetry` button, which only renders after a fetch actually fails.
- A passive live-match poll (`useLiveMatchPolling`) that only fires while a shown match is live, on a fixed interval.
- A client-side cache (`dashboardMatchesCache`, keyed by day, TTL 60s while anything live/imminent is shown else 5 min) that silently serves a same-session cached list, skipping the fetch entirely.

Direct finding this session: right after a backend rate-limit recovery, the dashboard served an incomplete match list (missing a finished Ligue 1 fixture) because that league's data hadn't successfully round-tripped through the cache yet at the moment it was last fetched. There was no way to ask the page to just check again.

## 2. Scope

**In scope:**
- A pull-to-refresh gesture on the Daily Edges page, mobile viewports only (no native gesture to replace on desktop, and `<main>` isn't even the scroll container there — see Section 3).
- Dragging down from the very top of the page shows a small indicator that tracks the drag, locks into a spinning state past a release threshold, and triggers a refresh.
- The refresh forces a real re-fetch of `/api/fixtures`, bypassing `dashboardMatchesCache`'s TTL short-circuit, then re-renders with whatever recommendations are already cached server-side.

**Out of scope (explicitly not doing):**
- Any change to what a refresh fetches or computes. No recommendation/prediction regeneration, no new backend endpoint, no new query parameters -- this reuses `DashboardPage`'s existing `getFixtures()` → `fixtureToMatch()` → bulk cached-recommendation-resolve flow verbatim, just forced to actually run.
- Other pages (Match Explorer, Bets, match detail). Daily Edges only for now; the hook built here is generic enough to reuse later if wanted, but no other page is wired to it in this change.
- A desktop/mouse equivalent (e.g. a visible button at `lg` and up). Out of scope -- not asked for, and desktop already has a real browser reload.
- Replacing or changing `useLiveMatchPolling`'s existing passive refresh. Both can coexist; the gesture is just a second, user-initiated way to trigger the same underlying reload.

## 3. Mechanism

New hook, `usePullToRefresh(onRefresh: () => void)`, in `app/frontend/components/MatchUI.tsx` (or a new small `lib/usePullToRefresh.ts` if that file is judged too large at implementation time -- a call for the plan, not this spec).

Mobile-only: `AppShell.tsx`'s `<main>` only becomes its own scroll container at the `lg` breakpoint (`lg:overflow-y-auto`); below `lg` the page scrolls via the document body. The hook checks viewport width (matching the existing `lg` breakpoint convention already used elsewhere in this file) and no-ops entirely at `lg` and up -- it attaches no listeners there.

Below `lg`, the hook attaches `touchstart`/`touchmove`/`touchend` listeners on `window` (not a scrollable div, since the body itself is what scrolls):

- `touchstart`: record the touch's start Y, but only arm the gesture if `window.scrollY === 0` at that instant. Starting a drag anywhere else (mid-scroll) is ignored entirely, so this never fights normal scrolling.
- `touchmove` (only while armed): compute `deltaY = currentY - startY`. If `deltaY <= 0` (dragging up, or back past the start), disarm -- treat as a normal scroll, let the browser handle it. If `deltaY > 0`, this is a pull: call `preventDefault()` on this move (armed + currently in a pull only, never unconditionally, so normal page scrolling elsewhere on the page is untouched) and report a resisted pull distance (e.g. `Math.min(deltaY * 0.5, MAX_PULL_PX)`, `MAX_PULL_PX` e.g. 100) to a pull-distance state the indicator renders from.
- `touchend` (only while armed): if the last reported pull distance cleared `REFRESH_THRESHOLD_PX` (e.g. 70px), lock the indicator into a spinning state and call `onRefresh()`. Otherwise animate the pull distance back to 0 (CSS transition, no JS animation loop needed) and disarm.
- The indicator stays in its spinning/locked state until the caller (`DashboardPage`) signals the refresh has finished (the hook exposes a `finishRefresh()` to call once the triggered reload's promise resolves), then animates back to 0.

No new dependency -- native `TouchEvent`s only, consistent with this file's existing "ladder" of reaching for a platform feature before a library (and there's no gesture library in `package.json` to begin with).

## 4. Visual indicator

A small component (e.g. `PullToRefreshIndicator`) rendered above `DashboardPage`'s match list, inside the same sticky header area already in `MatchUI.tsx`. Height driven by the hook's current pull distance (0 when idle, growing while dragging, fixed at the threshold height while spinning). Content: a spinner icon (reuse whatever icon this file already uses for other loading/retry affordances -- `@phosphor-icons/react` is already a dependency, no new icon asset needed), rotated by drag progress while pulling, animating a continuous spin once locked into the refreshing state. No "Pull to refresh" / "Release to refresh" text label -- the spinner's own rotation and lock state communicate this the same way iOS/Android's native indicators do, and this file's established pattern elsewhere favors compact, icon-led affordances over explanatory copy.

## 5. Wiring into DashboardPage

`DashboardPage` already has `retryTick`/`setRetryTick` driving its reload `useEffect`. The new piece is a cache-invalidation step the gesture needs that `retryTick` alone doesn't give it: bumping `retryTick` alone doesn't force a re-fetch if `dashboardMatchesCache` still considers today's entry fresh (exactly the TTL short-circuit this feature exists to bypass).

Add a small exported helper next to the existing cache functions:

```ts
export function invalidateDashboardMatchesCache(key: string): void {
  dashboardMatchesCache.delete(key);
}
```

`DashboardPage` calls `usePullToRefresh(() => { invalidateDashboardMatchesCache(dateString(asOf, sandboxMode)); setRetryTick((t) => t + 1); })`. The existing `load()` effect's cache check (`getDashboardMatchesCache(today)`) then naturally falls through to a real fetch, exactly like a cold load -- no branching logic added to `load()` itself. Once `load()`'s fetch resolves (success or error), it calls the hook's `finishRefresh()` so the indicator animates away.

## 6. Testing

- `usePullToRefresh` unit tests (jsdom, simulated `TouchEvent`s): a drag starting at `scrollY > 0` never arms; a drag starting at `scrollY === 0` that clears the threshold calls `onRefresh` exactly once; one that doesn't clear the threshold does not call it; an upward drag (negative deltaY) disarms without calling it; the hook no-ops above the `lg` breakpoint (no listeners attached / `onRefresh` never reachable).
- `DashboardPage` integration test: seed `dashboardMatchesCache` with a fresh (non-expired) entry for today, simulate a completed pull-to-refresh, assert `getFixtures`/the mocked fetch is called again despite the cache being fresh (proves the bypass), and that no recommendation-generation endpoint is touched (proves "whatever predictions are in place does not need to change").
