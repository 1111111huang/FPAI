# Daily Edges Pull-to-Refresh Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a mobile-only pull-to-refresh gesture to the Daily Edges page that forces a real re-fetch of fixtures (bypassing the client-side cache), without regenerating any predictions.

**Architecture:** A pure gesture hook (`usePullToRefresh`, native `TouchEvent`s on `window`, no new dependency) reports pull distance/lock state; a small presentational component (`PullToRefreshIndicator`) renders a spinner from that state; `DashboardPage` wires the two together, invalidating one cache entry and bumping its existing `retryTick` reload mechanism on refresh.

**Tech Stack:** React 18 (hooks), TypeScript, Tailwind, Vitest + Testing Library (`renderHook`, `act`). No new npm dependency.

**Design doc:** `docs/superpowers/specs/2026-10-16-daily-edges-pull-to-refresh-design.md`

---

## Context for the implementing engineer

- `app/frontend/components/MatchUI.tsx` is the single large file holding every page component (`DashboardPage`, `MatchExplorerPage`, `MatchAnalysisPage`, etc.) plus their shared helpers. It already has a `"use client";` directive at the top.
- The page scrolls differently by viewport width: `app/frontend/components/AppShell.tsx`'s `<main>` only becomes its own scroll container at the `lg` Tailwind breakpoint (`lg:overflow-y-auto`, `lg` = 1024px, no custom override in `tailwind.config.js`). Below `lg`, the page scrolls via the document body — `window.scrollY`, not a scrollable div's `scrollTop`.
- `DashboardPage` (same file, starts around line 1652) already has a `retryTick` state that forces its data-loading `useEffect` to re-run, and a module-level `dashboardMatchesCache` (lines ~300-334) that short-circuits that reload if a fresh-enough entry already exists for the day. Only `getDashboardMatchesCache`/`setDashboardMatchesCache` touch this map directly today; neither is exported.
- Standalone small components in this directory (`InstallAppBanner.tsx`, `ServiceWorkerRegistration.tsx`, `UserMenu.tsx`) each get their own file with a top-of-file `"use client";` directive, even when simple. Follow that pattern rather than inlining more into the already-large `MatchUI.tsx`.
- `@phosphor-icons/react` is already a dependency (see MatchUI.tsx's own icon import block) — use it for the spinner icon, no new package.
- Touch-event testing has no existing precedent in this codebase; `window.matchMedia` has no global polyfill in `app/frontend/vitest.setup.ts`, so it's `undefined` in every test unless a test mocks it itself.

---

### Task 1: `usePullToRefresh` hook

**Files:**
- Create: `app/frontend/lib/usePullToRefresh.ts`
- Test: `app/frontend/lib/__tests__/usePullToRefresh.test.ts`

- [ ] **Step 1: Write the failing tests**

```ts
// app/frontend/lib/__tests__/usePullToRefresh.test.ts
import { describe, expect, it, vi, beforeEach, afterEach } from "vitest";
import { act, renderHook } from "@testing-library/react";
import { usePullToRefresh } from "../usePullToRefresh";

function setScrollY(value: number): void {
  Object.defineProperty(window, "scrollY", { value, configurable: true });
}

function setDesktopViewport(isDesktop: boolean): void {
  window.matchMedia = vi.fn().mockImplementation((query: string) => ({
    matches: isDesktop,
    media: query,
    addListener: vi.fn(),
    removeListener: vi.fn(),
  })) as unknown as typeof window.matchMedia;
}

function dispatchTouch(type: "touchstart" | "touchmove" | "touchend", clientY?: number): void {
  const event = new Event(type, { bubbles: true, cancelable: true }) as unknown as TouchEvent;
  if (clientY !== undefined) {
    Object.defineProperty(event, "touches", { value: [{ clientY }], configurable: true });
  }
  act(() => {
    window.dispatchEvent(event);
  });
}

describe("usePullToRefresh", () => {
  beforeEach(() => {
    setScrollY(0);
  });

  afterEach(() => {
    // @ts-expect-error -- deleting a test-only stub, not a real browser API removal
    delete window.matchMedia;
  });

  it("a drag starting at scrollY > 0 never arms -- no refresh on release past the threshold", () => {
    setScrollY(50);
    const onRefresh = vi.fn();
    renderHook(() => usePullToRefresh({ onRefresh, refreshing: false }));

    dispatchTouch("touchstart", 0);
    dispatchTouch("touchmove", 200);
    dispatchTouch("touchend");

    expect(onRefresh).not.toHaveBeenCalled();
  });

  it("a drag starting at scrollY === 0 that clears the threshold calls onRefresh exactly once and locks", () => {
    const onRefresh = vi.fn();
    const { result } = renderHook(() => usePullToRefresh({ onRefresh, refreshing: false }));

    dispatchTouch("touchstart", 0);
    dispatchTouch("touchmove", 200); // 200 * 0.5 resistance = 100px, capped at 100, clears the 70px threshold
    dispatchTouch("touchend");

    expect(onRefresh).toHaveBeenCalledTimes(1);
    expect(result.current.locked).toBe(true);
  });

  it("a drag that doesn't clear the threshold does not call onRefresh", () => {
    const onRefresh = vi.fn();
    renderHook(() => usePullToRefresh({ onRefresh, refreshing: false }));

    dispatchTouch("touchstart", 0);
    dispatchTouch("touchmove", 50); // 50 * 0.5 resistance = 25px, under the 70px threshold
    dispatchTouch("touchend");

    expect(onRefresh).not.toHaveBeenCalled();
  });

  it("an upward drag (negative deltaY) disarms without calling onRefresh", () => {
    const onRefresh = vi.fn();
    renderHook(() => usePullToRefresh({ onRefresh, refreshing: false }));

    dispatchTouch("touchstart", 100);
    dispatchTouch("touchmove", 50); // deltaY = -50, upward
    dispatchTouch("touchend");

    expect(onRefresh).not.toHaveBeenCalled();
  });

  it("no-ops entirely at the lg breakpoint and up -- no listeners attached", () => {
    setDesktopViewport(true);
    const onRefresh = vi.fn();
    renderHook(() => usePullToRefresh({ onRefresh, refreshing: false }));

    dispatchTouch("touchstart", 0);
    dispatchTouch("touchmove", 200);
    dispatchTouch("touchend");

    expect(onRefresh).not.toHaveBeenCalled();
  });

  it("stays locked while the caller's refreshing stays true, and resets pullDistance/locked once it flips back to false", () => {
    const onRefresh = vi.fn();
    const { result, rerender } = renderHook(
      ({ refreshing }) => usePullToRefresh({ onRefresh, refreshing }),
      { initialProps: { refreshing: false } },
    );

    dispatchTouch("touchstart", 0);
    dispatchTouch("touchmove", 200);
    dispatchTouch("touchend");
    expect(result.current.locked).toBe(true);

    rerender({ refreshing: true });
    expect(result.current.locked).toBe(true);

    rerender({ refreshing: false });
    expect(result.current.locked).toBe(false);
    expect(result.current.pullDistance).toBe(0);
  });
});
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `cd app/frontend && npx vitest run lib/__tests__/usePullToRefresh.test.ts`
Expected: FAIL — `Cannot find module '../usePullToRefresh'` (the hook doesn't exist yet).

- [ ] **Step 3: Write the hook**

```ts
// app/frontend/lib/usePullToRefresh.ts
import { useEffect, useRef, useState } from "react";

const REFRESH_THRESHOLD_PX = 70;
const MAX_PULL_PX = 100;
const PULL_RESISTANCE = 0.5;
// Tailwind's default `lg` breakpoint (no override in tailwind.config.js) --
// matches the same lg: convention AppShell.tsx/MatchUI.tsx already use for
// "desktop" everywhere else in this app.
const DESKTOP_MIN_WIDTH_PX = 1024;

function isMobileViewport(): boolean {
  // No window.matchMedia (e.g. an unmocked test environment) defaults to
  // "mobile" -- harmless in a real browser, where matchMedia always exists.
  return typeof window.matchMedia !== "function" || !window.matchMedia(`(min-width: ${DESKTOP_MIN_WIDTH_PX}px)`).matches;
}

/**
 * Mobile-only pull-to-refresh gesture, driven by native TouchEvents on
 * `window` -- the page scrolls via the document body below the `lg`
 * breakpoint (AppShell.tsx's `<main>` only becomes its own scroll
 * container at `lg` and up), so there's no single scrollable element to
 * attach to, and no native gesture to replace at `lg` and up anyway.
 *
 * `refreshing` is caller-controlled: true while the caller's own triggered
 * refresh is in flight. The indicator stays locked at the threshold height
 * for as long as `refreshing` is true, and resets to 0 the render after it
 * flips back to false -- this is how the caller tells the hook "done."
 */
export function usePullToRefresh({
  onRefresh,
  refreshing,
}: {
  onRefresh: () => void;
  refreshing: boolean;
}): { pullDistance: number; locked: boolean; dragging: boolean } {
  const [pullDistance, setPullDistance] = useState(0);
  const [locked, setLocked] = useState(false);
  const [dragging, setDragging] = useState(false);
  const armedRef = useRef(false);
  const startYRef = useRef(0);

  useEffect(() => {
    if (!refreshing) {
      setPullDistance(0);
      setLocked(false);
    }
  }, [refreshing]);

  useEffect(() => {
    if (!isMobileViewport()) return;

    function handleTouchStart(e: TouchEvent): void {
      if (window.scrollY !== 0) return;
      armedRef.current = true;
      startYRef.current = e.touches[0].clientY;
      setDragging(true);
    }

    function handleTouchMove(e: TouchEvent): void {
      if (!armedRef.current) return;
      const deltaY = e.touches[0].clientY - startYRef.current;
      if (deltaY <= 0) {
        armedRef.current = false;
        setDragging(false);
        setPullDistance(0);
        return;
      }
      e.preventDefault();
      setPullDistance(Math.min(deltaY * PULL_RESISTANCE, MAX_PULL_PX));
    }

    function handleTouchEnd(): void {
      if (!armedRef.current) return;
      armedRef.current = false;
      setDragging(false);
      setPullDistance((current) => {
        if (current >= REFRESH_THRESHOLD_PX) {
          setLocked(true);
          onRefresh();
          return REFRESH_THRESHOLD_PX;
        }
        return 0;
      });
    }

    window.addEventListener("touchstart", handleTouchStart, { passive: true });
    window.addEventListener("touchmove", handleTouchMove, { passive: false });
    window.addEventListener("touchend", handleTouchEnd, { passive: true });
    return () => {
      window.removeEventListener("touchstart", handleTouchStart);
      window.removeEventListener("touchmove", handleTouchMove);
      window.removeEventListener("touchend", handleTouchEnd);
    };
  }, [onRefresh]);

  return { pullDistance, locked, dragging };
}
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `cd app/frontend && npx vitest run lib/__tests__/usePullToRefresh.test.ts`
Expected: PASS (6 tests)

- [ ] **Step 5: Commit**

```bash
git add app/frontend/lib/usePullToRefresh.ts app/frontend/lib/__tests__/usePullToRefresh.test.ts
git commit -m "feat(frontend): add usePullToRefresh gesture hook

Co-Authored-By: Claude Sonnet 5 <noreply@anthropic.com>"
```

---

### Task 2: `PullToRefreshIndicator` component

**Files:**
- Create: `app/frontend/components/PullToRefreshIndicator.tsx`
- Test: `app/frontend/components/__tests__/PullToRefreshIndicator.test.tsx`

- [ ] **Step 1: Write the failing tests**

```tsx
// app/frontend/components/__tests__/PullToRefreshIndicator.test.tsx
import { describe, expect, it } from "vitest";
import { render } from "@testing-library/react";
import { PullToRefreshIndicator } from "../PullToRefreshIndicator";

describe("PullToRefreshIndicator", () => {
  it("renders nothing when idle (pullDistance 0, not locked)", () => {
    const { container } = render(
      <PullToRefreshIndicator pullDistance={0} locked={false} dragging={false} />
    );
    expect(container).toBeEmptyDOMElement();
  });

  it("grows to match pullDistance while dragging", () => {
    const { container } = render(
      <PullToRefreshIndicator pullDistance={40} locked={false} dragging={true} />
    );
    const wrapper = container.firstElementChild as HTMLElement;
    expect(wrapper.style.height).toBe("40px");
  });

  it("locks at the threshold height and spins once locked", () => {
    const { container } = render(
      <PullToRefreshIndicator pullDistance={70} locked={true} dragging={false} />
    );
    const wrapper = container.firstElementChild as HTMLElement;
    expect(wrapper.style.height).toBe("70px");
    expect(container.querySelector(".animate-spin")).toBeInTheDocument();
  });
});
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `cd app/frontend && npx vitest run components/__tests__/PullToRefreshIndicator.test.tsx`
Expected: FAIL — `Cannot find module '../PullToRefreshIndicator'`

- [ ] **Step 3: Write the component**

```tsx
// app/frontend/components/PullToRefreshIndicator.tsx
"use client";

import { ArrowClockwise } from "@phosphor-icons/react";

const REFRESH_THRESHOLD_PX = 70;

/** Visual half of usePullToRefresh (lib/usePullToRefresh.ts) -- a small
 * spinner that grows in from the top as the page is pulled down, locks
 * into a spin once released past the threshold, and collapses back to 0
 * once the caller's refresh finishes. No "Pull to refresh"/"Release to
 * refresh" text -- the spinner's own rotation and lock state communicate
 * this the same way iOS/Android's native indicators do. */
export function PullToRefreshIndicator({
  pullDistance,
  locked,
  dragging,
}: {
  pullDistance: number;
  locked: boolean;
  dragging: boolean;
}) {
  const height = locked ? REFRESH_THRESHOLD_PX : pullDistance;
  if (height === 0) return null;
  const progress = Math.min(pullDistance / REFRESH_THRESHOLD_PX, 1);
  return (
    <div
      // No transition while actively dragging -- the indicator should
      // track the finger 1:1, not lag behind a CSS transition. Snapping
      // back (released under the threshold) or collapsing after a
      // finished refresh both animate.
      className={`flex items-center justify-center overflow-hidden ${
        dragging ? "" : "transition-[height] duration-200 ease-out"
      }`}
      style={{ height }}
      aria-hidden="true"
    >
      <ArrowClockwise
        size={20}
        className={locked ? "animate-spin text-ink-secondary" : "text-ink-secondary"}
        style={locked ? undefined : { transform: `rotate(${progress * 360}deg)` }}
      />
    </div>
  );
}
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `cd app/frontend && npx vitest run components/__tests__/PullToRefreshIndicator.test.tsx`
Expected: PASS (3 tests)

- [ ] **Step 5: Commit**

```bash
git add app/frontend/components/PullToRefreshIndicator.tsx app/frontend/components/__tests__/PullToRefreshIndicator.test.tsx
git commit -m "feat(frontend): add PullToRefreshIndicator component

Co-Authored-By: Claude Sonnet 5 <noreply@anthropic.com>"
```

---

### Task 3: Export a cache-invalidation helper from MatchUI.tsx

**Files:**
- Modify: `app/frontend/components/MatchUI.tsx` (near line 334, right after `setDashboardMatchesCache`)

- [ ] **Step 1: Add the export**

Find this existing function (around line 328-334):

```ts
function setDashboardMatchesCache(key: string, matches: Match[]): void {
  const now = Date.now();
  const ttlMs = matches.some((m) => isLiveOrImminent(m.kickoffIso, now))
    ? DASHBOARD_CACHE_LIVE_TTL_MS
    : DASHBOARD_CACHE_IDLE_TTL_MS;
  dashboardMatchesCache.set(key, { matches, fetchedAt: now, ttlMs });
}
```

Add immediately after it:

```ts
/** Forces the next load() run for `key` to skip the TTL short-circuit and
 * actually re-fetch -- used by DashboardPage's pull-to-refresh handler
 * (W238 follow-up). Unlike __resetDashboardMatchesCacheForTests below,
 * this is real app code, not a test-only escape hatch: it clears exactly
 * one day's entry rather than the whole cache, so a refresh triggered
 * while looking at "today" doesn't also throw away an unrelated day's
 * still-fresh cached list. */
export function invalidateDashboardMatchesCache(key: string): void {
  dashboardMatchesCache.delete(key);
}
```

- [ ] **Step 2: Typecheck**

Run: `cd app/frontend && npx tsc --noEmit`
Expected: no new errors (this is an additive export, nothing else references it yet).

- [ ] **Step 3: Commit**

```bash
git add app/frontend/components/MatchUI.tsx
git commit -m "feat(frontend): export invalidateDashboardMatchesCache for forced refresh

Co-Authored-By: Claude Sonnet 5 <noreply@anthropic.com>"
```

---

### Task 4: Wire the gesture into DashboardPage

**Files:**
- Modify: `app/frontend/components/MatchUI.tsx`
  - Import line (currently line 17)
  - New imports for the hook/component
  - `DashboardPage` function body (starts around line 1652)
  - `load()`'s three exit points (cache-hit return, success, catch) inside the same function
  - `DashboardPage`'s returned JSX (around line 1813-1814)
- Test: `app/frontend/components/__tests__/MatchUI.dashboardWindow.test.tsx`

- [ ] **Step 1: Add `act` to this test file's existing `@testing-library/react` import**

Find (line 12):

```ts
import { render, screen, waitFor } from "@testing-library/react";
```

Change to:

```ts
import { act, render, screen, waitFor } from "@testing-library/react";
```

- [ ] **Step 2: Write the failing integration test**

Add to the end of the existing `describe("Dashboard always shows the next 10 matches (date-grouped, not today-only)", ...)` block (this file already imports `getCachedRecommendationsBulk`, `getFixtures`, `fixture`, `DashboardPage`, `__resetDashboardMatchesCacheForTests` -- no other new imports needed):

```ts
  it("a completed pull-to-refresh forces a real re-fetch even though the client cache is still fresh", async () => {
    vi.mocked(getFixtures).mockResolvedValue([fixture("match-0", "2026-09-01T15:00:00Z")]);
    // getCachedRecommendationsBulk is left on its default auto-mock (resolves
    // undefined) rather than given an explicit .mockResolvedValue here --
    // this file doesn't reset that mock between tests/describe blocks, and
    // the only thing this test checks about it is call *count*, not what it
    // resolves to (resolveCachedRecommendations in MatchUI.tsx already
    // handles an undefined result via optional chaining, same as every
    // other test in this file that never configures it either).

    render(<DashboardPage />);
    await screen.findByText("match-0");
    // dashboardMatchesCache is now warm for today (W233) -- without the
    // pull-to-refresh bypass, nothing would trigger a second call here.
    expect(getFixtures).toHaveBeenCalledTimes(1);

    act(() => {
      window.dispatchEvent(
        Object.assign(new Event("touchstart", { bubbles: true }), { touches: [{ clientY: 0 }] })
      );
      window.dispatchEvent(
        Object.assign(new Event("touchmove", { bubbles: true, cancelable: true }), { touches: [{ clientY: 200 }] })
      );
      window.dispatchEvent(new Event("touchend", { bubbles: true }));
    });

    await waitFor(() => expect(getFixtures).toHaveBeenCalledTimes(2));
    // Proves the cache bypass specifically, not just an unrelated second
    // call: the fetch ran again despite nothing having expired.
    expect(getCachedRecommendationsBulk).toHaveBeenCalledTimes(2);
  });
```

- [ ] **Step 3: Run the test to verify it fails**

Run: `cd app/frontend && npx vitest run components/__tests__/MatchUI.dashboardWindow.test.tsx -t "pull-to-refresh"`
Expected: FAIL — `getFixtures` was called 1 time, not 2 (no gesture wiring exists yet, so the dispatched touch events do nothing).

- [ ] **Step 4: Add the new imports**

Change (current line 17):

```ts
import { type Dispatch, type SetStateAction, useEffect, useMemo, useState } from "react";
```

to:

```ts
import { type Dispatch, type SetStateAction, useCallback, useEffect, useMemo, useState } from "react";
```

Find (lines 55-60, the file's other `@/lib/...` and sibling-component imports):

```ts
import type { Bet, Fixture, MatchRecommendationOut } from "@/lib/types";
import { useSandboxAsOf } from "@/lib/useSandboxAsOf";
import { groupByDate, groupByLeague, sortMatches, LEAGUE_COUNTRY, LEAGUE_LABEL, type MatchSort } from "@/lib/dashboardMetrics";
import { AppShell } from "./AppShell";
import { DashboardRail } from "./DashboardRail";
import { LogBetModal } from "./LogBetModal";
```

Change to:

```ts
import type { Bet, Fixture, MatchRecommendationOut } from "@/lib/types";
import { useSandboxAsOf } from "@/lib/useSandboxAsOf";
import { usePullToRefresh } from "@/lib/usePullToRefresh";
import { groupByDate, groupByLeague, sortMatches, LEAGUE_COUNTRY, LEAGUE_LABEL, type MatchSort } from "@/lib/dashboardMetrics";
import { AppShell } from "./AppShell";
import { DashboardRail } from "./DashboardRail";
import { LogBetModal } from "./LogBetModal";
import { PullToRefreshIndicator } from "./PullToRefreshIndicator";
```

- [ ] **Step 5: Add `refreshing` state and the hook call to `DashboardPage`**

Find (around line 1673, end of `DashboardPage`'s existing `useState` declarations):

```ts
  const [railOpen, setRailOpen] = useState(false);

  useEffect(() => {
```

Change to:

```ts
  const [railOpen, setRailOpen] = useState(false);
  // W238: true while a pull-to-refresh-triggered reload is in flight --
  // read by usePullToRefresh below to know when to release its lock, and
  // reset to false at every exit point of load()'s own effect (cache hit,
  // success, error).
  const [refreshing, setRefreshing] = useState(false);
  const handlePullToRefresh = useCallback(() => {
    invalidateDashboardMatchesCache(dateString(asOf, sandboxMode));
    setRefreshing(true);
    setRetryTick((t) => t + 1);
  }, [asOf, sandboxMode]);
  const { pullDistance, locked, dragging } = usePullToRefresh({
    onRefresh: handlePullToRefresh,
    refreshing,
  });

  useEffect(() => {
```

- [ ] **Step 6: Reset `refreshing` at `load()`'s three exit points**

Find (around line 1695-1699, the cache-hit short-circuit):

```ts
      const cached = getDashboardMatchesCache(today);
      if (cached) {
        setMatches(cached);
        return;
      }
```

Change to:

```ts
      const cached = getDashboardMatchesCache(today);
      if (cached) {
        setMatches(cached);
        setRefreshing(false);
        return;
      }
```

Find (around line 1742-1746, the success path):

```ts
        setMatches(nearest);
        resolveCachedRecommendations(nearest).then((resolvedMatches) => {
          if (cancelled) return;
          setMatches(resolvedMatches);
          setDashboardMatchesCache(today, resolvedMatches);
        });
```

Change to:

```ts
        setMatches(nearest);
        resolveCachedRecommendations(nearest).then((resolvedMatches) => {
          if (cancelled) return;
          setMatches(resolvedMatches);
          setDashboardMatchesCache(today, resolvedMatches);
          setRefreshing(false);
        });
```

Find (around line 1747-1749, the catch path):

```ts
      } catch (err) {
        if (!cancelled) setError(err instanceof ApiError ? err.message : "Could not load fixtures.");
      }
```

Change to:

```ts
      } catch (err) {
        if (!cancelled) {
          setError(err instanceof ApiError ? err.message : "Could not load fixtures.");
          setRefreshing(false);
        }
      }
```

- [ ] **Step 7: Render the indicator**

Find (around line 1813-1814):

```tsx
        <div className="lg:-mt-8 lg:flex lg:items-start lg:gap-6">
          <div className="min-w-0 flex-1">
```

Change to:

```tsx
        <div className="lg:-mt-8 lg:flex lg:items-start lg:gap-6">
          <div className="min-w-0 flex-1">
            <PullToRefreshIndicator pullDistance={pullDistance} locked={locked} dragging={dragging} />
```

- [ ] **Step 8: Run the integration test to verify it passes**

Run: `cd app/frontend && npx vitest run components/__tests__/MatchUI.dashboardWindow.test.tsx`
Expected: PASS (every test in the file, including the new one)

- [ ] **Step 9: Run the full frontend suite and typecheck**

Run: `cd app/frontend && npx vitest run`
Expected: PASS, no new failures compared to the pre-change baseline.

Run: `cd app/frontend && npx tsc --noEmit`
Expected: no errors.

- [ ] **Step 10: Commit**

```bash
git add app/frontend/components/MatchUI.tsx app/frontend/components/__tests__/MatchUI.dashboardWindow.test.tsx
git commit -m "feat(frontend): wire pull-to-refresh into Daily Edges

Co-Authored-By: Claude Sonnet 5 <noreply@anthropic.com>"
```

---

## Self-review notes (for whoever executes this plan)

- **Spec coverage:** Section 3 (mechanism) → Task 1. Section 4 (visual indicator) → Task 2. Section 5 (wiring, cache invalidation) → Tasks 3-4. Section 6 (testing) → the test steps in Tasks 1, 2, and 4.
- **Why `usePullToRefresh` lives in `lib/`, not `MatchUI.tsx`:** per the design doc's own note that this was a call for the plan -- `lib/useSandboxAsOf.ts` is this codebase's existing precedent for a standalone hook file with no JSX, and `MatchUI.tsx` is already large enough that adding ~90 more lines there wasn't justified when a clean extraction was available.
- **Why no `"use client"` on `usePullToRefresh.ts`:** matches `lib/useSandboxAsOf.ts`'s existing precedent (hook-only files in this codebase don't carry the directive; it's only added to component files with JSX).
