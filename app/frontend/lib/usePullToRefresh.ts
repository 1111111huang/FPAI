import { useEffect, useRef, useState } from "react";

// Exported so PullToRefreshIndicator.tsx can match the same lock-trigger
// distance in its own height/progress math -- previously duplicated as two
// separate constants, found in final review (risk: a future threshold
// tweak in only one file silently desyncing the indicator from the gesture).
export const REFRESH_THRESHOLD_PX = 70;
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
  const pullDistanceRef = useRef(0);

  useEffect(() => {
    if (!refreshing) {
      setPullDistance(0);
      setLocked(false);
    }
  }, [refreshing]);

  useEffect(() => {
    if (!isMobileViewport()) return;

    function handleTouchStart(e: TouchEvent): void {
      // Found in code review: without this guard, starting a new drag while
      // a previous pull-to-refresh is still locked/in flight (refreshing
      // still true) and clearing the threshold again would call onRefresh()
      // a second time concurrently -- not destructive, but a redundant
      // concurrent refresh the design never intended.
      if (refreshing || window.scrollY !== 0) return;
      armedRef.current = true;
      startYRef.current = e.touches[0].clientY;
      pullDistanceRef.current = 0;
      setDragging(true);
    }

    function handleTouchMove(e: TouchEvent): void {
      if (!armedRef.current) return;
      const deltaY = e.touches[0].clientY - startYRef.current;
      if (deltaY <= 0) {
        armedRef.current = false;
        setDragging(false);
        pullDistanceRef.current = 0;
        setPullDistance(0);
        return;
      }
      e.preventDefault();
      const next = Math.min(deltaY * PULL_RESISTANCE, MAX_PULL_PX);
      pullDistanceRef.current = next;
      setPullDistance(next);
    }

    function handleTouchEnd(): void {
      if (!armedRef.current) return;
      armedRef.current = false;
      setDragging(false);
      if (pullDistanceRef.current >= REFRESH_THRESHOLD_PX) {
        setLocked(true);
        setPullDistance(REFRESH_THRESHOLD_PX);
        onRefresh();
      } else {
        setPullDistance(0);
      }
    }

    window.addEventListener("touchstart", handleTouchStart, { passive: true });
    window.addEventListener("touchmove", handleTouchMove, { passive: false });
    window.addEventListener("touchend", handleTouchEnd, { passive: true });
    return () => {
      window.removeEventListener("touchstart", handleTouchStart);
      window.removeEventListener("touchmove", handleTouchMove);
      window.removeEventListener("touchend", handleTouchEnd);
    };
  }, [onRefresh, refreshing]);

  return { pullDistance, locked, dragging };
}
