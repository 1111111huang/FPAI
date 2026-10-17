import { describe, expect, it, vi, beforeEach, afterEach } from "vitest";
import { useState } from "react";
import { act, renderHook } from "@testing-library/react";
import { usePullToRefresh } from "../usePullToRefresh";

// BUG-088 follow-up (found live 2026-10-17): transform-gpu alone didn't
// stop the bottom tab bar/header from detaching during a fast scroll on
// iOS -- the real fix is a dedicated scroll container (<main>) instead of
// document-level body scroll, same architecture desktop already used.
// usePullToRefresh's own "are we at the top" gate now checks that
// container's scrollTop instead of window.scrollY, which no longer moves
// at all once <main> is the thing scrolling.
function setScrollTop(value: number): void {
  let main = document.querySelector("main");
  if (!main) {
    main = document.createElement("main");
    document.body.appendChild(main);
  }
  Object.defineProperty(main, "scrollTop", { value, configurable: true });
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
    setScrollTop(0);
  });

  afterEach(() => {
    // @ts-expect-error -- deleting a test-only stub, not a real browser API removal
    delete window.matchMedia;
    document.querySelector("main")?.remove();
  });

  it("a drag starting at scrollY > 0 never arms -- no refresh on release past the threshold", () => {
    setScrollTop(50);
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

  it("does not re-arm a new drag while a previous refresh is still in flight (refreshing true)", () => {
    const onRefresh = vi.fn();
    const { rerender } = renderHook(
      ({ refreshing }) => usePullToRefresh({ onRefresh, refreshing }),
      { initialProps: { refreshing: false } },
    );

    dispatchTouch("touchstart", 0);
    dispatchTouch("touchmove", 200);
    dispatchTouch("touchend");
    expect(onRefresh).toHaveBeenCalledTimes(1);

    // Caller sets refreshing true (as DashboardPage's handlePullToRefresh
    // does) while the triggered reload is still in flight.
    rerender({ refreshing: true });

    // A second drag starts and clears the threshold again while still
    // refreshing -- must not re-arm, so onRefresh must not fire a second
    // time.
    dispatchTouch("touchstart", 0);
    dispatchTouch("touchmove", 200);
    dispatchTouch("touchend");

    expect(onRefresh).toHaveBeenCalledTimes(1);
  });

  it("does not double-fire onRefresh when the caller's onRefresh triggers a state update on the same render (regression)", () => {
    const callCount = { current: 0 };
    function useHarness() {
      const [, setTick] = useState(0);
      const onRefresh = () => {
        callCount.current += 1;
        setTick((t) => t + 1); // a realistic caller: onRefresh updates its own component state
      };
      return usePullToRefresh({ onRefresh, refreshing: false });
    }

    renderHook(() => useHarness());

    dispatchTouch("touchstart", 0);
    dispatchTouch("touchmove", 200);
    dispatchTouch("touchend");

    expect(callCount.current).toBe(1);
  });

  it("a bare tap after a successful refresh does not spuriously call onRefresh again (regression)", () => {
    const onRefresh = vi.fn();
    const { rerender } = renderHook(
      ({ refreshing }) => usePullToRefresh({ onRefresh, refreshing }),
      { initialProps: { refreshing: false } },
    );

    // A full pull-to-refresh, same as the other threshold-clearing test.
    dispatchTouch("touchstart", 0);
    dispatchTouch("touchmove", 200);
    dispatchTouch("touchend");
    expect(onRefresh).toHaveBeenCalledTimes(1);

    // Caller's refresh finishes.
    rerender({ refreshing: true });
    rerender({ refreshing: false });

    // A later bare tap at scrollY === 0 -- touchstart immediately followed
    // by touchend, no intervening touchmove at all (an ordinary tap, not a
    // drag). Must not re-trigger onRefresh just because a stale pull
    // distance was left over from the previous successful gesture.
    dispatchTouch("touchstart", 0);
    dispatchTouch("touchend");

    expect(onRefresh).toHaveBeenCalledTimes(1);
  });
});
