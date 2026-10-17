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
