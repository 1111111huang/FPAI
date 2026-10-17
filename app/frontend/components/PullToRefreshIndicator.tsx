"use client";

import { ArrowClockwise } from "@phosphor-icons/react";

import { REFRESH_THRESHOLD_PX } from "@/lib/usePullToRefresh";

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
