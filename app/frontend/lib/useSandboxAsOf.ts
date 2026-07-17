import { useEffect, useState } from "react";
import { getSandboxStatus } from "./api";

export type SandboxAsOf = {
  /** The sandbox as_of date (UTC midnight) when sandbox mode is active,
   * the real browser Date() otherwise. */
  asOf: Date;
  /** Whether asOf came from the sandbox endpoint (UTC-anchored calendar
   * date) or is the real browser clock (a real instant) -- consumers that
   * need to read asOf via Date getters (not just .toISOString()) must
   * branch on this: UTC getters for the sandbox case, local getters for
   * the real-clock case, since the two are not interchangeable (see
   * MatchCard's formatDay()). */
  sandboxMode: boolean;
};

/**
 * W30: resolves "today" for date-window purposes -- the sandbox as_of date
 * when sandbox mode is active, the real browser Date() otherwise. Fetches
 * the backend's GET /api/sandbox/status once per mount. Works for both a
 * human clicking through the real UI and Playwright-driven automated
 * checks, since both just read this hook's returned Date the same way.
 */
export function useSandboxAsOf(): SandboxAsOf {
  const [state, setState] = useState<SandboxAsOf>(() => ({ asOf: new Date(), sandboxMode: false }));

  useEffect(() => {
    let cancelled = false;
    getSandboxStatus()
      .then((status) => {
        if (!cancelled && status.sandbox_mode && status.as_of) {
          // Every consumer re-serializes this via .toISOString(), which is
          // always UTC -- constructing UTC midnight here (a bare
          // YYYY-MM-DD string parses as UTC midnight per ECMA-262) keeps
          // that round trip exact regardless of the browser's local
          // timezone. Appending a local-time suffix here would shift the
          // resulting date by a day in positive-UTC-offset timezones once
          // read back through .toISOString().
          setState({ asOf: new Date(status.as_of), sandboxMode: true });
        }
      })
      .catch(() => {
        // sandbox status endpoint unreachable/erroring -- fall back to the
        // real browser clock rather than blocking the page.
      });
    return () => {
      cancelled = true;
    };
  }, []);

  return state;
}
