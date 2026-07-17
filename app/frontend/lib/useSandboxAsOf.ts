import { useEffect, useState } from "react";
import { getSandboxStatus } from "./api";

/**
 * W30: resolves "today" for date-window purposes -- the sandbox as_of date
 * when sandbox mode is active, the real browser Date() otherwise. Fetches
 * the backend's GET /api/sandbox/status once per mount. Works for both a
 * human clicking through the real UI and Playwright-driven automated
 * checks, since both just read this hook's returned Date the same way.
 */
export function useSandboxAsOf(): Date {
  const [asOf, setAsOf] = useState<Date>(() => new Date());

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
          setAsOf(new Date(status.as_of));
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

  return asOf;
}
