// Deliberately does NOT use the global next-auth/react mock from
// vitest.setup.ts -- this is the one test that exercises the REAL
// SessionProvider wiring, to catch a regression of the exact bug where
// AppShell/UserMenu crashed every page because nothing wrapped the app
// in a SessionProvider. Every other test correctly uses the mock for
// convenience; this one exists specifically so that convenience can't
// hide a real wiring break again.
import { render } from "@testing-library/react";
import { describe, expect, it, vi } from "vitest";
import { Providers } from "../providers";

vi.unmock("next-auth/react"); // opt out of vitest.setup.ts's global mock for this file only

describe("Providers", () => {
  it("wraps children in a real SessionProvider without throwing", () => {
    // next-auth's SessionProvider fetches /api/auth/session on mount --
    // stub fetch so this stays a pure wiring check, not a network test.
    vi.stubGlobal(
      "fetch",
      vi.fn().mockResolvedValue({
        ok: true,
        json: async () => ({}),
      })
    );
    expect(() =>
      render(
        <Providers>
          <div>child</div>
        </Providers>
      )
    ).not.toThrow();
    vi.unstubAllGlobals();
  });
});
