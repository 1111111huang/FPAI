import { vi } from "vitest";
import "@testing-library/jest-dom/vitest";

// next/font/google relies on Next.js's own SWC/webpack compiler transform to
// work -- under Vitest (Vite's transform, not Next's build pipeline) the
// real import isn't callable at all ("Inter is not a function"). Standard
// Next.js testing guidance: stub it with a plain className/variable.
// Direct user spec (2026-09-17): Inter/IBM Plex Mono (app/layout.tsx)
// replaced the earlier Montserrat-for-the-wordmark-only setup (AppShell.tsx
// now just uses font-semibold, inheriting the new global Inter default --
// no separate next/font/google call of its own to mock anymore).
vi.mock("next/font/google", () => ({
  Inter: () => ({ className: "font-inter-mock", variable: "font-inter-mock-variable" }),
  IBM_Plex_Mono: () => ({ className: "font-plex-mono-mock", variable: "font-plex-mono-mock-variable" }),
}));

// AppShell (rendered by every page) now renders UserMenu, which calls
// next-auth/react's useSession -- that throws outside a <SessionProvider>.
// The real provider now lives in app/providers.tsx, but jsdom tests never go
// through Next's request/response cycle that provider relies on, so it's
// still mocked here. Global default so every existing test (none of which
// render one) keeps working unchanged; a test that cares about the
// signed-in UI overrides this with its own vi.mock("next-auth/react", ...)
// (see UserMenu.test.tsx).
vi.mock("next-auth/react", () => ({
  useSession: () => ({ data: null, status: "unauthenticated" }),
  signOut: vi.fn(),
}));

// AppShell (rendered by every page, directly or wrapped) also calls
// next/navigation's useRouter (89e5b1f, tab-click view-transition handler) --
// that throws outside a mounted App Router, same reason as useSession above.
// Global default so every test that mounts AppShell keeps working without
// its own per-file mock; a test that cares about a specific navigation
// method overrides this with its own vi.mock("next/navigation", ...) (see
// LogBetButton.test.tsx / app/login/__tests__/page.test.tsx).
//
// usePathname/useSearchParams are mocked here too -- a vi.mock factory
// replaces the *whole* module for a file, and MatchUI.tsx's LogBetButton
// calls both (no local mock of its own in e.g. MatchUI.test.tsx), so leaving
// them out would turn "no export defined" into a brand-new failure. Real
// next/navigation returns null for both when there's no mounted
// App Router (doesn't throw, unlike useRouter) -- matched here so files
// without their own override see the same behavior they always have.
vi.mock("next/navigation", () => ({
  useRouter: () => ({
    push: vi.fn(),
    replace: vi.fn(),
    back: vi.fn(),
    forward: vi.fn(),
    refresh: vi.fn(),
    prefetch: vi.fn(),
  }),
  usePathname: () => null,
  useSearchParams: () => null,
}));
