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
