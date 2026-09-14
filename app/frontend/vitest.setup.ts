import { vi } from "vitest";
import "@testing-library/jest-dom/vitest";

// next/font/google relies on Next.js's own SWC/webpack compiler transform to
// work -- under Vitest (Vite's transform, not Next's build pipeline) the
// real import isn't callable at all ("Montserrat is not a function").
// Standard Next.js testing guidance: stub it with a plain className.
vi.mock("next/font/google", () => ({
  Montserrat: () => ({ className: "font-montserrat-mock" }),
}));

// AppShell (rendered by every page) now renders UserMenu, which calls
// next-auth/react's useSession -- that throws outside a <SessionProvider>.
// Global default so every existing test (none of which render one) keeps
// working unchanged; a test that cares about the signed-in UI overrides
// this with its own vi.mock("next-auth/react", ...) (see UserMenu.test.tsx).
vi.mock("next-auth/react", () => ({
  useSession: () => ({ data: null, status: "unauthenticated" }),
  signOut: vi.fn(),
}));
