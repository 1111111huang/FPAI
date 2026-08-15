import { vi } from "vitest";
import "@testing-library/jest-dom/vitest";

// next/font/google relies on Next.js's own SWC/webpack compiler transform to
// work -- under Vitest (Vite's transform, not Next's build pipeline) the
// real import isn't callable at all ("Montserrat is not a function").
// Standard Next.js testing guidance: stub it with a plain className.
vi.mock("next/font/google", () => ({
  Montserrat: () => ({ className: "font-montserrat-mock" }),
}));
