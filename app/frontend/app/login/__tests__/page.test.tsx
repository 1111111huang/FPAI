import { render, screen } from "@testing-library/react";
import { describe, expect, it, vi, beforeEach } from "vitest";
import LoginPage from "../page";

const mockUseSession = vi.fn();
const mockSignIn = vi.fn();
const mockUseSearchParams = vi.fn();

vi.mock("next-auth/react", () => ({
  useSession: () => mockUseSession(),
  signIn: (...args: unknown[]) => mockSignIn(...args),
}));
vi.mock("next/navigation", () => ({
  useSearchParams: () => mockUseSearchParams(),
}));

describe("LoginPage", () => {
  beforeEach(() => {
    mockUseSession.mockReset();
    mockSignIn.mockReset();
    mockUseSearchParams.mockReset();
    mockUseSearchParams.mockReturnValue(new URLSearchParams());
    mockUseSession.mockReturnValue({ status: "unauthenticated" });
    // The redirect-loop guard's counter lives in sessionStorage (must
    // survive a hard reload, so it can't be component state) -- reset
    // between tests so one test's redirect attempt can't count toward
    // another's loop-guard threshold.
    sessionStorage.clear();
  });

  it("passes the callbackUrl query param through to signIn instead of a hardcoded path", async () => {
    mockUseSearchParams.mockReturnValue(new URLSearchParams("callbackUrl=%2Fbets%2Fsomething"));
    render(<LoginPage />);
    screen.getByRole("button", { name: /continue with google/i }).click();
    expect(mockSignIn).toHaveBeenCalledWith("google", { callbackUrl: "/bets/something" });
  });

  it("falls back to /bets when there is no callbackUrl param", () => {
    render(<LoginPage />);
    screen.getByRole("button", { name: /continue with google/i }).click();
    expect(mockSignIn).toHaveBeenCalledWith("google", { callbackUrl: "/bets" });
  });

  it("shows an explanation when redirected back with ?error=AccessDenied", () => {
    mockUseSearchParams.mockReturnValue(new URLSearchParams("error=AccessDenied"));
    render(<LoginPage />);
    expect(screen.getByText(/not authorized|not on the allowlist|access denied/i)).toBeInTheDocument();
  });

  it("redirects to /bets immediately if already signed in", () => {
    // A hard navigation (window.location.href), not router.push -- a soft
    // client nav here can replay a stale pre-login middleware redirect
    // from Next.js's router cache (found live, 2026-09-15: a Link
    // auto-prefetch of /bets before sign-in cached a "redirect to /login"
    // result that a real post-login push then kept replaying forever).
    // jsdom's window.location doesn't actually navigate on assignment, so
    // swap in a plain writable stub for just this test and read back what
    // got assigned to it.
    const originalLocation = window.location;
    // @ts-expect-error -- jsdom's window.location isn't normally
    // reassignable; deleting it first is the standard escape hatch.
    delete window.location;
    window.location = { href: "" } as Location;

    mockUseSession.mockReturnValue({ status: "authenticated" });
    render(<LoginPage />);
    expect(window.location.href).toBe("/bets");

    window.location = originalLocation;
  });

  it("stops redirecting and shows a message after repeated redirect attempts in a short window", () => {
    // Simulates arriving here already having bounced past the loop-guard
    // threshold (e.g. middleware keeps disagreeing with the client's own
    // session state) -- a real prior incident (2026-09-15) hit exactly
    // this as a fast, silent, infinite reload loop before this guard
    // existed.
    sessionStorage.setItem("login-redirect-attempts", JSON.stringify({ count: 3, firstAt: Date.now() }));
    const originalLocation = window.location;
    // @ts-expect-error -- see the test above.
    delete window.location;
    window.location = { href: "" } as Location;

    mockUseSession.mockReturnValue({ status: "authenticated" });
    render(<LoginPage />);

    expect(window.location.href).toBe(""); // never redirected
    expect(screen.getByText(/having trouble signing in/i)).toBeInTheDocument();

    window.location = originalLocation;
  });
});
