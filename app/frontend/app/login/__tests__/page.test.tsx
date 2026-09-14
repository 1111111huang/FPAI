import { render, screen } from "@testing-library/react";
import { describe, expect, it, vi, beforeEach } from "vitest";
import LoginPage from "../page";

const mockUseSession = vi.fn();
const mockSignIn = vi.fn();
const mockUseSearchParams = vi.fn();
const mockPush = vi.fn();

vi.mock("next-auth/react", () => ({
  useSession: () => mockUseSession(),
  signIn: (...args: unknown[]) => mockSignIn(...args),
}));
vi.mock("next/navigation", () => ({
  useSearchParams: () => mockUseSearchParams(),
  useRouter: () => ({ push: mockPush }),
}));

describe("LoginPage", () => {
  beforeEach(() => {
    mockUseSession.mockReset();
    mockSignIn.mockReset();
    mockUseSearchParams.mockReset();
    mockPush.mockReset();
    mockUseSearchParams.mockReturnValue(new URLSearchParams());
    mockUseSession.mockReturnValue({ status: "unauthenticated" });
  });

  it("passes the callbackUrl query param through to signIn instead of a hardcoded path", async () => {
    mockUseSearchParams.mockReturnValue(new URLSearchParams("callbackUrl=%2Fbets%2Fsomething"));
    render(<LoginPage />);
    screen.getByRole("button", { name: /sign in with google/i }).click();
    expect(mockSignIn).toHaveBeenCalledWith("google", { callbackUrl: "/bets/something" });
  });

  it("falls back to /bets when there is no callbackUrl param", () => {
    render(<LoginPage />);
    screen.getByRole("button", { name: /sign in with google/i }).click();
    expect(mockSignIn).toHaveBeenCalledWith("google", { callbackUrl: "/bets" });
  });

  it("shows an explanation when redirected back with ?error=AccessDenied", () => {
    mockUseSearchParams.mockReturnValue(new URLSearchParams("error=AccessDenied"));
    render(<LoginPage />);
    expect(screen.getByText(/not authorized|not on the allowlist|access denied/i)).toBeInTheDocument();
  });

  it("redirects to /bets immediately if already signed in", () => {
    mockUseSession.mockReturnValue({ status: "authenticated" });
    render(<LoginPage />);
    expect(mockPush).toHaveBeenCalledWith("/bets");
  });
});
