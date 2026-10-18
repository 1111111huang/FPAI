import { describe, expect, it, vi, beforeEach } from "vitest";

const mockGetSession = vi.fn();
vi.mock("next-auth/react", () => ({
  getSession: (...args: unknown[]) => mockGetSession(...args),
}));

import { ApiError } from "../api";

describe("ApiError", () => {
  beforeEach(() => {
    mockGetSession.mockReset();
  });

  // BUG-090 (found live 2026-10-17): backendProxy.ts's getServerSession()
  // re-reads the real session cookie fresh on every /api/bets/* call, but
  // the client-side useSession() hook (UserMenu and every other consumer)
  // only reflects whatever it last fetched -- so a session that's actually
  // expired can still show "signed in" in the sidebar while the same
  // request 401s against that same expired session (confirmed live: the
  // sidebar showed the signed-in email/Sign out while "Confirm bet" failed
  // with "Failed to log bet (401)" and the Bets page itself said "Your
  // session expired"). Constructing a 401 ApiError now forces every
  // useSession() consumer app-wide to refetch and catch up with the real
  // state immediately, instead of waiting for NextAuth's own next refetch.
  it("a 401 ApiError triggers a session refetch", () => {
    new ApiError("Failed to log bet (401)", 401);
    expect(mockGetSession).toHaveBeenCalledTimes(1);
  });

  it("a non-401 ApiError does not trigger a session refetch", () => {
    new ApiError("Failed to load bets (500)", 500);
    expect(mockGetSession).not.toHaveBeenCalled();
  });

  it("an ApiError with no status does not trigger a session refetch", () => {
    new ApiError("Network error");
    expect(mockGetSession).not.toHaveBeenCalled();
  });
});
