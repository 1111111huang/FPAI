// W210: coverage for forwardToBackend()'s two responsibilities -- reject
// unauthenticated callers, and attach the two proxy-only headers
// (X-User-Email, X-Internal-Secret) for an authenticated one.
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { getServerSession } from "next-auth";
import { forwardToBackend } from "./backendProxy";

vi.mock("next-auth", () => ({ getServerSession: vi.fn() }));

describe("forwardToBackend", () => {
  const originalFetch = global.fetch;

  beforeEach(() => {
    vi.mocked(getServerSession).mockReset();
    vi.stubEnv("INTERNAL_API_SECRET", "test-internal-secret");
    global.fetch = vi.fn().mockResolvedValue(new Response(null, { status: 200 }));
  });

  afterEach(() => {
    vi.unstubAllEnvs();
    global.fetch = originalFetch;
  });

  it("returns 401 without calling the backend when there is no session", async () => {
    vi.mocked(getServerSession).mockResolvedValue(null);

    const res = await forwardToBackend("/api/bets");

    expect(res.status).toBe(401);
    expect(global.fetch).not.toHaveBeenCalled();
  });

  it("returns 401 when the session has no user email", async () => {
    vi.mocked(getServerSession).mockResolvedValue({ user: {} } as never);

    const res = await forwardToBackend("/api/bets");

    expect(res.status).toBe(401);
    expect(global.fetch).not.toHaveBeenCalled();
  });

  it("attaches X-User-Email and X-Internal-Secret when a session is present", async () => {
    vi.mocked(getServerSession).mockResolvedValue({
      user: { email: "jane@gmail.com" },
    } as never);

    await forwardToBackend("/api/bets", { method: "GET" });

    expect(global.fetch).toHaveBeenCalledTimes(1);
    const [, init] = vi.mocked(global.fetch).mock.calls[0];
    const headers = new Headers(init!.headers);
    expect(headers.get("X-User-Email")).toBe("jane@gmail.com");
    expect(headers.get("X-Internal-Secret")).toBe("test-internal-secret");
  });

  it("forwards to the backend URL with the given path", async () => {
    vi.mocked(getServerSession).mockResolvedValue({
      user: { email: "jane@gmail.com" },
    } as never);

    await forwardToBackend("/api/bets/123");

    const [url] = vi.mocked(global.fetch).mock.calls[0];
    expect(String(url)).toContain("/api/bets/123");
  });
});
