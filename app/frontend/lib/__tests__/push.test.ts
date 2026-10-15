import { describe, expect, it, vi, beforeEach } from "vitest";
import { isPushSubscribed, subscribeToPush, unsubscribeFromPush } from "../push";

const mockSubscribe = vi.fn();
const mockGetSubscription = vi.fn();
const mockUnsubscribe = vi.fn();
const fakeFetch = vi.fn();

beforeEach(() => {
  mockSubscribe.mockReset();
  mockGetSubscription.mockReset();
  mockUnsubscribe.mockReset();
  fakeFetch.mockReset().mockResolvedValue({ ok: true });
  global.fetch = fakeFetch as unknown as typeof fetch;
  // vitest doesn't load .env.local (unlike `next dev`/`next build`), so the
  // VAPID public key the real app gets from env is unset here -- set a
  // fake one so subscribeToPush's "not configured" guard doesn't fire.
  process.env.NEXT_PUBLIC_VAPID_PUBLIC_KEY = "dGVzdC12YXBpZC1wdWJsaWMta2V5LTAxMjM0NTY3ODk";
  Object.defineProperty(window.navigator, "serviceWorker", {
    configurable: true,
    value: {
      ready: Promise.resolve({
        pushManager: { subscribe: mockSubscribe, getSubscription: mockGetSubscription },
      }),
    },
  });
});

describe("isPushSubscribed", () => {
  it("returns false when there is no active subscription", async () => {
    mockGetSubscription.mockResolvedValue(null);
    expect(await isPushSubscribed()).toBe(false);
  });

  it("returns true when a subscription exists", async () => {
    mockGetSubscription.mockResolvedValue({});
    expect(await isPushSubscribed()).toBe(true);
  });
});

describe("subscribeToPush", () => {
  it("subscribes via PushManager and posts the subscription to /api/push", async () => {
    mockSubscribe.mockResolvedValue({
      endpoint: "https://push.example/abc",
      toJSON: () => ({ keys: { p256dh: "key", auth: "auth" } }),
    });
    await subscribeToPush();
    expect(mockSubscribe).toHaveBeenCalledWith(expect.objectContaining({ userVisibleOnly: true }));
    expect(fakeFetch).toHaveBeenCalledWith(
      "/api/push",
      expect.objectContaining({ method: "POST" })
    );
    const body = JSON.parse(fakeFetch.mock.calls[0][1].body as string);
    expect(body).toEqual({ endpoint: "https://push.example/abc", p256dh: "key", auth: "auth" });
  });
});

describe("unsubscribeFromPush", () => {
  it("deletes the server-side subscription and unsubscribes locally", async () => {
    mockGetSubscription.mockResolvedValue({
      endpoint: "https://push.example/abc",
      unsubscribe: mockUnsubscribe,
    });
    await unsubscribeFromPush();
    expect(fakeFetch).toHaveBeenCalledWith(
      "/api/push",
      expect.objectContaining({ method: "DELETE" })
    );
    expect(mockUnsubscribe).toHaveBeenCalled();
  });

  it("does nothing when there is no active subscription", async () => {
    mockGetSubscription.mockResolvedValue(null);
    await unsubscribeFromPush();
    expect(fakeFetch).not.toHaveBeenCalled();
  });
});
