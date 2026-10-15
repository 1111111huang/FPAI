import { render } from "@testing-library/react";
import { describe, expect, it, vi, beforeEach } from "vitest";
import { ServiceWorkerRegistration } from "../ServiceWorkerRegistration";

describe("ServiceWorkerRegistration", () => {
  beforeEach(() => {
    vi.restoreAllMocks();
  });

  it("registers the service worker on mount when supported", () => {
    const register = vi.fn().mockResolvedValue(undefined);
    Object.defineProperty(window.navigator, "serviceWorker", {
      value: { register },
      configurable: true,
    });
    render(<ServiceWorkerRegistration />);
    expect(register).toHaveBeenCalledWith("/sw.js");
  });

  it("does nothing when serviceWorker is unsupported", () => {
    Object.defineProperty(window.navigator, "serviceWorker", {
      value: undefined,
      configurable: true,
    });
    expect(() => render(<ServiceWorkerRegistration />)).not.toThrow();
  });
});
