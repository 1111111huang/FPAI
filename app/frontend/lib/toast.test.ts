import { describe, expect, it, vi, beforeEach, afterEach } from "vitest";
import { showToast } from "./toast";

describe("showToast", () => {
  beforeEach(() => {
    vi.useFakeTimers();
    document.body.innerHTML = "";
  });

  afterEach(() => {
    vi.useRealTimers();
  });

  it("shows the message then removes itself", () => {
    showToast("Bet logged");
    expect(document.body.textContent).toContain("Bet logged");

    vi.runAllTimers();
    expect(document.body.textContent).not.toContain("Bet logged");
  });

  it("replaces an in-flight toast instead of stacking", () => {
    showToast("First");
    showToast("Second");
    expect(document.body.querySelectorAll("div").length).toBe(1);
    expect(document.body.textContent).toContain("Second");
  });
});
