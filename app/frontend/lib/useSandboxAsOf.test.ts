import { describe, expect, it, vi, beforeEach } from "vitest";
import { renderHook, waitFor } from "@testing-library/react";
import { useSandboxAsOf } from "./useSandboxAsOf";
import { getSandboxStatus } from "./api";

vi.mock("./api", () => ({ getSandboxStatus: vi.fn() }));

describe("useSandboxAsOf", () => {
  beforeEach(() => {
    vi.mocked(getSandboxStatus).mockReset();
  });

  it("stays on the real clock when sandbox mode is inactive", async () => {
    vi.mocked(getSandboxStatus).mockResolvedValue({ sandbox_mode: false, as_of: null });
    const before = new Date();

    const { result } = renderHook(() => useSandboxAsOf());

    await waitFor(() => expect(getSandboxStatus).toHaveBeenCalled());
    const after = new Date();
    expect(result.current.asOf.getTime()).toBeGreaterThanOrEqual(before.getTime());
    expect(result.current.asOf.getTime()).toBeLessThanOrEqual(after.getTime());
    expect(result.current.sandboxMode).toBe(false);
  });

  it("switches to the sandbox as_of date once fetched", async () => {
    vi.mocked(getSandboxStatus).mockResolvedValue({ sandbox_mode: true, as_of: "2026-03-01" });

    const { result } = renderHook(() => useSandboxAsOf());

    await waitFor(() => expect(result.current.asOf.toISOString().slice(0, 10)).toBe("2026-03-01"));
    expect(result.current.sandboxMode).toBe(true);
  });

  it("falls back to the real clock if the sandbox status call fails", async () => {
    vi.mocked(getSandboxStatus).mockRejectedValue(new Error("network error"));

    const { result } = renderHook(() => useSandboxAsOf());

    await waitFor(() => expect(getSandboxStatus).toHaveBeenCalled());
    expect(result.current.asOf).toBeInstanceOf(Date);
    expect(result.current.sandboxMode).toBe(false);
  });
});
