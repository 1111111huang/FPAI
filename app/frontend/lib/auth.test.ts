// W210: coverage for the invite-only allowlist gate -- ALLOWED_EMAILS
// parsing (getAllowedEmails, exported for this test) and the signIn
// callback that actually enforces it.
import { afterEach, describe, expect, it, vi } from "vitest";
import { authOptions, getAllowedEmails } from "./auth";
import type { User } from "next-auth";

afterEach(() => {
  vi.unstubAllEnvs();
});

describe("getAllowedEmails", () => {
  it("parses a comma-separated list, trimmed and lowercased", () => {
    vi.stubEnv("ALLOWED_EMAILS", " Jane@Gmail.com, bob@example.com ,");
    expect(getAllowedEmails()).toEqual(["jane@gmail.com", "bob@example.com"]);
  });

  it("is an empty list when unset", () => {
    vi.stubEnv("ALLOWED_EMAILS", "");
    expect(getAllowedEmails()).toEqual([]);
  });
});

describe("authOptions.callbacks.signIn", () => {
  const signIn = authOptions.callbacks!.signIn!;

  function user(email: string | null): User {
    return { id: "1", email, name: null, image: null } as User;
  }

  it("accepts an email on the allowlist", async () => {
    vi.stubEnv("ALLOWED_EMAILS", "fpai.deploy@gmail.com");
    await expect(signIn({ user: user("fpai.deploy@gmail.com") } as never)).resolves.toBe(true);
  });

  it("rejects an email not on the allowlist", async () => {
    vi.stubEnv("ALLOWED_EMAILS", "fpai.deploy@gmail.com");
    await expect(signIn({ user: user("intruder@gmail.com") } as never)).resolves.toBe(false);
  });

  it("is case-insensitive", async () => {
    vi.stubEnv("ALLOWED_EMAILS", "fpai.deploy@gmail.com");
    await expect(signIn({ user: user("FPAI.Deploy@Gmail.com") } as never)).resolves.toBe(true);
  });

  it("rejects a sign-in with no email at all", async () => {
    vi.stubEnv("ALLOWED_EMAILS", "fpai.deploy@gmail.com");
    await expect(signIn({ user: user(null) } as never)).resolves.toBe(false);
  });
});
