"use client";

import { signIn } from "next-auth/react";

export default function LoginPage() {
  return (
    <div className="flex flex-col items-center mt-16 gap-4">
      <h1 className="text-lg font-medium text-ink">Sign in</h1>
      <button
        type="button"
        onClick={() => signIn("google", { callbackUrl: "/bets" })}
        className="rounded-md border border-accent px-4 py-2 text-sm font-medium text-accent hover:bg-accent/10"
      >
        Sign in with Google
      </button>
    </div>
  );
}
