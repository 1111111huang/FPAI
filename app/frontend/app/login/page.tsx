"use client";

import { signIn } from "next-auth/react";

export default function LoginPage() {
  return (
    <div style={{ display: "flex", flexDirection: "column", alignItems: "center", marginTop: "4rem", gap: "1rem" }}>
      <h1>Sign in</h1>
      <button onClick={() => signIn("google", { callbackUrl: "/bets" })}>
        Sign in with Google
      </button>
    </div>
  );
}
