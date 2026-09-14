// W210: Google-only, invite-only sign-in. ALLOWED_EMAILS is a plain
// comma-separated env var, not a database table -- deliberately the
// simplest possible allowlist for a small, manually-curated user list.
// Google does the real authentication; this callback is the only gate on
// top of it. JWT session strategy (not database sessions) -- no shared
// session store needed between this app and FastAPI, and FastAPI never
// needs to read it at all (see app/backend/auth_deps.py's own docstring).
import type { NextAuthOptions } from "next-auth";
import GoogleProvider from "next-auth/providers/google";

export function getAllowedEmails(): string[] {
  return (process.env.ALLOWED_EMAILS ?? "")
    .split(",")
    .map((e) => e.trim().toLowerCase())
    .filter(Boolean);
}

export const authOptions: NextAuthOptions = {
  providers: [
    GoogleProvider({
      clientId: process.env.GOOGLE_CLIENT_ID!,
      clientSecret: process.env.GOOGLE_CLIENT_SECRET!,
    }),
  ],
  session: { strategy: "jwt" },
  callbacks: {
    async signIn({ user }) {
      if (!user.email) return false;
      const allowed = getAllowedEmails();
      return allowed.includes(user.email.toLowerCase());
    },
  },
  pages: {
    signIn: "/login",
  },
};
