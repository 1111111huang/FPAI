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
  // Production bug (2026-09-15): Edge Middleware's own getToken() never
  // sees the session cookie at all (confirmed live via a temp debug log --
  // req.cookies missing it entirely), even on a genuine hard top-level
  // reload, while /api/auth/session (a Node-runtime route, same request)
  // reads it fine. The only concrete difference between those two: the
  // default __Secure- prefixed cookie name (real over https, which this
  // deployment is). Testing/working around that specifically by pinning a
  // plain, unprefixed name -- still `secure: true` at the attribute level,
  // just not name-prefixed. If this fixes it, the prefix itself was the
  // problem (a known class of edge/CDN-layer quirk with __Secure-/__Host-
  // prefixed cookies); middleware.ts's own withAuth() call must use this
  // same name or it'll go on looking for the old one.
  cookies: {
    sessionToken: {
      name: "next-auth.session-token",
      options: { httpOnly: true, sameSite: "lax", path: "/", secure: true },
    },
  },
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
