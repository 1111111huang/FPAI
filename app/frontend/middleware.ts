// W210: gates only user-scoped pages -- the shared dashboard ("/") stays
// unauthenticated by design (Task decision: shared recommendations,
// private bet tracking only).
import { withAuth } from "next-auth/middleware";

export default withAuth({
  pages: { signIn: "/login" },
  // Must match lib/auth.ts's own cookies.sessionToken.name override exactly
  // -- see that file's comment for why this is no longer the default
  // __Secure-next-auth.session-token.
  cookies: { sessionToken: { name: "next-auth.session-token" } },
  callbacks: {
    // TEMP DIAGNOSTIC (2026-09-15) -- middleware's own getToken() rejects a
    // session cookie that /api/auth/session (a separate, Node runtime)
    // accepts as valid, in production, consistently. Logging what
    // middleware's Edge runtime actually sees for the raw cookie vs what
    // decoding it produces, to tell a genuinely-missing/differently-scoped
    // cookie apart from a decode-only failure. Remove once root-caused.
    authorized({ req, token }) {
      const raw = req.cookies.get("next-auth.session-token");
      console.log("[middleware-debug]", {
        hasCookie: !!raw,
        cookieLen: raw?.value?.length ?? 0,
        tokenPresent: !!token,
        tokenEmail: token?.email,
      });
      return !!token;
    },
  },
});

export const config = {
  matcher: ["/bets/:path*"],
};
