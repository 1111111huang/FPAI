// W210: gates only user-scoped pages -- the shared dashboard ("/") stays
// unauthenticated by design (Task decision: shared recommendations,
// private bet tracking only).
import { withAuth } from "next-auth/middleware";

export default withAuth({
  pages: { signIn: "/login" },
});

export const config = {
  matcher: ["/bets/:path*"],
};
