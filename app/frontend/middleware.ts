// W210: gates only user-scoped pages -- the shared dashboard ("/") stays
// unauthenticated by design (Task decision: shared recommendations,
// private bet tracking only).
export { default } from "next-auth/middleware";

export const config = {
  matcher: ["/bets/:path*"],
};
