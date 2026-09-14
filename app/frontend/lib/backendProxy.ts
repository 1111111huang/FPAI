// W210: every proxy route in app/api/bets/* goes through this so the
// INTERNAL_API_SECRET header (server-only, never sent to the browser) is
// attached exactly once, in exactly one place.
import { getServerSession } from "next-auth";
import { authOptions } from "@/lib/auth";

const API_BASE = process.env.NEXT_PUBLIC_API_BASE_URL ?? "http://localhost:8000";

export async function forwardToBackend(path: string, init: RequestInit = {}): Promise<Response> {
  const session = await getServerSession(authOptions);
  if (!session?.user?.email) {
    return new Response(JSON.stringify({ detail: "Unauthorized" }), { status: 401 });
  }
  const headers = new Headers(init.headers);
  headers.set("X-User-Email", session.user.email);
  headers.set("X-Internal-Secret", process.env.INTERNAL_API_SECRET!);
  if (init.body) headers.set("Content-Type", "application/json");
  return fetch(`${API_BASE}${path}`, { ...init, headers });
}
