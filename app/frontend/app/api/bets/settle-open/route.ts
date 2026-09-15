import { NextRequest } from "next/server";
import { forwardToBackend } from "@/lib/backendProxy";

export async function POST(request: NextRequest) {
  // Forwards ?blocking=false through untouched when the caller passes it
  // (BetTracker.tsx's automatic settle-on-page-load, W214) -- omitted
  // entirely for the explicit "Settle open bets" button, which keeps the
  // backend's own default blocking=true (a deliberate user-triggered
  // wait). See main.py's settle_open() and football_data_client.py's
  // RateLimitWouldBlock for why this distinction exists.
  const blocking = request.nextUrl.searchParams.get("blocking");
  const path = blocking !== null ? `/api/bets/settle-open?blocking=${blocking}` : "/api/bets/settle-open";
  const response = await forwardToBackend(path, { method: "POST" });
  return new Response(await response.text(), { status: response.status });
}
