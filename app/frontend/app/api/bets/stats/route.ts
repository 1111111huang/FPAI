import { forwardToBackend } from "@/lib/backendProxy";

export async function GET() {
  const response = await forwardToBackend("/api/bets/stats");
  return new Response(await response.text(), { status: response.status });
}
