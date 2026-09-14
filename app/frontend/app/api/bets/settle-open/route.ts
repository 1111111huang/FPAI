import { forwardToBackend } from "@/lib/backendProxy";

export async function POST() {
  const response = await forwardToBackend("/api/bets/settle-open", { method: "POST" });
  return new Response(await response.text(), { status: response.status });
}
