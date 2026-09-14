import { forwardToBackend } from "@/lib/backendProxy";

export async function POST(request: Request) {
  const body = await request.text();
  const response = await forwardToBackend("/api/bets/from-recommendation", { method: "POST", body });
  return new Response(await response.text(), { status: response.status });
}
