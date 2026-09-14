import { forwardToBackend } from "@/lib/backendProxy";

export async function GET() {
  const response = await forwardToBackend("/api/bets");
  return new Response(await response.text(), { status: response.status });
}

export async function POST(request: Request) {
  const body = await request.text();
  const response = await forwardToBackend("/api/bets/manual", { method: "POST", body });
  return new Response(await response.text(), { status: response.status });
}
