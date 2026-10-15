import { forwardToBackend } from "@/lib/backendProxy";

export async function POST(request: Request) {
  const body = await request.text();
  const response = await forwardToBackend("/api/push/subscribe", { method: "POST", body });
  return new Response(response.status === 204 ? null : await response.text(), { status: response.status });
}

export async function DELETE(request: Request) {
  const body = await request.text();
  const response = await forwardToBackend("/api/push/subscribe", { method: "DELETE", body });
  return new Response(response.status === 204 ? null : await response.text(), { status: response.status });
}
