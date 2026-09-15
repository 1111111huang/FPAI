import { forwardToBackend } from "@/lib/backendProxy";

export async function DELETE(request: Request, { params }: { params: { id: string } }) {
  const response = await forwardToBackend(`/api/bets/${params.id}`, { method: "DELETE" });
  return new Response(response.status === 204 ? null : await response.text(), { status: response.status });
}

export async function PATCH(request: Request, { params }: { params: { id: string } }) {
  const body = await request.text();
  const response = await forwardToBackend(`/api/bets/${params.id}`, { method: "PATCH", body });
  return new Response(await response.text(), { status: response.status });
}
