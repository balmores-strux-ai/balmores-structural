/** Production: same-origin /api/backend/* (rewritten to BACKEND_PROXY_URL). Override with NEXT_PUBLIC_API_URL if needed. */
function apiBase(): string {
  const explicit = process.env.NEXT_PUBLIC_API_URL;
  if (explicit !== undefined && explicit !== "") return explicit.replace(/\/$/, "");
  if (process.env.NODE_ENV === "production") return "/api/backend";
  return "http://localhost:8000";
}

export const API_URL = apiBase();

export async function sendChat(payload: Record<string, unknown>) {
  const res = await fetch(`${API_URL}/chat`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
  });
  if (!res.ok) throw new Error("Failed to send chat request");
  return res.json();
}

export async function queueVerify(projectId: string) {
  const res = await fetch(`${API_URL}/verify`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ project_id: projectId }),
  });
  if (!res.ok) throw new Error("Failed to queue verification");
  return res.json();
}
