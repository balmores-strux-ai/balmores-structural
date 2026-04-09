import { coerceRawPredictions, unwrapDetailedResults } from "./coercePredictions";

/** Production: same-origin /api/backend/* (rewritten to BACKEND_PROXY_URL). Override with NEXT_PUBLIC_API_URL if needed. */
function apiBase(): string {
  const explicit = process.env.NEXT_PUBLIC_API_URL;
  if (explicit !== undefined && explicit !== "") return explicit.replace(/\/$/, "");
  if (process.env.NODE_ENV === "production") return "/api/backend";
  return "http://localhost:8000";
}

export const API_URL = apiBase();

function authHeaders(): Record<string, string> {
  const k = process.env.NEXT_PUBLIC_API_KEY;
  if (typeof k === "string" && k.trim()) {
    return { "X-API-Key": k.trim() };
  }
  return {};
}

function jsonHeaders(): Record<string, string> {
  return { "Content-Type": "application/json", ...authHeaders() };
}

export type ChatResponsePayload = {
  project_id: string;
  messages: { role: string; content: string }[];
  assumptions?: string[];
  geometry?: unknown;
  state?: {
    span_x_m?: number;
    span_y_m?: number;
    stories?: number;
    bays_x?: number;
    bays_y?: number;
  };
  result_cards?: unknown[];
  detailed_results?: unknown;
  recommendations?: unknown[];
  follow_up_questions?: string[];
  confidence?: string;
  [key: string]: unknown;
};

/** Coerce nested prediction numbers so the UI never sees stringified floats as "not a number". */
function normalizeChatComplete(complete: ChatResponsePayload): ChatResponsePayload {
  const dr = unwrapDetailedResults(complete);
  if (dr && "raw_predictions" in dr) {
    dr.raw_predictions = coerceRawPredictions(dr.raw_predictions);
  }
  return complete;
}

async function parseError(res: Response): Promise<string> {
  try {
    const j = (await res.json()) as { error?: { message?: string }; request_id?: string };
    const m = j?.error?.message;
    const rid = j?.request_id ? ` (ref: ${j.request_id})` : "";
    return (m || `Request failed (${res.status})`) + rid;
  } catch {
    return `Request failed (${res.status})`;
  }
}

function sleep(ms: number): Promise<void> {
  return new Promise((r) => setTimeout(r, ms));
}

async function fetchWithRetry(
  url: string,
  init: RequestInit,
  opts?: { retries?: number; retryOn?: number[] },
): Promise<Response> {
  const retries = opts?.retries ?? 2;
  const retryOn = opts?.retryOn ?? [502, 503, 504];
  let last: Response | null = null;
  for (let attempt = 0; attempt <= retries; attempt++) {
    const res = await fetch(url, init);
    last = res;
    if (res.ok || !retryOn.includes(res.status) || attempt === retries) return res;
    await sleep(300 * 2 ** attempt);
  }
  return last!;
}

export async function sendChat(payload: Record<string, unknown>, opts?: { signal?: AbortSignal }) {
  const res = await fetchWithRetry(
    `${API_URL}/chat`,
    {
      method: "POST",
      headers: jsonHeaders(),
      body: JSON.stringify(payload),
      signal: opts?.signal,
    },
    { retries: 2 },
  );
  if (!res.ok) throw new Error(await parseError(res));
  const data = (await res.json()) as ChatResponsePayload;
  return normalizeChatComplete(data);
}

/**
 * NDJSON stream: meta → deltas → complete (full ChatResponse JSON).
 */
export async function sendChatStream(
  payload: Record<string, unknown>,
  opts: {
    signal?: AbortSignal;
    onDelta?: (accumulated: string) => void;
    onMeta?: (projectId: string) => void;
  },
): Promise<ChatResponsePayload> {
  const res = await fetch(`${API_URL}/chat/stream`, {
    method: "POST",
    headers: jsonHeaders(),
    body: JSON.stringify(payload),
    signal: opts.signal,
  });
  if (!res.ok) {
    if (res.status === 404) {
      const data = (await sendChat(payload, opts)) as ChatResponsePayload;
      const t = String(data.messages?.[0]?.content ?? "");
      if (t) opts.onDelta?.(t);
      return data;
    }
    throw new Error(await parseError(res));
  }
  const reader = res.body?.getReader();
  if (!reader) throw new Error("No response body");
  const decoder = new TextDecoder();
  let buffer = "";
  let accumulated = "";
  let complete: ChatResponsePayload | null = null;
  while (true) {
    const { done, value } = await reader.read();
    if (done) break;
    buffer += decoder.decode(value, { stream: true });
    const lines = buffer.split("\n");
    buffer = lines.pop() ?? "";
    for (const line of lines) {
      const trimmed = line.trim();
      if (!trimmed) continue;
      let obj: {
        type: string;
        project_id?: string;
        text?: string;
        data?: ChatResponsePayload;
      };
      try {
        obj = JSON.parse(trimmed) as typeof obj;
      } catch {
        continue;
      }
      if (obj.type === "meta" && obj.project_id) opts.onMeta?.(obj.project_id);
      if (obj.type === "delta" && typeof obj.text === "string") {
        accumulated += obj.text;
        opts.onDelta?.(accumulated);
      }
      if (obj.type === "complete" && obj.data) complete = obj.data;
    }
  }
  if (buffer.trim()) {
    try {
      const obj = JSON.parse(buffer.trim()) as { type: string; data?: ChatResponsePayload };
      if (obj.type === "complete" && obj.data) complete = obj.data;
    } catch {
      /* ignore trailing partial */
    }
  }
  if (!complete) throw new Error("Stream ended without complete payload");
  return normalizeChatComplete(complete);
}

export async function queueVerify(projectId: string) {
  const res = await fetchWithRetry(
    `${API_URL}/verify`,
    {
      method: "POST",
      headers: jsonHeaders(),
      body: JSON.stringify({ project_id: projectId }),
    },
    { retries: 2 },
  );
  if (!res.ok) throw new Error(await parseError(res));
  return res.json();
}

/** 3D frame preview payload (chat or FEA). */
export type ViewerGeometry = {
  nodes: { id: string; x: number; y: number; z: number }[];
  members: { id: string; start: string; end: string; kind: string }[];
  meta?: Record<string, unknown>;
};

/** PyNite parametric frame FEA (matches backend FeaBuildingRequest). */
export type FeaBuildingRequest = {
  bays_x?: number;
  bays_y?: number;
  stories?: number;
  span_x_m?: number;
  span_y_m?: number;
  bottom_story_height_m?: number;
  story_height_m?: number;
  floor_load_kpa?: number;
  two_way_fraction?: number;
  elastic_modulus_gpa?: number;
  poisson_ratio?: number;
  shear_modulus_gpa?: number | null;
  beam_width_m?: number;
  beam_depth_m?: number;
  column_width_m?: number;
  lateral_fx_total_kn?: number;
  check_statics?: boolean;
};

export type FeaResultCard = { label: string; value: string; unit?: string | null; tone?: string };

export type FeaBuildingResponse = {
  engine: string;
  load_combination: string;
  geometry: ViewerGeometry;
  result_cards: FeaResultCard[];
  assumptions: string[];
  summary_markdown: string;
  beams: { id: string; floor_z_m: number; M_max_kNm: number; V_max_kN: number; deflection_mm: number }[];
  columns: { id: string; P_max_kN: number }[];
  base_reactions_sample: { node: string; Rz_kN: number; Rx_kN: number }[];
  totals: Record<string, number>;
  pynite_path?: string;
};

export async function runFeaAnalyze(body: FeaBuildingRequest): Promise<FeaBuildingResponse> {
  const res = await fetchWithRetry(
    `${API_URL}/fea/analyze`,
    {
      method: "POST",
      headers: jsonHeaders(),
      body: JSON.stringify(body),
    },
    { retries: 1 },
  );
  if (!res.ok) throw new Error(await parseError(res));
  return (await res.json()) as FeaBuildingResponse;
}

export async function downloadEtabsExport(projectId: string, format: "txt" | "json"): Promise<void> {
  const path = format === "json" ? `/export/etabs/${projectId}/json` : `/export/etabs/${projectId}`;
  const res = await fetchWithRetry(
    `${API_URL}${path}`,
    { headers: authHeaders() },
    { retries: 2 },
  );
  if (!res.ok) throw new Error("Export failed — run an analysis first or check the server.");
  const blob = await res.blob();
  const a = document.createElement("a");
  const url = URL.createObjectURL(blob);
  a.href = url;
  a.download =
    format === "json"
      ? `balmores_etabs_${projectId.slice(0, 8)}.json`
      : `balmores_etabs_${projectId.slice(0, 8)}.txt`;
  a.rel = "noopener";
  a.click();
  URL.revokeObjectURL(url);
}
