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
  const retryOn = opts?.retryOn ?? [429, 502, 503, 504];
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

/** Parameters returned from the server after parsing the chat message (for diagrams / UI). */
export type FeaParsedModel = {
  analysis_type?: "beam_2d" | "frame_2d" | "building_3d";
  // 3D building
  spans_x_m?: number[];
  spans_y_m?: number[];
  story_heights_m?: number[];
  dl_kpa?: number;
  ll_kpa?: number;
  slab_sw_kpa?: number;
  wind_pressure_kpa?: number;
  lateral_roof_fraction_of_gravity?: number;
  two_way_fraction?: number;
  material_steel?: boolean;
  sbc_kpa?: number | null;
  // 2D frame
  spans_m?: number[];
  dl_kN_per_m?: number;
  ll_kN_per_m?: number;
  lateral_fx_per_floor_kN?: number;
  // 2D beam
  span_m?: number;
  support_left?: string;
  support_right?: string;
  cantilever_left_m?: number;
  cantilever_right_m?: number;
  point_loads?: { P_kN: number; x_m: number; case?: string }[];
  // Common section
  material?: string;
  beam_width_m?: number;
  beam_depth_m?: number;
  column_width_m?: number;
};

/** Diagram arrays returned for 2D beam/frame analyses. `[xs, ys]` pairs. */
export type FeaDiagrams = {
  /** beam_2d only */
  shear_kN?: [number[], number[]];
  moment_kNm?: [number[], number[]];
  deflection_mm?: [number[], number[]];
  /** frame_2d only */
  moment_per_level_kNm?: Record<string, [number[], number[]]>;
  shear_per_level_kN?: Record<string, [number[], number[]]>;
  x_label_m?: string;
};

/** Location-aware design criteria resolved by the backend. */
export type FeaDesignCriteria = {
  location_input?: string;
  matched_location?: string | null;
  country?: string;
  is_assumed?: boolean;
  loads?: { dl_kpa: number; ll_kpa: number; snow_kpa: number; notes?: string };
  wind?: {
    design_wind_speed_mps: number;
    pressure_kpa: number;
    exposure_category: string;
    importance_factor: number;
    code_basis: string;
  };
  seismic?: {
    zone: number;
    pga_g: number;
    base_shear_coeff: number;
    site_class: string;
    code_basis: string;
  };
  soil?: { sbc_kpa: number; description: string; code_basis: string };
  combos?: { uls: string[]; sls: string[]; governing: string };
  notes?: string[];
};

export type FeaPromptResponse = {
  analysis_type: "beam_2d" | "frame_2d" | "building_3d";
  input_summary: string;
  parse_notes: string[];
  parsed_model?: FeaParsedModel;
  engine: string;
  load_combination: string;
  geometry: ViewerGeometry;
  result_cards: FeaResultCard[];
  assumptions: string[];
  summary_markdown: string;
  beams: { id: string; floor_z_m: number; M_max_kNm: number; V_max_kN: number; deflection_mm: number }[];
  columns: {
    id: string;
    P_max_kN: number;
    My_max_kNm: number;
    Mz_max_kNm: number;
    T_max_kNm: number;
  }[];
  base_reactions: {
    node: string;
    x_m: number;
    y_m?: number;
    Rx_kN: number;
    Ry_kN?: number;
    Rz_kN?: number;
    Mx_kNm?: number;
    My_kNm?: number;
    Mz_kNm: number;
  }[];
  storey_drifts: {
    storey_index: number;
    z_top_m: number;
    height_m: number;
    max_drift_mm: number;
    drift_ratio_h: number;
  }[];
  p_delta_note: string;
  totals: Record<string, number | null | undefined>;
  diagrams?: FeaDiagrams;
  design_criteria?: FeaDesignCriteria;
  elapsed_ms?: number;
  executive_summary?: string;
  pynite_path?: string;
};

export async function analyzeFeaPrompt(
  message: string,
  opts?: { run_p_delta?: boolean },
): Promise<FeaPromptResponse> {
  const res = await fetchWithRetry(
    `${API_URL}/fea/analyze-prompt`,
    {
      method: "POST",
      headers: jsonHeaders(),
      body: JSON.stringify({
        message,
        run_p_delta: opts?.run_p_delta !== false,
      }),
    },
    { retries: 1 },
  );
  if (!res.ok) throw new Error(await parseError(res));
  return (await res.json()) as FeaPromptResponse;
}

/** Live-progress event emitted by the streaming endpoint. */
export type FeaProgressEvent =
  | {
      type: "stage";
      stage: string;
      label: string;
      progress?: number;
      elapsed_seconds?: number;
      estimated_total_seconds?: number;
    }
  | {
      type: "tick";
      progress: number;
      elapsed_seconds: number;
      estimated_total_seconds?: number;
      /** Set by /llm/ask/stream during the DeepSeek-R1 thinking phase. */
      phase?: "fea" | "llm_thinking";
      llm_elapsed_seconds?: number;
    }
  | { type: "complete"; data: FeaPromptResponse }
  | { type: "llm_token"; text: string }
  | { type: "error"; status?: number; message: string };

/** /llm/health response — used by frontend for the privacy-mode badge. */
export type LlmHealth = {
  enabled: boolean;
  ok: boolean;
  model: string;
  endpoint: string;
  loopback_only: boolean;
  installed_models?: string[];
  reason?: string;
};

export async function getLlmHealth(): Promise<LlmHealth | null> {
  try {
    const res = await fetch(`${API_URL}/llm/health`, {
      headers: authHeaders(),
      cache: "no-store",
    });
    if (!res.ok) return null;
    return (await res.json()) as LlmHealth;
  } catch {
    return null;
  }
}

export type LlmStreamEvent =
  | {
      type: "stage";
      stage: string;
      label: string;
      progress?: number;
      elapsed_seconds?: number;
    }
  | {
      type: "tick";
      progress: number;
      elapsed_seconds: number;
      phase?: "fea" | "llm_thinking";
      llm_elapsed_seconds?: number;
    }
  | { type: "llm_token"; text: string }
  | { type: "fea_ready"; data: FeaPromptResponse }
  | {
      type: "complete";
      data: FeaPromptResponse | null;
      llm_summary: string;
      rescue_note?: string | null;
      chat_only?: boolean;
    }
  | { type: "error"; status?: number; message: string };

/**
 * Stream a user prompt through PyNite + local DeepSeek-R1 commentary.
 * All inference stays on 127.0.0.1 by design.
 */
export async function askLlmStream(
  message: string,
  opts: {
    run_p_delta?: boolean;
    use_llm_summary?: boolean;
    signal?: AbortSignal;
    onProgress?: (ev: LlmStreamEvent) => void;
    onLlmToken?: (text: string, accumulated: string) => void;
    onFeaReady?: (data: FeaPromptResponse) => void;
  },
): Promise<{
  data: FeaPromptResponse | null;
  llm_summary: string;
  rescue_note?: string | null;
  chat_only?: boolean;
}> {
  const body = JSON.stringify({
    message,
    run_p_delta: opts.run_p_delta !== false,
    use_llm_summary: opts.use_llm_summary !== false,
  });

  const res = await fetch(`${API_URL}/llm/ask/stream`, {
    method: "POST",
    headers: jsonHeaders(),
    body,
    signal: opts.signal,
  });
  if (!res.ok || !res.body) {
    if (res.status === 404 || res.status === 405) {
      const data = await analyzeFeaPrompt(message, { run_p_delta: opts.run_p_delta });
      return { data, llm_summary: "" };
    }
    throw new Error(await parseError(res));
  }
  const reader = res.body.getReader();
  const decoder = new TextDecoder();
  let buffer = "";
  let accumulated = "";
  let complete: FeaPromptResponse | null = null;
  let feaReady: FeaPromptResponse | null = null;
  let summary = "";
  let rescueNote: string | null | undefined = undefined;
  let chatOnly = false;
  let sawComplete = false;
  let lastError: string | null = null;

  const consume = (line: string) => {
    const t = line.trim();
    if (!t) return;
    let ev: LlmStreamEvent;
    try {
      ev = JSON.parse(t) as LlmStreamEvent;
    } catch {
      return;
    }
    opts.onProgress?.(ev);
    if (ev.type === "llm_token") {
      accumulated += ev.text;
      opts.onLlmToken?.(ev.text, accumulated);
    } else if (ev.type === "fea_ready") {
      feaReady = ev.data;
      opts.onFeaReady?.(ev.data);
    } else if (ev.type === "complete") {
      sawComplete = true;
      complete = ev.data; // may be null in chat-only mode
      summary = ev.llm_summary || accumulated;
      rescueNote = ev.rescue_note ?? null;
      chatOnly = ev.chat_only ?? false;
    } else if (ev.type === "error") {
      lastError = ev.message;
    }
  };

  while (true) {
    const { done, value } = await reader.read();
    if (done) break;
    buffer += decoder.decode(value, { stream: true });
    const parts = buffer.split("\n");
    buffer = parts.pop() ?? "";
    for (const line of parts) consume(line);
  }
  if (buffer.trim()) consume(buffer);
  if (sawComplete) {
    return {
      data: complete,
      llm_summary: summary,
      rescue_note: rescueNote,
      chat_only: chatOnly,
    };
  }
  if (feaReady) {
    return {
      data: feaReady,
      llm_summary: accumulated.trim() || summary,
      rescue_note: rescueNote,
      chat_only: false,
    };
  }
  if (lastError) {
    throw new Error(lastError);
  }
  throw new Error("LLM stream ended without complete payload");
}

/**
 * Stream the analysis with STAAD-style progress events. Falls back to the
 * non-streaming endpoint if the server doesn't support streaming.
 */
export async function analyzeFeaPromptStream(
  message: string,
  opts: {
    run_p_delta?: boolean;
    signal?: AbortSignal;
    onProgress?: (ev: FeaProgressEvent) => void;
  },
): Promise<FeaPromptResponse> {
  const body = JSON.stringify({
    message,
    run_p_delta: opts.run_p_delta !== false,
  });

  let res: Response;
  try {
    res = await fetch(`${API_URL}/fea/analyze-prompt/stream`, {
      method: "POST",
      headers: jsonHeaders(),
      body,
      signal: opts.signal,
    });
  } catch (e) {
    return analyzeFeaPrompt(message, { run_p_delta: opts.run_p_delta });
  }

  if (!res.ok || !res.body) {
    if (res.status === 404 || res.status === 405) {
      return analyzeFeaPrompt(message, { run_p_delta: opts.run_p_delta });
    }
    throw new Error(await parseError(res));
  }

  const reader = res.body.getReader();
  const decoder = new TextDecoder();
  let buffer = "";
  let complete: FeaPromptResponse | null = null;
  let lastError: string | null = null;

  const consume = (line: string) => {
    const t = line.trim();
    if (!t) return;
    let ev: FeaProgressEvent;
    try {
      ev = JSON.parse(t) as FeaProgressEvent;
    } catch {
      return;
    }
    opts.onProgress?.(ev);
    if (ev.type === "complete") complete = ev.data;
    if (ev.type === "error") lastError = ev.message;
  };

  while (true) {
    const { done, value } = await reader.read();
    if (done) break;
    buffer += decoder.decode(value, { stream: true });
    const parts = buffer.split("\n");
    buffer = parts.pop() ?? "";
    for (const line of parts) consume(line);
  }
  if (buffer.trim()) consume(buffer);
  if (complete) return complete;
  // Stream ended early or returned an error event — retry once on the JSON endpoint.
  try {
    return await analyzeFeaPrompt(message, { run_p_delta: opts.run_p_delta });
  } catch (fallbackErr) {
    if (lastError) throw new Error(lastError);
    throw fallbackErr;
  }
}

/** Run PyNite with streaming progress; always falls back to the JSON endpoint. */
export async function runFeaAnalysisResilient(
  message: string,
  opts: {
    run_p_delta?: boolean;
    signal?: AbortSignal;
    onProgress?: (ev: FeaProgressEvent) => void;
  },
): Promise<FeaPromptResponse> {
  try {
    return await analyzeFeaPromptStream(message, opts);
  } catch {
    return analyzeFeaPrompt(message, { run_p_delta: opts.run_p_delta });
  }
}

function feaResultForLlmSummary(res: FeaPromptResponse): FeaPromptResponse {
  return {
    ...res,
    diagrams: {},
  };
}

/**
 * DeepSeek-R1 executive summary (recommendations + conclusion) after PyNite has finished.
 * Non-streaming — avoids NDJSON timeout issues on long solves.
 */
export async function summarizeFeaWithLlm(
  message: string,
  feaResult: FeaPromptResponse,
  opts?: { signal?: AbortSignal },
): Promise<string> {
  try {
    const res = await fetch(`${API_URL}/llm/summarize`, {
      method: "POST",
      headers: jsonHeaders(),
      body: JSON.stringify({ message, fea_result: feaResultForLlmSummary(feaResult) }),
      signal: opts?.signal,
    });
    if (!res.ok) {
      if (res.status === 404 || res.status === 405 || res.status === 403 || res.status === 429) {
        return "";
      }
      throw new Error(await parseError(res));
    }
    const j = (await res.json()) as { llm_summary?: string };
    return (j.llm_summary || "").trim();
  } catch {
    return "";
  }
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
