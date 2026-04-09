"use client";

import dynamic from "next/dynamic";
import { useEffect, useMemo, useRef, useState } from "react";
import BarMiniChart from "@/components/BarMiniChart";
import KeyboardHelp from "@/components/KeyboardHelp";
import Toast from "@/components/Toast";
import {
  sendChatStream,
  queueVerify,
  downloadEtabsExport,
  runFeaAnalyze,
  type FeaBuildingRequest,
  type FeaBuildingResponse,
  type ViewerGeometry,
} from "@/lib/api";
import { coerceRawPredictions, unwrapDetailedResults } from "@/lib/coercePredictions";
import { buildResultCardsFromSurface, reconcileSurfaceMetrics } from "@/lib/reconcileMetrics";
import AssistantMarkdown from "@/components/AssistantMarkdown";
import { loadPromptHistory, pushPromptHistory, type PromptHistoryItem } from "@/lib/promptHistory";

const ThreeViewer = dynamic(() => import("@/components/ThreeViewer"), {
  ssr: false,
  loading: () => (
    <div className="viewer-loading" role="status">
      Loading 3D preview…
    </div>
  ),
});

type Card = { label: string; value: string; unit?: string; tone?: string };
type Recommendation = { title: string; detail: string; severity: string };
type Message = {
  id: string;
  role: string;
  content: string;
  streaming?: boolean;
  metrics?: Card[];
};

const TABS = [
  "Results",
  "Physics",
  "Assumptions",
  "Member Schedule",
  "Recommendations",
  "Follow-ups",
  "Model outputs",
  "ETABS export",
] as const;

const FEA_TABS = ["Summary", "Assumptions", "Beams", "Columns", "Base reactions"] as const;

const PLACEHOLDER_PROMPT =
  "Example: 6-storey steel, 4×3 bays, 7 m × 6.5 m spans, fixed base, rigid diaphragm, braced, Canada code…";
const COORDINATE_HINT =
  "Enter — send · Shift+Enter — new line · Your message appears on the right; the assistant streams on the left. Esc stops generation.";

const APP_VERSION =
  typeof process !== "undefined" && process.env.NEXT_PUBLIC_APP_VERSION
    ? process.env.NEXT_PUBLIC_APP_VERSION
    : "0.1.0";

type MemberGroupRow = { group: string; scope: string; design_basis: string; suggested_section: string };

function genId(): string {
  if (typeof crypto !== "undefined" && crypto.randomUUID) return crypto.randomUUID();
  return `m-${Date.now()}-${Math.random().toString(36).slice(2, 9)}`;
}

function fmtBrainValue(v: number): string {
  if (!Number.isFinite(v)) return "—";
  const a = Math.abs(v);
  if (a > 0 && a < 1e-8) return v.toExponential(3);
  if (a >= 10000) return v.toLocaleString(undefined, { maximumFractionDigits: 0 });
  if (a >= 100) return v.toFixed(1);
  return v.toFixed(3);
}

const DEFAULT_FEA: FeaBuildingRequest = {
  bays_x: 3,
  bays_y: 3,
  stories: 4,
  span_x_m: 6,
  span_y_m: 6,
  bottom_story_height_m: 4,
  story_height_m: 3.5,
  floor_load_kpa: 10,
  two_way_fraction: 0.5,
  elastic_modulus_gpa: 200,
  poisson_ratio: 0.3,
  beam_width_m: 0.4,
  beam_depth_m: 0.75,
  column_width_m: 0.45,
  lateral_fx_total_kn: 0,
  check_statics: false,
};

export default function HomePage() {
  const [workspace, setWorkspace] = useState<"fea" | "assistant">("fea");
  const [feaForm, setFeaForm] = useState<FeaBuildingRequest>(() => ({ ...DEFAULT_FEA }));
  const [feaResult, setFeaResult] = useState<FeaBuildingResponse | null>(null);
  const [feaLoading, setFeaLoading] = useState(false);
  const [feaErr, setFeaErr] = useState<string | null>(null);
  const [feaTab, setFeaTab] = useState<(typeof FEA_TABS)[number]>("Summary");

  const [projectId, setProjectId] = useState<string | null>(null);
  const [code, setCode] = useState("Canada");
  const [prompt, setPrompt] = useState("");
  const [loading, setLoading] = useState(false);
  const [messages, setMessages] = useState<Message[]>([
    {
      id: "welcome",
      role: "assistant",
      content:
        "Neural assistant mode uses the trained checkpoint. For **code-verified** member forces and reactions, use **PyNite FEA** in the workspace switcher and enter grid, spans, loads, and section sizes.",
    },
  ]);
  const [promptHistory, setPromptHistory] = useState<PromptHistoryItem[]>([]);
  const [exportErr, setExportErr] = useState<string | null>(null);
  const abortRef = useRef<AbortController | null>(null);
  const chatEndRef = useRef<HTMLDivElement>(null);
  const [assumptions, setAssumptions] = useState<string[]>([]);
  const [geometry, setGeometry] = useState<ViewerGeometry | null>(null);
  const [cards, setCards] = useState<Card[]>([]);
  const [recommendations, setRecommendations] = useState<Recommendation[]>([]);
  const [questions, setQuestions] = useState<string[]>([]);
  const [confidence, setConfidence] = useState("Unknown");
  const [details, setDetails] = useState<Record<string, unknown> | null>(null);
  const [verifyStatus, setVerifyStatus] = useState<string>("Not started");
  const [activeTab, setActiveTab] = useState<(typeof TABS)[number]>("Results");
  const [toast, setToast] = useState<string | null>(null);
  const [helpOpen, setHelpOpen] = useState(false);

  const busy = loading || messages.some((m) => m.streaming);

  useEffect(() => {
    setPromptHistory(loadPromptHistory());
  }, []);

  useEffect(() => {
    if (!toast) return;
    const t = window.setTimeout(() => setToast(null), 2600);
    return () => window.clearTimeout(t);
  }, [toast]);

  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      if (e.key !== "?" || e.ctrlKey || e.metaKey || e.altKey) return;
      const el = e.target as HTMLElement | null;
      if (el && (el.tagName === "INPUT" || el.tagName === "TEXTAREA" || el.isContentEditable)) return;
      e.preventDefault();
      setHelpOpen(true);
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, []);

  useEffect(() => {
    return () => {
      abortRef.current?.abort();
    };
  }, []);

  useEffect(() => {
    if (!busy) return;
    const onKey = (e: KeyboardEvent) => {
      if (e.key === "Escape") {
        e.preventDefault();
        abortRef.current?.abort();
      }
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [busy]);

  useEffect(() => {
    chatEndRef.current?.scrollIntoView({ behavior: "smooth", block: "end" });
  }, [messages]);

  const chartData = useMemo(() => {
    if (workspace === "fea" && feaResult?.totals) {
      const t = feaResult.totals;
      return {
        forceItems: [
          { label: "Beam shear", value: t.max_beam_shear_kN ?? 0, unit: "kN" },
          { label: "Beam moment", value: t.max_beam_moment_kNm ?? 0, unit: "kNm" },
          { label: "Column |P|", value: t.max_column_axial_kN ?? 0, unit: "kN" },
          { label: "Σ base Rz", value: t.sum_base_Rz_kN ?? 0, unit: "kN" },
        ],
        driftItems: [
          { label: "Beam defl.", value: t.max_beam_deflection_mm ?? 0, unit: "mm" },
          { label: "Roof |DZ|", value: t.roof_max_DZ_mm ?? 0, unit: "mm" },
          { label: "Σ base Rx", value: t.sum_base_Rx_kN ?? 0, unit: "kN" },
        ],
      };
    }
    const raw = coerceRawPredictions(details?.raw_predictions);
    const spans = (details?._spans as { span_x_m?: number; span_y_m?: number } | undefined) || {};
    const surf = reconcileSurfaceMetrics(raw, spans);
    if (!surf) return null;
    return {
      forceItems: [
        { label: "Beam Shear", value: surf.beam_shear_kN, unit: "kN" },
        { label: "Beam Moment", value: surf.beam_moment_kNm, unit: "kNm" },
        { label: "Column Axial", value: surf.col_axial_kN, unit: "kN" },
        { label: "Joint Reaction V", value: surf.joint_reaction_vertical_kN, unit: "kN" },
      ],
      driftItems: [
        { label: "Max Drift", value: surf.max_drift_mm, unit: "mm" },
        { label: "Roof Disp.", value: surf.roof_disp_mm, unit: "mm" },
        { label: "Base Shear", value: surf.base_shear_kN, unit: "kN" },
      ],
    };
  }, [details, workspace, feaResult]);

  const displayGeometry: ViewerGeometry | null =
    workspace === "fea" && feaResult?.geometry ? feaResult.geometry : geometry;

  async function onRunFea() {
    setFeaErr(null);
    setFeaLoading(true);
    try {
      const res = await runFeaAnalyze(feaForm);
      setFeaResult(res);
      setFeaTab("Summary");
    } catch (e: unknown) {
      setFeaErr(e instanceof Error ? e.message : "FEA request failed");
    } finally {
      setFeaLoading(false);
    }
  }


  async function onSend() {
    const text = prompt.trim();
    if (!text || busy) return;
    abortRef.current?.abort();
    const ac = new AbortController();
    abortRef.current = ac;
    const userMsg: Message = { id: genId(), role: "user", content: text };
    const assistId = genId();
    setMessages((prev) => [...prev, userMsg, { id: assistId, role: "assistant", content: "", streaming: true }]);
    setPrompt("");
    setLoading(true);
    try {
      const res = await sendChatStream(
        { project_id: projectId, message: text, code },
        {
          signal: ac.signal,
          onMeta: (pid) => setProjectId(pid),
          onDelta: (acc) => {
            setMessages((prev) =>
              prev.map((m) => (m.id === assistId ? { ...m, content: acc, streaming: true } : m)),
            );
          },
        },
      );
      setProjectId(res.project_id);
      pushPromptHistory(text, res.project_id);
      setPromptHistory(loadPromptHistory());
      setAssumptions((res.assumptions as string[]) || []);
      setGeometry((res.geometry as ViewerGeometry | undefined) ?? null);
      const baseDr =
        unwrapDetailedResults(res) ??
        ((res.detailed_results || (res as { detailedResults?: unknown }).detailedResults) as Record<
          string,
          unknown
        > | null);
      const dr = baseDr ? { ...baseDr } : {};
      const rawPred = coerceRawPredictions(dr.raw_predictions);
      const st = res.state as { span_x_m?: number; span_y_m?: number } | undefined;
      const spans = { span_x_m: st?.span_x_m, span_y_m: st?.span_y_m };
      const clientSurf = reconcileSurfaceMetrics(rawPred, spans);
      let turnCards: Card[] = [];
      if (clientSurf && Object.keys(rawPred).length > 0) {
        turnCards = buildResultCardsFromSurface(clientSurf) as Card[];
        setCards(turnCards);
        setDetails({
          ...dr,
          raw_predictions: rawPred,
          display_metrics: clientSurf,
          _spans: spans,
        });
      } else {
        turnCards = (res.result_cards as Card[]) || [];
        setCards(turnCards);
        setDetails(Object.keys(dr).length ? dr : null);
      }
      setRecommendations((res.recommendations as Recommendation[]) || []);
      setQuestions((res.follow_up_questions as string[]) || []);
      setConfidence((res.confidence as string) || "Unknown");
      const finalText = String(res.messages?.[0]?.content ?? "");
      setMessages((prev) =>
        prev.map((m) =>
          m.id === assistId
            ? { ...m, content: finalText, streaming: false, metrics: turnCards.length ? turnCards : undefined }
            : m,
        ),
      );
    } catch (err: unknown) {
      const name = err && typeof err === "object" && "name" in err ? String((err as Error).name) : "";
      if (name === "AbortError") {
        setMessages((prev) => prev.filter((m) => m.id !== assistId));
        return;
      }
      const msg = err instanceof Error ? err.message : "Request failed";
      setMessages((prev) =>
        prev.map((m) =>
          m.id === assistId ? { ...m, content: `Error: ${msg}`, streaming: false } : m,
        ),
      );
    } finally {
      setLoading(false);
    }
  }

  async function onVerify() {
    if (!projectId) return;
    const res = await queueVerify(projectId);
    setVerifyStatus((res as { message?: string }).message ?? "OK");
  }

  function onQuestionClick(q: string) {
    setPrompt(q);
  }

  function onStop() {
    abortRef.current?.abort();
  }

  return (
    <div className="page">
      <div className="topbar">
        <div className="brand">
          <div className="brand-badge" aria-hidden />
          <div>
            <div className="brand-title">BALMORES STRUCTURAL</div>
            <div className="small-muted">Parametric 3D frame · PyNite FEA · optional neural assistant</div>
          </div>
        </div>
        <div className="topbar-actions">
          <div className="workspace-switch" role="tablist" aria-label="Workspace">
            <button
              type="button"
              role="tab"
              aria-selected={workspace === "fea"}
              className={`ws-btn ${workspace === "fea" ? "active" : ""}`}
              onClick={() => setWorkspace("fea")}
            >
              PyNite FEA
            </button>
            <button
              type="button"
              role="tab"
              aria-selected={workspace === "assistant"}
              className={`ws-btn ${workspace === "assistant" ? "active" : ""}`}
              onClick={() => setWorkspace("assistant")}
            >
              Neural assistant
            </button>
          </div>
          <button type="button" className="btn btn-outline btn-compact" onClick={() => setHelpOpen(true)}>
            Shortcuts
          </button>
          {workspace === "assistant" ? (
            <div className="confidence-chip" title="Model confidence">
              <span className={`dot ${confidence !== "Unknown" ? "live" : ""}`} aria-hidden />
              <span>Confidence · {confidence}</span>
            </div>
          ) : (
            <div className="confidence-chip" title="Analysis engine">
              <span className="dot live" aria-hidden />
              <span>Engine · PyNite FEM</span>
            </div>
          )}
        </div>
      </div>

      <div className="layout">
        <div className={`panel panel-chat panel-chat-gpt ${workspace === "fea" ? "panel-fea" : ""}`}>
          <div className="panel-header">
            <strong>{workspace === "fea" ? "Building input → PyNite" : "Analysis chat"}</strong>
            <span className="small-muted">
              {workspace === "fea"
                ? "Grid, spans, storey heights, floor load, steel E, sections — then run linear FEA"
                : "You → assistant · streaming · key outputs in-thread"}
            </span>
          </div>
          {workspace === "fea" ? (
            <div className="fea-scroll">
              <p className="fea-lead small-muted">
                Enter a regular bay grid and loading. The backend builds a 3D frame in{" "}
                <strong>Pynite</strong>, applies slab-equivalent UDLs on beams (two-way split), fixes all base nodes,
                and returns envelopes and reactions.
              </p>
              <div className="fea-form-grid">
                <label className="fea-field">
                  <span>Bays X</span>
                  <input
                    type="number"
                    min={1}
                    max={24}
                    value={feaForm.bays_x ?? 3}
                    onChange={(e) =>
                      setFeaForm((f) => ({ ...f, bays_x: Math.min(24, Math.max(1, parseInt(e.target.value, 10) || 1)) }))
                    }
                  />
                </label>
                <label className="fea-field">
                  <span>Bays Y</span>
                  <input
                    type="number"
                    min={1}
                    max={24}
                    value={feaForm.bays_y ?? 3}
                    onChange={(e) =>
                      setFeaForm((f) => ({ ...f, bays_y: Math.min(24, Math.max(1, parseInt(e.target.value, 10) || 1)) }))
                    }
                  />
                </label>
                <label className="fea-field">
                  <span>Storeys</span>
                  <input
                    type="number"
                    min={1}
                    max={50}
                    value={feaForm.stories ?? 4}
                    onChange={(e) =>
                      setFeaForm((f) => ({ ...f, stories: Math.min(50, Math.max(1, parseInt(e.target.value, 10) || 1)) }))
                    }
                  />
                </label>
                <label className="fea-field">
                  <span>Span X (m)</span>
                  <input
                    type="number"
                    step="0.1"
                    min={0.5}
                    value={feaForm.span_x_m ?? 6}
                    onChange={(e) => setFeaForm((f) => ({ ...f, span_x_m: parseFloat(e.target.value) || 6 }))}
                  />
                </label>
                <label className="fea-field">
                  <span>Span Y (m)</span>
                  <input
                    type="number"
                    step="0.1"
                    min={0.5}
                    value={feaForm.span_y_m ?? 6}
                    onChange={(e) => setFeaForm((f) => ({ ...f, span_y_m: parseFloat(e.target.value) || 6 }))}
                  />
                </label>
                <label className="fea-field">
                  <span>First storey h (m)</span>
                  <input
                    type="number"
                    step="0.1"
                    min={0.5}
                    value={feaForm.bottom_story_height_m ?? 4}
                    onChange={(e) =>
                      setFeaForm((f) => ({ ...f, bottom_story_height_m: parseFloat(e.target.value) || 4 }))
                    }
                  />
                </label>
                <label className="fea-field">
                  <span>Typical storey h (m)</span>
                  <input
                    type="number"
                    step="0.1"
                    min={0.5}
                    value={feaForm.story_height_m ?? 3.5}
                    onChange={(e) => setFeaForm((f) => ({ ...f, story_height_m: parseFloat(e.target.value) || 3.5 }))}
                  />
                </label>
                <label className="fea-field fea-field-wide">
                  <span>Floor pressure (kPa)</span>
                  <input
                    type="number"
                    step="0.5"
                    min={0.1}
                    value={feaForm.floor_load_kpa ?? 10}
                    onChange={(e) =>
                      setFeaForm((f) => ({ ...f, floor_load_kpa: parseFloat(e.target.value) || 10 }))
                    }
                  />
                  <small className="field-hint">DL+LL proxy on every elevated slab</small>
                </label>
                <label className="fea-field">
                  <span>Two-way split</span>
                  <input
                    type="number"
                    step="0.05"
                    min={0}
                    max={1}
                    value={feaForm.two_way_fraction ?? 0.5}
                    onChange={(e) =>
                      setFeaForm((f) => ({
                        ...f,
                        two_way_fraction: Math.min(1, Math.max(0, parseFloat(e.target.value) || 0.5)),
                      }))
                    }
                  />
                </label>
                <label className="fea-field">
                  <span>E (GPa)</span>
                  <input
                    type="number"
                    step="1"
                    min={1}
                    value={feaForm.elastic_modulus_gpa ?? 200}
                    onChange={(e) =>
                      setFeaForm((f) => ({ ...f, elastic_modulus_gpa: parseFloat(e.target.value) || 200 }))
                    }
                  />
                </label>
                <label className="fea-field">
                  <span>ν</span>
                  <input
                    type="number"
                    step="0.01"
                    min={0.01}
                    max={0.49}
                    value={feaForm.poisson_ratio ?? 0.3}
                    onChange={(e) =>
                      setFeaForm((f) => ({ ...f, poisson_ratio: parseFloat(e.target.value) || 0.3 }))
                    }
                  />
                </label>
                <label className="fea-field">
                  <span>Beam b (m)</span>
                  <input
                    type="number"
                    step="0.01"
                    min={0.05}
                    value={feaForm.beam_width_m ?? 0.4}
                    onChange={(e) =>
                      setFeaForm((f) => ({ ...f, beam_width_m: parseFloat(e.target.value) || 0.4 }))
                    }
                  />
                </label>
                <label className="fea-field">
                  <span>Beam h (m)</span>
                  <input
                    type="number"
                    step="0.01"
                    min={0.05}
                    value={feaForm.beam_depth_m ?? 0.75}
                    onChange={(e) =>
                      setFeaForm((f) => ({ ...f, beam_depth_m: parseFloat(e.target.value) || 0.75 }))
                    }
                  />
                </label>
                <label className="fea-field">
                  <span>Column square (m)</span>
                  <input
                    type="number"
                    step="0.01"
                    min={0.05}
                    value={feaForm.column_width_m ?? 0.45}
                    onChange={(e) =>
                      setFeaForm((f) => ({ ...f, column_width_m: parseFloat(e.target.value) || 0.45 }))
                    }
                  />
                </label>
                <label className="fea-field fea-field-wide">
                  <span>Optional roof lateral +X (kN total)</span>
                  <input
                    type="number"
                    step="10"
                    min={0}
                    value={feaForm.lateral_fx_total_kn ?? 0}
                    onChange={(e) =>
                      setFeaForm((f) => ({ ...f, lateral_fx_total_kn: Math.max(0, parseFloat(e.target.value) || 0) }))
                    }
                  />
                  <small className="field-hint">Split equally over all roof nodes; combined with gravity</small>
                </label>
                <label className="fea-field fea-check">
                  <input
                    type="checkbox"
                    checked={!!feaForm.check_statics}
                    onChange={(e) => setFeaForm((f) => ({ ...f, check_statics: e.target.checked }))}
                  />
                  <span>Verbose statics check (server log)</span>
                </label>
              </div>
              {feaErr ? <p className="export-err fea-err">{feaErr}</p> : null}
              <div className="fea-actions">
                <button type="button" className="btn btn-outline" onClick={() => setFeaForm({ ...DEFAULT_FEA })}>
                  Reset defaults
                </button>
                <button type="button" className="btn btn-send" onClick={() => void onRunFea()} disabled={feaLoading}>
                  {feaLoading ? "Running FEA…" : "Run PyNite analysis"}
                </button>
              </div>
            </div>
          ) : null}
          {workspace === "assistant" ? (
            <>
          <div className="chat-scroll" role="log" aria-live="polite" aria-relevant="additions text">
            <div className="chat-thread">
              {messages.map((m) => (
                <div key={m.id} className={`msg-row msg-row-${m.role}`}>
                  <div className={`msg-bubble msg ${m.role}${m.streaming ? " is-streaming" : ""}`}>
                    <div className="msg-meta">
                      <small>{m.role === "user" ? "You" : "Assistant"}</small>
                      {m.role === "assistant" && !m.streaming && m.content.trim() ? (
                        <button
                          type="button"
                          className="msg-copy"
                          onClick={() => {
                            void navigator.clipboard.writeText(m.content).then(() => setToast("Copied"));
                          }}
                        >
                          Copy
                        </button>
                      ) : null}
                    </div>
                    {m.role === "assistant" && m.streaming && !m.content.trim() ? (
                      <div className="typing-dots typing-dots-inline" aria-hidden>
                        <span className="typing-dot" />
                        <span className="typing-dot" />
                        <span className="typing-dot" />
                      </div>
                    ) : m.role === "assistant" ? (
                      <AssistantMarkdown content={m.content} streaming={!!m.streaming} />
                    ) : (
                      <div className="msg-body-plain">{m.content}</div>
                    )}
                    {m.role === "assistant" && !m.streaming && m.metrics && m.metrics.length > 0 ? (
                      <div className="msg-metrics" aria-label="Key structural outputs">
                        <div className="msg-metrics-title">Key outputs</div>
                        <div className="msg-metrics-grid">
                          {m.metrics.map((c) => (
                            <div key={c.label} className={`msg-metric-chip ${c.tone ? `tone-${c.tone}` : ""}`}>
                              <span className="msg-metric-label">{c.label}</span>
                              <span className="msg-metric-value">
                                {c.value}
                                {c.unit ? ` ${c.unit}` : ""}
                              </span>
                            </div>
                          ))}
                        </div>
                      </div>
                    ) : null}
                  </div>
                </div>
              ))}
              <div ref={chatEndRef} className="chat-end-anchor" />
            </div>
          </div>

          <details className="prompt-history-details">
            <summary className="prompt-history-summary">Recent prompts</summary>
            <div className="prompt-history-inner">
              {promptHistory.length === 0 ? (
                <p className="small-muted">Sent prompts are listed here for reuse.</p>
              ) : (
                <ul className="prompt-history-list">
                  {promptHistory.slice(0, 12).map((h) => (
                    <li key={h.id}>
                      <button
                        type="button"
                        className="prompt-history-item"
                        title={new Date(h.at).toLocaleString()}
                        onClick={() => setPrompt(h.text)}
                      >
                        {h.text.length > 72 ? `${h.text.slice(0, 72)}…` : h.text}
                      </button>
                    </li>
                  ))}
                </ul>
              )}
            </div>
          </details>

          <div className="input-wrap chat-composer">
            <div className="row row-tools">
              <select value={code} onChange={(e) => setCode(e.target.value)} className="code-select">
                <option>Canada</option>
                <option>US</option>
                <option>Philippines</option>
              </select>
              <button type="button" className="btn btn-outline" onClick={onVerify} disabled={!projectId}>
                Queue ETABS Verify
              </button>
            </div>
            <textarea
              value={prompt}
              onChange={(e) => setPrompt(e.target.value)}
              placeholder={PLACEHOLDER_PROMPT}
              onKeyDown={(e) => {
                if (e.key === "Enter" && (e.ctrlKey || e.metaKey)) {
                  e.preventDefault();
                  onSend();
                  return;
                }
                if (e.key === "Enter" && !e.shiftKey) {
                  e.preventDefault();
                  onSend();
                }
              }}
            />
            <div className="input-actions">
              <button type="button" className="btn btn-stop" onClick={onStop} disabled={!busy}>
                Stop
              </button>
              <button type="button" className="btn btn-send" onClick={onSend} disabled={busy}>
                {loading ? "Analyzing…" : busy ? "Replying…" : "Send"}
              </button>
            </div>
            <p className="hint-line">{COORDINATE_HINT}</p>
            <p className={`status-line ${verifyStatus === "Not started" ? "status-line--idle" : ""}`}>{verifyStatus}</p>
          </div>
            </>
          ) : null}
        </div>

        <div className="panel panel-viewer">
          <div className="panel-header">
            <strong>Live preview</strong>
            <span className="small-muted">Orbit · scroll zoom</span>
          </div>
          <div className="viewer-shell">
            <div className="canvas-wrap">
              <ThreeViewer geometry={displayGeometry} />
              <div className="overlay">
                <div className="tag">3D Model</div>
                <div className="tag">{workspace === "fea" ? "PyNite mesh" : "ETABS-calibrated"}</div>
              </div>
            </div>
            <div className="bottom-split">
              <BarMiniChart title="Beam / Column / Reaction" items={chartData?.forceItems || []} />
              <BarMiniChart title="Drift / Roof / Base" items={chartData?.driftItems || []} />
            </div>
          </div>
        </div>

        <div className="panel panel-results">
          <div className="panel-header">
            <strong>Results</strong>
            <span className="small-muted">
              {workspace === "fea" ? "PyNite envelopes · reactions" : "Tabs · full detail"}
            </span>
          </div>
          <div className="tabs">
            {workspace === "fea"
              ? FEA_TABS.map((t) => (
                  <button
                    key={t}
                    type="button"
                    className={`tab ${feaTab === t ? "active" : ""}`}
                    onClick={() => setFeaTab(t)}
                  >
                    {t}
                  </button>
                ))
              : TABS.map((t) => (
                  <button
                    key={t}
                    type="button"
                    className={`tab ${activeTab === t ? "active" : ""}`}
                    onClick={() => setActiveTab(t)}
                  >
                    {t}
                  </button>
                ))}
          </div>
          <div className="panel-body results-scroll">
            {workspace === "fea" ? (
              <>
                {feaTab === "Summary" && (
                  <div className="section">
                    {feaResult?.result_cards?.length ? (
                      <div className="result-grid fea-card-grid">
                        {feaResult.result_cards.map((card) => (
                          <div key={card.label} className="card">
                            <div className="card-label">{card.label}</div>
                            <div className={`card-value ${card.tone ? `tone-${card.tone}` : ""}`}>
                              {card.value}
                              {card.unit ? ` ${card.unit}` : ""}
                            </div>
                          </div>
                        ))}
                      </div>
                    ) : null}
                    {!feaResult ? (
                      <p className="small-muted">
                        Run <strong>PyNite analysis</strong> from the left panel to see forces, deflections, and reactions.
                      </p>
                    ) : (
                      <AssistantMarkdown content={feaResult.summary_markdown} streaming={false} />
                    )}
                    {feaResult ? (
                      <p className="small-muted fea-meta">
                        Combination <code>{feaResult.load_combination}</code>
                        {feaResult.engine ? (
                          <>
                            {" · "}
                            {feaResult.engine}
                          </>
                        ) : null}
                      </p>
                    ) : null}
                  </div>
                )}
                {feaTab === "Assumptions" && (
                  <div className="section">
                    {feaResult?.assumptions?.length ? (
                      feaResult.assumptions.map((a, i) => (
                        <div key={i} className="li">
                          {a}
                        </div>
                      ))
                    ) : (
                      <div className="small-muted">No run yet.</div>
                    )}
                  </div>
                )}
                {feaTab === "Beams" && (
                  <div className="section table-wrap">
                    <table className="table table-striped">
                      <thead>
                        <tr>
                          <th>Member</th>
                          <th>z (m)</th>
                          <th>|M| (kN·m)</th>
                          <th>|V| (kN)</th>
                          <th>δ (mm)</th>
                        </tr>
                      </thead>
                      <tbody>
                        {(feaResult?.beams ?? []).map((b) => (
                          <tr key={b.id}>
                            <td>
                              <code>{b.id}</code>
                            </td>
                            <td>{b.floor_z_m}</td>
                            <td>{b.M_max_kNm}</td>
                            <td>{b.V_max_kN}</td>
                            <td>{b.deflection_mm}</td>
                          </tr>
                        ))}
                      </tbody>
                    </table>
                    {!feaResult?.beams?.length ? <p className="small-muted">No data.</p> : null}
                  </div>
                )}
                {feaTab === "Columns" && (
                  <div className="section table-wrap">
                    <table className="table table-striped">
                      <thead>
                        <tr>
                          <th>Member</th>
                          <th>|P| (kN)</th>
                        </tr>
                      </thead>
                      <tbody>
                        {(feaResult?.columns ?? []).map((c) => (
                          <tr key={c.id}>
                            <td>
                              <code>{c.id}</code>
                            </td>
                            <td>{c.P_max_kN}</td>
                          </tr>
                        ))}
                      </tbody>
                    </table>
                    {!feaResult?.columns?.length ? <p className="small-muted">No data.</p> : null}
                  </div>
                )}
                {feaTab === "Base reactions" && (
                  <div className="section table-wrap">
                    <table className="table table-striped">
                      <thead>
                        <tr>
                          <th>Node</th>
                          <th>Rz (kN)</th>
                          <th>Rx (kN)</th>
                        </tr>
                      </thead>
                      <tbody>
                        {(feaResult?.base_reactions_sample ?? []).map((r) => (
                          <tr key={r.node}>
                            <td>
                              <code>{r.node}</code>
                            </td>
                            <td>{r.Rz_kN}</td>
                            <td>{r.Rx_kN}</td>
                          </tr>
                        ))}
                      </tbody>
                    </table>
                    <p className="small-muted">Sample base nodes; see Summary for summed reactions.</p>
                    {!feaResult?.base_reactions_sample?.length ? <p className="small-muted">No data.</p> : null}
                  </div>
                )}
              </>
            ) : (
              <>
            {activeTab === "Results" && (
              <div className="result-grid">
                {cards.length === 0 && (
                  <div className="small-muted empty-hint">Send a chat message to populate force and drift cards.</div>
                )}
                {cards.map((card) => (
                  <div key={card.label} className="card">
                    <div className="card-label">{card.label}</div>
                    <div className={`card-value ${card.tone ? `tone-${card.tone}` : ""}`}>
                      {card.value} {card.unit || ""}
                    </div>
                  </div>
                ))}
              </div>
            )}

            {activeTab === "Physics" && (
              <div className="section">
                <div className="physics-grid">
                  {details &&
                  details.physics_checks &&
                  typeof details.physics_checks === "object" ? (
                    <>
                      <div className="physics-card">
                        <strong>Physics checks</strong>
                        <div className="li">
                          Base shear:{" "}
                          {(details.physics_checks as { base_shear_check?: string }).base_shear_check}
                        </div>
                        <div className="li">
                          Column axial:{" "}
                          {(details.physics_checks as { column_axial_check?: string }).column_axial_check}
                        </div>
                        <div className="li">
                          Drift: {(details.physics_checks as { drift_compliance?: string }).drift_compliance}
                        </div>
                      </div>
                      <div className="physics-card">
                        <strong>Basis</strong>
                        <div className="li">
                          {(details.physics_checks as { physics_basis?: string }).physics_basis}
                        </div>
                        <div className="li">{(details.physics_checks as { units?: string }).units}</div>
                      </div>
                    </>
                  ) : null}
                </div>
                {!details?.physics_checks ? (
                  <div className="small-muted">Run analysis to see physics checks.</div>
                ) : null}
              </div>
            )}

            {activeTab === "Assumptions" && (
              <div className="section">
                {assumptions.length > 0 ? (
                  assumptions.map((a, i) => (
                    <div key={i} className="li">
                      {a}
                    </div>
                  ))
                ) : (
                  <div className="small-muted">No assumptions yet.</div>
                )}
              </div>
            )}

            {activeTab === "Member Schedule" && (
              <div className="section table-wrap">
                <table className="table table-striped">
                  <thead>
                    <tr>
                      <th>Group</th>
                      <th>Scope</th>
                      <th>Design basis</th>
                      <th>Suggested</th>
                    </tr>
                  </thead>
                  <tbody>
                    {(
                      (details?.member_schedule as { beam_groups?: MemberGroupRow[] } | undefined)?.beam_groups ?? []
                    ).map((g) => (
                      <tr key={`b-${g.group}`}>
                        <td>{g.group}</td>
                        <td>{g.scope}</td>
                        <td>{g.design_basis}</td>
                        <td>{g.suggested_section}</td>
                      </tr>
                    ))}
                    {(
                      (details?.member_schedule as { column_groups?: MemberGroupRow[] } | undefined)?.column_groups ??
                      []
                    ).map((g) => (
                      <tr key={`c-${g.group}`}>
                        <td>{g.group}</td>
                        <td>{g.scope}</td>
                        <td>{g.design_basis}</td>
                        <td>{g.suggested_section}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            )}

            {activeTab === "Recommendations" && (
              <div className="section">
                {recommendations.length > 0 ? (
                  recommendations.map((r, i) => (
                    <div key={i} className="li">
                      <strong>{r.title}</strong>
                      <div className="small-muted" style={{ marginTop: 6 }}>
                        {r.detail}
                      </div>
                    </div>
                  ))
                ) : (
                  <div className="small-muted">No recommendations yet.</div>
                )}
              </div>
            )}

            {activeTab === "Follow-ups" && (
              <div className="section">
                <p className="small-muted">Tap to paste into the input.</p>
                <div className="question-list">
                  {questions.length > 0 ? (
                    questions.map((q, i) => (
                      <div key={i} className="li question-chip" onClick={() => onQuestionClick(q)}>
                        {q}
                      </div>
                    ))
                  ) : (
                    <div className="small-muted">No follow-ups yet.</div>
                  )}
                </div>
              </div>
            )}

            {activeTab === "Model outputs" && (
              <div className="section">
                <p className="small-muted model-outputs-lead">
                  Raw neural targets from the checkpoint. Headline metrics use reconciled display values when{" "}
                  <code>max_*</code> heads are tiny.
                </p>
                {Array.isArray(details?.brain_targets) && (details!.brain_targets as unknown[]).length > 0 ? (
                  <div className="table-wrap model-outputs-scroll">
                    <table className="table table-striped">
                      <thead>
                        <tr>
                          <th>Key</th>
                          <th>Value</th>
                          <th>Unit</th>
                        </tr>
                      </thead>
                      <tbody>
                        {(
                          details!.brain_targets as {
                            key: string;
                            value: number;
                            unit: string;
                          }[]
                        ).map((row) => (
                          <tr key={row.key}>
                            <td>
                              <code className="brain-key">{row.key}</code>
                            </td>
                            <td>{fmtBrainValue(row.value)}</td>
                            <td>{row.unit || "—"}</td>
                          </tr>
                        ))}
                      </tbody>
                    </table>
                  </div>
                ) : (
                  <div className="small-muted">Run analysis to list all targets.</div>
                )}
              </div>
            )}

            {activeTab === "ETABS export" && (
              <div className="section">
                <p className="small-muted">
                  Download summary text or geometry JSON (not a binary .edb). Requires a project session from chat.
                </p>
                <div className="row etabs-row">
                  <button
                    type="button"
                    className="btn btn-outline"
                    disabled={!projectId}
                    onClick={async () => {
                      if (!projectId) return;
                      setExportErr(null);
                      try {
                        await downloadEtabsExport(projectId, "txt");
                      } catch (e: unknown) {
                        setExportErr(e instanceof Error ? e.message : "Download failed");
                      }
                    }}
                  >
                    .txt summary
                  </button>
                  <button
                    type="button"
                    className="btn btn-outline"
                    disabled={!projectId}
                    onClick={async () => {
                      if (!projectId) return;
                      setExportErr(null);
                      try {
                        await downloadEtabsExport(projectId, "json");
                      } catch (e: unknown) {
                        setExportErr(e instanceof Error ? e.message : "Download failed");
                      }
                    }}
                  >
                    Geometry JSON
                  </button>
                </div>
                {exportErr ? <p className="export-err">{exportErr}</p> : null}
              </div>
            )}
              </>
            )}
          </div>
        </div>
      </div>

      <KeyboardHelp open={helpOpen} onClose={() => setHelpOpen(false)} />
      <Toast message={toast} />

      <footer className="site-footer">
        <span>Balmores Structural</span>
        <span className="footer-sep" aria-hidden />
        <span className="small-muted">PyNite FEA · optional neural assistant</span>
        <span className="footer-sep" aria-hidden />
        <span className="footer-ver">v{APP_VERSION}</span>
      </footer>
    </div>
  );
}
