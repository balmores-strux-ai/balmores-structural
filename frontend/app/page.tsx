"use client";

import dynamic from "next/dynamic";
import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import AssistantMarkdown from "@/components/AssistantMarkdown";
import BeamDiagrams from "@/components/BeamDiagrams";
import FeaDiagrams from "@/components/FeaDiagrams";
import ProfileBadges from "@/components/ProfileBadges";
import {
  analyzeFeaPrompt,
  type FeaParsedModel,
  type FeaPromptResponse,
  type ViewerGeometry,
} from "@/lib/api";

const ThreeViewer = dynamic(() => import("@/components/ThreeViewer"), {
  ssr: false,
  loading: () => (
    <div className="viewer-loading" role="status">
      Loading 3D preview…
    </div>
  ),
});

const PLACEHOLDER = `Examples you can paste:

• Simply supported steel beam, span 8 m, UDL 15 kN/m DL, 10 kN/m LL, 40 kN point load at midspan.
• 2D RC moment frame, 3 bays of 6 m, 4 storeys at 3.5 m, 20 kN/m DL, 8 kN/m LL, 25 kN lateral per floor.
• 6-storey RC building, X-spans (6, 8, 6m), Y-spans (5, 5m), 3.5m storey heights, 4.5 kPa DL, 3 kPa LL, 200mm slab, 1 kPa wind, Seismic Zone 3.`;

const WELCOME_ASSISTANT = `**Balmores Structural — PyNite assistant**

I am powered by the **open-source PyNite** finite-element library, integrated directly into this app. Describe any of the three kinds of problems below in plain English and I will build, analyse, and report them:

1. **2D beam** — spans, supports (pin/roller/fixed/cantilever), UDLs, point loads. Output includes **shear / moment / deflection diagrams**.
2. **2D moment frame** — bay spans, storey heights, gravity and lateral per-floor loads. Output includes member envelopes, reactions and drift.
3. **3D building** — X- and Y-bay spans (m), storey heights, DL/LL (kPa), wind / seismic zone, SBC. Output includes a coloured 3D geometry, plan/elevation schematics, envelopes and reactions.

Tip: include explicit numbers and units, e.g. \`UDL 15 kN/m\`, \`3 bays of 6 m\`, \`X-spans (6, 8, 6m)\`.`;

type QuickPrompt = {
  variant: "beam" | "frame" | "building";
  label: string;
  prompt: string;
};

const QUICK_PROMPTS: QuickPrompt[] = [
  {
    variant: "beam",
    label: "2D beam · simply supported",
    prompt:
      "Simply supported steel beam, span 8 m, UDL 12 kN/m DL and 8 kN/m LL, with a 40 kN point load at midspan. Section 250 mm wide by 450 mm deep.",
  },
  {
    variant: "beam",
    label: "2D beam · fixed cantilever",
    prompt:
      "Concrete cantilever beam, fixed at the left, span 4 m, 25 kN point load at 4 m from the left, DL 8 kN/m, LL 4 kN/m.",
  },
  {
    variant: "frame",
    label: "2D frame · 3 bays × 4 storeys",
    prompt:
      "2D RC moment frame, 3 bays of 6 m, 4 storeys at 3.5 m, DL 20 kN/m LL 8 kN/m on each beam, 25 kN lateral per floor.",
  },
  {
    variant: "building",
    label: "3D RC building · zone 3",
    prompt:
      "6-storey RC building, X-spans (6, 8, 6m), Y-spans (5, 5m), 3.5m storey heights, 4.5 kPa DL, 3 kPa LL, 200mm slab, 1.0 kPa wind, Seismic Zone 3, 200 kPa SBC.",
  },
  {
    variant: "building",
    label: "3D steel building · irregular",
    prompt:
      "5-storey structural steel frame, X-spans (8, 10, 8m), Y-spans (6, 6m), 3.8m storey heights, 3 kPa DL, 4 kPa LL, 1.2 kPa wind, 150 kPa SBC.",
  },
];

function analysisLabel(t?: string): string {
  if (t === "beam_2d") return "2D beam";
  if (t === "frame_2d") return "2D frame";
  return "3D building";
}

const APP_VERSION =
  typeof process !== "undefined" && process.env.NEXT_PUBLIC_APP_VERSION
    ? process.env.NEXT_PUBLIC_APP_VERSION
    : "0.2.0";

type ChatMsg =
  | { id: string; role: "user"; content: string }
  | { id: string; role: "assistant"; content: string; isError?: boolean };

function uid() {
  return `${Date.now()}-${Math.random().toString(36).slice(2, 9)}`;
}

export default function HomePage() {
  const [messages, setMessages] = useState<ChatMsg[]>([
    { id: "welcome", role: "assistant", content: WELCOME_ASSISTANT },
  ]);
  const [draft, setDraft] = useState("");
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState<FeaPromptResponse | null>(null);
  const [pDelta, setPDelta] = useState(false);
  const chatEndRef = useRef<HTMLDivElement>(null);
  const textareaRef = useRef<HTMLTextAreaElement>(null);

  const geometry: ViewerGeometry | null = result?.geometry ?? null;

  const beamMomentById = useMemo(() => {
    const m: Record<string, number> = {};
    if (!result?.beams) return m;
    for (const b of result.beams) m[b.id] = b.M_max_kNm;
    return m;
  }, [result]);

  const columnAxialById = useMemo(() => {
    const m: Record<string, number> = {};
    if (!result?.columns) return m;
    for (const c of result.columns) m[c.id] = c.P_max_kN;
    return m;
  }, [result]);

  const parsedForDiagrams: FeaParsedModel | null = useMemo(() => {
    const p = result?.parsed_model;
    if (!p) return null;
    if (p.analysis_type === "building_3d" || (!p.analysis_type && Array.isArray(p.spans_x_m))) {
      if (!Array.isArray(p.spans_x_m) || !Array.isArray(p.spans_y_m)) return null;
      if (p.spans_x_m.length < 1 || p.spans_y_m.length < 1) return null;
      return p as FeaParsedModel;
    }
    return null;
  }, [result]);

  const is3D = result?.analysis_type === "building_3d" || (!result?.analysis_type && !!parsedForDiagrams);

  useEffect(() => {
    chatEndRef.current?.scrollIntoView({ behavior: "smooth", block: "end" });
  }, [messages, loading]);

  const runAnalysis = useCallback(async () => {
    const text = draft.trim();
    if (!text || loading) return;
    setLoading(true);
    const userMsg: ChatMsg = { id: uid(), role: "user", content: text };
    setMessages((prev) => [...prev, userMsg]);
    setDraft("");
    try {
      const res = await analyzeFeaPrompt(text, { run_p_delta: pDelta });
      setResult(res);
      const assistantBody = `${res.input_summary}\n\n${res.summary_markdown}`;
      setMessages((prev) => [
        ...prev,
        {
          id: uid(),
          role: "assistant",
          content: assistantBody,
        },
      ]);
    } catch (e: unknown) {
      setResult(null);
      const msg = e instanceof Error ? e.message : "Request failed";
      setMessages((prev) => [
        ...prev,
        {
          id: uid(),
          role: "assistant",
          content: `**Analysis failed**\n\n${msg}`,
          isError: true,
        },
      ]);
    } finally {
      setLoading(false);
    }
  }, [draft, loading, pDelta]);

  return (
    <div className="page page-fea-chat">
      <header className="topbar">
        <div className="brand">
          <div className="brand-badge" aria-hidden />
          <div>
            <div className="brand-title">BALMORES STRUCTURAL</div>
            <div className="small-muted">
              Natural-language FEA · PyNite kernel · 2D beams · 2D frames · 3D buildings
            </div>
          </div>
        </div>
        <div style={{ display: "flex", alignItems: "center", gap: 12 }}>
          {result ? (
            <span className="analysis-type-badge" title="Detected from your message">
              {analysisLabel(result.analysis_type)}
            </span>
          ) : null}
          <label className="pdelta-toggle small-muted">
            <input type="checkbox" checked={pDelta} onChange={(e) => setPDelta(e.target.checked)} />
            P-Δ analysis
          </label>
        </div>
      </header>

      <div className="layout layout-fea-3">
        <section className="panel panel-chat panel-fea-chatonly panel-chat-gpt" aria-label="Design chat">
          <div className="panel-header">
            <strong>Chat</strong>
            <span className="small-muted">Natural language → PyNite model</span>
          </div>
          <div className="fea-chat-thread-wrap">
            <div className="chat-thread fea-chat-thread">
              {messages.map((m) => (
                <div key={m.id} className={`msg-row ${m.role === "user" ? "msg-row-user" : "msg-row-assistant"}`}>
                  <div
                    className={`msg-bubble msg ${m.role} ${m.role === "assistant" && "isError" in m && m.isError ? "msg-error" : ""}`}
                  >
                    <div className="msg-meta">
                      <small>{m.role === "user" ? "You" : "PyNite assistant"}</small>
                    </div>
                    <AssistantMarkdown content={m.content} streaming={false} />
                  </div>
                </div>
              ))}
              {loading ? (
                <div className="msg-row msg-row-assistant">
                  <div className="msg-bubble msg assistant fea-chat-thinking">
                    <small>Running PyNite FEM…</small>
                    <p className="small-muted" style={{ margin: "6px 0 0" }}>
                      Building nodes, members, load cases, and ULS combination.
                    </p>
                  </div>
                </div>
              ) : null}
              <div ref={chatEndRef} />
            </div>
          </div>
          <div className="fea-chat-composer">
            <div className="fea-quick-prompts">
              {QUICK_PROMPTS.map((q, i) => (
                <button
                  key={i}
                  type="button"
                  className="fea-chip"
                  data-variant={q.variant}
                  disabled={loading}
                  onClick={() => {
                    setDraft(q.prompt);
                    textareaRef.current?.focus();
                  }}
                >
                  {q.label}
                </button>
              ))}
            </div>
            <textarea
              ref={textareaRef}
              className="fea-chat-textarea fea-composer-input"
              value={draft}
              onChange={(e) => setDraft(e.target.value)}
              placeholder={PLACEHOLDER}
              rows={4}
              disabled={loading}
              onKeyDown={(e) => {
                if (e.key === "Enter" && !e.shiftKey && (e.ctrlKey || e.metaKey)) {
                  e.preventDefault();
                  void runAnalysis();
                }
              }}
            />
            <div className="fea-chat-actions">
              <button type="button" className="btn btn-send" disabled={loading || !draft.trim()} onClick={() => void runAnalysis()}>
                {loading ? "Running…" : "Send & analyze"}
              </button>
            </div>
            <p className="hint-line small-muted">Enter — new line · Ctrl+Enter — send · PyNite from project library</p>
          </div>
        </section>

        <section className="panel panel-viewer fea-panel-visual" aria-label="3D model and diagrams">
          <div className="panel-header">
            <strong>3D + diagrams</strong>
            <span className="small-muted">
              {result ? `${result.engine} · ${result.load_combination}` : "Orbit · scroll zoom"}
            </span>
          </div>
          <div className="fea-visual-stack">
            <div className="viewer-shell viewer-shell-split canvas-wrap">
              <ThreeViewer geometry={geometry} beamMomentById={beamMomentById} columnAxialById={columnAxialById} />
              <div className="overlay fea-3d-overlay">
                <div className="tag">PyNite FEM</div>
                {result ? (
                  <div className="fea-3d-legend small-muted">
                    <span>Beams: hue = |M| (top envelopes)</span>
                    <span>Columns: hue = |P|</span>
                  </div>
                ) : null}
              </div>
            </div>
            <div className="fea-diagrams-scroll">
              {is3D ? (
                <FeaDiagrams
                  parsed={parsedForDiagrams}
                  storeyDrifts={result?.storey_drifts ?? []}
                  totals={result?.totals ?? {}}
                  loadCombination={result?.load_combination ?? "—"}
                />
              ) : null}
              {result?.diagrams ? <BeamDiagrams diagrams={result.diagrams} /> : null}
            </div>
          </div>
        </section>

        <section className="panel panel-results panel-fea-report" aria-label="Analysis results">
          <div className="panel-header">
            <strong>PyNite output</strong>
            <span className="small-muted">{result ? result.load_combination : "—"}</span>
          </div>
          <div className="panel-body results-scroll fea-report-body">
            {!result && !loading ? (
              <p className="small-muted empty-hint">Send a building description to see parsed inputs, reactions, members, and drift.</p>
            ) : null}

            {result ? (
              <>
                <div className="report-section">
                  <h3 className="report-h">Interpreted inputs</h3>
                  <AssistantMarkdown content={result.input_summary} streaming={false} />
                  {result.parse_notes.length ? (
                    <ul className="report-notes">
                      {result.parse_notes.map((n, i) => (
                        <li key={i}>{n}</li>
                      ))}
                    </ul>
                  ) : null}
                </div>

                <div className="report-section">
                  <h3 className="report-h">Key quantities</h3>
                  <div className="result-grid fea-kpi-grid">
                    {result.result_cards.map((c) => (
                      <div key={c.label} className="card">
                        <div className="card-label">{c.label}</div>
                        <div className={`card-value ${c.tone ? `tone-${c.tone}` : ""}`}>
                          {c.value}
                          {c.unit ? ` ${c.unit}` : ""}
                        </div>
                      </div>
                    ))}
                  </div>
                </div>

                <div className="report-section">
                  <h3 className="report-h">Analysis summary</h3>
                  <AssistantMarkdown content={result.summary_markdown} streaming={false} />
                  {result.p_delta_note ? (
                    <p className="small-muted report-pdelta">{result.p_delta_note}</p>
                  ) : null}
                  <p className="small-muted">
                    Solver: <strong>Integrated PyNite FEM</strong> (open-source, MIT-licensed).
                  </p>
                </div>

                <div className="report-section">
                  <h3 className="report-h">Support reactions (ULS)</h3>
                  <div className="table-wrap table-scroll">
                    <table className="table table-striped table-compact">
                      <thead>
                        <tr>
                          <th>Node</th>
                          <th>x,y (m)</th>
                          <th>Rx</th>
                          <th>Ry</th>
                          <th>Rz</th>
                          <th>Mx</th>
                          <th>My</th>
                          <th>Mz</th>
                        </tr>
                      </thead>
                      <tbody>
                        {result.base_reactions.map((r) => (
                          <tr key={r.node}>
                            <td>
                              <code>{r.node}</code>
                            </td>
                            <td>
                              {r.x_m},{r.y_m}
                            </td>
                            <td>{r.Rx_kN}</td>
                            <td>{r.Ry_kN}</td>
                            <td>{r.Rz_kN}</td>
                            <td>{r.Mx_kNm}</td>
                            <td>{r.My_kNm}</td>
                            <td>{r.Mz_kNm}</td>
                          </tr>
                        ))}
                      </tbody>
                    </table>
                  </div>
                  <p className="small-muted">Forces kN, moments kN·m — PyNite sign convention.</p>
                </div>

                {result.storey_drifts.length ? (
                  <div className="report-section">
                    <h3 className="report-h">Storey drift (horizontal, ULS)</h3>
                    <div className="table-wrap">
                      <table className="table table-striped table-compact">
                        <thead>
                          <tr>
                            <th>Storey</th>
                            <th>z top (m)</th>
                            <th>h (m)</th>
                            <th>Max drift (mm)</th>
                            <th>Drift / h</th>
                          </tr>
                        </thead>
                        <tbody>
                          {result.storey_drifts.map((s) => (
                            <tr key={s.storey_index}>
                              <td>{s.storey_index}</td>
                              <td>{s.z_top_m}</td>
                              <td>{s.height_m}</td>
                              <td>{s.max_drift_mm}</td>
                              <td>{s.drift_ratio_h}</td>
                            </tr>
                          ))}
                        </tbody>
                      </table>
                    </div>
                  </div>
                ) : null}

                <div className="report-section">
                  <h3 className="report-h">Beams — envelope (|M|, |V|, deflection)</h3>
                  <div className="table-wrap table-scroll">
                    <table className="table table-striped table-compact">
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
                        {result.beams.map((b) => (
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
                  </div>
                </div>

                {result.columns.length ? (
                  <div className="report-section">
                    <h3 className="report-h">Columns — P, M, T envelope</h3>
                    <div className="table-wrap table-scroll">
                      <table className="table table-striped table-compact">
                        <thead>
                          <tr>
                            <th>Member</th>
                            <th>|P| (kN)</th>
                            <th>|My| (kN·m)</th>
                            <th>|Mz| (kN·m)</th>
                            <th>|T| (kN·m)</th>
                          </tr>
                        </thead>
                        <tbody>
                          {result.columns.map((c) => (
                            <tr key={c.id}>
                              <td>
                                <code>{c.id}</code>
                              </td>
                              <td>{c.P_max_kN}</td>
                              <td>{c.My_max_kNm}</td>
                              <td>{c.Mz_max_kNm}</td>
                              <td>{c.T_max_kNm}</td>
                            </tr>
                          ))}
                        </tbody>
                      </table>
                    </div>
                  </div>
                ) : null}

                <div className="report-section">
                  <h3 className="report-h">Assumptions & limits</h3>
                  <ul className="report-assumptions">
                    {result.assumptions.map((a, i) => (
                      <li key={i}>{a}</li>
                    ))}
                  </ul>
                  {result.totals.max_bearing_on_column_footing_kPa != null ? (
                    <p className="small-muted">
                      Rough column-only bearing estimate: ~{Number(result.totals.max_bearing_on_column_footing_kPa).toFixed(1)}{" "}
                      kPa vs specified SBC (if given). Not a footing design.
                    </p>
                  ) : null}
                </div>
              </>
            ) : null}
          </div>
        </section>
      </div>

      <div
        style={{
          padding: "28px 16px 10px",
          borderTop: "1px solid rgba(255,255,255,0.06)",
          background: "rgba(8,10,15,0.6)",
        }}
      >
        <ProfileBadges align="center" showOrcidPill />
      </div>

      <footer className="site-footer">
        <span>
          © 2026 Balmores Laboratory — Developed by{" "}
          <a href="/about" rel="author" style={{ color: "inherit", textDecoration: "underline", textUnderlineOffset: 3 }}>
            Louie Doniego Balmores
          </a>
        </span>
        <span className="footer-sep" aria-hidden />
        <span className="small-muted">Integrated PyNite open-source FEM · verify with your code</span>
        <span className="footer-sep" aria-hidden />
        <a href="/about" style={{ color: "inherit" }}>About</a>
        <span className="footer-sep" aria-hidden />
        <a href="/cv" style={{ color: "inherit" }}>CV</a>
        <span className="footer-sep" aria-hidden />
        <a href="/research" style={{ color: "inherit" }}>Research</a>
        <span className="footer-sep" aria-hidden />
        <span className="footer-ver">v{APP_VERSION}</span>
      </footer>
    </div>
  );
}
