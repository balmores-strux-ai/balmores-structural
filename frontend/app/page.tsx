"use client";

import dynamic from "next/dynamic";
import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import AssistantMarkdown from "@/components/AssistantMarkdown";
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

const PLACEHOLDER = `Example: 5-storey RC frame, X-spans (10, 12, 7m), Y-spans (8, 10, 6, 5, 12m), 5m storey heights, 4.5 kPa DL, 3.0 kPa LL, 200mm slab, 1.2 kPa wind, Seismic Zone 4, 150 kPa SBC.`;

const WELCOME_ASSISTANT = `**PyNite structural chat**

Describe a building like you would to a colleague: **number of storeys**, **X- and Y-bay spans in metres**, **DL / LL in kPa**, optional **slab thickness (mm)**, **wind (kPa)**, **seismic zone**, and **SBC (kPa)**.

I parse your message into a **3D irregular frame**, run **PyNite** from this project’s FEM library, and show **3D geometry** (members coloured by demand), **plan / elevation schematics**, **drift**, and **reactions**.

Tip: include explicit span lists, e.g. \`X-spans (6, 8, 6m) and Y-spans (5, 5m)\`.`;

const QUICK_PROMPTS = [
  "4-storey RC building, X-spans (6, 6, 6m), Y-spans (5, 5m), 4m storey heights, 4.5 kPa DL, 2 kPa LL, 200mm slab, Seismic Zone 3.",
  "5-storey steel frame, X-spans (8, 10, 8m), Y-spans (6, 6m), 3.5m floors, 3 kPa DL, 4 kPa LL, 1.0 kPa wind, 200 kPa SBC.",
];

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
    if (!p || !Array.isArray(p.spans_x_m) || !Array.isArray(p.spans_y_m)) return null;
    if (p.spans_x_m.length < 1 || p.spans_y_m.length < 1) return null;
    return p as FeaParsedModel;
  }, [result]);

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
            <div className="small-muted">Chat → PyNite 3D FEA · diagrams · coloured demand map</div>
          </div>
        </div>
        <label className="pdelta-toggle small-muted">
          <input type="checkbox" checked={pDelta} onChange={(e) => setPDelta(e.target.checked)} />
          P-Δ analysis
        </label>
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
                  disabled={loading}
                  onClick={() => {
                    setDraft(q);
                    textareaRef.current?.focus();
                  }}
                >
                  Quick example {i + 1}
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
              <FeaDiagrams
                parsed={parsedForDiagrams}
                storeyDrifts={result?.storey_drifts ?? []}
                totals={result?.totals ?? {}}
                loadCombination={result?.load_combination ?? "—"}
              />
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
                  <p className="small-muted report-pdelta">{result.p_delta_note}</p>
                  {result.pynite_path ? (
                    <p className="small-muted">FEM source: <code className="fea-code-path">{result.pynite_path}</code></p>
                  ) : null}
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
        <span className="small-muted">PyNite open-source FEM · verify with your code</span>
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
