"use client";

import dynamic from "next/dynamic";
import { useEffect, useRef, useState } from "react";
import AssistantMarkdown from "@/components/AssistantMarkdown";
import { analyzeFeaPrompt, type FeaPromptResponse, type ViewerGeometry } from "@/lib/api";

const ThreeViewer = dynamic(() => import("@/components/ThreeViewer"), {
  ssr: false,
  loading: () => (
    <div className="viewer-loading" role="status">
      Loading 3D preview…
    </div>
  ),
});

const PLACEHOLDER = `Example: Design a 5-storey 3D RC frame with X-spans (10, 12, 7m) and Y-spans (8, 10, 6, 5, 12m) at 5m heights, 4.5kPa DL, 3.0kPa LL, 200mm slabs, 1.2kPa wind, Seismic Zone 4, 150kPa SBC.`;

const APP_VERSION =
  typeof process !== "undefined" && process.env.NEXT_PUBLIC_APP_VERSION
    ? process.env.NEXT_PUBLIC_APP_VERSION
    : "0.2.0";

export default function HomePage() {
  const [prompt, setPrompt] = useState("");
  const [loading, setLoading] = useState(false);
  const [err, setErr] = useState<string | null>(null);
  const [result, setResult] = useState<FeaPromptResponse | null>(null);
  const [pDelta, setPDelta] = useState(true);
  const chatEndRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    chatEndRef.current?.scrollIntoView({ behavior: "smooth", block: "end" });
  }, [result, err, loading]);

  const geometry: ViewerGeometry | null = result?.geometry ?? null;

  async function onAnalyze() {
    const text = prompt.trim();
    if (!text || loading) return;
    setErr(null);
    setLoading(true);
    try {
      const res = await analyzeFeaPrompt(text, { run_p_delta: pDelta });
      setResult(res);
    } catch (e: unknown) {
      setResult(null);
      setErr(e instanceof Error ? e.message : "Request failed");
    } finally {
      setLoading(false);
    }
  }

  return (
    <div className="page page-fea-chat">
      <header className="topbar">
        <div className="brand">
          <div className="brand-badge" aria-hidden />
          <div>
            <div className="brand-title">BALMORES STRUCTURAL</div>
            <div className="small-muted">Describe the building · PyNite 3D FEA · design-oriented outputs</div>
          </div>
        </div>
        <label className="pdelta-toggle small-muted">
          <input type="checkbox" checked={pDelta} onChange={(e) => setPDelta(e.target.checked)} />
          P-Δ analysis
        </label>
      </header>

      <div className="layout layout-fea-3">
        <section className="panel panel-chat panel-fea-chatonly" aria-label="Design brief">
          <div className="panel-header">
            <strong>Your brief</strong>
            <span className="small-muted">Storeys, X/Y spans (m), kPa loads, slab mm, wind, zone, SBC</span>
          </div>
          <div className="fea-chat-scroll">
            <p className="small-muted fea-chat-intro">
              One message is parsed into an irregular 3D frame, analyzed in{" "}
              <strong>PyNite</strong> (DL/LL load cases, ULS combination, optional wind and simplified seismic
              lateral, optional P-Δ). Results appear in the right column.
            </p>
            <textarea
              className="fea-chat-textarea"
              value={prompt}
              onChange={(e) => setPrompt(e.target.value)}
              placeholder={PLACEHOLDER}
              rows={14}
              onKeyDown={(e) => {
                if (e.key === "Enter" && (e.ctrlKey || e.metaKey)) {
                  e.preventDefault();
                  void onAnalyze();
                }
              }}
            />
            {err ? <p className="export-err">{err}</p> : null}
            <div className="fea-chat-actions">
              <button type="button" className="btn btn-send" disabled={loading} onClick={() => void onAnalyze()}>
                {loading ? "Running PyNite…" : "Analyze structure"}
              </button>
            </div>
            <p className="hint-line small-muted">Ctrl+Enter — run · Esc — clear focus</p>
            <div ref={chatEndRef} />
          </div>
        </section>

        <section className="panel panel-viewer" aria-label="3D model">
          <div className="panel-header">
            <strong>3D frame</strong>
            <span className="small-muted">Orbit · scroll zoom</span>
          </div>
          <div className="viewer-shell viewer-shell-tall">
            <div className="canvas-wrap">
              <ThreeViewer geometry={geometry} />
              <div className="overlay">
                <div className="tag">PyNite FEM</div>
              </div>
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
              <p className="small-muted empty-hint">Run an analysis to see interpreted inputs, reactions, members, and drift.</p>
            ) : null}

            {result ? (
              <>
                <div className="report-section">
                  <h3 className="report-h">Input summary</h3>
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

      <footer className="site-footer">
        <span>Balmores Structural</span>
        <span className="footer-sep" aria-hidden />
        <span className="small-muted">PyNite open-source FEM · verify with your code</span>
        <span className="footer-sep" aria-hidden />
        <span className="footer-ver">v{APP_VERSION}</span>
      </footer>
    </div>
  );
}
