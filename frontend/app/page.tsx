"use client";

import { useMemo, useState } from "react";
import ThreeViewer from "@/components/ThreeViewer";
import BarMiniChart from "@/components/BarMiniChart";
import { sendChat, queueVerify } from "@/lib/api";

type Card = { label: string; value: string; unit?: string; tone?: string };
type Recommendation = { title: string; detail: string; severity: string };
type Message = { role: string; content: string };

const TABS = ["Results", "Physics", "Assumptions", "Member Schedule", "Recommendations", "Follow-ups"] as const;

const initialPrompt = `Design-check a 4-storey steel building, 12m x 18m, 3 bays by 3 bays, fc 30 MPa, fy 420 MPa, SBC 150 kPa, fixed supports, rigid diaphragm, braced frame, use Canada code. Give beam shear, beam moments, column axial, joint reactions, drift in mm, and recommended beam/column groups.`;
const COORDINATE_HINT = "Tip: Use natural language, coordinates (30x40m), or dimensions like '4 storey, 3x4 bays, 6m spans'.";

export default function HomePage() {
  const [projectId, setProjectId] = useState<string | null>(null);
  const [code, setCode] = useState("Canada");
  const [prompt, setPrompt] = useState(initialPrompt);
  const [loading, setLoading] = useState(false);
  const [messages, setMessages] = useState<Message[]>([
    { role: "assistant", content: "Describe your structure in natural language. You can also give dimensions, supports, materials, SBC, or coordinate-style input." },
  ]);
  const [assumptions, setAssumptions] = useState<string[]>([]);
  const [geometry, setGeometry] = useState<any>(null);
  const [cards, setCards] = useState<Card[]>([]);
  const [recommendations, setRecommendations] = useState<Recommendation[]>([]);
  const [questions, setQuestions] = useState<string[]>([]);
  const [confidence, setConfidence] = useState("Unknown");
  const [details, setDetails] = useState<any>(null);
  const [verifyStatus, setVerifyStatus] = useState<string>("Not started");
  const [activeTab, setActiveTab] = useState<(typeof TABS)[number]>("Results");

  const chartData = useMemo(() => {
    if (!details?.raw_predictions) return null;
    return {
      forceItems: [
        { label: "Beam Shear", value: details.raw_predictions.max_beam_shear_kN || 0, unit: "kN" },
        { label: "Beam Moment", value: details.raw_predictions.max_beam_moment_kNm || details.raw_predictions.beam_m3_grav_kNm || 0, unit: "kNm" },
        { label: "Column Axial", value: details.raw_predictions.max_column_axial_kN || details.raw_predictions.col_p_grav_kN || 0, unit: "kN" },
        { label: "Joint Reaction V", value: details.raw_predictions.max_joint_reaction_vertical_kN || 0, unit: "kN" },
      ],
      driftItems: [
        { label: "Max Drift", value: details.raw_predictions.max_drift_mm || 0, unit: "mm" },
        { label: "Roof Disp.", value: (details.raw_predictions.worst_roof_disp_m || 0) * 1000, unit: "mm" },
        { label: "Base Shear", value: details.raw_predictions.worst_base_shear_kN || 0, unit: "kN" },
      ],
    };
  }, [details]);

  async function onSend() {
    if (!prompt.trim()) return;
    setLoading(true);
    try {
      const res = await sendChat({ project_id: projectId, message: prompt, code });
      setProjectId(res.project_id);
      setMessages((prev) => [...prev, { role: "user", content: prompt }, ...res.messages]);
      setAssumptions(res.assumptions || []);
      setGeometry(res.geometry);
      setCards(res.result_cards || []);
      setRecommendations(res.recommendations || []);
      setQuestions(res.follow_up_questions || []);
      setConfidence(res.confidence || "Unknown");
      setDetails(res.detailed_results || null);
      setPrompt("");
    } catch (err: any) {
      setMessages((prev) => [...prev, { role: "assistant", content: `Error: ${err.message}` }]);
    } finally {
      setLoading(false);
    }
  }

  async function onVerify() {
    if (!projectId) return;
    const res = await queueVerify(projectId);
    setVerifyStatus(res.message);
  }

  function onQuestionClick(q: string) {
    setPrompt(q);
  }

  return (
    <div className="page">
      <div className="topbar">
        <div className="brand">
          <div className="brand-badge" />
            <div>
              <div>BALMORES STRUCTURAL</div>
              <div className="small-muted">Physics-informed · ETABS-calibrated · Instant structural analysis</div>
            </div>
        </div>
        <div className="small-muted">Confidence: {confidence}</div>
      </div>

      <div className="layout">
        <div className="panel">
          <div className="panel-header">
            <strong>Chat Input</strong>
            <span className="small-muted">Natural language / coordinates / assumptions</span>
          </div>
          <div className="chat-scroll">
            {messages.map((m, idx) => (
              <div key={`${m.role}-${idx}`} className={`msg ${m.role}`}>
                <small>{m.role === "user" ? "You" : "Structural Assistant"}</small>
                <div>{m.content}</div>
              </div>
            ))}
          </div>
          <div className="input-wrap">
            <div className="row">
              <select value={code} onChange={(e) => setCode(e.target.value)}>
                <option>Canada</option>
                <option>US</option>
                <option>Philippines</option>
              </select>
              <button className="btn" onClick={onVerify} disabled={!projectId}>
                Queue ETABS Verify
              </button>
            </div>
            <textarea
              value={prompt}
              onChange={(e) => setPrompt(e.target.value)}
              placeholder="Describe the building, loads, supports, code, fc/fy, SBC, coordinates, or ask a follow-up."
              onKeyDown={(e) => {
                if (e.key === "Enter" && !e.shiftKey) {
                  e.preventDefault();
                  onSend();
                }
              }}
            />
            <button className="btn" onClick={onSend} disabled={loading}>
              {loading ? "Thinking..." : "Send"}
            </button>
            <div className="small-muted">{COORDINATE_HINT}</div>
            <div className="small-muted">{verifyStatus}</div>
          </div>
        </div>

        <div className="panel">
          <div className="viewer-shell">
            <div className="canvas-wrap">
              <ThreeViewer geometry={geometry} />
              <div className="overlay">
                <div className="tag">3D Model</div>
                <div className="tag">Live browser output</div>
                <div className="tag">ETABS-calibrated</div>
              </div>
            </div>
            <div className="bottom-split">
              <BarMiniChart title="Beam / Column / Reaction Summary" items={chartData?.forceItems || []} />
              <BarMiniChart title="Drift / Roof / Base Summary" items={chartData?.driftItems || []} />
            </div>
          </div>
        </div>

        <div className="panel">
          <div className="panel-header">
            <strong>Results & Recommendations</strong>
            <span className="small-muted">Instant conceptual analysis</span>
          </div>
          <div className="tabs">
            {TABS.map((t) => (
              <button
                key={t}
                className={`tab ${activeTab === t ? "active" : ""}`}
                onClick={() => setActiveTab(t)}
              >
                {t}
              </button>
            ))}
          </div>
          <div className="panel-body" style={{ maxHeight: "calc(100vh - 220px)", overflow: "auto" }}>
            {activeTab === "Results" && (
              <div className="result-grid">
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
                  {details?.physics_checks && (
                    <>
                      <div className="physics-card">
                        <strong>Physics Checks</strong>
                        <div className="li">Base shear: {details.physics_checks.base_shear_check}</div>
                        <div className="li">Column axial: {details.physics_checks.column_axial_check}</div>
                        <div className="li">Drift compliance: {details.physics_checks.drift_compliance}</div>
                      </div>
                      <div className="physics-card">
                        <strong>Analysis Basis</strong>
                        <div className="li">{details.physics_checks.physics_basis}</div>
                        <div className="li">Units: {details.physics_checks.units}</div>
                        <div className="li small-muted">{details.physics_checks.equilibrium_note}</div>
                      </div>
                      {details?.model_info && (
                        <div className="physics-card small-muted">
                          <strong>Model</strong>
                          <div className="li">Trained on {details.model_info.dataset_rows} ETABS samples</div>
                          <div className="li">{details.model_info.training}</div>
                        </div>
                      )}
                    </>
                  )}
                </div>
                {!details?.physics_checks && (
                  <div className="li small-muted">Run analysis to see physics-informed checks.</div>
                )}
              </div>
            )}

            {activeTab === "Assumptions" && (
              <div className="section">
                <div className="assumption-list">
                  {assumptions.length > 0 ? (
                    assumptions.map((a, i) => <div key={i} className="li">{a}</div>)
                  ) : (
                    <div className="li small-muted">No assumptions yet. Send a message to analyze.</div>
                  )}
                </div>
              </div>
            )}

            {activeTab === "Member Schedule" && (
              <div className="section">
                <table className="table">
                  <thead>
                    <tr>
                      <th>Group</th>
                      <th>Scope</th>
                      <th>Design Basis</th>
                      <th>Suggested Section</th>
                    </tr>
                  </thead>
                  <tbody>
                    {details?.member_schedule?.beam_groups?.map((g: any) => (
                      <tr key={g.group}>
                        <td>{g.group}</td>
                        <td>{g.scope}</td>
                        <td>{g.design_basis}</td>
                        <td>{g.suggested_section}</td>
                      </tr>
                    ))}
                    {details?.member_schedule?.column_groups?.map((g: any) => (
                      <tr key={g.group}>
                        <td>{g.group}</td>
                        <td>{g.scope}</td>
                        <td>{g.design_basis}</td>
                        <td>{g.suggested_section}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
                {!details?.member_schedule?.beam_groups?.length && !details?.member_schedule?.column_groups?.length && (
                  <div className="li small-muted">No member schedule yet. Send a message to analyze.</div>
                )}
              </div>
            )}

            {activeTab === "Recommendations" && (
              <div className="section">
                <div className="recommendation-list">
                  {recommendations.length > 0 ? (
                    recommendations.map((r, i) => (
                      <div key={i} className="li">
                        <strong>{r.title}</strong>
                        <div className="small-muted" style={{ marginTop: 6 }}>{r.detail}</div>
                      </div>
                    ))
                  ) : (
                    <div className="li small-muted">No recommendations yet. Send a message to analyze.</div>
                  )}
                </div>
              </div>
            )}

            {activeTab === "Follow-ups" && (
              <div className="section">
                <p className="small-muted" style={{ marginBottom: 10 }}>
                  Click a question to add it to the input.
                </p>
                <div className="question-list">
                  {questions.length > 0 ? (
                    questions.map((q, i) => (
                      <div
                        key={i}
                        className="li question-chip"
                        onClick={() => onQuestionClick(q)}
                      >
                        {q}
                      </div>
                    ))
                  ) : (
                    <div className="li small-muted">No follow-up questions yet. Send a message to analyze.</div>
                  )}
                </div>
              </div>
            )}
          </div>
        </div>
      </div>
    </div>
  );
}
