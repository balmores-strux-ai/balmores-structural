"use client";

import { useCallback, useEffect, useRef, useState } from "react";
import AnalysisProgress from "@/components/AnalysisProgress";
import AssistantMarkdown from "@/components/AssistantMarkdown";
import BeamDiagrams, { type DiagramVisibility } from "@/components/BeamDiagrams";
import BuildingSupportLoadPanel from "@/components/BuildingSupportLoadPanel";
import DesignCriteriaCard from "@/components/DesignCriteriaCard";
import ProfileBadges from "@/components/ProfileBadges";
import PromptInputPreview from "@/components/PromptInputPreview";
import SteelSectionAdvisor from "@/components/SteelSectionAdvisor";
import StructureModel2D from "@/components/StructureModel2D";
import StructureModel3D from "@/components/StructureModel3D";
import { downloadFeaEtabsExports } from "@/lib/exportFeaEtabs";
import { downloadAndTryOpenDocx, feaResultToDocxBlob } from "@/lib/exportFeaDocx";
import {
  analyzeFeaPromptStream,
  type FeaProgressEvent,
  type FeaPromptResponse,
} from "@/lib/api";

const PLACEHOLDER = `Examples you can paste:

- Simply supported steel beam, span 8 m, UDL 15 kN/m DL, 10 kN/m LL, 40 kN point load at midspan.
- Continuous concrete beam, 4 spans of 6 m, DL 12 kN/m, LL 8 kN/m.
- 2D RC moment frame, 3 bays of 6 m, 4 storeys at 3.5 m, DL 20 kN/m, LL 8 kN/m, 25 kN lateral per floor.
- 30-storey RC building in Manila, X-spans (6, 8, 6m), Y-spans (5, 5m), 3.5m storey heights, 4.5 kPa DL, 3 kPa LL, 200mm slab.
   (Wind, seismic zone and SBC are auto-resolved from the location.)`;

const TYPING_PREVIEWS = [
  "30-storey RC tower in Cebu, X-spans (6, 8, 12, 8, 6m), Y-spans (5, 9, 9, 5m), 3.5m storey heights, 4.5 kPa DL, 3 kPa LL, 200mm slab.",
  "2D steel moment frame, 5 bays of 7 m, 6 storeys at 3.6 m, DL 18 kN/m, LL 10 kN/m on every beam, 35 kN wind per floor.",
  "Continuous concrete beam, spans (6, 8, 10, 8, 6 m), 6 supports. DL 22 kN/m, LL 18 kN/m. Left and right ends fixed.",
  "25-storey structural steel tower in Makati, X-spans (8, 10, 8m), Y-spans (6, 6m), 3.8m storey heights, 3 kPa DL, 4 kPa LL.",
];

const WELCOME_ASSISTANT = `**Balmores Structural - PyNite assistant**

I am powered by the **open-source PyNite** finite-element library, integrated directly into this app. Tell me about a structure in plain English and I will build, analyse, and report it. I support:

1. **2D beams** - simply supported, fixed-fixed, cantilevers, **continuous beams with 2 / 3 / 4 / 5 supports**.
2. **2D moment frames** - bay spans, storey heights, gravity + lateral.
3. **3D buildings** - irregular X / Y spans, up to **60 storeys**, P-Delta, drift, base reactions.

**Type a city in the Philippines** (e.g. *"in Manila"*, *"in Cebu"*, *"in Quezon City"*, *"in Davao"*) and I will look up the local **wind speed**, **seismic zone**, **PGA**, **SBC** and **NSCP 2015** context - every assumption is shown in the design-criteria table on the right.

Tip: include explicit numbers and units (e.g. \`UDL 15 kN/m\`, \`X-spans (6, 8, 6m)\`).`;

type QuickPrompt = {
  variant: "beam" | "frame" | "building";
  label: string;
  prompt: string;
  sampleKey: string;
};

const QUICK_PROMPTS: QuickPrompt[] = [
  {
    variant: "beam",
    label: "Beam - simply supported, point load",
    sampleKey: "beam-simple-point",
    prompt:
      "Simply supported steel beam, span 8 m, UDL 12 kN/m DL and 8 kN/m LL, with a 40 kN point load at midspan. Section 250 mm wide by 450 mm deep.",
  },
  {
    variant: "beam",
    label: "Beam - fixed cantilever (4 m)",
    sampleKey: "beam-fixed-cantilever",
    prompt:
      "Concrete cantilever beam, fixed at the left, span 4 m, 25 kN point load at 4 m from the left, DL 8 kN/m, LL 4 kN/m.",
  },
  {
    variant: "beam",
    label: "Continuous - 2 spans of 6 m",
    sampleKey: "cont-2x6",
    prompt:
      "Continuous concrete beam, 2 spans of 6 m (3 supports), DL 15 kN/m, LL 10 kN/m. Beam 300 x 600 mm.",
  },
  {
    variant: "beam",
    label: "Continuous - 3 spans (5, 6, 5 m)",
    sampleKey: "cont-5-6-5",
    prompt:
      "Continuous steel beam with 3 spans of 5, 6, 5 m (4 supports). DL 18 kN/m, LL 12 kN/m, 50 kN point load at 8.5 m from the left.",
  },
  {
    variant: "beam",
    label: "Continuous - 4 spans of 7 m",
    sampleKey: "cont-4x7",
    prompt:
      "Continuous reinforced-concrete beam, 4 spans of 7 m (5 supports), DL 20 kN/m, LL 15 kN/m. Beam 350 x 700 mm.",
  },
  {
    variant: "beam",
    label: "Continuous - 5 spans (6, 8, 10, 8, 6 m)",
    sampleKey: "cont-6-8-10-8-6",
    prompt:
      "Continuous concrete beam, spans (6, 8, 10, 8, 6 m), 6 supports. DL 22 kN/m, LL 18 kN/m. Beam 400 x 800 mm. Left and right ends fixed.",
  },

  {
    variant: "frame",
    label: "Frame - 3 bays x 4 storeys (RC)",
    sampleKey: "frame-3bay-4sty-rc",
    prompt:
      "2D RC moment frame, 3 bays of 6 m, 4 storeys at 3.5 m, DL 20 kN/m LL 8 kN/m on each beam, 25 kN lateral per floor.",
  },
  {
    variant: "frame",
    label: "Frame - 5 bays x 6 storeys (steel)",
    sampleKey: "frame-5bay-6sty-steel",
    prompt:
      "2D structural steel moment frame, 5 bays of 7 m, 6 storeys at 3.6 m, DL 18 kN/m, LL 10 kN/m on every beam, 35 kN wind per floor.",
  },
  {
    variant: "frame",
    label: "Frame - single-bay portal x 2 storeys",
    sampleKey: "frame-portal-2sty",
    prompt:
      "2D RC portal frame, 1 bay of 8 m, 2 storeys at 4 m, DL 25 kN/m, LL 12 kN/m, 40 kN lateral per floor.",
  },
  {
    variant: "frame",
    label: "Frame - industrial 4 bays x 1 storey",
    sampleKey: "frame-industrial-4bay",
    prompt:
      "2D steel moment frame, 4 bays of 9 m, single storey 6 m high, DL 8 kN/m, LL 6 kN/m, 60 kN wind per floor.",
  },

  {
    variant: "building",
    label: "Building - 6-storey RC, Manila",
    sampleKey: "building-6-manila",
    prompt:
      "6-storey RC building in Manila, X-spans (6, 8, 6m), Y-spans (5, 5m), 3.5m storey heights, 4.5 kPa DL, 3 kPa LL, 200mm slab.",
  },
  {
    variant: "building",
    label: "Building - 30-storey RC, Cebu",
    sampleKey: "building-30-cebu",
    prompt:
      "30-storey RC tower in Cebu, X-spans (6, 8, 12, 8, 6m), Y-spans (5, 9, 9, 5m), 3.5m storey heights, 4.5 kPa DL, 3 kPa LL, 200mm slab.",
  },
  {
    variant: "building",
    label: "Building - 25-storey steel, Makati",
    sampleKey: "building-25-makati",
    prompt:
      "25-storey structural steel tower in Makati, X-spans (8, 10, 8m), Y-spans (6, 6m), 3.8m storey heights, 3 kPa DL, 4 kPa LL.",
  },
  {
    variant: "building",
    label: "Building - 12-storey RC, Iloilo",
    sampleKey: "building-12-iloilo",
    prompt:
      "12-storey RC building in Iloilo, X-spans (7, 7, 7m), Y-spans (6, 6m), 3.6m storey heights, 4 kPa DL, 3 kPa LL, 180 mm slab.",
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
    : "0.6.0";

type ChatMsg =
  | { id: string; role: "user"; content: string }
  | { id: string; role: "assistant"; content: string; isError?: boolean };

function uid() {
  return `${Date.now()}-${Math.random().toString(36).slice(2, 9)}`;
}

function fmtNum(value: unknown, decimals = 2): string {
  if (value === null || value === undefined) return "-";
  const n = typeof value === "number" ? value : Number(value);
  if (!Number.isFinite(n)) return "-";
  return n.toLocaleString(undefined, {
    minimumFractionDigits: decimals,
    maximumFractionDigits: decimals,
  });
}

function fmtDriftRatio(r: unknown): string {
  const n = typeof r === "number" ? r : Number(r);
  if (!Number.isFinite(n) || n <= 0) return "-";
  return `1/${Math.round(1 / n).toLocaleString()}`;
}

type SampleBundle = {
  version: number;
  defaultKey: string;
  samples: Record<string, FeaPromptResponse>;
};

async function loadSampleBundle(): Promise<SampleBundle | null> {
  try {
    const r = await fetch("/sample-results.json", { cache: "force-cache" });
    return r.ok ? ((await r.json()) as SampleBundle) : null;
  } catch {
    return null;
  }
}

function shouldUseInstantDemo(text: string): boolean {
  const t = text.toLowerCase();
  return /\b60[\s-]*(storey|story|floor)\b/.test(t) || /\btaipei\b/.test(t);
}

export default function HomePage() {
  const [messages, setMessages] = useState<ChatMsg[]>([
    { id: "welcome", role: "assistant", content: WELCOME_ASSISTANT },
  ]);
  const [draft, setDraft] = useState("");
  const [typedPlaceholder, setTypedPlaceholder] = useState("");
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState<FeaPromptResponse | null>(null);
  const [sampleBundle, setSampleBundle] = useState<SampleBundle | null>(null);
  const [pDelta, setPDelta] = useState(false);
  const [progressEvent, setProgressEvent] = useState<FeaProgressEvent | null>(null);
  const [demoMode, setDemoMode] = useState(true);
  const [exportingDoc, setExportingDoc] = useState(false);
  const [showReactArrows, setShowReactArrows] = useState(true);
  const [showLoadArrows, setShowLoadArrows] = useState(true);
  const [diagVis, setDiagVis] = useState<DiagramVisibility>({
    beamShear: true,
    beamMoment: true,
    beamDeflection: true,
    frameMoment: true,
    frameShear: true,
  });
  const chatEndRef = useRef<HTMLDivElement>(null);
  const textareaRef = useRef<HTMLTextAreaElement>(null);

  useEffect(() => {
    let cancelled = false;
    loadSampleBundle()
      .then((bundle) => {
        if (!cancelled && bundle) {
          setSampleBundle(bundle);
          setResult(bundle.samples[bundle.defaultKey] ?? null);
          setDemoMode(true);
        }
      })
      .catch(() => {});
    return () => {
      cancelled = true;
    };
  }, []);

  useEffect(() => {
    chatEndRef.current?.scrollIntoView({ behavior: "smooth", block: "end" });
  }, [messages, loading]);

  useEffect(() => {
    let idx = 0;
    let pos = 0;
    let pause = 0;
    const timer = window.setInterval(() => {
      const phrase = TYPING_PREVIEWS[idx % TYPING_PREVIEWS.length];
      if (pos <= phrase.length) {
        setTypedPlaceholder(phrase.slice(0, pos));
        pos += 2;
      } else if (pause < 10) {
        pause += 1;
      } else {
        idx += 1;
        pos = 0;
        pause = 0;
      }
    }, 55);
    return () => window.clearInterval(timer);
  }, []);

  const runAnalysis = useCallback(async () => {
    const text = draft.trim();
    if (!text || loading) return;
    setLoading(true);
    setProgressEvent(null);
    const userMsg: ChatMsg = { id: uid(), role: "user", content: text };
    setMessages((prev) => [...prev, userMsg]);
    setDraft("");
    try {
      if (shouldUseInstantDemo(text)) {
        const bundle = sampleBundle ?? (await loadSampleBundle());
        const demo = bundle?.samples[bundle.defaultKey];
        if (demo) {
          if (bundle) setSampleBundle(bundle);
          setResult(demo);
          setDemoMode(true);
          setMessages((prev) => [
            ...prev,
            {
              id: uid(),
              role: "assistant",
              content:
                "**Instant verified showcase loaded**\n\nThe 60-storey stress-test is intentionally not used as the default because its preliminary bare-frame drift is not appropriate for client presentation without a core/wall system. For a polished and faster first impression, I loaded the precomputed **30-storey RC tower in Cebu, Philippines** sample with full tables, drift, reactions, beams, columns, design criteria, handcalcs, ETABS export, and Word report.\n\nPhilippines-only location mode is active for this release while the NSCP dataset is polished.",
            },
          ]);
          return;
        }
      }
      const res = await analyzeFeaPromptStream(text, {
        run_p_delta: pDelta,
        onProgress: (ev) => setProgressEvent(ev),
      });
      setResult(res);
      setDemoMode(false);
      const elapsedSeconds =
        typeof res.elapsed_ms === "number" ? (res.elapsed_ms / 1000).toFixed(1) : null;
      const assistantBody = `${res.input_summary}\n\n${res.summary_markdown}${
        elapsedSeconds ? `\n\n_Solved in ${elapsedSeconds} s by the integrated PyNite kernel._` : ""
      }`;
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
      setProgressEvent(null);
    }
  }, [draft, loading, pDelta, sampleBundle]);

  const has2DDiagrams =
    !!result?.diagrams &&
    (result.analysis_type === "beam_2d" || result.analysis_type === "frame_2d");

  const has2DModelView =
    result && (result.analysis_type === "beam_2d" || result.analysis_type === "frame_2d");

  const exportWord = useCallback(async () => {
    if (!result) return;
    setExportingDoc(true);
    try {
      const blob = await feaResultToDocxBlob(
        result,
        `Balmores_Structural_${result.analysis_type}_${result.load_combination}`,
      );
      downloadAndTryOpenDocx(
        blob,
        `Balmores_Structural_${result.load_combination.replace(/\s+/g, "_")}.docx`,
      );
    } finally {
      setExportingDoc(false);
    }
  }, [result]);

  return (
    <div className="page page-fea-chat">
      <header className="topbar">
        <div className="brand">
          <div className="brand-badge" aria-hidden />
          <div>
            <div className="brand-title">BALMORES STRUCTURAL</div>
            <div className="small-muted">
              Natural-language FEA - PyNite kernel - 2D beams - continuous beams - 2D frames - 3D buildings
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
            P-Delta analysis
          </label>
        </div>
      </header>

      <div className="layout layout-fea-2">
        <section className="panel panel-chat panel-fea-chatonly panel-chat-gpt" aria-label="Design chat">
          <div className="panel-header">
            <strong>Chat</strong>
            <span className="small-muted">Natural language to PyNite model</span>
          </div>
          <div className="fea-chat-thread-wrap">
            <div className="chat-thread fea-chat-thread">
              {messages.map((m) => (
                <div
                  key={m.id}
                  className={`msg-row ${m.role === "user" ? "msg-row-user" : "msg-row-assistant"}`}
                >
                  <div
                    className={`msg-bubble msg ${m.role} ${
                      m.role === "assistant" && "isError" in m && m.isError ? "msg-error" : ""
                    }`}
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
                    <AnalysisProgress event={progressEvent} active={loading} fallbackTotal={6} />
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
                    const sample = sampleBundle?.samples[q.sampleKey];
                    if (sample) {
                      setResult(sample);
                      setDemoMode(true);
                    }
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
              placeholder={typedPlaceholder || PLACEHOLDER}
              rows={4}
              disabled={loading}
              onKeyDown={(e) => {
                if (e.key === "Enter" && !e.shiftKey && (e.ctrlKey || e.metaKey)) {
                  e.preventDefault();
                  void runAnalysis();
                }
              }}
            />
            <PromptInputPreview text={draft} />
            <div className="fea-chat-actions">
              <button
                type="button"
                className="btn btn-send"
                disabled={loading || !draft.trim()}
                onClick={() => void runAnalysis()}
              >
                {loading ? "Running..." : "Send & analyze"}
              </button>
            </div>
            <p className="hint-line small-muted">
              Enter - new line - Ctrl+Enter - send - type a Philippine city for NSCP wind / seismic / SBC
            </p>
          </div>
        </section>

        <section className="panel panel-results panel-fea-report" aria-label="Analysis results">
          <div className="panel-header fea-report-header">
            <div>
              <strong>PyNite output</strong>
              <span className="small-muted">
                {result ? ` - ${result.engine} - ${result.load_combination}` : "-"}
              </span>
            </div>
            <div className="fea-report-header-actions">
              {result ? (
                <>
                  <button
                    type="button"
                    className="btn btn-export"
                    disabled={exportingDoc}
                    onClick={() => void exportWord()}
                    title="Export full tables and narrative to Microsoft Word (.docx)"
                  >
                    {exportingDoc ? "Preparing..." : "Export report (.docx)"}
                  </button>
                  <button
                    type="button"
                    className="btn btn-ghost"
                    onClick={() => downloadFeaEtabsExports(result)}
                    title="Download ETABS-oriented text and JSON model files from this result"
                  >
                    ETABS model (.txt/.json)
                  </button>
                  {demoMode ? (
                    <button
                      type="button"
                      className="btn btn-ghost"
                      onClick={() => {
                        setResult(null);
                        setDemoMode(false);
                      }}
                    >
                      Clear sample
                    </button>
                  ) : null}
                </>
              ) : null}
            </div>
          </div>
          <div className="panel-body results-scroll fea-report-body">
            {!result && !loading ? (
              <p className="small-muted empty-hint">
                Send a building description to see parsed inputs, design criteria, reactions,
                members, and drift. A 60-storey sample may load on open.
              </p>
            ) : null}

            {result ? (
              <>
                {demoMode ? (
                  <div className="demo-banner" role="status">
                    <span className="demo-banner-badge">Sample</span>
                    <span>
                      Preloaded instant PyNite sample output so users see results immediately.
                      Select any example chip to swap the prompt text and report instantly.
                    </span>
                  </div>
                ) : null}
                {result.design_criteria ? (
                  <div className="report-section">
                    <DesignCriteriaCard criteria={result.design_criteria} />
                  </div>
                ) : null}

                {(has2DModelView || result.analysis_type === "building_3d") ? (
                  <div className="report-section">
                    <div className="report-h-row">
                      <h3 className="report-h">
                        {has2DModelView ? "2D model perspective" : "3D model support/load view"}
                      </h3>
                      {has2DModelView ? (
                        <span className="report-count small-muted">
                          {result.geometry.nodes.length} nodes - {result.geometry.members.length} members
                        </span>
                      ) : null}
                    </div>
                    <div className="model-view-controls">
                      {has2DModelView ? (
                        <>
                          <label>
                            <input
                              type="checkbox"
                              checked={showReactArrows}
                              onChange={(e) => setShowReactArrows(e.target.checked)}
                            />
                            Reaction arrows
                          </label>
                          <label>
                            <input
                              type="checkbox"
                              checked={showLoadArrows}
                              onChange={(e) => setShowLoadArrows(e.target.checked)}
                            />
                            Applied-load arrows
                          </label>
                          <label>
                            <input
                              type="checkbox"
                              checked={diagVis.beamShear || diagVis.frameShear}
                              onChange={(e) =>
                                setDiagVis((v) => ({
                                  ...v,
                                  beamShear: e.target.checked,
                                  frameShear: e.target.checked,
                                }))
                              }
                            />
                            Shear diagram
                          </label>
                          <label>
                            <input
                              type="checkbox"
                              checked={diagVis.beamMoment || diagVis.frameMoment}
                              onChange={(e) =>
                                setDiagVis((v) => ({
                                  ...v,
                                  beamMoment: e.target.checked,
                                  frameMoment: e.target.checked,
                                }))
                              }
                            />
                            Moment diagram
                          </label>
                          {result.analysis_type === "beam_2d" ? (
                            <label>
                              <input
                                type="checkbox"
                                checked={diagVis.beamDeflection}
                                onChange={(e) =>
                                  setDiagVis((v) => ({ ...v, beamDeflection: e.target.checked }))
                                }
                              />
                              Deflection diagram
                            </label>
                          ) : null}
                        </>
                      ) : null}
                    </div>
                    {has2DModelView ? (
                      <StructureModel2D
                        result={result}
                        showReactions={showReactArrows}
                        showLoads={showLoadArrows}
                      />
                    ) : (
                      <>
                        <StructureModel3D result={result} />
                        <BuildingSupportLoadPanel result={result} />
                      </>
                    )}
                  </div>
                ) : null}

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
                  <SteelSectionAdvisor result={result} />
                </div>

                {has2DDiagrams && result.diagrams ? (
                  <div className="report-section">
                    <h3 className="report-h">Shear / moment / deflection</h3>
                    <BeamDiagrams diagrams={result.diagrams} visibility={diagVis} />
                  </div>
                ) : null}

                <div className="report-section">
                  <h3 className="report-h">Analysis summary</h3>
                  <AssistantMarkdown content={result.summary_markdown} streaming={false} />
                  {result.p_delta_note ? (
                    <p className="small-muted report-pdelta">{result.p_delta_note}</p>
                  ) : null}
                  <p className="small-muted">
                    Solver: <strong>Integrated PyNite FEM</strong> (open-source, MIT-licensed).
                    {typeof result.elapsed_ms === "number"
                      ? ` Wall-clock: ${(result.elapsed_ms / 1000).toFixed(2)} s.`
                      : ""}
                  </p>
                </div>

                {result.base_reactions.length ? (
                  <div className="report-section">
                    <div className="report-h-row">
                      <h3 className="report-h">Support reactions (ULS)</h3>
                      <span className="report-count small-muted">
                        {result.base_reactions.length} nodes
                      </span>
                    </div>
                    <div className="table-wrap table-scroll">
                      <table className="table table-striped table-compact table-numeric">
                        <thead>
                          <tr>
                            <th>Node</th>
                            <th className="col-coord">x, y (m)</th>
                            <th className="col-num">Rx (kN)</th>
                            <th className="col-num">Ry (kN)</th>
                            <th className="col-num">Rz (kN)</th>
                            <th className="col-num">Mx (kN-m)</th>
                            <th className="col-num">My (kN-m)</th>
                            <th className="col-num">Mz (kN-m)</th>
                          </tr>
                        </thead>
                        <tbody>
                          {result.base_reactions.map((r) => (
                            <tr key={r.node}>
                              <td>
                                <code>{r.node}</code>
                              </td>
                              <td className="col-coord">
                                {fmtNum(r.x_m, 2)}, {fmtNum(r.y_m, 2)}
                              </td>
                              <td className="col-num">{fmtNum(r.Rx_kN)}</td>
                              <td className="col-num">{fmtNum(r.Ry_kN)}</td>
                              <td className="col-num">{fmtNum(r.Rz_kN)}</td>
                              <td className="col-num">{fmtNum(r.Mx_kNm)}</td>
                              <td className="col-num">{fmtNum(r.My_kNm)}</td>
                              <td className="col-num">{fmtNum(r.Mz_kNm)}</td>
                            </tr>
                          ))}
                        </tbody>
                      </table>
                    </div>
                    <p className="small-muted">Forces kN, moments kN-m - PyNite sign convention.</p>
                  </div>
                ) : null}

                {result.storey_drifts.length ? (
                  <div className="report-section">
                    <div className="report-h-row">
                      <h3 className="report-h">Storey drift (horizontal, ULS)</h3>
                      <span className="report-count small-muted">
                        {result.storey_drifts.length} storeys
                      </span>
                    </div>
                    <div className="table-wrap">
                      <table className="table table-striped table-compact table-numeric">
                        <thead>
                          <tr>
                            <th className="col-num">Storey</th>
                            <th className="col-num">z top (m)</th>
                            <th className="col-num">h (m)</th>
                            <th className="col-num">Max drift (mm)</th>
                            <th className="col-num">Drift ratio</th>
                          </tr>
                        </thead>
                        <tbody>
                          {result.storey_drifts.map((s) => (
                            <tr key={s.storey_index}>
                              <td className="col-num">{s.storey_index}</td>
                              <td className="col-num">{fmtNum(s.z_top_m, 2)}</td>
                              <td className="col-num">{fmtNum(s.height_m, 2)}</td>
                              <td className="col-num">{fmtNum(s.max_drift_mm, 2)}</td>
                              <td className="col-num">{fmtDriftRatio(s.drift_ratio_h)}</td>
                            </tr>
                          ))}
                        </tbody>
                      </table>
                    </div>
                    <p className="small-muted">
                      Drift ratio reported as 1 / N of storey height (smaller denominators = softer).
                    </p>
                  </div>
                ) : null}

                {result.beams.length ? (
                  <div className="report-section">
                    <div className="report-h-row">
                      <h3 className="report-h">Beams - envelope (|M|, |V|, deflection)</h3>
                      <span className="report-count small-muted">
                        {result.beams.length} rows (sorted by |M|)
                      </span>
                    </div>
                    <div className="table-wrap table-scroll">
                      <table className="table table-striped table-compact table-numeric">
                        <thead>
                          <tr>
                            <th>Member</th>
                            <th className="col-num">z (m)</th>
                            <th className="col-num">|M| (kN-m)</th>
                            <th className="col-num">|V| (kN)</th>
                            <th className="col-num">delta (mm)</th>
                          </tr>
                        </thead>
                        <tbody>
                          {result.beams.map((b) => (
                            <tr key={b.id}>
                              <td>
                                <code>{b.id}</code>
                              </td>
                              <td className="col-num">{fmtNum(b.floor_z_m, 2)}</td>
                              <td className="col-num">{fmtNum(b.M_max_kNm)}</td>
                              <td className="col-num">{fmtNum(b.V_max_kN)}</td>
                              <td className="col-num">{fmtNum(b.deflection_mm)}</td>
                            </tr>
                          ))}
                        </tbody>
                      </table>
                    </div>
                  </div>
                ) : null}

                {result.columns.length ? (
                  <div className="report-section">
                    <div className="report-h-row">
                      <h3 className="report-h">Columns - P, M, T envelope</h3>
                      <span className="report-count small-muted">
                        {result.columns.length} rows (sorted by |P|)
                      </span>
                    </div>
                    <div className="table-wrap table-scroll">
                      <table className="table table-striped table-compact table-numeric">
                        <thead>
                          <tr>
                            <th>Member</th>
                            <th className="col-num">|P| (kN)</th>
                            <th className="col-num">|My| (kN-m)</th>
                            <th className="col-num">|Mz| (kN-m)</th>
                            <th className="col-num">|T| (kN-m)</th>
                          </tr>
                        </thead>
                        <tbody>
                          {result.columns.map((c) => (
                            <tr key={c.id}>
                              <td>
                                <code>{c.id}</code>
                              </td>
                              <td className="col-num">{fmtNum(c.P_max_kN)}</td>
                              <td className="col-num">{fmtNum(c.My_max_kNm)}</td>
                              <td className="col-num">{fmtNum(c.Mz_max_kNm)}</td>
                              <td className="col-num">{fmtNum(c.T_max_kNm)}</td>
                            </tr>
                          ))}
                        </tbody>
                      </table>
                    </div>
                  </div>
                ) : null}

                <div className="report-section">
                  <h3 className="report-h">Assumptions &amp; limits</h3>
                  <ul className="report-assumptions">
                    {result.assumptions.map((a, i) => (
                      <li key={i}>{a}</li>
                    ))}
                  </ul>
                  {result.totals.max_bearing_on_column_footing_kPa != null ? (
                    <p className="small-muted">
                      Rough column-only bearing estimate: ~
                      {fmtNum(result.totals.max_bearing_on_column_footing_kPa, 1)} kPa vs specified
                      SBC (if given). Not a footing design.
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
          (c) 2026 Balmores Lab - Developed by{" "}
          <a
            href="/about"
            rel="author"
            style={{ color: "inherit", textDecoration: "underline", textUnderlineOffset: 3 }}
          >
            Louie Doniego Balmores
          </a>
        </span>
        <span className="footer-sep" aria-hidden />
        <span className="small-muted">Integrated PyNite open-source FEM - verify with your code</span>
        <span className="footer-sep" aria-hidden />
        <a href="/about" style={{ color: "inherit" }}>
          About
        </a>
        <span className="footer-sep" aria-hidden />
        <a href="/cv" style={{ color: "inherit" }}>
          CV
        </a>
        <span className="footer-sep" aria-hidden />
        <a href="/research" style={{ color: "inherit" }}>
          Research
        </a>
        <span className="footer-sep" aria-hidden />
        <span className="footer-ver">v{APP_VERSION}</span>
      </footer>
    </div>
  );
}

