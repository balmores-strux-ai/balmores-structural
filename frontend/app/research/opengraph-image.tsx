import { ImageResponse } from "next/og";

export const runtime = "edge";
export const alt = "Research - Balmores Lab - AI-Driven Structural Engineering";
export const size = { width: 1200, height: 630 };
export const contentType = "image/png";

export default function OpengraphImage() {
  return new ImageResponse(
    (
      <div
        style={{
          width: "100%",
          height: "100%",
          display: "flex",
          flexDirection: "column",
          justifyContent: "space-between",
          padding: "72px 80px",
          background:
            "radial-gradient(1000px 500px at 100% 0%, rgba(99,102,241,0.35), transparent 60%), #080a0f",
          color: "#e6edf3",
          fontFamily: "system-ui, sans-serif",
        }}
      >
        <span style={{ fontSize: 22, letterSpacing: "0.28em", color: "#cbd5e1", fontWeight: 700 }}>
          BALMORES - LAB / RESEARCH
        </span>
        <div>
          <div style={{ fontSize: 56, fontWeight: 700, color: "#f8fafc", lineHeight: 1.1 }}>
            AI-Driven Structural Engineering
          </div>
          <div style={{ fontSize: 28, color: "#a5b4fc", marginTop: 20 }}>
            Neural surrogates · NL-to-FEM · Physics-informed ML
          </div>
          <div style={{ fontSize: 22, color: "#94a3b8", marginTop: 16 }}>
            Louie Doniego Balmores — Structural Engineer & AI Researcher
          </div>
        </div>
        <span style={{ fontSize: 22, color: "#64748b" }}>balmoreslab.com/research</span>
      </div>
    ),
    { ...size },
  );
}
