import { ImageResponse } from "next/og";

export const runtime = "edge";
export const alt = "Louie Doniego Balmores — Structural Engineer & AI Researcher";
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
            "radial-gradient(1000px 500px at 0% 0%, rgba(99,102,241,0.35), transparent 60%), radial-gradient(1000px 500px at 100% 100%, rgba(14,165,233,0.30), transparent 60%), #080a0f",
          color: "#e6edf3",
          fontFamily: "system-ui, sans-serif",
        }}
      >
        <div style={{ display: "flex", alignItems: "center", gap: 16 }}>
          <div
            style={{
              width: 44,
              height: 44,
              borderRadius: 10,
              background:
                "linear-gradient(135deg, #6366f1 0%, #0ea5e9 100%)",
            }}
          />
          <span
            style={{
              fontSize: 22,
              letterSpacing: "0.28em",
              color: "#cbd5e1",
              fontWeight: 700,
            }}
          >
            BALMORES · LAB
          </span>
        </div>

        <div style={{ display: "flex", flexDirection: "column" }}>
          <div
            style={{
              fontSize: 20,
              letterSpacing: "0.22em",
              textTransform: "uppercase",
              color: "#60a5fa",
              marginBottom: 24,
              fontWeight: 600,
            }}
          >
            Structural Engineer · AI Researcher
          </div>
          <div
            style={{
              fontSize: 86,
              fontWeight: 700,
              lineHeight: 1.04,
              letterSpacing: "-0.02em",
              color: "#f8fafc",
            }}
          >
            Louie Doniego
          </div>
          <div
            style={{
              fontSize: 86,
              fontWeight: 700,
              lineHeight: 1.04,
              letterSpacing: "-0.02em",
              color: "#f8fafc",
              marginBottom: 28,
            }}
          >
            Balmores
          </div>
          <div
            style={{
              fontSize: 28,
              color: "#94a3b8",
              maxWidth: 1040,
              lineHeight: 1.35,
            }}
          >
            Licensed Civil Engineer · PRC Philippines (2013) · 10+ years in
            high-performance structural design · AI-driven structural
            optimization research at Balmores Laboratory.
          </div>
        </div>

        <div
          style={{
            display: "flex",
            justifyContent: "space-between",
            alignItems: "flex-end",
            color: "#cbd5e1",
            fontSize: 22,
          }}
        >
          <span style={{ fontWeight: 600 }}>balmoreslab.com</span>
          <span style={{ color: "#64748b" }}>PyNite · PyTorch · Next.js</span>
        </div>
      </div>
    ),
    { ...size },
  );
}
