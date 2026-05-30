import { ImageResponse } from "next/og";
import { SANDRA_AGCAOILI } from "@/lib/research-team";

export const runtime = "edge";
export const alt = `${SANDRA_AGCAOILI.name} - AI Researcher | Balmores Lab`;
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
            "radial-gradient(1000px 500px at 100% 0%, rgba(14,165,233,0.35), transparent 60%), radial-gradient(1000px 500px at 0% 100%, rgba(99,102,241,0.25), transparent 60%), #080a0f",
          color: "#e6edf3",
          fontFamily: "system-ui, sans-serif",
        }}
      >
        <span style={{ fontSize: 22, letterSpacing: "0.28em", color: "#cbd5e1", fontWeight: 700 }}>
          BALMORES - LAB / PROFILE
        </span>
        <div>
          <div style={{ fontSize: 18, letterSpacing: "0.22em", textTransform: "uppercase", color: "#7dd3fc", marginBottom: 20 }}>
            AI Researcher · Research Partner
          </div>
          <div style={{ fontSize: 88, fontWeight: 700, color: "#f8fafc", lineHeight: 1.02, letterSpacing: "-0.02em" }}>
            Sandra Agcaoili
          </div>
          <div style={{ fontSize: 30, color: "#a5b4fc", marginTop: 16 }}>
            PhD Artificial Intelligence · UP Diliman
          </div>
          <div style={{ fontSize: 24, color: "#94a3b8", marginTop: 16 }}>
            Based in Singapore · AAP Member · Balmores Lab
          </div>
        </div>
        <span style={{ fontSize: 22, color: "#64748b" }}>balmoreslab.com/about/sandra-agcaoili</span>
      </div>
    ),
    { ...size },
  );
}
