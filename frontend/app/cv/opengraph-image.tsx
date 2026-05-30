import { ImageResponse } from "next/og";
import { PERSON_NAME, JOB_TITLE } from "@/lib/seo";

export const runtime = "edge";
export const alt = `CV - ${PERSON_NAME}`;
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
            "radial-gradient(1000px 500px at 0% 100%, rgba(14,165,233,0.30), transparent 60%), #080a0f",
          color: "#e6edf3",
          fontFamily: "system-ui, sans-serif",
        }}
      >
        <span style={{ fontSize: 22, letterSpacing: "0.28em", color: "#cbd5e1", fontWeight: 700 }}>
          BALMORES - LAB / CV
        </span>
        <div>
          <div style={{ fontSize: 72, fontWeight: 700, color: "#f8fafc", lineHeight: 1.05 }}>
            {PERSON_NAME}
          </div>
          <div style={{ fontSize: 32, color: "#a5b4fc", marginTop: 16 }}>{JOB_TITLE}</div>
          <div style={{ fontSize: 24, color: "#94a3b8", marginTop: 20, maxWidth: 900 }}>
            Registered Civil Engineer (PRC PH, 2013) · 10+ years structural design · AI research
          </div>
        </div>
        <span style={{ fontSize: 22, color: "#64748b" }}>balmoreslab.com/cv</span>
      </div>
    ),
    { ...size },
  );
}
