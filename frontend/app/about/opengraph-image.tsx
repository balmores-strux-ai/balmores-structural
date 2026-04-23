import { ImageResponse } from "next/og";

export const runtime = "edge";
export const alt = "About Louie Doniego Balmores — Official Profile";
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
          padding: "72px 80px",
          background:
            "radial-gradient(1000px 500px at 100% 0%, rgba(99,102,241,0.35), transparent 60%), radial-gradient(1000px 500px at 0% 100%, rgba(236,72,153,0.22), transparent 60%), #080a0f",
          color: "#e6edf3",
          fontFamily: "system-ui, sans-serif",
        }}
      >
        <div
          style={{
            display: "flex",
            alignItems: "center",
            justifyContent: "space-between",
          }}
        >
          <span
            style={{
              fontSize: 22,
              letterSpacing: "0.28em",
              color: "#cbd5e1",
              fontWeight: 700,
            }}
          >
            BALMORES · LAB  /  ABOUT
          </span>
          <span style={{ color: "#94a3b8", fontSize: 20 }}>
            balmoreslab.com/about
          </span>
        </div>

        <div
          style={{
            display: "flex",
            flexDirection: "column",
            marginTop: 60,
            flex: 1,
            justifyContent: "center",
          }}
        >
          <div
            style={{
              fontSize: 18,
              letterSpacing: "0.22em",
              textTransform: "uppercase",
              color: "#a5b4fc",
              marginBottom: 22,
            }}
          >
            Official Profile
          </div>
          <div
            style={{
              fontSize: 92,
              fontWeight: 700,
              lineHeight: 1.02,
              color: "#f8fafc",
              letterSpacing: "-0.02em",
            }}
          >
            Louie Doniego Balmores
          </div>
          <div
            style={{
              fontSize: 34,
              color: "#a5b4fc",
              marginTop: 16,
              fontWeight: 500,
            }}
          >
            Structural Engineer & AI Researcher
          </div>
        </div>

        <div
          style={{
            display: "flex",
            flexWrap: "wrap",
            gap: 14,
            marginTop: 40,
          }}
        >
          {[
            "Registered Civil Engineer · PRC PH",
            "Nov 2013 · Seq. No. 350",
            "P.Eng Candidate (PEO, 2027)",
            "PE Candidate (USA, 2028)",
            "Founder, Balmores Laboratory",
          ].map((t) => (
            <span
              key={t}
              style={{
                padding: "10px 18px",
                fontSize: 20,
                border: "1px solid rgba(99,102,241,0.45)",
                background: "rgba(99,102,241,0.10)",
                color: "#c7d2fe",
                borderRadius: 999,
              }}
            >
              {t}
            </span>
          ))}
        </div>
      </div>
    ),
    { ...size },
  );
}
