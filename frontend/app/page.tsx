import type { CSSProperties } from "react";

const shell: CSSProperties = {
  position: "fixed",
  inset: 0,
  background: "#0b0d12",
  color: "#e6e8ee",
  display: "grid",
  gridTemplateRows: "1fr auto",
  textAlign: "center",
  fontFamily: "ui-sans-serif, system-ui, -apple-system, Segoe UI, Roboto, sans-serif",
  zIndex: 9999,
};

const stage: CSSProperties = {
  display: "grid",
  placeItems: "center",
  padding: 32,
};

const title: CSSProperties = {
  fontSize: "clamp(28px, 6vw, 64px)",
  letterSpacing: "0.04em",
  fontWeight: 700,
  margin: 0,
  lineHeight: 1.15,
};

const foot: CSSProperties = {
  padding: "16px 24px",
  borderTop: "1px solid rgba(255,255,255,0.08)",
  fontSize: 13,
  letterSpacing: "0.02em",
  opacity: 0.78,
};

export default function Page() {
  return (
    <div style={shell}>
      <main style={stage}>
        <h1 style={title}>Project Underway by Louie Balmores</h1>
      </main>
      <footer style={foot}>© 2026 Balmoreslab Developed by Louie Balmores</footer>
    </div>
  );
}
