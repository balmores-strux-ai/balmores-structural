import type { CSSProperties } from "react";

const overlay: CSSProperties = {
  position: "fixed",
  inset: 0,
  background: "#0b0d12",
  color: "#e6e8ee",
  display: "grid",
  placeItems: "center",
  textAlign: "center",
  fontFamily: "ui-sans-serif, system-ui, -apple-system, Segoe UI, Roboto, sans-serif",
  padding: 32,
  zIndex: 9999,
};

const title: CSSProperties = {
  fontSize: "clamp(28px, 6vw, 64px)",
  letterSpacing: "0.18em",
  fontWeight: 700,
  marginBottom: 16,
};

const sub: CSSProperties = {
  fontSize: "clamp(14px, 2vw, 18px)",
  opacity: 0.75,
};

export default function Page() {
  return (
    <main style={overlay}>
      <div>
        <div style={title}>UNDER CONSTRUCTION</div>
        <div style={sub}>Developing by Louie Balmores</div>
      </div>
    </main>
  );
}
