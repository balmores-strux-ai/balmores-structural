"use client";

export default function BarMiniChart({
  title,
  items,
}: {
  title: string;
  items: { label: string; value: number; unit?: string }[];
}) {
  const max = Math.max(...items.map((x) => x.value), 1);
  return (
    <div className="chart-card">
      <div style={{ fontWeight: 600, marginBottom: 12, fontSize: 13 }}>
        {title}
      </div>
      <div style={{ display: "grid", gap: 12 }}>
        {items.map((item) => (
          <div key={item.label}>
            <div
              className="small-muted"
              style={{ display: "flex", justifyContent: "space-between", marginBottom: 4 }}
            >
              <span>{item.label}</span>
              <span>
                {item.value.toFixed(1)} {item.unit || ""}
              </span>
            </div>
            <div
              style={{
                height: 8,
                background: "rgba(255,255,255,.06)",
                borderRadius: 999,
                overflow: "hidden",
              }}
            >
              <div
                style={{
                  width: `${(item.value / max) * 100}%`,
                  height: "100%",
                  background: "linear-gradient(90deg, #3b82f6, #06b6d4)",
                  borderRadius: 999,
                }}
              />
            </div>
          </div>
        ))}
      </div>
    </div>
  );
}
