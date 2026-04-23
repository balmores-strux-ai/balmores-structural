"use client";

export default function BarMiniChart({
  title,
  items,
}: {
  title: string;
  items: { label: string; value: number; unit?: string }[];
}) {
  const nums = items.map((x) => {
    const v = typeof x.value === "number" ? x.value : Number(x.value);
    return Number.isFinite(v) ? v : 0;
  });
  const max = Math.max(...nums, 1e-9);
  return (
    <div className="chart-card">
      <div className="chart-title">{title}</div>
      <div className="chart-rows">
        {items.map((item, i) => {
          const val = nums[i] ?? 0;
          return (
            <div key={item.label}>
              <div className="chart-row-head small-muted">
                <span>{item.label}</span>
                <span>
                  {val.toFixed(1)} {item.unit || ""}
                </span>
              </div>
              <div className="chart-bar-track">
                <div
                  className="chart-bar-fill"
                  style={{ width: `${(val / max) * 100}%` }}
                />
              </div>
            </div>
          );
        })}
      </div>
    </div>
  );
}
