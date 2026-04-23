"use client";

import type { FeaParsedModel } from "@/lib/api";

type DriftRow = {
  storey_index: number;
  z_top_m: number;
  height_m: number;
  max_drift_mm: number;
  drift_ratio_h: number;
};

type Totals = Record<string, number | null | undefined>;

function cumsum(xs: number[]): number[] {
  const o: number[] = [];
  let s = 0;
  for (const x of xs) {
    s += x;
    o.push(s);
  }
  return o;
}

/** Schematic plan: bay grid from parsed spans (metres → SVG). */
function PlanSchematic({ sx, sy }: { sx: number[]; sy: number[] }) {
  const W = sx.reduce((a, b) => a + b, 0) || 1;
  const H = sy.reduce((a, b) => a + b, 0) || 1;
  const pad = 8;
  const vw = 260;
  const vh = 180;
  const scale = Math.min((vw - pad * 2) / W, (vh - pad * 2) / H);
  const ox = pad + (vw - pad * 2 - W * scale) / 2;
  const oy = pad + (vh - pad * 2 - H * scale) / 2;

  const xLines = [0, ...cumsum(sx)].map((x) => ox + x * scale);
  const yLines = [0, ...cumsum(sy)].map((y) => oy + y * scale);

  return (
    <svg
      className="fea-diag-svg"
      viewBox={`0 0 ${vw} ${vh}`}
      aria-label="Plan schematic: column grid from X and Y spans"
    >
      <rect x={ox} y={oy} width={W * scale} height={H * scale} fill="rgba(59,130,246,0.06)" stroke="rgba(59,130,246,0.35)" strokeWidth={1} rx={4} />
      {xLines.map((x, i) => (
        <line key={`x${i}`} x1={x} y1={oy} x2={x} y2={oy + H * scale} stroke="rgba(232,236,242,0.35)" strokeWidth={i === 0 || i === xLines.length - 1 ? 1.2 : 0.7} />
      ))}
      {yLines.map((y, i) => (
        <line key={`y${i}`} x1={ox} y1={y} x2={ox + W * scale} y2={y} stroke="rgba(232,236,242,0.35)" strokeWidth={i === 0 || i === yLines.length - 1 ? 1.2 : 0.7} />
      ))}
      <text x={ox + (W * scale) / 2} y={oy - 2} textAnchor="middle" fill="var(--muted)" fontSize={9} fontFamily="var(--font-mono)">
        Plan {W.toFixed(1)} × {H.toFixed(1)} m
      </text>
    </svg>
  );
}

/** Elevation: stacked storeys along Z (schematic). */
function ElevationSchematic({ heights }: { heights: number[] }) {
  const total = heights.reduce((a, b) => a + b, 0) || 1;
  const vw = 260;
  const vh = 160;
  const pad = 14;
  const barW = 48;
  const x0 = vw / 2 - barW / 2;
  let z = vh - pad;
  const segs: { y: number; h: number; lab: string }[] = [];
  heights.forEach((h, i) => {
    const drawH = (h / total) * (vh - pad * 2);
    segs.push({ y: z - drawH, h: drawH, lab: `${h}m` });
    z -= drawH;
  });

  return (
    <svg className="fea-diag-svg" viewBox={`0 0 ${vw} ${vh}`} aria-label="Elevation: storey heights">
      <text x={vw / 2} y={12} textAnchor="middle" fill="var(--muted)" fontSize={9} fontFamily="var(--font-mono)">
        Elevation · {heights.length} storeys · Σh {total.toFixed(1)} m
      </text>
      {segs.map((s, i) => (
        <g key={i}>
          <rect x={x0} y={s.y} width={barW} height={s.h} fill={i % 2 === 0 ? "rgba(147,197,253,0.25)" : "rgba(103,232,249,0.2)"} stroke="rgba(232,236,242,0.25)" strokeWidth={1} rx={2} />
          <text x={x0 + barW + 6} y={s.y + s.h / 2 + 3} fill="var(--muted)" fontSize={8} fontFamily="var(--font-mono)">
            L{i + 1} {s.lab}
          </text>
        </g>
      ))}
    </svg>
  );
}

function DriftBars({ drifts }: { drifts: DriftRow[] }) {
  if (!drifts.length) return <p className="small-muted fea-diag-empty">No drift rows.</p>;
  const maxMm = Math.max(...drifts.map((d) => d.max_drift_mm), 1e-6);
  return (
    <div className="fea-drift-bars" role="img" aria-label="Storey drift diagram">
      {drifts.map((d) => (
        <div key={d.storey_index} className="fea-drift-row">
          <span className="fea-drift-label">S{d.storey_index}</span>
          <div className="fea-drift-track">
            <div
              className="fea-drift-fill"
              style={{ width: `${Math.min(100, (d.max_drift_mm / maxMm) * 100)}%` }}
              title={`${d.max_drift_mm} mm`}
            />
          </div>
          <span className="fea-drift-val">{d.max_drift_mm.toFixed(2)} mm</span>
          <span className="fea-drift-ratio">{(d.drift_ratio_h * 1000).toFixed(2)}‰ h</span>
        </div>
      ))}
    </div>
  );
}

function EquilibriumBars({ totals }: { totals: Totals }) {
  const rx = Number(totals.sum_base_Rx_kN ?? 0);
  const ry = Number(totals.sum_base_Ry_kN ?? 0);
  const rz = Number(totals.sum_base_Rz_kN ?? 0);
  const m = Math.max(Math.abs(rx), Math.abs(ry), Math.abs(rz), 1e-6);
  const rows = [
    { k: "ΣRx", v: rx, c: "rgba(248, 113, 113, 0.65)" },
    { k: "ΣRy", v: ry, c: "rgba(52, 211, 153, 0.65)" },
    { k: "ΣRz", v: rz, c: "rgba(96, 165, 250, 0.75)" },
  ];
  return (
    <div className="fea-eq-bars" aria-label="Global vertical and horizontal equilibrium at supports">
      {rows.map((r) => (
        <div key={r.k} className="fea-eq-row">
          <span className="fea-eq-label">{r.k}</span>
          <div className="fea-eq-track">
            <div
              className="fea-eq-fill"
              style={{
                width: `${(Math.abs(r.v) / m) * 100}%`,
                background: r.c,
                marginLeft: r.v < 0 ? "auto" : 0,
                marginRight: r.v >= 0 ? "auto" : 0,
              }}
            />
          </div>
          <span className="fea-eq-val">{r.v.toFixed(1)} kN</span>
        </div>
      ))}
      <p className="small-muted fea-eq-note">Base reactions from PyNite (ULS); check statics in model.</p>
    </div>
  );
}

export default function FeaDiagrams({
  parsed,
  storeyDrifts,
  totals,
  loadCombination,
}: {
  parsed: FeaParsedModel | null;
  storeyDrifts: DriftRow[];
  totals: Totals;
  loadCombination: string;
}) {
  if (!parsed?.spans_x_m?.length || !parsed?.spans_y_m?.length) {
    return (
      <div className="fea-diagrams">
        <p className="small-muted">Run an analysis to see plan/elevation schematics and drift diagrams.</p>
      </div>
    );
  }

  const sh = parsed.story_heights_m ?? [3.5];

  return (
    <div className="fea-diagrams">
      <div className="fea-diag-head">
        <strong>Diagrams</strong>
        <span className="small-muted">{loadCombination}</span>
      </div>
      <div className="fea-diag-grid">
        <div className="fea-diag-card">
          <div className="fea-diag-title">Plan grid</div>
          <PlanSchematic sx={parsed.spans_x_m} sy={parsed.spans_y_m} />
        </div>
        <div className="fea-diag-card">
          <div className="fea-diag-title">Storey stack</div>
          <ElevationSchematic heights={sh} />
        </div>
        <div className="fea-diag-card fea-diag-card-wide">
          <div className="fea-diag-title">Storey drift (horizontal, max per level)</div>
          <DriftBars drifts={storeyDrifts} />
        </div>
        <div className="fea-diag-card fea-diag-card-wide">
          <div className="fea-diag-title">Base resultant (global)</div>
          <EquilibriumBars totals={totals} />
        </div>
      </div>
      <dl className="fea-diag-params">
        <div>
          <dt>DL + slab SW</dt>
          <dd>
            {parsed.dl_kpa?.toFixed(2)} + {parsed.slab_sw_kpa?.toFixed(2)} kPa
          </dd>
        </div>
        <div>
          <dt>LL</dt>
          <dd>{parsed.ll_kpa?.toFixed(2)} kPa</dd>
        </div>
        <div>
          <dt>Wind</dt>
          <dd>{(parsed.wind_pressure_kpa ?? 0).toFixed(2)} kPa</dd>
        </div>
        <div>
          <dt>Lateral (roof / grav)</dt>
          <dd>{((parsed.lateral_roof_fraction_of_gravity ?? 0) * 100).toFixed(0)}%</dd>
        </div>
      </dl>
    </div>
  );
}
