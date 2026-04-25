"use client";

import type { FeaDiagrams } from "@/lib/api";

type Pair = [number[], number[]];

function Chart({
  title,
  unit,
  data,
  fill,
  stroke,
}: {
  title: string;
  unit: string;
  data?: Pair;
  fill: string;
  stroke: string;
}) {
  if (!data || data[0].length < 2) {
    return (
      <div className="beam-diag-card">
        <div className="beam-diag-title">{title}</div>
        <p className="small-muted" style={{ padding: "8px 10px" }}>
          No data.
        </p>
      </div>
    );
  }
  const [xs, ys] = data;
  const minX = Math.min(...xs);
  const maxX = Math.max(...xs);
  const absMax = Math.max(1e-6, ...ys.map((v) => Math.abs(v)));
  const vw = 320;
  const vh = 120;
  const padL = 34;
  const padR = 10;
  const padT = 10;
  const padB = 22;
  const W = vw - padL - padR;
  const H = vh - padT - padB;

  const sx = (x: number) => padL + ((x - minX) / (maxX - minX || 1)) * W;
  const sy = (v: number) => padT + H / 2 - (v / absMax) * (H / 2 - 4);

  const areaPoints: string[] = [];
  for (let i = 0; i < xs.length; i++) areaPoints.push(`${sx(xs[i])},${sy(ys[i])}`);
  const areaPath = `M${sx(minX)},${sy(0)} L${areaPoints.join(" L")} L${sx(maxX)},${sy(0)} Z`;
  const linePath = `M${areaPoints.join(" L")}`;

  const yMax = absMax;
  const yMin = -absMax;
  const maxIdx = ys.reduce((best, v, i) => (Math.abs(v) > Math.abs(ys[best]) ? i : best), 0);

  return (
    <div className="beam-diag-card">
      <div className="beam-diag-title">
        {title}
        <span className="beam-diag-peak">
          peak {ys[maxIdx].toFixed(2)} {unit} @ x = {xs[maxIdx].toFixed(2)} m
        </span>
      </div>
      <svg viewBox={`0 0 ${vw} ${vh}`} className="beam-diag-svg">
        <rect x={padL} y={padT} width={W} height={H} fill="rgba(13,18,28,0.5)" stroke="rgba(148,163,184,0.2)" rx={4} />
        <line
          x1={padL}
          x2={padL + W}
          y1={sy(0)}
          y2={sy(0)}
          stroke="rgba(148,163,184,0.45)"
          strokeDasharray="4 3"
        />
        <path d={areaPath} fill={fill} opacity={0.35} />
        <path d={linePath} fill="none" stroke={stroke} strokeWidth={1.6} />
        <text x={4} y={padT + 10} fill="var(--muted)" fontSize={8} fontFamily="var(--font-mono)">
          {yMax.toFixed(1)}
        </text>
        <text x={4} y={padT + H - 2} fill="var(--muted)" fontSize={8} fontFamily="var(--font-mono)">
          {yMin.toFixed(1)}
        </text>
        <text x={padL} y={vh - 6} fill="var(--muted)" fontSize={8} fontFamily="var(--font-mono)">
          {minX.toFixed(2)} m
        </text>
        <text x={padL + W - 24} y={vh - 6} fill="var(--muted)" fontSize={8} fontFamily="var(--font-mono)">
          {maxX.toFixed(2)} m
        </text>
      </svg>
    </div>
  );
}

export type DiagramVisibility = {
  beamShear: boolean;
  beamMoment: boolean;
  beamDeflection: boolean;
  frameMoment: boolean;
  frameShear: boolean;
};

const DEFAULT_VIS: DiagramVisibility = {
  beamShear: true,
  beamMoment: true,
  beamDeflection: true,
  frameMoment: true,
  frameShear: true,
};

export default function BeamDiagrams({
  diagrams,
  visibility = DEFAULT_VIS,
}: {
  diagrams?: FeaDiagrams | null;
  visibility?: Partial<DiagramVisibility>;
}) {
  if (!diagrams) return null;
  const v = { ...DEFAULT_VIS, ...visibility };
  const hasBeam =
    diagrams.shear_kN && diagrams.moment_kNm && diagrams.deflection_mm;
  const hasFrame =
    diagrams.moment_per_level_kNm && Object.keys(diagrams.moment_per_level_kNm).length > 0;
  if (!hasBeam && !hasFrame) return null;

  return (
    <div className="beam-diagrams">
      <div className="beam-diag-head">
        <strong>Shear / moment / deflection</strong>
        <span className="small-muted">PyNite member arrays · ULS combo</span>
      </div>
      {hasBeam ? (
        <div className="beam-diag-grid">
          {v.beamShear ? (
            <Chart
              title="Shear V (kN)"
              unit="kN"
              data={diagrams.shear_kN}
              fill="rgba(52, 211, 153, 0.45)"
              stroke="rgba(52, 211, 153, 0.95)"
            />
          ) : null}
          {v.beamMoment ? (
            <Chart
              title="Moment M (kN·m)"
              unit="kN·m"
              data={diagrams.moment_kNm}
              fill="rgba(251, 191, 36, 0.35)"
              stroke="rgba(251, 191, 36, 0.95)"
            />
          ) : null}
          {v.beamDeflection ? (
            <Chart
              title="Deflection δ (mm)"
              unit="mm"
              data={diagrams.deflection_mm}
              fill="rgba(96, 165, 250, 0.35)"
              stroke="rgba(96, 165, 250, 0.95)"
            />
          ) : null}
        </div>
      ) : null}
      {hasFrame && diagrams.moment_per_level_kNm ? (
        <div className="beam-diag-grid">
          {v.frameMoment
            ? Object.entries(diagrams.moment_per_level_kNm).map(([k, pair]) => (
                <Chart
                  key={`M${k}`}
                  title={`Moment · storey ${k.replace("level_", "")}`}
                  unit="kN·m"
                  data={pair as Pair}
                  fill="rgba(251, 191, 36, 0.35)"
                  stroke="rgba(251, 191, 36, 0.95)"
                />
              ))
            : null}
          {v.frameShear && diagrams.shear_per_level_kN
            ? Object.entries(diagrams.shear_per_level_kN).map(([k, pair]) => (
                <Chart
                  key={`V${k}`}
                  title={`Shear · storey ${k.replace("level_", "")}`}
                  unit="kN"
                  data={pair as Pair}
                  fill="rgba(52, 211, 153, 0.45)"
                  stroke="rgba(52, 211, 153, 0.95)"
                />
              ))
            : null}
        </div>
      ) : null}
    </div>
  );
}
