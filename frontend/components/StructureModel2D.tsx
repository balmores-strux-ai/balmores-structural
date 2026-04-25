"use client";

import type React from "react";
import type { FeaPromptResponse } from "@/lib/api";

type Pt = { px: number; py: number };

function parseFrameId(id: string): { i: number; k: number } | null {
  const m = /^N_(\d+)_(\d+)$/.exec(id);
  if (!m) return null;
  return { i: +m[1], k: +m[2] };
}

function projectCabinet(
  x: number,
  z: number,
  xMin: number,
  xMax: number,
  zMin: number,
  zMax: number,
  w: number,
  h: number,
  pad: number,
): Pt {
  const depthSkew = 0.32;
  const xSpan = xMax - xMin || 1;
  const zSpan = zMax - zMin || 1;
  const kx = (w - 2 * pad) / (xSpan + zSpan * depthSkew);
  const kz = (h - 2 * pad) / zSpan;
  const px = pad + (x - xMin) * kx + z * depthSkew * kx;
  const py = pad + (zMax - z) * kz;
  return { px, py: py };
}

type SupportKind = "pin" | "roller" | "fixed" | "free" | string;

function SupportMark({
  x,
  y,
  kind,
  scale = 1,
}: {
  x: number;
  y: number;
  kind: SupportKind;
  scale?: number;
}) {
  const s = 8 * scale;
  const kl = (kind || "roller").toLowerCase();
  if (kl === "free")
    return <circle cx={x} cy={y} r={1.5} fill="var(--muted)" opacity={0.5} />;

  if (kl === "roller") {
    return (
      <g>
        <line x1={x - s * 0.4} y1={y} x2={x + s * 0.4} y2={y} stroke="var(--ink)" strokeWidth={1.2} />
        <circle cx={x} cy={y - 4} r={2.2} fill="none" stroke="var(--ink)" strokeWidth={1.2} />
        <line x1={x - s} y1={y + 2} x2={x + s} y2={y + 2} stroke="var(--ink)" strokeWidth={1} />
      </g>
    );
  }
  if (kl === "pin") {
    return (
      <g>
        <path d={`M${x} ${y - 2} L${x - s * 0.55} ${y + s * 0.5} L${x + s * 0.55} ${y + s * 0.5} Z`} fill="none" stroke="var(--ink)" strokeWidth={1.2} />
        <line x1={x - s} y1={y + s * 0.5} x2={x + s} y2={y + s * 0.5} stroke="var(--ink)" strokeWidth={1.2} />
      </g>
    );
  }
  /* fixed + default */
  return (
    <g>
      <path d={`M${x} ${y - 2} L${x - s * 0.5} ${y + s * 0.55} L${x + s * 0.5} ${y + s * 0.55} Z`} fill="rgba(148,163,184,0.3)" stroke="var(--ink)" strokeWidth={1.2} />
      <line x1={x - s} y1={y + s * 0.6} x2={x - s * 0.3} y2={y + s * 0.6} stroke="var(--ink)" strokeWidth={1.2} />
      <line x1={x - s * 0.1} y1={y + s * 0.6} x2={x + s * 0.1} y2={y + s * 0.6} stroke="var(--ink)" strokeWidth={1.2} />
      <line x1={x + s * 0.3} y1={y + s * 0.6} x2={x + s} y2={y + s * 0.6} stroke="var(--ink)" strokeWidth={1.2} />
    </g>
  );
}

function arrow(
  x0: number,
  y0: number,
  x1: number,
  y1: number,
  color: string,
  label: string,
) {
  const dx = x1 - x0;
  const dy = y1 - y0;
  const len = Math.hypot(dx, dy) || 1;
  const ux = dx / len;
  const uy = dy / len;
  const ah = 6;
  const bx = x1 - ux * ah;
  const by = y1 - uy * ah;
  const perp = 3;
  return (
    <g>
      <line x1={x0} y1={y0} x2={x1} y2={y1} stroke={color} strokeWidth={1.5} />
      <path
        d={`M${x1},${y1} L${bx - uy * perp},${by + ux * perp} L${bx + uy * perp},${by - ux * perp} Z`}
        fill={color}
      />
      <text x={x1 + 5} y={y1 - 2} fill={color} fontSize={8} fontFamily="var(--font-mono)">
        {label}
      </text>
    </g>
  );
}

export default function StructureModel2D({
  result,
  showReactions,
  showLoads,
}: {
  result: FeaPromptResponse;
  showReactions: boolean;
  showLoads: boolean;
}) {
  const { geometry, base_reactions, analysis_type } = result;
  const nodes = geometry.nodes;
  if (!nodes.length) return null;

  const isFrame = analysis_type === "frame_2d";
  const isBeam = analysis_type === "beam_2d";
  if (!isFrame && !isBeam) return null;

  const meta = geometry.meta as
    | {
        support_kinds?: string[];
        dl_kN_per_m?: number;
        ll_kN_per_m?: number;
        lateral_fx_per_floor_kN?: number;
        plane?: string;
      }
    | undefined;

  const W = 420;
  const H = 260;
  const pad = 28;

  const byId: Record<string, (typeof nodes)[0]> = {};
  for (const n of nodes) byId[n.id] = n;

  let xMin = Infinity,
    xMax = -Infinity,
    zMin = Infinity,
    zMax = -Infinity;
  for (const n of nodes) {
    const xw = n.x;
    const zw = n.z;
    xMin = Math.min(xMin, xw);
    xMax = Math.max(xMax, xw);
    zMin = Math.min(zMin, zw);
    zMax = Math.max(zMax, zw);
  }
  if (!Number.isFinite(xMin)) return null;
  if (xMax - xMin < 1e-6) {
    xMin -= 1;
    xMax += 1;
  }
  if (zMax - zMin < 1e-6) {
    zMin -= 0.1;
    zMax += 1;
  }

  const proj = (x: number, z: number) => projectCabinet(x, z, xMin, xMax, zMin, zMax, W, H, pad);

  const nodeList = [...nodes];
  if (isFrame) {
    nodeList.sort((a, b) => {
      const pa = parseFrameId(a.id);
      const pb = parseFrameId(b.id);
      if (pa && pb) return pa.k - pb.k || pa.i - pb.i;
      return a.id.localeCompare(b.id);
    });
  } else {
    nodeList.sort((a, b) => a.x - b.x);
  }

  const numById: Record<string, number> = {};
  nodeList.forEach((n, idx) => {
    numById[n.id] = idx + 1;
  });

  const reactMax = Math.max(
    1e-3,
    ...base_reactions.map((r) => Math.max(Math.abs(r.Rx_kN), Math.abs(r.Ry_kN ?? 0), Math.abs(r.Rz_kN ?? 0))),
  );
  const arrScale = 32 / reactMax;

  const kSupport = meta?.support_kinds || [];

  return (
    <div className="structure-model-2d">
      <div className="structure-model-2d-head">
        <strong>Elevation (cabinet / oblique view)</strong>
        <span className="small-muted">Global X across · Z up · {meta?.plane || "FEM view"}</span>
      </div>
      <svg
        className="structure-model-2d-svg"
        viewBox={`0 0 ${W} ${H}`}
        width="100%"
        height="auto"
        role="img"
        aria-label="2D structural model with node numbers"
      >
        <defs>
          <pattern id="udl" width={10} height={20} patternUnits="userSpaceOnUse">
            <line x1={5} y1={2} x2={5} y2={10} stroke="rgba(56, 189, 248, 0.8)" strokeWidth={1} markerEnd="url(#arrd)" />
          </pattern>
          <marker id="arrd" markerWidth={4} markerHeight={4} refX={2} refY={2} orient="auto">
            <path d="M0,0 L4,2 L0,4 z" fill="rgba(56, 189, 248, 0.9)" />
          </marker>
        </defs>
        <rect x={0} y={0} width={W} height={H} fill="rgba(11, 15, 24, 0.55)" stroke="rgba(148,163,184,0.12)" rx={6} />

        {geometry.members.map((m) => {
          const a = byId[m.start];
          const b = byId[m.end];
          if (!a || !b) return null;
          const p0 = proj(a.x, a.z);
          const p1 = proj(b.x, b.z);
          return (
            <line
              key={m.id}
              x1={p0.px}
              y1={p0.py}
              x2={p1.px}
              y2={p1.py}
              stroke="rgba(191, 219, 254, 0.85)"
              strokeWidth={m.kind === "column" ? 2.4 : 1.8}
              strokeLinecap="round"
            />
          );
        })}

        {isBeam &&
        showLoads &&
        (Number(meta?.dl_kN_per_m) > 0 || Number(meta?.ll_kN_per_m) > 0) ? (
          <g>
            {geometry.members
              .filter((m) => m.kind === "beam")
              .map((m) => {
                const a = byId[m.start];
                const b = byId[m.end];
                if (!a || !b) return null;
                const nSeg = 6;
                const el: React.ReactNode[] = [];
                for (let s = 0; s < nSeg; s++) {
                  const t1 = (s + 0.5) / nSeg;
                  const xb = a.x + t1 * (b.x - a.x);
                  const zb = a.z + t1 * (b.z - a.z);
                  const pB = proj(xb, zb);
                  el.push(
                    <line
                      key={`${m.id}-w${s}`}
                      x1={pB.px}
                      y1={pB.py - 4}
                      x2={pB.px}
                      y2={pB.py + 10}
                      stroke="rgba(56, 189, 248, 0.8)"
                      strokeWidth={1.1}
                    />,
                  );
                }
                return <g key={m.id}>{el}</g>;
              })}
            <text x={pad} y={16} fill="var(--accent)" fontSize={9} fontFamily="var(--font-mono)">
              w = {(Number(meta?.dl_kN_per_m) || 0) + (Number(meta?.ll_kN_per_m) || 0) > 0
                ? `1.2·DL+1.6·LL (ULS)  ·  wDL ${meta?.dl_kN_per_m ?? "—"}  wLL ${meta?.ll_kN_per_m ?? "—"} kN/m`
                : "Loads as interpreted"}
            </text>
          </g>
        ) : null}

        {isFrame && showLoads && (Number(meta?.lateral_fx_per_floor_kN) || 0) > 0
          ? nodeList.map((n) => {
                const p = parseFrameId(n.id);
                if (!p || p.i !== 0 || p.k < 1) return null;
                const pt = proj(n.x, n.z);
                const w = 22;
                return (
                  <g key={n.id + "-lat"}>
                    {arrow(pt.px - w, pt.py, pt.px, pt.py, "rgba(250, 204, 21, 0.95)", `Fx`)}
                    <text x={pt.px - w - 2} y={pt.py - 4} fontSize={7} fill="rgba(250, 204, 21, 0.95)">
                      {String(meta?.lateral_fx_per_floor_kN)} kN
                    </text>
                  </g>
                );
              })
          : null}

        {isFrame && showLoads && (Number(meta?.dl_kN_per_m) > 0 || Number(meta?.ll_kN_per_m) > 0) ? (
          <g>
            {geometry.members
              .filter((m) => m.kind === "beam")
              .filter((m) => m.id.includes("B_0_"))
              .map((m) => {
                const a = byId[m.start];
                const b = byId[m.end];
                if (!a || !b) return null;
                const midX = (a.x + b.x) / 2;
                const midZ = (a.z + b.z) / 2;
                const p0 = proj(midX, midZ);
                return (
                  <g key={m.id + "-g"}>
                    <line x1={p0.px - 20} y1={p0.py + 6} x2={p0.px + 20} y2={p0.py + 6} stroke="rgba(34, 197, 94, 0.4)" />
                    {[0, 0.3, 0.6, 0.9].map((t) => {
                      const tx = a.x + t * (b.x - a.x);
                      const tz = a.z + t * (b.z - a.z);
                      const p = proj(tx, tz);
                      return (
                        <line
                          key={t}
                          x1={p.px}
                          y1={p.py - 1}
                          x2={p.px}
                          y2={p.py - 8}
                          stroke="rgba(34, 197, 94, 0.85)"
                          strokeWidth={1}
                        />
                      );
                    })}
                    <text x={p0.px - 30} y={p0.py + 18} fontSize={8} fill="rgba(34, 197, 94, 0.95)">
                      w = DL+LL (kN/m)
                    </text>
                  </g>
                );
              })}
          </g>
        ) : null}

        {nodeList.map((n) => {
          const p = proj(n.x, n.z);
          return (
            <g key={n.id}>
              <circle
                cx={p.px}
                cy={p.py}
                r={5}
                fill="rgba(10, 15, 25, 0.95)"
                stroke="var(--ink)"
                strokeWidth={1.2}
              />
              <text
                x={p.px - 2}
                y={p.py + 2}
                fontSize={8}
                fill="var(--ink)"
                fontWeight={600}
                fontFamily="var(--font-mono)"
                style={{ userSelect: "none" }}
              >
                {numById[n.id]}
              </text>
            </g>
          );
        })}

        {isBeam
          ? nodeList.map((n, j) => {
              const p = proj(n.x, n.z);
              const k = (kSupport[j] as SupportKind) || "roller";
              return <SupportMark key={n.id + "-s"} x={p.px} y={p.py + 5} kind={k} scale={0.9} />;
            })
          : nodeList.map((n) => {
              const p = parseFrameId(n.id);
              if (!p) return null;
              if (p.k !== 0) return null;
              const pt = proj(n.x, n.z);
              return <SupportMark key={n.id + "-s"} x={pt.px} y={pt.py + 6} kind="fixed" />;
            })}

        {showReactions
          ? base_reactions.map((r) => {
              const n = byId[r.node];
              if (!n) return null;
              const p = proj(n.x, n.z);
              const rx = r.Rx_kN;
              const ry = r.Ry_kN ?? 0;
              const rz = r.Rz_kN ?? 0;
              const vert = isBeam ? ry : ry;
              const hor = isBeam ? rx : rx;
              const vMag = isBeam ? Math.abs(vert) : Math.max(Math.abs(vert), Math.abs(rz)) * 0.6;
              const hMag = Math.abs(hor);
              const g: React.ReactNode[] = [];
              if (hMag > 1e-4) {
                const len = hMag * arrScale;
                g.push(arrow(p.px, p.py, p.px - Math.sign(hor) * len, p.py, "rgba(244, 114, 182, 0.95)", `Rx`));
              }
              if (vMag > 1e-4) {
                const lenV = (vert > 0 ? -1 : 1) * vMag * arrScale;
                g.push(arrow(p.px, p.py, p.px, p.py - lenV, "rgba(167, 139, 250, 0.95)", isBeam ? "Ry" : "Ry"));
              }
              return <g key={r.node + "-r"}>{g}</g>;
            })
          : null}
      </svg>
      <p className="small-muted structure-model-2d-legend">
        <strong>Nodes</strong> 1…{nodeList.length} = sequence along members (axes in metres). &nbsp; Supports:{" "}
        <span className="leg-fixed">▲ fixed</span> ·<span className="leg-pin"> ∇ pin</span> ·
        <span className="leg-roll"> ⊙ roller</span>
        {showLoads ? " · Cyan/amber/green: schematic loads (not to scale in all cases)" : null}
        {showReactions ? " · Pink = horizontal R · Violet = vertical R" : null}
      </p>
    </div>
  );
}
