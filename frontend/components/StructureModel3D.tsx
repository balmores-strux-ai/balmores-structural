"use client";

import type { FeaPromptResponse } from "@/lib/api";

type Node3 = { id: string; x: number; y: number; z: number };

function iso(n: Node3, ext: { minX: number; minY: number; maxX: number; maxY: number; maxZ: number }) {
  const W = 430;
  const H = 310;
  const sx = 5.4;
  const sy = 3.8;
  const sz = Math.min(2.3, 210 / Math.max(ext.maxZ || 1, 1));
  const cx = W / 2 - ((ext.maxX - ext.minX) - (ext.maxY - ext.minY)) * sx * 0.5;
  const by = H - 48;
  return {
    x: cx + (n.x - ext.minX) * sx - (n.y - ext.minY) * sy,
    y: by - n.z * sz + (n.x - ext.minX) * sx * 0.18 + (n.y - ext.minY) * sy * 0.2,
  };
}

export default function StructureModel3D({ result }: { result: FeaPromptResponse }) {
  if (result.analysis_type !== "building_3d") return null;
  const nodes = result.geometry.nodes;
  const members = result.geometry.members;
  if (!nodes.length) return null;

  const ext = nodes.reduce(
    (a, n) => ({
      minX: Math.min(a.minX, n.x),
      minY: Math.min(a.minY, n.y),
      maxX: Math.max(a.maxX, n.x),
      maxY: Math.max(a.maxY, n.y),
      maxZ: Math.max(a.maxZ, n.z),
    }),
    { minX: Infinity, minY: Infinity, maxX: -Infinity, maxY: -Infinity, maxZ: 0 },
  );
  const byId: Record<string, Node3> = {};
  for (const n of nodes) byId[n.id] = n;

  const levels = [...new Set(nodes.map((n) => n.z))].sort((a, b) => a - b);
  const roofZ = levels[levels.length - 1] ?? 0;
  const showLevels = new Set([0, roofZ, ...levels.filter((_, i) => i % Math.max(1, Math.floor(levels.length / 5)) === 0)]);
  const visibleMembers = members.filter((m) => {
    const a = byId[m.start];
    const b = byId[m.end];
    if (!a || !b) return false;
    return showLevels.has(a.z) || showLevels.has(b.z) || m.kind === "column";
  });
  const labelNodes = nodes.filter((n) => {
    const isCorner = (n.x === ext.minX || n.x === ext.maxX) && (n.y === ext.minY || n.y === ext.maxY);
    const isBaseEdge = n.z === 0 && (n.x === ext.minX || n.x === ext.maxX || n.y === ext.minY || n.y === ext.maxY);
    return isBaseEdge || (isCorner && n.z === roofZ);
  });
  const nodeNumber: Record<string, number> = {};
  nodes.forEach((n, i) => {
    nodeNumber[n.id] = i + 1;
  });

  return (
    <div className="structure-model-3d">
      <div className="structure-model-2d-head">
        <strong>3D wireframe preview</strong>
        <span className="small-muted">isometric - visible node numbers + support names</span>
      </div>
      <svg viewBox="0 0 430 310" className="structure-model-2d-svg" role="img" aria-label="3D building frame wireframe with support nodes">
        <rect x={0} y={0} width={430} height={310} rx={8} fill="rgba(8,12,20,0.65)" stroke="rgba(148,163,184,0.13)" />
        {visibleMembers.map((m) => {
          const a = byId[m.start];
          const b = byId[m.end];
          if (!a || !b) return null;
          const p0 = iso(a, ext);
          const p1 = iso(b, ext);
          const isCol = m.kind === "column";
          return (
            <line
              key={m.id}
              x1={p0.x}
              y1={p0.y}
              x2={p1.x}
              y2={p1.y}
              stroke={isCol ? "rgba(196,181,253,0.74)" : "rgba(125,211,252,0.58)"}
              strokeWidth={isCol ? 1.4 : 1}
            />
          );
        })}
        {nodes
          .filter((n) => n.z === 0 && (n.x === ext.minX || n.x === ext.maxX || n.y === ext.minY || n.y === ext.maxY))
          .map((n) => {
            const p = iso(n, ext);
            return (
              <g key={n.id}>
                <path d={`M${p.x} ${p.y + 2} l-5 9 h10 z`} fill="rgba(167,139,250,0.28)" stroke="#c4b5fd" strokeWidth={1} />
              </g>
            );
          })}
        {labelNodes.map((n) => {
          const p = iso(n, ext);
          return (
            <g key={n.id + "-label"}>
              <circle cx={p.x} cy={p.y} r={3.4} fill="#0f172a" stroke="#e0f2fe" strokeWidth={1} />
              <text x={p.x + 5} y={p.y - 4} fontSize={8} fill="#e0f2fe" fontFamily="var(--font-mono)">
                #{nodeNumber[n.id]} {n.id}
              </text>
            </g>
          );
        })}
        <text x={14} y={22} fill="#bae6fd" fontSize={10} fontFamily="var(--font-mono)">
          Fixed supports at all base nodes - perimeter node numbers shown for readability
        </text>
      </svg>
    </div>
  );
}
