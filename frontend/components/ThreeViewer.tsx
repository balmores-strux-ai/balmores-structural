"use client";

import { Canvas, useThree } from "@react-three/fiber";
import { Line, OrbitControls } from "@react-three/drei";
import { useEffect, useMemo } from "react";
import * as THREE from "three";

type NodeT = { id: string; x: number; y: number; z: number };
type MemberT = { id: string; start: string; end: string; kind: string };

const COLORS = {
  beam: "#67e8f9",
  column: "#93c5fd",
  brace: "#fbbf24",
  cantilever: "#f87171",
} as const;

function hslForIntensity(t: number, hueCold = 210, hueHot = 12): string {
  const x = Math.max(0, Math.min(1, t));
  const h = hueCold + x * (hueHot - hueCold);
  return `hsl(${h}, 82%, 58%)`;
}

function MemberLines({
  nodes,
  members,
  beamMomentById,
  columnAxialById,
}: {
  nodes: NodeT[];
  members: MemberT[];
  beamMomentById: Record<string, number>;
  columnAxialById: Record<string, number>;
}) {
  const nodeMap = useMemo(() => {
    const map = new Map<string, [number, number, number]>();
    for (const n of nodes) map.set(n.id, [n.x, n.z, n.y]);
    return map;
  }, [nodes]);

  const beamVals = useMemo(() => Object.values(beamMomentById).filter((v) => Number.isFinite(v)), [beamMomentById]);
  const colVals = useMemo(() => Object.values(columnAxialById).filter((v) => Number.isFinite(v)), [columnAxialById]);
  const bMin = beamVals.length ? Math.min(...beamVals) : 0;
  const bMax = beamVals.length ? Math.max(...beamVals) : 1;
  const cMin = colVals.length ? Math.min(...colVals) : 0;
  const cMax = colVals.length ? Math.max(...colVals) : 1;

  return (
    <>
      {members.map((m) => {
        const a = nodeMap.get(m.start);
        const b = nodeMap.get(m.end);
        if (!a || !b) return null;

        let color = (COLORS as Record<string, string>)[m.kind] ?? COLORS.beam;
        if (m.kind === "beam" && beamMomentById[m.id] !== undefined) {
          const v = beamMomentById[m.id];
          const t = bMax > bMin ? (v - bMin) / (bMax - bMin) : 0.5;
          color = hslForIntensity(t, 200, 8);
        } else if (m.kind === "column" && columnAxialById[m.id] !== undefined) {
          const v = columnAxialById[m.id];
          const t = cMax > cMin ? (v - cMin) / (cMax - cMin) : 0.5;
          color = hslForIntensity(t, 230, 28);
        }

        return <Line key={m.id} points={[a, b]} color={color} lineWidth={2} />;
      })}
    </>
  );
}

/** Frame lines have no volume; fit camera from node coordinates (same X,Z,Y mapping as members). */
function FitCamera({ nodes }: { nodes: NodeT[] }) {
  const { camera } = useThree();
  useEffect(() => {
    if (!nodes.length) return;
    let minX = Infinity,
      minY = Infinity,
      minZ = Infinity,
      maxX = -Infinity,
      maxY = -Infinity,
      maxZ = -Infinity;
    for (const n of nodes) {
      const x = n.x;
      const y = n.z;
      const z = n.y;
      minX = Math.min(minX, x);
      maxX = Math.max(maxX, x);
      minY = Math.min(minY, y);
      maxY = Math.max(maxY, y);
      minZ = Math.min(minZ, z);
      maxZ = Math.max(maxZ, z);
    }
    const cx = (minX + maxX) / 2;
    const cy = (minY + maxY) / 2;
    const cz = (minZ + maxZ) / 2;
    const size = Math.max(maxX - minX, maxY - minY, maxZ - minZ, 1);
    const dist = size * 1.85;
    const pos = new THREE.Vector3(cx + dist * 0.75, cy + dist * 0.55, cz + dist * 0.75);
    camera.position.copy(pos);
    camera.lookAt(cx, cy, cz);
    camera.updateProjectionMatrix();
  }, [nodes, camera]);
  return null;
}

export default function ThreeViewer({
  geometry,
  beamMomentById,
  columnAxialById,
}: {
  geometry: { nodes: NodeT[]; members: MemberT[]; meta?: Record<string, unknown> } | null;
  /** |M| envelope (kN·m) per beam member id — drives hue on beams. */
  beamMomentById?: Record<string, number>;
  /** |P| envelope (kN) per column member id — drives hue on columns. */
  columnAxialById?: Record<string, number>;
}) {
  const bMap = beamMomentById ?? {};
  const cMap = columnAxialById ?? {};

  return (
    <Canvas camera={{ position: [24, 20, 24], fov: 50 }}>
      <color attach="background" args={["#080a0f"]} />
      <fog attach="fog" args={["#080a0f", 28, 130]} />
      <ambientLight intensity={0.55} />
      <hemisphereLight args={["#c7e8ff", "#0a0c10", 0.35]} />
      <directionalLight position={[12, 28, 14]} intensity={1.15} color="#e8f4ff" />
      <directionalLight position={[-18, 8, -10]} intensity={0.35} color="#818cf8" />
      <gridHelper args={[80, 40, "#1e3a4f", "#0c1219"]} />
      {geometry && geometry.nodes.length > 0 ? (
        <>
          <FitCamera nodes={geometry.nodes} />
          <MemberLines
            nodes={geometry.nodes}
            members={geometry.members}
            beamMomentById={bMap}
            columnAxialById={cMap}
          />
        </>
      ) : null}
      <OrbitControls makeDefault enableDamping dampingFactor={0.08} />
    </Canvas>
  );
}
