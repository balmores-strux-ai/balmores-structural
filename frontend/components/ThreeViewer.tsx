"use client";

import { Canvas } from "@react-three/fiber";
import { OrbitControls, Line } from "@react-three/drei";
import { useMemo } from "react";

type NodeT = { id: string; x: number; y: number; z: number };
type MemberT = { id: string; start: string; end: string; kind: string };

const COLORS = {
  beam: "#67e8f9",
  column: "#93c5fd",
  brace: "#fbbf24",
  cantilever: "#f87171",
} as const;

function MemberLines({
  nodes,
  members,
}: {
  nodes: NodeT[];
  members: MemberT[];
}) {
  const nodeMap = useMemo(() => {
    const map = new Map<string, [number, number, number]>();
    for (const n of nodes) map.set(n.id, [n.x, n.z, n.y]);
    return map;
  }, [nodes]);

  return (
    <>
      {members.map((m) => {
        const a = nodeMap.get(m.start);
        const b = nodeMap.get(m.end);
        if (!a || !b) return null;
        const color = (COLORS as Record<string, string>)[m.kind] ?? COLORS.beam;
        return (
          <Line
            key={m.id}
            points={[a, b]}
            color={color}
            lineWidth={1.8}
          />
        );
      })}
    </>
  );
}

export default function ThreeViewer({
  geometry,
}: {
  geometry: { nodes: NodeT[]; members: MemberT[]; meta?: Record<string, unknown> } | null;
}) {
  return (
    <Canvas camera={{ position: [24, 20, 24], fov: 50 }}>
      <color attach="background" args={["#0c0f14"]} />
      <ambientLight intensity={0.7} />
      <directionalLight position={[10, 20, 10]} intensity={1.3} />
      <gridHelper args={[80, 40, "#1e293b", "#0f172a"]} />
      {geometry && <MemberLines nodes={geometry.nodes} members={geometry.members} />}
      <OrbitControls enableDamping dampingFactor={0.08} />
    </Canvas>
  );
}
