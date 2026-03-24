'use client';

import { Canvas } from '@react-three/fiber';
import { OrbitControls, Line } from '@react-three/drei';
import { ProjectState } from '@/types';
import { useMemo } from 'react';

function StructureModel({ project }: { project?: ProjectState }) {
  const data = useMemo(() => {
    if (!project) return { columns: [], beamsX: [], beamsY: [] as number[][][] };

    const xCoords = [0, ...project.grid_x_m];
    const yCoords = [0, ...project.grid_y_m];
    const zCoords = [0];
    for (const h of project.story_heights_m) zCoords.push(zCoords[zCoords.length - 1] + h);

    const columns: number[][][] = [];
    const beamsX: number[][][] = [];
    const beamsY: number[][][] = [];

    for (const x of xCoords) {
      for (const y of yCoords) {
        for (let i = 0; i < zCoords.length - 1; i++) {
          columns.push([[x, zCoords[i], y], [x, zCoords[i + 1], y]]);
        }
      }
    }

    for (let k = 1; k < zCoords.length; k++) {
      const z = zCoords[k];
      for (const y of yCoords) {
        for (let i = 0; i < xCoords.length - 1; i++) {
          beamsX.push([[xCoords[i], z, y], [xCoords[i + 1], z, y]]);
        }
      }
      for (const x of xCoords) {
        for (let j = 0; j < yCoords.length - 1; j++) {
          beamsY.push([[x, z, yCoords[j]], [x, z, yCoords[j + 1]]]);
        }
      }
    }

    return { columns, beamsX, beamsY };
  }, [project]);

  return (
    <>
      <ambientLight intensity={1.2} />
      <directionalLight position={[15, 20, 10]} intensity={1.5} />
      <gridHelper args={[40, 20, '#2a3750', '#182030']} position={[10, 0, 10]} />
      {data.columns.map((seg, idx) => (
        <Line key={`c-${idx}`} points={seg as any} color="#87b5ff" lineWidth={1.4} />
      ))}
      {data.beamsX.map((seg, idx) => (
        <Line key={`bx-${idx}`} points={seg as any} color="#7af0c8" lineWidth={1.4} />
      ))}
      {data.beamsY.map((seg, idx) => (
        <Line key={`by-${idx}`} points={seg as any} color="#7af0c8" lineWidth={1.4} />
      ))}
      <OrbitControls />
    </>
  );
}

export default function ViewerPanel({ project }: { project?: ProjectState }) {
  return (
    <section className="panel viewerPanel">
      <div className="panelTitle">3D model preview</div>
      <div className="viewerFrame">
        <Canvas camera={{ position: [22, 20, 22], fov: 45 }}>
          <StructureModel project={project} />
        </Canvas>
      </div>
    </section>
  );
}
