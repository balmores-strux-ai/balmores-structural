"use client";

import type { FeaPromptResponse } from "@/lib/api";

/**
 * 3D building: no single-plane wireframe in this build — a precise textual + pictogram
 * of supports and load path for engineers (matches PyNite model assumptions).
 */
export default function BuildingSupportLoadPanel({ result }: { result: FeaPromptResponse }) {
  if (result.analysis_type !== "building_3d") return null;
  const meta = result.geometry?.meta as
    | {
        grid?: { nx?: number; ny?: number; n_levels?: number; storeys?: number; bays_x?: number; bays_y?: number };
        supports_caption?: string;
        loads_caption?: string;
      }
    | undefined;
  if (!meta?.grid && !meta?.supports_caption) return null;

  const g = meta.grid;
  return (
    <div className="bldg-support-panel">
      <div className="bldg-support-panel-title">
        <strong>3D model — supports &amp; loads</strong>
        <span className="small-muted">Grid-level summary (X / Y / Z in metres)</span>
      </div>
      {g ? (
        <ul className="bldg-support-grid-dl">
          <li>
            <span>Column grid</span>{" "}
            <strong>
              {g.nx} × {g.ny}
            </strong>{" "}
            plan nodes, <strong>{g.n_levels ?? g.storeys} vertical levels (storeys: {g.storeys})</strong> ·
            {g.bays_x != null && g.bays_y != null ? (
              <span>
                {" "}
                {g.bays_x} bay(s) in X, {g.bays_y} bay(s) in Y
              </span>
            ) : null}
          </li>
        </ul>
      ) : null}
      {meta.supports_caption ? <p className="bldg-support-line">{meta.supports_caption}</p> : null}
      {meta.loads_caption ? <p className="bldg-support-line bldg-loads">{meta.loads_caption}</p> : null}
      <div className="bldg-support-schematic" aria-hidden>
        <div className="bldg-slab">Roof + floors</div>
        <div className="bldg-arrows" />
        <div className="bldg-bases">
          <span className="bldg-fix">⛶</span>
          <span className="bldg-fix">⛶</span>
          <span className="bldg-fix">⛶</span>
        </div>
        <p className="small-muted bldg-cap">Each ⛶ = fixed base (6 DOF) at a column; wind / seismic applied per solver assumptions in PyNite.</p>
      </div>
    </div>
  );
}
