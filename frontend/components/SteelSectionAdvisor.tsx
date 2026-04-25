"use client";

import type { FeaPromptResponse } from "@/lib/api";

type WShape = {
  name: string;
  weightKgM: number;
  sxCm3: number;
  areaCm2: number;
};

const FY_MPA = 345;
const PHI_B = 0.9;
const PHI_C = 0.9;

const W_SHAPES: WShape[] = [
  { name: "W250x33", weightKgM: 33, sxCm3: 450, areaCm2: 42 },
  { name: "W310x39", weightKgM: 39, sxCm3: 650, areaCm2: 50 },
  { name: "W360x51", weightKgM: 51, sxCm3: 950, areaCm2: 65 },
  { name: "W410x60", weightKgM: 60, sxCm3: 1250, areaCm2: 76 },
  { name: "W460x74", weightKgM: 74, sxCm3: 1700, areaCm2: 94 },
  { name: "W530x82", weightKgM: 82, sxCm3: 2200, areaCm2: 104 },
  { name: "W610x101", weightKgM: 101, sxCm3: 3100, areaCm2: 129 },
  { name: "W690x125", weightKgM: 125, sxCm3: 4300, areaCm2: 159 },
  { name: "W760x147", weightKgM: 147, sxCm3: 5600, areaCm2: 187 },
  { name: "W920x201", weightKgM: 201, sxCm3: 9200, areaCm2: 256 },
];

function phiMn(shape: WShape) {
  return (PHI_B * FY_MPA * shape.sxCm3) / 1000;
}

function phiPn(shape: WShape) {
  // Short-column educational estimate only. Slenderness/KL-r is not checked here.
  return PHI_C * FY_MPA * shape.areaCm2 * 0.1;
}

function pickShape(mu: number, pu = 0): { shape: WShape; util: number; mode: string } {
  let best = W_SHAPES[W_SHAPES.length - 1];
  let bestUtil = 999;
  let bestMode = "flexure";
  for (const s of W_SHAPES) {
    const flex = mu > 0 ? mu / phiMn(s) : 0;
    const axial = pu > 0 ? pu / phiPn(s) : 0;
    const util = Math.max(flex, axial);
    if (util >= 0.7 && util <= 0.9) return { shape: s, util, mode: axial > flex ? "axial" : "flexure" };
    if (util < 0.9 && util > 0.45 && Math.abs(0.8 - util) < Math.abs(0.8 - bestUtil)) {
      best = s;
      bestUtil = util;
      bestMode = axial > flex ? "axial" : "flexure";
    }
  }
  if (bestUtil !== 999) return { shape: best, util: bestUtil, mode: bestMode };
  const last = W_SHAPES[W_SHAPES.length - 1];
  const util = Math.max(mu / phiMn(last), pu / phiPn(last));
  return { shape: last, util, mode: pu / phiPn(last) > mu / phiMn(last) ? "axial" : "flexure" };
}

export function steelAdviceForResult(result: FeaPromptResponse) {
  const muBeam = Math.max(0, ...result.beams.map((b) => Math.abs(b.M_max_kNm || 0)));
  const puCol = Math.max(0, ...result.columns.map((c) => Math.abs(c.P_max_kN || 0)));
  const muCol = Math.max(0, ...result.columns.map((c) => Math.max(Math.abs(c.My_max_kNm || 0), Math.abs(c.Mz_max_kNm || 0))));
  const beam = pickShape(muBeam);
  const col = pickShape(muCol, puCol);
  return { muBeam, puCol, muCol, beam, col };
}

export default function SteelSectionAdvisor({ result }: { result: FeaPromptResponse }) {
  const a = steelAdviceForResult(result);
  if (!a.muBeam && !a.puCol) return null;
  return (
    <div className="steel-advisor">
      <div className="steel-advisor-head">
        <strong>Preliminary steel W-shape advisor</strong>
        <span className="small-muted">target utilization about 0.70 to 0.90 where catalogue permits</span>
      </div>
      <div className="steel-advisor-grid">
        <div className="steel-advisor-card">
          <span>Typical beam</span>
          <b>{a.beam.shape.name}</b>
          <p>
            Mu = {a.muBeam.toFixed(1)} kN-m; phiMn = {phiMn(a.beam.shape).toFixed(1)} kN-m; UR ={" "}
            {a.beam.util.toFixed(2)}
          </p>
        </div>
        <div className="steel-advisor-card">
          <span>Typical column / gravity member</span>
          <b>{a.col.shape.name}</b>
          <p>
            Pu = {a.puCol.toFixed(1)} kN, Mu = {a.muCol.toFixed(1)} kN-m; governing {a.col.mode}; UR ={" "}
            {a.col.util.toFixed(2)}
          </p>
        </div>
      </div>
      <p className="small-muted">
        Basis: NSCP 2015 Volume 1, Chapter 5 Structural Steel references AISC 360 LRFD. This quick sizing uses
        phi_b = 0.90, Fy = 345 MPa and plastic/elastic section modulus screening only. Final design must check
        unbraced length, local buckling, shear, combined P-M interaction, deflection, connections, and NSCP load
        combinations.
      </p>
    </div>
  );
}
