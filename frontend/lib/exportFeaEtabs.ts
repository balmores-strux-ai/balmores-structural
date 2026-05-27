import type { FeaPromptResponse } from "@/lib/api";
import { downloadFeaE2k } from "@/lib/exportFeaE2k";

function fmt(n: unknown, d = 4): string {
  const v = typeof n === "number" ? n : Number(n);
  return Number.isFinite(v) ? v.toFixed(d) : "";
}

export function buildFeaEtabsText(res: FeaPromptResponse): string {
  const lines: string[] = [];
  lines.push("BALMORES STRUCTURAL - ETABS-ORIENTED MODEL EXPORT");
  lines.push("=".repeat(72));
  lines.push("NOTE: Use the companion .e2k file for direct ETABS import (File → Import → ETABS .e2k).");
  lines.push("This text file is a human-readable cross-check of the same model.");
  lines.push("");
  lines.push(`Analysis type: ${res.analysis_type}`);
  lines.push(`Load combination used in structural analysis: ${res.load_combination}`);
  lines.push(`Engine: ${res.engine}`);
  lines.push("");
  lines.push("INTERPRETED INPUTS");
  lines.push(res.input_summary.replace(/\*\*/g, ""));
  lines.push("");
  if (res.design_criteria) {
    lines.push("DESIGN CRITERIA");
    lines.push(`Location: ${res.design_criteria.matched_location || res.design_criteria.location_input || "assumed"}`);
    lines.push(`Wind q: ${res.design_criteria.wind?.pressure_kpa ?? ""} kPa`);
    lines.push(`Seismic zone: ${res.design_criteria.seismic?.zone ?? ""}; PGA: ${res.design_criteria.seismic?.pga_g ?? ""} g`);
    lines.push(`SBC: ${res.design_criteria.soil?.sbc_kpa ?? ""} kPa`);
    lines.push("");
  }
  lines.push(`NODES (${res.geometry.nodes.length})`);
  for (const n of res.geometry.nodes) {
    lines.push(`${n.id}, X=${fmt(n.x)}, Y=${fmt(n.y)}, Z=${fmt(n.z)}`);
  }
  lines.push("");
  lines.push(`FRAME OBJECTS (${res.geometry.members.length})`);
  for (const m of res.geometry.members) {
    lines.push(`${m.id}, ${m.start}, ${m.end}, type=${m.kind}`);
  }
  lines.push("");
  lines.push("SUPPORT RESTRAINTS");
  if (res.analysis_type === "building_3d") {
    lines.push("All base nodes (Z=0): U1,U2,U3,R1,R2,R3 fixed.");
  } else {
    lines.push("See support reactions table and geometry.meta.support_kinds for beam/frame support idealization.");
  }
  lines.push("");
  lines.push("LOAD COMBINATIONS");
  const uls = res.design_criteria?.combos?.uls || ["1.2DL + 1.6LL (+ lateral when present)"];
  for (const c of uls) lines.push(`ULS candidate: ${c}`);
  lines.push("");
  lines.push("RESULTS TO VERIFY");
  for (const c of res.result_cards) lines.push(`${c.label}: ${c.value}${c.unit ? " " + c.unit : ""}`);
  lines.push("");
  lines.push("END");
  return lines.join("\n");
}

/** Primary export: CSI ETABS .e2k for File → Import → ETABS (.e2k). */
export function downloadFeaEtabsExports(res: FeaPromptResponse) {
  downloadFeaE2k(res);
}

/** Legacy text + JSON bundle (optional cross-check). */
export function downloadFeaEtabsLegacyBundle(res: FeaPromptResponse) {
  const base = `balmores_etabs_${res.analysis_type}_${Date.now()}`;
  const payload = {
    format: "balmores_etabs_fea_prompt_v2",
    generated_at: new Date().toISOString(),
    analysis_type: res.analysis_type,
    parsed_model: res.parsed_model,
    design_criteria: res.design_criteria,
    geometry: res.geometry,
    load_combination: res.load_combination,
    result_cards: res.result_cards,
    base_reactions: res.base_reactions,
    storey_drifts: res.storey_drifts,
    beams: res.beams,
    columns: res.columns,
    assumptions: res.assumptions,
  };
  const files = [
    { name: `${base}.txt`, type: "text/plain", data: buildFeaEtabsText(res) },
    { name: `${base}.json`, type: "application/json", data: JSON.stringify(payload, null, 2) },
  ];
  for (const f of files) {
    const blob = new Blob([f.data], { type: f.type });
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = f.name;
    a.click();
    window.setTimeout(() => URL.revokeObjectURL(url), 30_000);
  }
}
