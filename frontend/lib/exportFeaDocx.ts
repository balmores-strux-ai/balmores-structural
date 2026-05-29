import {
  AlignmentType,
  Document,
  HeadingLevel,
  Packer,
  Paragraph,
  Table,
  TableCell,
  TableRow,
  TextRun,
  WidthType,
} from "docx";
import type { FeaPromptResponse, FeaDesignCriteria } from "@/lib/api";

function p(text: string, opts?: { bold?: boolean; size?: number }) {
  return new Paragraph({
    children: [
      new TextRun({ text, bold: opts?.bold, size: opts?.size != null ? Math.round(opts.size * 2) : 22 }),
    ],
  });
}

function cell(text: string, w?: { pct: number }): TableCell {
  return new TableCell({
    children: [new Paragraph({ children: [new TextRun({ text: String(text) })] })],
    width: w ? { size: w.pct, type: WidthType.PERCENTAGE } : undefined,
  });
}

function h1(text: string) {
  return new Paragraph({
    heading: HeadingLevel.HEADING_1,
    children: [new TextRun({ text, bold: true })],
  });
}

function fmt(n: unknown, d = 2): string {
  if (n === null || n === undefined) return "—";
  const v = typeof n === "number" ? n : Number(n);
  if (!Number.isFinite(v)) return "—";
  return v.toFixed(d);
}

function criteriaToParagraphs(c: FeaDesignCriteria | undefined): Paragraph[] {
  if (!c) return [p("Design criteria: not supplied for this run.")];
  const out: Paragraph[] = [
    p(`Location: ${c.matched_location || c.location_input || "—"}`),
    p(
      `Wind: V = ${c.wind ? fmt(c.wind.design_wind_speed_mps, 0) : "—"} m/s, q = ${c.wind ? fmt(c.wind.pressure_kpa, 2) : "—"} kPa (${c.wind?.code_basis || "—"})`,
    ),
    p(
      `Seismic: zone ${c.seismic?.zone ?? "—"}, PGA = ${c.seismic ? fmt(c.seismic.pga_g, 2) : "—"} g (${c.seismic?.code_basis || "—"})`,
    ),
    p(`SBC: ${c.soil ? fmt(c.soil.sbc_kpa, 0) : "—"} kPa — ${c.soil?.description || "—"}`),
  ];
  if (c.combos?.uls?.length) {
    out.push(
      p("ULS combinations: " + c.combos.uls.join(" · ")),
    );
  }
  out.push(p("Code basis / source note: NSCP 2015 Volume 1 is used for Philippines site parameters. Wind loading is from NSCP Chapter 2, Section 207 (Wind Loads). Earthquake parameters are from NSCP Chapter 2, Section 208 (Earthquake Loads). Structural steel follows NSCP Chapter 5, which references AISC 360 LRFD provisions. Exact page numbers depend on the printed/PDF edition, so the report cites chapter and section references instead of a potentially wrong page number."));
  out.push(p("Snow load: for the listed Philippines city database, snow load is taken as 0.00 kPa unless explicitly entered by the user."));
  return out;
}

function steelSizingParagraphs(res: FeaPromptResponse): Paragraph[] {
  const shapes = [
    { name: "W250x33", sx: 450, area: 42 },
    { name: "W310x39", sx: 650, area: 50 },
    { name: "W360x51", sx: 950, area: 65 },
    { name: "W410x60", sx: 1250, area: 76 },
    { name: "W460x74", sx: 1700, area: 94 },
    { name: "W530x82", sx: 2200, area: 104 },
    { name: "W610x101", sx: 3100, area: 129 },
    { name: "W690x125", sx: 4300, area: 159 },
    { name: "W760x147", sx: 5600, area: 187 },
    { name: "W920x201", sx: 9200, area: 256 },
  ];
  const phi = 0.9;
  const fy = 345;
  const phiMn = (sx: number) => (phi * fy * sx) / 1000;
  const phiPn = (area: number) => phi * fy * area * 0.1;
  const mu = Math.max(0, ...res.beams.map((b) => Math.abs(b.M_max_kNm || 0)));
  const pu = Math.max(0, ...res.columns.map((c) => Math.abs(c.P_max_kN || 0)));
  if (!mu && !pu) return [];
  const pick = (m: number, pAxial = 0) => {
    let best = shapes[shapes.length - 1];
    let bestU = 999;
    for (const s of shapes) {
      const u = Math.max(m ? m / phiMn(s.sx) : 0, pAxial ? pAxial / phiPn(s.area) : 0);
      if (u >= 0.7 && u <= 0.9) return { s, u };
      if (u < 0.9 && Math.abs(0.8 - u) < Math.abs(0.8 - bestU)) {
        best = s;
        bestU = u;
      }
    }
    return { s: best, u: Math.max(m / phiMn(best.sx), pAxial / phiPn(best.area)) };
  };
  const beam = pick(mu);
  const col = pick(Math.max(0, ...res.columns.map((c) => Math.max(Math.abs(c.My_max_kNm || 0), Math.abs(c.Mz_max_kNm || 0)))), pu);
  return [
    h1("6. Preliminary W-shape steel sizing"),
    p(`Beam trial section: ${beam.s.name}; Mu = ${fmt(mu, 1)} kN-m; phiMn = ${fmt(phiMn(beam.s.sx), 1)} kN-m; utilization = ${fmt(beam.u, 2)}.`),
    p(`Column/gravity trial section: ${col.s.name}; Pu = ${fmt(pu, 1)} kN; phiPn(short-column screen) = ${fmt(phiPn(col.s.area), 1)} kN; utilization = ${fmt(col.u, 2)}.`),
    p("Assumptions: Fy = 345 MPa, phi_b = phi_c = 0.90, W-shape catalogue screening only. This is a preliminary selection aiming for roughly 0.70 to 0.90 utilization where possible; final NSCP/AISC design must include unbraced length, KL/r, local buckling, shear, P-M interaction, deflection, connections, and constructability."),
  ];
}

function handcalcParagraphs(res: FeaPromptResponse): Paragraph[] {
  const out: Paragraph[] = [h1("7. Engineer hand-checks / teaching calculations")];
  if (res.analysis_type === "beam_2d") {
    const span = res.parsed_model?.span_m;
    const dl = res.parsed_model?.dl_kN_per_m ?? 0;
    const ll = res.parsed_model?.ll_kN_per_m ?? 0;
    if (typeof span === "number" && span > 0) {
      const wu = 1.2 * dl + 1.6 * ll;
      out.push(p(`ULS distributed load: wu = 1.2DL + 1.6LL = ${fmt(wu)} kN/m.`));
      out.push(p(`Simple-beam reference (for reasonableness only): M = wu L^2 / 8 = ${fmt((wu * span * span) / 8)} kN·m.`));
      out.push(p(`Simple-beam reference shear: V = wu L / 2 = ${fmt((wu * span) / 2)} kN.`));
      out.push(p("Compare these hand values against the PyNite FEM envelope. Continuous/fixed/cantilever cases will differ because end fixity changes moment distribution."));
    }
  } else if (res.analysis_type === "frame_2d") {
    out.push(p("Frame hand-check: base shear equilibrium should approximately balance the applied lateral floor loads in the ULS combination."));
    out.push(p(`Reported Σ base Rx = ${fmt(res.totals.sum_base_Rx_kN)} kN; this should be opposite in sign to the total applied lateral push.`));
    out.push(p("Drift check: storey drift ratio = Δ / h. A common preliminary serviceability target is about h/400 to h/500; final NSCP checks require the correct system, importance, Cd/R, and load level."));
  } else {
    const dc = res.design_criteria;
    out.push(p("Building hand-check: velocity pressure q = 0.613 V^2 / 1000 kPa using V in m/s."));
    if (dc?.wind?.design_wind_speed_mps) {
      const v = dc.wind.design_wind_speed_mps;
      out.push(p(`For V = ${fmt(v, 0)} m/s, q = 0.613 × ${fmt(v, 0)}^2 / 1000 = ${fmt(dc.wind.pressure_kpa, 3)} kPa.`));
    }
    out.push(p("Base reaction check: ΣRz should be close to the factored gravity load carried by the frame in the solved ULS combination."));
    out.push(p(`Reported ΣRz = ${fmt(res.totals.sum_base_Rz_kN, 1)} kN; estimated gravity from parser = ${fmt(res.totals.estimated_gravity_kN, 1)} kN.`));
    out.push(p("Drift check: Δ/h is reported per storey. Values larger than about 0.0025 (h/400) should be treated as a stiffness warning for client submissions unless the final code load level permits otherwise."));
  }
  return out;
}

function diagramParagraphs(res: FeaPromptResponse): Paragraph[] {
  const d = res.diagrams;
  if (!d) return [];
  const out: Paragraph[] = [h1("8. Diagram data / graph points")];
  const addPair = (name: string, pair?: [number[], number[]]) => {
    if (!pair || pair[0].length < 2) return;
    const xs = pair[0];
    const ys = pair[1];
    const maxIdx = ys.reduce((best, v, i) => (Math.abs(v) > Math.abs(ys[best]) ? i : best), 0);
    out.push(p(`${name}: peak = ${fmt(ys[maxIdx])} at x = ${fmt(xs[maxIdx])} m.`));
    const sample = xs.slice(0, 10).map((x, i) => `(${fmt(x, 2)}, ${fmt(ys[i], 2)})`).join("; ");
    out.push(p(`First graph points: ${sample}${xs.length > 10 ? "; ..." : ""}`));
  };
  addPair("Shear V", d.shear_kN as [number[], number[]] | undefined);
  addPair("Moment M", d.moment_kNm as [number[], number[]] | undefined);
  addPair("Deflection δ", d.deflection_mm as [number[], number[]] | undefined);
  if (d.moment_per_level_kNm) {
    for (const [k, pair] of Object.entries(d.moment_per_level_kNm).slice(0, 4)) {
      addPair(`Frame moment ${k}`, pair as [number[], number[]]);
    }
  }
  if (d.shear_per_level_kN) {
    for (const [k, pair] of Object.entries(d.shear_per_level_kN).slice(0, 4)) {
      addPair(`Frame shear ${k}`, pair as [number[], number[]]);
    }
  }
  return out.length > 1 ? out : [];
}

function executiveSummaryParagraphs(res: FeaPromptResponse): Paragraph[] {
  const raw = (res.executive_summary || "").trim();
  if (!raw) return [];
  const out: Paragraph[] = [h1("Executive summary — recommendations and conclusion")];
  for (const block of raw.split(/\n{2,}/)) {
    const text = block.replace(/\*\*/g, "").replace(/^#+\s*/gm, "").trim();
    if (text) out.push(p(text));
  }
  return out;
}

export async function feaResultToDocxBlob(res: FeaPromptResponse, title = "Balmores Structural FEA report"): Promise<Blob> {
  const dateStr = new Date().toLocaleString("en-PH", { dateStyle: "long", timeStyle: "short" });

  const children: (Paragraph | Table)[] = [
    new Paragraph({
      heading: HeadingLevel.TITLE,
      alignment: AlignmentType.CENTER,
      children: [new TextRun({ text: title, bold: true, size: 56 })],
    }),
    new Paragraph({
      alignment: AlignmentType.CENTER,
      children: [
        new TextRun({
          text: "PyNite integrated FEM (MIT) · " + (res.load_combination || "ULS"),
          italics: true,
          size: 24,
        }),
      ],
    }),
    p(`Generated: ${dateStr} · ${res.engine} · type: ${res.analysis_type}`),
    p("Prepared for engineering review. Verify with a licensed professional engineer; this is not a design certificate.", { size: 9 }),
    h1("1. Project summary"),
    p(res.input_summary.replace(/\*\*/g, "")),
    ...executiveSummaryParagraphs(res),
    h1("2. Design parameters (NSCP / location)"),
    ...criteriaToParagraphs(res.design_criteria),
    h1("3. Structural analysis method and justification"),
    p("The analysis model is assembled from the interpreted natural-language input and solved using PyNite finite elements. Nodes, members, supports, and loads are explicitly listed in the web output and ETABS-oriented export."),
    p("For beams and frames, the model uses elastic beam-column elements. For 3D buildings, gravity loads are tributary-area converted to frame line loads; wind/seismic are simplified preliminary lateral actions based on the resolved NSCP-style criteria."),
    p("This is a professional preliminary analysis package for engineering review, education, and model setup. Final client submissions must still be checked against the governing NSCP load combinations, member design provisions, P-Delta stability, diaphragm/core behavior, and licensed engineer judgment."),
    h1("4. Governing PyNite result summary"),
  ];

  for (const card of res.result_cards) {
    children.push(
      p(
        `${card.label}: ${card.value}${card.unit ? " " + card.unit : ""}`,
      ),
    );
  }
  if (res.summary_markdown) {
    children.push(
      h1("5. Methodology summary (from PyNite output)"),
      p(res.summary_markdown.replace(/\*\*/g, "")),
    );
  } else {
    children.push(h1("5. Methodology"));
  }
  if (res.p_delta_note) children.push(p(`P-Δ: ${res.p_delta_note}`));
  if (typeof res.elapsed_ms === "number") {
    children.push(p(`CPU wall time (solver, ms): ${res.elapsed_ms}`));
  }

  children.push(...handcalcParagraphs(res));
  children.push(...diagramParagraphs(res));
  children.push(...steelSizingParagraphs(res));

  if (res.base_reactions.length) {
    children.push(
      h1("9. Support reactions (ULS) — kN, kN-m"),
    );
    const hRow = new TableRow({
      children: [
        cell("Node", { pct: 10 }),
        cell("x (m)"),
        cell("y (m)"),
        cell("Rx"),
        cell("Ry"),
        cell("Rz"),
        cell("Mz"),
      ],
    });
    const rows: TableRow[] = [hRow];
    for (const r of res.base_reactions) {
      rows.push(
        new TableRow({
          children: [
            cell(r.node, { pct: 10 }),
            cell(fmt(r.x_m, 2)),
            cell(fmt("y_m" in r && r.y_m != null ? r.y_m : undefined, 2)),
            cell(fmt(r.Rx_kN)),
            cell(fmt(r.Ry_kN)),
            cell(fmt(r.Rz_kN)),
            cell(fmt(r.Mz_kNm)),
          ],
        }),
      );
    }
    children.push(
      new Table({ width: { size: 100, type: WidthType.PERCENTAGE }, rows }),
    );
  }

  if (res.storey_drifts.length) {
    children.push(
      h1("10. Storey drift"),
    );
    const dRows: TableRow[] = [
      new TableRow({
        children: [cell("Sty"), cell("z top (m)"), cell("h (m)"), cell("drift (mm)"), cell("h ratio")],
      }),
    ];
    for (const s of res.storey_drifts) {
      dRows.push(
        new TableRow({
          children: [
            cell(String(s.storey_index)),
            cell(fmt(s.z_top_m, 2)),
            cell(fmt(s.height_m, 2)),
            cell(fmt(s.max_drift_mm, 2)),
            cell(fmt(s.drift_ratio_h, 4)),
          ],
        }),
      );
    }
    children.push(
      new Table({ width: { size: 100, type: WidthType.PERCENTAGE }, rows: dRows }),
    );
  }

  if (res.beams.length) {
    children.push(
      h1("11. Beams (envelopes, excerpt)"),
    );
    const bRows: TableRow[] = [
      new TableRow({ children: [cell("Member"), cell("|M| (kN·m)"), cell("|V| (kN)"), cell("δ (mm)")] }),
    ];
    for (const b of res.beams) {
      bRows.push(
        new TableRow({
          children: [
            cell(b.id, { pct: 25 }),
            cell(fmt(b.M_max_kNm)),
            cell(fmt(b.V_max_kN)),
            cell(fmt(b.deflection_mm, 2)),
          ],
        }),
      );
    }
    children.push(
      new Table({ width: { size: 100, type: WidthType.PERCENTAGE }, rows: bRows }),
    );
  }
  if (res.columns.length) {
    children.push(
      h1("12. Columns (envelopes, excerpt)"),
    );
    const cRows: TableRow[] = [
      new TableRow({
        children: [cell("Member"), cell("|P| (kN)"), cell("|My|"), cell("|Mz|"), cell("|T|")],
      }),
    ];
    for (const c of res.columns) {
      cRows.push(
        new TableRow({
          children: [
            cell(c.id, { pct: 22 }),
            cell(fmt(c.P_max_kN)),
            cell(fmt(c.My_max_kNm)),
            cell(fmt(c.Mz_max_kNm)),
            cell(fmt(c.T_max_kNm)),
          ],
        }),
      );
    }
    children.push(
      new Table({ width: { size: 100, type: WidthType.PERCENTAGE }, rows: cRows }),
    );
  }
  children.push(
    h1("13. Assumptions and limits"),
  );
  for (const a of res.assumptions) {
    children.push(
      p(a),
    );
  }
  children.push(
    p("End of report.", { bold: true }),
  );

  const doc = new Document({
    sections: [
      {
        properties: {},
        children,
      },
    ],
  });
  return Packer.toBlob(doc);
}

export function downloadAndTryOpenDocx(blob: Blob, fileName: string) {
  const f = fileName.replace(/[\\/]+/g, "-");
  const url = URL.createObjectURL(blob);
  const a = document.createElement("a");
  a.href = url;
  a.download = f.endsWith(".docx") ? f : `${f}.docx`;
  a.style.display = "none";
  document.body.appendChild(a);
  a.click();
  document.body.removeChild(a);
  window.setTimeout(() => URL.revokeObjectURL(url), 60_000);
}
