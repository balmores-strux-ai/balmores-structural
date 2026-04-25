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
    p("This document reproduces the application output for your records. Verify with a professional engineer; not a design certificate.", { size: 9 }),
    h1("1. Interpreted inputs"),
    p(res.input_summary),
    h1("2. Design parameters (NSCP / location)"),
    ...criteriaToParagraphs(res.design_criteria),
    h1("3. Governing result summary"),
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
      h1("4. Methodology summary (from PyNite output)"),
      p(res.summary_markdown.replace(/\*\*/g, "")),
    );
  } else {
    children.push(h1("4. Methodology"));
  }
  if (res.p_delta_note) children.push(p(`P-Δ: ${res.p_delta_note}`));
  if (typeof res.elapsed_ms === "number") {
    children.push(p(`CPU wall time (solver, ms): ${res.elapsed_ms}`));
  }

  if (res.base_reactions.length) {
    children.push(
      h1("5. Support reactions (ULS) — kN, kN·m"),
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
      h1("6. Storey drift"),
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
      h1("7. Beams (envelopes, excerpt)"),
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
      h1("8. Columns (envelopes, excerpt)"),
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
    h1("9. Assumptions and limits"),
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
