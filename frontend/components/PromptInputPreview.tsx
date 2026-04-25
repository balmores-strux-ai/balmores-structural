"use client";

type Preview = {
  kind: string;
  framing: string;
  sections: string;
  loads: string;
  notes: string;
};

function numsFromBlock(text: string, axis: "x" | "y"): number[] {
  const re = new RegExp(`${axis}\\s*[- ]?spans?\\s*\\(([^)]*)\\)`, "i");
  const m = re.exec(text);
  if (!m) return [];
  return (m[1].match(/\d+(?:\.\d+)?/g) || []).map(Number).filter((n) => n > 0);
}

function firstNumber(text: string, re: RegExp): number | null {
  const m = re.exec(text);
  if (!m) return null;
  const hit = m.slice(1).find((v) => v != null);
  return hit ? Number(hit) : null;
}

function previewFromText(text: string): Preview {
  const t = text.trim();
  if (!t) {
    return {
      kind: "Live model preview",
      framing: "Start typing a beam, 2D frame, or building prompt.",
      sections: "Example: beam 300 x 600 mm, W-section, RC column/core, slab thickness.",
      loads: "Add DL, LL, wind, seismic, SBC, city, and supports when known.",
      notes: "Missing data will be assumed and shown in the report.",
    };
  }
  const low = t.toLowerCase();
  const isBuilding = /\b(storey|story|building|tower|x-spans|y-spans)\b/.test(low);
  const isFrame = !isBuilding && /\b(frame|portal|bay|storey|story)\b/.test(low);
  const xSpans = numsFromBlock(t, "x");
  const ySpans = numsFromBlock(t, "y");
  const spans = (t.match(/spans?\s*(?:of|=|:)?\s*\(?([0-9.,\s]+)\s*m/i)?.[1].match(/\d+(?:\.\d+)?/g) || [])
    .map(Number)
    .filter((n) => n > 0);
  const stories = firstNumber(low, /(\d+)\s*[- ]?(?:storey|story|floor)/);
  const height = firstNumber(low, /(\d+(?:\.\d+)?)\s*m\s*(?:storey\s*)?heights?/);
  const dl = firstNumber(low, /(\d+(?:\.\d+)?)\s*kpa\s*dl|dl\s*(\d+(?:\.\d+)?)\s*kpa/i);
  const ll = firstNumber(low, /(\d+(?:\.\d+)?)\s*kpa\s*ll|ll\s*(\d+(?:\.\d+)?)\s*kpa/i);
  const sec = low.match(/(\d{2,4})\s*(?:x|×|by)\s*(\d{2,4})\s*mm/i);

  if (isBuilding) {
    return {
      kind: "3D building frame",
      framing:
        xSpans.length || ySpans.length
          ? `Plan grid: X ${xSpans.join(" + ") || "assume 6+6+6"} m; Y ${ySpans.join(" + ") || "assume 6+6"} m; ${stories || "?"} storeys @ ${height || 3.5} m.`
          : `Plan grid will assume X 3 bays x 6 m and Y 2 bays x 6 m if not specified; ${stories || "?"} storeys.`,
      sections: sec
        ? `User section hint: ${sec[1]} x ${sec[2]} mm. Tall buildings auto-scale preliminary beam/column/core equivalent sizes.`
        : "Section layout: RC/steel inferred from text; tall buildings get preliminary high-rise stiffness assumptions.",
      loads: `Loads: DL ${dl ?? "assume 4.5"} kPa, LL ${ll ?? "assume 3.0"} kPa; PH city resolves NSCP wind/seismic/SBC.`,
      notes: "Output will include 3D nodes, fixed supports, member envelopes, drift, ETABS export, and report hand-checks.",
    };
  }
  if (isFrame) {
    return {
      kind: "2D moment frame",
      framing: `Elevation frame: ${spans.length ? spans.join(" + ") : "assumed"} m bay layout; ${stories || "?"} storeys @ ${height || 3.5} m.`,
      sections: sec ? `Section hint: ${sec[1]} x ${sec[2]} mm.` : "Section layout: beam/column sizes assumed if not entered.",
      loads: "Gravity line loads on beams plus lateral nodal loads per floor if entered.",
      notes: "The 2D preview will show node numbers, supports, load arrows, reactions, and S/M diagrams.",
    };
  }
  return {
    kind: "2D beam / continuous beam",
    framing: spans.length ? `Beam spans: ${spans.join(" + ")} m.` : "Beam span/support layout will be parsed from span, supports, and continuous keywords.",
    sections: sec ? `Section hint: ${sec[1]} x ${sec[2]} mm.` : "Section layout: steel/RC section assumed if not entered.",
    loads: "DL/LL UDLs and point loads are converted into PyNite load cases and ULS.",
    notes: "Preview/result will show support symbols, node numbers, reactions, shear, moment, and deflection.",
  };
}

export default function PromptInputPreview({ text }: { text: string }) {
  const p = previewFromText(text);
  return (
    <div className="prompt-input-preview" aria-live="polite">
      <div className="prompt-preview-title">
        <strong>{p.kind}</strong>
        <span>framing + section preview</span>
      </div>
      <div className="prompt-preview-grid">
        <div>
          <b>Framing layout</b>
          <p>{p.framing}</p>
        </div>
        <div>
          <b>Section layout</b>
          <p>{p.sections}</p>
        </div>
        <div>
          <b>Loads / site</b>
          <p>{p.loads}</p>
        </div>
        <div>
          <b>Assumption policy</b>
          <p>{p.notes}</p>
        </div>
      </div>
    </div>
  );
}
