/** Coerce brain prediction map from JSON (numbers may arrive as strings). */

export function coerceRawPredictions(raw: unknown): Record<string, number> {
  if (!raw || typeof raw !== "object") return {};
  const out: Record<string, number> = {};
  for (const [k, v] of Object.entries(raw as Record<string, unknown>)) {
    if (typeof v === "number" && Number.isFinite(v)) {
      out[k] = v;
      continue;
    }
    if (typeof v === "string" && v.trim() !== "") {
      const n = Number(v);
      if (Number.isFinite(n)) out[k] = n;
    }
  }
  return out;
}

export function unwrapDetailedResults(res: {
  detailed_results?: unknown;
  detailedResults?: unknown;
}): Record<string, unknown> | null {
  const dr = res.detailed_results ?? (res as { detailedResults?: unknown }).detailedResults;
  if (!dr || typeof dr !== "object") return null;
  return dr as Record<string, unknown>;
}
