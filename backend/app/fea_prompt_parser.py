"""
Heuristic parser: natural-language structural brief → parameters for PyNite irregular grid FEA.
Not a full NLP model; tuned for explicit numbers (spans, kPa, storeys) like design prompts.
"""
from __future__ import annotations

import re
from typing import Any, Dict, List, Tuple


def _floats_in_parens(s: str) -> List[float]:
    out: List[float] = []
    for part in re.split(r"[,;]", s):
        part = part.strip().lower().replace("m", "").strip()
        if not part:
            continue
        try:
            out.append(float(part))
        except ValueError:
            continue
    return out


def _find_spans_block(text: str, axis: str) -> List[float] | None:
    t = text.lower()
    # X-spans (10, 12, 7m) or x spans: 10, 12, 7
    for pat in (
        rf"{axis}\s*[- ]?spans?\s*\(([^)]+)\)",
        rf"{axis}\s*[- ]?spans?\s*[:=]\s*([^\n]+)",
    ):
        m = re.search(pat, t, re.I)
        if m:
            vals = _floats_in_parens(m.group(1))
            if vals:
                return vals
    return None


def _story_count(text: str) -> int | None:
    t = text.lower()
    for pat in (
        r"(\d+)\s*[-]?\s*(?:storey|story|stories)\b",
        r"(\d+)\s*[-]?\s*floors?\b",
        r"(\d+)\s*[-]?\s*levels?\b",
    ):
        m = re.search(pat, t)
        if m:
            n = int(m.group(1))
            if 1 <= n <= 80:
                return n
    return None


def _kpa_after(text: str, keywords: Tuple[str, ...]) -> float | None:
    t = text.lower()
    for kw in keywords:
        m = re.search(rf"(\d+(?:\.\d+)?)\s*kpa\s*(?:\(|,|\s|$|{re.escape(kw)})", t)
        if not m:
            m = re.search(rf"(\d+(?:\.\d+)?)\s*kpa\s*(?:dl|dead|ll|live|wl|wind|sbc)", t)
        # tighter: number before keyword
        m2 = re.search(rf"(\d+(?:\.\d+)?)\s*kpa[^.\n]*?{re.escape(kw)}", t)
        if m2:
            return float(m2.group(1))
    return None


def _parse_dl_ll(text: str) -> Tuple[float | None, float | None]:
    dl = ll = None
    t = text.replace("–", "-").lower()
    for m in re.finditer(r"(\d+(?:\.\d+)?)\s*kpa\s*,?\s*(?:dl|dead|d\.l\.|dead\s+load)\b", t):
        dl = float(m.group(1))
    for m in re.finditer(r"(\d+(?:\.\d+)?)\s*kpa\s*,?\s*(?:ll|live|l\.l\.|live\s+load)\b", t):
        ll = float(m.group(1))
    return dl, ll


def _slab_mm(text: str) -> float | None:
    m = re.search(r"(\d+)\s*mm\s*(?:slab|thick)", text.lower())
    if m:
        return float(m.group(1)) / 1000.0
    return None


def _sbc_kpa(text: str) -> float | None:
    m = re.search(r"(\d+(?:\.\d+)?)\s*kpa\s*(?:sbc|bearing|soil)", text.lower())
    if m:
        return float(m.group(1))
    m2 = re.search(r"(?:sbc|bearing)\s*(?:=|:)?\s*(\d+(?:\.\d+)?)\s*kpa", text.lower())
    if m2:
        return float(m2.group(1))
    return None


def _wind_kpa(text: str) -> float | None:
    m = re.search(r"(\d+(?:\.\d+)?)\s*kpa\s*(?:wl|wind)", text.lower())
    if m:
        return float(m.group(1))
    return None


def _uniform_story_height_m(text: str) -> float | None:
    m = re.search(r"at\s+(\d+(?:\.\d+)?)\s*m\s*(?:storey|story|floor|height)", text.lower())
    if m:
        return float(m.group(1))
    m2 = re.search(r"(\d+(?:\.\d+)?)\s*m\s*(?:storey|story|floor)\s*heights?", text.lower())
    if m2:
        return float(m2.group(1))
    return None


def parse_structural_prompt(text: str) -> Tuple[Dict[str, Any], List[str]]:
    """
    Returns (params_dict, assumptions/warnings).
    params_dict keys match run_irregular_frame_analysis kwargs where possible.
    """
    notes: List[str] = []
    raw = text.strip()
    if len(raw) < 12:
        raise ValueError("Describe the building with storeys, spans (m), and loads (kPa).")

    stories = _story_count(raw)
    if not stories:
        raise ValueError("Could not find number of storeys (e.g. '5-storey' or '5 stories').")

    sx = _find_spans_block(raw, "x")
    sy = _find_spans_block(raw, "y")
    if not sx or not sy:
        raise ValueError(
            "Could not parse X- and Y-span lists. Use e.g. "
            "'X-spans (10, 12, 7m) and Y-spans (8, 10, 6, 5, 12m)'."
        )

    sh = _uniform_story_height_m(raw)
    if not sh:
        m = re.search(r"(\d+(?:\.\d+)?)\s*m\b", raw.lower())
        sh = float(m.group(1)) if m else None
    if not sh:
        sh = 3.5
        notes.append("No storey height found; using 3.5 m typical.")

    story_heights_m = [float(sh)] * stories

    dl, ll = _parse_dl_ll(raw)
    if dl is None:
        dl = 5.0
        notes.append("Dead load (DL) not found; using 5.0 kPa.")
    if ll is None:
        ll = 2.0
        notes.append("Live load (LL) not found; using 2.0 kPa.")

    slab_t = _slab_mm(raw) or 0.0
    slab_sw_kpa = 25.0 * slab_t if slab_t > 0 else 0.0
    if slab_t > 0:
        notes.append(f"Slab self-weight ~{slab_sw_kpa:.2f} kPa added to DL (25 kN/m^3 * thickness).")

    sbc = _sbc_kpa(raw)
    wpress = _wind_kpa(raw)

    rc = bool(re.search(r"\brc\b|reinforced\s+concrete|concrete\s+frame", raw.lower()))
    steel = bool(re.search(r"\bsteel\b|structural\s+steel", raw.lower()))
    material_steel = steel and not rc
    if not rc and not steel:
        rc = True
        notes.append("Material not specified; assuming reinforced concrete (E ≈ 30 GPa).")

    seismic_zone = None
    mz = re.search(r"(?:seismic\s+)?zone\s+(\d)\b", raw.lower())
    if mz:
        seismic_zone = int(mz.group(1))
    lateral_fraction = 0.0
    if seismic_zone and seismic_zone >= 3:
        lateral_fraction = min(0.12, 0.04 + 0.02 * seismic_zone)
        notes.append(
            f"Seismic zone {seismic_zone}: approximate equivalent lateral force "
            f"{lateral_fraction:.0%} of estimated gravity (roof nodal push; educational model)."
        )

    params: Dict[str, Any] = {
        "spans_x_m": sx,
        "spans_y_m": sy,
        "story_heights_m": story_heights_m,
        "dl_kpa": dl,
        "ll_kpa": ll,
        "slab_sw_kpa": slab_sw_kpa,
        "wind_pressure_kpa": wpress or 0.0,
        "lateral_roof_fraction_of_gravity": lateral_fraction,
        "material_steel": material_steel,
        "sbc_kpa": sbc,
        "two_way_fraction": 0.5,
        "beam_width_m": 0.40 if rc else 0.35,
        "beam_depth_m": 0.65 if rc else 0.55,
        "column_width_m": 0.50 if rc else 0.40,
    }

    if wpress:
        notes.append(f"Wind pressure {wpress} kPa applied to windward façade per storey (simplified).")

    return params, notes
