"""
Natural-language → structural-analysis parameters.

Detects whether the user described a **2D beam**, a **2D moment frame**, or a **3D building**
and returns a parameter dict ready for the matching ``run_*_analysis`` function in
:mod:`app.pynite_fea`.

The parser is heuristic (regex-based), not an LLM; it is tuned for engineer-style prompts
with explicit numbers, e.g. ``span 6 m, UDL 15 kN/m, simply supported``.

When the prompt mentions a recognised location (e.g. ``in Manila``,
``location: Cebu``), the parser also resolves wind / seismic / soil parameters
through :mod:`app.design_criteria` and embeds them into the returned dict so the
analysis kernel and UI can show the user the exact code basis used.
"""
from __future__ import annotations

import re
from typing import Any, Dict, List, Optional, Tuple

from .design_criteria import (
    DesignCriteriaResult,
    detect_location_in_text,
    resolve_design_criteria,
)


# ---------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------


def _floats_in_text(s: str) -> List[float]:
    out: List[float] = []
    for part in re.split(r"[,;x×]", s):
        part = part.strip().lower().replace("m", "").replace("meter", "").replace("metre", "").strip()
        if not part:
            continue
        try:
            out.append(float(part))
        except ValueError:
            continue
    return out


def _find_spans_block(text: str, axis: str) -> List[float] | None:
    t = text.lower()
    for pat in (
        rf"{axis}\s*[- ]?spans?\s*\(([^)]+)\)",
        rf"{axis}\s*[- ]?spans?\s*[:=]\s*([^\n.]+)",
        rf"spans?\s+in\s+{axis}\s*[:=]?\s*\(([^)]+)\)",
    ):
        m = re.search(pat, t, re.I)
        if m:
            vals = _floats_in_text(m.group(1))
            if vals:
                return vals
    return None


def _story_count(text: str) -> int | None:
    t = text.lower()
    for pat in (
        r"(\d+)\s*[-]?\s*(?:storeys|stories|storey|story)\b",
        r"(\d+)\s*[-]?\s*floors?\b",
        r"(\d+)\s*[-]?\s*levels?\b",
    ):
        m = re.search(pat, t)
        if m:
            n = int(m.group(1))
            if 1 <= n <= 200:
                return n
    return None


def _parse_continuous_beam_spans(text: str) -> List[float] | None:
    """Detect explicit multi-span continuous-beam descriptions.

    Recognised forms:
      * 'spans (4, 5, 6 m)' or 'spans of 4, 5, 6 m' or 'spans=4,5,6m'
      * 'continuous beam with 3 supports'  → 2 equal spans
      * 'continuous beam, 3 spans of 6 m'  → 3 equal spans of 6 m
      * '4-span beam, span 6 m'            → 4 equal spans of 6 m
    """
    t = text.lower()

    m = re.search(r"\bspans?\s*(?:=|:|of|are)?\s*\(([^)]+)\)", t)
    if m:
        vals = _floats_in_text(m.group(1))
        if len(vals) >= 2:
            return vals

    m = re.search(r"\bspans?\s*(?:of|=|:|are)?\s*([\d.,;\s×x]+?)\s*m\b", t)
    if m and ("," in m.group(1) or ";" in m.group(1) or "x" in m.group(1) or "×" in m.group(1)):
        vals = _floats_in_text(m.group(1))
        if len(vals) >= 2:
            return vals

    m = re.search(
        r"(\d+)\s*[-]?\s*(?:spans?|bays?)\s*(?:of|with)?\s*(\d+(?:\.\d+)?)\s*m",
        t,
    )
    if m:
        n = int(m.group(1))
        L = float(m.group(2))
        if 2 <= n <= 12 and L > 0:
            return [L] * n

    m = re.search(
        r"continuous\s+beam[^\n.]*?(\d+)\s*supports?",
        t,
    )
    if m:
        s = int(m.group(1))
        if 3 <= s <= 12:
            single = _parse_span_single(text) or 6.0
            return [single] * (s - 1)

    m = re.search(r"continuous\s+beam[^\n.]*?(\d+)\s*spans?", t)
    if m:
        s = int(m.group(1))
        if 2 <= s <= 12:
            single = _parse_span_single(text) or 6.0
            return [single] * s

    return None


def _parse_dl_ll_kpa(text: str) -> Tuple[float | None, float | None]:
    dl = ll = None
    t = text.replace("–", "-").lower()
    for m in re.finditer(r"(\d+(?:\.\d+)?)\s*kpa\s*,?\s*(?:dl|dead|d\.l\.|dead\s+load)\b", t):
        dl = float(m.group(1))
    for m in re.finditer(r"(\d+(?:\.\d+)?)\s*kpa\s*,?\s*(?:ll|live|l\.l\.|live\s+load)\b", t):
        ll = float(m.group(1))
    # also support "dl = 4 kpa"
    if dl is None:
        m = re.search(r"\b(?:dl|dead(?:\s*load)?)\s*(?:=|:)?\s*(\d+(?:\.\d+)?)\s*kpa", t)
        if m:
            dl = float(m.group(1))
    if ll is None:
        m = re.search(r"\b(?:ll|live(?:\s*load)?)\s*(?:=|:)?\s*(\d+(?:\.\d+)?)\s*kpa", t)
        if m:
            ll = float(m.group(1))
    return dl, ll


def _parse_dl_ll_kNm(text: str) -> Tuple[float | None, float | None]:
    """Parse DL / LL given in kN/m (line loads on a beam)."""
    dl = ll = None
    t = text.lower()
    # "DL 12 kN/m" style: keyword BEFORE the number
    for m in re.finditer(
        r"(?:dl|dead(?:\s*load)?)\s*(?:=|:)?\s*(\d+(?:\.\d+)?)\s*(?:kn/m|kn\s*per\s*m)\b", t
    ):
        dl = float(m.group(1))
    for m in re.finditer(
        r"(?:ll|live(?:\s*load)?)\s*(?:=|:)?\s*(\d+(?:\.\d+)?)\s*(?:kn/m|kn\s*per\s*m)\b", t
    ):
        ll = float(m.group(1))
    # "12 kN/m DL" style: keyword AFTER the number (only non-digit chars between).
    if dl is None:
        for m in re.finditer(
            r"(\d+(?:\.\d+)?)\s*(?:kn/m|kn\s*per\s*m)[^\d\n.]*?(?:dl|dead)\b", t
        ):
            dl = float(m.group(1))
    if ll is None:
        for m in re.finditer(
            r"(\d+(?:\.\d+)?)\s*(?:kn/m|kn\s*per\s*m)[^\d\n.]*?(?:ll|live)\b", t
        ):
            ll = float(m.group(1))
    return dl, ll


def _parse_udl_total(text: str) -> float | None:
    """Match a total UDL like '15 kN/m udl' or 'udl of 12 kN/m'."""
    t = text.lower()
    m = re.search(r"(?:udl|uniform(?:ly)?\s*(?:distributed)?\s*load)\s*(?:of|=|:)?\s*(\d+(?:\.\d+)?)\s*(?:kn/m|kn\s*per\s*m)", t)
    if m:
        return float(m.group(1))
    m = re.search(r"(\d+(?:\.\d+)?)\s*(?:kn/m|kn\s*per\s*m)\s*(?:udl|uniform)", t)
    if m:
        return float(m.group(1))
    return None


_POINT_LOAD_RE = re.compile(
    r"(\d+(?:\.\d+)?)\s*kn\s*(?:point\s*load)?\s*(?:at|@)\s*"
    r"(\d+(?:\.\d+)?)\s*m(?:\s*(?:from\s*(?:the\s*)?left|from\s*support))?",
    re.I,
)

_MIDSPAN_RE = re.compile(r"(\d+(?:\.\d+)?)\s*kn\s*(?:point\s*load)?\s*(?:at\s*(?:the\s*)?midspan|at\s*mid-span)", re.I)


def _parse_point_loads(text: str, span_m: float | None) -> List[Dict[str, float]]:
    out: List[Dict[str, float]] = []
    for m in _POINT_LOAD_RE.finditer(text):
        out.append({"P_kN": float(m.group(1)), "x_m": float(m.group(2)), "case": "LL"})
    if span_m:
        for m in _MIDSPAN_RE.finditer(text):
            out.append({"P_kN": float(m.group(1)), "x_m": float(span_m) / 2.0, "case": "LL"})
    return out


def _parse_span_single(text: str) -> float | None:
    t = text.lower()
    m = re.search(r"(?:span|length|l\s*=)\s*(?:of|=|:)?\s*(\d+(?:\.\d+)?)\s*m\b", t)
    if m:
        return float(m.group(1))
    m = re.search(r"(\d+(?:\.\d+)?)\s*m\s*(?:long|span)\b", t)
    if m:
        return float(m.group(1))
    return None


def _parse_supports(text: str) -> Tuple[str, str]:
    t = text.lower()
    if "cantilever" in t or "fixed end" in t or "propped" in t:
        if "propped" in t:
            return "fixed", "pin"
        if re.search(r"cantilever\s+(?:from|at)?\s*(?:the\s*)?right", t):
            return "fixed", "free"
        return "fixed", "free"
    if "fixed-fixed" in t or "both ends fixed" in t or "fully fixed" in t:
        return "fixed", "fixed"
    if "simply supported" in t or "simple span" in t or "simply-supported" in t:
        return "pin", "roller"
    if "overhang" in t:
        return "pin", "roller"
    return "pin", "roller"


def _parse_overhangs(text: str) -> Tuple[float, float]:
    t = text.lower()
    cL = cR = 0.0
    m = re.search(r"(\d+(?:\.\d+)?)\s*m\s*(?:left\s*)?overhang", t)
    if m:
        cL = float(m.group(1))
    m = re.search(r"(\d+(?:\.\d+)?)\s*m\s*right\s*overhang", t)
    if m:
        cR = float(m.group(1))
    m = re.search(r"overhangs?\s*(?:of)?\s*(\d+(?:\.\d+)?)\s*m(?:\s*each)?", t)
    if m and cL == 0.0 and cR == 0.0:
        cL = cR = float(m.group(1))
    return cL, cR


def _slab_mm(text: str) -> float | None:
    m = re.search(r"(\d+)\s*mm\s*(?:slab|thick)", text.lower())
    if m:
        return float(m.group(1)) / 1000.0
    return None


def _sbc_kpa(text: str) -> float | None:
    t = text.lower()
    m = re.search(r"(\d+(?:\.\d+)?)\s*kpa\s*(?:sbc|bearing|soil)", t)
    if m:
        return float(m.group(1))
    m2 = re.search(r"(?:sbc|bearing)\s*(?:=|:)?\s*(\d+(?:\.\d+)?)\s*kpa", t)
    if m2:
        return float(m2.group(1))
    return None


def _wind_kpa(text: str) -> float | None:
    m = re.search(r"(\d+(?:\.\d+)?)\s*kpa\s*(?:wl|wind)", text.lower())
    if m:
        return float(m.group(1))
    return None


def _uniform_story_height_m(text: str) -> float | None:
    t = text.lower()
    m = re.search(r"(\d+(?:\.\d+)?)\s*m\s*(?:storey|story|floor)\s*heights?", t)
    if m:
        return float(m.group(1))
    m = re.search(r"(?:storey|story)\s*height[s]?\s*(?:of|=|:)?\s*(\d+(?:\.\d+)?)\s*m", t)
    if m:
        return float(m.group(1))
    m = re.search(r"(?:storeys|stories|storey|story|floors|floor|levels|level)\s+at\s+(\d+(?:\.\d+)?)\s*m", t)
    if m:
        return float(m.group(1))
    m = re.search(r"(\d+(?:\.\d+)?)\s*m\s+(?:storey|story|stories|storeys|floor|floors)\b", t)
    if m:
        return float(m.group(1))
    return None


def _material(text: str) -> Tuple[bool, bool, bool]:
    """Return (is_rc, is_steel, material_detected)."""
    t = text.lower()
    rc = bool(re.search(r"\brc\b|reinforced\s+concrete|concrete\s+(?:frame|beam|building)", t))
    steel = bool(re.search(r"\bsteel\b|structural\s+steel", t))
    return rc, steel, (rc or steel)


# ---------------------------------------------------------------------------
# Analysis-type detection
# ---------------------------------------------------------------------------


def _detect_analysis_type(text: str) -> str:
    t = text.lower()
    has_x_spans = bool(re.search(r"x\s*[- ]?spans?\b", t))
    has_y_spans = bool(re.search(r"y\s*[- ]?spans?\b", t))
    has_storey = bool(re.search(r"\b(?:storey|story|stories|floors?|levels?)\b", t))

    # Explicit 3D building: both X- and Y-span lists.
    if has_x_spans and has_y_spans:
        return "building_3d"

    # Explicit 2D signals.
    if re.search(
        r"\b(?:2d|2-d|planar|plane)\b|\bportal\s+frame\b|\b2d\s+moment\s+frame\b",
        t,
    ):
        return "frame_2d" if re.search(r"\bframe\b|\bbay", t) else "beam_2d"

    # "N bays of X m" with storeys → 2D frame.
    if re.search(r"\b\d+\s*[-]?\s*bays?\s+of\b", t) and has_storey:
        return "frame_2d"
    if re.search(r"\bmoment\s+frame\b", t) and not (has_x_spans and has_y_spans):
        return "frame_2d"

    # Beam keywords without any storeys / building / frame.
    if re.search(r"\bbeam\b", t) and not has_storey and not re.search(r"\bframe\b|\bbuilding\b", t):
        return "beam_2d"

    # Single frame without storeys → 2D frame.
    if re.search(r"\bframe\b", t) and not has_storey and not (has_x_spans and has_y_spans):
        return "frame_2d"

    # Default to 3D building when we see storeys / building keywords.
    if re.search(r"\bbuilding\b", t) or has_storey:
        return "building_3d"

    return "building_3d"


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


def parse_structural_prompt(text: str) -> Tuple[Dict[str, Any], List[str]]:
    """Return ``(params, notes)``.

    ``params`` always contains ``analysis_type`` ∈ ``{"beam_2d","frame_2d","building_3d"}``.
    A separate key ``design_criteria_payload`` carries the wind / seismic / soil
    table resolved from the user's location (or a clearly-tagged moderate
    fallback when no location was given).
    """
    notes: List[str] = []
    raw = text.strip()
    if len(raw) < 8:
        raise ValueError("Describe the structure with spans, loads, and supports (minimum 8 characters).")

    analysis_type = _detect_analysis_type(raw)

    if analysis_type == "beam_2d":
        params = _parse_beam(raw, notes)
    elif analysis_type == "frame_2d":
        params = _parse_frame_2d(raw, notes)
    else:
        params = _parse_building_3d(raw, notes)

    # Design-criteria resolution (location → wind / seismic / soil)
    location = detect_location_in_text(raw)
    user_wind = _wind_kpa(raw)
    user_zone = None
    mz = re.search(r"(?:seismic\s+)?zone\s+(\d)\b", raw.lower())
    if mz:
        user_zone = int(mz.group(1))
    user_sbc = _sbc_kpa(raw)

    user_dl_kpa, user_ll_kpa = _parse_dl_ll_kpa(raw)

    dc = resolve_design_criteria(
        location=location,
        user_dl_kpa=user_dl_kpa,
        user_ll_kpa=user_ll_kpa,
        user_wind_kpa=user_wind,
        user_seismic_zone=user_zone,
        user_sbc_kpa=user_sbc,
    )
    notes.extend(dc.notes)
    params["design_criteria_payload"] = dc.to_dict()

    # If the building branch did not get explicit wind/seismic/sbc but the
    # location resolver supplied them, back-fill the analysis inputs so the
    # solve actually uses them (transparent to the user — assumptions are
    # listed in the design-criteria notes).
    if analysis_type == "building_3d":
        if not params.get("wind_pressure_kpa"):
            params["wind_pressure_kpa"] = float(dc.wind.pressure_kpa)
            notes.append(
                f"Wind pressure auto-applied from design criteria: q = "
                f"{dc.wind.pressure_kpa} kPa (V = {dc.wind.design_wind_speed_mps} m/s)."
            )
        if params.get("sbc_kpa") is None:
            params["sbc_kpa"] = float(dc.soil.sbc_kpa)
        # Seismic equivalent base shear → roof-node lateral push approximation
        if not params.get("lateral_roof_fraction_of_gravity"):
            params["lateral_roof_fraction_of_gravity"] = float(dc.seismic.base_shear_coeff)
            notes.append(
                f"Seismic lateral push set to {dc.seismic.base_shear_coeff:.1%} of gravity "
                f"(zone {dc.seismic.zone}, PGA ≈ {dc.seismic.pga_g} g)."
            )

    return params, notes


# ---------------------------------------------------------------------------
# Branch 1: 2D beam
# ---------------------------------------------------------------------------


def _parse_beam(text: str, notes: List[str]) -> Dict[str, Any]:
    cont_spans = _parse_continuous_beam_spans(text)

    if cont_spans and len(cont_spans) >= 2:
        return _parse_continuous(text, cont_spans, notes)

    span = _parse_span_single(text)
    if not span:
        raise ValueError("Could not find beam span. Try 'simply supported beam, span 6 m, UDL 15 kN/m'.")

    sup_l, sup_r = _parse_supports(text)
    cL, cR = _parse_overhangs(text)

    dl, ll = _parse_dl_ll_kNm(text)
    udl = _parse_udl_total(text)
    if udl and dl is None and ll is None:
        dl = udl
        notes.append(f"Interpreted UDL {udl} kN/m as dead load on beam (pass 'LL=…' to split).")
    if dl is None and ll is None:
        dl = 10.0
        ll = 5.0
        notes.append("No explicit beam loads found; defaulting to DL = 10 kN/m, LL = 5 kN/m.")
    dl = float(dl or 0.0)
    ll = float(ll or 0.0)

    pts = _parse_point_loads(text, span)

    rc, steel, material_detected = _material(text)
    if not material_detected:
        steel = True
        notes.append("Material not specified; assuming structural steel (E ≈ 200 GPa).")

    params: Dict[str, Any] = {
        "analysis_type": "beam_2d",
        "span_m": span,
        "support_left": sup_l,
        "support_right": sup_r,
        "cantilever_left_m": cL,
        "cantilever_right_m": cR,
        "dl_kN_per_m": dl,
        "ll_kN_per_m": ll,
        "point_loads": pts,
        "material": "steel" if steel else "concrete",
        "beam_width_m": 0.25 if steel else 0.30,
        "beam_depth_m": 0.45 if steel else 0.60,
    }
    if pts:
        notes.append(f"Detected {len(pts)} point load(s) on the beam: " + "; ".join(f"{int(p['P_kN'])} kN @ {p['x_m']} m" for p in pts))
    notes.append(f"Beam supports: {sup_l} (left) – {sup_r} (right)" + (f" · overhangs L={cL}m, R={cR}m" if (cL or cR) else ""))
    return params


def _parse_continuous(text: str, spans: List[float], notes: List[str]) -> Dict[str, Any]:
    """Build params for the multi-span continuous-beam solver."""
    n_supports = len(spans) + 1

    dl, ll = _parse_dl_ll_kNm(text)
    udl = _parse_udl_total(text)
    if udl and dl is None and ll is None:
        dl = udl
        notes.append(f"Interpreted UDL {udl} kN/m as dead load on every span.")
    if dl is None and ll is None:
        dl = 10.0
        ll = 5.0
        notes.append("No explicit loads on continuous beam; defaulting to DL = 10 kN/m, LL = 5 kN/m.")
    dl = float(dl or 0.0)
    ll = float(ll or 0.0)

    rc, steel, material_detected = _material(text)
    if not material_detected:
        steel = True
        notes.append("Material not specified; assuming structural steel (E ≈ 200 GPa).")

    # Support kinds: pin at first, rollers elsewhere unless user calls out fixed ends
    sup_kinds = ["pin"] + ["roller"] * (n_supports - 1)
    t = text.lower()
    if "fixed end" in t or "fixed-fixed" in t or "fully fixed" in t:
        sup_kinds[0] = "fixed"
        sup_kinds[-1] = "fixed"
    elif "left fixed" in t:
        sup_kinds[0] = "fixed"
    elif "right fixed" in t:
        sup_kinds[-1] = "fixed"

    pts = _parse_point_loads(text, sum(spans))

    params: Dict[str, Any] = {
        "analysis_type": "beam_2d",
        "spans_m": spans,
        "support_kinds": sup_kinds,
        "dl_kN_per_m": dl,
        "ll_kN_per_m": ll,
        "point_loads": pts,
        "material": "steel" if steel else "concrete",
        "beam_width_m": 0.25 if steel else 0.30,
        "beam_depth_m": 0.45 if steel else 0.60,
    }
    notes.append(
        f"Continuous beam: {len(spans)} spans ({', '.join(str(round(s, 2)) for s in spans)} m), "
        f"{n_supports} supports → {' – '.join(s.title() for s in sup_kinds)}."
    )
    return params


# ---------------------------------------------------------------------------
# Branch 2: 2D moment frame
# ---------------------------------------------------------------------------


def _parse_frame_2d(text: str, notes: List[str]) -> Dict[str, Any]:
    sx = _find_spans_block(text, "x") or _find_spans_block(text, "span") or []
    if not sx:
        m = re.search(r"(\d+)\s*[-]?\s*bays?\b", text.lower())
        if m:
            n = int(m.group(1))
            single = _parse_span_single(text) or 6.0
            sx = [single] * n
        else:
            single = _parse_span_single(text) or 6.0
            sx = [single]
            notes.append(f"Frame bays not given; assuming single bay of {single} m.")

    stories = _story_count(text) or 1
    sh = _uniform_story_height_m(text) or 3.5
    story_heights_m = [sh] * stories

    dl_kn_per_m = ll_kn_per_m = None
    dl_line, ll_line = _parse_dl_ll_kNm(text)
    if dl_line is not None:
        dl_kn_per_m = dl_line
    if ll_line is not None:
        ll_kn_per_m = ll_line
    if dl_kn_per_m is None or ll_kn_per_m is None:
        dl_kpa, ll_kpa = _parse_dl_ll_kpa(text)
        # If user gave kPa, assume 5 m tributary width (a common 2D-frame convention)
        trib = 5.0
        if dl_kn_per_m is None and dl_kpa is not None:
            dl_kn_per_m = float(dl_kpa) * trib
            notes.append(f"DL given in kPa; converted to {dl_kn_per_m:.2f} kN/m on beams using 5 m tributary width.")
        if ll_kn_per_m is None and ll_kpa is not None:
            ll_kn_per_m = float(ll_kpa) * trib
            notes.append(f"LL given in kPa; converted to {ll_kn_per_m:.2f} kN/m on beams using 5 m tributary width.")

    if dl_kn_per_m is None:
        dl_kn_per_m = 15.0
        notes.append("Dead load not found; using 15 kN/m on each beam.")
    if ll_kn_per_m is None:
        ll_kn_per_m = 6.0
        notes.append("Live load not found; using 6 kN/m on each beam.")

    lateral = 0.0
    m = re.search(r"(\d+(?:\.\d+)?)\s*kn\s*(?:lateral|wind|seismic|per\s*floor|each\s*storey)", text.lower())
    if m:
        lateral = float(m.group(1))
    elif _wind_kpa(text):
        lateral = float(_wind_kpa(text)) * 5.0 * sh  # trib façade area per floor
        notes.append(f"Wind pressure interpreted as ~{lateral:.1f} kN lateral point load per floor.")

    rc, steel, material_detected = _material(text)
    if not material_detected:
        rc = True
        notes.append("Material not specified; assuming reinforced concrete frame (E ≈ 30 GPa).")

    params: Dict[str, Any] = {
        "analysis_type": "frame_2d",
        "spans_m": sx,
        "story_heights_m": story_heights_m,
        "dl_kN_per_m": float(dl_kn_per_m),
        "ll_kN_per_m": float(ll_kn_per_m),
        "lateral_fx_per_floor_kN": float(lateral),
        "material": "steel" if steel else "concrete",
        "beam_width_m": 0.30 if steel else 0.35,
        "beam_depth_m": 0.50 if steel else 0.65,
        "column_width_m": 0.35 if steel else 0.50,
    }
    return params


# ---------------------------------------------------------------------------
# Branch 3: 3D building (original behaviour, slightly polished)
# ---------------------------------------------------------------------------


def _parse_building_3d(text: str, notes: List[str]) -> Dict[str, Any]:
    stories = _story_count(text)
    if not stories:
        raise ValueError("Could not find the number of storeys (e.g. '5-storey' or '5 stories').")

    sx = _find_spans_block(text, "x")
    sy = _find_spans_block(text, "y")
    if not sx or not sy:
        raise ValueError(
            "Could not parse X- and Y-span lists. Use e.g. "
            "'X-spans (6, 8, 6 m) and Y-spans (5, 5 m)'."
        )

    sh = _uniform_story_height_m(text)
    if not sh:
        m = re.search(r"(\d+(?:\.\d+)?)\s*m\s*(?:storey|story|floor|height)", text.lower())
        sh = float(m.group(1)) if m else None
    if not sh:
        sh = 3.5
        notes.append("No storey height found; using 3.5 m typical.")
    story_heights_m = [float(sh)] * stories

    dl, ll = _parse_dl_ll_kpa(text)
    if dl is None:
        dl = 5.0
        notes.append("Dead load (DL) not found; using 5.0 kPa.")
    if ll is None:
        ll = 2.0
        notes.append("Live load (LL) not found; using 2.0 kPa.")

    slab_t = _slab_mm(text) or 0.0
    slab_sw_kpa = 25.0 * slab_t if slab_t > 0 else 0.0
    if slab_t > 0:
        notes.append(f"Slab self-weight ~{slab_sw_kpa:.2f} kPa added to DL (25 kN/m³ × thickness).")

    sbc = _sbc_kpa(text)
    wpress = _wind_kpa(text)

    rc, steel, material_detected = _material(text)
    material_steel = steel and not rc
    if not material_detected:
        rc = True
        notes.append("Material not specified; assuming reinforced concrete (E ≈ 30 GPa).")

    seismic_zone = None
    mz = re.search(r"(?:seismic\s+)?zone\s+(\d)\b", text.lower())
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
        "analysis_type": "building_3d",
        "spans_x_m": sx,
        "spans_y_m": sy,
        "story_heights_m": story_heights_m,
        "dl_kpa": float(dl),
        "ll_kpa": float(ll),
        "slab_sw_kpa": float(slab_sw_kpa),
        "wind_pressure_kpa": float(wpress or 0.0),
        "lateral_roof_fraction_of_gravity": lateral_fraction,
        "material_steel": bool(material_steel),
        "sbc_kpa": sbc,
        "two_way_fraction": 0.5,
        "beam_width_m": 0.40 if rc else 0.35,
        "beam_depth_m": 0.65 if rc else 0.55,
        "column_width_m": 0.50 if rc else 0.40,
    }
    if wpress:
        notes.append(f"Wind pressure {wpress} kPa applied to windward façade per storey (simplified).")
    return params
