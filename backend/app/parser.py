from __future__ import annotations

import math
import re
from typing import List, Tuple
from .schemas import ProjectState


CODE_DEFAULTS = {
    "Canada": {"dead_kpa": 2.5, "live_kpa": 3.0, "snow_kpa": 1.0, "wind": 0.035, "eq": 0.08},
    "US": {"dead_kpa": 2.3, "live_kpa": 2.5, "snow_kpa": 0.6, "wind": 0.03, "eq": 0.07},
    "Philippines": {"dead_kpa": 2.5, "live_kpa": 2.0, "snow_kpa": 0.0, "wind": 0.04, "eq": 0.10},
}


def _find_number(pattern: str, text: str) -> float | None:
    m = re.search(pattern, text, re.I)
    return float(m.group(1)) if m else None


def _find_int(pattern: str, text: str) -> int | None:
    m = re.search(pattern, text, re.I)
    return int(m.group(1)) if m else None


def _parse_coordinates(text: str) -> tuple[str | None, float | None, float | None]:
    triplets = re.findall(r'(-?\d+(?:\.\d+)?)\s*,\s*(-?\d+(?:\.\d+)?)\s*,\s*(-?\d+(?:\.\d+)?)', text)
    if not triplets:
        return None, None, None
    pts = [(float(a), float(b), float(c)) for a, b, c in triplets]
    max_x = max(p[0] for p in pts) - min(p[0] for p in pts)
    max_y = max(p[1] for p in pts) - min(p[1] for p in pts)
    return text, abs(max_x), abs(max_y)


def merge_user_message(state: ProjectState, message: str, code: str) -> ProjectState:
    text = message.strip()
    low = text.lower()
    defaults = CODE_DEFAULTS.get(code, CODE_DEFAULTS["Canada"])
    assumptions: List[str] = []

    stories = _find_int(r'(\d+)\s*storey|(?:\b)(\d+)\s*story', low)
    if stories is None:
        m = re.search(r'(\d+)\s*storey|(?:\b)(\d+)\s*story', low)
        if m:
            stories = int(next(g for g in m.groups() if g))
    if stories is not None:
        state.stories = stories

    bays = re.search(r'(\d+)\s*bays?\s*(?:x|by)\s*(\d+)\s*bays?', low)
    if bays:
        state.bays_x = int(bays.group(1))
        state.bays_y = int(bays.group(2))
    else:
        bx = _find_int(r'(\d+)\s*bays?\s*x', low)
        by = _find_int(r'(\d+)\s*bays?\s*y', low)
        if bx:
            state.bays_x = bx
        if by:
            state.bays_y = by

    # Plan dimensions: "30x40", "30 x 40", "30x40m", "30 by 40 m", "30m x 40m"
    dim = re.search(r'(\d+(?:\.\d+)?)\s*(?:m|meter|metre)s?\s*(?:x|by|\*)\s*(\d+(?:\.\d+)?)\s*(?:m|meter|metre)s?', low)
    if not dim:
        dim = re.search(r'(\d+(?:\.\d+)?)\s*(?:x|by|\*)\s*(\d+(?:\.\d+)?)\s*(?:m|meter|metre)?', low)
    if not dim:
        dim = re.search(r'plan\s+(\d+(?:\.\d+)?)\s*(?:x|by)\s*(\d+(?:\.\d+)?)', low)
    if not dim:
        # Two standalone numbers: "30 40" or "30, 40" as plan dims (if context suggests dimensions)
        dim = re.search(r'(?:^|,|\.)\s*(\d+(?:\.\d+)?)\s*[,x\s]\s*(\d+(?:\.\d+)?)\s*(?:m|$)', low)
    if dim:
        px = float(dim.group(1))
        py = float(dim.group(2))
        # If plan total (>20m): divide by bays; else treat as span per bay
        if px > 20 or py > 20:  # likely plan dimensions (e.g. 30x40m)
            state.span_x_m = round(px / max(state.bays_x, 1), 3)
            state.span_y_m = round(py / max(state.bays_y, 1), 3)
        else:
            state.span_x_m = round(px, 3)
            state.span_y_m = round(py, 3)
    else:
        sx = _find_number(r'span\s*x\s*(\d+(?:\.\d+)?)', low) or _find_number(r'(\d+(?:\.\d+)?)\s*m\s*span', low)
        sy = _find_number(r'span\s*y\s*(\d+(?:\.\d+)?)', low)
        if sx:
            state.span_x_m = sx
        if sy:
            state.span_y_m = sy

    sh = _find_number(r'(\d+(?:\.\d+)?)\s*m\s*(?:story|storey)\s*height', low)
    if sh:
        state.story_height_m = sh
        state.bottom_story_height_m = max(sh, state.bottom_story_height_m)

    fc = _find_number(r'fc\s*(\d+(?:\.\d+)?)', low) or _find_number(r'f\'?c\s*(\d+(?:\.\d+)?)', low)
    if fc:
        state.fc_mpa = fc

    fy = _find_number(r'fy\s*(\d+(?:\.\d+)?)', low)
    if fy:
        state.fy_mpa = fy

    sbc = _find_number(r'sbc\s*(\d+(?:\.\d+)?)', low)
    if sbc:
        state.sbc_kpa = sbc

    dl = _find_number(r'dead\s*load\s*(\d+(?:\.\d+)?)', low)
    ll = _find_number(r'live\s*load\s*(\d+(?:\.\d+)?)', low)
    if dl:
        state.dead_kpa = dl
    if ll:
        state.live_kpa = ll

    wx = _find_number(r'wind\s*(?:coefficient|coeff)?\s*x\s*(\d+(?:\.\d+)?)', low)
    wy = _find_number(r'wind\s*(?:coefficient|coeff)?\s*y\s*(\d+(?:\.\d+)?)', low)
    eqx = _find_number(r'seismic\s*(?:coefficient|coeff)?\s*x\s*(\d+(?:\.\d+)?)', low)
    eqy = _find_number(r'seismic\s*(?:coefficient|coeff)?\s*y\s*(\d+(?:\.\d+)?)', low)
    if wx: state.wind_coeff_x = wx
    if wy: state.wind_coeff_y = wy
    if eqx: state.eq_coeff_x = eqx
    if eqy: state.eq_coeff_y = eqy

    twist = _find_number(r'twist\s*(?:angle)?\s*(\d+(?:\.\d+)?)', low)
    if twist is not None:
        state.twist_total_deg = twist

    skew = _find_number(r'skew\s*(?:angle)?\s*(\d+(?:\.\d+)?)', low)
    if skew is not None:
        state.plan_skew_deg = skew

    setback = _find_number(r'setback\s*(?:ratio)?\s*(\d+(?:\.\d+)?)', low)
    if setback is not None:
        state.setback_ratio = setback

    ecc = _find_number(r'eccentricity\s*(?:ratio)?\s*(\d+(?:\.\d+)?)', low)
    if ecc is not None:
        state.eccentricity_ratio = ecc

    cant = _find_number(r'cantilever\s*(?:length)?\s*(\d+(?:\.\d+)?)', low)
    if cant is not None:
        state.cantilever_length_m = cant
        state.cantilever_fraction = min(0.35, cant / max(state.span_x_m, state.span_y_m, 1e-6))

    if "pinned" in low and "fixed" in low:
        state.support_mode = "mixed"
    elif "pinned" in low:
        state.support_mode = "pinned"
    elif "fixed" in low:
        state.support_mode = "fixed"

    if "semi rigid diaphragm" in low or "semi-rigid diaphragm" in low:
        state.diaphragm_mode = "semi_rigid"
    elif "flexible diaphragm" in low:
        state.diaphragm_mode = "flexible"
    elif "rigid diaphragm" in low:
        state.diaphragm_mode = "rigid"

    if "unbraced" in low or "no bracing" in low:
        state.brace_mode = "unbraced"
    elif "mixed bracing" in low:
        state.brace_mode = "mixed"
    elif "braced" in low:
        state.brace_mode = "braced"

    if "l-shape" in low or "l shape" in low:
        state.plan_shape = "L"
    elif "t-shape" in low or "t shape" in low:
        state.plan_shape = "T"
    elif "u-shape" in low or "u shape" in low:
        state.plan_shape = "U"
    elif "setback" in low:
        state.plan_shape = "setback"
    elif "podium" in low and "tower" in low:
        state.plan_shape = "podium_tower"
    else:
        state.plan_shape = state.plan_shape or "rect"

    if "concrete" in low or "rc" in low:
        state.building_type = "concrete"
    elif "steel" in low:
        state.building_type = "steel"

    coords_raw, max_x, max_y = _parse_coordinates(text)
    if coords_raw and max_x and max_y:
        state.coordinates_raw = coords_raw
        state.span_x_m = round(max_x / max(state.bays_x, 1), 3)
        state.span_y_m = round(max_y / max(state.bays_y, 1), 3)

    # Defaults if never specified
    if "dead_kpa" not in state.model_fields_set:
        state.dead_kpa = state.dead_kpa or defaults["dead_kpa"]
    if "live_kpa" not in state.model_fields_set:
        state.live_kpa = state.live_kpa or defaults["live_kpa"]
    if state.snow_kpa == 0 and code == "Canada":
        state.snow_kpa = defaults["snow_kpa"]
    if state.wind_coeff_x == 0:
        state.wind_coeff_x = defaults["wind"]
    if state.wind_coeff_y == 0:
        state.wind_coeff_y = defaults["wind"]
    if state.eq_coeff_x == 0:
        state.eq_coeff_x = defaults["eq"]
    if state.eq_coeff_y == 0:
        state.eq_coeff_y = defaults["eq"]

    # assumptions / prompts
    if "live load" not in low and ll is None:
        assumptions.append(f"Assumed live load = {state.live_kpa:.2f} kPa based on {code} defaults.")
    if "dead load" not in low and dl is None:
        assumptions.append(f"Assumed super dead load = {state.dead_kpa:.2f} kPa based on {code} defaults.")
    if "support" not in low and "fixed" not in low and "pinned" not in low:
        assumptions.append(f"Assumed support mode = {state.support_mode}.")
    if "diaphragm" not in low:
        assumptions.append(f"Assumed diaphragm mode = {state.diaphragm_mode}.")
    if "braced" not in low and "unbraced" not in low:
        assumptions.append(f"Assumed brace mode = {state.brace_mode}.")
    if fc is None:
        assumptions.append(f"Assumed fc = {state.fc_mpa:.1f} MPa.")
    if fy is None:
        assumptions.append(f"Assumed fy = {state.fy_mpa:.1f} MPa.")
    if sbc is None:
        assumptions.append(f"Assumed SBC = {state.sbc_kpa:.1f} kPa.")
    if sh is None:
        assumptions.append(f"Assumed typical storey height = {state.story_height_m:.2f} m.")

    state.code = code
    state.assumptions = assumptions
    state.conversation_notes.append(text)
    return state


def follow_up_questions(state: ProjectState) -> List[str]:
    questions: List[str] = []
    if state.support_mode == "mixed":
        questions.append("Do you want the ground supports mostly fixed with selected pinned points, or a 50/50 mixed support assumption?")
    if state.brace_mode == "unbraced" and state.stories >= 15:
        questions.append("This is a tall unbraced case. Should I keep it as a moment-frame-only assumption or add a stiff core/braced direction?")
    if state.cantilever_length_m > 0 and state.cantilever_fraction == 0:
        questions.append("How many floors carry the cantilever?")
    if state.plan_shape != "rect":
        questions.append("Should I treat the irregular plan as architectural only, or also reduce diaphragm regularity and increase torsion sensitivity?")
    if state.coordinates_raw:
        questions.append("Do you want me to treat the supplied coordinates as the governing geometry instead of the bay-based grid?")
    return questions
