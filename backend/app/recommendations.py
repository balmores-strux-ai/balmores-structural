from __future__ import annotations

from typing import Any, Dict, List
from .schemas import Recommendation, ProjectState


def build_recommendations(state: ProjectState, pred: Dict[str, float]) -> List[Recommendation]:
    out: List[Recommendation] = []

    drift_mm = pred.get("max_drift_mm", 0.0)
    beam_m = pred.get("max_beam_moment_kNm", pred.get("beam_m3_grav_kNm", 0.0))
    beam_v = pred.get("max_beam_shear_kN", 0.0)
    col_p = pred.get("max_column_axial_kN", pred.get("col_p_grav_kN", 0.0))
    brace_p = pred.get("brace_p_eq_kN", 0.0)
    dcr = pred.get("max_dcr_proxy", 0.0)

    if drift_mm > 60:
        out.append(Recommendation(
            title="Drift control is a concern",
            detail=f"Predicted maximum drift is about {drift_mm:.1f} mm. Increase lateral stiffness, add bracing/core action, or shorten spans.",
            severity="warning",
        ))
    else:
        out.append(Recommendation(
            title="Drift is within conceptual range",
            detail=f"Predicted maximum drift is about {drift_mm:.1f} mm under the current assumptions.",
            severity="info",
        ))

    if state.brace_mode == "unbraced" and state.stories >= 15:
        out.append(Recommendation(
            title="Tall unbraced frame warning",
            detail="This looks like a drift-sensitive moment-frame case. ETABS verification is strongly recommended before trusting local member actions.",
            severity="warning",
        ))

    if state.support_mode == "pinned":
        out.append(Recommendation(
            title="Pinned-base sensitivity",
            detail="Pinned or semi-pinned support assumptions usually increase overall flexibility and drift. Re-check if fixed base is more realistic.",
            severity="info",
        ))

    out.append(Recommendation(
        title="Beam design direction",
        detail=f"Use beam grouping around conceptual demand levels of moment ≈ {beam_m:.1f} kNm and shear ≈ {beam_v:.1f} kN.",
        severity="info",
    ))

    out.append(Recommendation(
        title="Column design direction",
        detail=f"Use lower-storey column groups around axial demand ≈ {col_p:.1f} kN, reducing in upper storeys.",
        severity="info",
    ))

    if brace_p > 0.0:
        out.append(Recommendation(
            title="Brace demand present",
            detail=f"Predicted brace axial demand reaches about {brace_p:.1f} kN. Keep brace connections and gusset assumptions explicit in later verification.",
            severity="info",
        ))

    if dcr > 1.0:
        out.append(Recommendation(
            title="Demand-capacity proxy exceeds 1.0",
            detail="The conceptual DCR proxy is high. Treat this as a likely overstressed direction until verified in ETABS.",
            severity="warning",
        ))

    return out
