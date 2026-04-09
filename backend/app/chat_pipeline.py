from __future__ import annotations

import math
from typing import List, Protocol

from .schemas import ChatRequest, ChatResponse, ResultCard, ProjectState, ChatMessage
from .parser import merge_user_message, follow_up_questions
from .inference import (
    brain_target_rows,
    build_geometry,
    build_member_schedule,
    build_physics_checks,
    confidence_label,
    feature_dict_from_state,
    surface_metrics_from_brain,
)
from .model_loader import get_brain
from .recommendations import build_recommendations


class ProjectStore(Protocol):
    def create_project(self, state: ProjectState) -> str: ...

    def has_project(self, project_id: str) -> bool: ...

    def get_state(self, project_id: str) -> ProjectState: ...

    def save_state(self, project_id: str, state: ProjectState) -> None: ...

    def append_message(self, project_id: str, role: str, content: str) -> None: ...

    def get_messages(self, project_id: str) -> List[ChatMessage]: ...


def _sanitize_pred(d: dict) -> dict:
    """Ensure JSON-safe floats (no NaN/inf) for API and streaming clients."""
    out: dict = {}
    for k, v in d.items():
        if isinstance(v, float):
            out[k] = 0.0 if not math.isfinite(v) else v
        else:
            out[k] = v
    return out


def _make_cards(m: dict) -> list[ResultCard]:
    max_drift = m["max_drift_mm"]
    return [
        ResultCard(label="Max drift", value=f"{max_drift:.1f}", unit="mm", tone="warning" if max_drift > 60 else "good"),
        ResultCard(label="Beam max shear", value=f"{m['beam_shear_kN']:.1f}", unit="kN"),
        ResultCard(label="Beam max moment", value=f"{m['beam_moment_kNm']:.1f}", unit="kNm"),
        ResultCard(label="Column axial", value=f"{m['col_axial_kN']:.1f}", unit="kN"),
        ResultCard(label="Joint reaction V", value=f"{m['joint_reaction_vertical_kN']:.1f}", unit="kN"),
        ResultCard(label="Roof disp worst", value=f"{m['roof_disp_mm']:.1f}", unit="mm"),
        ResultCard(label="Base shear worst", value=f"{m['base_shear_kN']:.1f}", unit="kN"),
        ResultCard(label="DCR proxy", value=f"{m['dcr_proxy']:.2f}", unit=None, tone="warning" if m["dcr_proxy"] > 1.0 else "good"),
    ]


def run_chat(store: ProjectStore, req: ChatRequest) -> ChatResponse:
    """Single inference path for /chat and /chat/stream."""
    if req.project_id and store.has_project(req.project_id):
        state = store.get_state(req.project_id)
    else:
        state = ProjectState(code=req.code)
        project_id = store.create_project(state)
        req.project_id = project_id

    state = merge_user_message(state, req.message, req.code)
    store.save_state(req.project_id, state)
    store.append_message(req.project_id, "user", req.message)

    features = feature_dict_from_state(state)
    pred = _sanitize_pred(get_brain().predict(features))
    surf = surface_metrics_from_brain(pred, features)
    geom = build_geometry(state)
    recs = build_recommendations(state, pred, surf)
    cards = _make_cards(surf)
    member_schedule = build_member_schedule(state, pred)

    summary = (
        f"I interpreted your project as a {state.stories}-storey {state.building_type} building "
        f"with {state.bays_x}x{state.bays_y} bays, spans {state.span_x_m:.2f} m x {state.span_y_m:.2f} m, "
        f"{state.support_mode} supports, {state.diaphragm_mode} diaphragm, and {state.brace_mode} brace mode."
    )
    conclusion = (
        f"ETABS-calibrated brain outputs: max drift ≈ {surf['max_drift_mm']:.1f} mm; "
        f"beam shear ≈ {surf['beam_shear_kN']:.1f} kN; beam moment ≈ {surf['beam_moment_kNm']:.1f} kNm; "
        f"column axial ≈ {surf['col_axial_kN']:.1f} kN; joint reaction V ≈ {surf['joint_reaction_vertical_kN']:.1f} kN; "
        f"roof displacement ≈ {surf['roof_disp_mm']:.1f} mm; base shear ≈ {surf['base_shear_kN']:.1f} kN; "
        f"DCR proxy ≈ {surf['dcr_proxy']:.2f}."
    )
    ai_message = summary + " " + conclusion

    store.append_message(req.project_id, "assistant", ai_message)

    charts = {
        "beamColumnSummary": {
            "beamShear": surf["beam_shear_kN"],
            "beamMoment": surf["beam_moment_kNm"],
            "columnAxial": surf["col_axial_kN"],
            "jointReactionV": surf["joint_reaction_vertical_kN"],
        },
        "driftAndBase": {
            "maxDriftMm": surf["max_drift_mm"],
            "roofDispMm": surf["roof_disp_mm"],
            "baseShear": surf["base_shear_kN"],
        },
    }

    physics_checks = build_physics_checks(state, pred, features, surf)
    detailed = {
        "raw_predictions": pred,
        "display_metrics": surf,
        "brain_targets": brain_target_rows(pred),
        "member_schedule": member_schedule,
        "physics_checks": physics_checks,
        "model_info": {
            "dataset_rows": get_brain().dataset_rows,
            "training": "Physics-informed, ETABS-calibrated",
            "targets": list(get_brain().target_columns),
        },
        "analysis_summary": {
            "max_beam_end_shear_kN": pred.get("max_beam_end_shear_kN", 0.0),
            "max_beam_end_moment_kNm": pred.get("max_beam_end_moment_kNm", 0.0),
            "frame_temp_axial_kN": pred.get("frame_p_temp_kN", 0.0),
            "base_mz_eq_kNm": pred.get("base_mz_eq_kNm", 0.0),
        },
    }

    return ChatResponse(
        project_id=req.project_id,
        messages=[
            ChatMessage(role="assistant", content=ai_message),
        ],
        state=state,
        assumptions=state.assumptions,
        geometry=geom,
        result_cards=cards,
        detailed_results=detailed,
        recommendations=recs,
        charts=charts,
        follow_up_questions=follow_up_questions(state),
        confidence=confidence_label(state),
        etabs_verification_available=True,
    )
