from __future__ import annotations

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from .schemas import ChatRequest, ChatResponse, VerifyRequest, VerifyResponse, ResultCard
from .schemas import ProjectState, ChatMessage
from .store import STORE
from .parser import merge_user_message, follow_up_questions
from .inference import build_geometry, feature_dict_from_state, confidence_label, build_member_schedule
from .model_loader import BRAIN
from .recommendations import build_recommendations

app = FastAPI(title="BALMORES STRUCTURAL", version="0.1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def _pred_val(pred: dict, primary: str, fallback: str | None = None, default: float = 0.0) -> float:
    """Use fallback when primary is missing or zero (so trained outputs show correctly)."""
    v = pred.get(primary, default)
    if v is not None and float(v) != 0:
        return float(v)
    if fallback:
        return float(pred.get(fallback, default))
    return default


def _make_cards(pred: dict) -> list[ResultCard]:
    max_drift = _pred_val(pred, "max_drift_mm") or (_pred_val(pred, "worst_roof_disp_m") * 1000.0)
    beam_shear = _pred_val(pred, "max_beam_shear_kN")
    beam_moment = _pred_val(pred, "max_beam_moment_kNm", "beam_m3_grav_kNm")
    col_axial = _pred_val(pred, "max_column_axial_kN", "col_p_grav_kN")
    joint_v = _pred_val(pred, "max_joint_reaction_vertical_kN")
    roof_disp = _pred_val(pred, "worst_roof_disp_m") * 1000.0
    base_shear = _pred_val(pred, "worst_base_shear_kN") or max(
        pred.get("base_fx_eqx_kN", 0) or 0,
        pred.get("base_fy_eqy_kN", 0) or 0,
        pred.get("base_fx_wx_kN", 0) or 0,
        pred.get("base_fy_wy_kN", 0) or 0,
    )
    dcr = _pred_val(pred, "max_dcr_proxy")
    return [
        ResultCard(label="Max drift", value=f"{max_drift:.1f}", unit="mm", tone="warning" if max_drift > 60 else "good"),
        ResultCard(label="Beam max shear", value=f"{beam_shear:.1f}", unit="kN"),
        ResultCard(label="Beam max moment", value=f"{beam_moment:.1f}", unit="kNm"),
        ResultCard(label="Column axial", value=f"{col_axial:.1f}", unit="kN"),
        ResultCard(label="Joint reaction V", value=f"{joint_v:.1f}", unit="kN"),
        ResultCard(label="Roof disp worst", value=f"{roof_disp:.1f}", unit="mm"),
        ResultCard(label="Base shear worst", value=f"{base_shear:.1f}", unit="kN"),
        ResultCard(label="DCR proxy", value=f"{dcr:.2f}", unit=None, tone="warning" if dcr > 1.0 else "good"),
    ]


@app.get("/health")
def health() -> dict:
    return {
        "status": "ok",
        "dataset_rows": BRAIN.dataset_rows,
        "feature_count": len(BRAIN.feature_columns),
        "target_count": len(BRAIN.target_columns),
    }


@app.post("/chat", response_model=ChatResponse)
def chat(req: ChatRequest) -> ChatResponse:
    if req.project_id and req.project_id in STORE.projects:
        state = STORE.get_state(req.project_id)
    else:
        state = ProjectState(code=req.code)
        project_id = STORE.create_project(state)
        req.project_id = project_id

    state = merge_user_message(state, req.message, req.code)
    STORE.save_state(req.project_id, state)
    STORE.append_message(req.project_id, "user", req.message)

    features = feature_dict_from_state(state)
    pred = BRAIN.predict(features)
    geom = build_geometry(state)
    recs = build_recommendations(state, pred)
    cards = _make_cards(pred)
    member_schedule = build_member_schedule(state, pred)

    max_drift = _pred_val(pred, "max_drift_mm") or (_pred_val(pred, "worst_roof_disp_m") * 1000.0)
    beam_shear = _pred_val(pred, "max_beam_shear_kN")
    beam_m = _pred_val(pred, "max_beam_moment_kNm", "beam_m3_grav_kNm")
    col_p = _pred_val(pred, "max_column_axial_kN", "col_p_grav_kN")
    joint_v = _pred_val(pred, "max_joint_reaction_vertical_kN")
    roof_disp = _pred_val(pred, "worst_roof_disp_m") * 1000.0
    base_shear = _pred_val(pred, "worst_base_shear_kN") or max(
        pred.get("base_fx_eqx_kN", 0) or 0,
        pred.get("base_fy_eqy_kN", 0) or 0,
        pred.get("base_fx_wx_kN", 0) or 0,
        pred.get("base_fy_wy_kN", 0) or 0,
    )

    summary = (
        f"I interpreted your project as a {state.stories}-storey {state.building_type} building "
        f"with {state.bays_x}x{state.bays_y} bays, spans {state.span_x_m:.2f} m x {state.span_y_m:.2f} m, "
        f"{state.support_mode} supports, {state.diaphragm_mode} diaphragm, and {state.brace_mode} brace mode."
    )
    conclusion = (
        f"Predicted max drift is about {max_drift:.1f} mm, beam max shear about "
        f"{beam_shear:.1f} kN, beam max moment about {beam_m:.1f} kNm, "
        f"and column axial about {col_p:.1f} kN."
    )
    ai_message = summary + " " + conclusion

    STORE.append_message(req.project_id, "assistant", ai_message)

    charts = {
        "beamColumnSummary": {
            "beamShear": beam_shear,
            "beamMoment": beam_m,
            "columnAxial": col_p,
            "jointReactionV": joint_v,
        },
        "driftAndBase": {
            "maxDriftMm": max_drift,
            "roofDispMm": roof_disp,
            "baseShear": base_shear,
        },
    }

    detailed = {
        "raw_predictions": pred,
        "member_schedule": member_schedule,
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


@app.post("/verify", response_model=VerifyResponse)
def verify(req: VerifyRequest) -> VerifyResponse:
    return VerifyResponse(
        project_id=req.project_id,
        status="queued",
        message="Prototype mode: ETABS verification worker is not connected yet. Wire your real ETABS API service here next.",
    )
