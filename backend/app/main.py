from __future__ import annotations

import hmac
import json
import os
import uuid
from typing import Iterator

from fastapi import Depends, FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, Response, StreamingResponse
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.middleware.gzip import GZipMiddleware

from .chat_pipeline import run_chat
from .middleware_extra import (
    AccessLogMiddleware,
    MaxBodyMiddleware,
    ProcessTimeMiddleware,
)
from .schemas import (
    ChatRequest,
    ChatResponse,
    FeaBuildingRequest,
    FeaBuildingResponse,
    FeaPromptRequest,
    FeaPromptResponse,
    VerifyRequest,
    VerifyResponse,
)
from .store import get_store
from .inference import build_geometry
from .model_loader import MODEL_PATH, get_brain
from .etabs_export import build_etabs_export_json, build_etabs_export_text

try:
    import sentry_sdk

    if os.getenv("SENTRY_DSN"):
        sentry_sdk.init(
            dsn=os.environ["SENTRY_DSN"],
            traces_sample_rate=float(os.getenv("SENTRY_TRACES_SAMPLE_RATE", "0.1")),
            environment=os.getenv("SENTRY_ENVIRONMENT", "production"),
        )
except ImportError:
    pass


def require_api_key_if_configured(request: Request) -> None:
    """If API_KEY is set, require X-API-Key or Authorization: Bearer (inlined so deploy never misses app.deps)."""
    expected = os.getenv("API_KEY", "").strip()
    if not expected:
        return
    provided = (request.headers.get("x-api-key") or "").strip()
    if not provided:
        auth = request.headers.get("authorization") or ""
        if auth.lower().startswith("bearer "):
            provided = auth[7:].strip()
    if not provided or len(provided) != len(expected):
        raise HTTPException(status_code=401, detail="Invalid or missing API key")
    if not hmac.compare_digest(provided, expected):
        raise HTTPException(status_code=401, detail="Invalid or missing API key")


class RequestIDMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request, call_next):
        rid = request.headers.get("X-Request-ID") or str(uuid.uuid4())
        request.state.request_id = rid
        response = await call_next(request)
        response.headers["X-Request-ID"] = rid
        return response


def _cors_origins() -> list[str]:
    raw = os.getenv("ALLOWED_ORIGINS", "*").strip()
    parts = [o.strip() for o in raw.split(",") if o.strip()]
    return parts if parts else ["*"]


app = FastAPI(title="BALMORES STRUCTURAL", version="0.1.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=_cors_origins(),
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
app.add_middleware(RequestIDMiddleware)
app.add_middleware(MaxBodyMiddleware)
app.add_middleware(ProcessTimeMiddleware)
app.add_middleware(AccessLogMiddleware)
app.add_middleware(GZipMiddleware, minimum_size=800)


@app.exception_handler(HTTPException)
async def http_error_handler(request: Request, exc: HTTPException) -> JSONResponse:
    rid = getattr(request.state, "request_id", None) or "-"
    detail = exc.detail if isinstance(exc.detail, str) else str(exc.detail)
    return JSONResponse(
        status_code=exc.status_code,
        content={"error": {"code": "http_error", "message": detail}, "request_id": rid},
    )


@app.get("/health")
def health() -> dict:
    out: dict = {
        "status": "ok",
        "brain_pt_path": str(MODEL_PATH),
        "store": "sql" if os.getenv("DATABASE_URL") else "memory",
        "fea": "ok",
    }
    try:
        from .pynite_fea import pynite_available

        out["pynite_path_ok"] = pynite_available()
    except Exception as e:
        out["pynite_path_ok"] = False
        out["pynite_error"] = str(e)[:200]

    try:
        brain = get_brain()
        pm = brain.physics_training_manifest or {}
        out["display_metrics_pipeline"] = "surface_metrics_from_brain(pred,features)+sanitize_pred"
        out["physics_informed"] = bool(pm.get("physics_informed"))
        out["dataset_rows"] = brain.dataset_rows
        out["feature_count"] = len(brain.feature_columns)
        out["target_count"] = len(brain.target_columns)
        methods = pm.get("methods")
        if methods:
            out["brain_physics_methods"] = methods
        vmae = brain.metrics.get("val_mean_mae_all_targets")
        if vmae is not None:
            out["val_mean_mae_all_targets"] = vmae
        out["brain_status"] = "loaded"
    except Exception as e:
        out["brain_status"] = "unavailable"
        out["brain_error"] = str(e)[:300]
    return out


@app.get("/metrics")
def prometheus_metrics() -> Response:
    """Prometheus text when METRICS_ENABLED=1 (in-process counters; single-worker accurate)."""
    if os.getenv("METRICS_ENABLED", "").lower() not in ("1", "true", "yes"):
        raise HTTPException(status_code=404, detail="Metrics disabled")
    from .metrics_state import prometheus_text

    return Response(
        content=prometheus_text(),
        media_type="text/plain; version=0.0.4; charset=utf-8",
    )


@app.get("/ready")
def ready() -> dict:
    """Readiness: optional brain + optional DB ping for orchestrators."""
    out: dict = {"status": "ready", "brain_pt_path": str(MODEL_PATH)}
    try:
        get_brain()
        out["brain"] = "ok"
    except Exception as e:
        out["brain"] = "skipped"
        out["brain_note"] = str(e)[:200]
    dsn = os.getenv("DATABASE_URL", "").strip()
    if not dsn:
        out["database"] = "skipped"
        return out
    try:
        from sqlalchemy import text

        store = get_store()
        if hasattr(store, "_engine"):
            with store._engine.connect() as conn:  # type: ignore[attr-defined]
                conn.execute(text("SELECT 1"))
            out["database"] = "ok"
        else:
            out["database"] = "memory"
    except Exception as e:
        out["status"] = "not_ready"
        out["database"] = "error"
        out["database_error"] = str(e)[:200]
    return out


@app.post(
    "/fea/analyze",
    response_model=FeaBuildingResponse,
    dependencies=[Depends(require_api_key_if_configured)],
)
def fea_analyze(req: FeaBuildingRequest) -> FeaBuildingResponse:
    """3D frame FEA via vendored PyNite (parametric grid, gravity UDL + optional roof lateral)."""
    from .pynite_fea import run_parametric_frame_analysis
    from .schemas import GeometryPayload, ResultCard

    e_mpa = float(req.elastic_modulus_gpa) * 1000.0
    g_mpa = float(req.shear_modulus_gpa) * 1000.0 if req.shear_modulus_gpa is not None else None

    try:
        raw = run_parametric_frame_analysis(
            bays_x=req.bays_x,
            bays_y=req.bays_y,
            stories=req.stories,
            span_x_m=req.span_x_m,
            span_y_m=req.span_y_m,
            bottom_story_height_m=req.bottom_story_height_m,
            story_height_m=req.story_height_m,
            floor_load_kpa=req.floor_load_kpa,
            two_way_fraction=req.two_way_fraction,
            e_mpa=e_mpa,
            nu=req.poisson_ratio,
            g_mpa=g_mpa,
            beam_width_m=req.beam_width_m,
            beam_depth_m=req.beam_depth_m,
            column_width_m=req.column_width_m,
            lateral_fx_total_kn=req.lateral_fx_total_kn,
            check_statics=req.check_statics,
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"PyNite analysis failed: {e}") from e

    return FeaBuildingResponse(
        engine=raw["engine"],
        load_combination=raw["load_combination"],
        geometry=GeometryPayload.model_validate(raw["geometry"]),
        result_cards=[ResultCard(**c) for c in raw["result_cards"]],
        assumptions=raw["assumptions"],
        summary_markdown=raw["summary_markdown"],
        beams=raw["beams"],
        columns=raw["columns"],
        base_reactions_sample=raw["base_reactions_sample"],
        totals=raw["totals"],
        pynite_path=raw.get("pynite_path", ""),
    )


@app.post(
    "/fea/analyze-prompt",
    response_model=FeaPromptResponse,
    dependencies=[Depends(require_api_key_if_configured)],
)
def fea_analyze_prompt(req: FeaPromptRequest) -> FeaPromptResponse:
    """Parse a design brief in plain English, then run irregular-grid PyNite FEA (DL/LL, wind, optional seismic push)."""
    from .fea_prompt_parser import parse_structural_prompt
    from .pynite_fea import run_irregular_frame_analysis
    from .schemas import GeometryPayload, ResultCard

    try:
        params, parse_notes = parse_structural_prompt(req.message)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e

    mat = "Steel" if params.get("material_steel") else "Concrete"
    sx = params["spans_x_m"]
    sy = params["spans_y_m"]
    sh = params["story_heights_m"]
    input_summary = (
        "**Model read from your text**\n\n"
        f"- **Storeys:** {len(sh)} · typical height {sh[0]:.2f} m each (uniform).\n"
        f"- **X spans (m):** {', '.join(str(v) for v in sx)} → {len(sx)} bays, plan length **{sum(sx):.2f} m**.\n"
        f"- **Y spans (m):** {', '.join(str(v) for v in sy)} → {len(sy)} bays, plan width **{sum(sy):.2f} m**.\n"
        f"- **Loads:** DL **{params['dl_kpa']:.2f}** kPa + slab SW **{params['slab_sw_kpa']:.2f}** kPa on beams; "
        f"LL **{params['ll_kpa']:.2f}** kPa.\n"
        f"- **Material:** {mat} (default section sizes for this demo).\n"
    )
    if params.get("wind_pressure_kpa"):
        input_summary += f"- **Wind:** {params['wind_pressure_kpa']} kPa on façade (simplified nodal pattern).\n"
    if params.get("lateral_roof_fraction_of_gravity", 0) > 0:
        input_summary += (
            f"- **Seismic (placeholder):** roof shear ≈ **{params['lateral_roof_fraction_of_gravity']:.0%}** "
            "of estimated gravity.\n"
        )
    if params.get("sbc_kpa") is not None:
        input_summary += f"- **Allowable bearing (your input):** **{params['sbc_kpa']}** kPa.\n"

    try:
        raw = run_irregular_frame_analysis(**params, run_p_delta=req.run_p_delta)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"PyNite analysis failed: {e}") from e

    return FeaPromptResponse(
        input_summary=input_summary,
        parse_notes=parse_notes,
        engine=raw["engine"],
        load_combination=raw["load_combination"],
        geometry=GeometryPayload.model_validate(raw["geometry"]),
        result_cards=[ResultCard(**c) for c in raw["result_cards"]],
        assumptions=raw["assumptions"],
        summary_markdown=raw["summary_markdown"],
        beams=raw["beams"],
        columns=raw["columns"],
        base_reactions=raw["base_reactions"],
        storey_drifts=raw["storey_drifts"],
        p_delta_note=raw["p_delta_note"],
        totals=raw["totals"],
        pynite_path=raw.get("pynite_path", ""),
    )


@app.post("/chat", response_model=ChatResponse, dependencies=[Depends(require_api_key_if_configured)])
def chat(req: ChatRequest) -> ChatResponse:
    store = get_store()
    return run_chat(store, req)


def _ndjson_chunks(req: ChatRequest) -> Iterator[str]:
    store = get_store()
    resp = run_chat(store, req)
    text = resp.messages[0].content if resp.messages else ""
    yield json.dumps({"type": "meta", "project_id": resp.project_id}) + "\n"
    step = max(8, min(24, max(len(text) // 40, 8)))
    if not text:
        yield json.dumps({"type": "delta", "text": ""}) + "\n"
    else:
        for i in range(0, len(text), step):
            yield json.dumps({"type": "delta", "text": text[i : i + step]}) + "\n"
    yield json.dumps({"type": "complete", "data": resp.model_dump(mode="json")}) + "\n"


@app.post("/chat/stream", dependencies=[Depends(require_api_key_if_configured)])
def chat_stream(req: ChatRequest) -> StreamingResponse:
    return StreamingResponse(
        _ndjson_chunks(req),
        media_type="application/x-ndjson",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@app.post("/verify", response_model=VerifyResponse, dependencies=[Depends(require_api_key_if_configured)])
def verify(req: VerifyRequest) -> VerifyResponse:
    return VerifyResponse(
        project_id=req.project_id,
        status="queued",
        message="Prototype mode: ETABS verification worker is not connected yet. Wire your real ETABS API service here next.",
    )


@app.get("/export/etabs/{project_id}", dependencies=[Depends(require_api_key_if_configured)])
def export_etabs_txt(project_id: str) -> Response:
    store = get_store()
    if not store.has_project(project_id):
        raise HTTPException(status_code=404, detail="Unknown project_id")
    state = store.get_state(project_id)
    geom = build_geometry(state)
    text = build_etabs_export_text(state, geom.model_dump())
    fname = f"balmores_etabs_{project_id[:8]}.txt"
    return Response(
        content=text.encode("utf-8"),
        media_type="text/plain; charset=utf-8",
        headers={"Content-Disposition": f'attachment; filename="{fname}"'},
    )


@app.get("/export/etabs/{project_id}/json", dependencies=[Depends(require_api_key_if_configured)])
def export_etabs_json(project_id: str) -> Response:
    store = get_store()
    if not store.has_project(project_id):
        raise HTTPException(status_code=404, detail="Unknown project_id")
    state = store.get_state(project_id)
    geom = build_geometry(state)
    raw = build_etabs_export_json(project_id, state, geom.model_dump())
    fname = f"balmores_etabs_{project_id[:8]}.json"
    return Response(
        content=raw.encode("utf-8"),
        media_type="application/json; charset=utf-8",
        headers={"Content-Disposition": f'attachment; filename="{fname}"'},
    )
