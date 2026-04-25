from __future__ import annotations

import asyncio
import hmac
import json
import os
import time
import uuid
from typing import AsyncIterator, Iterator

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


@app.on_event("startup")
def _prewarm() -> None:
    """Warm imports + a tiny PyNite solve so the first user request is FAST.

    PyNite eagerly imports matplotlib & prettytable at module load (≈1.5–3 s
    on cold workers), and scipy.sparse pulls a large compiled extension. Doing
    a 1×1×1 trivial solve here means every real request — including the very
    first — pays only the analysis time, never the import cost.
    """
    try:
        from .pynite_fea import run_parametric_frame_analysis  # eager import

        run_parametric_frame_analysis(
            bays_x=1,
            bays_y=1,
            stories=1,
            span_x_m=4.0,
            span_y_m=4.0,
            bottom_story_height_m=3.0,
            story_height_m=3.0,
            floor_load_kpa=2.0,
        )
    except Exception:
        # Pre-warm is best-effort — never block startup if it fails on Render.
        pass


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


def _run_prompt_pipeline(req: FeaPromptRequest) -> FeaPromptResponse:
    from .fea_prompt_parser import parse_structural_prompt
    from .pynite_fea import (
        run_beam_analysis,
        run_frame_2d_analysis,
        run_irregular_frame_analysis,
    )
    from .schemas import GeometryPayload, ResultCard

    try:
        params, parse_notes = parse_structural_prompt(req.message)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e

    atype = params.pop("analysis_type", "building_3d")
    design_criteria = params.pop("design_criteria_payload", {})

    t0 = time.perf_counter()
    try:
        if atype == "beam_2d":
            input_summary = _summary_beam(params)
            raw = run_beam_analysis(**params)
        elif atype == "frame_2d":
            input_summary = _summary_frame_2d(params)
            raw = run_frame_2d_analysis(**params, run_p_delta=bool(req.run_p_delta))
        else:
            input_summary = _summary_building(params)
            raw = run_irregular_frame_analysis(**params, run_p_delta=bool(req.run_p_delta))
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"PyNite analysis failed: {e}") from e
    elapsed_ms = int((time.perf_counter() - t0) * 1000)

    parsed_model = {k: v for k, v in params.items() if _is_jsonable(v)}
    parsed_model["analysis_type"] = atype

    return FeaPromptResponse(
        analysis_type=raw.get("analysis_type", atype),
        input_summary=input_summary,
        parse_notes=parse_notes,
        parsed_model=parsed_model,
        engine=raw["engine"],
        load_combination=raw["load_combination"],
        geometry=GeometryPayload.model_validate(raw["geometry"]),
        result_cards=[ResultCard(**c) for c in raw["result_cards"]],
        assumptions=raw["assumptions"],
        summary_markdown=raw["summary_markdown"],
        beams=raw.get("beams", []),
        columns=raw.get("columns", []),
        base_reactions=raw.get("base_reactions", []),
        storey_drifts=raw.get("storey_drifts", []),
        p_delta_note=raw.get("p_delta_note", ""),
        totals=raw.get("totals", {}),
        diagrams=raw.get("diagrams", {}),
        design_criteria=design_criteria,
        elapsed_ms=elapsed_ms,
        pynite_path=raw.get("pynite_path", ""),
    )


@app.post(
    "/fea/analyze-prompt",
    response_model=FeaPromptResponse,
    dependencies=[Depends(require_api_key_if_configured)],
)
def fea_analyze_prompt(req: FeaPromptRequest) -> FeaPromptResponse:
    """Parse a natural-language structural brief (2D beam, 2D frame, or 3D building) and solve in PyNite."""
    return _run_prompt_pipeline(req)


# ---------------------------------------------------------------------------
# Streaming endpoint: STAAD-style live progress while solving
# ---------------------------------------------------------------------------


def _estimate_solve_seconds(message: str) -> float:
    """Heuristic: estimate kernel runtime so the progress bar feels honest."""
    from .fea_prompt_parser import _detect_analysis_type, _story_count
    atype = _detect_analysis_type(message)
    if atype == "beam_2d":
        return 1.0
    if atype == "frame_2d":
        return 2.5
    n = _story_count(message) or 6
    # Empirical fit to local benchmark: ~0.45 s/storey for 5x5 bay grid.
    return max(2.0, min(180.0, 1.5 + 0.45 * n))


async def _ndjson_progress(req: FeaPromptRequest) -> AsyncIterator[str]:
    """NDJSON stream: progress events while the solve runs in a worker thread."""
    estimated = _estimate_solve_seconds(req.message)
    started_at = time.perf_counter()

    yield json.dumps(
        {
            "type": "stage",
            "stage": "parse",
            "label": "Parsing your description",
            "estimated_total_seconds": round(estimated, 1),
        }
    ) + "\n"
    await asyncio.sleep(0.05)

    yield json.dumps(
        {
            "type": "stage",
            "stage": "build_model",
            "label": "Building nodes, members, sections, supports",
        }
    ) + "\n"

    loop = asyncio.get_running_loop()
    solver_task = loop.run_in_executor(None, _run_prompt_pipeline, req)

    stages = [
        (0.10, "build_model", "Assembling stiffness blocks"),
        (0.25, "loads", "Applying gravity, wind and seismic load cases"),
        (0.40, "factor", "Sparse Cholesky factorisation of K"),
        (0.55, "solve", "Solving K·u = F (load combinations)"),
        (0.75, "pdelta", "P-Δ second-order iteration"),
        (0.88, "post", "Extracting member envelopes and storey drift"),
        (0.96, "package", "Formatting tables and design criteria"),
    ]
    next_stage_idx = 0
    poll_period = 0.4

    while not solver_task.done():
        await asyncio.sleep(poll_period)
        elapsed = time.perf_counter() - started_at
        progress = min(0.95, elapsed / max(estimated, 0.1))

        while next_stage_idx < len(stages) and progress >= stages[next_stage_idx][0]:
            _, sname, slabel = stages[next_stage_idx]
            yield json.dumps(
                {
                    "type": "stage",
                    "stage": sname,
                    "label": slabel,
                    "progress": round(progress, 3),
                    "elapsed_seconds": round(elapsed, 1),
                }
            ) + "\n"
            next_stage_idx += 1

        yield json.dumps(
            {
                "type": "tick",
                "progress": round(progress, 3),
                "elapsed_seconds": round(elapsed, 1),
                "estimated_total_seconds": round(estimated, 1),
            }
        ) + "\n"

    try:
        result = await solver_task
    except HTTPException as he:
        yield json.dumps(
            {
                "type": "error",
                "status": he.status_code,
                "message": he.detail if isinstance(he.detail, str) else str(he.detail),
            }
        ) + "\n"
        return
    except Exception as e:  # noqa: BLE001
        yield json.dumps({"type": "error", "status": 500, "message": str(e)}) + "\n"
        return

    yield json.dumps(
        {
            "type": "stage",
            "stage": "done",
            "label": "Solve complete",
            "progress": 1.0,
            "elapsed_seconds": round(time.perf_counter() - started_at, 1),
        }
    ) + "\n"
    yield json.dumps({"type": "complete", "data": result.model_dump(mode="json")}) + "\n"


@app.post(
    "/fea/analyze-prompt/stream",
    dependencies=[Depends(require_api_key_if_configured)],
)
async def fea_analyze_prompt_stream(req: FeaPromptRequest) -> StreamingResponse:
    """Same analysis as ``/fea/analyze-prompt`` but yields NDJSON progress events.

    Each line is one of:
      * ``{"type":"stage", ...}``     — solver stage transitions
      * ``{"type":"tick",  ...}``     — periodic progress percent + elapsed time
      * ``{"type":"complete","data":{...}}`` — final FeaPromptResponse payload
      * ``{"type":"error", ...}``     — fatal error
    """
    return StreamingResponse(
        _ndjson_progress(req),
        media_type="application/x-ndjson",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


def _is_jsonable(value) -> bool:
    try:
        json.dumps(value)
        return True
    except (TypeError, ValueError):
        return False


def _summary_beam(p: dict) -> str:
    pts = p.get("point_loads") or []
    line_loads = (
        f"DL **{p.get('dl_kN_per_m', 0):.2f}** kN/m + LL **{p.get('ll_kN_per_m', 0):.2f}** kN/m"
    )
    pt_str = (
        "; ".join(f"**{pl['P_kN']} kN** @ {pl['x_m']} m" for pl in pts) if pts else "none"
    )
    if p.get("spans_m"):
        spans = [float(s) for s in p["spans_m"]]
        support_kinds = p.get("support_kinds") or []
        support_label = " – ".join(str(k).title() for k in support_kinds) or "continuous supports"
        return (
            "**2D continuous beam model read from your text**\n\n"
            f"- **Spans:** {', '.join(f'{s:g}' for s in spans)} m → {len(spans)} span(s), total **{sum(spans):.2f} m**.\n"
            f"- **Supports:** {support_label}\n"
            f"- **Distributed loads:** {line_loads}\n"
            f"- **Point loads:** {pt_str}\n"
            f"- **Material:** {str(p.get('material','concrete')).title()} · section {p.get('beam_width_m',0.3)} × {p.get('beam_depth_m',0.6)} m\n"
        )
    return (
        "**2D beam model read from your text**\n\n"
        f"- **Span:** {p['span_m']:.2f} m · **supports:** {p.get('support_left','pin').title()} – {p.get('support_right','roller').title()}\n"
        f"- **Overhangs:** left {p.get('cantilever_left_m', 0):.2f} m, right {p.get('cantilever_right_m', 0):.2f} m\n"
        f"- **Distributed loads:** {line_loads}\n"
        f"- **Point loads:** {pt_str}\n"
        f"- **Material:** {str(p.get('material','concrete')).title()} · section {p.get('beam_width_m',0.3)} × {p.get('beam_depth_m',0.6)} m\n"
    )


def _summary_frame_2d(p: dict) -> str:
    sx = p.get("spans_m", [])
    sh = p.get("story_heights_m", [])
    lat = p.get("lateral_fx_per_floor_kN", 0)
    return (
        "**2D moment frame model read from your text**\n\n"
        f"- **Bays (X spans, m):** {', '.join(f'{x:g}' for x in sx)} → {len(sx)} bay(s), width **{sum(sx):.2f} m**.\n"
        f"- **Storeys:** {len(sh)} · heights {', '.join(f'{h:g}' for h in sh)} m.\n"
        f"- **Loads:** DL **{p.get('dl_kN_per_m', 0):.2f}** kN/m, LL **{p.get('ll_kN_per_m', 0):.2f}** kN/m on every beam.\n"
        f"- **Lateral per floor:** **{lat:.2f}** kN at windward column (ULS).\n"
        f"- **Material:** {str(p.get('material','concrete')).title()} · beam {p.get('beam_width_m',0.3)}×{p.get('beam_depth_m',0.6)} m, column {p.get('column_width_m',0.45)} m square.\n"
    )


def _summary_building(p: dict) -> str:
    mat = "Steel" if p.get("material_steel") else "Concrete"
    sx = p["spans_x_m"]
    sy = p["spans_y_m"]
    sh = p["story_heights_m"]
    out = (
        "**3D building model read from your text**\n\n"
        f"- **Storeys:** {len(sh)} · typical height {sh[0]:.2f} m each (uniform).\n"
        f"- **X spans (m):** {', '.join(str(v) for v in sx)} → {len(sx)} bays, plan length **{sum(sx):.2f} m**.\n"
        f"- **Y spans (m):** {', '.join(str(v) for v in sy)} → {len(sy)} bays, plan width **{sum(sy):.2f} m**.\n"
        f"- **Loads:** DL **{p['dl_kpa']:.2f}** kPa + slab SW **{p['slab_sw_kpa']:.2f}** kPa on beams; "
        f"LL **{p['ll_kpa']:.2f}** kPa.\n"
        f"- **Material:** {mat} (default section sizes).\n"
    )
    if p.get("wind_pressure_kpa"):
        out += f"- **Wind:** {p['wind_pressure_kpa']} kPa on façade (simplified nodal pattern).\n"
    if p.get("lateral_roof_fraction_of_gravity", 0) > 0:
        out += (
            f"- **Seismic (placeholder):** roof shear ≈ **{p['lateral_roof_fraction_of_gravity']:.0%}** "
            "of estimated gravity.\n"
        )
    if p.get("sbc_kpa") is not None:
        out += f"- **Allowable bearing (your input):** **{p['sbc_kpa']}** kPa.\n"
    return out


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
