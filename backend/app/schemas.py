from __future__ import annotations

from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field


class ChatRequest(BaseModel):
    project_id: Optional[str] = None
    message: str = Field(..., max_length=32000, min_length=1)
    code: str = Field(default="Canada")
    use_defaults: bool = True


class GeometryNode(BaseModel):
    id: str
    x: float
    y: float
    z: float


class GeometryMember(BaseModel):
    id: str
    start: str
    end: str
    kind: str
    group: Optional[str] = None


class GeometryPayload(BaseModel):
    nodes: List[GeometryNode]
    members: List[GeometryMember]
    meta: Dict[str, Any] = Field(default_factory=dict)


class ResultCard(BaseModel):
    label: str
    value: str
    unit: Optional[str] = None
    tone: str = "neutral"


class Recommendation(BaseModel):
    title: str
    detail: str
    severity: str = "info"


class ProjectState(BaseModel):
    code: str = "Canada"
    building_type: str = "steel"
    stories: int = 4
    bays_x: int = 3
    bays_y: int = 3
    span_x_m: float = 6.0
    span_y_m: float = 6.0
    story_height_m: float = 3.5
    bottom_story_height_m: float = 4.0
    support_mode: str = "fixed"
    diaphragm_mode: str = "rigid"
    brace_mode: str = "braced"
    plan_shape: str = "rect"
    fc_mpa: float = 30.0
    fy_mpa: float = 420.0
    sbc_kpa: float = 150.0
    dead_kpa: float = 2.5
    live_kpa: float = 3.0
    snow_kpa: float = 0.5
    wind_coeff_x: float = 0.03
    wind_coeff_y: float = 0.03
    eq_coeff_x: float = 0.08
    eq_coeff_y: float = 0.08
    twist_total_deg: float = 0.0
    plan_skew_deg: float = 0.0
    lean_total_m: float = 0.0
    setback_ratio: float = 0.0
    eccentricity_ratio: float = 0.0
    cantilever_length_m: float = 0.0
    cantilever_fraction: float = 0.0
    coordinates_raw: Optional[str] = None
    assumptions: List[str] = Field(default_factory=list)
    conversation_notes: List[str] = Field(default_factory=list)


class ChatMessage(BaseModel):
    role: str
    content: str


class ChatResponse(BaseModel):
    project_id: str
    messages: List[ChatMessage]
    state: ProjectState
    assumptions: List[str]
    geometry: GeometryPayload
    result_cards: List[ResultCard]
    detailed_results: Dict[str, Any]
    recommendations: List[Recommendation]
    charts: Dict[str, Any]
    follow_up_questions: List[str]
    confidence: str
    etabs_verification_available: bool = True


class VerifyRequest(BaseModel):
    project_id: str


class VerifyResponse(BaseModel):
    project_id: str
    status: str
    message: str


class FeaBuildingRequest(BaseModel):
    """Parametric 3D frame for PyNite gravity (+ optional lateral) analysis."""

    bays_x: int = Field(default=3, ge=1, le=24)
    bays_y: int = Field(default=3, ge=1, le=24)
    stories: int = Field(default=4, ge=1, le=50)
    span_x_m: float = Field(default=6.0, gt=0, le=80.0)
    span_y_m: float = Field(default=6.0, gt=0, le=80.0)
    bottom_story_height_m: float = Field(default=4.0, gt=0, le=30.0)
    story_height_m: float = Field(default=3.5, gt=0, le=30.0)
    floor_load_kpa: float = Field(
        default=10.0,
        gt=0,
        le=200.0,
        description="Equivalent floor pressure (DL+LL proxy) in kPa, applied on all elevated slabs.",
    )
    two_way_fraction: float = Field(default=0.5, ge=0.0, le=1.0)
    elastic_modulus_gpa: float = Field(default=200.0, gt=0, le=300.0)
    poisson_ratio: float = Field(default=0.3, gt=0, lt=0.499)
    shear_modulus_gpa: Optional[float] = Field(default=None, description="If omitted, G = E / (2(1+ν)).")
    beam_width_m: float = Field(default=0.40, gt=0, le=5.0)
    beam_depth_m: float = Field(default=0.75, gt=0, le=8.0)
    column_width_m: float = Field(default=0.45, gt=0, le=5.0)
    lateral_fx_total_kn: float = Field(
        default=0.0,
        ge=0,
        le=1.0e7,
        description="Optional +X nodal push at roof (kN total), split equally among roof nodes.",
    )
    check_statics: bool = False


class FeaBuildingResponse(BaseModel):
    engine: str
    load_combination: str
    geometry: GeometryPayload
    result_cards: List[ResultCard]
    assumptions: List[str]
    summary_markdown: str
    beams: List[Dict[str, Any]]
    columns: List[Dict[str, Any]]
    base_reactions_sample: List[Dict[str, Any]]
    totals: Dict[str, Any]
    pynite_path: str = ""
