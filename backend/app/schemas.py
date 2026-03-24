from __future__ import annotations

from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field


class ChatRequest(BaseModel):
    project_id: Optional[str] = None
    message: str
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
