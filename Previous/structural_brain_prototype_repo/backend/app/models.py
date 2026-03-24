from typing import List, Literal
from pydantic import BaseModel, Field

CodeOption = Literal["Canada", "US", "Philippines"]

class Assumption(BaseModel):
    field: str
    value: str
    reason: str

class ProjectRequest(BaseModel):
    prompt: str = Field(min_length=5)
    code: CodeOption

class ProjectState(BaseModel):
    project_name: str
    code: CodeOption
    stories: int
    plan_x_m: float
    plan_y_m: float
    story_heights_m: List[float]
    grid_x_m: List[float]
    grid_y_m: List[float]
    material_system: str
    fc_mpa: float
    fy_mpa: float
    sbc_kpa: float
    support_type: str
    diaphragm_type: str
    coordinate_mode: bool
    raw_input: str

class MemberSummary(BaseModel):
    name: str
    type: Literal["beam", "column"]
    max_shear_kN: float
    max_moment_kNm: float
    axial_kN: float
    deflection_mm: float = 0.0
    group: str | None = None

class AnalysisResults(BaseModel):
    roof_drift_x_mm: float
    roof_drift_y_mm: float
    story_drift_max_mm: float
    base_shear_x_kN: float
    base_shear_y_kN: float
    beam_moment_max_kNm: float
    beam_shear_max_kN: float
    column_axial_max_kN: float
    joint_reaction_max_kN: float
    period_1_s: float
    governing_direction: str
    confidence: str
    recommendations: List[str]
    conclusion: str
    members: List[MemberSummary]
    assumptions: List[Assumption]

class AnalyzeResponse(BaseModel):
    project: ProjectState
    analysis: AnalysisResults
    follow_up_question: str

class VerificationJobRequest(BaseModel):
    project: ProjectState

class VerificationJobResponse(BaseModel):
    job_id: str
    status: str
    message: str
