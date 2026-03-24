from fastapi import APIRouter
from app.models import AnalyzeResponse, ProjectRequest
from app.services.parser import parse_project
from app.services.inference import run_prototype_inference
from app.services.reporting import build_follow_up_question

router = APIRouter(prefix="/api/chat", tags=["chat"])

@router.post("/analyze", response_model=AnalyzeResponse)
def analyze_prompt(req: ProjectRequest) -> AnalyzeResponse:
    project = parse_project(req)
    analysis = run_prototype_inference(project)
    response = AnalyzeResponse(project=project, analysis=analysis, follow_up_question="")
    response.follow_up_question = build_follow_up_question(response)
    return response
