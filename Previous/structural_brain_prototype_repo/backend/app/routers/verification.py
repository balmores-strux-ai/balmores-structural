from uuid import uuid4
from fastapi import APIRouter, BackgroundTasks
from app.models import VerificationJobRequest, VerificationJobResponse

router = APIRouter(prefix="/api/verification", tags=["verification"])

def fake_verification_job(job_id: str) -> None:
    # Replace later with:
    # 1) ETABS model generation
    # 2) ETABS run
    # 3) extraction + report build
    print(f"Prototype verification job queued: {job_id}")

@router.post("/jobs", response_model=VerificationJobResponse)
def create_verification_job(req: VerificationJobRequest, background_tasks: BackgroundTasks) -> VerificationJobResponse:
    job_id = str(uuid4())
    background_tasks.add_task(fake_verification_job, job_id)
    return VerificationJobResponse(
        job_id=job_id,
        status="queued",
        message="Prototype verification job queued. Replace this stub with your ETABS worker later."
    )
