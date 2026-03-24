from app.models import AnalyzeResponse

def build_follow_up_question(resp: AnalyzeResponse) -> str:
    if resp.analysis.story_drift_max_mm > 18:
        return "Do you want me to stiffen the frame or add bracing and compare the updated drift?"
    if resp.analysis.beam_moment_max_kNm > 250:
        return "Do you want me to reduce beam demand by changing bay spacing or adding stiffness?"
    return "Do you want me to generate an ETABS verification job or compare fixed vs pinned supports next?"
