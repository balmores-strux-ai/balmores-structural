import re
from app.models import ProjectState, ProjectRequest

def _first_number(pattern: str, text: str, default: float) -> float:
    m = re.search(pattern, text, re.I)
    return float(m.group(1)) if m else default

def parse_project(req: ProjectRequest) -> ProjectState:
    text = req.prompt.strip()

    stories = int(_first_number(r"(\d+)\s*storey|(?:\d+)\s*story", text, 4))
    # Safer story extraction
    m_story = re.search(r"(\d+)\s*storey|([0-9]+)\s*story", text, re.I)
    if m_story:
      stories = int(next(g for g in m_story.groups() if g))

    plan_match = re.search(r"(\d+(?:\.\d+)?)\s*m\s*(?:x|by)\s*(\d+(?:\.\d+)?)\s*m", text, re.I)
    if plan_match:
        plan_x = float(plan_match.group(1))
        plan_y = float(plan_match.group(2))
    else:
        plan_x, plan_y = 12.0, 18.0

    bays_match = re.search(r"(\d+)\s*bays?\s*(?:each way|in both directions)", text, re.I)
    if bays_match:
        bays_x = bays_y = int(bays_match.group(1))
    else:
        bx = re.search(r"(\d+)\s*bays?\s*in\s*x", text, re.I)
        by = re.search(r"(\d+)\s*bays?\s*in\s*y", text, re.I)
        bays_x = int(bx.group(1)) if bx else 3
        bays_y = int(by.group(1)) if by else 3

    story_h = _first_number(r"(\d+(?:\.\d+)?)\s*m\s*(?:storey|story)\s*height", text, 3.5)

    material_system = "RC" if re.search(r"concrete|rc|reinforced concrete", text, re.I) else "Steel"
    fc = _first_number(r"fc\s*(\d+(?:\.\d+)?)", text, 28.0)
    fy = _first_number(r"fy\s*(\d+(?:\.\d+)?)", text, 415.0)
    sbc = _first_number(r"sbc\s*(\d+(?:\.\d+)?)", text, 150.0)

    support_type = "pinned" if re.search(r"pinned", text, re.I) else "fixed"
    if re.search(r"mixed support", text, re.I):
        support_type = "mixed"

    diaphragm_type = "flexible" if re.search(r"flexible diaphragm", text, re.I) else "rigid"
    if re.search(r"semi.?rigid", text, re.I):
        diaphragm_type = "semi-rigid"

    grid_x = [round(plan_x / bays_x, 3)] * bays_x
    grid_y = [round(plan_y / bays_y, 3)] * bays_y
    story_heights = [story_h] * stories

    return ProjectState(
        project_name=f"{stories}-storey {material_system} structure",
        code=req.code,
        stories=stories,
        plan_x_m=plan_x,
        plan_y_m=plan_y,
        story_heights_m=story_heights,
        grid_x_m=grid_x,
        grid_y_m=grid_y,
        material_system=material_system,
        fc_mpa=fc,
        fy_mpa=fy,
        sbc_kpa=sbc,
        support_type=support_type,
        diaphragm_type=diaphragm_type,
        coordinate_mode=bool(re.search(r"node|coordinate|staad", text, re.I)),
        raw_input=text
    )
