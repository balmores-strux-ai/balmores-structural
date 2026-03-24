from __future__ import annotations

from app.models import AnalysisResults, MemberSummary, ProjectState, Assumption
from app.services.defaults import build_assumptions

def _member_group(moment: float) -> str:
    if moment < 120:
        return "B1 / C1"
    if moment < 250:
        return "B2 / C2"
    if moment < 450:
        return "B3 / C3"
    return "B4 / C4"

def run_prototype_inference(project: ProjectState) -> AnalysisResults:
    assumptions: list[Assumption] = build_assumptions(project)

    total_height = sum(project.story_heights_m)
    area = project.plan_x_m * project.plan_y_m
    stiffness_factor = 1.18 if project.support_type == "fixed" else 0.92
    diaphragm_factor = 1.08 if project.diaphragm_type == "rigid" else 0.95

    material_factor = 1.0 if project.material_system == "RC" else 0.88
    period_1 = round(0.045 * total_height**0.9 * (1.0 / max(material_factor, 0.6)), 3)

    base_shear_x = area * project.stories * 1.7 * material_factor
    base_shear_y = base_shear_x * 1.08

    roof_drift_x = total_height * 3.1 / stiffness_factor / diaphragm_factor
    roof_drift_y = roof_drift_x * 1.12
    story_drift_max = max(roof_drift_x, roof_drift_y) / max(project.stories, 1)

    beam_moment = (project.plan_x_m / max(len(project.grid_x_m), 1)) * 28.0 * project.stories * (1.15 if project.material_system == "RC" else 1.0)
    beam_shear = beam_moment * 0.62
    column_axial = area * project.stories * 9.2 / max((len(project.grid_x_m) + 1) * (len(project.grid_y_m) + 1), 1)
    joint_reaction = max(base_shear_x, base_shear_y) / max((len(project.grid_x_m) + 1) * 2, 1)

    members: list[MemberSummary] = []
    n_beams = max(len(project.grid_y_m) + 1, 1) * len(project.grid_x_m) * project.stories
    n_cols = (len(project.grid_x_m) + 1) * (len(project.grid_y_m) + 1) * project.stories

    for i in range(min(n_beams, 16)):
        factor = 0.72 + (i % 5) * 0.08
        m = beam_moment * factor
        v = beam_shear * factor
        d = roof_drift_x * 0.14 * factor
        members.append(MemberSummary(
            name=f"B{i+1}",
            type="beam",
            max_shear_kN=round(v, 1),
            max_moment_kNm=round(m, 1),
            axial_kN=round(v * 0.09, 1),
            deflection_mm=round(d, 1),
            group=_member_group(m)
        ))

    for i in range(min(n_cols, 16)):
        factor = 0.75 + (i % 4) * 0.09
        p = column_axial * factor
        m = beam_moment * 0.42 * factor
        v = beam_shear * 0.55 * factor
        members.append(MemberSummary(
            name=f"C{i+1}",
            type="column",
            max_shear_kN=round(v, 1),
            max_moment_kNm=round(m, 1),
            axial_kN=round(p, 1),
            deflection_mm=0.0,
            group=_member_group(m)
        ))

    recommendations = [
        "Check Y-direction stiffness because drift is slightly higher in Y than X.",
        "Group beams and columns into a small number of construction families instead of using one unique size per member.",
        "Use ETABS verification if this becomes a serious design option or if irregularity increases."
    ]

    conclusion = (
        f"This {project.stories}-storey {project.material_system} prototype is governed mainly by "
        f"{'Y-direction drift' if roof_drift_y >= roof_drift_x else 'X-direction drift'}. "
        f"The current response is suitable for rapid concept screening, not final sign-off."
    )

    return AnalysisResults(
        roof_drift_x_mm=round(roof_drift_x, 1),
        roof_drift_y_mm=round(roof_drift_y, 1),
        story_drift_max_mm=round(story_drift_max, 1),
        base_shear_x_kN=round(base_shear_x, 1),
        base_shear_y_kN=round(base_shear_y, 1),
        beam_moment_max_kNm=round(beam_moment, 1),
        beam_shear_max_kN=round(beam_shear, 1),
        column_axial_max_kN=round(column_axial, 1),
        joint_reaction_max_kN=round(joint_reaction, 1),
        period_1_s=period_1,
        governing_direction="Y drift" if roof_drift_y >= roof_drift_x else "X drift",
        confidence="Prototype / moderate",
        recommendations=recommendations,
        conclusion=conclusion,
        members=members,
        assumptions=assumptions
    )
