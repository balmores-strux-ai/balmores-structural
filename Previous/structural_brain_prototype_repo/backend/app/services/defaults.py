from app.models import Assumption, ProjectState

def build_assumptions(project: ProjectState) -> list[Assumption]:
    assumptions: list[Assumption] = []

    assumptions.append(Assumption(
        field="Live load",
        value="2.5 kPa",
        reason="Prototype default for general building occupancy when user did not specify live load."
    ))
    assumptions.append(Assumption(
        field="Wind / seismic simplification",
        value="Prototype code-linked default envelope",
        reason="Prototype mode uses simplified defaults until ETABS verification is requested."
    ))

    if project.support_type == "fixed":
        assumptions.append(Assumption(
            field="Support type",
            value="Fixed supports",
            reason="Input did not explicitly override fixed support behavior."
        ))
    if project.diaphragm_type == "rigid":
        assumptions.append(Assumption(
            field="Diaphragm type",
            value="Rigid diaphragm",
            reason="Prototype default for regular floor systems unless user specifies flexible or semi-rigid behavior."
        ))

    return assumptions
