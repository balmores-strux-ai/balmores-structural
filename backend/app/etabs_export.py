from __future__ import annotations

import json
from typing import Any, Dict

from .schemas import ProjectState
from .inference import build_geometry


def _geom_dict(geom: Any) -> Dict[str, Any]:
    if hasattr(geom, "model_dump"):
        return geom.model_dump()
    return geom.dict()


def build_etabs_export_text(state: ProjectState, geometry: Dict[str, Any]) -> str:
    """Human-readable package for CSI ETABS manual setup or scripting (not a binary .edb)."""
    lines: list[str] = []
    lines.append("BALMORES STRUCTURAL — ETABS-oriented model summary")
    lines.append("=" * 64)
    lines.append(
        "NOTE: This is a text + geometry export from your interpreted inputs. "
        "It is NOT a CSI ETABS .edb file. Open ETABS and create a model using the grids, "
        "story data, and loads below, or import the companion JSON in your own workflow."
    )
    lines.append("")
    lines.append("DESIGN BASIS")
    lines.append(f"  Code / region: {state.code}")
    lines.append(f"  Building type: {state.building_type}")
    lines.append(f"  Plan shape: {state.plan_shape}")
    lines.append("")
    lines.append("GEOMETRY — GRID")
    lines.append(f"  Bays (X × Y): {state.bays_x} × {state.bays_y}")
    lines.append(f"  Span X: {state.span_x_m:.4f} m   Span Y: {state.span_y_m:.4f} m")
    lines.append(f"  Stories: {state.stories}")
    lines.append(f"  Bottom story height: {state.bottom_story_height_m:.4f} m")
    lines.append(f"  Typical story height: {state.story_height_m:.4f} m")
    lines.append(f"  Supports: {state.support_mode}")
    lines.append(f"  Diaphragm: {state.diaphragm_mode}")
    lines.append(f"  Bracing mode: {state.brace_mode}")
    lines.append("")
    lines.append("MATERIALS / SITE (interpreted)")
    lines.append(f"  fc: {state.fc_mpa:.2f} MPa   fy: {state.fy_mpa:.2f} MPa")
    lines.append(f"  SBC: {state.sbc_kpa:.2f} kPa")
    lines.append("")
    lines.append("LOADS (kPa — verify in ETABS load patterns)")
    lines.append(f"  Dead: {state.dead_kpa:.3f}   Live: {state.live_kpa:.3f}   Snow: {state.snow_kpa:.3f}")
    lines.append(f"  Wind coeff X/Y: {state.wind_coeff_x:.4f} / {state.wind_coeff_y:.4f}")
    lines.append(f"  Seismic coeff X/Y: {state.eq_coeff_x:.4f} / {state.eq_coeff_y:.4f}")
    lines.append("")
    if state.coordinates_raw:
        lines.append("RAW COORDINATES / NOTES FROM USER")
        lines.append(f"  {state.coordinates_raw}")
        lines.append("")
    if state.assumptions:
        lines.append("ASSUMPTIONS")
        for a in state.assumptions:
            lines.append(f"  • {a}")
        lines.append("")
    nodes = geometry.get("nodes") or []
    members = geometry.get("members") or []
    lines.append(f"NODES ({len(nodes)}) — coordinates in metres (X, Y, Z)")
    for n in nodes:
        lines.append(f"  {n['id']}:  X={float(n['x']):.4f}  Y={float(n['y']):.4f}  Z={float(n['z']):.4f}")
    lines.append("")
    lines.append(f"MEMBERS ({len(members)}) — start/end node IDs")
    for m in members:
        g = m.get("group") or ""
        extra = f"  [{g}]" if g else ""
        lines.append(f"  {m['id']}: {m['start']} → {m['end']}  kind={m['kind']}{extra}")
    lines.append("")
    lines.append("END OF FILE")
    return "\n".join(lines)


def build_etabs_export_json(project_id: str, state: ProjectState, geometry: Dict[str, Any]) -> str:
    payload = {
        "format": "balmores_etabs_export_v1",
        "project_id": project_id,
        "state": state.model_dump(),
        "geometry": geometry,
    }
    return json.dumps(payload, indent=2, ensure_ascii=False)
