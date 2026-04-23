"""
PyNite-based structural analysis kernel for the Balmores Structural website.

Wraps the vendored PyNite finite element library (open-source, MIT-licensed) and exposes
three analysis modes:

* :func:`run_beam_analysis`              – 2D simply-supported / overhanging / fixed beams
* :func:`run_frame_2d_analysis`          – 2D portal / multi-bay multi-storey moment frames
* :func:`run_parametric_frame_analysis`  – regular 3D building frame grid
* :func:`run_irregular_frame_analysis`   – irregular-bay 3D building frame

The vendored PyNite sources are added to ``sys.path`` at import time.
Resolution order:

1. ``BALMORES_PYNITE_ROOT`` environment variable (must contain a ``Pynite/`` package).
2. Nested ``Pynite-main/Pynite-main`` (how GitHub zips are usually unpacked).
3. Flat ``Pynite-main``.
4. ``structural_analysis_pynite`` (legacy location).
"""
from __future__ import annotations

import math
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

_REPO_ROOT = Path(__file__).resolve().parents[2]


def _has_pynite_pkg(root: Path) -> bool:
    return (root / "Pynite" / "__init__.py").is_file()


def _resolve_pynite_root() -> Path:
    env = os.environ.get("BALMORES_PYNITE_ROOT", "").strip()
    if env:
        p = Path(env)
        if _has_pynite_pkg(p):
            return p
    candidates = [
        _REPO_ROOT / "Pynite-main" / "Pynite-main",
        _REPO_ROOT / "Pynite-main",
        _REPO_ROOT / "structural_analysis_pynite",
    ]
    for p in candidates:
        if _has_pynite_pkg(p):
            return p
    return _REPO_ROOT / "Pynite-main" / "Pynite-main"


_PYNITE_ROOT = _resolve_pynite_root()
if _PYNITE_ROOT.is_dir() and str(_PYNITE_ROOT) not in sys.path:
    sys.path.insert(0, str(_PYNITE_ROOT))


def pynite_available() -> bool:
    return _has_pynite_pkg(_PYNITE_ROOT)


def pynite_root_label() -> str:
    """Short, user-safe label for the FEM source (no filesystem paths shown)."""
    return "Integrated PyNite FEM (open-source)"


def _mag(res: Any) -> float:
    v = res[0] if isinstance(res, tuple) else res
    return abs(float(v))


def _safe_member_scalar(getter, default: float = 0.0) -> float:
    """PyNite may raise KeyError on node displacements for a combo after P-Δ; keep extraction alive."""
    try:
        return _mag(getter())
    except (KeyError, TypeError, ValueError, AttributeError):
        return default


def _z_levels(
    stories: int,
    bottom_story_height_m: float,
    story_height_m: float,
) -> List[float]:
    z = [0.0, float(bottom_story_height_m)]
    for _ in range(1, int(stories)):
        z.append(z[-1] + float(story_height_m))
    return z


def run_parametric_frame_analysis(
    *,
    bays_x: int,
    bays_y: int,
    stories: int,
    span_x_m: float,
    span_y_m: float,
    bottom_story_height_m: float,
    story_height_m: float,
    floor_load_kpa: float,
    two_way_fraction: float = 0.5,
    e_mpa: float = 200_000.0,
    nu: float = 0.3,
    g_mpa: float | None = None,
    beam_width_m: float = 0.40,
    beam_depth_m: float = 0.75,
    column_width_m: float = 0.45,
    lateral_fx_total_kn: float = 0.0,
    check_statics: bool = False,
) -> Dict[str, Any]:
    """
    Regular grid 3D frame; gravity as member UDL (two-way slab split); optional +X nodal wind at roof.

    Units: m, kN, kN·m; E, G in MPa internally converted to kN/m² for PyNite (1 MPa = 1000 kN/m²).
    """
    if not pynite_available():
        raise RuntimeError(f"PyNite folder not found at {_PYNITE_ROOT}")

    from Pynite import FEModel3D  # noqa: WPS433 — after sys.path

    bx, by = int(bays_x), int(bays_y)
    st = int(stories)
    if bx < 1 or by < 1:
        raise ValueError("bays_x and bays_y must be at least 1")
    if st < 1 or st > 50:
        raise ValueError("stories must be between 1 and 50")
    if span_x_m <= 0 or span_y_m <= 0:
        raise ValueError("spans must be positive")

    XS = [i * float(span_x_m) for i in range(bx + 1)]
    YS = [j * float(span_y_m) for j in range(by + 1)]
    ZS = _z_levels(st, bottom_story_height_m, story_height_m)
    nx, ny, nz = len(XS), len(YS), len(ZS)

    node_count = nx * ny * nz
    if node_count > 8000:
        raise ValueError("Model too large for this demo API; reduce bays or stories.")

    Q = float(floor_load_kpa)
    two = max(0.0, min(1.0, float(two_way_fraction)))

    def tributary_y(j: int) -> float:
        if j == 0:
            return (YS[1] - YS[0]) / 2
        if j == ny - 1:
            return (YS[j] - YS[j - 1]) / 2
        return (YS[j] - YS[j - 1]) / 2 + (YS[j + 1] - YS[j]) / 2

    def tributary_x(i: int) -> float:
        if i == 0:
            return (XS[1] - XS[0]) / 2
        if i == nx - 1:
            return (XS[i] - XS[i - 1]) / 2
        return (XS[i] - XS[i - 1]) / 2 + (XS[i + 1] - XS[i]) / 2

    def w_x_beam(j: int) -> float:
        return two * Q * tributary_y(j)

    def w_y_beam(i: int) -> float:
        return two * Q * tributary_x(i)

    def node_name(i: int, j: int, k: int) -> str:
        return f"n_{i}_{j}_{k}"

    E_knm2 = float(e_mpa) * 1000.0
    if g_mpa is None:
        G_knm2 = E_knm2 / (2.0 * (1.0 + float(nu)))
    else:
        G_knm2 = float(g_mpa) * 1000.0
    rho = 77.0

    m = FEModel3D()
    m.add_material("Steel", E_knm2, G_knm2, float(nu), rho)

    b_b, h_b = float(beam_width_m), float(beam_depth_m)
    A_b = b_b * h_b
    Iy_b = h_b * b_b**3 / 12
    Iz_b = b_b * h_b**3 / 12
    J_b = (b_b * h_b**3 + h_b * b_b**3) / 12

    b_c = float(column_width_m)
    A_c = b_c**2
    Iy_c = b_c**4 / 12
    Iz_c = Iy_c
    J_c = 0.5 * Iy_c

    m.add_section("Beam", A_b, Iy_b, Iz_b, J_b)
    m.add_section("Column", A_c, Iy_c, Iz_c, J_c)

    for k in range(nz):
        for j in range(ny):
            for i in range(nx):
                m.add_node(node_name(i, j, k), XS[i], YS[j], ZS[k])

    for i in range(nx):
        for j in range(ny):
            m.def_support(node_name(i, j, 0), True, True, True, True, True, True)

    beam_x_members: List[Tuple[str, int, int, int]] = []
    beam_y_members: List[Tuple[str, int, int, int]] = []
    col_members: List[str] = []

    for k in range(nz - 1):
        for j in range(ny):
            for i in range(nx):
                name = f"c_{i}_{j}_{k}"
                m.add_member(name, node_name(i, j, k), node_name(i, j, k + 1), "Steel", "Column")
                col_members.append(name)

    for k in range(1, nz):
        for j in range(ny):
            for i in range(nx - 1):
                name = f"bx_{i}_{j}_{k}"
                m.add_member(name, node_name(i, j, k), node_name(i + 1, j, k), "Steel", "Beam")
                beam_x_members.append((name, i, j, k))
        for i in range(nx):
            for j in range(ny - 1):
                name = f"by_{i}_{j}_{k}"
                m.add_member(name, node_name(i, j, k), node_name(i, j + 1, k), "Steel", "Beam")
                beam_y_members.append((name, i, j, k))

    case_g = "GRAVITY"
    for name, i, j, k in beam_x_members:
        w = w_x_beam(j)
        m.add_member_dist_load(name, "FZ", -w, -w, case=case_g)
    for name, i, j, k in beam_y_members:
        w = w_y_beam(i)
        m.add_member_dist_load(name, "FZ", -w, -w, case=case_g)

    case_w = "WIND_X"
    lat = float(lateral_fx_total_kn)
    if lat != 0.0:
        roof_k = nz - 1
        roof_nodes = [node_name(i, j, roof_k) for i in range(nx) for j in range(ny)]
        n_r = len(roof_nodes)
        fx_each = lat / n_r if n_r else 0.0
        for nid in roof_nodes:
            m.add_node_load(nid, "FX", fx_each, case=case_w)

    if lat != 0.0:
        m.add_load_combo("COMBO", {case_g: 1.0, case_w: 1.0})
        combo = "COMBO"
    else:
        m.add_load_combo("COMBO", {case_g: 1.0})
        combo = "COMBO"

    m.analyze(check_statics=check_statics)

    max_m_beam = 0.0
    max_v_beam = 0.0
    max_defl_mm = 0.0
    beam_rows: List[Dict[str, Any]] = []

    for name, i, j, k in beam_x_members + beam_y_members:
        mem = m.members[name]
        Mz = _mag(mem.max_moment("Mz", combo))
        My = _mag(mem.max_moment("My", combo))
        mm = max(My, Mz)
        Vy = _mag(mem.max_shear("Fy", combo))
        Vz = _mag(mem.max_shear("Fz", combo))
        vm = max(Vy, Vz)
        max_m_beam = max(max_m_beam, mm)
        max_v_beam = max(max_v_beam, vm)
        try:
            d_y = _mag(mem.max_deflection("dy", combo))
            d_z = _mag(mem.max_deflection("dz", combo))
            dloc = max(d_y, d_z) * 1000.0
        except Exception:
            dloc = 0.0
        max_defl_mm = max(max_defl_mm, dloc)
        beam_rows.append(
            {
                "id": name,
                "floor_z_m": ZS[k],
                "M_max_kNm": round(mm, 3),
                "V_max_kN": round(vm, 3),
                "deflection_mm": round(dloc, 3),
            }
        )

    beam_rows.sort(key=lambda r: -float(r["M_max_kNm"]))
    top_beams = beam_rows[:16]

    max_p_col = 0.0
    col_rows: List[Dict[str, Any]] = []
    for name in col_members:
        mem = m.members[name]
        p = _mag(mem.max_axial(combo))
        max_p_col = max(max_p_col, p)
        col_rows.append({"id": name, "P_max_kN": round(p, 3)})
    col_rows.sort(key=lambda r: -float(r["P_max_kN"]))
    top_cols = col_rows[:16]

    roof_k = nz - 1
    max_disp_z_mm = 0.0
    max_disp_xy_mm = 0.0
    for i in range(nx):
        for j in range(ny):
            n = m.nodes[node_name(i, j, roof_k)]
            try:
                dz = abs(float(n.DZ.get(combo, 0.0))) * 1000.0
                dx = abs(float(n.DX.get(combo, 0.0))) * 1000.0
                dy = abs(float(n.DY.get(combo, 0.0))) * 1000.0
                max_disp_z_mm = max(max_disp_z_mm, dz)
                max_disp_xy_mm = max(max_disp_xy_mm, (dx * dx + dy * dy) ** 0.5)
            except Exception:
                pass

    sum_fz_base = 0.0
    sum_fx_base = 0.0
    base_samples: List[Dict[str, Any]] = []
    for i in range(nx):
        for j in range(ny):
            nid = node_name(i, j, 0)
            n = m.nodes[nid]
            try:
                fz = float(n.RxnFZ.get(combo, 0.0))
                fx = float(n.RxnFX.get(combo, 0.0))
                sum_fz_base += fz
                sum_fx_base += fx
                if len(base_samples) < 4:
                    base_samples.append({"node": nid, "Rz_kN": round(fz, 2), "Rx_kN": round(fx, 2)})
            except Exception:
                pass

    total_grav_applied = Q * (nx - 1) * (ny - 1) * (nz - 1) * span_x_m * span_y_m
    statics_note = (
        f"Approx. total floor area loaded ≈ {Q} kPa × plan × ({nz - 1}) floors; "
        f"sum vertical reactions at base ≈ {sum_fz_base:.1f} kN (sign convention per PyNite)."
    )

    geometry_nodes = [
        {"id": node_name(i, j, k), "x": XS[i], "y": YS[j], "z": ZS[k]}
        for k in range(nz)
        for j in range(ny)
        for i in range(nx)
    ]
    geometry_members: List[Dict[str, str]] = []
    for k in range(nz - 1):
        for j in range(ny):
            for i in range(nx):
                geometry_members.append(
                    {
                        "id": f"c_{i}_{j}_{k}",
                        "start": node_name(i, j, k),
                        "end": node_name(i, j, k + 1),
                        "kind": "column",
                    }
                )
    for k in range(1, nz):
        for j in range(ny):
            for i in range(nx - 1):
                geometry_members.append(
                    {
                        "id": f"bx_{i}_{j}_{k}",
                        "start": node_name(i, j, k),
                        "end": node_name(i + 1, j, k),
                        "kind": "beam",
                    }
                )
        for i in range(nx):
            for j in range(ny - 1):
                geometry_members.append(
                    {
                        "id": f"by_{i}_{j}_{k}",
                        "start": node_name(i, j, k),
                        "end": node_name(i, j + 1, k),
                        "kind": "beam",
                    }
                )

    assumptions = [
        f"Regular {bx}×{by} bay grid, {span_x_m} m × {span_y_m} m spans; {st} storeys above grade.",
        f"Storey heights: first elevated {bottom_story_height_m} m, typical upper {story_height_m} m.",
        f"Steel E = {e_mpa / 1000.0:.0f} GPa (as entered); G from ν = {nu} unless overridden.",
        f"Beam section (rect.): {beam_width_m} m × {beam_depth_m} m; column (square): {column_width_m} m.",
        f"Floor equivalent pressure (DL+LL proxy): {Q} kPa on all elevated slabs; {two:.0%} of load to each beam system (two-way split).",
        "Supports: fixed base (6 DOF) at all base nodes.",
    ]
    if lat != 0.0:
        assumptions.append(
            f"Wind (+global X): total {lat} kN applied equally to all {nx*ny} roof nodes (pushover-style lateral push, not code wind pressure)."
        )

    narrative = (
        f"**PyNite 3D frame** — load combination `{combo}`.\n\n"
        f"- Max beam |M| ≈ **{max_m_beam:.2f} kN·m**; max beam |V| ≈ **{max_v_beam:.2f} kN**.\n"
        f"- Max column |P| ≈ **{max_p_col:.2f} kN**.\n"
        f"- Roof level: max |vertical disp.| ≈ **{max_disp_z_mm:.2f} mm**; "
        f"horizontal resultant displacement magnitude ≈ **{max_disp_xy_mm:.2f} mm** (at roof nodes).\n"
        f"- Sum base Rz ≈ **{sum_fz_base:.1f} kN**; sum base Rx ≈ **{sum_fx_base:.1f} kN**.\n\n"
        f"{statics_note}\n\n"
        "_Educational prototype — verify members and loads against your design code._"
    )

    result_cards = [
        {"label": "Max beam moment", "value": f"{max_m_beam:.2f}", "unit": "kN·m", "tone": "neutral"},
        {"label": "Max beam shear", "value": f"{max_v_beam:.2f}", "unit": "kN", "tone": "neutral"},
        {"label": "Max column axial", "value": f"{max_p_col:.2f}", "unit": "kN", "tone": "neutral"},
        {
            "label": "Max beam defl. (local)",
            "value": f"{max_defl_mm:.2f}",
            "unit": "mm",
            "tone": "warning" if max_defl_mm > (float(span_x_m) * 1000.0) / 360.0 else "good",
        },
        {"label": "Roof |DZ| max", "value": f"{max_disp_z_mm:.2f}", "unit": "mm", "tone": "neutral"},
        {"label": "Σ base vertical R", "value": f"{sum_fz_base:.1f}", "unit": "kN", "tone": "neutral"},
        {"label": "Σ base shear Rx", "value": f"{sum_fx_base:.1f}", "unit": "kN", "tone": "neutral"},
    ]

    return {
        "analysis_type": "building_3d",
        "engine": "PyNite",
        "pynite_path": pynite_root_label(),
        "load_combination": combo,
        "geometry": {"nodes": geometry_nodes, "members": geometry_members, "meta": {"source": "fea_parametric_frame"}},
        "result_cards": result_cards,
        "assumptions": assumptions,
        "summary_markdown": narrative,
        "beams": top_beams,
        "columns": top_cols,
        "base_reactions_sample": base_samples,
        "totals": {
            "max_beam_moment_kNm": round(max_m_beam, 4),
            "max_beam_shear_kN": round(max_v_beam, 4),
            "max_column_axial_kN": round(max_p_col, 4),
            "max_beam_deflection_mm": round(max_defl_mm, 4),
            "roof_max_DZ_mm": round(max_disp_z_mm, 4),
            "sum_base_Rz_kN": round(sum_fz_base, 4),
            "sum_base_Rx_kN": round(sum_fx_base, 4),
            "approx_total_floor_load_kN": round(total_grav_applied, 2),
        },
    }


def _cum_axis(spans: List[float]) -> List[float]:
    xs = [0.0]
    for s in spans:
        xs.append(xs[-1] + float(s))
    return xs


def run_irregular_frame_analysis(
    *,
    spans_x_m: List[float],
    spans_y_m: List[float],
    story_heights_m: List[float],
    dl_kpa: float,
    ll_kpa: float,
    slab_sw_kpa: float = 0.0,
    wind_pressure_kpa: float = 0.0,
    lateral_roof_fraction_of_gravity: float = 0.0,
    two_way_fraction: float = 0.5,
    material_steel: bool = False,
    beam_width_m: float = 0.40,
    beam_depth_m: float = 0.65,
    column_width_m: float = 0.50,
    run_p_delta: bool = True,
    check_statics: bool = False,
    sbc_kpa: float | None = None,
    retain_model: bool = False,
) -> Dict[str, Any]:
    """
    Irregular bay grid from span lists; DL + LL load cases; optional façade wind per storey;
    optional roof shear from fraction of estimated gravity (seismic placeholder); ULS = 1.2DL+1.6LL+lat.
    Optionally runs PyNite ``analyze_PDelta`` on ULS.

    If ``retain_model`` is True, the returned dict includes ``_femodel`` and ``_analysis_combo`` for
    desktop tools (not JSON-serializable). Omit when serving HTTP APIs.
    """
    if not pynite_available():
        raise RuntimeError(f"PyNite folder not found at {_PYNITE_ROOT}")

    from Pynite import FEModel3D  # noqa: WPS433

    sx = [float(x) for x in spans_x_m if float(x) > 0]
    sy = [float(y) for y in spans_y_m if float(y) > 0]
    if len(sx) < 1 or len(sy) < 1:
        raise ValueError("spans_x_m and spans_y_m must each have at least one positive span.")

    sh = [float(h) for h in story_heights_m if float(h) > 0]
    if not sh:
        raise ValueError("story_heights_m must have at least one positive height.")

    XS = _cum_axis(sx)
    YS = _cum_axis(sy)
    ZS = [0.0]
    for h in sh:
        ZS.append(ZS[-1] + h)

    nx, ny, nz = len(XS), len(YS), len(ZS)
    if nx * ny * nz > 12000:
        raise ValueError("Model too large for this service; reduce spans or storeys.")

    dl_eff = float(dl_kpa) + float(slab_sw_kpa)
    ll = float(ll_kpa)
    two = max(0.0, min(1.0, float(two_way_fraction)))

    def tributary_y(j: int) -> float:
        if j == 0:
            return (YS[1] - YS[0]) / 2
        if j == ny - 1:
            return (YS[j] - YS[j - 1]) / 2
        return (YS[j] - YS[j - 1]) / 2 + (YS[j + 1] - YS[j]) / 2

    def tributary_x(i: int) -> float:
        if i == 0:
            return (XS[1] - XS[0]) / 2
        if i == nx - 1:
            return (XS[i] - XS[i - 1]) / 2
        return (XS[i] - XS[i - 1]) / 2 + (XS[i + 1] - XS[i]) / 2

    def node_name(i: int, j: int, k: int) -> str:
        return f"n_{i}_{j}_{k}"

    if material_steel:
        e_mpa, nu, rho, mat_name = 200_000.0, 0.3, 77.0, "Steel"
    else:
        e_mpa, nu, rho, mat_name = 30_000.0, 0.2, 25.0, "Concrete"

    E_knm2 = e_mpa * 1000.0
    G_knm2 = E_knm2 / (2.0 * (1.0 + nu))

    m = FEModel3D()
    m.add_material(mat_name, E_knm2, G_knm2, nu, rho)

    b_b, h_b = float(beam_width_m), float(beam_depth_m)
    A_b = b_b * h_b
    Iy_b = h_b * b_b**3 / 12
    Iz_b = b_b * h_b**3 / 12
    J_b = (b_b * h_b**3 + h_b * b_b**3) / 12

    b_c = float(column_width_m)
    A_c = b_c**2
    Iy_c = b_c**4 / 12
    Iz_c = Iy_c
    J_c = 0.5 * Iy_c

    m.add_section("Beam", A_b, Iy_b, Iz_b, J_b)
    m.add_section("Column", A_c, Iy_c, Iz_c, J_c)

    for k in range(nz):
        for j in range(ny):
            for i in range(nx):
                m.add_node(node_name(i, j, k), XS[i], YS[j], ZS[k])

    for i in range(nx):
        for j in range(ny):
            m.def_support(node_name(i, j, 0), True, True, True, True, True, True)

    beam_x_members: List[Tuple[str, int, int, int]] = []
    beam_y_members: List[Tuple[str, int, int, int]] = []
    col_members: List[str] = []

    for k in range(nz - 1):
        for j in range(ny):
            for i in range(nx):
                name = f"c_{i}_{j}_{k}"
                m.add_member(name, node_name(i, j, k), node_name(i, j, k + 1), mat_name, "Column")
                col_members.append(name)

    for k in range(1, nz):
        for j in range(ny):
            for i in range(nx - 1):
                name = f"bx_{i}_{j}_{k}"
                m.add_member(name, node_name(i, j, k), node_name(i + 1, j, k), mat_name, "Beam")
                beam_x_members.append((name, i, j, k))
        for i in range(nx):
            for j in range(ny - 1):
                name = f"by_{i}_{j}_{k}"
                m.add_member(name, node_name(i, j, k), node_name(i, j + 1, k), mat_name, "Beam")
                beam_y_members.append((name, i, j, k))

    case_dl = "DL"
    case_ll = "LL"
    case_wind = "WIND"
    case_eq = "EQ_ROOF"

    for name, i, j, k in beam_x_members:
        wy = tributary_y(j)
        m.add_member_dist_load(name, "FZ", -two * dl_eff * wy, -two * dl_eff * wy, case=case_dl)
        m.add_member_dist_load(name, "FZ", -two * ll * wy, -two * ll * wy, case=case_ll)
    for name, i, j, k in beam_y_members:
        wx = tributary_x(i)
        m.add_member_dist_load(name, "FZ", -two * dl_eff * wx, -two * dl_eff * wx, case=case_dl)
        m.add_member_dist_load(name, "FZ", -two * ll * wx, -two * ll * wx, case=case_ll)

    wp = float(wind_pressure_kpa)
    if wp > 0.0:
        depth_m = YS[-1]
        for k in range(1, nz):
            hseg = ZS[k] - ZS[k - 1]
            f_kn = wp * depth_m * hseg
            nn = nx * ny
            f_each = f_kn / nn if nn else 0.0
            for i in range(nx):
                for j in range(ny):
                    m.add_node_load(node_name(i, j, k), "FX", f_each, case=case_wind)

    lat_frac = float(lateral_roof_fraction_of_gravity)
    plan_area = XS[-1] * YS[-1]
    grav_est = (dl_eff + ll) * plan_area * max(0, nz - 1)
    roof_k = nz - 1
    if lat_frac > 0.0 and roof_k >= 1:
        f_eq = lat_frac * grav_est
        nn = nx * ny
        fx_each = f_eq / nn if nn else 0.0
        for i in range(nx):
            for j in range(ny):
                m.add_node_load(node_name(i, j, roof_k), "FX", fx_each, case=case_eq)

    uls_parts: Dict[str, float] = {case_dl: 1.2, case_ll: 1.6}
    if wp > 0.0:
        uls_parts[case_wind] = 1.0
    if lat_frac > 0.0:
        uls_parts[case_eq] = 1.0
    m.add_load_combo("ULS", uls_parts)
    combo = "ULS"

    m.analyze(check_statics=check_statics)

    p_delta_note = "First-order linear analysis only (P-Δ not requested)."
    if run_p_delta:
        try:
            # NOTE: `combo_tags` in PyNite filters by a combo's tags, *not* its
            # name. Our ULS combo has no tags, so passing `combo_tags=["ULS"]`
            # would make `_identify_combos` return an empty list. P-Δ would
            # then run zero iterations while `_prepare_model` inside it has
            # already cleared every node's displacement dict, zeroing all
            # member forces and deflections. We want P-Δ on the only combo
            # we built, so we simply don't pass `combo_tags`.
            m.analyze_PDelta(log=False, check_stability=True, max_iter=30, sparse=True)
            p_delta_note = "P-Δ (second-order) analysis completed for ULS in PyNite."
        except Exception as ex:
            # Fall back to the already-computed first-order results. Re-run
            # the first-order solver so displacements (which P-Δ's prepare
            # step may have cleared) are restored.
            try:
                m.analyze(check_statics=False)
            except Exception:
                pass
            p_delta_note = f"P-Δ not applied: {str(ex)[:200]}. Showing first-order ULS member forces."

    def hor_mm(nid: str) -> float:
        n = m.nodes[nid]
        try:
            dx = float(n.DX.get(combo, 0.0))
            dy = float(n.DY.get(combo, 0.0))
            return (dx * dx + dy * dy) ** 0.5 * 1000.0
        except Exception:
            return 0.0

    storey_drifts: List[Dict[str, Any]] = []
    for k in range(1, nz):
        h_m = ZS[k] - ZS[k - 1]
        worst_mm = 0.0
        worst_ratio = 0.0
        for i in range(nx):
            for j in range(ny):
                d_rel = abs(hor_mm(node_name(i, j, k)) - hor_mm(node_name(i, j, k - 1)))
                ratio = d_rel / (h_m * 1000.0) if h_m > 0 else 0.0
                if d_rel > worst_mm:
                    worst_mm = d_rel
                    worst_ratio = ratio
        storey_drifts.append(
            {
                "storey_index": k,
                "z_top_m": round(ZS[k], 4),
                "height_m": round(h_m, 4),
                "max_drift_mm": round(worst_mm, 4),
                "drift_ratio_h": round(worst_ratio, 6),
            }
        )

    max_m_beam = max_v_beam = max_defl_mm = 0.0
    beam_rows: List[Dict[str, Any]] = []
    for name, i, j, k in beam_x_members + beam_y_members:
        mem = m.members[name]
        Mz = _safe_member_scalar(lambda: mem.max_moment("Mz", combo))
        My = _safe_member_scalar(lambda: mem.max_moment("My", combo))
        mm = max(My, Mz)
        Vy = _safe_member_scalar(lambda: mem.max_shear("Fy", combo))
        Vz = _safe_member_scalar(lambda: mem.max_shear("Fz", combo))
        vm = max(Vy, Vz)
        max_m_beam = max(max_m_beam, mm)
        max_v_beam = max(max_v_beam, vm)
        d_y = _safe_member_scalar(lambda: mem.max_deflection("dy", combo))
        d_z = _safe_member_scalar(lambda: mem.max_deflection("dz", combo))
        dloc = max(d_y, d_z) * 1000.0
        max_defl_mm = max(max_defl_mm, dloc)
        beam_rows.append(
            {
                "id": name,
                "floor_z_m": ZS[k],
                "M_max_kNm": round(mm, 3),
                "V_max_kN": round(vm, 3),
                "deflection_mm": round(dloc, 3),
            }
        )
    beam_rows.sort(key=lambda r: -float(r["M_max_kNm"]))
    top_beams = beam_rows[:40]

    max_p_col = 0.0
    col_rows: List[Dict[str, Any]] = []
    for name in col_members:
        mem = m.members[name]
        p = _safe_member_scalar(lambda: mem.max_axial(combo))
        My = _safe_member_scalar(lambda: mem.max_moment("My", combo))
        Mz = _safe_member_scalar(lambda: mem.max_moment("Mz", combo))
        T = _safe_member_scalar(lambda: mem.max_torque(combo))
        max_p_col = max(max_p_col, p)
        col_rows.append(
            {
                "id": name,
                "P_max_kN": round(p, 3),
                "My_max_kNm": round(My, 3),
                "Mz_max_kNm": round(Mz, 3),
                "T_max_kNm": round(T, 3),
            }
        )
    col_rows.sort(key=lambda r: -float(r["P_max_kN"]))
    top_cols = col_rows[:40]

    max_disp_z_mm = max_disp_xy_mm = 0.0
    for i in range(nx):
        for j in range(ny):
            n = m.nodes[node_name(i, j, roof_k)]
            try:
                dz = abs(float(n.DZ.get(combo, 0.0))) * 1000.0
                dx = float(n.DX.get(combo, 0.0))
                dy = float(n.DY.get(combo, 0.0))
                max_disp_z_mm = max(max_disp_z_mm, dz)
                max_disp_xy_mm = max(max_disp_xy_mm, (dx * dx + dy * dy) ** 0.5 * 1000.0)
            except Exception:
                pass

    base_reactions: List[Dict[str, Any]] = []
    sum_fz = sum_fx = sum_fy = 0.0
    for i in range(nx):
        for j in range(ny):
            nid = node_name(i, j, 0)
            n = m.nodes[nid]
            try:
                fx = float(n.RxnFX.get(combo, 0.0))
                fy = float(n.RxnFY.get(combo, 0.0))
                fz = float(n.RxnFZ.get(combo, 0.0))
                mx = float(n.RxnMX.get(combo, 0.0))
                my = float(n.RxnMY.get(combo, 0.0))
                mz = float(n.RxnMZ.get(combo, 0.0))
                sum_fx += fx
                sum_fy += fy
                sum_fz += fz
                base_reactions.append(
                    {
                        "node": nid,
                        "x_m": round(XS[i], 3),
                        "y_m": round(YS[j], 3),
                        "Rx_kN": round(fx, 2),
                        "Ry_kN": round(fy, 2),
                        "Rz_kN": round(fz, 2),
                        "Mx_kNm": round(mx, 2),
                        "My_kNm": round(my, 2),
                        "Mz_kNm": round(mz, 2),
                    }
                )
            except Exception:
                pass

    max_bearing_kpa = None
    if sbc_kpa is not None and plan_area > 0 and nx * ny > 0:
        max_vert = max(abs(r["Rz_kN"]) for r in base_reactions) if base_reactions else 0.0
        foot_a = column_width_m**2
        if foot_a > 0:
            max_bearing_kpa = max_vert / foot_a

    geometry_nodes = [
        {"id": node_name(i, j, k), "x": XS[i], "y": YS[j], "z": ZS[k]}
        for k in range(nz)
        for j in range(ny)
        for i in range(nx)
    ]
    geometry_members: List[Dict[str, str]] = []
    for k in range(nz - 1):
        for j in range(ny):
            for i in range(nx):
                geometry_members.append(
                    {
                        "id": f"c_{i}_{j}_{k}",
                        "start": node_name(i, j, k),
                        "end": node_name(i, j, k + 1),
                        "kind": "column",
                    }
                )
    for k in range(1, nz):
        for j in range(ny):
            for i in range(nx - 1):
                geometry_members.append(
                    {
                        "id": f"bx_{i}_{j}_{k}",
                        "start": node_name(i, j, k),
                        "end": node_name(i + 1, j, k),
                        "kind": "beam",
                    }
                )
        for i in range(nx):
            for j in range(ny - 1):
                geometry_members.append(
                    {
                        "id": f"by_{i}_{j}_{k}",
                        "start": node_name(i, j, k),
                        "end": node_name(i, j + 1, k),
                        "kind": "beam",
                    }
                )

    span_x_max = max(sx) if sx else 0.0
    assumptions = [
        f"Irregular grid: X spans {sx} m ({len(sx)} bays), Y spans {sy} m ({len(sy)} bays).",
        f"Vertical: {len(sh)} storeys, level z = {', '.join(f'{z:.2f}' for z in ZS)} m.",
        f"Loads: DL {dl_kpa:.2f} kPa + slab SW {slab_sw_kpa:.2f} kPa on beams; LL {ll_kpa:.2f} kPa; "
        f"two-way split {two:.0%} to each beam system.",
        f"ULS = 1.2·DL + 1.6·LL"
        + (" + 1.0·WIND" if wp > 0 else "")
        + (" + 1.0·EQ_roof" if lat_frac > 0 else "")
        + ".",
        f"Material: {mat_name}, E ≈ {e_mpa/1000:.0f} GPa; beam {beam_width_m}×{beam_depth_m} m; column {column_width_m} m square.",
        p_delta_note,
        "Fixed base at all base nodes; verify all assumptions against your building code.",
    ]
    if sbc_kpa is not None:
        assumptions.append(f"Specified SBC {sbc_kpa} kPa — compare to estimated max footing pressure ≈ {max_bearing_kpa:.1f} kPa (column area only, rough).")

    narrative = (
        f"**PyNite 3D frame (irregular grid)** — **{combo}**.\n\n"
        f"- **Beams:** max |M| ≈ **{max_m_beam:.2f} kN·m**; max |V| ≈ **{max_v_beam:.2f} kN**; "
        f"max local deflection ≈ **{max_defl_mm:.2f} mm**.\n"
        f"- **Columns:** max |P| ≈ **{max_p_col:.2f} kN** (see table for My, Mz, T).\n"
        f"- **Roof:** max horizontal disp. ≈ **{max_disp_xy_mm:.2f} mm**; max vertical ≈ **{max_disp_z_mm:.2f} mm**.\n"
        f"- **Base:** ΣRz ≈ **{sum_fz:.1f} kN**, ΣRx ≈ **{sum_fx:.1f} kN**, ΣRy ≈ **{sum_fy:.1f} kN**.\n"
        f"- **P-Δ:** {p_delta_note}\n"
    )

    result_cards = [
        {"label": "Max beam |M|", "value": f"{max_m_beam:.2f}", "unit": "kN·m", "tone": "neutral"},
        {"label": "Max beam |V|", "value": f"{max_v_beam:.2f}", "unit": "kN", "tone": "neutral"},
        {"label": "Max beam defl.", "value": f"{max_defl_mm:.2f}", "unit": "mm", "tone": "neutral"},
        {"label": "Max column |P|", "value": f"{max_p_col:.2f}", "unit": "kN", "tone": "neutral"},
        {
            "label": "Max storey drift",
            "value": f"{max((float(s['max_drift_mm']) for s in storey_drifts), default=0.0):.2f}",
            "unit": "mm",
            "tone": "neutral",
        },
        {"label": "Σ base Rz", "value": f"{sum_fz:.1f}", "unit": "kN", "tone": "neutral"},
    ]

    out: Dict[str, Any] = {
        "analysis_type": "building_3d",
        "engine": "PyNite",
        "pynite_path": pynite_root_label(),
        "load_combination": combo,
        "geometry": {"nodes": geometry_nodes, "members": geometry_members, "meta": {"source": "fea_irregular_frame"}},
        "result_cards": result_cards,
        "assumptions": assumptions,
        "summary_markdown": narrative,
        "beams": top_beams,
        "columns": top_cols,
        "base_reactions": base_reactions,
        "storey_drifts": storey_drifts,
        "p_delta_note": p_delta_note,
        "totals": {
            "max_beam_moment_kNm": round(max_m_beam, 4),
            "max_beam_shear_kN": round(max_v_beam, 4),
            "max_column_axial_kN": round(max_p_col, 4),
            "max_beam_deflection_mm": round(max_defl_mm, 4),
            "roof_max_DZ_mm": round(max_disp_z_mm, 4),
            "roof_horizontal_mm": round(max_disp_xy_mm, 4),
            "sum_base_Rz_kN": round(sum_fz, 4),
            "sum_base_Rx_kN": round(sum_fx, 4),
            "sum_base_Ry_kN": round(sum_fy, 4),
            "estimated_gravity_kN": round(grav_est, 2),
            "max_bearing_on_column_footing_kPa": round(max_bearing_kpa, 3) if max_bearing_kpa is not None else None,
        },
    }
    if retain_model:
        out["_femodel"] = m
        out["_analysis_combo"] = combo
    return out


def _sample_series(fn, L: float, n: int) -> List[List[float]]:
    """Return [x_list_m, y_list] by sampling ``fn(x)`` at ``n`` evenly-spaced stations."""
    if n < 2:
        n = 2
    xs: List[float] = []
    ys: List[float] = []
    for i in range(n):
        x = L * i / (n - 1)
        try:
            y = float(fn(x))
        except Exception:
            y = 0.0
        xs.append(round(x, 4))
        ys.append(round(y, 4))
    return [xs, ys]


def _beam_section_rect(b_m: float, h_m: float) -> Tuple[float, float, float, float]:
    A = b_m * h_m
    Iz = b_m * h_m**3 / 12.0
    Iy = h_m * b_m**3 / 12.0
    J = (b_m * h_m**3 + h_m * b_m**3) / 12.0
    return A, Iy, Iz, J


# ---------------------------------------------------------------------------
# 2D beam analysis — single span or multi-span continuous
# ---------------------------------------------------------------------------


def run_beam_analysis(
    *,
    span_m: float,
    support_left: str = "pin",
    support_right: str = "roller",
    cantilever_left_m: float = 0.0,
    cantilever_right_m: float = 0.0,
    udl_kN_per_m: float = 0.0,
    dl_kN_per_m: float = 0.0,
    ll_kN_per_m: float = 0.0,
    point_loads: Optional[List[Dict[str, float]]] = None,
    material: str = "steel",
    beam_width_m: float = 0.30,
    beam_depth_m: float = 0.60,
    n_samples: int = 41,
) -> Dict[str, Any]:
    """
    2D prismatic beam FEA via PyNite (bends in X–Y plane).

    * ``support_left``/``support_right``: ``pin`` | ``roller`` | ``fixed`` | ``free``.
    * ``cantilever_left_m`` / ``cantilever_right_m``: optional overhangs beyond the supports.
    * Loads default to kN and kN/m. If ``udl_kN_per_m`` is given it is treated as total DL+LL
      (useful when the prompt only provides a single magnitude); otherwise DL/LL are summed.
    * Output: envelopes, diagrams (shear/moment/deflection arrays), reactions, ULS combo = 1.2DL+1.6LL.
    """
    if not pynite_available():
        raise RuntimeError("PyNite package not found on disk.")
    from Pynite import FEModel3D  # noqa: WPS433

    L = float(span_m)
    if L <= 0:
        raise ValueError("span_m must be positive")
    cL = max(0.0, float(cantilever_left_m))
    cR = max(0.0, float(cantilever_right_m))

    if material.lower().startswith("steel"):
        e_mpa, nu, rho, mat_name = 200_000.0, 0.3, 77.0, "Steel"
    else:
        e_mpa, nu, rho, mat_name = 30_000.0, 0.2, 25.0, "Concrete"
    E_knm2 = e_mpa * 1000.0
    G_knm2 = E_knm2 / (2.0 * (1.0 + nu))

    A, Iy, Iz, J = _beam_section_rect(beam_width_m, beam_depth_m)

    m = FEModel3D()
    m.add_material(mat_name, E_knm2, G_knm2, nu, rho)
    m.add_section("Beam", A, Iy, Iz, J)

    # Axis convention: the beam runs along global X; gravity acts along −Y.
    nodes_xy: List[Tuple[str, float]] = []
    if cL > 0:
        nodes_xy.append(("A", -cL))
    nodes_xy.append(("SL", 0.0))
    nodes_xy.append(("SR", L))
    if cR > 0:
        nodes_xy.append(("B", L + cR))
    nodes_xy.sort(key=lambda kv: kv[1])
    for name, xv in nodes_xy:
        m.add_node(name, float(xv), 0.0, 0.0)

    # One continuous beam member per pair of adjacent nodes.
    members: List[Tuple[str, str, str, float, float]] = []
    for idx in range(len(nodes_xy) - 1):
        a, xa = nodes_xy[idx]
        b, xb = nodes_xy[idx + 1]
        name = f"M{idx + 1}"
        m.add_member(name, a, b, mat_name, "Beam")
        members.append((name, a, b, xa, xb))

    # 2D-beam constraint: at every node, restrain DZ (out-of-plane), RX (torsion),
    # RY (out-of-plane bending) so the 3D PyNite model behaves as a planar beam.
    for name, _xv in nodes_xy:
        m.def_support(name, False, False, True, True, True, False)

    def _def_support(node: str, kind: str) -> None:
        k = kind.lower()
        if k == "pin":
            # DX, DY additionally restrained (DZ/RX/RY already fixed above)
            m.def_support(node, True, True, True, True, True, False)
        elif k == "roller":
            m.def_support(node, False, True, True, True, True, False)
        elif k == "fixed":
            m.def_support(node, True, True, True, True, True, True)
        elif k == "free":
            return
        else:
            raise ValueError(f"unknown support kind '{kind}'")

    _def_support("SL", support_left)
    _def_support("SR", support_right)

    if support_left.lower() == "free" and support_right.lower() == "free":
        raise ValueError("At least one of support_left / support_right must restrain the beam.")
    # Make sure at least one node restrains DX for axial stability.
    if support_left.lower() not in ("pin", "fixed") and support_right.lower() not in ("pin", "fixed"):
        m.def_support("SL", True, True, True, True, True, False)

    # Loads — distribute UDL over all members in their local length
    case_dl = "DL"
    case_ll = "LL"
    if udl_kN_per_m and (dl_kN_per_m == 0.0 and ll_kN_per_m == 0.0):
        dl_w = float(udl_kN_per_m)
        ll_w = 0.0
    else:
        dl_w = float(dl_kN_per_m)
        ll_w = float(ll_kN_per_m)

    for name, a, b, xa, xb in members:
        if dl_w:
            m.add_member_dist_load(name, "Fy", -dl_w, -dl_w, case=case_dl)
        if ll_w:
            m.add_member_dist_load(name, "Fy", -ll_w, -ll_w, case=case_ll)

    for pl in point_loads or []:
        mag = float(pl.get("P_kN", 0.0))
        loc = float(pl.get("x_m", L / 2.0))
        case = str(pl.get("case", "LL")).upper() or "LL"
        # find member hosting this global x
        for name, a, b, xa, xb in members:
            if xa - 1e-9 <= loc <= xb + 1e-9:
                local_x = max(0.0, loc - xa)
                m.add_member_pt_load(name, "Fy", -abs(mag), local_x, case=case)
                break

    m.add_load_combo("SLS", {case_dl: 1.0, case_ll: 1.0})
    m.add_load_combo("ULS", {case_dl: 1.2, case_ll: 1.6})
    combo = "ULS"
    m.analyze(check_statics=False)

    # Build global-x series for shear / moment / deflection across all members
    shear_xy: List[List[float]] = [[], []]
    moment_xy: List[List[float]] = [[], []]
    defl_xy: List[List[float]] = [[], []]
    for name, a, b, xa, xb in members:
        mem = m.members[name]
        Lm = xb - xa
        per_mem = max(8, int(n_samples * Lm / max(1e-6, (L + cL + cR))))
        for i in range(per_mem + 1):
            lx = Lm * i / per_mem
            gx = xa + lx
            try:
                s = float(mem.shear("Fy", lx, combo))
            except Exception:
                s = 0.0
            try:
                mm = float(mem.moment("Mz", lx, combo))
            except Exception:
                mm = 0.0
            try:
                d = float(mem.deflection("dy", lx, combo)) * 1000.0
            except Exception:
                d = 0.0
            shear_xy[0].append(round(gx, 4))
            shear_xy[1].append(round(s, 4))
            moment_xy[0].append(round(gx, 4))
            moment_xy[1].append(round(mm, 4))
            defl_xy[0].append(round(gx, 4))
            defl_xy[1].append(round(d, 4))

    V_pos = max(shear_xy[1]) if shear_xy[1] else 0.0
    V_neg = min(shear_xy[1]) if shear_xy[1] else 0.0
    M_pos = max(moment_xy[1]) if moment_xy[1] else 0.0
    M_neg = min(moment_xy[1]) if moment_xy[1] else 0.0
    d_peak_mm = max((abs(v) for v in defl_xy[1]), default=0.0)
    Vmax = max(abs(V_pos), abs(V_neg))
    Mmax = max(abs(M_pos), abs(M_neg))

    reactions: List[Dict[str, Any]] = []
    for nid in ("SL", "SR"):
        if nid not in m.nodes:
            continue
        n = m.nodes[nid]
        try:
            rx = float(n.RxnFX.get(combo, 0.0))
            ry = float(n.RxnFY.get(combo, 0.0))
            mz = float(n.RxnMZ.get(combo, 0.0))
        except Exception:
            rx = ry = mz = 0.0
        reactions.append(
            {
                "node": nid,
                "x_m": float(m.nodes[nid].X),
                "Rx_kN": round(rx, 3),
                "Ry_kN": round(ry, 3),
                "Mz_kNm": round(mz, 3),
            }
        )

    geometry_nodes = [
        {"id": name, "x": float(m.nodes[name].X), "y": 0.0, "z": 0.0}
        for name, _x in nodes_xy
    ]
    geometry_members = [
        {"id": name, "start": a, "end": b, "kind": "beam"}
        for name, a, b, _xa, _xb in members
    ]

    support_summary = f"{support_left.title()} – {support_right.title()}"
    if cL or cR:
        support_summary += f" · overhangs L={cL}m, R={cR}m"

    mat_label = mat_name
    defl_limit_mm = (L * 1000.0) / 360.0
    tone_defl = "warning" if d_peak_mm > defl_limit_mm else "good"

    w_total = (1.2 * dl_w + 1.6 * ll_w)
    simply_supported_check_M = w_total * L * L / 8.0 if dl_w or ll_w else None
    simply_supported_check_V = w_total * L / 2.0 if dl_w or ll_w else None

    result_cards = [
        {"label": "Max |M|", "value": f"{Mmax:.2f}", "unit": "kN·m", "tone": "neutral"},
        {"label": "Max |V|", "value": f"{Vmax:.2f}", "unit": "kN", "tone": "neutral"},
        {"label": "Max deflection", "value": f"{d_peak_mm:.2f}", "unit": "mm", "tone": tone_defl},
        {"label": "L/360 limit", "value": f"{defl_limit_mm:.2f}", "unit": "mm", "tone": "neutral"},
        {"label": "Reaction SL (Ry)", "value": f"{reactions[0]['Ry_kN'] if reactions else 0:.2f}", "unit": "kN", "tone": "neutral"},
        {"label": "Reaction SR (Ry)", "value": f"{reactions[-1]['Ry_kN'] if reactions else 0:.2f}", "unit": "kN", "tone": "neutral"},
    ]

    assumptions = [
        f"Single-span beam L = {L} m, {support_summary}.",
        f"Section: rectangular {beam_width_m} m × {beam_depth_m} m ({mat_label}, E ≈ {e_mpa/1000:.0f} GPa).",
        f"Loads: DL = {dl_w} kN/m, LL = {ll_w} kN/m (+ {len(point_loads or [])} point loads).",
        "Load combinations built: SLS = 1.0DL + 1.0LL, ULS = 1.2DL + 1.6LL (analysis uses ULS).",
        "Results come from PyNite (Euler–Bernoulli beam element, integrated open-source FEM).",
    ]

    check_line = ""
    if simply_supported_check_M is not None and cL == 0 and cR == 0:
        check_line = (
            f"\n\n_Sanity check (simply-supported ULS):_ w·L²/8 ≈ **{simply_supported_check_M:.2f} kN·m**, "
            f"w·L/2 ≈ **{simply_supported_check_V:.2f} kN** — compare against FEA above."
        )

    narrative = (
        f"**PyNite 2D beam** — ULS = 1.2DL + 1.6LL.\n\n"
        f"- Envelope |M|: **{Mmax:.2f} kN·m** (M⁺ {M_pos:.2f}, M⁻ {M_neg:.2f}).\n"
        f"- Envelope |V|: **{Vmax:.2f} kN** (V⁺ {V_pos:.2f}, V⁻ {V_neg:.2f}).\n"
        f"- Max vertical deflection: **{d_peak_mm:.2f} mm** vs L/360 = {defl_limit_mm:.2f} mm.\n"
        f"- Reactions: SL Ry = {reactions[0]['Ry_kN']:.2f} kN, SR Ry = {reactions[-1]['Ry_kN']:.2f} kN."
        f"{check_line}"
    )

    return {
        "analysis_type": "beam_2d",
        "engine": "PyNite",
        "pynite_path": pynite_root_label(),
        "load_combination": combo,
        "geometry": {
            "nodes": geometry_nodes,
            "members": geometry_members,
            "meta": {"source": "fea_beam_2d"},
        },
        "result_cards": result_cards,
        "assumptions": assumptions,
        "summary_markdown": narrative,
        "beams": [
            {
                "id": name,
                "floor_z_m": 0.0,
                "M_max_kNm": round(Mmax, 3),
                "V_max_kN": round(Vmax, 3),
                "deflection_mm": round(d_peak_mm, 3),
            }
            for name, _a, _b, _xa, _xb in members
        ],
        "columns": [],
        "base_reactions": reactions,
        "storey_drifts": [],
        "p_delta_note": "Linear first-order analysis (P-Δ not applicable for isolated beams).",
        "totals": {
            "max_beam_moment_kNm": round(Mmax, 4),
            "max_beam_shear_kN": round(Vmax, 4),
            "max_beam_deflection_mm": round(d_peak_mm, 4),
            "deflection_limit_L_over_360_mm": round(defl_limit_mm, 4),
            "sum_base_Rz_kN": round(sum(r["Ry_kN"] for r in reactions), 4),
        },
        "diagrams": {
            "shear_kN": shear_xy,
            "moment_kNm": moment_xy,
            "deflection_mm": defl_xy,
            "x_label_m": "x along beam (m)",
        },
    }


# ---------------------------------------------------------------------------
# 2D frame analysis — single-plane multi-bay, multi-storey moment frame
# ---------------------------------------------------------------------------


def run_frame_2d_analysis(
    *,
    spans_m: List[float],
    story_heights_m: List[float],
    dl_kN_per_m: float = 15.0,
    ll_kN_per_m: float = 6.0,
    lateral_fx_per_floor_kN: float = 0.0,
    material: str = "concrete",
    beam_width_m: float = 0.30,
    beam_depth_m: float = 0.60,
    column_width_m: float = 0.45,
    run_p_delta: bool = False,
    n_diag_samples: int = 17,
) -> Dict[str, Any]:
    """
    Planar moment frame in PyNite (lies in X–Y plane; Z is fully restrained).

    * ``spans_m``     – list of bay lengths along X (m)
    * ``story_heights_m`` – list of storey heights along Y (m)
    * Gravity UDL on every beam. ``lateral_fx_per_floor_kN`` applies a nodal +X load
      at the left-most column of every storey to produce drift.
    """
    if not pynite_available():
        raise RuntimeError("PyNite package not found on disk.")
    from Pynite import FEModel3D  # noqa: WPS433

    sx = [float(s) for s in spans_m if float(s) > 0]
    sh = [float(h) for h in story_heights_m if float(h) > 0]
    if not sx or not sh:
        raise ValueError("frame_2d requires at least one span and one storey height")

    XS = _cum_axis(sx)
    YS = [0.0]
    for h in sh:
        YS.append(YS[-1] + h)

    nx, ny = len(XS), len(YS)

    if material.lower().startswith("steel"):
        e_mpa, nu, rho, mat_name = 200_000.0, 0.3, 77.0, "Steel"
    else:
        e_mpa, nu, rho, mat_name = 30_000.0, 0.2, 25.0, "Concrete"
    E_knm2 = e_mpa * 1000.0
    G_knm2 = E_knm2 / (2.0 * (1.0 + nu))

    Ab, Iyb, Izb, Jb = _beam_section_rect(beam_width_m, beam_depth_m)
    bc = float(column_width_m)
    Ac, Iyc, Izc, Jc = _beam_section_rect(bc, bc)

    m = FEModel3D()
    m.add_material(mat_name, E_knm2, G_knm2, nu, rho)
    m.add_section("Beam", Ab, Iyb, Izb, Jb)
    m.add_section("Column", Ac, Iyc, Izc, Jc)

    def node(i: int, k: int) -> str:
        return f"N_{i}_{k}"

    for k in range(ny):
        for i in range(nx):
            m.add_node(node(i, k), XS[i], YS[k], 0.0)

    # Pin out-of-plane (Z + RX + RY) translation/rotation at every node to stay planar.
    for k in range(ny):
        for i in range(nx):
            nid = node(i, k)
            if k == 0:
                m.def_support(nid, True, True, True, True, True, True)
            else:
                m.def_support(nid, False, False, True, True, True, False)

    col_members: List[str] = []
    beam_members: List[Tuple[str, int, int]] = []

    for k in range(ny - 1):
        for i in range(nx):
            name = f"C_{i}_{k}"
            m.add_member(name, node(i, k), node(i, k + 1), mat_name, "Column")
            col_members.append(name)

    for k in range(1, ny):
        for i in range(nx - 1):
            name = f"B_{i}_{k}"
            m.add_member(name, node(i, k), node(i + 1, k), mat_name, "Beam")
            beam_members.append((name, i, k))

    case_dl, case_ll, case_lat = "DL", "LL", "LAT"
    for name, _i, _k in beam_members:
        m.add_member_dist_load(name, "Fy", -float(dl_kN_per_m), -float(dl_kN_per_m), case=case_dl)
        m.add_member_dist_load(name, "Fy", -float(ll_kN_per_m), -float(ll_kN_per_m), case=case_ll)

    if lateral_fx_per_floor_kN:
        for k in range(1, ny):
            m.add_node_load(node(0, k), "FX", float(lateral_fx_per_floor_kN), case=case_lat)

    uls = {case_dl: 1.2, case_ll: 1.6}
    if lateral_fx_per_floor_kN:
        uls[case_lat] = 1.0
    m.add_load_combo("ULS", uls)
    combo = "ULS"

    m.analyze(check_statics=False)
    p_delta_note = "First-order linear analysis only."
    if run_p_delta:
        try:
            # See note in run_irregular_frame_analysis: passing combo_tags=[name]
            # is a bug in PyNite's contract — tags are distinct from combo names.
            m.analyze_PDelta(log=False, check_stability=True, max_iter=30, sparse=True)
            p_delta_note = "P-Δ (second-order) analysis completed on ULS."
        except Exception as ex:
            try:
                m.analyze(check_statics=False)
            except Exception:
                pass
            p_delta_note = f"P-Δ skipped: {str(ex)[:160]}"

    max_M = max_V = max_def = 0.0
    beams_out: List[Dict[str, Any]] = []
    moment_diag: Dict[str, List[List[float]]] = {}
    shear_diag: Dict[str, List[List[float]]] = {}
    for name, _i, k in beam_members:
        mem = m.members[name]
        Mmax = _safe_member_scalar(lambda: mem.max_moment("Mz", combo))
        Vmax = _safe_member_scalar(lambda: mem.max_shear("Fy", combo))
        try:
            dmax = abs(float(mem.max_deflection("dy", combo))) * 1000.0
        except Exception:
            dmax = 0.0
        max_M = max(max_M, Mmax)
        max_V = max(max_V, Vmax)
        max_def = max(max_def, dmax)
        beams_out.append(
            {
                "id": name,
                "floor_z_m": YS[k],
                "M_max_kNm": round(Mmax, 3),
                "V_max_kN": round(Vmax, 3),
                "deflection_mm": round(dmax, 3),
            }
        )
        # One representative beam per storey gets a diagram (first beam of each level)
        if (_i == 0) and (f"level_{k}" not in moment_diag):
            L = mem.L()
            npts = max(9, n_diag_samples)
            xs = [L * i / (npts - 1) for i in range(npts)]
            moment_diag[f"level_{k}"] = [
                [round(x, 3) for x in xs],
                [round(float(mem.moment("Mz", x, combo)), 3) for x in xs],
            ]
            shear_diag[f"level_{k}"] = [
                [round(x, 3) for x in xs],
                [round(float(mem.shear("Fy", x, combo)), 3) for x in xs],
            ]

    beams_out.sort(key=lambda r: -r["M_max_kNm"])

    max_P = 0.0
    cols_out: List[Dict[str, Any]] = []
    for name in col_members:
        mem = m.members[name]
        P = _safe_member_scalar(lambda: mem.max_axial(combo))
        Mz = _safe_member_scalar(lambda: mem.max_moment("Mz", combo))
        max_P = max(max_P, P)
        cols_out.append(
            {
                "id": name,
                "P_max_kN": round(P, 3),
                "My_max_kNm": 0.0,
                "Mz_max_kNm": round(Mz, 3),
                "T_max_kNm": 0.0,
            }
        )
    cols_out.sort(key=lambda r: -r["P_max_kN"])

    reactions: List[Dict[str, Any]] = []
    sum_fx = sum_fy = 0.0
    for i in range(nx):
        nid = node(i, 0)
        n = m.nodes[nid]
        try:
            rx = float(n.RxnFX.get(combo, 0.0))
            ry = float(n.RxnFY.get(combo, 0.0))
            mz = float(n.RxnMZ.get(combo, 0.0))
        except Exception:
            rx = ry = mz = 0.0
        sum_fx += rx
        sum_fy += ry
        reactions.append(
            {
                "node": nid,
                "x_m": round(XS[i], 3),
                "y_m": 0.0,
                "Rx_kN": round(rx, 2),
                "Ry_kN": round(ry, 2),
                "Rz_kN": round(ry, 2),  # alias so UI table reads cleanly
                "Mx_kNm": 0.0,
                "My_kNm": 0.0,
                "Mz_kNm": round(mz, 2),
            }
        )

    storey_drifts: List[Dict[str, Any]] = []
    for k in range(1, ny):
        h_m = YS[k] - YS[k - 1]
        worst = 0.0
        for i in range(nx):
            try:
                d_top = float(m.nodes[node(i, k)].DX.get(combo, 0.0)) * 1000.0
                d_bot = float(m.nodes[node(i, k - 1)].DX.get(combo, 0.0)) * 1000.0
                rel = abs(d_top - d_bot)
                if rel > worst:
                    worst = rel
            except Exception:
                continue
        storey_drifts.append(
            {
                "storey_index": k,
                "z_top_m": round(YS[k], 3),
                "height_m": round(h_m, 3),
                "max_drift_mm": round(worst, 3),
                "drift_ratio_h": round(worst / (h_m * 1000.0), 6) if h_m else 0.0,
            }
        )

    geometry_nodes = [
        {"id": node(i, k), "x": XS[i], "y": 0.0, "z": YS[k]}
        for k in range(ny)
        for i in range(nx)
    ]
    geometry_members: List[Dict[str, str]] = []
    for k in range(ny - 1):
        for i in range(nx):
            geometry_members.append(
                {"id": f"C_{i}_{k}", "start": node(i, k), "end": node(i, k + 1), "kind": "column"}
            )
    for k in range(1, ny):
        for i in range(nx - 1):
            geometry_members.append(
                {"id": f"B_{i}_{k}", "start": node(i, k), "end": node(i + 1, k), "kind": "beam"}
            )

    assumptions = [
        f"2D moment frame: {len(sx)} bays (spans {sx} m), {len(sh)} storeys (heights {sh} m).",
        f"Material: {mat_name}, E ≈ {e_mpa/1000:.0f} GPa; beam {beam_width_m}×{beam_depth_m} m; column {column_width_m} m square.",
        f"Loads: DL = {dl_kN_per_m} kN/m, LL = {ll_kN_per_m} kN/m on all beams; lateral Fx per floor = {lateral_fx_per_floor_kN} kN.",
        "Out-of-plane DOFs restrained at every node to keep the model planar.",
        "ULS combination = 1.2·DL + 1.6·LL" + (" + 1.0·LAT" if lateral_fx_per_floor_kN else "") + ".",
        p_delta_note,
    ]

    narrative = (
        f"**PyNite 2D moment frame** — ULS.\n\n"
        f"- Max beam |M| ≈ **{max_M:.2f} kN·m**, max beam |V| ≈ **{max_V:.2f} kN**.\n"
        f"- Max column axial |P| ≈ **{max_P:.2f} kN**.\n"
        f"- Peak beam deflection ≈ **{max_def:.2f} mm**.\n"
        f"- Base equilibrium: ΣRx ≈ **{sum_fx:.1f} kN**, ΣRy ≈ **{sum_fy:.1f} kN**.\n"
    )

    result_cards = [
        {"label": "Max beam |M|", "value": f"{max_M:.2f}", "unit": "kN·m", "tone": "neutral"},
        {"label": "Max beam |V|", "value": f"{max_V:.2f}", "unit": "kN", "tone": "neutral"},
        {"label": "Max beam defl.", "value": f"{max_def:.2f}", "unit": "mm", "tone": "neutral"},
        {"label": "Max column |P|", "value": f"{max_P:.2f}", "unit": "kN", "tone": "neutral"},
        {
            "label": "Max storey drift",
            "value": f"{max((s['max_drift_mm'] for s in storey_drifts), default=0.0):.2f}",
            "unit": "mm",
            "tone": "neutral",
        },
        {"label": "Σ base Ry", "value": f"{sum_fy:.1f}", "unit": "kN", "tone": "neutral"},
    ]

    return {
        "analysis_type": "frame_2d",
        "engine": "PyNite",
        "pynite_path": pynite_root_label(),
        "load_combination": combo,
        "geometry": {
            "nodes": geometry_nodes,
            "members": geometry_members,
            "meta": {"source": "fea_frame_2d"},
        },
        "result_cards": result_cards,
        "assumptions": assumptions,
        "summary_markdown": narrative,
        "beams": beams_out,
        "columns": cols_out,
        "base_reactions": reactions,
        "storey_drifts": storey_drifts,
        "p_delta_note": p_delta_note,
        "totals": {
            "max_beam_moment_kNm": round(max_M, 4),
            "max_beam_shear_kN": round(max_V, 4),
            "max_column_axial_kN": round(max_P, 4),
            "max_beam_deflection_mm": round(max_def, 4),
            "sum_base_Rx_kN": round(sum_fx, 4),
            "sum_base_Ry_kN": round(sum_fy, 4),
            "sum_base_Rz_kN": round(sum_fy, 4),
        },
        "diagrams": {
            "moment_per_level_kNm": moment_diag,
            "shear_per_level_kN": shear_diag,
            "x_label_m": "x along beam (m)",
        },
    }


