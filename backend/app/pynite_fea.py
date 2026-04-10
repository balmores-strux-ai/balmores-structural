"""
Parametric 3D building frame in PyNite (gravity + optional lateral), for /fea/analyze.

Adds repo-root ``Pynite-main`` to ``sys.path`` so the vendored library is used without pip install.
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

_REPO_ROOT = Path(__file__).resolve().parents[2]
_PYNITE_ROOT = _REPO_ROOT / "Pynite-main"
if _PYNITE_ROOT.is_dir() and str(_PYNITE_ROOT) not in sys.path:
    sys.path.insert(0, str(_PYNITE_ROOT))


def pynite_available() -> bool:
    return _PYNITE_ROOT.is_dir()


def _mag(res: Any) -> float:
    v = res[0] if isinstance(res, tuple) else res
    return abs(float(v))


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
        "engine": "PyNite",
        "pynite_path": str(_PYNITE_ROOT),
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
) -> Dict[str, Any]:
    """
    Irregular bay grid from span lists; DL + LL load cases; optional façade wind per storey;
    optional roof shear from fraction of estimated gravity (seismic placeholder); ULS = 1.2DL+1.6LL+lat.
    Optionally runs PyNite ``analyze_PDelta`` on ULS.
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
            m.analyze_PDelta(log=False, check_stability=True, max_iter=30, sparse=True, combo_tags=[combo])
            p_delta_note = "P-Δ (second-order) analysis completed for ULS in PyNite."
        except Exception as ex:
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
    top_beams = beam_rows[:40]

    max_p_col = 0.0
    col_rows: List[Dict[str, Any]] = []
    for name in col_members:
        mem = m.members[name]
        p = _mag(mem.max_axial(combo))
        My = _mag(mem.max_moment("My", combo))
        Mz = _mag(mem.max_moment("Mz", combo))
        try:
            T = _mag(mem.max_torque(combo))
        except Exception:
            T = 0.0
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

    return {
        "engine": "PyNite",
        "pynite_path": str(_PYNITE_ROOT),
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
