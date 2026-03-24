from __future__ import annotations

import math
from typing import Any, Dict, List, Tuple

from .schemas import GeometryPayload, GeometryNode, GeometryMember, ProjectState
from .model_loader import BRAIN


def clamp(v: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, v))


def make_i_section(depth: float, scale: float = 1.0) -> Tuple[float, float, float, float]:
    d = float(depth) * float(scale)
    bf = 0.45 * d
    tf = clamp(0.040 * d, 0.018, 0.090)
    tw = clamp(0.020 * d, 0.010, 0.045)
    return d, bf, tf, tw


def i_area(d: float, bf: float, tf: float, tw: float) -> float:
    web_h = max(d - 2.0 * tf, 0.0)
    return 2.0 * bf * tf + web_h * tw


def i_Ix(d: float, bf: float, tf: float, tw: float) -> float:
    web_h = max(d - 2.0 * tf, 0.0)
    i_web = tw * web_h**3 / 12.0
    y = d / 2.0 - tf / 2.0
    i_flange = 2.0 * (((bf * tf**3) / 12.0) + bf * tf * y**2)
    return i_web + i_flange


def i_Iy(d: float, bf: float, tf: float, tw: float) -> float:
    web_h = max(d - 2.0 * tf, 0.0)
    i_web = web_h * tw**3 / 12.0
    i_flange = 2.0 * (tf * bf**3 / 12.0)
    return i_web + i_flange


def i_J(d: float, bf: float, tf: float, tw: float) -> float:
    web_h = max(d - 2.0 * tf, 0.0)
    return (2.0 * bf * tf**3 + web_h * tw**3) / 3.0


def support_mode_id(mode: str) -> float:
    return {"fixed": 1.0, "pinned": 2.0, "mixed": 3.0}.get(mode, 1.0)


def diaphragm_mode_id(mode: str) -> float:
    return {"rigid": 1.0, "semi_rigid": 2.0, "flexible": 3.0}.get(mode, 1.0)


def structural_system_id(building_type: str, brace_mode: str) -> float:
    base = 1.0 if building_type == "steel" else 2.0
    brace = {"braced": 0.0, "unbraced": 1.0, "mixed": 2.0}.get(brace_mode, 0.0)
    return base * 10.0 + brace


def build_geometry(state: ProjectState) -> GeometryPayload:
    nodes: List[GeometryNode] = []
    members: List[GeometryMember] = []

    plan_x = state.bays_x * state.span_x_m
    plan_y = state.bays_y * state.span_y_m
    z_levels = [0.0]
    z_levels.append(state.bottom_story_height_m)
    for _ in range(1, state.stories):
        z_levels.append(z_levels[-1] + state.story_height_m)

    for k, z in enumerate(z_levels):
        for i in range(state.bays_x + 1):
            for j in range(state.bays_y + 1):
                nid = f"N_{k}_{i}_{j}"
                nodes.append(GeometryNode(id=nid, x=i * state.span_x_m, y=j * state.span_y_m, z=z))

    for k in range(len(z_levels) - 1):
        for i in range(state.bays_x + 1):
            for j in range(state.bays_y + 1):
                members.append(GeometryMember(
                    id=f"C_{k}_{i}_{j}",
                    start=f"N_{k}_{i}_{j}",
                    end=f"N_{k+1}_{i}_{j}",
                    kind="column",
                    group=None,
                ))

    for k in range(1, len(z_levels)):
        for i in range(state.bays_x):
            for j in range(state.bays_y + 1):
                members.append(GeometryMember(
                    id=f"BX_{k}_{i}_{j}",
                    start=f"N_{k}_{i}_{j}",
                    end=f"N_{k}_{i+1}_{j}",
                    kind="beam",
                    group=None,
                ))
        for i in range(state.bays_x + 1):
            for j in range(state.bays_y):
                members.append(GeometryMember(
                    id=f"BY_{k}_{i}_{j}",
                    start=f"N_{k}_{i}_{j}",
                    end=f"N_{k}_{i}_{j+1}",
                    kind="beam",
                    group=None,
                ))

    if state.brace_mode in {"braced", "mixed"}:
        for k in range(len(z_levels) - 1):
            members.append(GeometryMember(
                id=f"BRX_{k}",
                start=f"N_{k}_0_0",
                end=f"N_{k+1}_1_0",
                kind="brace",
                group=None,
            ))
            members.append(GeometryMember(
                id=f"BRY_{k}",
                start=f"N_{k}_{state.bays_x}_{state.bays_y}",
                end=f"N_{k+1}_{max(state.bays_x-1,0)}_{state.bays_y}",
                kind="brace",
                group=None,
            ))

    if state.cantilever_length_m > 0:
        for k in range(1, len(z_levels)):
            start = f"N_{k}_{state.bays_x}_{state.bays_y}"
            end_id = f"CN_{k}"
            nodes.append(GeometryNode(id=end_id, x=plan_x + state.cantilever_length_m, y=plan_y, z=z_levels[k]))
            members.append(GeometryMember(
                id=f"CANT_{k}",
                start=start,
                end=end_id,
                kind="cantilever",
                group=None,
            ))

    return GeometryPayload(
        nodes=nodes,
        members=members,
        meta={
            "plan_x_m": plan_x,
            "plan_y_m": plan_y,
            "stories": state.stories,
        },
    )


def feature_dict_from_state(state: ProjectState) -> Dict[str, float]:
    stories = float(state.stories)
    bays_x = float(state.bays_x)
    bays_y = float(state.bays_y)
    span_x = float(state.span_x_m)
    span_y = float(state.span_y_m)
    story_h = float(state.story_height_m)
    bottom_h = float(state.bottom_story_height_m)
    total_h = bottom_h + max(state.stories - 1, 0) * story_h
    plan_x = bays_x * span_x
    plan_y = bays_y * span_y
    plan_area = plan_x * plan_y
    aspect_ratio = plan_x / max(plan_y, 1e-6)

    brace_x = 1.0 if state.brace_mode in {"braced", "mixed"} else 0.0
    brace_y = 1.0 if state.brace_mode == "braced" else 0.0
    brace_density = 0.9 if state.brace_mode == "braced" else 0.25 if state.brace_mode == "mixed" else 0.01

    beam_scale = 1.0
    col_scale = 1.08 if state.support_mode == "pinned" else 1.0
    brace_scale = 1.0 if state.brace_mode != "unbraced" else 0.1
    beam_stiff = 0.94 if state.support_mode == "pinned" else 1.0
    col_stiff = 0.92 if state.support_mode == "pinned" else 1.03
    brace_stiff = 1.0 if state.brace_mode != "unbraced" else 0.15

    beam_d = clamp(max(max(span_x, span_y) / 18.0, 0.45), 0.45, 1.20)
    col_d = clamp(max(0.70, 0.018 * total_h + 0.03 * max(span_x, span_y) * math.sqrt(max(stories, 1))), 0.70, 2.20)
    brace_d = clamp(max(0.35, 0.65 * beam_d), 0.30, 0.95)

    bsec = make_i_section(beam_d, beam_scale)
    csec = make_i_section(col_d, col_scale)
    rsec = make_i_section(brace_d, brace_scale)

    A_beam = i_area(*bsec)
    I_beam = i_Ix(*bsec)
    Iy_beam = i_Iy(*bsec)
    J_beam = i_J(*bsec)

    A_col = i_area(*csec)
    I_col = i_Ix(*csec)
    Iy_col = i_Iy(*csec)
    J_col = i_J(*csec)

    A_br = i_area(*rsec)
    I_br = i_Ix(*rsec)
    Iy_br = i_Iy(*rsec)
    J_br = i_J(*rsec)

    E = 200e6
    alpha = 1.2e-5
    rho = 800.203873598369
    fy_kpa = state.fy_mpa * 1000.0

    selfwt = 1.8 if state.building_type == "steel" else 2.4
    total_weight = plan_area * ((selfwt + state.dead_kpa + 0.25 * state.live_kpa) * max(stories - 1.0, 0.0) + (selfwt + state.dead_kpa + 0.25 * state.live_kpa + state.snow_kpa))
    total_mass = total_weight / 9.81

    Vwx = total_weight * state.wind_coeff_x
    Vwy = total_weight * state.wind_coeff_y
    Veqx = total_weight * state.eq_coeff_x
    Veqy = total_weight * state.eq_coeff_y
    Vimpx = total_weight * 0.02
    Vimpy = total_weight * 0.02
    Vblastx = total_weight * 0.03
    Vblasty = total_weight * 0.03

    perimeter_cols = 2.0 * (bays_x + bays_y + 2.0)
    brace_mult_x = 1.0 + 3.0 * brace_x
    brace_mult_y = 1.0 + 3.0 * brace_y
    k_story_x = perimeter_cols * 12.0 * E * I_col / max(story_h**3, 1e-9) * brace_mult_x * col_stiff
    k_story_y = perimeter_cols * 12.0 * E * I_col / max(story_h**3, 1e-9) * brace_mult_y * col_stiff
    k_total_x = max(k_story_x / max(stories, 1.0), 1e-9)
    k_total_y = max(k_story_y / max(stories, 1.0), 1e-9)

    drift_wx = Vwx / k_total_x
    drift_wy = Vwy / k_total_y
    drift_eqx = Veqx / k_total_x
    drift_eqy = Veqy / k_total_y
    drift_impx = Vimpx / k_total_x
    drift_impy = Vimpy / k_total_y
    drift_blastx = Vblastx / k_total_x
    drift_blasty = Vblasty / k_total_y

    ecc = state.eccentricity_ratio * max(plan_x, plan_y, 1.0)
    theta_w = Vwx * ecc / max(E * J_col, 1e-9)
    theta_eq = Veqx * ecc / max(E * J_col, 1e-9)
    theta_blast = Vblastx * ecc / max(E * J_col, 1e-9)

    beam_line_load = (state.dead_kpa + state.live_kpa + state.snow_kpa) * max(span_x, span_y)
    beam_m_proxy = beam_line_load * max(span_x, span_y) ** 2 / 8.0
    beam_defl_proxy = 5.0 * beam_line_load * max(span_x, span_y) ** 4 / max(384.0 * E * I_beam, 1e-9)
    col_p_proxy = total_weight / max((bays_x + 1.0) * (bays_y + 1.0), 1.0)
    brace_p_proxy = max(Veqx, Veqy) / max(4.0, 1.0)
    temp_delta = 20.0
    temp_axial = E * A_col * alpha * temp_delta
    cant_m = beam_m_proxy * max(state.cantilever_fraction, 0.05)
    cant_defl = beam_defl_proxy * max(state.cantilever_fraction, 0.05)

    Pcr = math.pi**2 * E * I_col / max(story_h**2, 1e-9)
    buckling_ratio = col_p_proxy / max(Pcr, 1e-9)
    beam_myield = fy_kpa * (I_beam / max(bsec[0] / 2.0, 1e-9))
    beam_plastic_ratio = beam_m_proxy / max(beam_myield, 1e-9)
    col_axial_yield_ratio = col_p_proxy / max(A_col * fy_kpa, 1e-9)

    Tx = 2.0 * math.pi * math.sqrt(max(total_mass / k_total_x, 1e-12))
    Ty = 2.0 * math.pi * math.sqrt(max(total_mass / k_total_y, 1e-12))
    Tt = 2.0 * math.pi * math.sqrt(max(total_mass * (plan_x**2 + plan_y**2) / max(E * J_col, 1e-9), 1e-12))

    thermal_exp = alpha * temp_delta * total_h
    thermal_strain = alpha * temp_delta
    thermal_energy_density = 0.5 * E * thermal_strain**2 / 1000.0
    thermal_stress = E * thermal_strain
    thermal_stress_ratio = thermal_stress / max(fy_kpa, 1e-9)

    beam_slender = max(span_x, span_y) / max(beam_d, 1e-9)
    col_slender = story_h / max(col_d, 1e-9)
    shear_flex = max(span_x, span_y) / max(A_beam, 1e-9) * 1e-3
    axial_short = col_p_proxy * total_h / max(E * A_col, 1e-9)
    strain_energy = 0.5 * total_weight * max(drift_eqx, drift_eqy)
    p_delta = col_p_proxy * max(drift_eqx, drift_eqy) / max(beam_m_proxy, 1e-9)
    ltb = beam_m_proxy / max(E * Iy_beam, 1e-9)
    mass_irreg = 1.0 + abs(state.setback_ratio) + abs(state.eccentricity_ratio)
    modal_density = stories / max(Tx + Ty + Tt, 1e-9)
    skybridge = 0.0
    cluster_irreg = 1.0 + abs(state.twist_total_deg) / 30.0 + abs(state.plan_skew_deg) / 30.0
    floor_stiff_var = abs(state.setback_ratio) + abs(state.plan_skew_deg) / 90.0
    support_irreg = 0.5 if state.support_mode == "mixed" else 0.0
    diaphragm_irreg = 0.5 if state.diaphragm_mode != "rigid" else 0.0
    blast_impulse = Vblastx * max(story_h, 1.0)
    impact_energy = Vimpx * max(story_h, 1.0)
    story_shear_grad = worst = max(Veqx, Veqy) / max(stories, 1.0)
    robustness = max(0.1, 1.5 - buckling_ratio - support_irreg)
    cant_ratio = state.cantilever_fraction

    support_fixity_index = {"fixed": 1.0, "pinned": 0.35, "mixed": 0.65}[state.support_mode]
    diaphragm_constraint_index = {"rigid": 1.0, "semi_rigid": 0.65, "flexible": 0.35}[state.diaphragm_mode]

    # New physics features (round5 v3 physics enhanced)
    moment_magnification_B1_proxy = 1.0 / max(1.0 - col_p_proxy / max(Pcr, 1e-9), 0.2)
    shear_deformation_ratio_proxy = (E / 77e6) * (I_beam / max(A_beam, 1e-9)) / max(max(span_x, span_y)**2, 1e-9)
    overturning_moment_index = max(Veqx, Veqy) * total_h / max(total_weight * max(plan_x, plan_y) * 0.5, 1e-9)
    story_drift_regularity_proxy = 1.0 - min(0.5, story_shear_grad / max(max(Veqx, Veqy), 1e-9) * story_h)
    foundation_flexibility_proxy = 1.0 - 0.3 * (1.0 - support_fixity_index)
    second_order_amplification_proxy = 1.0 / max(1.0 - buckling_ratio * 0.5, 0.3)
    diaphragm_shear_flow_index = diaphragm_constraint_index * math.sqrt(plan_area) / max(total_h, 1e-9)
    modal_participation_proxy = 0.85 + 0.15 * (1.0 - mass_irreg / 2.0)

    feature_dict = {
        "stories": stories, "bays_x": bays_x, "bays_y": bays_y,
        "span_x_m": span_x, "span_y_m": span_y, "story_height_m": story_h, "total_height_m": total_h,
        "plan_x_m": plan_x, "plan_y_m": plan_y, "plan_area_m2": plan_area,
        "slender_x": total_h / max(plan_x, 1e-9), "slender_y": total_h / max(plan_y, 1e-9),
        "aspect_ratio": aspect_ratio, "aerodynamic_slenderness": total_h / max((plan_x + plan_y) / 2.0, 1e-9),
        "brace_x": brace_x, "brace_y": brace_y, "brace_density": brace_density,
        "beam_scale": beam_scale, "col_scale": col_scale, "brace_scale": brace_scale,
        "beam_stiffness_modifier": beam_stiff, "col_stiffness_modifier": col_stiff, "brace_stiffness_modifier": brace_stiff,
        "super_dead_kpa": state.dead_kpa, "live_kpa": state.live_kpa, "snow_kpa": state.snow_kpa, "selfwt_floor_kpa": selfwt,
        "wind_coeff_x": state.wind_coeff_x, "wind_coeff_y": state.wind_coeff_y, "eq_coeff_x": state.eq_coeff_x, "eq_coeff_y": state.eq_coeff_y,
        "temp_delta_c": 20.0, "plan_skew_deg": state.plan_skew_deg, "twist_total_deg": state.twist_total_deg,
        "lean_total_m": state.lean_total_m, "setback_ratio": state.setback_ratio, "eccentricity_ratio": state.eccentricity_ratio,
        "aero_shape_factor": 1.0, "tower_gap_m": 0.0, "podium_levels": 1.0 if state.plan_shape == "podium_tower" else 0.0,
        "shape_irregularity_index": 0.2 if state.plan_shape == "rect" else 0.8,
        "out_of_plane_irregularity": abs(state.plan_skew_deg) / 90.0,
        "cluster_tower_count": 1.0, "bridge_levels_n": 0.0, "bridge_links_per_level": 0.0, "bridge_span_m": 0.0,
        "beam_release_fraction": 0.15 if state.support_mode == "pinned" else 0.02,
        "beam_release_mode_id": 2.0 if state.support_mode == "pinned" else 1.0,
        "support_mode_id": support_mode_id(state.support_mode), "diaphragm_mode_id": diaphragm_mode_id(state.diaphragm_mode),
        "support_fixity_index": support_fixity_index, "diaphragm_constraint_index": diaphragm_constraint_index,
        "mass_source_factor": 1.0, "live_load_reduction_factor": 0.9 if stories >= 10 else 1.0,
        "wind_attack_angle_deg": 0.0, "k_factor_col": 1.0 if state.brace_mode != "unbraced" else 1.4,
        "soft_story_factor": 1.0, "structural_system_id": structural_system_id(state.building_type, state.brace_mode),
        "floor_stiffness_gradient": 1.0, "in_floor_stiffness_variation": 0.08, "story_mass_gradient": 1.0,
        "live_impact_factor": 1.05, "site_class_factor": 1.0 if state.code != "Philippines" else 1.1,
        "damping_ratio_model": 0.02, "response_mod_factor": 5.0 if state.brace_mode == "braced" else 3.5, "redundancy_factor": 1.0,
        "panel_zone_flexibility": 0.12, "snow_drift_factor": 1.0, "ponding_factor": 1.0,
        "cantilever_fraction": state.cantilever_fraction, "cantilever_length_m": state.cantilever_length_m, "cantilever_levels_n": max(1.0, stories * state.cantilever_fraction) if state.cantilever_fraction > 0 else 0.0,
        "analysis_dimensionality_id": 3.0, "two_d_branch_flag": 0.0,
        "impact_coeff_x": 0.02, "impact_coeff_y": 0.02, "blast_coeff_x": 0.03, "blast_coeff_y": 0.03,
        "A_beam_m2": A_beam, "I_beam_m4": I_beam, "Iy_beam_m4": Iy_beam, "J_beam_m4": J_beam,
        "A_col_m2": A_col, "I_col_m4": I_col, "Iy_col_m4": Iy_col, "J_col_m4": J_col,
        "A_br_m2": A_br, "I_br_m4": I_br, "Iy_br_m4": Iy_br, "J_br_m4": J_br,
        "total_weight_kN": total_weight, "total_mass_kNs2_m": total_mass,
        "V_wx_proxy_kN": Vwx, "V_wy_proxy_kN": Vwy, "V_eqx_proxy_kN": Veqx, "V_eqy_proxy_kN": Veqy,
        "V_imp_x_proxy_kN": Vimpx, "V_imp_y_proxy_kN": Vimpy, "V_blast_x_proxy_kN": Vblastx, "V_blast_y_proxy_kN": Vblasty,
        "drift_proxy_wx_m": drift_wx, "drift_proxy_wy_m": drift_wy, "drift_proxy_eqx_m": drift_eqx, "drift_proxy_eqy_m": drift_eqy,
        "drift_proxy_impx_m": drift_impx, "drift_proxy_impy_m": drift_impy, "drift_proxy_blastx_m": drift_blastx, "drift_proxy_blasty_m": drift_blasty,
        "torsion_theta_w_rad": theta_w, "torsion_theta_eq_rad": theta_eq, "torsion_theta_blast_rad": theta_blast,
        "beam_m_proxy_kNm": beam_m_proxy, "beam_defl_proxy_m": beam_defl_proxy, "col_p_proxy_kN": col_p_proxy, "brace_p_proxy_kN": brace_p_proxy,
        "temp_axial_proxy_kN": temp_axial, "cantilever_m_proxy_kNm": cant_m, "cantilever_defl_proxy_m": cant_defl, "Pcr_proxy_kN": Pcr,
        "buckling_ratio": buckling_ratio, "beam_myield_proxy_kNm": beam_myield, "beam_plastic_ratio": beam_plastic_ratio,
        "col_axial_yield_ratio": col_axial_yield_ratio, "T_proxy_x_s": Tx, "T_proxy_y_s": Ty, "T_torsion_proxy_s": Tt,
        "thermal_free_expansion_m": thermal_exp, "thermal_strain": thermal_strain, "thermal_energy_density_kNm_m3": thermal_energy_density,
        "thermal_stress_kPa": thermal_stress, "thermal_stress_ratio": thermal_stress_ratio,
        "beam_slenderness_proxy": beam_slender, "col_slenderness_proxy": col_slender, "shear_flexibility_proxy": shear_flex,
        "axial_shortening_proxy_m": axial_short, "strain_energy_proxy_kNm": strain_energy, "p_delta_index_proxy": p_delta,
        "lateral_torsional_index": ltb, "mass_irregularity_proxy": mass_irreg, "modal_density_proxy": modal_density,
        "skybridge_coupling_index": skybridge, "cluster_irregularity_proxy": cluster_irreg, "floor_stiffness_variation_index": floor_stiff_var,
        "support_irregularity_index": support_irreg, "diaphragm_irregularity_index": diaphragm_irreg,
        "blast_impulse_proxy_kN_m": blast_impulse, "impact_energy_proxy_kNm": impact_energy, "story_shear_gradient_proxy": story_shear_grad,
        "robustness_proxy": robustness, "cantilever_ratio": cant_ratio,
        "moment_magnification_B1_proxy": moment_magnification_B1_proxy,
        "shear_deformation_ratio_proxy": shear_deformation_ratio_proxy,
        "overturning_moment_index": overturning_moment_index,
        "story_drift_regularity_proxy": story_drift_regularity_proxy,
        "foundation_flexibility_proxy": foundation_flexibility_proxy,
        "second_order_amplification_proxy": second_order_amplification_proxy,
        "diaphragm_shear_flow_index": diaphragm_shear_flow_index,
        "modal_participation_proxy": modal_participation_proxy,
        "brace_added": max(0.0, stories * 2.0 if state.brace_mode in {"braced", "mixed"} else 0.0),
        "brace_attempted": max(1.0, stories * 2.0),
        "bridge_members_added": 0.0, "cantilever_members_added": max(0.0, stories if state.cantilever_length_m > 0 else 0.0),
        "released_beams_n": max(0.0, stories * 2.0 if state.support_mode == "pinned" else stories * 0.25),
        "transformed_points_n": max(1.0, (state.bays_x + 1) * (state.bays_y + 1) * (state.stories + 1)),
        "diaphragm_assigned_n": max(1.0, state.stories),
        "fixed_support_points_n": max(1.0, (state.bays_x + 1) * (state.bays_y + 1) * (1.0 if state.support_mode == "fixed" else 0.4 if state.support_mode == "mixed" else 0.0)),
        "pinned_support_points_n": max(0.0, (state.bays_x + 1) * (state.bays_y + 1) * (1.0 if state.support_mode == "pinned" else 0.6 if state.support_mode == "mixed" else 0.0)),
        "Vwx_assigned_kN": Vwx, "Vwy_assigned_kN": Vwy, "Veqx_assigned_kN": Veqx, "Veqy_assigned_kN": Veqy,
        "Vimpx_assigned_kN": Vimpx, "Vimpy_assigned_kN": Vimpy, "Vblastx_assigned_kN": Vblastx, "Vblasty_assigned_kN": Vblasty,
    }

    # Fill missing feature columns defensively
    for col in BRAIN.feature_columns:
        feature_dict.setdefault(col, 1e-9)
        if feature_dict[col] == 0:
            feature_dict[col] = 1e-9
    return feature_dict


def _pred_fallback(pred: Dict[str, float], primary: str, fallback: str, default: float = 0.0) -> float:
    v = pred.get(primary, default)
    if v is not None and float(v) != 0:
        return float(v)
    return float(pred.get(fallback, default))


def build_member_schedule(state: ProjectState, predictions: Dict[str, float]) -> Dict[str, Any]:
    beam_m = _pred_fallback(predictions, "max_beam_moment_kNm", "beam_m3_grav_kNm")
    beam_v = predictions.get("max_beam_shear_kN") or predictions.get("max_beam_end_shear_kN", 0.0) or 1e-6
    col_p = _pred_fallback(predictions, "max_column_axial_kN", "col_p_grav_kN")
    story_count = max(state.stories, 1)

    beam_groups = []
    column_groups = []

    beam_groups.append({
        "group": "B1",
        "scope": "Typical interior beams",
        "design_basis": f"Moment {beam_m*0.75:.1f} kNm, Shear {beam_v*0.70:.1f} kN",
        "suggested_section": "Deeper beam group",
    })
    beam_groups.append({
        "group": "B2",
        "scope": "Perimeter / secondary beams",
        "design_basis": f"Moment {beam_m*0.45:.1f} kNm, Shear {beam_v*0.50:.1f} kN",
        "suggested_section": "Secondary beam group",
    })
    if state.cantilever_length_m > 0:
        beam_groups.append({
            "group": "B3",
            "scope": "Cantilever beams",
            "design_basis": f"Moment {predictions.get('cantilever_m3_grav_kNm', 0.0):.1f} kNm",
            "suggested_section": "Cantilever beam group",
        })

    column_groups.append({
        "group": "C1",
        "scope": "Base to lower storeys",
        "design_basis": f"Axial {col_p:.1f} kN",
        "suggested_section": "Largest column group",
    })
    column_groups.append({
        "group": "C2",
        "scope": "Mid storeys",
        "design_basis": f"Axial {col_p*0.65:.1f} kN",
        "suggested_section": "Medium column group",
    })
    column_groups.append({
        "group": "C3",
        "scope": "Upper storeys",
        "design_basis": f"Axial {col_p*0.35:.1f} kN",
        "suggested_section": "Light column group",
    })

    return {"beam_groups": beam_groups, "column_groups": column_groups}


def confidence_label(state: ProjectState) -> str:
    in_range = 0
    total = 0
    # cheap range check on key fields
    keys = ["stories", "bays_x", "bays_y", "span_x_m", "span_y_m", "story_height_m", "total_height_m"]
    fdict = feature_dict_from_state(state)
    for k in keys:
        if k in BRAIN.training_ranges:
            total += 1
            lo, hi = BRAIN.training_ranges[k]
            if lo <= fdict[k] <= hi:
                in_range += 1
    ratio = in_range / max(total, 1)
    if ratio > 0.85:
        return "High"
    if ratio > 0.55:
        return "Moderate"
    return "Low"
