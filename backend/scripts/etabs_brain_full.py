"""
ETABS full pipeline: connect via comtypes, create parametric steel deck models,
run analysis, extract results, append to teacher CSV/parquet, then retrain the
structural brain. SELF-CONTAINED — all logic inline.

Requires: ETABS installed, comtypes, psutil, pandas, torch, numpy
  pip install comtypes psutil pandas torch numpy

Usage:
  cd backend
  python scripts/etabs_brain_full.py

Connects to ETABS via comtypes, creates parametric models, runs analysis,
extracts results, appends to CSV, and trains the brain.
"""
from __future__ import annotations

import gc
import json
import math
import os
import sys
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

# --- CONFIG ---
SCRIPT_VERSION = "etabs_brain_full_round6_100k_roadmap"

CONFIG = {
    # Long-term dataset goal (env: ETABS_TARGET_TOTAL). Each run adds at most MAX_NEW_PER_RUN.
    "TARGET_TOTAL_MODELS": 100_000,
    # Cap per run so one session does not try to build 100k models at once (weeks of runtime).
    "MAX_NEW_PER_RUN": 2500,
    "TARGET_NEW_MODELS_IF_NO_EXISTING": 2500,
    "RANDOM_SEED": 42,
    "APPEND_DATASET": True,
    "PRESENT_UNITS": 6,  # kN-m-C (SI)
    "RESTART_EVERY_N_MODELS": 200,
    "RESTART_AFTER_CONSECUTIVE_FAILS": 5,
    "RESTART_SLOW_MODEL_THRESHOLD_S": 60,
}


def _env_int(name: str, default: int) -> int:
    raw = os.environ.get(name, "").strip()
    if not raw:
        return default
    try:
        return int(raw)
    except ValueError:
        return default


def _env_bool(name: str) -> bool:
    return os.environ.get(name, "").strip().lower() in ("1", "true", "yes", "on")

# Path resolution — output files go to backend/data and backend/models
_BACKEND = Path(__file__).resolve().parent.parent
DATASET_CSV = _BACKEND / "data" / "etabs_parametric_structural_teacher.csv"
DATASET_PARQUET = _BACKEND / "data" / "etabs_parametric_structural_teacher.parquet"
MODEL_FILE = _BACKEND / "models" / "etabs_parametric_structural_brain.pt"
METRICS_JSON = _BACKEND / "data" / "etabs_parametric_structural_metrics.json"
PROGRESS_JSON = _BACKEND / "data" / "etabs_progress_status.json"
FAILED_SAMPLES_CSV = _BACKEND / "data" / "etabs_failed_samples_log.csv"
TEMP_MODEL_DIR = _BACKEND / "data" / "etabs_temp_models"

# Add parent for app imports if needed
sys.path.insert(0, str(_BACKEND))

# Units: 6 = kN-m-C (SI)
ETABS_UNITS_KNM_C = 6

# Parametric ranges for steel deck
STORY_RANGE = (2, 8)
BAYS_X_RANGE = (2, 6)
BAYS_Y_RANGE = (2, 5)
SPAN_X_RANGE = (4.0, 9.0)  # m
SPAN_Y_RANGE = (4.0, 9.0)  # m
STORY_HEIGHT_RANGE = (3.0, 5.0)
BOTTOM_STORY_MULT = (0.9, 1.2)
DEAD_KPA_RANGE = (2.0, 3.5)
LIVE_KPA_RANGE = (2.0, 5.0)

# Demand grouping: 6 groups (Dead, Live, Wind X/Y, EQ X/Y)
DEMAND_GROUPS_CREATED = 6

# 14 archetypes (full round4 spec)
ARCHETYPES = [
    "lowrise_wide", "midrise_regular", "highrise_regular",
    "slender_tower", "hybrid_braced", "long_span_hall",
    "twisted_tower", "leaning_tower", "setback_tower",
    "twin_tower_podium", "mega_extreme", "triple_twin_mega_cluster",
    "twisted_quad_bridge_cluster", "two_d_branch",
]

# Load pattern types (CSI ETABS)
LOAD_PAT_DEAD = 1
LOAD_PAT_LIVE = 2
LOAD_PAT_SNOW = 3
LOAD_PAT_WIND = 4
LOAD_PAT_QUAKE = 5
LOAD_PAT_TEMP = 6
LOAD_PAT_OTHER = 7

# --- FEATURE COLUMNS (inference pipeline + 8 new physics columns) ---
FEATURE_COLUMNS: List[str] = [
    "stories", "bays_x", "bays_y", "span_x_m", "span_y_m", "story_height_m", "total_height_m",
    "plan_x_m", "plan_y_m", "plan_area_m2", "slender_x", "slender_y", "aspect_ratio", "aerodynamic_slenderness",
    "brace_x", "brace_y", "brace_density", "beam_scale", "col_scale", "brace_scale",
    "beam_stiffness_modifier", "col_stiffness_modifier", "brace_stiffness_modifier",
    "super_dead_kpa", "live_kpa", "snow_kpa", "selfwt_floor_kpa", "temp_delta_c",
    "wind_coeff_x", "wind_coeff_y", "eq_coeff_x", "eq_coeff_y",
    "Vwx_assigned_kN", "Vwy_assigned_kN", "Veqx_assigned_kN", "Veqy_assigned_kN",
    "A_beam_m2", "I_beam_m4", "A_col_m2", "I_col_m4", "A_br_m2", "I_br_m4",
    "total_weight_kN", "V_wx_proxy_kN", "V_wy_proxy_kN", "V_eqx_proxy_kN", "V_eqy_proxy_kN",
    "drift_proxy_wx_m", "drift_proxy_wy_m", "drift_proxy_eqx_m", "drift_proxy_eqy_m",
    "beam_m_proxy_kNm", "col_p_proxy_kN", "brace_p_proxy_kN", "temp_axial_proxy_kN", "Pcr_proxy_kN",
    "buckling_ratio", "T_proxy_x_s", "T_proxy_y_s", "thermal_free_expansion_m",
    "Iy_beam_m4", "J_beam_m4", "Iy_col_m4", "J_col_m4", "Iy_br_m4", "J_br_m4", "total_mass_kNs2_m",
    "torsion_theta_w_rad", "torsion_theta_eq_rad", "beam_defl_proxy_m", "beam_myield_proxy_kNm",
    "beam_plastic_ratio", "col_axial_yield_ratio", "beam_slenderness_proxy", "col_slenderness_proxy",
    "strain_energy_proxy_kNm", "T_torsion_proxy_s", "thermal_strain", "thermal_energy_density_kNm_m3",
    "beam_release_fraction", "beam_release_mode_id", "cluster_tower_count", "bridge_levels_n",
    "bridge_links_per_level", "bridge_span_m", "bridge_members_added", "cantilever_fraction",
    "cantilever_length_m", "cantilever_levels_n", "cantilever_members_added",
    "support_mode_id", "diaphragm_mode_id", "support_fixity_index", "diaphragm_constraint_index",
    "fixed_support_points_n", "pinned_support_points_n", "diaphragm_assigned_n",
    "floor_stiffness_gradient", "in_floor_stiffness_variation", "story_mass_gradient",
    "live_impact_factor", "site_class_factor", "damping_ratio_model", "response_mod_factor",
    "redundancy_factor", "panel_zone_flexibility", "snow_drift_factor", "ponding_factor",
    "mass_source_factor", "live_load_reduction_factor", "wind_attack_angle_deg", "k_factor_col",
    "soft_story_factor", "structural_system_id",
    "impact_coeff_x", "impact_coeff_y", "blast_coeff_x", "blast_coeff_y",
    "V_imp_x_proxy_kN", "V_imp_y_proxy_kN", "V_blast_x_proxy_kN", "V_blast_y_proxy_kN",
    "drift_proxy_impx_m", "drift_proxy_impy_m", "drift_proxy_blastx_m", "drift_proxy_blasty_m",
    "torsion_theta_blast_rad", "cantilever_m_proxy_kNm", "cantilever_defl_proxy_m",
    "thermal_stress_kPa", "thermal_stress_ratio", "floor_stiffness_variation_index",
    "support_irregularity_index", "diaphragm_irregularity_index",
    "blast_impulse_proxy_kN_m", "impact_energy_proxy_kNm", "story_shear_gradient_proxy",
    "robustness_proxy", "cantilever_ratio",
    "Vimpx_assigned_kN", "Vimpy_assigned_kN", "Vblastx_assigned_kN", "Vblasty_assigned_kN",
    "plan_skew_deg", "twist_total_deg", "lean_total_m", "setback_ratio", "eccentricity_ratio",
    "aero_shape_factor", "tower_gap_m", "podium_levels", "shape_irregularity_index",
    "out_of_plane_irregularity", "brace_added", "brace_attempted", "released_beams_n",
    "transformed_points_n", "shear_flexibility_proxy", "axial_shortening_proxy_m", "p_delta_index_proxy",
    "lateral_torsional_index", "mass_irregularity_proxy", "modal_density_proxy",
    "skybridge_coupling_index", "cluster_irregularity_proxy",
    "moment_magnification_B1_proxy", "shear_deformation_ratio_proxy", "overturning_moment_index",
    "story_drift_regularity_proxy", "foundation_flexibility_proxy", "second_order_amplification_proxy",
    "diaphragm_shear_flow_index", "modal_participation_proxy",
]
_seen_f = set()
FEATURE_COLUMNS = [c for c in FEATURE_COLUMNS if not (c in _seen_f or _seen_f.add(c))]

TARGET_COLUMNS: List[str] = [
    "max_beam_shear_kN", "max_beam_end_shear_kN", "max_beam_moment_kNm", "max_column_axial_kN",
    "max_joint_reaction_vertical_kN", "max_beam_deflection_mm", "max_drift_mm", "max_beam_end_moment_kNm",
]


def _get(row: Dict[str, float], key: str, default: float = 0.0) -> float:
    v = row.get(key, default)
    if v is None or (isinstance(v, float) and math.isnan(v)):
        return default
    return float(v)


def choose_round4_archetype(stories: float, bays_x: float, bays_y: float,
                            plan_x: float, plan_y: float, total_h: float,
                            brace_density: float = 0.0, rng: Optional[np.random.Generator] = None) -> str:
    """Choose archetype from geometry. Balanced sampling with failure adaptation (14 archetypes)."""
    n_stories = int(stories)
    plan_area = plan_x * plan_y if plan_x > 0 and plan_y > 0 else 1.0
    slen_x = total_h / plan_x if plan_x > 0 else 0.0
    slen_y = total_h / plan_y if plan_y > 0 else 0.0
    slen = (slen_x + slen_y) / 2.0
    aspect = plan_x / plan_y if plan_y > 0 else 1.0
    n_bays = int(bays_x * bays_y)

    if rng is not None and rng.random() < 0.15:
        return rng.choice(ARCHETYPES)

    if slen > 3.5 and n_bays <= 12:
        return "slender_tower"
    if n_bays >= 40 and n_stories <= 6 and max(plan_x, plan_y) > 50:
        return "long_span_hall"
    if brace_density > 0.3 and n_stories >= 6:
        return "hybrid_braced"
    if n_stories >= 25 and n_bays >= 30:
        return "mega_extreme"
    if aspect > 2.5 or aspect < 0.4:
        if n_stories >= 15:
            return "twisted_tower" if (rng is not None and rng.random() < 0.5) else "leaning_tower"
        return "two_d_branch"

    if n_stories <= 4:
        return "lowrise_wide" if n_bays >= 16 else "lowrise_wide"
    if n_stories <= 12:
        return "midrise_regular"
    if n_stories <= 20:
        return "highrise_regular"
    return "slender_tower"


def sample_building_params(rng: np.random.Generator, force_archetype: Optional[str] = None) -> Dict[str, float]:
    """Sample parametric building parameters. Per-archetype logic for 14 archetypes."""
    archetype = force_archetype or rng.choice(ARCHETYPES)
    if archetype == "lowrise_wide":
        num_stories = int(rng.integers(2, 5))
        bays_x = int(rng.integers(4, 7))
        bays_y = int(rng.integers(4, 6))
        span_x = float(rng.uniform(5.0, 8.0))
        span_y = float(rng.uniform(5.0, 8.0))
    elif archetype == "midrise_regular":
        num_stories = int(rng.integers(5, 13))
        bays_x = int(rng.integers(3, 6))
        bays_y = int(rng.integers(3, 5))
        span_x = float(rng.uniform(5.0, 7.5))
        span_y = float(rng.uniform(5.0, 7.5))
    elif archetype == "highrise_regular":
        num_stories = int(rng.integers(13, 22))
        bays_x = int(rng.integers(4, 7))
        bays_y = int(rng.integers(4, 6))
        span_x = float(rng.uniform(4.5, 7.0))
        span_y = float(rng.uniform(4.5, 7.0))
    elif archetype == "slender_tower":
        num_stories = int(rng.integers(15, 25))
        bays_x = int(rng.integers(2, 4))
        bays_y = int(rng.integers(2, 4))
        span_x = float(rng.uniform(4.0, 6.0))
        span_y = float(rng.uniform(4.0, 6.0))
    elif archetype == "hybrid_braced":
        num_stories = int(rng.integers(8, 15))
        bays_x = int(rng.integers(3, 5))
        bays_y = int(rng.integers(3, 5))
        span_x = float(rng.uniform(5.0, 7.0))
        span_y = float(rng.uniform(5.0, 7.0))
    elif archetype == "long_span_hall":
        num_stories = int(rng.integers(2, 6))
        bays_x = int(rng.integers(6, 10))
        bays_y = int(rng.integers(4, 8))
        span_x = float(rng.uniform(8.0, 12.0))
        span_y = float(rng.uniform(6.0, 10.0))
    elif archetype == "twisted_tower":
        num_stories = int(rng.integers(12, 20))
        bays_x = int(rng.integers(3, 5))
        bays_y = int(rng.integers(3, 5))
        span_x = float(rng.uniform(5.0, 7.0))
        span_y = float(rng.uniform(5.0, 7.0))
    elif archetype == "leaning_tower":
        num_stories = int(rng.integers(10, 18))
        bays_x = int(rng.integers(2, 4))
        bays_y = int(rng.integers(2, 4))
        span_x = float(rng.uniform(4.5, 6.5))
        span_y = float(rng.uniform(4.5, 6.5))
    elif archetype == "setback_tower":
        num_stories = int(rng.integers(12, 20))
        bays_x = int(rng.integers(4, 6))
        bays_y = int(rng.integers(4, 6))
        span_x = float(rng.uniform(5.0, 7.0))
        span_y = float(rng.uniform(5.0, 7.0))
    elif archetype == "twin_tower_podium":
        num_stories = int(rng.integers(8, 15))
        bays_x = int(rng.integers(4, 6))
        bays_y = int(rng.integers(4, 6))
        span_x = float(rng.uniform(5.0, 7.0))
        span_y = float(rng.uniform(5.0, 7.0))
    elif archetype == "mega_extreme":
        num_stories = int(rng.integers(25, 35))
        bays_x = int(rng.integers(6, 9))
        bays_y = int(rng.integers(5, 8))
        span_x = float(rng.uniform(5.0, 7.0))
        span_y = float(rng.uniform(5.0, 7.0))
    elif archetype == "triple_twin_mega_cluster":
        num_stories = int(rng.integers(15, 25))
        bays_x = int(rng.integers(5, 7))
        bays_y = int(rng.integers(5, 7))
        span_x = float(rng.uniform(5.0, 7.0))
        span_y = float(rng.uniform(5.0, 7.0))
    elif archetype == "twisted_quad_bridge_cluster":
        num_stories = int(rng.integers(10, 18))
        bays_x = int(rng.integers(4, 6))
        bays_y = int(rng.integers(4, 6))
        span_x = float(rng.uniform(5.0, 7.0))
        span_y = float(rng.uniform(5.0, 7.0))
    elif archetype == "two_d_branch":
        num_stories = int(rng.integers(4, 12))
        bays_x = int(rng.integers(2, 4)) if rng.random() < 0.5 else int(rng.integers(6, 10))
        bays_y = int(rng.integers(6, 10)) if bays_x <= 4 else int(rng.integers(2, 4))
        span_x = float(rng.uniform(4.0, 8.0))
        span_y = float(rng.uniform(4.0, 8.0))
    else:
        num_stories = int(rng.integers(STORY_RANGE[0], STORY_RANGE[1] + 1))
        bays_x = int(rng.integers(BAYS_X_RANGE[0], BAYS_X_RANGE[1] + 1))
        bays_y = int(rng.integers(BAYS_Y_RANGE[0], BAYS_Y_RANGE[1] + 1))
        span_x = float(rng.uniform(SPAN_X_RANGE[0], SPAN_X_RANGE[1]))
        span_y = float(rng.uniform(SPAN_Y_RANGE[0], SPAN_Y_RANGE[1]))

    lines_x = bays_x + 1
    lines_y = bays_y + 1
    story_height = float(rng.uniform(STORY_HEIGHT_RANGE[0], STORY_HEIGHT_RANGE[1]))
    bottom_mult = float(rng.uniform(BOTTOM_STORY_MULT[0], BOTTOM_STORY_MULT[1]))
    bottom_story_height = story_height * bottom_mult
    dead_kpa = float(rng.uniform(DEAD_KPA_RANGE[0], DEAD_KPA_RANGE[1]))
    live_kpa = float(rng.uniform(LIVE_KPA_RANGE[0], LIVE_KPA_RANGE[1]))
    total_h = bottom_story_height + (num_stories - 1) * story_height
    plan_x = bays_x * span_x
    plan_y = bays_y * span_y

    brace_x = 1.0 if archetype == "hybrid_braced" and rng.random() < 0.7 else (1.0 if rng.random() < 0.25 else 0.0)
    brace_y = 1.0 if brace_x > 0 and rng.random() < 0.6 else 0.0
    brace_density = float(rng.uniform(0.2, 0.8)) if brace_x > 0 else float(rng.uniform(0.01, 0.15))
    snow_kpa = float(rng.uniform(0.5, 2.0)) if rng.random() < 0.4 else 0.0
    temp_delta = float(rng.uniform(15.0, 35.0))
    cantilever_fraction = float(rng.uniform(0.0, 0.15)) if rng.random() < 0.2 else 0.0
    cantilever_length = float(rng.uniform(1.0, 3.0)) if cantilever_fraction > 0 else 0.0
    plan_skew = float(rng.uniform(-8.0, 8.0)) if rng.random() < 0.15 else 0.0
    twist_deg = float(rng.uniform(-5.0, 5.0)) if rng.random() < 0.1 else 0.0
    lean_m = float(rng.uniform(0.0, 0.5)) if rng.random() < 0.08 else 0.0
    setback_ratio = float(rng.uniform(0.1, 0.3)) if archetype == "setback_tower" and rng.random() < 0.6 else (float(rng.uniform(0.0, 0.15)) if rng.random() < 0.1 else 0.0)
    beam_release_frac = float(rng.uniform(0.02, 0.12)) if rng.random() < 0.15 else 0.02
    beam_release_mode = 2.0 if beam_release_frac > 0.05 else 1.0

    return {
        "stories": float(num_stories),
        "bays_x": float(bays_x),
        "bays_y": float(bays_y),
        "lines_x": float(lines_x),
        "lines_y": float(lines_y),
        "span_x_m": span_x,
        "span_y_m": span_y,
        "story_height_m": story_height,
        "bottom_story_height_m": bottom_story_height,
        "total_height_m": total_h,
        "dead_kpa": dead_kpa,
        "live_kpa": live_kpa,
        "plan_x_m": plan_x,
        "plan_y_m": plan_y,
        "archetype": archetype,
        "brace_x": brace_x,
        "brace_y": brace_y,
        "brace_density": brace_density,
        "snow_kpa": snow_kpa,
        "temp_delta_c": temp_delta,
        "cantilever_fraction": cantilever_fraction,
        "cantilever_length_m": cantilever_length,
        "plan_skew_deg": plan_skew,
        "twist_total_deg": twist_deg,
        "lean_total_m": lean_m,
        "setback_ratio": setback_ratio,
        "beam_release_fraction": beam_release_frac,
        "beam_release_mode_id": beam_release_mode,
    }


def estimate_generation_only_fields(row: Dict[str, Any]) -> Dict[str, float]:
    """Fill proxy/calculated fields from geometry (for ETABS-generated rows)."""
    out = dict(row)
    plan_x = _get(row, "plan_x_m", 20.0)
    plan_y = _get(row, "plan_y_m", 20.0)
    plan_area = plan_x * plan_y if plan_x > 0 and plan_y > 0 else 400.0
    story_h = _get(row, "story_height_m", 3.0)
    total_h = _get(row, "total_height_m", 30.0)
    stories = _get(row, "stories", 5.0)
    bays_x = _get(row, "bays_x", 4.0)
    bays_y = _get(row, "bays_y", 4.0)
    span_x = _get(row, "span_x_m", 6.0)
    span_y = _get(row, "span_y_m", 6.0)
    dead_kpa = _get(row, "super_dead_kpa", row.get("dead_kpa", 2.5))
    live_kpa = _get(row, "live_kpa", 3.0)
    selfwt = 1.8

    slender_x = total_h / plan_x if plan_x > 0 else 0.0
    slender_y = total_h / plan_y if plan_y > 0 else 0.0
    aspect_ratio = plan_x / plan_y if plan_y > 0 else 1.0
    aero_slen = (slender_x + slender_y) / 2.0

    E = 200e6
    G = 77e6
    alpha = 1.2e-5
    fy_kpa = 345000.0

    beam_d = max(max(span_x, span_y) / 18.0, 0.45)
    col_d = max(0.70, 0.018 * total_h + 0.03 * max(span_x, span_y) * math.sqrt(max(stories, 1)))
    brace_d = max(0.35, 0.65 * beam_d)

    def _i_area(d: float, bf: float, tf: float, tw: float) -> float:
        return 2.0 * bf * tf + max(d - 2.0 * tf, 0.0) * tw

    def _i_Ix(d: float, bf: float, tf: float, tw: float) -> float:
        wh = max(d - 2.0 * tf, 0.0)
        return tw * wh**3 / 12.0 + 2.0 * (bf * tf**3 / 12.0 + bf * tf * (d/2 - tf/2)**2)

    def _i_Iy(d: float, bf: float, tf: float, tw: float) -> float:
        wh = max(d - 2.0 * tf, 0.0)
        return wh * tw**3 / 12.0 + 2.0 * (tf * bf**3 / 12.0)

    def _i_J(d: float, bf: float, tf: float, tw: float) -> float:
        wh = max(d - 2.0 * tf, 0.0)
        return (2.0 * bf * tf**3 + wh * tw**3) / 3.0

    bf_b = 0.45 * beam_d
    tf_b = max(0.018, min(0.090, 0.04 * beam_d))
    tw_b = max(0.010, min(0.045, 0.02 * beam_d))
    A_beam = _i_area(beam_d, bf_b, tf_b, tw_b)
    I_beam = _i_Ix(beam_d, bf_b, tf_b, tw_b)
    Iy_beam = _i_Iy(beam_d, bf_b, tf_b, tw_b)
    J_beam = _i_J(beam_d, bf_b, tf_b, tw_b)

    bf_c = 0.45 * col_d
    tf_c = max(0.018, min(0.090, 0.04 * col_d))
    tw_c = max(0.010, min(0.045, 0.02 * col_d))
    A_col = _i_area(col_d, bf_c, tf_c, tw_c)
    I_col = _i_Ix(col_d, bf_c, tf_c, tw_c)
    Iy_col = _i_Iy(col_d, bf_c, tf_c, tw_c)
    J_col = _i_J(col_d, bf_c, tf_c, tw_c)

    A_br = A_beam * 0.5
    I_br = I_beam * 0.3
    Iy_br = Iy_beam * 0.3
    J_br = J_beam * 0.3

    total_weight = plan_area * ((selfwt + dead_kpa + 0.25 * live_kpa) * max(stories - 1.0, 0.0) +
                                (selfwt + dead_kpa + 0.25 * live_kpa))
    total_mass = total_weight / 9.81

    wind_x, wind_y = 0.03, 0.025
    eq_x, eq_y = 0.08, 0.07
    Vwx = total_weight * wind_x
    Vwy = total_weight * wind_y
    Veqx = total_weight * eq_x
    Veqy = total_weight * eq_y

    perimeter = 2.0 * (bays_x + bays_y + 2.0)
    k_story = perimeter * 12.0 * E * I_col / max(story_h**3, 1e-9)
    k_total = max(k_story / max(stories, 1.0), 1e-9)
    drift_wx = Vwx / k_total
    drift_wy = Vwy / k_total
    drift_eqx = Veqx / k_total
    drift_eqy = Veqy / k_total

    beam_line = (dead_kpa + live_kpa) * max(span_x, span_y)
    beam_m_proxy = beam_line * max(span_x, span_y)**2 / 8.0
    beam_defl = 5.0 * beam_line * max(span_x, span_y)**4 / max(384.0 * E * I_beam, 1e-9)
    n_cols = (bays_x + 1.0) * (bays_y + 1.0)
    col_p_proxy = total_weight / max(n_cols, 1.0)
    brace_p_proxy = max(Veqx, Veqy) / 4.0
    Pcr = math.pi**2 * E * I_col / max(story_h**2, 1e-9)
    buckling_ratio = col_p_proxy / max(Pcr, 1e-9)
    temp_delta = 20.0
    temp_axial = E * A_col * alpha * temp_delta
    Tx = 2.0 * math.pi * math.sqrt(max(total_mass / k_total, 1e-12))
    Ty = Tx
    Tt = 2.0 * math.pi * math.sqrt(max(total_mass * (plan_x**2 + plan_y**2) / max(E * J_col, 1e-9), 1e-12))
    thermal_exp = alpha * temp_delta * total_h
    thermal_strain = alpha * temp_delta
    thermal_energy = 0.5 * E * thermal_strain**2 / 1000.0
    thermal_stress = E * thermal_strain
    thermal_stress_ratio = thermal_stress / max(fy_kpa, 1e-9)
    beam_myield = fy_kpa * (I_beam / max(beam_d / 2.0, 1e-9))
    beam_plastic_ratio = beam_m_proxy / max(beam_myield, 1e-9)
    col_axial_yield_ratio = col_p_proxy / max(A_col * fy_kpa, 1e-9)
    beam_slender = max(span_x, span_y) / max(beam_d, 1e-9)
    col_slender = story_h / max(col_d, 1e-9)
    ecc = 0.05 * max(plan_x, plan_y)
    theta_w = Vwx * ecc / max(E * J_col, 1e-9)
    theta_eq = Veqx * ecc / max(E * J_col, 1e-9)
    theta_blast = 0.03 * total_weight * ecc / max(E * J_col, 1e-9)
    story_shear_grad = max(Veqx, Veqy) / max(stories, 1.0)
    strain_energy = 0.5 * total_weight * max(drift_eqx, drift_eqy)
    p_delta = col_p_proxy * max(drift_eqx, drift_eqy) / max(beam_m_proxy, 1e-9)
    shear_flex = max(span_x, span_y) / max(A_beam, 1e-9) * 1e-3
    axial_short = col_p_proxy * total_h / max(E * A_col, 1e-9)
    mass_irreg = 1.0
    modal_density = stories / max(Tx + Ty + Tt, 1e-9)
    support_fixity = 1.0
    diaphragm_constraint = 1.0

    moment_magnification_B1 = 1.0 / max(1.0 - col_p_proxy / max(Pcr, 1e-9), 0.2)
    shear_deformation_ratio = (E / G) * (I_beam / max(A_beam, 1e-9)) / max(max(span_x, span_y)**2, 1e-9)
    overturning_moment_index = max(Veqx, Veqy) * total_h / max(total_weight * max(plan_x, plan_y) * 0.5, 1e-9)
    story_drift_regularity = 1.0 - min(0.5, story_shear_grad / max(max(Veqx, Veqy), 1e-9) * story_h)
    foundation_flexibility = 1.0 - 0.3 * (1.0 - support_fixity)
    second_order_amplification = 1.0 / max(1.0 - buckling_ratio * 0.5, 0.3)
    diaphragm_shear_flow = diaphragm_constraint * math.sqrt(plan_area) / max(total_h, 1e-9)
    modal_participation = 0.85 + 0.15 * (1.0 - mass_irreg / 2.0)

    out.update({
        "plan_area_m2": plan_area, "slender_x": slender_x, "slender_y": slender_y,
        "aspect_ratio": aspect_ratio, "aerodynamic_slenderness": aero_slen,
        "A_beam_m2": A_beam, "I_beam_m4": I_beam, "Iy_beam_m4": Iy_beam, "J_beam_m4": J_beam,
        "A_col_m2": A_col, "I_col_m4": I_col, "Iy_col_m4": Iy_col, "J_col_m4": J_col,
        "A_br_m2": A_br, "I_br_m4": I_br, "Iy_br_m4": Iy_br, "J_br_m4": J_br,
        "total_weight_kN": total_weight, "total_mass_kNs2_m": total_mass,
        "V_wx_proxy_kN": Vwx, "V_wy_proxy_kN": Vwy, "V_eqx_proxy_kN": Veqx, "V_eqy_proxy_kN": Veqy,
        "Vwx_assigned_kN": Vwx, "Vwy_assigned_kN": Vwy, "Veqx_assigned_kN": Veqx, "Veqy_assigned_kN": Veqy,
        "wind_coeff_x": wind_x, "wind_coeff_y": wind_y, "eq_coeff_x": eq_x, "eq_coeff_y": eq_y,
        "drift_proxy_wx_m": drift_wx, "drift_proxy_wy_m": drift_wy,
        "drift_proxy_eqx_m": drift_eqx, "drift_proxy_eqy_m": drift_eqy,
        "beam_m_proxy_kNm": beam_m_proxy, "col_p_proxy_kN": col_p_proxy, "brace_p_proxy_kN": brace_p_proxy,
        "temp_axial_proxy_kN": temp_axial, "Pcr_proxy_kN": Pcr, "buckling_ratio": buckling_ratio,
        "T_proxy_x_s": Tx, "T_proxy_y_s": Ty, "T_torsion_proxy_s": Tt,
        "thermal_free_expansion_m": thermal_exp, "temp_delta_c": temp_delta,
        "thermal_strain": thermal_strain, "thermal_energy_density_kNm_m3": thermal_energy,
        "thermal_stress_kPa": thermal_stress, "thermal_stress_ratio": thermal_stress_ratio,
        "beam_defl_proxy_m": beam_defl, "beam_myield_proxy_kNm": beam_myield,
        "beam_plastic_ratio": beam_plastic_ratio, "col_axial_yield_ratio": col_axial_yield_ratio,
        "beam_slenderness_proxy": beam_slender, "col_slenderness_proxy": col_slender,
        "strain_energy_proxy_kNm": strain_energy, "p_delta_index_proxy": p_delta,
        "shear_flexibility_proxy": shear_flex, "axial_shortening_proxy_m": axial_short,
        "mass_irregularity_proxy": mass_irreg, "modal_density_proxy": modal_density,
        "story_shear_gradient_proxy": story_shear_grad,
        "torsion_theta_w_rad": theta_w, "torsion_theta_eq_rad": theta_eq,
        "torsion_theta_blast_rad": theta_blast,
        "V_imp_x_proxy_kN": total_weight * 0.02, "V_imp_y_proxy_kN": total_weight * 0.02,
        "V_blast_x_proxy_kN": total_weight * 0.03, "V_blast_y_proxy_kN": total_weight * 0.03,
        "drift_proxy_impx_m": drift_wx * 0.5, "drift_proxy_impy_m": drift_wy * 0.5,
        "drift_proxy_blastx_m": drift_eqx * 0.3, "drift_proxy_blasty_m": drift_eqy * 0.3,
        "cantilever_m_proxy_kNm": 0.0, "cantilever_defl_proxy_m": 0.0,
        "impact_coeff_x": 0.02, "impact_coeff_y": 0.02, "blast_coeff_x": 0.03, "blast_coeff_y": 0.03,
        "Vimpx_assigned_kN": total_weight * 0.02, "Vimpy_assigned_kN": total_weight * 0.02,
        "Vblastx_assigned_kN": total_weight * 0.03, "Vblasty_assigned_kN": total_weight * 0.03,
        "moment_magnification_B1_proxy": moment_magnification_B1,
        "shear_deformation_ratio_proxy": shear_deformation_ratio,
        "overturning_moment_index": overturning_moment_index,
        "story_drift_regularity_proxy": story_drift_regularity,
        "foundation_flexibility_proxy": foundation_flexibility,
        "second_order_amplification_proxy": second_order_amplification,
        "diaphragm_shear_flow_index": diaphragm_shear_flow,
        "modal_participation_proxy": modal_participation,
        "support_mode_id": 1.0, "diaphragm_mode_id": 1.0,
        "support_fixity_index": support_fixity, "diaphragm_constraint_index": diaphragm_constraint,
        "brace_x": 0.0, "brace_y": 0.0, "brace_density": 0.01, "beam_scale": 1.0, "col_scale": 1.0, "brace_scale": 0.0,
        "beam_stiffness_modifier": 1.0, "col_stiffness_modifier": 1.0, "brace_stiffness_modifier": 1.0,
        "snow_kpa": 0.0, "selfwt_floor_kpa": selfwt,
        "super_dead_kpa": dead_kpa, "plan_skew_deg": 0.0, "twist_total_deg": 0.0, "lean_total_m": 0.0,
        "setback_ratio": 0.0, "eccentricity_ratio": 0.0, "aero_shape_factor": 1.0,
        "tower_gap_m": 0.0, "podium_levels": 0.0, "shape_irregularity_index": 0.2,
        "out_of_plane_irregularity": 0.0, "structural_system_id": 2.0,
        "floor_stiffness_gradient": 1.0, "in_floor_stiffness_variation": 0.08, "story_mass_gradient": 1.0,
        "live_impact_factor": 1.05, "site_class_factor": 1.0, "damping_ratio_model": 0.02,
        "response_mod_factor": 3.5, "redundancy_factor": 1.0, "panel_zone_flexibility": 0.12,
        "snow_drift_factor": 1.0, "ponding_factor": 1.0, "mass_source_factor": 1.0,
        "live_load_reduction_factor": 0.9 if stories >= 10 else 1.0, "wind_attack_angle_deg": 0.0,
        "k_factor_col": 1.0, "soft_story_factor": 1.0,
        "floor_stiffness_variation_index": 0.0, "support_irregularity_index": 0.0,
        "diaphragm_irregularity_index": 0.0, "blast_impulse_proxy_kN_m": Vwx * story_h,
        "impact_energy_proxy_kNm": Vwx * story_h, "robustness_proxy": 1.0, "cantilever_ratio": 0.0,
        "lateral_torsional_index": beam_m_proxy / max(E * Iy_beam, 1e-9),
        "skybridge_coupling_index": 0.0, "cluster_irregularity_proxy": 1.0,
        "beam_release_fraction": 0.02, "beam_release_mode_id": 1.0,
        "cluster_tower_count": 1.0, "bridge_levels_n": 0.0, "bridge_links_per_level": 0.0,
        "bridge_span_m": 0.0, "bridge_members_added": 0.0, "cantilever_fraction": 0.0,
        "cantilever_length_m": 0.0, "cantilever_levels_n": 0.0, "cantilever_members_added": 0.0,
        "fixed_support_points_n": n_cols, "pinned_support_points_n": 0.0,
        "diaphragm_assigned_n": max(1.0, stories), "transformed_points_n": n_cols * (stories + 1.0),
        "released_beams_n": 0.0,
    })
    return out


def build_physics_features(row: Dict[str, Any]) -> Dict[str, float]:
    """Build full physics feature dict. Adds 8 new physics computations."""
    out = dict(row)
    pcr_proxy = _get(row, "Pcr_proxy_kN", 1e9)
    col_p_proxy = _get(row, "col_p_proxy_kN", 1e3)
    drift_eqx = _get(row, "drift_proxy_eqx_m", 0.001)
    drift_eqy = _get(row, "drift_proxy_eqy_m", 0.001)
    veqx = _get(row, "V_eqx_proxy_kN", 100.0)
    veqy = _get(row, "V_eqy_proxy_kN", 100.0)
    total_weight = _get(row, "total_weight_kN", 1000.0)
    plan_x = _get(row, "plan_x_m", 20.0)
    plan_y = _get(row, "plan_y_m", 20.0)
    story_h = _get(row, "story_height_m", 3.0)
    h_total = _get(row, "total_height_m", 30.0)
    plan_area = _get(row, "plan_area_m2", 400.0)
    sx = _get(row, "span_x_m", 6.0)
    sy = _get(row, "span_y_m", 6.0)
    a_beam = _get(row, "A_beam_m2", 0.01)
    i_beam = _get(row, "I_beam_m4", 1e-5)
    story_shear_gradient_proxy = _get(row, "story_shear_gradient_proxy", 100.0)
    support_fixity_index = _get(row, "support_fixity_index", 1.0)
    buckling_ratio = _get(row, "buckling_ratio", 0.1)
    diaphragm_constraint_index = _get(row, "diaphragm_constraint_index", 1.0)
    mass_irregularity_proxy = _get(row, "mass_irregularity_proxy", 1.0)
    e_mod, g_mod = 200e6, 77e6

    moment_magnification_B1_proxy = 1.0 / max(1.0 - col_p_proxy / max(pcr_proxy, 1e-9), 0.2)
    shear_deformation_ratio_proxy = (e_mod / g_mod) * (i_beam / max(a_beam, 1e-9)) / max(max(sx, sy)**2, 1e-9)
    overturning_moment_index = max(veqx, veqy) * h_total / max(total_weight * max(plan_x, plan_y) * 0.5, 1e-9)
    story_drift_regularity_proxy = 1.0 - min(0.5, story_shear_gradient_proxy / max(max(veqx, veqy), 1e-9) * story_h)
    foundation_flexibility_proxy = 1.0 - 0.3 * (1.0 - support_fixity_index)
    second_order_amplification_proxy = 1.0 / max(1.0 - buckling_ratio * 0.5, 0.3)
    diaphragm_shear_flow_index = diaphragm_constraint_index * math.sqrt(plan_area) / max(h_total, 1e-9)
    modal_participation_proxy = 0.85 + 0.15 * (1.0 - mass_irregularity_proxy / 2.0)

    out["moment_magnification_B1_proxy"] = float(moment_magnification_B1_proxy)
    out["shear_deformation_ratio_proxy"] = float(shear_deformation_ratio_proxy)
    out["overturning_moment_index"] = float(overturning_moment_index)
    out["story_drift_regularity_proxy"] = float(story_drift_regularity_proxy)
    out["foundation_flexibility_proxy"] = float(foundation_flexibility_proxy)
    out["second_order_amplification_proxy"] = float(second_order_amplification_proxy)
    out["diaphragm_shear_flow_index"] = float(diaphragm_shear_flow_index)
    out["modal_participation_proxy"] = float(modal_participation_proxy)
    return out


# --- StructuralBrain: 448 width, 3 res blocks ---
class ResidualBlock(nn.Module):
    def __init__(self, width: int, dropout: float = 0.0) -> None:
        super().__init__()
        self.fc1 = nn.Linear(width, width)
        self.fc2 = nn.Linear(width, width)
        self.act = nn.GELU()
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = self.act(self.fc1(x))
        x = self.dropout(self.fc2(x))
        return self.act(x + residual)


class StructuralBrain(nn.Module):
    """448-width network with 3 residual blocks."""
    def __init__(self, in_dim: int, out_dim: int, width: int = 448, head_dim: int = 224, dropout: float = 0.05) -> None:
        super().__init__()
        self.embed = nn.Sequential(
            nn.Linear(in_dim, width),
            nn.GELU(),
            nn.Linear(width, width),
            nn.GELU(),
        )
        self.res1 = ResidualBlock(width, dropout=dropout)
        self.res2 = ResidualBlock(width, dropout=dropout)
        self.res3 = ResidualBlock(width, dropout=dropout)
        self.head = nn.Sequential(
            nn.Linear(width, head_dim),
            nn.GELU(),
            nn.Linear(head_dim, out_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.embed(x)
        x = self.res1(x)
        x = self.res2(x)
        x = self.res3(x)
        return self.head(x)


def kill_stale_etabs() -> None:
    """Kill any running ETABS processes."""
    try:
        import psutil
        for proc in psutil.process_iter(["name", "pid"]):
            try:
                name = (proc.info.get("name") or "").lower()
                if "etabs" in name or "sap" in name:
                    proc.terminate()
                    proc.wait(timeout=5)
                    print(f"  Killed stale process: {name} (PID {proc.info.get('pid')})")
            except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.TimeoutExpired):
                pass
    except ImportError:
        pass


def connect_etabs() -> Any:
    """Connect to ETABS via comtypes. Returns SapModel."""
    try:
        import comtypes.client
    except ImportError:
        raise ImportError("Install comtypes: pip install comtypes")
    helper = comtypes.client.CreateObject("ETABSv1.Helper")
    try:
        helper = helper.QueryInterface(comtypes.gen.ETABSv1.cHelper)
    except AttributeError:
        pass
    etabs_obj = helper.CreateObjectProgID("CSI.ETABS.API.ETABSObject")
    etabs_obj.ApplicationStart()
    return etabs_obj.SapModel


def set_units(sap_model: Any, units: int = ETABS_UNITS_KNM_C) -> int:
    return sap_model.SetPresentUnits(units)


def api_ok(ret: int) -> bool:
    return ret == 0


def extract_ret(res: Any, default: Any = None) -> Any:
    if res is None:
        return default
    if isinstance(res, (tuple, list)) and len(res) >= 1:
        return res[0] if res[0] == 0 else res
    return res


def unpack_name_and_ret(res: Any) -> Tuple[Any, Any]:
    if res is None or not isinstance(res, (tuple, list)):
        return None, -1
    if len(res) >= 2:
        return res[1], res[0]
    return res[0] if len(res) > 0 else None, -1


def get_all_frames(sap_model: Any) -> List[str]:
    try:
        ret, names = sap_model.FrameObj.GetAllNames()
        if api_ok(ret) and names:
            return list(names) if isinstance(names, (list, tuple)) else [names]
    except Exception:
        pass
    return []


def get_coord_cartesian(sap_model: Any, frame_name: str) -> Optional[Tuple[float, float, float, float, float, float]]:
    try:
        ret, xi, yi, zi, xj, yj, zj = sap_model.FrameObj.GetCoordCartesian(frame_name, 0, 0, 0, 0, 0, 0)
        if api_ok(ret):
            return (xi, yi, zi, xj, yj, zj)
    except Exception:
        pass
    return None


def get_frame_force_group(sap_model: Any, case_name: str) -> Dict[str, List[float]]:
    out: Dict[str, List[float]] = {"P": [], "V2": [], "V3": [], "M2": [], "M3": []}
    try:
        res = sap_model.Results.FrameForce("", case_name, 0)
        if not res or not isinstance(res, (tuple, list)) or len(res) < 11:
            return out
        out["P"] = list(res[5]) if hasattr(res[5], "__iter__") and res[5] else []
        out["V2"] = list(res[6]) if hasattr(res[6], "__iter__") and res[6] else []
        out["V3"] = list(res[7]) if hasattr(res[7], "__iter__") and res[7] else []
        out["M2"] = list(res[9]) if hasattr(res[9], "__iter__") and res[9] else []
        out["M3"] = list(res[10]) if hasattr(res[10], "__iter__") and res[10] else []
    except Exception:
        pass
    return out


def get_base_reaction(sap_model: Any, case_name: str) -> Tuple[float, float, float]:
    try:
        res = sap_model.Results.BaseReact(case_name, 0)
        if res and isinstance(res, (tuple, list)):
            if len(res) >= 7:
                return (float(res[3] or 0), float(res[4] or 0), float(res[5] or 0))
            if len(res) >= 3:
                return (float(res[0] or 0), float(res[1] or 0), float(res[2] or 0))
    except Exception:
        pass
    return 0.0, 0.0, 0.0


def get_joint_displ(sap_model: Any, case_name: str) -> float:
    try:
        res = sap_model.Results.JointDispl("", case_name, 0)
        if res and isinstance(res, (tuple, list)) and len(res) >= 8:
            u3 = res[7] if len(res) > 7 else (res[5] if len(res) > 5 else [])
            if hasattr(u3, "__iter__") and not isinstance(u3, str):
                vals = [abs(float(x)) for x in u3]
                return max(vals) if vals else 0.0
    except Exception:
        pass
    return 0.0


def select_case_for_output(sap_model: Any, case_name: str) -> None:
    try:
        sap_model.Results.Setup.DeselectAllCasesAndCombosForOutput()
        sap_model.Results.Setup.SetCaseSelectedForOutput(case_name, True)
    except Exception:
        pass


def safe_save_model(sap_model: Any, path: Path) -> bool:
    try:
        return sap_model.File.Save(str(path)) == 0
    except Exception:
        return False


def initialize_new_model(sap_model: Any) -> int:
    return sap_model.InitializeNewModel()


def new_steel_deck(sap_model: Any, num_stories: int, story_height: float, bottom_story_height: float,
                   num_lines_x: int, num_lines_y: int, bay_width_x: float, bay_width_y: float) -> int:
    try:
        return sap_model.File.NewSteelDeck(
            num_stories, story_height, bottom_story_height,
            num_lines_x, num_lines_y, bay_width_x, bay_width_y,
        )
    except Exception as e:
        print(f"  NewSteelDeck failed: {e}")
        return -1


def add_load_patterns_and_cases(sap_model: Any, dead_kpa: float, live_kpa: float,
                                 snow_kpa: float = 0.0, wind_x: float = 0.03, wind_y: float = 0.025,
                                 eq_x: float = 0.08, eq_y: float = 0.07, temp_delta: float = 20.0,
                                 imp_x: float = 0.02, imp_y: float = 0.02,
                                 blast_x: float = 0.03, blast_y: float = 0.03) -> bool:
    """Add full load patterns: DEAD, LIVE, SNOW, WX, WY, EQX, EQY, TEMP, IMPX, IMPY, BLASTX, BLASTY."""
    patterns = [
        ("DEAD", LOAD_PAT_DEAD),
        ("LIVE", LOAD_PAT_LIVE),
        ("SNOW", LOAD_PAT_SNOW),
        ("WX", LOAD_PAT_WIND),
        ("WY", LOAD_PAT_WIND),
        ("EQX", LOAD_PAT_QUAKE),
        ("EQY", LOAD_PAT_QUAKE),
        ("TEMP", LOAD_PAT_TEMP),
        ("IMPX", LOAD_PAT_OTHER),
        ("IMPY", LOAD_PAT_OTHER),
        ("BLASTX", LOAD_PAT_OTHER),
        ("BLASTY", LOAD_PAT_OTHER),
    ]
    for name, ptype in patterns:
        if sap_model.LoadPatterns.Add(name, ptype) != 0:
            pass
    for lc in ["DEAD", "LIVE", "SNOW", "WX", "WY", "EQX", "EQY", "TEMP", "IMPX", "IMPY", "BLASTX", "BLASTY"]:
        try:
            sap_model.LoadCases.StaticLinear.SetCase(lc)
        except Exception:
            pass
    return True


def assign_area_load_to_floors(sap_model: Any, dead_kpa: float, live_kpa: float, snow_kpa: float = 0.0) -> bool:
    try:
        ret, areas = sap_model.AreaObj.GetAllNames()
        names = areas if isinstance(areas, (list, tuple)) else ([areas] if areas else [])
        if not names:
            return True
        for name in names:
            try:
                sap_model.AreaObj.SetLoadUniform(name, "DEAD", dead_kpa, 3, True)
                sap_model.AreaObj.SetLoadUniform(name, "LIVE", live_kpa, 3, True)
                if snow_kpa > 0:
                    sap_model.AreaObj.SetLoadUniform(name, "SNOW", snow_kpa, 3, True)
            except Exception:
                pass
        return True
    except Exception:
        return True


def assign_frame_load_to_beams(sap_model: Any, dead_kpa: float, live_kpa: float, snow_kpa: float = 0.0) -> bool:
    try:
        ret, names = sap_model.FrameObj.GetAllNames()
        frame_list = names if isinstance(names, (list, tuple)) else ([names] if names else [])
        w_dead, w_live = dead_kpa * 3.0, live_kpa * 3.0
        w_snow = snow_kpa * 3.0 if snow_kpa > 0 else 0.0
        for name in frame_list:
            try:
                sap_model.FrameObj.SetLoadDistributed(name, "DEAD", 1, w_dead, w_dead, 0, 1, "Local")
                sap_model.FrameObj.SetLoadDistributed(name, "LIVE", 1, w_live, w_live, 0, 1, "Local")
                if w_snow > 0:
                    sap_model.FrameObj.SetLoadDistributed(name, "SNOW", 1, w_snow, w_snow, 0, 1, "Local")
            except Exception:
                pass
        return True
    except Exception:
        return True


def assign_lateral_loads(sap_model: Any, total_weight: float, story_heights: List[float],
                         v_wx: float, v_wy: float, v_eqx: float, v_eqy: float,
                         v_impx: float = 0.0, v_impy: float = 0.0,
                         v_blastx: float = 0.0, v_blasty: float = 0.0) -> bool:
    """Assign wind/earthquake/impact/blast lateral loads by height (inverse triangular)."""
    if not story_heights:
        return True
    try:
        ret, points = sap_model.PointObj.GetAllNames()
        pt_list = points if isinstance(points, (list, tuple)) else ([points] if points else [])
        if not pt_list:
            return True
        h_total = sum(story_heights)
        if h_total <= 0:
            return True
        z_levels = [0.0]
        for h in story_heights:
            z_levels.append(z_levels[-1] + h)
        for pt in pt_list:
            try:
                _, _, _, z = sap_model.PointObj.GetCoordCartesian(pt, 0, 0, 0)
                z_val = float(z) if z is not None else 0.0
                factor = (z_val / h_total) if h_total > 0 else 0.5
                n_stories = len(story_heights)
                frac = (factor * 2.0 / (n_stories + 1)) if n_stories > 0 else 1.0
                if v_wx > 0:
                    sap_model.PointObj.SetLoadForce(pt, "WX", 1, v_wx * frac, 0, 0, 0, 0, 0)
                if v_wy > 0:
                    sap_model.PointObj.SetLoadForce(pt, "WY", 2, v_wy * frac, 0, 0, 0, 0, 0)
                if v_eqx > 0:
                    sap_model.PointObj.SetLoadForce(pt, "EQX", 1, v_eqx * frac, 0, 0, 0, 0, 0)
                if v_eqy > 0:
                    sap_model.PointObj.SetLoadForce(pt, "EQY", 2, v_eqy * frac, 0, 0, 0, 0, 0)
            except Exception:
                pass
        return True
    except Exception:
        return True


def assign_temperature_load(sap_model: Any, temp_delta: float = 20.0) -> bool:
    try:
        ret, names = sap_model.FrameObj.GetAllNames()
        frame_list = names if isinstance(names, (list, tuple)) else ([names] if names else [])
        for name in frame_list:
            try:
                sap_model.FrameObj.SetLoadTemperature(name, "TEMP", 1, temp_delta)
            except Exception:
                pass
        return True
    except Exception:
        return True


def assign_impact_and_blast_loads(sap_model: Any, v_impx: float, v_impy: float,
                                  v_blastx: float, v_blasty: float,
                                  story_heights: List[float]) -> bool:
    """Assign impact and blast lateral loads at floor levels."""
    if not story_heights or (v_impx <= 0 and v_impy <= 0 and v_blastx <= 0 and v_blasty <= 0):
        return True
    try:
        ret, points = sap_model.PointObj.GetAllNames()
        pt_list = points if isinstance(points, (list, tuple)) else ([points] if points else [])
        if not pt_list:
            return True
        n_levels = len(story_heights) + 1
        frac = 1.0 / max(n_levels, 1)
        for pt in pt_list[:min(len(pt_list), 50)]:
            try:
                if v_impx > 0:
                    sap_model.PointObj.SetLoadForce(pt, "IMPX", 1, v_impx * frac, 0, 0, 0, 0, 0)
                if v_impy > 0:
                    sap_model.PointObj.SetLoadForce(pt, "IMPY", 2, v_impy * frac, 0, 0, 0, 0, 0)
                if v_blastx > 0:
                    sap_model.PointObj.SetLoadForce(pt, "BLASTX", 1, v_blastx * frac, 0, 0, 0, 0, 0)
                if v_blasty > 0:
                    sap_model.PointObj.SetLoadForce(pt, "BLASTY", 2, v_blasty * frac, 0, 0, 0, 0, 0)
            except Exception:
                pass
        return True
    except Exception:
        return True


def section_pack_from_params(beam_d: float, col_d: float, brace_d: float,
                              beam_scale: float, col_scale: float, brace_scale: float) -> Dict[str, Tuple[float, float, float, float]]:
    """Compute I-section params (d, bf, tf, tw) from depths and scales."""
    def _make(d: float, scale: float) -> Tuple[float, float, float, float]:
        d = d * scale
        bf = 0.45 * d
        tf = max(0.018, min(0.090, 0.04 * d))
        tw = max(0.010, min(0.045, 0.02 * d))
        return (d, bf, tf, tw)
    return {
        "beam": _make(beam_d, beam_scale),
        "col": _make(col_d, col_scale),
        "brace": _make(brace_d, brace_scale),
    }


def define_sections(sap_model: Any, section_pack: Dict[str, Tuple[float, float, float, float]]) -> bool:
    """Define beam, col, brace I-sections in ETABS."""
    try:
        prop_frame = sap_model.PropFrame
        for name, (d, bf, tf, tw) in [("BeamSec", section_pack["beam"]),
                                        ("ColSec", section_pack["col"]),
                                        ("BraceSec", section_pack["brace"])]:
            if prop_frame.SetISection(name, "Steel", d, bf, tf, tw, bf, tf, tw) != 0:
                pass
        return True
    except Exception:
        return True


def classify_and_assign_frame_sections(sap_model: Any, beam_sec: str, col_sec: str, brace_sec: str) -> bool:
    """Classify frames as beam/column/brace and assign sections. Heuristic: vertical=col, diagonal=brace, else beam."""
    try:
        ret, names = sap_model.FrameObj.GetAllNames()
        frame_list = names if isinstance(names, (list, tuple)) else ([names] if names else [])
        for name in frame_list:
            try:
                _, xi, yi, zi, xj, yj, zj = sap_model.FrameObj.GetCoordCartesian(name, 0, 0, 0, 0, 0, 0)
                dz = abs(float(zj or 0) - float(zi or 0))
                dx = abs(float(xj or 0) - float(xi or 0))
                dy = abs(float(yj or 0) - float(yi or 0))
                span = math.sqrt(dx*dx + dy*dy + 1e-12)
                if dz > span * 0.8:
                    sap_model.FrameObj.SetProp(name, col_sec)
                elif dz > span * 0.3:
                    sap_model.FrameObj.SetProp(name, brace_sec)
                else:
                    sap_model.FrameObj.SetProp(name, beam_sec)
            except Exception:
                sap_model.FrameObj.SetProp(name, beam_sec)
        return True
    except Exception:
        return True


def apply_global_transform_to_points(sap_model: Any, plan_skew_deg: float, twist_deg: float,
                                      lean_m: float, setback_ratio: float) -> int:
    """Apply plan skew, twist, lean, setback to point coordinates. Returns count of transformed points."""
    if abs(plan_skew_deg) < 0.01 and abs(twist_deg) < 0.01 and abs(lean_m) < 0.001 and abs(setback_ratio) < 0.01:
        return 0
    try:
        ret, pts = sap_model.PointObj.GetAllNames()
        pt_list = pts if isinstance(pts, (list, tuple)) else ([pts] if pts else [])
        n = 0
        for pt in pt_list:
            try:
                _, x, y, z = sap_model.PointObj.GetCoordCartesian(pt, 0, 0, 0)
                x, y, z = float(x or 0), float(y or 0), float(z or 0)
                if abs(plan_skew_deg) > 0.01:
                    rad = math.radians(plan_skew_deg)
                    x, y = x * math.cos(rad) - y * math.sin(rad), x * math.sin(rad) + y * math.cos(rad)
                if abs(lean_m) > 0.001 and z > 0:
                    x += lean_m * (x / max(abs(x) + 1e-9, 1))
                    y += lean_m * (y / max(abs(y) + 1e-9, 1))
                if abs(setback_ratio) > 0.01 and z > 0:
                    x *= 1.0 - setback_ratio * (z / 100.0)
                    y *= 1.0 - setback_ratio * (z / 100.0)
                sap_model.PointObj.ChangeCoordCartesian(pt, x, y, z)
                n += 1
            except Exception:
                pass
        return n
    except Exception:
        return 0


def add_braces(sap_model: Any, num_stories: int, story_height: float, bottom_story_height: float,
               lines_x: int, lines_y: int, span_x: float, span_y: float,
               brace_x: float, brace_y: float, brace_density: float,
               brace_sec: str = "BraceSec", rng: Optional[np.random.Generator] = None) -> Tuple[int, int]:
    """Add diagonal braces. brace_x/brace_y=1 to add in that dir, brace_density 0-1. Returns (added, attempted)."""
    if brace_x <= 0 and brace_y <= 0:
        return 0, 0
    added, attempted = 0, 0
    rng = rng or np.random.default_rng(42)
    try:
        z_levels = [0.0, bottom_story_height]
        for _ in range(num_stories - 1):
            z_levels.append(z_levels[-1] + story_height)
        for k in range(len(z_levels) - 1):
            z1, z2 = z_levels[k], z_levels[k + 1]
            for i in range(lines_x - 1):
                for j in range(lines_y - 1):
                    if rng.random() > brace_density:
                        continue
                    attempted += 1
                    x1, y1 = i * span_x, j * span_y
                    x2, y2 = (i + 1) * span_x, (j + 1) * span_y
                    if brace_x > 0:
                        try:
                            sap_model.FrameObj.AddByCoord(x1, y1, z1, x2, y2, z2, f"BR_{k}_{i}_{j}", brace_sec)
                            added += 1
                        except Exception:
                            pass
                    if brace_y > 0 and i != j:
                        x2b, y2b = (i + 1) * span_x, j * span_y
                        try:
                            sap_model.FrameObj.AddByCoord(x1, y1, z1, x2b, y2b, z2, f"BRy_{k}_{i}_{j}", brace_sec)
                            added += 1
                        except Exception:
                            pass
        return added, attempted
    except Exception:
        return added, attempted


def add_skybridges(sap_model: Any, cluster_tower_count: int, bridge_levels_n: int,
                   bridge_span_m: float, tower_gap_m: float) -> int:
    """Add skybridge beams between cluster towers."""
    if cluster_tower_count < 2 or bridge_levels_n <= 0 or bridge_span_m <= 0:
        return 0
    added = 0
    try:
        gap = tower_gap_m if tower_gap_m > 0 else 5.0
        for level in range(int(bridge_levels_n)):
            z = 10.0 + level * 4.0
            try:
                sap_model.FrameObj.AddByCoord(0, 0, z, gap + bridge_span_m, 0, z, f"SKY_{level}", "BeamSec")
                added += 1
            except Exception:
                pass
        return added
    except Exception:
        return added


def add_cantilever_beams(sap_model: Any, params: Dict[str, float]) -> int:
    """Add cantilever beams extending from perimeter."""
    cant_frac = params.get("cantilever_fraction", 0.0)
    cant_len = params.get("cantilever_length_m", 0.0)
    if cant_frac <= 0 or cant_len <= 0:
        return 0
    added = 0
    try:
        lines_x = int(params["lines_x"])
        lines_y = int(params["lines_y"])
        span_x = params["span_x_m"]
        span_y = params["span_y_m"]
        num_stories = int(params["stories"])
        story_h = params["story_height_m"]
        bottom_h = params["bottom_story_height_m"]
        z_levels = [bottom_h]
        for _ in range(num_stories - 1):
            z_levels.append(z_levels[-1] + story_h)
        n_cant = max(1, int(num_stories * cant_frac))
        for k in range(min(n_cant, len(z_levels))):
            z = z_levels[k]
            x1, y1 = (lines_x - 1) * span_x, (lines_y - 1) * span_y
            x2, y2 = x1 + cant_len, y1
            try:
                sap_model.FrameObj.AddByCoord(x1, y1, z, x2, y2, z, f"CANT_{k}", "BeamSec")
                added += 1
            except Exception:
                pass
        return added
    except Exception:
        return added


def apply_beam_releases(sap_model: Any, fraction: float, mode: str = "major_one_end") -> int:
    """Apply moment releases to beams. mode: major_one_end, major_both."""
    if fraction <= 0:
        return 0
    released = 0
    try:
        ret, names = sap_model.FrameObj.GetAllNames()
        frame_list = names if isinstance(names, (list, tuple)) else ([names] if names else [])
        n_release = max(1, int(len(frame_list) * fraction))
        for i, name in enumerate(frame_list[:n_release]):
            try:
                if mode == "major_both":
                    sap_model.FrameObj.SetReleases(name, [True, False, False, False, False, False],
                                                   [True, False, False, False, False, False])
                else:
                    sap_model.FrameObj.SetReleases(name, [True, False, False, False, False, False],
                                                   [False, False, False, False, False, False])
                released += 1
            except Exception:
                pass
        return released
    except Exception:
        return released


def run_analysis(sap_model: Any) -> bool:
    try:
        return sap_model.Analyze.RunAnalysis() == 0
    except Exception:
        return False


def select_case_for_output(sap_model: Any, case_name: str) -> None:
    try:
        sap_model.Results.Setup.DeselectAllCasesAndCombosForOutput()
        sap_model.Results.Setup.SetCaseSelectedForOutput(case_name, True)
    except Exception:
        pass


def extract_frame_force(sap_model: Any, case_name: str = "DEAD") -> Dict[str, float]:
    out = {
        "max_beam_shear_kN": 1e-6, "max_beam_end_shear_kN": 1e-6, "max_beam_moment_kNm": 1e-6,
        "max_column_axial_kN": 1e-6, "max_beam_end_moment_kNm": 1e-6,
    }
    try:
        res = sap_model.Results.FrameForce("", case_name, 0)
        if not res or not isinstance(res, (tuple, list)) or len(res) < 6:
            return out
        p = res[5] if len(res) > 5 else []
        v2 = res[6] if len(res) > 6 else []
        v3 = res[7] if len(res) > 7 else []
        m2 = res[9] if len(res) > 9 else []
        m3 = res[10] if len(res) > 10 else []
        for arr in (p,):
            if arr:
                out["max_column_axial_kN"] = max(out["max_column_axial_kN"], max(abs(float(x)) for x in arr))
        for arr in (v2, v3):
            if arr:
                out["max_beam_shear_kN"] = max(out["max_beam_shear_kN"], max(abs(float(x)) for x in arr))
                out["max_beam_end_shear_kN"] = out["max_beam_shear_kN"]
        for arr in (m2, m3):
            if arr:
                out["max_beam_moment_kNm"] = max(out["max_beam_moment_kNm"], max(abs(float(x)) for x in arr))
                out["max_beam_end_moment_kNm"] = out["max_beam_moment_kNm"]
    except Exception:
        pass
    return out


def extract_base_react(sap_model: Any, case_name: str = "DEAD") -> Tuple[float, float]:
    try:
        res = sap_model.Results.BaseReact(case_name, 0)
        if res:
            r = res if isinstance(res, (tuple, list)) else [res]
            if len(r) >= 5:
                return float(r[3] or 0), float(r[4] or 0)
            if len(r) >= 2:
                return float(r[0] or 0), float(r[1] or 0)
    except Exception:
        pass
    return 0.0, 0.0


def extract_joint_displ(sap_model: Any, case_name: str = "DEAD") -> float:
    try:
        res = sap_model.Results.JointDispl("", case_name, 0)
        if res and isinstance(res, (tuple, list)) and len(res) >= 8:
            u3 = res[7] if len(res) > 7 else res[5] if len(res) > 5 else []
            if hasattr(u3, "__iter__") and not isinstance(u3, str):
                vals = [abs(float(x)) for x in u3]
                if vals:
                    return max(vals)
    except Exception:
        pass
    return 0.0


def build_one_etabs_model(sap_model: Any, params: Dict[str, float], rng: Optional[np.random.Generator] = None) -> bool:
    """Create one parametric steel deck model with braces, cantilevers, full loads, sections."""
    ret = initialize_new_model(sap_model)
    if ret != 0:
        return False
    num_stories = int(params["stories"])
    story_height = params["story_height_m"]
    bottom_story_height = params["bottom_story_height_m"]
    lines_x = int(params["lines_x"])
    lines_y = int(params["lines_y"])
    span_x = params["span_x_m"]
    span_y = params["span_y_m"]
    plan_x = params["plan_x_m"]
    plan_y = params["plan_y_m"]
    total_h = params["total_height_m"]
    snow_kpa = params.get("snow_kpa", 0.0)
    temp_delta = params.get("temp_delta_c", 20.0)
    brace_x = params.get("brace_x", 0.0)
    brace_y = params.get("brace_y", 0.0)
    brace_density = params.get("brace_density", 0.1)
    beam_scale = params.get("beam_scale", 1.0)
    col_scale = params.get("col_scale", 1.0)
    brace_scale = params.get("brace_scale", 1.0)
    beam_release_frac = params.get("beam_release_fraction", 0.02)
    beam_release_mode = "major_both" if params.get("beam_release_mode_id", 1.0) >= 2 else "major_one_end"
    plan_skew = params.get("plan_skew_deg", 0.0)
    twist = params.get("twist_total_deg", 0.0)
    lean = params.get("lean_total_m", 0.0)
    setback = params.get("setback_ratio", 0.0)

    ret = new_steel_deck(sap_model, num_stories, story_height, bottom_story_height,
                         lines_x, lines_y, span_x, span_y)
    if ret != 0:
        return False

    beam_d = max(max(span_x, span_y) / 18.0, 0.45)
    col_d = max(0.70, 0.018 * total_h + 0.03 * max(span_x, span_y) * math.sqrt(max(num_stories, 1)))
    brace_d = max(0.35, 0.65 * beam_d)
    section_pack = section_pack_from_params(beam_d, col_d, brace_d, beam_scale, col_scale, brace_scale)
    define_sections(sap_model, section_pack)
    classify_and_assign_frame_sections(sap_model, "BeamSec", "ColSec", "BraceSec")

    add_load_patterns_and_cases(sap_model, params["dead_kpa"], params["live_kpa"],
                                snow_kpa=snow_kpa)
    assign_area_load_to_floors(sap_model, params["dead_kpa"], params["live_kpa"], snow_kpa)
    assign_frame_load_to_beams(sap_model, params["dead_kpa"], params["live_kpa"], snow_kpa)

    total_weight = plan_x * plan_y * 6.0 * num_stories
    wind_x, wind_y = 0.03, 0.025
    eq_x, eq_y = 0.08, 0.07
    v_wx = total_weight * wind_x
    v_wy = total_weight * wind_y
    v_eqx = total_weight * eq_x
    v_eqy = total_weight * eq_y
    v_impx = total_weight * 0.02
    v_impy = total_weight * 0.02
    v_blastx = total_weight * 0.03
    v_blasty = total_weight * 0.03
    story_heights = [bottom_story_height] + [story_height] * (num_stories - 1)
    assign_lateral_loads(sap_model, total_weight, story_heights, v_wx, v_wy, v_eqx, v_eqy,
                         v_impx, v_impy, v_blastx, v_blasty)
    assign_temperature_load(sap_model, temp_delta)
    assign_impact_and_blast_loads(sap_model, v_impx, v_impy, v_blastx, v_blasty, story_heights)

    n_transformed = apply_global_transform_to_points(sap_model, plan_skew, twist, lean, setback)
    brace_added, brace_attempted = add_braces(sap_model, num_stories, story_height, bottom_story_height,
               lines_x, lines_y, span_x, span_y, brace_x, brace_y, brace_density, "BraceSec", rng)
    cant_added = add_cantilever_beams(sap_model, params)
    released_n = apply_beam_releases(sap_model, beam_release_frac, beam_release_mode)

    params["brace_added"] = float(brace_added)
    params["brace_attempted"] = float(brace_attempted)
    params["cantilever_members_added"] = float(cant_added)
    params["released_beams_n"] = float(released_n)
    params["transformed_points_n"] = float(n_transformed)

    return True


LOAD_CASES_FOR_EXTRACT = ["DEAD", "LIVE", "SNOW", "WX", "WY", "EQX", "EQY", "TEMP", "IMPX", "IMPY", "BLASTX", "BLASTY"]


def _demand_group(val: float, p33: float, p66: float) -> str:
    if val <= p33:
        return "LOW"
    if val <= p66:
        return "MED"
    return "HIGH"


def run_analysis_and_extract(sap_model: Any, params: Dict[str, float]) -> Optional[Dict[str, Any]]:
    """Run analysis and extract FrameForce, BaseReact, JointDispl from ALL load cases."""
    if not run_analysis(sap_model):
        return None

    all_shear, all_moment, all_axial, all_drift = [], [], [], []
    for case in LOAD_CASES_FOR_EXTRACT:
        try:
            select_case_for_output(sap_model, case)
            frame_out = extract_frame_force(sap_model, case)
            all_shear.append(frame_out.get("max_beam_shear_kN", 0.0) or 1e-6)
            all_moment.append(frame_out.get("max_beam_moment_kNm", 0.0) or 1e-6)
            all_axial.append(frame_out.get("max_column_axial_kN", 0.0) or 1e-6)
            max_d = get_joint_displ(sap_model, case)
            all_drift.append(max_d * 1000.0)
        except Exception:
            pass

    max_shear = max(all_shear) if all_shear else 1e-6
    max_moment = max(all_moment) if all_moment else 1e-6
    max_axial = max(all_axial) if all_axial else 1e-6
    max_drift_mm = max(all_drift) if all_drift else 0.0

    select_case_for_output(sap_model, "DEAD")
    base_fx, base_fy, _ = get_base_reaction(sap_model, "DEAD")
    frame_out = extract_frame_force(sap_model, "DEAD")

    plan_x = params["plan_x_m"]
    plan_y = params["plan_y_m"]
    total_h = params["total_height_m"]
    slender_x = total_h / plan_x if plan_x > 0 else 0.0
    slender_y = total_h / plan_y if plan_y > 0 else 0.0
    aspect_ratio = plan_x / plan_y if plan_y > 0 else 1.0
    archetype = params.get("archetype") or choose_round4_archetype(
        params["stories"], params["bays_x"], params["bays_y"], plan_x, plan_y, total_h,
        params.get("brace_density", 0.0))

    sorted_shear = sorted(all_shear) if all_shear else [1e-6]
    sorted_axial = sorted(all_axial) if all_axial else [1e-6]
    p33_s, p66_s = sorted_shear[len(sorted_shear)//3] if len(sorted_shear) >= 2 else 1e-6, sorted_shear[2*len(sorted_shear)//3] if len(sorted_shear) >= 3 else max_shear
    p33_a, p66_a = sorted_axial[len(sorted_axial)//3] if len(sorted_axial) >= 2 else 1e-6, sorted_axial[2*len(sorted_axial)//3] if len(sorted_axial) >= 3 else max_axial
    beam_demand = f"BEAM_{_demand_group(max_shear, p33_s, p66_s)}_DEMAND"
    col_demand = f"COL_{_demand_group(max_axial, p33_a, p66_a)}_DEMAND"

    row: Dict[str, Any] = {
        "archetype": archetype,
        "brace_added": params.get("brace_added", 0.0), "brace_attempted": params.get("brace_attempted", 0.0),
        "stories": params["stories"], "bays_x": params["bays_x"], "bays_y": params["bays_y"],
        "lines_x": params["lines_x"], "lines_y": params["lines_y"],
        "span_x_m": params["span_x_m"], "span_y_m": params["span_y_m"],
        "story_height_m": params["story_height_m"], "bottom_story_height_m": params["bottom_story_height_m"],
        "total_height_m": params["total_height_m"], "plan_x_m": plan_x, "plan_y_m": plan_y,
        "plan_area_m2": plan_x * plan_y, "slender_x": slender_x, "slender_y": slender_y,
        "aspect_ratio": aspect_ratio, "aerodynamic_slenderness": (slender_x + slender_y) / 2.0,
        "brace_x": params.get("brace_x", 0.0), "brace_y": params.get("brace_y", 0.0),
        "brace_density": params.get("brace_density", 0.0),
        "beam_scale": params.get("beam_scale", 1.0), "col_scale": params.get("col_scale", 1.0),
        "brace_scale": params.get("brace_scale", 1.0),
        "beam_stiffness_modifier": 1.0, "col_stiffness_modifier": 1.0, "brace_stiffness_modifier": 1.0,
        "super_dead_kpa": params["dead_kpa"], "live_kpa": params["live_kpa"],
        "snow_kpa": params.get("snow_kpa", 0.0), "selfwt_floor_kpa": 1.8,
        "max_beam_shear_kN": max_shear, "max_beam_end_shear_kN": max_shear,
        "max_beam_moment_kNm": max_moment,
        "max_column_axial_kN": max_axial,
        "max_beam_end_moment_kNm": max_moment,
        "max_beam_deflection_mm": max_drift_mm, "max_drift_mm": max_drift_mm,
        "max_joint_reaction_vertical_kN": abs(base_fy) if base_fy != 0 else max_axial,
        "demand_groups_created": 12,
        "beam_demand_group": beam_demand, "col_demand_group": col_demand,
        "support_mode_id": 1.0, "diaphragm_mode_id": 1.0, "structural_system_id": 2.0,
        "analysis_dimensionality_id": 6.0,
    }
    row = estimate_generation_only_fields(row)
    row = build_physics_features(row)
    return row


def get_csv_columns(csv_path: Path) -> List[str]:
    if not csv_path.exists():
        return []
    df = pd.read_csv(csv_path, nrows=0)
    return list(df.columns)


def get_current_row_count(csv_path: Path) -> int:
    if not csv_path.exists():
        return 0
    return len(pd.read_csv(csv_path))


def collect_dataset(sap_model: Any, target_new: int, start_id: int, csv_columns: List[str],
                   rng: np.random.Generator, failed_log: List[Dict[str, Any]],
                   get_etabs_obj_fn: Optional[Callable[[], Any]] = None) -> List[Dict[str, Any]]:
    """Collect target_new samples. ETABS restart logic: every N models, after consecutive fails, slow model."""
    rows: List[Dict[str, Any]] = []
    ok, fail, consec_fail = 0, 0, 0
    restart_every = CONFIG.get("RESTART_EVERY_N_MODELS", 200)
    restart_after_fail = CONFIG.get("RESTART_AFTER_CONSECUTIVE_FAILS", 5)
    restart_slow_s = CONFIG.get("RESTART_SLOW_MODEL_THRESHOLD_S", 60)
    model_count = 0

    for i in range(target_new):
        sample_id = start_id + i
        params = sample_building_params(rng)
        params["sample_id"] = sample_id

        t0 = time.time()
        if not build_one_etabs_model(sap_model, params, rng):
            failed_log.append({"sample_id": sample_id, "reason": "build_failed", "params": str(params)[:200]})
            fail += 1
            consec_fail += 1
            if consec_fail >= restart_after_fail and get_etabs_obj_fn:
                kill_stale_etabs()
                time.sleep(3)
                try:
                    sap_model = get_etabs_obj_fn()
                    set_units(sap_model, ETABS_UNITS_KNM_C)
                    consec_fail = 0
                    print(f"  Restarted ETABS after {restart_after_fail} consecutive fails")
                except Exception as ex:
                    print(f"  Restart failed: {ex}")
            continue

        result = run_analysis_and_extract(sap_model, params)
        elapsed = time.time() - t0
        model_count += 1

        if result is None:
            failed_log.append({"sample_id": sample_id, "reason": "analysis_extract_failed", "params": str(params)[:200]})
            fail += 1
            consec_fail += 1
            if consec_fail >= restart_after_fail and get_etabs_obj_fn:
                kill_stale_etabs()
                time.sleep(3)
                try:
                    sap_model = get_etabs_obj_fn()
                    set_units(sap_model, ETABS_UNITS_KNM_C)
                    consec_fail = 0
                    print(f"  Restarted ETABS after {restart_after_fail} consecutive fails")
                except Exception as ex:
                    print(f"  Restart failed: {ex}")
            continue

        consec_fail = 0
        result["sample_id"] = sample_id
        rows.append(result)
        ok += 1

        if elapsed > restart_slow_s and get_etabs_obj_fn and model_count > 0:
            kill_stale_etabs()
            time.sleep(3)
            try:
                sap_model = get_etabs_obj_fn()
                set_units(sap_model, ETABS_UNITS_KNM_C)
                print(f"  Restarted ETABS (slow model: {elapsed:.1f}s)")
            except Exception as ex:
                print(f"  Restart failed: {ex}")
        elif model_count >= restart_every and get_etabs_obj_fn:
            kill_stale_etabs()
            time.sleep(3)
            try:
                sap_model = get_etabs_obj_fn()
                set_units(sap_model, ETABS_UNITS_KNM_C)
                model_count = 0
                print(f"  Restarted ETABS (every {restart_every} models)")
            except Exception as ex:
                print(f"  Restart failed: {ex}")

        if (i + 1) % 50 == 0:
            print(f"  Progress: {i+1}/{target_new} (ok={ok}, fail={fail})")
        gc.collect()
    return rows


# Physics-critical targets: higher weight = ETABS-level precision
_PHYSICS_TARGET_WEIGHTS = {
    "max_beam_moment_kNm": 3.0, "max_column_axial_kN": 3.0, "max_drift_mm": 3.0,
    "max_beam_shear_kN": 2.2, "max_beam_end_shear_kN": 2.2, "max_beam_end_moment_kNm": 2.2,
    "max_beam_deflection_mm": 2.0, "max_joint_reaction_vertical_kN": 1.6,
}


def train_brain(df: pd.DataFrame, output_path: Path, epochs: int = 800, patience: int = 150,
                width: int = 448, head_dim: int = 224) -> None:
    """Train StructuralBrain on the dataset with physics-informed weighted loss."""
    n_samples = len(df)
    X_raw = np.zeros((n_samples, len(FEATURE_COLUMNS)), dtype=np.float64)
    for i, c in enumerate(FEATURE_COLUMNS):
        if c in df.columns:
            X_raw[:, i] = df[c].fillna(0).astype(float).values
        X_raw[:, i] = np.maximum(X_raw[:, i], 1e-12)

    y_raw = np.zeros((n_samples, len(TARGET_COLUMNS)), dtype=np.float64)
    for i, c in enumerate(TARGET_COLUMNS):
        if c in df.columns:
            y_raw[:, i] = df[c].fillna(0).astype(float).values
        y_raw[:, i] = np.maximum(y_raw[:, i], 1e-12)

    X_log = np.log(X_raw)
    y_log = np.log(y_raw)
    x_mean = np.mean(X_log, axis=0)
    x_std = np.std(X_log, axis=0)
    x_std[x_std < 1e-12] = 1.0
    y_mean = np.mean(y_log, axis=0)
    y_std = np.std(y_log, axis=0)
    y_std[y_std < 1e-12] = 1.0
    X = ((X_log - x_mean) / x_std).astype(np.float32)
    y = ((y_log - y_mean) / y_std).astype(np.float32)
    X = np.clip(X, -6.0, 6.0)
    y = np.clip(y, -6.0, 6.0)

    rng = np.random.default_rng(42)
    idx = np.arange(n_samples)
    rng.shuffle(idx)
    n_train = max(int(0.85 * n_samples), n_samples - 100)
    n_train = min(n_train, n_samples - 50)
    train_idx, val_idx = idx[:n_train], idx[n_train:]

    X_train, y_train = X[train_idx], y[train_idx]
    X_val, y_val = X[val_idx], y[val_idx]

    in_dim = X.shape[1]
    out_dim = y.shape[1]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    target_weights = np.ones(out_dim, dtype=np.float32)
    for i, c in enumerate(TARGET_COLUMNS):
        if c in _PHYSICS_TARGET_WEIGHTS:
            target_weights[i] = _PHYSICS_TARGET_WEIGHTS[c]
    target_weights_t = torch.tensor(target_weights, dtype=torch.float32).to(device)

    model = StructuralBrain(in_dim, out_dim, width=width, head_dim=head_dim)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode="min", factor=0.5, patience=30, min_lr=1e-6)
    model = model.to(device)

    train_ds = torch.utils.data.TensorDataset(
        torch.tensor(X_train, dtype=torch.float32),
        torch.tensor(y_train, dtype=torch.float32),
    )
    train_loader = torch.utils.data.DataLoader(train_ds, batch_size=64, shuffle=True)
    x_val_t = torch.tensor(X_val, dtype=torch.float32).to(device)
    y_val_t = torch.tensor(y_val, dtype=torch.float32).to(device)

    best_val_loss = float("inf")
    best_state = None
    bad_epochs = 0

    for ep in range(epochs):
        model.train()
        total_loss = 0.0
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad()
            pred = model(xb)
            w = target_weights_t.unsqueeze(0).expand_as(pred)
            loss = (w * (pred - yb) ** 2).mean()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            total_loss += loss.item()
        train_loss = total_loss / len(train_loader)

        model.eval()
        with torch.no_grad():
            val_pred = model(x_val_t)
            w = target_weights_t.unsqueeze(0).expand_as(val_pred)
            val_loss = (w * (val_pred - y_val_t) ** 2).mean().item()
        scheduler.step(val_loss)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            bad_epochs = 0
        else:
            bad_epochs += 1

        if (ep + 1) % 25 == 0 or ep == 0:
            print(f"  Epoch {ep+1:4d} | train={train_loss:.6f} | val={val_loss:.6f} | best={best_val_loss:.6f}")

        if bad_epochs >= patience:
            print(f"\n  Early stopping at epoch {ep+1}")
            break

    if best_state:
        model.load_state_dict(best_state)

    x_scaler = {"mean": x_mean.tolist(), "std": x_std.tolist()}
    y_scaler = {"mean": y_mean.tolist(), "std": y_std.tolist()}
    training_ranges = {}
    for i, c in enumerate(FEATURE_COLUMNS):
        if c in ("stories", "bays_x", "bays_y", "span_x_m", "span_y_m", "story_height_m", "total_height_m"):
            col_vals = X_raw[:, i]
            training_ranges[c] = [float(np.min(col_vals)), float(np.max(col_vals))]

    output_path.parent.mkdir(parents=True, exist_ok=True)
    bundle = {
        "feature_columns": FEATURE_COLUMNS,
        "target_columns": TARGET_COLUMNS,
        "model_state_dict": model.cpu().state_dict(),
        "x_scaler": x_scaler,
        "y_scaler": y_scaler,
        "training_ranges": training_ranges,
        "dataset_rows": int(n_samples),
        "config": {"epochs": epochs, "width": width, "head_dim": head_dim, "physics_weights": True},
    }
    torch.save(bundle, output_path)
    print(f"\nSaved model: {output_path}")


def main() -> None:
    print("=" * 70)
    print("ETABS Brain Full Pipeline — Parametric Models + Analysis + Train")
    print(f"  Version: {SCRIPT_VERSION}")
    print("=" * 70)
    print(f"  Default TARGET_TOTAL_MODELS: {CONFIG['TARGET_TOTAL_MODELS']} (override: ETABS_TARGET_TOTAL)")
    print(f"  Default MAX_NEW_PER_RUN: {CONFIG['MAX_NEW_PER_RUN']} (override: ETABS_MAX_NEW_PER_RUN)")
    print(f"  DATASET_CSV: {DATASET_CSV}")
    print(f"  MODEL_FILE: {MODEL_FILE}")
    print(f"  Demand groups: {DEMAND_GROUPS_CREATED}")
    print()

    DATASET_CSV.parent.mkdir(parents=True, exist_ok=True)
    MODEL_FILE.parent.mkdir(parents=True, exist_ok=True)
    TEMP_MODEL_DIR.mkdir(parents=True, exist_ok=True)

    csv_columns = get_csv_columns(DATASET_CSV)
    if not csv_columns:
        csv_columns = list(FEATURE_COLUMNS) + list(TARGET_COLUMNS) + [
            "sample_id", "archetype", "demand_groups_created",
            "beam_demand_group", "col_demand_group",
        ]

    current_count = get_current_row_count(DATASET_CSV)
    target_total = _env_int("ETABS_TARGET_TOTAL", CONFIG["TARGET_TOTAL_MODELS"])
    max_new = _env_int("ETABS_MAX_NEW_PER_RUN", CONFIG["MAX_NEW_PER_RUN"])
    gap = max(0, target_total - current_count)
    target_new = min(max_new, gap)
    if current_count == 0 and target_new == 0:
        target_new = min(max_new, CONFIG["TARGET_NEW_MODELS_IF_NO_EXISTING"])
    start_id = current_count + 1

    print(f"  Current rows: {current_count}")
    print(f"  Target total (roadmap): {target_total}")
    print(f"  Max new this run: {max_new}")
    print(f"  New samples to generate: {target_new}")
    print()

    if target_new > 0:
        kill_stale_etabs()
        time.sleep(2)
        print("Connecting to ETABS...")
        try:
            sap_model = connect_etabs()
        except Exception as e:
            print(f"  ERROR: Could not connect to ETABS: {e}")
            sys.exit(1)
        set_units(sap_model, ETABS_UNITS_KNM_C)

        rng = np.random.default_rng(CONFIG["RANDOM_SEED"])
        failed_log: List[Dict[str, Any]] = []
        rows = collect_dataset(sap_model, target_new, start_id, csv_columns, rng, failed_log,
                               get_etabs_obj_fn=connect_etabs)
        print(f"\n  Collected {len(rows)} valid samples (failed: {len(failed_log)})")

        if failed_log:
            FAILED_SAMPLES_CSV.parent.mkdir(parents=True, exist_ok=True)
            pd.DataFrame(failed_log).to_csv(FAILED_SAMPLES_CSV, mode="a", header=not FAILED_SAMPLES_CSV.exists(), index=False)

        if not rows:
            print("  No rows to append. Exiting.")
            sys.exit(1)

        df_new = pd.DataFrame(rows)
        for c in csv_columns:
            if c not in df_new.columns:
                df_new[c] = 0.0
        col_order = [c for c in csv_columns if c in df_new.columns]
        df_new = df_new.reindex(columns=col_order, fill_value=0.0)

        if CONFIG["APPEND_DATASET"] and DATASET_CSV.exists():
            df_exist = pd.read_csv(DATASET_CSV)
            df_out = pd.concat([df_exist, df_new], ignore_index=True)
        else:
            df_out = df_new

        df_out.to_csv(DATASET_CSV, index=False)
        print(f"  Saved CSV: {DATASET_CSV} ({len(df_out)} rows)")

        try:
            df_out.to_parquet(DATASET_PARQUET, index=False)
            print(f"  Saved parquet: {DATASET_PARQUET}")
        except Exception as e:
            print(f"  Parquet save skipped: {e}")

        progress_data = {"rows": len(df_out), "last_run": time.strftime("%Y-%m-%d %H:%M:%S")}
        PROGRESS_JSON.parent.mkdir(parents=True, exist_ok=True)
        with open(PROGRESS_JSON, "w") as f:
            json.dump(progress_data, f, indent=2)
    else:
        df_out = pd.read_csv(DATASET_CSV)
        print("  No new samples needed. Using existing dataset for training.")

    skip_train = _env_bool("ETABS_SKIP_TRAIN")
    if skip_train:
        print("\nETABS_SKIP_TRAIN=1 — skipping inline training.")
        print("  Run next:  cd backend")
        print("            python scripts/train_10hr.py --hours 10")
        print("  Or:        run-train-10hr.bat")
    else:
        print("\nTraining structural brain (short pass)...")
        train_brain(df_out, MODEL_FILE)
    metrics = {"script_version": SCRIPT_VERSION, "dataset_rows": len(df_out)}
    with open(METRICS_JSON, "w") as f:
        json.dump(metrics, f, indent=2)
    print("\nDone.")


if __name__ == "__main__":
    main()
