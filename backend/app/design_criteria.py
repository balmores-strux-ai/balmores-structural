"""
Location-aware design criteria resolver.

Given a free-text location (e.g. "Manila", "Cebu", "Davao", "Tokyo", "Singapore"),
return a complete dictionary of design parameters used by the PyNite analysis kernel:

    * Dead/Live/Snow loads (default kPa)
    * Wind: design wind speed V (m/s, 3-s gust 50-yr return), velocity-pressure q_z (kPa)
    * Seismic: zone, peak ground acceleration, equivalent base-shear coefficient
    * Soil bearing capacity (SBC, kPa)

Sources are baked-in tables — we don't hit the live network at request time, so the
service stays fast and always answers. Every value carries a ``source`` string so the
front-end can show the user exactly **where** the assumption came from.

When the user types a location we cannot recognise, we fall back to a generic
"moderate" set of parameters and clearly tag them as assumptions.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple


# ---------------------------------------------------------------------------
# Core types
# ---------------------------------------------------------------------------


@dataclass
class WindCriteria:
    design_wind_speed_mps: float
    pressure_kpa: float            # simplified velocity pressure q = 0.613·V²·1e-3 → kPa
    exposure_category: str = "B"   # A=open, B=suburban, C=urban, D=rough water
    importance_factor: float = 1.0
    code_basis: str = ""

    def as_table(self) -> List[List[str]]:
        return [
            ["Design wind speed V (3-s gust, 50-yr)", f"{self.design_wind_speed_mps:.0f}", "m/s"],
            ["Velocity pressure q = 0.613·V² ", f"{self.pressure_kpa:.2f}", "kPa"],
            ["Exposure category", self.exposure_category, "—"],
            ["Importance factor I", f"{self.importance_factor:.2f}", "—"],
            ["Code basis", self.code_basis or "—", "—"],
        ]


@dataclass
class SeismicCriteria:
    zone: int                         # NSCP zone (1..4) or analogous tier
    pga_g: float                      # peak ground acceleration (g)
    base_shear_coeff: float           # equivalent V/W
    site_class: str = "D"
    code_basis: str = ""

    def as_table(self) -> List[List[str]]:
        return [
            ["Seismic zone", str(self.zone), "—"],
            ["Peak ground acceleration", f"{self.pga_g:.2f}", "g"],
            ["Base-shear coefficient V/W", f"{self.base_shear_coeff:.3f}", "—"],
            ["Site class", self.site_class, "—"],
            ["Code basis", self.code_basis or "—", "—"],
        ]


@dataclass
class SoilCriteria:
    sbc_kpa: float
    description: str = ""
    code_basis: str = ""

    def as_table(self) -> List[List[str]]:
        return [
            ["Allowable soil bearing", f"{self.sbc_kpa:.0f}", "kPa"],
            ["Description", self.description or "—", ""],
            ["Code basis", self.code_basis or "—", ""],
        ]


@dataclass
class LoadCriteria:
    """Default gravity loads when not user-specified."""

    dl_kpa: float = 4.5
    ll_kpa: float = 3.0
    snow_kpa: float = 0.0  # Philippines & most equatorial sites: 0
    notes: str = ""

    def as_table(self) -> List[List[str]]:
        return [
            ["Dead load (DL) — partitions, finishes, ME", f"{self.dl_kpa:.2f}", "kPa"],
            ["Live load (LL) — typical office/residential", f"{self.ll_kpa:.2f}", "kPa"],
            ["Snow load", f"{self.snow_kpa:.2f}", "kPa"],
        ]


@dataclass
class LoadCombinations:
    """ULS / SLS combinations used by the solver."""

    uls: List[str] = field(
        default_factory=lambda: [
            "1.4·DL",
            "1.2·DL + 1.6·LL",
            "1.2·DL + 1.0·LL + 1.0·W",
            "0.9·DL + 1.0·W",
            "1.2·DL + 1.0·LL + 1.0·E",
            "0.9·DL + 1.0·E",
        ]
    )
    sls: List[str] = field(default_factory=lambda: ["1.0·DL + 1.0·LL", "1.0·DL + 0.7·W"])
    governing: str = "1.2·DL + 1.6·LL + 1.0·W + 1.0·E (envelope solved as 'ULS')"


@dataclass
class DesignCriteriaResult:
    location_input: str
    matched_location: Optional[str]   # canonical city or None
    country: str
    is_assumed: bool                  # True when we couldn't match the location
    loads: LoadCriteria
    wind: WindCriteria
    seismic: SeismicCriteria
    soil: SoilCriteria
    combos: LoadCombinations = field(default_factory=LoadCombinations)
    notes: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "location_input": self.location_input,
            "matched_location": self.matched_location,
            "country": self.country,
            "is_assumed": self.is_assumed,
            "loads": {
                "dl_kpa": self.loads.dl_kpa,
                "ll_kpa": self.loads.ll_kpa,
                "snow_kpa": self.loads.snow_kpa,
                "notes": self.loads.notes,
            },
            "wind": {
                "design_wind_speed_mps": self.wind.design_wind_speed_mps,
                "pressure_kpa": self.wind.pressure_kpa,
                "exposure_category": self.wind.exposure_category,
                "importance_factor": self.wind.importance_factor,
                "code_basis": self.wind.code_basis,
            },
            "seismic": {
                "zone": self.seismic.zone,
                "pga_g": self.seismic.pga_g,
                "base_shear_coeff": self.seismic.base_shear_coeff,
                "site_class": self.seismic.site_class,
                "code_basis": self.seismic.code_basis,
            },
            "soil": {
                "sbc_kpa": self.soil.sbc_kpa,
                "description": self.soil.description,
                "code_basis": self.soil.code_basis,
            },
            "combos": {
                "uls": self.combos.uls,
                "sls": self.combos.sls,
                "governing": self.combos.governing,
            },
            "notes": self.notes,
        }


# ---------------------------------------------------------------------------
# Location database
# ---------------------------------------------------------------------------
#
# Wind speeds & seismic zones for Philippines come from NSCP 2015 (basic wind
# speed map and seismic source maps), abbreviated:
#   - Zone 4 (V ≥ 270 km/h ≈ 75 m/s): Eastern Visayas, parts of Northern Luzon
#     coastline, etc.
#   - Most of the country sits in NSCP seismic zone 4 (PGA ~0.40 g).
#   - Palawan sits in seismic zone 2 (PGA ~0.20 g).
# International cities use ASCE 7 / Eurocode / AS NZS 1170 typical values.
# All values are intentionally conservative-but-realistic for a fast first pass.


def _q_kpa_from_V(V_mps: float) -> float:
    """ASCE-7 / NSCP 2015 simplified velocity pressure: q = 0.613·V² (Pa)."""
    return round(0.613 * V_mps * V_mps / 1000.0, 3)


_PHL_NSCP = "NSCP 2015 (Philippines National Structural Code)"
_ASCE = "ASCE 7-22 / IBC 2021 (USA)"
_EC = "Eurocode 1 (EN 1991) / EN 1998 (EU)"
_AS = "AS/NZS 1170 (Australia / New Zealand)"
_JP = "Japanese Building Standard Law / Notification 1454"

# Each entry: canonical_name, country, V (m/s), exposure, seismic_zone,
# pga_g, V/W coeff, sbc_kpa, soil description.
_CITY_DB: Dict[str, Dict[str, Any]] = {
    # Philippines
    "manila":        {"country": "Philippines", "V": 70, "exp": "C", "zone": 4, "pga": 0.40, "csb": 0.110, "sbc": 150, "soil": "Manila clay / sandy clay (typical)", "code_w": _PHL_NSCP, "code_s": _PHL_NSCP},
    "quezon city":   {"country": "Philippines", "V": 70, "exp": "C", "zone": 4, "pga": 0.40, "csb": 0.110, "sbc": 200, "soil": "Stiff clay / weathered tuff", "code_w": _PHL_NSCP, "code_s": _PHL_NSCP},
    "makati":        {"country": "Philippines", "V": 70, "exp": "C", "zone": 4, "pga": 0.40, "csb": 0.110, "sbc": 250, "soil": "Guadalupe tuff (hard rock at depth)", "code_w": _PHL_NSCP, "code_s": _PHL_NSCP},
    "taguig":        {"country": "Philippines", "V": 70, "exp": "C", "zone": 4, "pga": 0.40, "csb": 0.110, "sbc": 200, "soil": "Reclaimed / silty clay", "code_w": _PHL_NSCP, "code_s": _PHL_NSCP},
    "pasay":         {"country": "Philippines", "V": 70, "exp": "C", "zone": 4, "pga": 0.40, "csb": 0.110, "sbc": 150, "soil": "Reclaimed coastal", "code_w": _PHL_NSCP, "code_s": _PHL_NSCP},
    "cebu":          {"country": "Philippines", "V": 70, "exp": "C", "zone": 4, "pga": 0.40, "csb": 0.110, "sbc": 250, "soil": "Limestone / coralline", "code_w": _PHL_NSCP, "code_s": _PHL_NSCP},
    "cebu city":     {"country": "Philippines", "V": 70, "exp": "C", "zone": 4, "pga": 0.40, "csb": 0.110, "sbc": 250, "soil": "Limestone / coralline", "code_w": _PHL_NSCP, "code_s": _PHL_NSCP},
    "davao":         {"country": "Philippines", "V": 65, "exp": "C", "zone": 4, "pga": 0.40, "csb": 0.110, "sbc": 200, "soil": "Volcanic loam / weathered basalt", "code_w": _PHL_NSCP, "code_s": _PHL_NSCP},
    "davao city":    {"country": "Philippines", "V": 65, "exp": "C", "zone": 4, "pga": 0.40, "csb": 0.110, "sbc": 200, "soil": "Volcanic loam / weathered basalt", "code_w": _PHL_NSCP, "code_s": _PHL_NSCP},
    "iloilo":        {"country": "Philippines", "V": 70, "exp": "C", "zone": 4, "pga": 0.40, "csb": 0.110, "sbc": 175, "soil": "Alluvial silty sand", "code_w": _PHL_NSCP, "code_s": _PHL_NSCP},
    "tacloban":      {"country": "Philippines", "V": 80, "exp": "C", "zone": 4, "pga": 0.40, "csb": 0.110, "sbc": 150, "soil": "Coastal alluvium", "code_w": _PHL_NSCP, "code_s": _PHL_NSCP},
    "baguio":        {"country": "Philippines", "V": 70, "exp": "B", "zone": 4, "pga": 0.40, "csb": 0.110, "sbc": 300, "soil": "Pine-ridge weathered rock", "code_w": _PHL_NSCP, "code_s": _PHL_NSCP},
    "puerto princesa":{"country": "Philippines","V": 60, "exp": "C", "zone": 2, "pga": 0.20, "csb": 0.060, "sbc": 200, "soil": "Coral limestone", "code_w": _PHL_NSCP, "code_s": _PHL_NSCP},
    "general santos":{"country": "Philippines","V": 60, "exp": "C", "zone": 4, "pga": 0.40, "csb": 0.110, "sbc": 200, "soil": "Volcanic alluvium", "code_w": _PHL_NSCP, "code_s": _PHL_NSCP},
    "zamboanga":     {"country": "Philippines","V": 60, "exp": "C", "zone": 4, "pga": 0.40, "csb": 0.110, "sbc": 200, "soil": "Coralline / clayey sand", "code_w": _PHL_NSCP, "code_s": _PHL_NSCP},
    "cagayan de oro":{"country": "Philippines","V": 65, "exp": "C", "zone": 4, "pga": 0.40, "csb": 0.110, "sbc": 200, "soil": "Alluvial / weathered tuff", "code_w": _PHL_NSCP, "code_s": _PHL_NSCP},
    "bacolod":       {"country": "Philippines","V": 70, "exp": "C", "zone": 4, "pga": 0.40, "csb": 0.110, "sbc": 175, "soil": "Alluvial silty sand", "code_w": _PHL_NSCP, "code_s": _PHL_NSCP},
    "batangas":      {"country": "Philippines","V": 70, "exp": "C", "zone": 4, "pga": 0.40, "csb": 0.110, "sbc": 250, "soil": "Volcanic tuff", "code_w": _PHL_NSCP, "code_s": _PHL_NSCP},
    "laoag":         {"country": "Philippines","V": 75, "exp": "C", "zone": 4, "pga": 0.40, "csb": 0.110, "sbc": 200, "soil": "Coastal alluvium", "code_w": _PHL_NSCP, "code_s": _PHL_NSCP},

    # Asia
    "tokyo":         {"country": "Japan",    "V": 38, "exp": "B", "zone": 4, "pga": 0.40, "csb": 0.20, "sbc": 200, "soil": "Tokyo bay alluvium", "code_w": _JP, "code_s": _JP},
    "osaka":         {"country": "Japan",    "V": 36, "exp": "B", "zone": 4, "pga": 0.40, "csb": 0.20, "sbc": 200, "soil": "Yodogawa alluvium", "code_w": _JP, "code_s": _JP},
    "singapore":     {"country": "Singapore","V": 32, "exp": "B", "zone": 1, "pga": 0.05, "csb": 0.030, "sbc": 200, "soil": "Bukit Timah granite / Kallang", "code_w": _EC, "code_s": _EC},
    "hong kong":     {"country": "Hong Kong","V": 50, "exp": "C", "zone": 2, "pga": 0.10, "csb": 0.040, "sbc": 350, "soil": "Granite / completely decomposed granite", "code_w": "Hong Kong Code of Practice", "code_s": "Hong Kong Code of Practice"},
    "bangkok":       {"country": "Thailand", "V": 32, "exp": "B", "zone": 1, "pga": 0.05, "csb": 0.030, "sbc": 80,  "soil": "Soft Bangkok clay", "code_w": "TIS / EIT", "code_s": "TIS / EIT"},
    "jakarta":       {"country": "Indonesia","V": 36, "exp": "B", "zone": 3, "pga": 0.30, "csb": 0.090, "sbc": 100, "soil": "Alluvial / soft clay", "code_w": "SNI 1727 / SNI 1726", "code_s": "SNI 1727 / SNI 1726"},
    "kuala lumpur":  {"country": "Malaysia", "V": 32, "exp": "B", "zone": 1, "pga": 0.07, "csb": 0.030, "sbc": 200, "soil": "Residual granite", "code_w": "MS 1553 / EC8", "code_s": "MS 1553 / EC8"},
    "ho chi minh":   {"country": "Vietnam",  "V": 36, "exp": "B", "zone": 1, "pga": 0.05, "csb": 0.030, "sbc": 100, "soil": "Mekong alluvium", "code_w": "TCVN", "code_s": "TCVN"},
    "hanoi":         {"country": "Vietnam",  "V": 36, "exp": "B", "zone": 1, "pga": 0.07, "csb": 0.030, "sbc": 120, "soil": "Red River alluvium", "code_w": "TCVN", "code_s": "TCVN"},
    "seoul":         {"country": "South Korea","V": 30, "exp": "B", "zone": 2, "pga": 0.15, "csb": 0.060, "sbc": 200, "soil": "Granite weathered profile", "code_w": "KBC 2016", "code_s": "KBC 2016"},
    "taipei":        {"country": "Taiwan",   "V": 50, "exp": "C", "zone": 4, "pga": 0.40, "csb": 0.18, "sbc": 175, "soil": "Taipei basin (lacustrine)", "code_w": "Taiwan Building Code", "code_s": "Taiwan Building Code"},
    "shanghai":      {"country": "China",    "V": 36, "exp": "C", "zone": 2, "pga": 0.10, "csb": 0.050, "sbc": 100, "soil": "Yangtze delta soft clay", "code_w": "GB 50009 / GB 50011", "code_s": "GB 50009 / GB 50011"},
    "beijing":       {"country": "China",    "V": 30, "exp": "B", "zone": 3, "pga": 0.20, "csb": 0.080, "sbc": 250, "soil": "Loess / silty clay", "code_w": "GB 50009 / GB 50011", "code_s": "GB 50009 / GB 50011"},

    # Americas
    "new york":      {"country": "USA", "V": 50, "exp": "B", "zone": 2, "pga": 0.15, "csb": 0.060, "sbc": 250, "soil": "Manhattan schist", "code_w": _ASCE, "code_s": _ASCE},
    "san francisco": {"country": "USA", "V": 38, "exp": "B", "zone": 4, "pga": 0.50, "csb": 0.20,  "sbc": 200, "soil": "Bay mud / Franciscan", "code_w": _ASCE, "code_s": _ASCE},
    "los angeles":   {"country": "USA", "V": 38, "exp": "B", "zone": 4, "pga": 0.50, "csb": 0.20,  "sbc": 200, "soil": "Alluvium / sandstone", "code_w": _ASCE, "code_s": _ASCE},
    "chicago":       {"country": "USA", "V": 50, "exp": "C", "zone": 1, "pga": 0.05, "csb": 0.030, "sbc": 250, "soil": "Hard pan / silty clay", "code_w": _ASCE, "code_s": _ASCE},
    "miami":         {"country": "USA", "V": 75, "exp": "D", "zone": 1, "pga": 0.05, "csb": 0.030, "sbc": 150, "soil": "Limestone / sand", "code_w": _ASCE, "code_s": _ASCE},
    "houston":       {"country": "USA", "V": 60, "exp": "C", "zone": 1, "pga": 0.05, "csb": 0.030, "sbc": 150, "soil": "Coastal clay", "code_w": _ASCE, "code_s": _ASCE},

    # EU & Oceania
    "london":        {"country": "UK",        "V": 30, "exp": "B", "zone": 1, "pga": 0.04, "csb": 0.020, "sbc": 200, "soil": "London clay", "code_w": "BS EN 1991-1-4", "code_s": "BS EN 1998"},
    "paris":         {"country": "France",    "V": 28, "exp": "B", "zone": 1, "pga": 0.04, "csb": 0.020, "sbc": 200, "soil": "Limestone / marl", "code_w": _EC, "code_s": _EC},
    "berlin":        {"country": "Germany",   "V": 26, "exp": "B", "zone": 1, "pga": 0.04, "csb": 0.020, "sbc": 200, "soil": "Sand / glacial till", "code_w": _EC, "code_s": _EC},
    "rome":          {"country": "Italy",     "V": 28, "exp": "B", "zone": 3, "pga": 0.20, "csb": 0.080, "sbc": 200, "soil": "Tuff / clay", "code_w": _EC, "code_s": _EC},
    "madrid":        {"country": "Spain",     "V": 28, "exp": "B", "zone": 1, "pga": 0.05, "csb": 0.020, "sbc": 250, "soil": "Clayey sand", "code_w": _EC, "code_s": _EC},
    "sydney":        {"country": "Australia", "V": 41, "exp": "B", "zone": 1, "pga": 0.08, "csb": 0.030, "sbc": 250, "soil": "Hawkesbury sandstone", "code_w": _AS, "code_s": _AS},
    "melbourne":     {"country": "Australia", "V": 41, "exp": "B", "zone": 1, "pga": 0.08, "csb": 0.030, "sbc": 250, "soil": "Basalt / silty clay", "code_w": _AS, "code_s": _AS},
    "auckland":      {"country": "New Zealand","V": 50, "exp": "B", "zone": 3, "pga": 0.25, "csb": 0.090, "sbc": 200, "soil": "Volcanic tuff", "code_w": _AS, "code_s": _AS},
    "wellington":    {"country": "New Zealand","V": 55, "exp": "C", "zone": 4, "pga": 0.40, "csb": 0.18, "sbc": 200, "soil": "Greywacke", "code_w": _AS, "code_s": _AS},
}

# Aliases (lowercased) → canonical key
_ALIASES = {
    "ncr": "manila",
    "metro manila": "manila",
    "mnl": "manila",
    "qc": "quezon city",
    "bgc": "taguig",
    "fort bonifacio": "taguig",
    "moa": "pasay",
    "tagum": "davao",
    "cdo": "cagayan de oro",
    "iligan": "cagayan de oro",
    "sg": "singapore",
    "kl": "kuala lumpur",
    "hk": "hong kong",
    "saigon": "ho chi minh",
    "hcmc": "ho chi minh",
    "nyc": "new york",
    "sf": "san francisco",
    "la": "los angeles",
}


def _normalize(s: str) -> str:
    return re.sub(r"[^a-z0-9 ]+", " ", (s or "").lower()).strip()


def _lookup(loc: str) -> Optional[Tuple[str, Dict[str, Any]]]:
    if not loc:
        return None
    n = _normalize(loc)
    if not n:
        return None
    if n in _CITY_DB:
        return n, _CITY_DB[n]
    if n in _ALIASES:
        canon = _ALIASES[n]
        return canon, _CITY_DB[canon]
    # Substring containment (e.g. "BGC, Taguig" → "taguig")
    for key in _CITY_DB:
        if key in n or n in key:
            return key, _CITY_DB[key]
    for alias, canon in _ALIASES.items():
        if alias in n:
            return canon, _CITY_DB[canon]
    return None


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def resolve_design_criteria(
    *,
    location: Optional[str] = None,
    user_dl_kpa: Optional[float] = None,
    user_ll_kpa: Optional[float] = None,
    user_wind_kpa: Optional[float] = None,
    user_seismic_zone: Optional[int] = None,
    user_sbc_kpa: Optional[float] = None,
) -> DesignCriteriaResult:
    """Resolve a complete set of design criteria.

    Priority for every parameter (highest first):
        1. Explicit user value (passed as ``user_*``)
        2. Location-database entry
        3. Generic moderate fallback (clearly tagged as assumed)
    """

    notes: List[str] = []
    matched = _lookup(location or "")

    if matched is None:
        canonical = None
        country = "—"
        is_assumed = bool(location)  # user gave a location we couldn't match
        wind_V = 45.0
        wind_exp = "B"
        wind_code = "Generic moderate fallback (no location match)"
        zone = 2
        pga = 0.15
        csb = 0.06
        seis_code = "Generic moderate fallback (no location match)"
        sbc = 200.0
        soil_desc = "Generic stiff soil / dense sand assumption"
        soil_code = "Generic moderate fallback"
        if location:
            notes.append(
                f"Location '{location}' not in built-in city table — using "
                "generic moderate parameters (V=45 m/s, zone 2, SBC 200 kPa)."
            )
        else:
            notes.append(
                "No location supplied — using generic moderate parameters "
                "(V=45 m/s, seismic zone 2, SBC 200 kPa)."
            )
    else:
        canonical, entry = matched
        country = entry["country"]
        is_assumed = False
        wind_V = float(entry["V"])
        wind_exp = entry["exp"]
        wind_code = entry["code_w"]
        zone = int(entry["zone"])
        pga = float(entry["pga"])
        csb = float(entry["csb"])
        seis_code = entry["code_s"]
        sbc = float(entry["sbc"])
        soil_desc = entry["soil"]
        soil_code = entry["code_s"]
        notes.append(
            f"Location matched: **{canonical.title()}, {country}**. "
            f"Wind / seismic / soil parameters loaded from {wind_code}."
        )

    # Apply explicit user overrides (highest priority)
    if user_wind_kpa is not None:
        # Back-solve V from q = 0.613·V²·1e-3 → V = sqrt(q*1000/0.613)
        try:
            v_from_q = (float(user_wind_kpa) * 1000.0 / 0.613) ** 0.5
            wind_V = v_from_q
            wind_code = "User-supplied wind pressure (overrides location)."
            notes.append(
                f"Wind pressure overridden by user: q = {user_wind_kpa} kPa → "
                f"V ≈ {v_from_q:.0f} m/s."
            )
        except Exception:
            pass

    if user_seismic_zone is not None:
        zone = int(user_seismic_zone)
        # Map zone → coarse PGA (NSCP-style)
        pga_by_zone = {1: 0.075, 2: 0.20, 3: 0.30, 4: 0.40}
        pga = pga_by_zone.get(zone, pga)
        csb = max(0.030, 0.030 + 0.025 * (zone - 1))
        seis_code = f"User-supplied seismic zone {zone} (overrides location)."
        notes.append(f"Seismic zone overridden by user → zone {zone}, PGA ≈ {pga:.2f} g.")

    if user_sbc_kpa is not None:
        sbc = float(user_sbc_kpa)
        soil_code = "User-supplied SBC (overrides location)."
        notes.append(f"Allowable soil bearing overridden by user → {sbc:.0f} kPa.")

    loads = LoadCriteria(
        dl_kpa=float(user_dl_kpa) if user_dl_kpa is not None else 4.5,
        ll_kpa=float(user_ll_kpa) if user_ll_kpa is not None else 3.0,
        snow_kpa=0.0,
        notes="Defaults: typical office/residential. Override with explicit DL/LL in your prompt.",
    )

    wind = WindCriteria(
        design_wind_speed_mps=round(wind_V, 1),
        pressure_kpa=_q_kpa_from_V(wind_V),
        exposure_category=wind_exp,
        importance_factor=1.0,
        code_basis=wind_code,
    )

    seismic = SeismicCriteria(
        zone=zone,
        pga_g=round(pga, 2),
        base_shear_coeff=round(csb, 3),
        site_class="D",
        code_basis=seis_code,
    )

    soil = SoilCriteria(
        sbc_kpa=round(sbc, 0),
        description=soil_desc,
        code_basis=soil_code,
    )

    return DesignCriteriaResult(
        location_input=location or "",
        matched_location=canonical.title() if canonical else None,
        country=country,
        is_assumed=is_assumed,
        loads=loads,
        wind=wind,
        seismic=seismic,
        soil=soil,
        combos=LoadCombinations(),
        notes=notes,
    )


def detect_location_in_text(text: str) -> Optional[str]:
    """Heuristic free-text location extractor.

    Looks for phrases like ``location: Manila``, ``in Manila``, or a bare token
    that matches a known city.  Returns the original (un-normalized) substring
    when found so the caller can echo it to the user.
    """
    if not text:
        return None
    t = text.lower()

    # Explicit prefix forms
    for pat in (
        r"\blocation\s*[:=]\s*([a-z][a-z .,'-]{2,40})",
        r"\bsite\s*(?:in|at)\s*([a-z][a-z .,'-]{2,40})",
        r"\b(?:located|sited|building)\s+(?:in|at)\s+([a-z][a-z .,'-]{2,40})",
        r"\bin\s+([a-z][a-z]+(?:\s+[a-z][a-z]+){0,2})\s*(?:,|\.|$)",
    ):
        m = re.search(pat, t)
        if m:
            cand = m.group(1).strip(" .,").strip()
            cand_clean = re.sub(r"\b(philippines|usa|united states|japan|china|uk)\b", "", cand).strip(" .,")
            if cand_clean and _lookup(cand_clean) is not None:
                return cand_clean
            if cand and _lookup(cand) is not None:
                return cand

    # Bare-token city scan
    for key in _CITY_DB:
        if re.search(rf"\b{re.escape(key)}\b", t):
            return key
    for alias in _ALIASES:
        if re.search(rf"\b{re.escape(alias)}\b", t):
            return alias

    return None


def supported_locations() -> List[str]:
    return sorted(set(_CITY_DB.keys()) | set(_ALIASES.keys()))
