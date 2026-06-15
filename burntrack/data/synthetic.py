"""
synthetic.py — Unified synthetic dataset generator for BurnTrack.
=================================================================

Merged from:
    - rothermel/generate_synthetic_dataset.py (v3 engine, per-fuel-model generation)
    - data_pipeline/synthetic_dataset.py (literature-calibrated African zones)

Generates synthetic training data with controlled biases between Rothermel
baseline ROS and "observed" ROS. Uses ALL fuel models (50+, not just 6).

Bias sources (literature):
    1. Wind measurement error (anemometer calibration)
    2. Moisture underestimation (sun-exposed sensor)
    3. Wrong fuel model identification
    4. Canyon/local wind effects not captured by Rothermel
    5. Micro-topographic slope vs macro (SRTM)

Correction factor range: 0.2 – 5.0 (expanded from 0.3–3.0).
"""

import warnings
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")


# =============================================================================
# IMPORTS: Try burntrack.engine first, fall back to rothermel/
# =============================================================================

_ENGINE_AVAILABLE = False
_FUEL_MODELS_AVAILABLE = False

try:
    from burntrack.engine import (
        RothermelEngine,
        FuelModel as EngineFuelModel,
        MoistureInputs,
        EnvironmentalConditions,
        RothermelOutput,
    )

    _ENGINE_AVAILABLE = True
except ImportError:
    try:
        import sys, os

        _rothermel_dir = os.path.join(
            os.path.dirname(__file__), "..", "..", "rothermel"
        )
        if os.path.isdir(_rothermel_dir):
            sys.path.insert(0, _rothermel_dir)
        from rothermel_engine_v3 import (
            RothermelEngine,
            FuelModel as EngineFuelModel,
            MoistureInputs,
            EnvironmentalConditions,
            RothermelOutput,
        )

        _ENGINE_AVAILABLE = True
    except ImportError:
        pass

try:
    from burntrack.engine import fuel_models as _fm

    ALL_FUEL_MODELS = _fm.ALL_FUEL_MODELS
    get_fuel_model_fn = _fm.get_fuel_model
    FuelModelData = _fm.FuelModel
    _FUEL_MODELS_AVAILABLE = True
except ImportError:
    try:
        import sys, os

        _rothermel_dir = os.path.join(
            os.path.dirname(__file__), "..", "..", "rothermel"
        )
        if os.path.isdir(_rothermel_dir) and "rothermel" not in sys.modules:
            sys.path.insert(0, _rothermel_dir)
        from fuel_models import ALL_FUEL_MODELS, get_fuel_model as get_fuel_model_fn

        FuelModelData = None  # accessed via get_fuel_model_fn
        _FUEL_MODELS_AVAILABLE = True
    except ImportError:
        ALL_FUEL_MODELS = {}
        get_fuel_model_fn = None
        FuelModelData = None

if not _ENGINE_AVAILABLE:
    print(
        "  [WARN] RothermelEngine not available. "
        "Synthetic dataset will use empirical ROS estimates."
    )

if not _FUEL_MODELS_AVAILABLE:
    print(
        "  [WARN] Fuel models not available. "
        "Synthetic dataset will use default parameters."
    )


# =============================================================================
# CONFIGURATION
# =============================================================================

@dataclass
class SyntheticConfig:
    """Configuration for the synthetic dataset generator.

    Attributes:
        n_samples: Total number of samples to generate.
        random_seed: Random seed for reproducibility.
        noise_level: Residual noise level (fraction, e.g. 0.12).
        correction_min: Minimum correction factor bound.
        correction_max: Maximum correction factor bound.
        output_path: Default output CSV path.
    """

    n_samples: int = 50000
    random_seed: int = 42
    noise_level: float = 0.12
    correction_min: float = 0.2
    correction_max: float = 5.0
    output_path: str = "synthetic_dataset.csv"


# =============================================================================
# AFRICA CLIMATE ZONES (from data_pipeline/synthetic_dataset.py)
# =============================================================================

AFRICA_CLIMATE_ZONES = {
    "sahel": {
        "temp_mean": 35.0,
        "temp_std": 5.0,
        "rh_mean": 20.0,
        "rh_std": 10.0,
        "wind_mean": 4.0,
        "wind_std": 2.5,
        "slope_mean": 3.0,
        "slope_std": 4.0,
        "vpd_bias": 2.5,
        "fuel_models": ["AF_SAHEL_GRASS", "GR1", "AF_GRASSLAND_FERTILE"],
        "bias_profile": "dry_grass_over",
    },
    "south_africa_fynbos": {
        "temp_mean": 25.0,
        "temp_std": 6.0,
        "rh_mean": 45.0,
        "rh_std": 15.0,
        "wind_mean": 5.5,
        "wind_std": 3.0,
        "slope_mean": 15.0,
        "slope_std": 10.0,
        "vpd_bias": 1.5,
        "fuel_models": ["AF_FYNBOS"],
        "bias_profile": "fynbos_under",
    },
    "south_africa_miombo": {
        "temp_mean": 28.0,
        "temp_std": 5.0,
        "rh_mean": 55.0,
        "rh_std": 12.0,
        "wind_mean": 3.0,
        "wind_std": 1.5,
        "slope_mean": 5.0,
        "slope_std": 6.0,
        "vpd_bias": 1.2,
        "fuel_models": ["AF_MIOMBO"],
        "bias_profile": "miombo_under",
    },
    "madagascar": {
        "temp_mean": 30.0,
        "temp_std": 4.0,
        "rh_mean": 65.0,
        "rh_std": 15.0,
        "wind_mean": 3.5,
        "wind_std": 2.0,
        "slope_mean": 10.0,
        "slope_std": 12.0,
        "vpd_bias": 1.0,
        "fuel_models": ["AF_STEPPE", "AF_GRASSLAND_FERTILE"],
        "bias_profile": "mixed",
    },
    "burkina": {
        "temp_mean": 33.0,
        "temp_std": 4.0,
        "rh_mean": 35.0,
        "rh_std": 15.0,
        "wind_mean": 3.0,
        "wind_std": 1.5,
        "slope_mean": 2.0,
        "slope_std": 3.0,
        "vpd_bias": 2.0,
        "fuel_models": ["AF_SAHEL_GRASS", "GR1"],
        "bias_profile": "savanna_under",
    },
}

# =============================================================================
# BIAS PROFILES (from data_pipeline/synthetic_dataset.py)
# =============================================================================

BIAS_PROFILES = {
    "dry_grass_over": {
        "base_factor": 0.78,
        "factor_std": 0.08,
        "conditions": {"vpd_threshold": 3.0, "rh_max": 25},
    },
    "fynbos_under": {
        "base_factor": 1.25,
        "factor_std": 0.06,
        "conditions": {"slope_min": 10},
    },
    "miombo_under": {
        "base_factor": 1.07,
        "factor_std": 0.03,
        "conditions": {"rh_min": 40},
    },
    "savanna_under": {
        "base_factor": 1.15,
        "factor_std": 0.05,
        "conditions": {"temp_min": 30},
    },
    "mixed": {
        "base_factor": 1.0,
        "factor_std": 0.10,
        "conditions": {},
    },
    "wind_over": {
        "base_factor": 0.85,
        "factor_std": 0.05,
        "conditions": {"wind_min": 6.0},
    },
    "humid_under": {
        "base_factor": 1.12,
        "factor_std": 0.03,
        "conditions": {"rh_min": 70},
    },
}

# =============================================================================
# FUEL MODEL ENCODING (expanded to ALL fuel models, not just 6)
# =============================================================================

FUEL_MODEL_ENCODING: Dict[str, float] = {}
if _FUEL_MODELS_AVAILABLE and ALL_FUEL_MODELS:
    for _i, _code in enumerate(sorted(ALL_FUEL_MODELS.keys())):
        FUEL_MODEL_ENCODING[_code] = float(_i + 1)
else:
    FUEL_MODEL_ENCODING = {
        "AF_STEPPE": 1.0,
        "AF_MIOMBO": 2.0,
        "AF_FYNBOS": 3.0,
        "AF_SAHEL_GRASS": 4.0,
        "GR1": 5.0,
        "AF_GRASSLAND_FERTILE": 6.0,
    }

FUEL_MODEL_INV = {v: k for k, v in FUEL_MODEL_ENCODING.items()}

# =============================================================================
# BIAS CONFIG FROM v3 GENERATOR (for sample-level bias simulation)
# =============================================================================

_BIAS_CONFIG = {
    "wind_error_std": 0.15,
    "moisture_bias": -0.03,
    "moisture_error_std": 0.05,
    "slope_error_std": 0.10,
    "fuel_loading_error": 0.10,
    "local_wind_effect": 0.20,
}


# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def convert_fuel_to_engine(fuel) -> "EngineFuelModel":
    """Convert a FuelModel (from fuel_models.py) to the engine's FuelModel."""
    return EngineFuelModel(
        name=fuel.code,
        w_1h=fuel.w_1h,
        w_10h=fuel.w_10h,
        w_100h=fuel.w_100h,
        w_live_herb=fuel.w_live_herb,
        w_live_woody=fuel.w_live_woody,
        sigma_1h=fuel.sigma_1h,
        sigma_10h=fuel.sigma_10h,
        sigma_100h=fuel.sigma_100h,
        sigma_live_herb=fuel.sigma_live_herb,
        sigma_live_woody=fuel.sigma_live_woody,
        delta=fuel.delta,
        mx=fuel.mx,
        h_dead=fuel.h_dead,
        h_live=fuel.h_live,
    )


def compute_vpd(temp_c: float, rh_percent: float) -> float:
    """Vapor pressure deficit in kPa."""
    es = 0.6108 * np.exp(17.27 * temp_c / (temp_c + 237.3))
    vpd = es * (1.0 - rh_percent / 100.0)
    return max(0.0, vpd)


def compute_dfmc(temp_c: float, vpd: float) -> float:
    """Dead fuel moisture content in percent."""
    dfmc = 30.0 - 2.5 * vpd - 0.1 * temp_c
    return float(np.clip(dfmc, 3.0, 40.0))


def compute_wind_mid_flame(wind_10m: float, fuel_height: float = 0.3) -> float:
    """Mid-flame wind speed from 10m wind."""
    if fuel_height < 0.6:
        return wind_10m * 0.4
    elif fuel_height < 2.0:
        return wind_10m * 0.6
    return wind_10m * 0.8


def compute_weighted_sav(fuel) -> float:
    """Compute weighted SAV [m²/m³] for a FuelModel.

    Uses the same formula as RothermelEngine._compute_weighted_sav.
    """
    w_1h, w_10h, w_100h = fuel.w_1h, fuel.w_10h, fuel.w_100h
    w_live_herb, w_live_woody = fuel.w_live_herb, fuel.w_live_woody
    sigma_1h = getattr(fuel, "sigma_1h", 0.0)
    sigma_10h = getattr(fuel, "sigma_10h", 0.0)
    sigma_100h = getattr(fuel, "sigma_100h", 0.0)
    sigma_live_herb = getattr(fuel, "sigma_live_herb", 0.0)
    sigma_live_woody = getattr(fuel, "sigma_live_woody", 0.0)

    w_dead = w_1h + w_10h + w_100h
    w_live = w_live_herb + w_live_woody
    w_total = w_dead + w_live

    if w_total <= 0:
        return 0.0

    sav_dead = 0.0
    if w_dead > 0:
        sav_dead = (
            w_1h * sigma_1h + w_10h * sigma_10h + w_100h * sigma_100h
        ) / w_dead

    sav_live = 0.0
    if w_live > 0:
        sav_live = (w_live_herb * sigma_live_herb + w_live_woody * sigma_live_woody) / w_live

    return (w_dead * sav_dead + w_live * sav_live) / w_total


# =============================================================================
# BIAS MODIFIERS (from data_pipeline/synthetic_dataset.py)
# =============================================================================

def _apply_bias_modifiers(
    base_factor: float,
    temp_c: float,
    rh_percent: float,
    wind_speed: float,
    vpd_kpa: float,
    slope_deg: float,
) -> float:
    """Apply literature-documented bias modifiers based on conditions.

    Modifiers:
        - VPD > 3 kPa  : Rothermel overestimates (Cruz 2015) -> reduce factor
        - Wind > 6 m/s : Rothermel overestimates (Andrews 2018) -> reduce factor
        - RH > 70%     : Rothermel underestimates (Rothermel 1972) -> increase factor
        - Slope > 25°  : Rothermel underestimates (Alexander 1985) -> increase factor
    """
    factor = base_factor

    if vpd_kpa > 3.0:
        factor *= np.random.uniform(0.92, 0.98)
    elif vpd_kpa > 2.0:
        factor *= np.random.uniform(0.95, 1.0)

    if wind_speed > 6.0:
        factor *= np.random.uniform(0.90, 0.96)
    elif wind_speed > 4.0:
        factor *= np.random.uniform(0.95, 1.0)

    if rh_percent > 70.0:
        factor *= np.random.uniform(1.05, 1.12)
    elif rh_percent > 50.0:
        factor *= np.random.uniform(1.0, 1.05)

    if slope_deg > 25.0:
        factor *= np.random.uniform(1.05, 1.15)
    elif slope_deg > 15.0:
        factor *= np.random.uniform(1.0, 1.08)

    return factor


# =============================================================================
# PER-SAMPLE GENERATION
# =============================================================================

def generate_sample(config: SyntheticConfig = None) -> dict:
    """Generate a single synthetic sample.

    Uses the full Rothermel engine when available, falls back to
    empirical ROS estimation otherwise. Applies both v3-style
    controlled bias and literature-based zone/profiling.

    Args:
        config: Optional SyntheticConfig. Uses defaults if None.

    Returns:
        Dict with all features and targets for one sample.
    """
    if config is None:
        config = SyntheticConfig()

    np.random.seed(None)  # reseed for true randomness per call

    # Select a zone from Africa climate zones
    zones = list(AFRICA_CLIMATE_ZONES.keys())
    weights = [3, 2, 2, 2, 2]
    zone = np.random.choice(zones, p=np.array(weights) / sum(weights))
    zone_params = AFRICA_CLIMATE_ZONES[zone]

    # Generate meteorology
    temp_c = np.random.normal(zone_params["temp_mean"], zone_params["temp_std"])
    temp_c = np.clip(temp_c, 5.0, 48.0)

    rh_percent = np.random.normal(zone_params["rh_mean"], zone_params["rh_std"])
    rh_percent = np.clip(rh_percent, 5.0, 95.0)

    wind_speed = np.random.normal(zone_params["wind_mean"], zone_params["wind_std"])
    wind_speed = max(0.0, wind_speed)

    slope_deg = np.random.normal(zone_params["slope_mean"], zone_params["slope_std"])
    slope_deg = np.clip(slope_deg, 0.0, 45.0)
    slope_pct = np.tan(np.radians(slope_deg)) * 100.0

    vpd = compute_vpd(temp_c, rh_percent)
    dfmc = compute_dfmc(temp_c, vpd)

    # Select a fuel model
    fuel_code = np.random.choice(zone_params["fuel_models"])
    fuel = get_fuel_model_fn(fuel_code) if get_fuel_model_fn else None

    # Moisture values
    m_1h = np.random.uniform(0.03, 0.25)
    m_10h = m_1h + np.random.uniform(-0.02, 0.05)
    m_10h = np.clip(m_10h, 0.03, 0.30)
    m_100h = m_10h + np.random.uniform(-0.02, 0.05)
    m_100h = np.clip(m_100h, 0.05, 0.35)
    m_live_herb = np.random.uniform(0.5, 1.5)
    m_live_woody = np.random.uniform(0.6, 2.0)
    angle = np.random.choice([0, 45, 90, 135, 180])

    # Compute ROS via Rothermel engine or fallback
    if _ENGINE_AVAILABLE and fuel is not None:
        engine = RothermelEngine()
        fuel_eng = convert_fuel_to_engine(fuel)
        fuel_height = fuel.delta if hasattr(fuel, "delta") else 0.3

        moisture = MoistureInputs(
            m_1h=m_1h,
            m_10h=m_10h,
            m_100h=m_100h,
            m_live_herb=m_live_herb,
            m_live_woody=m_live_woody,
        )
        wind_mid = compute_wind_mid_flame(wind_speed, fuel_height)
        conditions = EnvironmentalConditions(
            wind_speed=wind_mid, slope_pct=slope_pct, angle_wind_slope=angle
        )
        output = engine.compute(fuel_eng, moisture, conditions)
        ros_rothermel = output.ros if output.ros > 0 else 0.1

        phi_w = output.phi_w
        phi_s = output.phi_s
        phi_eff = output.phi_eff
        beta = output.beta
        beta_opt = output.beta_opt
        beta_ratio = output.beta / output.beta_opt if output.beta_opt > 0 else 0.0
        gamma = output.gamma
        eta_M = output.eta_M
        eta_S = output.eta_S
        I_R = output.reaction_intensity
        xi = output.xi
        tau = output.tau
        sigma_m2_m3 = compute_weighted_sav(fuel)

        w_total = fuel.w_total if hasattr(fuel, "w_total") else 1.0
        w_dead = fuel.w_dead if hasattr(fuel, "w_dead") else 0.5
        w_live = fuel.w_live if hasattr(fuel, "w_live") else 0.5
        delta_m = fuel.delta if hasattr(fuel, "delta") else 0.3
        mx_pct = fuel.mx if hasattr(fuel, "mx") else 15.0
        h_dead = fuel.h_dead if hasattr(fuel, "h_dead") else 18608.0
    else:
        # Empirical fallback ROS estimation
        ros_rothermel = 0.5 + 0.05 * temp_c - 0.01 * rh_percent
        ros_rothermel += 0.3 * wind_speed + 0.02 * slope_deg
        ros_rothermel = max(0.1, ros_rothermel)

        phi_w = 0.0
        phi_s = 0.0
        phi_eff = 0.0
        beta = 0.0
        beta_opt = 0.0
        beta_ratio = 0.0
        gamma = 0.0
        eta_M = 0.0
        eta_S = 0.0
        I_R = 0.0
        xi = 0.0
        tau = 0.0
        sigma_m2_m3 = 0.0
        w_total = 1.0
        w_dead = 0.5
        w_live = 0.5
        delta_m = 0.3
        mx_pct = 15.0
        h_dead = 18608.0

    if ros_rothermel <= 0:
        ros_rothermel = 0.1

    # --- v3-style controlled bias ---
    wind_error = np.random.normal(1.0, _BIAS_CONFIG["wind_error_std"])
    wind_error = np.clip(wind_error, 0.5, 1.5)

    moisture_bias = _BIAS_CONFIG["moisture_bias"] + np.random.normal(
        0, _BIAS_CONFIG["moisture_error_std"]
    )

    local_effect = np.random.normal(1.0, _BIAS_CONFIG["local_wind_effect"])
    local_effect = np.clip(local_effect, 0.5, 1.5)

    vent_effect = wind_error**0.5
    moisture_effect = 1.0 + 2.0 * max(0, -moisture_bias)

    ros_terrain_v3 = ros_rothermel * vent_effect * local_effect * moisture_effect
    ros_terrain_v3 *= np.random.normal(1.0, 0.05)

    # --- Literature-based bias ---
    bias_profile = zone_params["bias_profile"]
    bias_config = BIAS_PROFILES[bias_profile]
    base_factor = bias_config["base_factor"]

    literature_factor = _apply_bias_modifiers(
        base_factor, temp_c, rh_percent, wind_speed, vpd, slope_deg
    )
    noise = np.random.normal(1.0, config.noise_level)
    literature_factor *= noise
    literature_factor = np.clip(
        literature_factor, config.correction_min, config.correction_max
    )
    ros_terrain_lit = ros_rothermel * literature_factor

    # Merge both bias estimates (weighted average)
    ros_terrain = 0.6 * ros_terrain_v3 + 0.4 * ros_terrain_lit
    ros_terrain = max(0.1, ros_terrain)

    delta_ros = ros_terrain - ros_rothermel

    # Satellite proxy features
    ndvi = np.clip(np.random.normal(0.35 - 0.005 * temp_c, 0.1), -0.2, 0.8)
    ndwi = np.clip(np.random.normal(-0.1 - 0.003 * temp_c, 0.15), -0.6, 0.4)
    lst_c = temp_c + np.random.uniform(5.0, 20.0)

    return {
        "delta_ros": delta_ros,
        "ros_rothermel": ros_rothermel,
        "ros_terrain": ros_terrain,
        "temp_c": temp_c,
        "rh_percent": rh_percent,
        "wind_speed_ms": compute_wind_mid_flame(wind_speed, delta_m),
        "wind_10m": wind_speed,
        "vpd_kpa": vpd,
        "slope_deg": slope_deg,
        "slope_pct": slope_pct,
        "angle_wind_slope": angle,
        "fuel_model_code": fuel_code,
        "fuel_model_encoded": FUEL_MODEL_ENCODING.get(fuel_code, 0.0),
        "w_total_kg_m2": w_total,
        "w_dead_kg_m2": w_dead,
        "w_live_kg_m2": w_live,
        "delta_m": delta_m,
        "sigma_m2_m3": sigma_m2_m3,
        "mx_percent": mx_pct,
        "h_dead_kj_kg": h_dead,
        "phi_w": phi_w,
        "phi_s": phi_s,
        "phi_eff": phi_eff,
        "beta": beta,
        "beta_opt": beta_opt,
        "beta_ratio": beta_ratio,
        "gamma": gamma,
        "eta_M": eta_M,
        "eta_S": eta_S,
        "I_R_kW_m2": I_R,
        "xi": xi,
        "tau_min": tau,
        "ndvi": ndvi,
        "ndwi": ndwi,
        "lst_c": lst_c,
        "dfmc_percent": dfmc,
        "m_1h": m_1h,
        "m_10h": m_10h,
        "m_100h": m_100h,
        "m_live_herb": m_live_herb,
        "m_live_woody": m_live_woody,
        "zone": zone,
        "correction_factor": ros_terrain / max(ros_rothermel, 0.01),
    }


# =============================================================================
# MAIN DATASET GENERATOR
# =============================================================================

def generate_synthetic_dataset(
    n_samples: int = 50000, output_path: str = "synthetic_dataset.csv"
) -> pd.DataFrame:
    """Generate a complete synthetic dataset for BurnTrack corrector training.

    Uses ALL available fuel models, literature-calibrated African climate
    zones, and both v3-engine bias simulation and literature-based profiling.

    Args:
        n_samples: Total number of samples to generate.
        output_path: CSV output file path.

    Returns:
        DataFrame with all features and targets.
    """
    config = SyntheticConfig(n_samples=n_samples, output_path=output_path)

    # Determine fuel models to use
    if _FUEL_MODELS_AVAILABLE and ALL_FUEL_MODELS:
        fuel_codes = list(ALL_FUEL_MODELS.keys())
        n_per_fuel = max(1, n_samples // len(fuel_codes))
    else:
        fuel_codes = list(FUEL_MODEL_ENCODING.keys())
        n_per_fuel = max(1, n_samples // len(fuel_codes))

    print(
        f"Generating {n_samples} synthetic samples "
        f"({n_per_fuel} per model × {len(fuel_codes)} fuel models)..."
    )
    print(f"  Engine available: {_ENGINE_AVAILABLE}")
    print(f"  Fuel models: {len(fuel_codes)}")
    print(f"  Correction range: {config.correction_min} – {config.correction_max}")
    print(f"  Noise level: {config.noise_level:.0%}")

    all_samples = []
    rejected = 0

    for i, fuel_code in enumerate(fuel_codes):
        fuel_samples = []
        attempts = 0
        max_attempts = n_per_fuel * 5

        while len(fuel_samples) < n_per_fuel and attempts < max_attempts:
            sample = _generate_sample_for_fuel(fuel_code, config)
            attempts += 1
            if sample is not None:
                fuel_samples.append(sample)
            else:
                rejected += 1

        all_samples.extend(fuel_samples)

        if (i + 1) % 10 == 0 or i == len(fuel_codes) - 1:
            print(
                f"  [{i + 1}/{len(fuel_codes)}] {fuel_code}: "
                f"{len(fuel_samples)}/{n_per_fuel} OK"
            )

    df = pd.DataFrame(all_samples)
    df.to_csv(output_path, index=False)

    print()
    print("=" * 60)
    print("SYNTHETIC DATASET GENERATED")
    print("=" * 60)
    print(f"  Valid samples   : {len(df):,}")
    print(f"  Rejected        : {rejected:,}")
    print(f"  Fuel models     : {df['fuel_model_code'].nunique()}")
    print(f"  File            : {output_path}")
    print()
    print("--- TARGET STATISTICS (delta_ros) ---")
    print(f"  Mean : {df['delta_ros'].mean():.3f} m/min")
    print(f"  Std  : {df['delta_ros'].std():.3f} m/min")
    print(f"  Min  : {df['delta_ros'].min():.3f} m/min")
    print(f"  Max  : {df['delta_ros'].max():.3f} m/min")
    print()
    print("--- ROS STATISTICS ---")
    print(f"  ROS Rothermel mean : {df['ros_rothermel'].mean():.2f} m/min")
    print(f"  ROS terrain mean   : {df['ros_terrain'].mean():.2f} m/min")

    return df


def _generate_sample_for_fuel(fuel_code: str, config: SyntheticConfig) -> Optional[dict]:
    """Generate a single sample for a specific fuel model.

    Returns None if the sample is rejected (e.g. ROS <= 0 or ROS > 50).
    """
    fuel = get_fuel_model_fn(fuel_code) if get_fuel_model_fn else None
    if fuel is None:
        return None

    # Override the zone-specific fuel model selection
    zone = np.random.choice(list(AFRICA_CLIMATE_ZONES.keys()))
    zone_params = AFRICA_CLIMATE_ZONES[zone]

    temp_c = np.random.normal(zone_params["temp_mean"], zone_params["temp_std"])
    temp_c = np.clip(temp_c, 5.0, 48.0)
    rh_percent = np.random.normal(zone_params["rh_mean"], zone_params["rh_std"])
    rh_percent = np.clip(rh_percent, 5.0, 95.0)
    wind_speed = np.random.normal(zone_params["wind_mean"], zone_params["wind_std"])
    wind_speed = max(0.0, wind_speed)
    slope_deg = np.random.normal(zone_params["slope_mean"], zone_params["slope_std"])
    slope_deg = np.clip(slope_deg, 0.0, 45.0)
    slope_pct = np.tan(np.radians(slope_deg)) * 100.0

    vpd = compute_vpd(temp_c, rh_percent)
    dfmc = compute_dfmc(temp_c, vpd)

    m_1h = np.random.uniform(0.03, 0.25)
    m_10h = m_1h + np.random.uniform(-0.02, 0.05)
    m_10h = np.clip(m_10h, 0.03, 0.30)
    m_100h = m_10h + np.random.uniform(-0.02, 0.05)
    m_100h = np.clip(m_100h, 0.05, 0.35)
    m_live_herb = np.random.uniform(0.5, 1.5)
    m_live_woody = np.random.uniform(0.6, 2.0)
    angle = np.random.choice([0, 45, 90, 135, 180])

    if _ENGINE_AVAILABLE:
        engine = RothermelEngine()
        fuel_eng = convert_fuel_to_engine(fuel)
        fuel_height = fuel.delta

        moisture = MoistureInputs(
            m_1h=m_1h,
            m_10h=m_10h,
            m_100h=m_100h,
            m_live_herb=m_live_herb,
            m_live_woody=m_live_woody,
        )
        wind_mid = compute_wind_mid_flame(wind_speed, fuel_height)
        conditions = EnvironmentalConditions(
            wind_speed=wind_mid, slope_pct=slope_pct, angle_wind_slope=angle
        )
        output = engine.compute(fuel_eng, moisture, conditions)
        ros_rothermel = output.ros

        phi_w = output.phi_w
        phi_s = output.phi_s
        phi_eff = output.phi_eff
        beta = output.beta
        beta_opt = output.beta_opt
        beta_ratio = output.beta / output.beta_opt if output.beta_opt > 0 else 0.0
        gamma = output.gamma
        eta_M = output.eta_M
        eta_S = output.eta_S
        I_R = output.reaction_intensity
        xi = output.xi
        tau = output.tau
        sigma_m2_m3 = compute_weighted_sav(fuel)
    else:
        ros_rothermel = 0.5 + 0.05 * temp_c - 0.01 * rh_percent
        ros_rothermel += 0.3 * wind_speed + 0.02 * slope_deg
        ros_rothermel = max(0.1, ros_rothermel)

        phi_w = 0.0
        phi_s = 0.0
        phi_eff = 0.0
        beta = 0.0
        beta_opt = 0.0
        beta_ratio = 0.0
        gamma = 0.0
        eta_M = 0.0
        eta_S = 0.0
        I_R = 0.0
        xi = 0.0
        tau = 0.0
        sigma_m2_m3 = 1500.0

    if ros_rothermel <= 0 or ros_rothermel > 50:
        return None

    # v3-style bias
    wind_error = np.random.normal(1.0, _BIAS_CONFIG["wind_error_std"])
    wind_error = np.clip(wind_error, 0.5, 1.5)
    moisture_bias = _BIAS_CONFIG["moisture_bias"] + np.random.normal(
        0, _BIAS_CONFIG["moisture_error_std"]
    )
    local_effect = np.random.normal(1.0, _BIAS_CONFIG["local_wind_effect"])
    local_effect = np.clip(local_effect, 0.5, 1.5)

    vent_effect = wind_error**0.5
    moisture_effect = 1.0 + 2.0 * max(0, -moisture_bias)
    ros_terrain_v3 = ros_rothermel * vent_effect * local_effect * moisture_effect
    ros_terrain_v3 *= np.random.normal(1.0, 0.05)

    # literature-based bias
    bias_profile = zone_params["bias_profile"]
    bias_config = BIAS_PROFILES[bias_profile]
    base_factor = bias_config["base_factor"]
    literature_factor = _apply_bias_modifiers(
        base_factor, temp_c, rh_percent, wind_speed, vpd, slope_deg
    )
    noise = np.random.normal(1.0, config.noise_level)
    literature_factor *= noise
    literature_factor = np.clip(
        literature_factor, config.correction_min, config.correction_max
    )
    ros_terrain_lit = ros_rothermel * literature_factor

    ros_terrain = 0.6 * ros_terrain_v3 + 0.4 * ros_terrain_lit
    ros_terrain = max(0.1, ros_terrain)
    delta_ros = ros_terrain - ros_rothermel

    ndvi = np.clip(np.random.normal(0.35 - 0.005 * temp_c, 0.1), -0.2, 0.8)
    ndwi = np.clip(np.random.normal(-0.1 - 0.003 * temp_c, 0.15), -0.6, 0.4)
    lst_c = temp_c + np.random.uniform(5.0, 20.0)

    return {
        "delta_ros": delta_ros,
        "ros_rothermel": ros_rothermel,
        "ros_terrain": ros_terrain,
        "temp_c": temp_c,
        "rh_percent": rh_percent,
        "wind_speed_ms": compute_wind_mid_flame(wind_speed, fuel.delta),
        "wind_10m": wind_speed,
        "vpd_kpa": vpd,
        "slope_deg": slope_deg,
        "slope_pct": slope_pct,
        "angle_wind_slope": angle,
        "fuel_model_code": fuel_code,
        "fuel_model_encoded": FUEL_MODEL_ENCODING.get(fuel_code, 0.0),
        "w_total_kg_m2": fuel.w_total,
        "w_dead_kg_m2": fuel.w_dead,
        "w_live_kg_m2": fuel.w_live,
        "delta_m": fuel.delta,
        "sigma_m2_m3": sigma_m2_m3,
        "mx_percent": fuel.mx,
        "h_dead_kj_kg": fuel.h_dead,
        "phi_w": phi_w,
        "phi_s": phi_s,
        "phi_eff": phi_eff,
        "beta": beta,
        "beta_opt": beta_opt,
        "beta_ratio": beta_ratio,
        "gamma": gamma,
        "eta_M": eta_M,
        "eta_S": eta_S,
        "I_R_kW_m2": I_R,
        "xi": xi,
        "tau_min": tau,
        "ndvi": ndvi,
        "ndwi": ndwi,
        "lst_c": lst_c,
        "dfmc_percent": dfmc,
        "m_1h": m_1h,
        "m_10h": m_10h,
        "m_100h": m_100h,
        "m_live_herb": m_live_herb,
        "m_live_woody": m_live_woody,
        "zone": zone,
        "correction_factor": ros_terrain / max(ros_rothermel, 0.01),
    }


# =============================================================================
# TRAIN / VAL / TEST SPLIT
# =============================================================================

def generate_train_val_test_split(
    df: pd.DataFrame = None,
    n_samples: int = 50000,
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    output_dir: str = ".",
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Create stratified train/val/test split and save to CSVs.

    Split is stratified by fuel model to prevent data leakage.

    Args:
        df: Existing DataFrame. If None, generates a new one via
            generate_synthetic_dataset.
        n_samples: Number of samples (only used if df is None).
        train_ratio: Fraction for training.
        val_ratio: Fraction for validation.
        output_dir: Directory for output CSVs.

    Returns:
        Tuple of (train_df, val_df, test_df).
    """
    from sklearn.model_selection import train_test_split

    import os as _os

    if df is None:
        df = generate_synthetic_dataset(
            n_samples=n_samples,
            output_path=_os.path.join(output_dir, "synthetic_dataset.csv"),
        )

    train_list, val_list, test_list = [], [], []

    for fuel_code in df["fuel_model_code"].unique():
        fuel_df = df[df["fuel_model_code"] == fuel_code]

        if len(fuel_df) < 10:
            train_list.append(fuel_df)
            continue

        train_fuel, temp_fuel = train_test_split(
            fuel_df, train_size=train_ratio, random_state=42
        )
        val_fuel, test_fuel = train_test_split(
            temp_fuel,
            train_size=val_ratio / (1 - train_ratio),
            random_state=42,
        )

        train_list.append(train_fuel)
        val_list.append(val_fuel)
        test_list.append(test_fuel)

    train_df = pd.concat(train_list).reset_index(drop=True)
    val_df = pd.concat(val_list).reset_index(drop=True)
    test_df = pd.concat(test_list).reset_index(drop=True)

    train_df.to_csv(f"{output_dir}/train.csv", index=False)
    val_df.to_csv(f"{output_dir}/val.csv", index=False)
    test_df.to_csv(f"{output_dir}/test.csv", index=False)

    print()
    print("=" * 60)
    print("TRAIN / VAL / TEST SPLIT")
    print("=" * 60)
    print(f"  Train : {len(train_df):,} ({len(train_df)/len(df)*100:.1f}%)")
    print(f"  Val   : {len(val_df):,} ({len(val_df)/len(df)*100:.1f}%)")
    print(f"  Test  : {len(test_df):,} ({len(test_df)/len(df)*100:.1f}%)")
    print(f"  Files saved in {output_dir}/")

    return train_df, val_df, test_df
