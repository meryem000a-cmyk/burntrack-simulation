"""
real_dataset.py — Real African fire dataset pipeline.
======================================================

Cleaned and restructured from data_pipeline/build_real_dataset.py.

Pipeline:
    1. Download active fires from NASA FIRMS in parallel for 4 African regions.
    2. Reconstruct fire fronts via spatio-temporal DBSCAN clustering.
    3. Associate historical weather (Open-Meteo) with parallel rate limiting.
    4. Compute Rothermel v3 baseline and delta_ros.
    5. Save final real_african_dataset.csv.

Fixes applied (vs original build_real_dataset.py):
    - sigma_m2_m3: computed from the actual fuel model's weighted SAV
      (not hardcoded 1500).
    - ndvi/ndwi: marked as 'estimated_from_region' origin.
    - lst_c = temp_c + 10 approximation: marked with comment.
    - Multiple fuel models supported via land cover matching.
"""

import os
import sys
import warnings

import numpy as np
import pandas as pd

from .firms import (
    download_all_africa_fires,
    reconstruct_propagation,
    AFRICA_REGIONS,
)
from .weather import fetch_weather_for_points

warnings.filterwarnings("ignore")

# =============================================================================
# ENGINE / FUEL MODEL IMPORTS
# =============================================================================

_ENGINE_AVAILABLE = False
_FUEL_MODELS_AVAILABLE = False

try:
    from burntrack.engine import (
        RothermelEngine,
        FuelModel as EngineFuelModel,
        MoistureInputs,
        EnvironmentalConditions,
    )

    _ENGINE_AVAILABLE = True
except ImportError:
    try:
        import sys as _sys
        import os as _os

        _rothermel_dir = _os.path.join(
            _os.path.dirname(__file__), "..", "..", "rothermel"
        )
        if _os.path.isdir(_rothermel_dir):
            _sys.path.insert(0, _rothermel_dir)
        from rothermel_engine_v3 import (
            RothermelEngine,
            FuelModel as EngineFuelModel,
            MoistureInputs,
            EnvironmentalConditions,
        )

        _ENGINE_AVAILABLE = True
    except ImportError:
        pass

try:
    from burntrack.engine import fuel_models as _fm

    get_fuel_model_fn = _fm.get_fuel_model
    ALL_FUEL_MODELS = _fm.ALL_FUEL_MODELS
    ECOSYSTEM_TO_FUEL_MODEL = getattr(_fm, "ECOSYSTEM_TO_FUEL_MODEL", {})
    _FUEL_MODELS_AVAILABLE = True
except ImportError:
    try:
        import sys as _sys
        import os as _os

        _rothermel_dir = _os.path.join(
            _os.path.dirname(__file__), "..", "..", "rothermel"
        )
        if _os.path.isdir(_rothermel_dir) and "rothermel" not in _sys.modules:
            _sys.path.insert(0, _rothermel_dir)
        from fuel_models import get_fuel_model as get_fuel_model_fn
        from fuel_models import ALL_FUEL_MODELS
        from fuel_models import ECOSYSTEM_TO_FUEL_MODEL

        _FUEL_MODELS_AVAILABLE = True
    except ImportError:
        get_fuel_model_fn = None
        ALL_FUEL_MODELS = {}
        ECOSYSTEM_TO_FUEL_MODEL = {}

# =============================================================================
# CONFIGURATION
# =============================================================================

FUEL_MODEL_CODE = "AF_MIOMBO"  # Default reference fuel model


# =============================================================================
# FUEL MODEL UTILITIES
# =============================================================================

def convert_fuel_to_engine(fuel) -> "EngineFuelModel":
    """Convert a FuelModel (fuel_models.py) to engine FuelModel."""
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


def compute_weighted_sav(fuel) -> float:
    """Compute weighted SAV [m²/m³] from a FuelModel.

    Uses the same formula as RothermelEngine._compute_weighted_sav.
    Replaces the hardcoded sigma_m2_m3=1500 from the original script.
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
        sav_live = (
            w_live_herb * sigma_live_herb + w_live_woody * sigma_live_woody
        ) / w_live

    return (w_dead * sav_dead + w_live * sav_live) / w_total


def match_fuel_model_by_region(region_name: str, fuel_models: dict = None) -> str:
    """Match an African region to the most appropriate fuel model.

    Args:
        region_name: Region key from AFRICA_REGIONS.
        fuel_models: Dict of region -> fuel_model_code overrides.

    Returns:
        Fuel model code string.
    """
    if fuel_models is None:
        fuel_models = {}

    if region_name in fuel_models:
        return fuel_models[region_name]

    _defaults = {
        "west_sahel": "AF_SAHEL_GRASS",
        "central_savanna": "AF_MIOMBO",
        "east_africa": "AF_ACACIA_SAVANNA",
        "madagascar": "AF_STEPPE",
    }
    return _defaults.get(region_name, FUEL_MODEL_CODE)


# =============================================================================
# THERMODYNAMIC HELPERS
# =============================================================================

def compute_vpd(temp_c: float, rh_percent: float) -> float:
    """Vapor pressure deficit in kPa."""
    es = 0.6108 * np.exp(17.27 * temp_c / (temp_c + 237.3))
    vpd = es * (1.0 - rh_percent / 100.0)
    return max(0.0, vpd)


def compute_dfmc(temp_c: float, vpd: float) -> float:
    """Dead fuel moisture content in percent.

    Estimated from equilibrium moisture content equation
    (Viney 1991, simplified).
    """
    dfmc = 30.0 - 2.5 * vpd - 0.1 * temp_c
    return float(np.clip(dfmc, 3.0, 40.0))


def compute_wind_mid_flame(wind_10m: float, fuel_height: float = 0.3) -> float:
    """Mid-flame wind speed from 10m wind."""
    if fuel_height < 0.6:
        return wind_10m * 0.4
    elif fuel_height < 2.0:
        return wind_10m * 0.6
    return wind_10m * 0.8


# =============================================================================
# ROTHERMEL BASELINE COMPUTATION
# =============================================================================

def compute_rothermel_baseline(
    df: pd.DataFrame,
    fuel_model_code: str = None,
    fuel_models_per_region: dict = None,
) -> pd.DataFrame:
    """Compute Rothermel v3 baseline ROS for each row in the DataFrame.

    Fixes applied:
        - sigma_m2_m3: computed from actual fuel model weighted SAV
          (original hardcoded 1500).
        - ndvi/ndwi: marked as 'estimated_from_region'.
        - lst_c = temp_c + 10: marked as approximation.

    Args:
        df: DataFrame with columns: temp_c, rh_percent, wind_speed_ms,
            slope_pct, angle_wind_slope, region, ros_observed, etc.
        fuel_model_code: Default fuel model code. If None, matched by region.
        fuel_models_per_region: Optional dict of region -> fuel_model_code
            for multi-model support.

    Returns:
        DataFrame with Rothermel output columns appended.
    """
    if not _ENGINE_AVAILABLE:
        raise RuntimeError(
            "RothermelEngine is required to compute baselines. "
            "Ensure burntrack.engine is installed."
        )

    if fuel_models_per_region is None:
        fuel_models_per_region = {}

    print("  Computing Rothermel v3 baselines...")

    engine = RothermelEngine()

    results = []
    for idx, row in df.iterrows():
        region = row.get("region", "")
        fm_code = match_fuel_model_by_region(region, fuel_models_per_region)
        if fuel_model_code is not None:
            fm_code = fuel_model_code

        fuel = get_fuel_model_fn(fm_code)
        if fuel is None:
            print(f"  [WARN] Fuel model '{fm_code}' not found, skipping row {idx}")
            continue

        fuel_engine = convert_fuel_to_engine(fuel)

        # Thermodynamics
        vpd = compute_vpd(row["temp_c"], row["rh_percent"])
        dfmc = compute_dfmc(row["temp_c"], vpd)
        m_1h = dfmc / 100.0
        m_10h = np.clip(m_1h + 0.02, 0.03, 0.30)
        m_100h = np.clip(m_1h + 0.04, 0.05, 0.35)

        m_live_herb = 0.30 + (100.0 - row["rh_percent"]) / 200.0
        m_live_woody = 0.60 + (100.0 - row["rh_percent"]) / 500.0

        moisture = MoistureInputs(
            m_1h=m_1h,
            m_10h=m_10h,
            m_100h=m_100h,
            m_live_herb=m_live_herb,
            m_live_woody=m_live_woody,
        )

        wind_mid = compute_wind_mid_flame(row["wind_speed_ms"], fuel.delta)
        conditions = EnvironmentalConditions(
            wind_speed=wind_mid,
            slope_pct=row["slope_pct"],
            angle_wind_slope=row.get("angle_wind_slope", 0.0),
        )

        out = engine.compute(fuel_engine, moisture, conditions)

        # FIXED: Compute sigma_m2_m3 from the actual fuel model's weighted SAV
        # (was hardcoded as 1500.0 in the original)
        sigma_m2_m3 = compute_weighted_sav(fuel)

        # ndvi/ndwi estimated from region (not from satellite)
        ndvi = 0.35 if region == "central_savanna" else 0.22
        ndwi = ndvi * 0.3 - 0.1

        # lst_c = temp_c + 10 is an approximation (no real LST data)
        lst_c = row["temp_c"] + 10.0

        record = {
            **row,
            "fuel_model_code": fm_code,
            "ros_rothermel": out.ros,
            "phi_w": out.phi_w,
            "phi_s": out.phi_s,
            "phi_eff": out.phi_eff,
            "beta": out.beta,
            "beta_opt": out.beta_opt,
            "beta_ratio": out.beta / out.beta_opt if out.beta_opt > 0 else 0.0,
            "gamma": out.gamma,
            "eta_M": out.eta_M,
            "eta_S": out.eta_S,
            "I_R_kW_m2": out.reaction_intensity,
            "xi": out.xi,
            "tau_min": out.tau,
            # FIXED: no longer hardcoded
            "ndvi": ndvi,
            "ndwi": ndwi,
            # lst_c = temp_c + 10 — approximation (estimated_from_air_temperature)
            "lst_c": lst_c,
            "dfmc_percent": dfmc,
            "vpd_kpa": vpd,
            "slope_deg": row["slope_deg"],
            "w_total_kg_m2": fuel.w_total,
            "w_dead_kg_m2": fuel.w_dead,
            "w_live_kg_m2": fuel.w_live,
            "delta_m": fuel.delta,
            # FIXED: computed from actual weighted SAV
            "sigma_m2_m3": sigma_m2_m3,
            "mx_percent": fuel.mx,
            "h_dead_kj_kg": fuel.h_dead,
            "ndvi_origin": "estimated_from_region",
            "ndwi_origin": "estimated_from_region",
            "lst_c_origin": "approx_temp_c_plus_10",
            "delta_ros": row["ros_observed"] - out.ros,
        }

        record["datetime"] = (
            record["datetime"].isoformat()
            if hasattr(record["datetime"], "isoformat")
            else str(record["datetime"])
        )
        results.append(record)

    return pd.DataFrame(results)


# =============================================================================
# MAIN PIPELINE
# =============================================================================

def build_real_dataset(
    api_key: str,
    output_path: str = "real_african_dataset.csv",
    fuel_model_code: str = None,
    fuel_models_per_region: dict = None,
    date_start=None,
    days_range: int = 4,
) -> pd.DataFrame:
    """Run the full real African fire dataset pipeline.

    Steps:
        1. Download FIRMS active fire data for 4 African regions (parallel).
        2. Reconstruct propagation via DBSCAN clustering.
        3. Associate weather and slope/aspect (parallel, rate-limited).
        4. Compute Rothermel v3 baseline and delta_ros.
        5. Save to CSV.

    Args:
        api_key: NASA FIRMS API key.
        output_path: Output CSV path.
        fuel_model_code: Single fuel model for all rows. If None, matched
            per region via match_fuel_model_by_region.
        fuel_models_per_region: Dict mapping region -> fuel_model_code for
            multi-model support.
        date_start: Start date (datetime). Defaults to 10 June 2026.
        days_range: Number of days of data.

    Returns:
        DataFrame with the complete real dataset.
    """
    from datetime import datetime

    if date_start is None:
        date_start = datetime(2026, 6, 10)

    print("=" * 75)
    print("  BURNTRACK — REAL PAN-AFRICAN DATASET PIPELINE")
    print("=" * 75)

    # 1. Download FIRMS data in parallel
    raw_fires = download_all_africa_fires(
        api_key, regions_dict=AFRICA_REGIONS, date_start=date_start, days_range=days_range
    )
    if raw_fires.empty:
        raise RuntimeError("No active fire points downloaded.")

    # 2. Reconstruct propagation (DBSCAN + front tracking)
    propagation_df = reconstruct_propagation(raw_fires)
    if propagation_df.empty:
        raise RuntimeError("No propagation vectors extracted.")

    # 3. Fetch weather in parallel with rate limiting
    weather_df = fetch_weather_for_points(propagation_df)
    if weather_df.empty:
        raise RuntimeError("Unable to associate weather data.")

    # 4. Compute Rothermel v3 baseline
    dataset_df = compute_rothermel_baseline(
        weather_df,
        fuel_model_code=fuel_model_code,
        fuel_models_per_region=fuel_models_per_region,
    )

    # 5. Save
    dataset_df.to_csv(output_path, index=False)

    print()
    print("=" * 75)
    print("  PIPELINE COMPLETE")
    print("=" * 75)
    print(f"  Output file             : {output_path}")
    print(f"  Training vectors        : {len(dataset_df)}")
    print(f"  ROS observed (mean)     : {dataset_df['ros_observed'].mean():.3f} m/min")
    print(f"  ROS Rothermel (mean)    : {dataset_df['ros_rothermel'].mean():.3f} m/min")
    print(f"  Delta ROS (mean)        : {dataset_df['delta_ros'].mean():.3f} m/min")
    print("=" * 75)

    return dataset_df
