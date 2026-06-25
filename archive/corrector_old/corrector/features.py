"""
Feature engineering for BurnTrack fire behavior correction.

This module provides two feature engineering interfaces:

1. ``FeaturesEngineering`` — Original robot-sensor-based features for real-time inference
2. ``CorrectorFeatureExtractor`` — Full 55+ feature vector for ML corrector training

The corrector feature vector includes:
  - Rothermel raw outputs (13 features)
  - Derived physics ratios (12 features)
  - Weather + fire weather indices (11 features)
  - Fuel characteristics (10 features)
  - Fuel moisture (5 features)
  - Satellite indices (4 features)
  - Topography (5 features)

Total: 60+ features before fuel model embedding.
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Dict, Optional, Tuple, List
import math
import logging

logger = logging.getLogger(__name__)


# ============================================================================
# Data classes for sensor/API inputs (preserved for backward compat)
# ============================================================================

@dataclass
class RobotSensors:
    """
    Raw sensor data from a field robot.
    
    Attributes:
        temp_air: Air temperature (°C)
        rh: Relative humidity (%)
        wind_speed: Wind speed measured by robot (m/s)
        slope_deg: Terrain slope (degrees)
        aspect_deg: Slope aspect (degrees, optional)
        surface_temp: Surface temperature (°C, optional, IR sensor)
        co_ppm: CO concentration (ppm, optional, MQ-135)
        co2_ppm: CO2 concentration (ppm, optional, MQ-135)
    """
    temp_air: float
    rh: float
    wind_speed: float
    slope_deg: float
    aspect_deg: Optional[float] = None
    surface_temp: Optional[float] = None
    co_ppm: Optional[float] = None
    co2_ppm: Optional[float] = None


@dataclass
class WeatherAPI:
    """
    Weather data from external API (Open-Meteo, ERA5, etc.).
    
    Attributes:
        temp_2m: Temperature at 2m (°C)
        wind_10m: Wind speed at 10m (m/s)
        wind_gust: Wind gust speed (m/s, optional)
        precip_1h: Precipitation over 1h (mm, optional)
        pressure: Atmospheric pressure (hPa, optional)
        dew_point: Dew point temperature (°C, optional)
    """
    temp_2m: float
    wind_10m: float
    wind_gust: Optional[float] = None
    precip_1h: Optional[float] = None
    pressure: Optional[float] = None
    dew_point: Optional[float] = None


@dataclass
class SatelliteData:
    """
    Satellite-derived data (Google Earth Engine or equivalent).
    
    Attributes:
        ndvi: Normalized Difference Vegetation Index [-1, 1]
        ndwi: Normalized Difference Water Index [-1, 1]
        lst: Land Surface Temperature (°C)
        evi: Enhanced Vegetation Index [-1, 1] (optional)
        savi: Soil Adjusted Vegetation Index [-1, 1] (optional)
        burned_area: Recent burned area (m², optional)
    """
    ndvi: float
    ndwi: float
    lst: float
    evi: Optional[float] = None
    savi: Optional[float] = None
    burned_area: Optional[float] = None


# ============================================================================
# Core physics functions (used by both old and new interfaces)
# ============================================================================

def compute_vpd(temp_c: float, rh_percent: float) -> float:
    """Vapor Pressure Deficit (kPa).
    
    VPD = e_s × (1 - RH/100) where e_s = 0.6108 × exp(17.27T / (T+237.3))
    
    Args:
        temp_c: Air temperature (°C)
        rh_percent: Relative humidity (%)
    
    Returns:
        VPD in kPa (≥ 0)
    """
    es = 0.6108 * np.exp(17.27 * temp_c / (temp_c + 237.3))
    ea = es * (rh_percent / 100.0)
    return max(0.0, es - ea)


def compute_dfmc(temp_c: float, vpd: float) -> float:
    """Dead Fuel Moisture Content (%) — Nelson 2000 equilibrium model.
    
    DFMC ≈ 30 - 2.5·VPD - 0.1·T, clipped to [3, 40]%.
    
    Args:
        temp_c: Air temperature (°C)
        vpd: Vapor pressure deficit (kPa)
    
    Returns:
        DFMC in %, bounded [3, 40]
    """
    dfmc = 30.0 - 2.5 * vpd - 0.1 * temp_c
    return float(np.clip(dfmc, 3.0, 40.0))


def compute_dead_fuel_moistures(dfmc_percent: float) -> Dict[str, float]:
    """Compute 1h, 10h, 100h dead fuel moistures from DFMC.
    
    Larger timelag classes have higher moisture due to slower drying.
    
    Args:
        dfmc_percent: Dead fuel moisture content (%)
    
    Returns:
        Dict with 'm_1h', 'm_10h', 'm_100h' as fractions [0, 1]
    """
    m_1h = dfmc_percent / 100.0
    m_10h = float(np.clip(m_1h + 0.02, 0.03, 0.35))
    m_100h = float(np.clip(m_1h + 0.04, 0.05, 0.40))
    return {'m_1h': m_1h, 'm_10h': m_10h, 'm_100h': m_100h}


def compute_live_herb_moisture(
    ndvi: float, rh_percent: float, month: int,
    hemisphere: str = 'south',
) -> float:
    """Estimate Live Herbaceous Moisture Content (LHMC) from NDVI.
    
    Based on the relationship: as NDVI increases (greener vegetation),
    LHMC increases. Curing occurs when NDVI drops.
    
    Rothermel curing stages:
      - Uncured: LHMC > 120% → no fuel load transfer
      - Partially cured: 98% < LHMC < 120%
      - Fully cured: LHMC < 98% → dead 1h fuel load transfer
    
    Args:
        ndvi: Normalized Difference Vegetation Index [-1, 1]
        rh_percent: Relative humidity (%)
        month: Month of year (1-12)
        hemisphere: 'south' or 'north' for seasonal adjustment
    
    Returns:
        LHMC as percentage, bounded [30, 250]
    """
    # NDVI-based LHMC estimation (literature-calibrated)
    base_lhmc = 30.0 + 170.0 * max(0.0, ndvi)
    
    # Seasonal adjustment
    if hemisphere == 'south':
        # Southern Africa: dry season = May-October
        seasonal_factor = 1.0 - 0.3 * np.sin(np.pi * (month - 1) / 6.0)
    else:
        # Sahel/North Africa: dry season = November-April
        seasonal_factor = 1.0 - 0.3 * np.sin(np.pi * (month + 5) / 6.0)
    
    # RH adjustment
    rh_factor = 0.5 + 0.5 * (rh_percent / 100.0)
    
    lhmc = base_lhmc * seasonal_factor * rh_factor
    return float(np.clip(lhmc, 30.0, 250.0))


def compute_wind_mid_flame(wind_10m: float, fuel_height: float = 0.3) -> float:
    """Convert 10m wind speed to midflame wind speed.
    
    U_mid ≈ U_10m × reduction_factor, depending on fuel bed depth.
    
    Args:
        wind_10m: Wind speed at 10m height (m/s)
        fuel_height: Fuel bed depth (m)
    
    Returns:
        Midflame wind speed (m/s)
    """
    if fuel_height < 0.6:
        return wind_10m * 0.4
    elif fuel_height < 2.0:
        return wind_10m * 0.6
    else:
        return wind_10m * 0.8


# ============================================================================
# Canadian Forest Fire Weather Index (FWI) System
# Reference: Van Wagner (1987)
# ============================================================================

def compute_ffmc(
    temp_c: float, rh_percent: float, wind_kmh: float, precip_mm: float,
    prev_ffmc: float = 85.0,
) -> float:
    """Fine Fuel Moisture Code (FFMC).
    
    Represents moisture content of litter and fine fuels on the forest floor.
    Range: 0-101, higher = drier. Values > 87 indicate high fire danger.
    
    Args:
        temp_c: Temperature (°C)
        rh_percent: Relative humidity (%)
        wind_kmh: Wind speed (km/h)
        precip_mm: 24h precipitation (mm)
        prev_ffmc: Previous day's FFMC (default: 85 for startup)
    
    Returns:
        FFMC value [0, 101]
    """
    # Convert previous FFMC to moisture content
    mo = 147.2 * (101.0 - prev_ffmc) / (59.5 + prev_ffmc)
    
    # Rain effect
    if precip_mm > 0.5:
        rf = precip_mm - 0.5
        if mo <= 150.0:
            mr = mo + 42.5 * rf * np.exp(-100.0 / (251.0 - mo)) * (1.0 - np.exp(-6.93 / rf))
        else:
            mr = mo + 42.5 * rf * np.exp(-100.0 / (251.0 - mo)) * (1.0 - np.exp(-6.93 / rf)) + \
                 0.0015 * (mo - 150.0)**2 * rf**0.5
        mo = min(mr, 250.0)
    
    # Drying/wetting phase
    # Equilibrium moisture contents
    ed = (0.942 * rh_percent**0.679 + 11.0 * np.exp((rh_percent - 100.0) / 10.0) +
          0.18 * (21.1 - temp_c) * (1.0 - np.exp(-0.115 * rh_percent)))
    
    ew = (0.618 * rh_percent**0.753 + 10.0 * np.exp((rh_percent - 100.0) / 10.0) +
          0.18 * (21.1 - temp_c) * (1.0 - np.exp(-0.115 * rh_percent)))
    
    if mo > ed:
        # Drying
        k0 = 0.424 * (1.0 - (rh_percent / 100.0)**1.7) + \
             0.0694 * wind_kmh**0.5 * (1.0 - (rh_percent / 100.0)**8)
        kd = k0 * 0.581 * np.exp(0.0365 * temp_c)
        m = ed + (mo - ed) * 10.0**(-kd)
    elif mo < ew:
        # Wetting
        k1 = 0.424 * (1.0 - ((100.0 - rh_percent) / 100.0)**1.7) + \
             0.0694 * wind_kmh**0.5 * (1.0 - ((100.0 - rh_percent) / 100.0)**8)
        kw = k1 * 0.581 * np.exp(0.0365 * temp_c)
        m = ew - (ew - mo) * 10.0**(-kw)
    else:
        m = mo
    
    # Convert back to FFMC
    ffmc = 59.5 * (250.0 - m) / (147.2 + m)
    return float(np.clip(ffmc, 0.0, 101.0))


def compute_dmc(
    temp_c: float, rh_percent: float, precip_mm: float,
    lat: float, month: int,
    prev_dmc: float = 6.0,
) -> float:
    """Duff Moisture Code (DMC).
    
    Represents moisture in loosely compacted organic layers (duff).
    Higher = drier. Range: 0 to ∞ (typically 0-300).
    
    Args:
        temp_c: Temperature (°C)
        rh_percent: Relative humidity (%)
        precip_mm: 24h precipitation (mm)
        lat: Latitude for day-length adjustment
        month: Month (1-12)
        prev_dmc: Previous day's DMC (default: 6)
    
    Returns:
        DMC value (≥ 0)
    """
    temp = max(temp_c, -1.1)  # Minimum effective temperature
    
    # Day-length adjustment factors (Van Wagner 1987, Table 1)
    day_length = _effective_day_length(lat, month)
    
    # Rain effect
    po = prev_dmc
    if precip_mm > 1.5:
        re = 0.92 * precip_mm - 1.27
        mo = 20.0 + np.exp(5.6348 - po / 43.43)
        if po <= 33.0:
            b = 100.0 / (0.5 + 0.3 * po)
        elif po <= 65.0:
            b = 14.0 - 1.3 * np.log(po)
        else:
            b = 6.2 * np.log(po) - 17.2
        mr = mo + 1000.0 * re / (48.77 + b * re)
        po = 244.72 - 43.43 * np.log(mr - 20.0)
        po = max(po, 0.0)
    
    # Drying phase
    if temp > -1.1:
        k = 1.894 * (temp + 1.1) * (100.0 - rh_percent) * day_length * 1e-6
    else:
        k = 0.0
    
    return max(0.0, po + 100.0 * k)


def compute_dc(
    temp_c: float, precip_mm: float,
    lat: float, month: int,
    prev_dc: float = 15.0,
) -> float:
    """Drought Code (DC).
    
    Represents moisture in deep, compact organic layers.
    Higher = drier/more drought. Range: 0 to ∞ (typically 0-800).
    
    Args:
        temp_c: Temperature (°C)
        precip_mm: 24h precipitation (mm)
        lat: Latitude
        month: Month (1-12)
        prev_dc: Previous day's DC (default: 15)
    
    Returns:
        DC value (≥ 0)
    """
    temp = max(temp_c, -2.8)
    
    # Day-length factor for DC
    fl = _dc_day_length_factor(lat, month)
    
    # Rain effect
    po = prev_dc
    if precip_mm > 2.8:
        rd = 0.83 * precip_mm - 1.27
        qo = 800.0 * np.exp(-po / 400.0)
        qr = qo + 3.937 * rd
        dr = 400.0 * np.log(800.0 / qr)
        po = max(dr, 0.0)
    
    # Drying
    if temp > -2.8:
        v = 0.36 * (temp + 2.8) + fl
        v = max(v, 0.0)
    else:
        v = 0.0
    
    return max(0.0, po + 0.5 * v)


def compute_isi(ffmc: float, wind_kmh: float) -> float:
    """Initial Spread Index (ISI).
    
    Combines FFMC and wind into a single index of fire spread rate.
    
    Args:
        ffmc: Fine Fuel Moisture Code
        wind_kmh: Wind speed (km/h)
    
    Returns:
        ISI value (≥ 0)
    """
    m = 147.2 * (101.0 - ffmc) / (59.5 + ffmc)
    fw = np.exp(0.05039 * wind_kmh)
    ff = 91.9 * np.exp(-0.1386 * m) * (1.0 + m**5.31 / (4.93e7))
    return 0.208 * fw * ff


def compute_bui(dmc: float, dc: float) -> float:
    """Buildup Index (BUI).
    
    Combines DMC and DC into an index of total fuel available.
    
    Args:
        dmc: Duff Moisture Code
        dc: Drought Code
    
    Returns:
        BUI value (≥ 0)
    """
    if dmc <= 0.4 * dc:
        bui = 0.8 * dmc * dc / (dmc + 0.4 * dc) if (dmc + 0.4 * dc) > 0 else 0.0
    else:
        bui = dmc - (1.0 - 0.8 * dc / (dmc + 0.4 * dc)) * \
              (0.92 + (0.0114 * dmc)**1.7) if (dmc + 0.4 * dc) > 0 else 0.0
    return max(0.0, bui)


def compute_fwi(isi: float, bui: float) -> float:
    """Fire Weather Index (FWI).
    
    Final fire danger rating combining ISI and BUI.
    
    Args:
        isi: Initial Spread Index
        bui: Buildup Index
    
    Returns:
        FWI value (≥ 0)
    """
    if bui <= 80.0:
        fd = 0.626 * bui**0.809 + 2.0
    else:
        fd = 1000.0 / (25.0 + 108.64 * np.exp(-0.023 * bui))
    
    b = 0.1 * isi * fd
    
    if b > 1.0:
        fwi = np.exp(2.72 * (0.434 * np.log(b))**0.647)
    else:
        fwi = b
    
    return max(0.0, fwi)


def compute_fire_weather_indices(
    temp_c: float, rh_percent: float, wind_speed_ms: float,
    precip_mm: float = 0.0,
    lat: float = 0.0, month: int = 1,
    prev_day_indices: Optional[Dict[str, float]] = None,
) -> Dict[str, float]:
    """Compute complete Canadian FWI System from weather observations.
    
    Args:
        temp_c: Temperature (°C)
        rh_percent: Relative humidity (%)
        wind_speed_ms: Wind speed (m/s) — converted to km/h internally
        precip_mm: 24h precipitation (mm)
        lat: Latitude for day-length adjustments
        month: Month (1-12)
        prev_day_indices: Previous day's FWI indices for continuity
    
    Returns:
        Dict with keys: 'fwi_ffmc', 'fwi_dmc', 'fwi_dc', 'fwi_isi', 'fwi_bui', 'fwi'
    """
    wind_kmh = wind_speed_ms * 3.6
    
    prev_ffmc = prev_day_indices.get('fwi_ffmc', 85.0) if prev_day_indices else 85.0
    prev_dmc = prev_day_indices.get('fwi_dmc', 6.0) if prev_day_indices else 6.0
    prev_dc = prev_day_indices.get('fwi_dc', 15.0) if prev_day_indices else 15.0
    
    ffmc = compute_ffmc(temp_c, rh_percent, wind_kmh, precip_mm, prev_ffmc)
    dmc = compute_dmc(temp_c, rh_percent, precip_mm, lat, month, prev_dmc)
    dc = compute_dc(temp_c, precip_mm, lat, month, prev_dc)
    isi = compute_isi(ffmc, wind_kmh)
    bui = compute_bui(dmc, dc)
    fwi = compute_fwi(isi, bui)
    
    return {
        'fwi_ffmc': ffmc,
        'fwi_dmc': dmc,
        'fwi_dc': dc,
        'fwi_isi': isi,
        'fwi_bui': bui,
        'fwi': fwi,
    }


# ============================================================================
# Helper functions for FWI
# ============================================================================

def _effective_day_length(lat: float, month: int) -> float:
    """Effective day-length for DMC calculation (Van Wagner 1987, Table 1)."""
    # Day-length factors by latitude band and month
    if lat > 0:
        # Northern hemisphere
        dl = [6.5, 7.5, 9.0, 12.8, 13.9, 13.9, 12.4, 10.9, 9.4, 8.0, 7.0, 6.0]
    else:
        # Southern hemisphere — shifted by 6 months
        dl = [10.9, 9.4, 8.0, 7.0, 6.0, 6.5, 7.5, 9.0, 12.8, 13.9, 13.9, 12.4]
    
    # Interpolate for latitude magnitude
    factor = min(abs(lat) / 45.0, 1.0)
    equatorial_dl = 9.0
    return equatorial_dl + factor * (dl[month - 1] - equatorial_dl)


def _dc_day_length_factor(lat: float, month: int) -> float:
    """Day-length factor for DC calculation."""
    if lat > 0:
        fl = [-1.6, -1.6, -1.6, 0.9, 3.8, 5.8, 6.4, 5.0, 2.4, 0.4, -1.6, -1.6]
    else:
        fl = [6.4, 5.0, 2.4, 0.4, -1.6, -1.6, -1.6, -1.6, -1.6, 0.9, 3.8, 5.8]
    return fl[month - 1]


# ============================================================================
# Derived physics features for corrector
# ============================================================================

def compute_derived_physics_features(row: dict) -> dict:
    """Compute derived physics features from Rothermel outputs and environment.
    
    These capture nonlinear interactions that the Rothermel model handles poorly,
    serving as corrective signals for the ML ensemble.
    
    Args:
        row: Dict containing Rothermel outputs and environmental data.
             Required keys: 'beta', 'beta_opt', 'phi_w', 'phi_s', 'phi_eff',
             'm_1h', 'm_10h', 'm_100h', 'm_live_herb', 'I_R_kW_m2'/'reaction_intensity',
             'tau_min'/'residence_time'/'tau', 'ros_rothermel'/'ros',
             'wind_speed_ms'/'wind_speed', 'angle_wind_slope', 'slope_pct',
             'fireline_intensity'
    
    Returns:
        Dict of derived feature values
    """
    features = {}
    
    # --- Rothermel residual ratios (model breakdown indicators) ---
    beta = row.get('beta', 0.0)
    beta_opt = row.get('beta_opt', 0.001)
    features['beta_beta_opt_ratio'] = beta / max(beta_opt, 1e-6)
    
    phi_w = row.get('phi_w', 0.0)
    phi_s = row.get('phi_s', 0.0)
    phi_eff = row.get('phi_eff', 0.0)
    features['phi_w_phi_eff_ratio'] = phi_w / max(phi_eff, 1e-6)
    features['phi_s_phi_eff_ratio'] = phi_s / max(phi_eff, 1e-6)
    features['wind_dominance'] = phi_w / max(phi_w + phi_s, 1e-6)
    
    # --- Fuel moisture gradients (timelag class dynamics) ---
    m_1h = row.get('m_1h', 0.05)
    m_10h = row.get('m_10h', 0.07)
    m_100h = row.get('m_100h', 0.09)
    m_live_herb = row.get('m_live_herb', 0.5)
    
    features['moisture_gradient_1h_10h'] = m_1h - m_10h
    features['moisture_gradient_10h_100h'] = m_10h - m_100h
    
    dead_avg = (m_1h + m_10h + m_100h) / 3.0
    features['moisture_dead_live_ratio'] = dead_avg / max(m_live_herb, 0.01)
    
    # --- Energy balance features ---
    I_R = row.get('I_R_kW_m2', row.get('reaction_intensity', 0.0))
    tau = row.get('tau_min', row.get('residence_time', row.get('tau', 0.1)))
    features['energy_release_rate'] = I_R * max(tau, 0.001)
    
    ros = row.get('ros_rothermel', row.get('ros', 0.01))
    fi = row.get('fireline_intensity', 0.01)
    features['ros_to_intensity_ratio'] = ros / max(fi, 0.01)
    
    # --- Wind-fuel interaction ---
    wind = row.get('wind_speed_ms', row.get('wind_speed', 0.0))
    sigma = row.get('sigma_m2_m3', row.get('sigma_1h', 0.0))
    features['wind_sav_product'] = wind * sigma
    features['phi_w_per_unit_wind'] = phi_w / max(wind, 0.1)
    
    # --- Topographic ---
    angle_ws = row.get('angle_wind_slope', 0.0)
    features['wind_slope_alignment'] = float(np.cos(np.radians(angle_ws)))
    
    slope_pct = row.get('slope_pct', 0.0)
    features['slope_effectiveness'] = phi_s / max(slope_pct, 0.1) if slope_pct > 0 else 0.0
    
    return features


# ============================================================================
# Complete feature extraction for corrector training/inference
# ============================================================================

# Standard feature names in order, used for building feature matrices
ROTHERMEL_FEATURES = [
    'ros_rothermel', 'phi_w', 'phi_s', 'phi_eff',
    'beta', 'beta_opt', 'gamma', 'eta_M', 'eta_S',
    'I_R_kW_m2', 'xi', 'tau_min', 'fireline_intensity',
]

DERIVED_PHYSICS_FEATURES = [
    'beta_beta_opt_ratio', 'phi_w_phi_eff_ratio', 'phi_s_phi_eff_ratio',
    'wind_dominance', 'moisture_gradient_1h_10h', 'moisture_gradient_10h_100h',
    'moisture_dead_live_ratio', 'energy_release_rate', 'ros_to_intensity_ratio',
    'wind_sav_product', 'phi_w_per_unit_wind', 'wind_slope_alignment',
]

WEATHER_FEATURES = [
    'temp_c', 'rh_percent', 'wind_speed_ms', 'vpd_kpa',
    'fwi_ffmc', 'fwi_dmc', 'fwi_dc', 'fwi_isi', 'fwi_bui', 'fwi',
]

FUEL_FEATURES = [
    'w_total_kg_m2', 'w_dead_kg_m2', 'w_live_kg_m2', 'w_live_ratio',
    'delta_m', 'sigma_m2_m3', 'mx_percent', 'h_dead_kj_kg',
]

MOISTURE_FEATURES = [
    'm_1h', 'm_10h', 'm_100h', 'm_live_herb', 'm_live_woody',
]

SATELLITE_FEATURES = [
    'ndvi', 'ndwi', 'lst_c', 'ndvi_anomaly',
]

TOPOGRAPHY_FEATURES = [
    'slope_pct', 'slope_deg', 'aspect_deg', 'angle_wind_slope',
]

# Full ordered list of continuous features (excluding fuel_model embedding)
ALL_CONTINUOUS_FEATURES = (
    ROTHERMEL_FEATURES +
    DERIVED_PHYSICS_FEATURES +
    WEATHER_FEATURES +
    FUEL_FEATURES +
    MOISTURE_FEATURES +
    SATELLITE_FEATURES +
    TOPOGRAPHY_FEATURES
)

# NDVI reference stats for anomaly computation
NDVI_REFERENCE_MEAN = 0.35
NDVI_REFERENCE_STD = 0.15


class CorrectorFeatureExtractor:
    """Complete feature extractor for the BurnTrack corrector ensemble.
    
    Extracts 60+ features from raw fire observation data including:
    - Rothermel physics outputs
    - Derived physics ratios and interactions
    - Canadian FWI system indices
    - Fuel characteristics
    - Satellite-derived indices
    - Topographic features
    
    Example::
    
        extractor = CorrectorFeatureExtractor()
        features = extractor.extract_row(row_dict)
        feature_matrix = extractor.extract_dataframe(df)
    """
    
    def __init__(
        self,
        feature_names: Optional[List[str]] = None,
        ndvi_ref_mean: float = NDVI_REFERENCE_MEAN,
        ndvi_ref_std: float = NDVI_REFERENCE_STD,
    ):
        """Initialize the feature extractor.
        
        Args:
            feature_names: Explicit list of features to extract. If None, uses all.
            ndvi_ref_mean: Reference NDVI mean for anomaly calculation
            ndvi_ref_std: Reference NDVI std for anomaly calculation
        """
        self.feature_names = feature_names or list(ALL_CONTINUOUS_FEATURES)
        self.ndvi_ref_mean = ndvi_ref_mean
        self.ndvi_ref_std = ndvi_ref_std
    
    def extract_row(self, row: dict) -> dict:
        """Extract all features from a single observation row.
        
        Args:
            row: Dict containing raw observation data. Expected keys vary
                 depending on which features are requested.
        
        Returns:
            Dict mapping feature names to float values
        """
        features = {}
        
        # --- Rothermel raw outputs ---
        features['ros_rothermel'] = row.get('ros_rothermel', row.get('ros', 0.0))
        features['phi_w'] = row.get('phi_w', 0.0)
        features['phi_s'] = row.get('phi_s', 0.0)
        features['phi_eff'] = row.get('phi_eff', 0.0)
        features['beta'] = row.get('beta', 0.0)
        features['beta_opt'] = row.get('beta_opt', 0.0)
        features['gamma'] = row.get('gamma', 0.0)
        features['eta_M'] = row.get('eta_M', 0.0)
        features['eta_S'] = row.get('eta_S', 0.0)
        features['I_R_kW_m2'] = row.get('I_R_kW_m2', row.get('reaction_intensity', 0.0))
        features['xi'] = row.get('xi', 0.0)
        features['tau_min'] = row.get('tau_min', row.get('residence_time', row.get('tau', 0.0)))
        features['fireline_intensity'] = row.get('fireline_intensity', 0.0)
        
        # --- Derived physics ---
        derived = compute_derived_physics_features(row)
        features.update(derived)
        
        # --- Weather ---
        temp_c = row.get('temp_c', 25.0)
        rh = row.get('rh_percent', 50.0)
        wind = row.get('wind_speed_ms', row.get('wind_speed', 0.0))
        
        features['temp_c'] = temp_c
        features['rh_percent'] = rh
        features['wind_speed_ms'] = wind
        
        vpd = row.get('vpd_kpa', compute_vpd(temp_c, rh))
        features['vpd_kpa'] = vpd
        
        # FWI indices
        lat = row.get('latitude', row.get('lat', 0.0))
        date_str = row.get('date', row.get('datetime', ''))
        month = _extract_month(date_str) if date_str else 1
        precip = row.get('precip_mm', 0.0)
        
        fwi_indices = compute_fire_weather_indices(
            temp_c, rh, wind, precip, lat, month,
            prev_day_indices=row.get('prev_day_indices'),
        )
        features.update(fwi_indices)
        
        # --- Fuel characteristics ---
        features['w_total_kg_m2'] = row.get('w_total_kg_m2', 0.0)
        features['w_dead_kg_m2'] = row.get('w_dead_kg_m2', 0.0)
        features['w_live_kg_m2'] = row.get('w_live_kg_m2', 0.0)
        w_total = features['w_total_kg_m2']
        features['w_live_ratio'] = features['w_live_kg_m2'] / max(w_total, 0.001)
        features['delta_m'] = row.get('delta_m', row.get('delta', 0.0))
        features['sigma_m2_m3'] = row.get('sigma_m2_m3', row.get('sigma_1h', 0.0))
        features['mx_percent'] = row.get('mx_percent', row.get('mx', 15.0))
        features['h_dead_kj_kg'] = row.get('h_dead_kj_kg', row.get('h_dead', 18608.0))
        
        # --- Fuel moisture ---
        dfmc = row.get('dfmc_percent', compute_dfmc(temp_c, vpd))
        dead_moistures = compute_dead_fuel_moistures(dfmc)
        features['m_1h'] = row.get('m_1h', dead_moistures['m_1h'])
        features['m_10h'] = row.get('m_10h', dead_moistures['m_10h'])
        features['m_100h'] = row.get('m_100h', dead_moistures['m_100h'])
        
        ndvi = row.get('ndvi', 0.3)
        hemisphere = 'south' if lat < 0 else 'north'
        lhmc = compute_live_herb_moisture(ndvi, rh, month, hemisphere)
        features['m_live_herb'] = row.get('m_live_herb', lhmc / 100.0)
        features['m_live_woody'] = row.get('m_live_woody', 0.5)
        
        # --- Satellite ---
        features['ndvi'] = ndvi
        features['ndwi'] = row.get('ndwi', -0.1)
        features['lst_c'] = row.get('lst_c', row.get('lst', temp_c + 10.0))
        features['ndvi_anomaly'] = (ndvi - self.ndvi_ref_mean) / max(self.ndvi_ref_std, 0.01)
        
        # --- Topography ---
        features['slope_pct'] = row.get('slope_pct', 0.0)
        slope_deg = row.get('slope_deg', np.degrees(np.arctan(row.get('slope_pct', 0.0) / 100.0)))
        features['slope_deg'] = slope_deg
        features['aspect_deg'] = row.get('aspect_deg', row.get('slope_aspect_deg', 0.0))
        features['angle_wind_slope'] = row.get('angle_wind_slope', 0.0)
        
        # Filter to only requested features
        return {k: float(v) for k, v in features.items() if k in self.feature_names}
    
    def extract_dataframe(
        self, df, include_fuel_idx: bool = True
    ) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """Extract feature matrix from a DataFrame.
        
        Args:
            df: pandas DataFrame with observation rows
            include_fuel_idx: Whether to also return fuel model indices
        
        Returns:
            Tuple of (feature_matrix [n_samples, n_features], fuel_indices [n_samples])
            fuel_indices is None if include_fuel_idx=False or no fuel_model_code column
        """
        rows = []
        fuel_indices = []
        
        for idx, row_series in df.iterrows():
            row_dict = row_series.to_dict()
            features = self.extract_row(row_dict)
            # Ensure consistent ordering
            row_vec = [features.get(name, 0.0) for name in self.feature_names]
            rows.append(row_vec)
            
            if include_fuel_idx:
                fuel_code = row_dict.get('fuel_model_code', '')
                fuel_indices.append(_encode_fuel_model(fuel_code))
        
        X = np.array(rows, dtype=np.float64)
        
        if include_fuel_idx and fuel_indices:
            fuel_idx = np.array(fuel_indices, dtype=np.int64)
        else:
            fuel_idx = None
        
        return X, fuel_idx
    
    def get_feature_names(self) -> List[str]:
        """Return ordered list of feature names."""
        return list(self.feature_names)
    
    def get_n_features(self) -> int:
        """Return number of continuous features."""
        return len(self.feature_names)


# ============================================================================
# Fuel model encoding
# ============================================================================

# Complete fuel model encoding (matching mlp.py FUEL_MODEL_ENCODING)
FUEL_MODEL_ENCODING = {
    # BEHAVE Standard — GR (9 codes)
    'GR1': 0, 'GR2': 1, 'GR3': 2, 'GR4': 3, 'GR5': 4,
    'GR6': 5, 'GR7': 6, 'GR8': 7, 'GR9': 8,
    # BEHAVE Standard — GS (4 codes)
    'GS1': 9, 'GS2': 10, 'GS3': 11, 'GS4': 12,
    # BEHAVE Standard — SH (9 codes)
    'SH1': 13, 'SH2': 14, 'SH3': 15, 'SH4': 16, 'SH5': 17,
    'SH6': 18, 'SH7': 19, 'SH8': 20, 'SH9': 21,
    # North Africa (10 codes)
    'AF_STEPPE': 22, 'AF_STEPPE_DENSE': 23, 'AF_ARGAN': 24,
    'AF_CHENE_LIEGE': 25, 'AF_CEDRE': 26, 'AF_MAQUIS': 27,
    'AF_CEREALES': 28, 'AF_PALMIER': 29, 'AF_TAMARIX': 30, 'AF_JUJUBIER': 31,
    # Sub-Saharan Africa (18 codes)
    'AF_SAHEL_GRASS': 32, 'AF_SAHEL_WOODED': 33,
    'AF_SUDAN_GRASS': 34, 'AF_SUDAN_WOODED': 35,
    'AF_MIOMBO': 36, 'AF_MIOMBO_DENSE': 37, 'AF_MOPANE': 38,
    'AF_ACACIA_SAVANNA': 39, 'AF_GRASSLAND_FERTILE': 40,
    'AF_FYNBOS': 41, 'AF_FYNBOS_YOUNG': 42,
    'AF_BUSHVELD': 43, 'AF_BAOBAB': 44, 'AF_FOREST_DRY': 45,
    'AF_AFROMONTANE': 46, 'AF_MANGROVE': 47,
    'AF_RANGE_DEGRADED': 48, 'AF_RANGE_INTACT': 49,
}

N_FUEL_MODELS = len(FUEL_MODEL_ENCODING)

# Reverse mapping
FUEL_MODEL_DECODING = {v: k for k, v in FUEL_MODEL_ENCODING.items()}


def _encode_fuel_model(code: str) -> int:
    """Encode a fuel model code to integer index."""
    return FUEL_MODEL_ENCODING.get(code, 0)


def _extract_month(date_str: str) -> int:
    """Extract month from various date string formats."""
    if not date_str:
        return 1
    try:
        # Try ISO format YYYY-MM-DD
        parts = str(date_str).split('-')
        if len(parts) >= 2:
            return int(parts[1])
        # Try datetime with T separator
        if 'T' in str(date_str):
            return int(str(date_str).split('T')[0].split('-')[1])
    except (ValueError, IndexError):
        pass
    return 1


# ============================================================================
# Original FeaturesEngineering class (preserved for backward compatibility)
# ============================================================================

class FeaturesEngineering:
    """Original sensor-based feature engineering (backward compatible).
    
    Computes derived features from robot sensors, weather API, and satellite
    data for real-time inference on the robot.
    """
    
    def __init__(self):
        self.ndvi_reference_mean = NDVI_REFERENCE_MEAN
        self.ndvi_reference_std = NDVI_REFERENCE_STD
    
    def compute_vpd(self, temp_air: float, rh: float) -> float:
        """Compute Vapor Pressure Deficit (VPD) in kPa."""
        return compute_vpd(temp_air, rh)
    
    def compute_dfmc(self, temp_air: float, vpd: float) -> float:
        """Compute Dead Fuel Moisture Content (DFMC) in %."""
        return compute_dfmc(temp_air, vpd)
    
    def compute_dfmc_precip_adjusted(
        self, dfmc_base: float, precip_1h: Optional[float],
        hours_since_rain: int = 0,
    ) -> float:
        """Adjust DFMC for recent precipitation."""
        if precip_1h is None or precip_1h <= 0:
            drying_factor = 1.0 - np.exp(-0.1 * hours_since_rain)
            return min(dfmc_base + 10.0 * (1.0 - drying_factor), 40.0)
        moisture_increase = min(precip_1h * 5.0, 20.0)
        return min(dfmc_base + moisture_increase, 40.0)
    
    def compute_delta_t_surf_air(
        self, surface_temp: Optional[float], temp_air: float,
    ) -> float:
        """Compute surface-air temperature difference."""
        if surface_temp is None:
            return 0.0
        return surface_temp - temp_air
    
    def compute_wind_ratio(self, wind_robot: float, wind_10m: float) -> float:
        """Compute robot/weather wind ratio."""
        if wind_10m < 0.1:
            return 1.0
        ratio = wind_robot / wind_10m
        return float(np.clip(ratio, 0.1, 3.0))
    
    def compute_wind_mid_flame(self, wind_10m: float, fuel_height: float = 0.3) -> float:
        """Convert 10m wind to midflame wind speed."""
        return compute_wind_mid_flame(wind_10m, fuel_height)
    
    def compute_stress_index(
        self, vpd: float, ndvi: float, lst: float, temp_air: float,
    ) -> float:
        """Compute vegetation stress index [0, 1]."""
        vpd_term = min(vpd / 5.0, 2.0)
        ndvi_term = max(0.0, 1.0 - ndvi)
        temp_term = max(0.0, (lst - temp_air) / 10.0)
        stress = vpd_term * ndvi_term * temp_term
        return float(np.clip(stress, 0.0, 1.0))
    
    def compute_ndvi_anomaly(self, ndvi: float) -> float:
        """Compute NDVI anomaly (z-score)."""
        if self.ndvi_reference_std == 0:
            return 0.0
        return (ndvi - self.ndvi_reference_mean) / self.ndvi_reference_std
    
    def compute_danger_proxy(
        self, vpd: float, dfmc: float, wind_speed: float, ndvi: float,
    ) -> float:
        """Compute fire danger proxy index [0, 1]."""
        vpd_score = min(vpd / 5.0, 1.0)
        dfmc_score = max(0.0, 1.0 - dfmc / 25.0)
        wind_score = min(wind_speed / 10.0, 1.0)
        ndvi_score = max(0.0, 1.0 - abs(ndvi - 0.3) / 0.7)
        danger = 0.3 * vpd_score + 0.3 * dfmc_score + 0.25 * wind_score + 0.15 * ndvi_score
        return float(np.clip(danger, 0.0, 1.0))
    
    def compute_all_features(
        self, robot: RobotSensors, weather: WeatherAPI,
        satellite: SatelliteData, fuel_height: float = 0.3,
    ) -> Dict[str, float]:
        """Compute all features in one pass (original interface)."""
        vpd = self.compute_vpd(robot.temp_air, robot.rh)
        dfmc = self.compute_dfmc(robot.temp_air, vpd)
        dfmc_adj = self.compute_dfmc_precip_adjusted(
            dfmc, weather.precip_1h, hours_since_rain=0
        )
        delta_t = self.compute_delta_t_surf_air(robot.surface_temp, robot.temp_air)
        wind_ratio = self.compute_wind_ratio(robot.wind_speed, weather.wind_10m)
        wind_mid = self.compute_wind_mid_flame(weather.wind_10m, fuel_height)
        stress = self.compute_stress_index(vpd, satellite.ndvi, satellite.lst, robot.temp_air)
        ndvi_anomaly = self.compute_ndvi_anomaly(satellite.ndvi)
        danger_proxy = self.compute_danger_proxy(vpd, dfmc, robot.wind_speed, satellite.ndvi)
        
        return {
            'temp_air': robot.temp_air,
            'rh': robot.rh,
            'vpd': round(vpd, 3),
            'dfmc': round(dfmc, 3),
            'dfmc_adjusted': round(dfmc_adj, 3),
            'wind_speed': robot.wind_speed,
            'wind_mid_flame': round(wind_mid, 3),
            'slope_deg': robot.slope_deg,
            'temp_2m': weather.temp_2m,
            'wind_10m': weather.wind_10m,
            'wind_gust': weather.wind_gust if weather.wind_gust else 0.0,
            'precip_1h': weather.precip_1h if weather.precip_1h else 0.0,
            'pressure': weather.pressure if weather.pressure else 1013.25,
            'ndvi': satellite.ndvi,
            'ndwi': satellite.ndwi,
            'lst': satellite.lst,
            'evi': satellite.evi if satellite.evi else satellite.ndvi,
            'savi': satellite.savi if satellite.savi else satellite.ndvi,
            'delta_t_surf_air': round(delta_t, 3),
            'wind_ratio': round(wind_ratio, 3),
            'stress_index': round(stress, 3),
            'ndvi_anomaly': round(ndvi_anomaly, 3),
            'danger_proxy': round(danger_proxy, 3),
        }


if __name__ == "__main__":
    print("=" * 60)
    print("CORRECTOR FEATURE EXTRACTOR — TEST")
    print("=" * 60)
    
    # Test the new CorrectorFeatureExtractor
    extractor = CorrectorFeatureExtractor()
    
    test_row = {
        'ros_rothermel': 5.5,
        'phi_w': 4.9,
        'phi_s': 0.007,
        'phi_eff': 4.9,
        'beta': 0.005,
        'beta_opt': 0.006,
        'gamma': 15.5,
        'eta_M': 0.54,
        'eta_S': 0.42,
        'I_R_kW_m2': 1503.0,
        'xi': 0.041,
        'tau_min': 0.17,
        'fireline_intensity': 120.0,
        'temp_c': 38.0,
        'rh_percent': 20.0,
        'wind_speed_ms': 4.0,
        'slope_pct': 5.0,
        'slope_deg': 2.86,
        'aspect_deg': 180.0,
        'angle_wind_slope': 30.0,
        'ndvi': 0.25,
        'ndwi': -0.15,
        'lst_c': 48.0,
        'w_total_kg_m2': 1.3,
        'w_dead_kg_m2': 0.75,
        'w_live_kg_m2': 0.55,
        'delta_m': 0.5,
        'sigma_m2_m3': 1500.0,
        'mx_percent': 25.0,
        'h_dead_kj_kg': 19500.0,
        'fuel_model_code': 'AF_MIOMBO',
        'date': '2024-08-15',
        'latitude': -12.0,
    }
    
    features = extractor.extract_row(test_row)
    
    print(f"\n--- {len(features)} FEATURES EXTRACTED ---")
    for name, value in sorted(features.items()):
        print(f"  {name:30s}: {value:.4f}")
    
    print(f"\n--- FWI INDICES ---")
    fwi_features = {k: v for k, v in features.items() if k.startswith('fwi')}
    for k, v in fwi_features.items():
        print(f"  {k}: {v:.2f}")
    
    print(f"\n--- DERIVED PHYSICS ---")
    for name in DERIVED_PHYSICS_FEATURES:
        if name in features:
            print(f"  {name:30s}: {features[name]:.4f}")
