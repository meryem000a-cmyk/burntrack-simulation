"""
fetch_satellite.py — Satellite data retrieval for BurnTrack.
=============================================================

Provides NDVI, NDWI, LST, and land-cover retrieval with a dual approach:

    1. **Primary**: Google Earth Engine Python API (``earthengine-api``)
       using Sentinel-2, MODIS, and Copernicus datasets.
    2. **Fallback**: Offline, climate-based estimation from latitude, month,
       and optional fuel model code — no internet required.

The offline fallback produces reasonable estimates calibrated against
published remote-sensing literature on African biomes (Archibald et al.
2010; Rouse et al. 1974; Huete et al. 2002; Eva & Lambin 2000).

Usage::

    from burntrack.data.fetch_satellite import fetch_ndvi, fetch_ndwi, fetch_lst

    ndvi = fetch_ndvi(lat=12.5, lon=-1.5, date='2024-01-15',
                      fuel_model_code='AF_SAHEL_GRASS')
    ndwi = fetch_ndwi(lat=12.5, lon=-1.5, date='2024-01-15')
    lst  = fetch_lst(lat=12.5, lon=-1.5, date='2024-01-15', temp_c=35.0)
"""

import logging
import math
import warnings
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple

import numpy as np

try:
    import pandas as pd
    _PANDAS_AVAILABLE = True
except ImportError:
    _PANDAS_AVAILABLE = False

logger = logging.getLogger(__name__)

# ============================================================================
# GEE initialisation (lazy)
# ============================================================================

_GEE_INITIALISED: Optional[bool] = None  # None = not tried yet


def _init_gee() -> bool:
    """Initialise Google Earth Engine.  Returns ``True`` on success.

    The result is cached so repeated calls are cheap.  Authentication is
    attempted via ``ee.Initialize(opt_url=...)`` which works for service
    accounts and ``earthengine authenticate``-d users.
    """
    global _GEE_INITIALISED
    if _GEE_INITIALISED is not None:
        return _GEE_INITIALISED

    try:
        import ee  # type: ignore[import-untyped]
        ee.Initialize(opt_url="https://earthengine-highvolume.googleapis.com")
        _GEE_INITIALISED = True
        logger.info("Google Earth Engine initialised successfully.")
    except Exception as exc:
        _GEE_INITIALISED = False
        logger.warning(
            "GEE initialisation failed (%s). Using offline fallback.", exc
        )
    return _GEE_INITIALISED


# ============================================================================
# GEE data functions
# ============================================================================

def fetch_ndvi_gee(
    lat: float,
    lon: float,
    date: str,
    buffer_m: int = 500,
) -> Optional[float]:
    """Fetch NDVI from Sentinel-2 or MODIS via GEE.

    Strategy:
        1. Try Sentinel-2 SR Harmonized (10 m, ~5-day revisit).
        2. Fall back to MODIS MOD13Q1 (250 m, 16-day composite).

    A 30-day look-back window before *date* is used to find the most
    recent cloud-free observation.

    Args:
        lat: Latitude in decimal degrees.
        lon: Longitude in decimal degrees.
        date: ISO-format date string (YYYY-MM-DD).
        buffer_m: Buffer radius in metres for spatial averaging.

    Returns:
        Median NDVI in [-1, 1] or ``None`` if retrieval fails.
    """
    if not _init_gee():
        return None

    try:
        import ee  # type: ignore[import-untyped]

        point = ee.Geometry.Point([lon, lat])
        region = point.buffer(buffer_m)
        end_date = ee.Date(date)
        start_date = end_date.advance(-30, "day")

        # ---- Sentinel-2 attempt ----
        try:
            s2 = (
                ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED")
                .filterBounds(region)
                .filterDate(start_date, end_date)
                .filter(ee.Filter.lt("CLOUDY_PIXEL_PERCENTAGE", 30))
                .select(["B4", "B8"])  # Red, NIR
            )

            if s2.size().getInfo() > 0:
                median = s2.median()
                ndvi = median.normalizedDifference(["B8", "B4"]).rename("NDVI")
                stats = ndvi.reduceRegion(
                    reducer=ee.Reducer.median(),
                    geometry=region,
                    scale=10,
                    maxPixels=1e6,
                )
                val = stats.get("NDVI").getInfo()
                if val is not None and -1 <= val <= 1:
                    logger.debug(
                        "NDVI from Sentinel-2: %.3f at (%.4f, %.4f)", val, lat, lon
                    )
                    return float(val)
        except Exception as s2_exc:
            logger.debug("Sentinel-2 NDVI failed: %s", s2_exc)

        # ---- MODIS MOD13Q1 fallback ----
        try:
            modis = (
                ee.ImageCollection("MODIS/061/MOD13Q1")
                .filterBounds(region)
                .filterDate(start_date, end_date)
                .select(["NDVI"])
            )

            if modis.size().getInfo() > 0:
                median = modis.median()
                # MOD13Q1 NDVI has scale factor 0.0001
                scaled = median.multiply(0.0001)
                stats = scaled.reduceRegion(
                    reducer=ee.Reducer.median(),
                    geometry=region,
                    scale=250,
                    maxPixels=1e6,
                )
                val = stats.get("NDVI").getInfo()
                if val is not None and -1 <= val <= 1:
                    logger.debug(
                        "NDVI from MODIS MOD13Q1: %.3f at (%.4f, %.4f)",
                        val, lat, lon,
                    )
                    return float(val)
        except Exception as modis_exc:
            logger.debug("MODIS NDVI failed: %s", modis_exc)

    except Exception as exc:
        logger.warning("GEE NDVI retrieval error: %s", exc)

    return None


def fetch_ndwi_gee(
    lat: float,
    lon: float,
    date: str,
    buffer_m: int = 500,
) -> Optional[float]:
    """Fetch NDWI (vegetation moisture) from Sentinel-2 or MODIS via GEE.

    Uses the McFeeters-style NDWI:  (Green − NIR) / (Green + NIR)
    and Gao-style moisture index:   (NIR − SWIR) / (NIR + SWIR).

    For fire-behaviour purposes the Gao-style *moisture* NDWI using the
    SWIR band (B11 on Sentinel-2, band 6 on MODIS) is preferred.

    Args:
        lat: Latitude in decimal degrees.
        lon: Longitude in decimal degrees.
        date: ISO-format date string (YYYY-MM-DD).
        buffer_m: Buffer radius in metres for spatial averaging.

    Returns:
        NDWI in [-1, 1] or ``None`` if unavailable.
    """
    if not _init_gee():
        return None

    try:
        import ee  # type: ignore[import-untyped]

        point = ee.Geometry.Point([lon, lat])
        region = point.buffer(buffer_m)
        end_date = ee.Date(date)
        start_date = end_date.advance(-30, "day")

        # ---- Sentinel-2 (Gao moisture NDWI = (B8 - B11)/(B8 + B11)) ----
        try:
            s2 = (
                ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED")
                .filterBounds(region)
                .filterDate(start_date, end_date)
                .filter(ee.Filter.lt("CLOUDY_PIXEL_PERCENTAGE", 30))
                .select(["B8", "B11"])  # NIR, SWIR-1
            )

            if s2.size().getInfo() > 0:
                median = s2.median()
                ndwi = median.normalizedDifference(["B8", "B11"]).rename("NDWI")
                stats = ndwi.reduceRegion(
                    reducer=ee.Reducer.median(),
                    geometry=region,
                    scale=20,  # B11 is 20 m
                    maxPixels=1e6,
                )
                val = stats.get("NDWI").getInfo()
                if val is not None and -1 <= val <= 1:
                    logger.debug(
                        "NDWI from Sentinel-2: %.3f at (%.4f, %.4f)",
                        val, lat, lon,
                    )
                    return float(val)
        except Exception as s2_exc:
            logger.debug("Sentinel-2 NDWI failed: %s", s2_exc)

        # ---- MODIS MOD09GA (NIR=band 2, SWIR=band 6) ----
        try:
            modis = (
                ee.ImageCollection("MODIS/061/MOD09GA")
                .filterBounds(region)
                .filterDate(start_date, end_date)
                .select(["sur_refl_b02", "sur_refl_b06"])
            )

            if modis.size().getInfo() > 0:
                median = modis.median()
                ndwi = median.normalizedDifference(
                    ["sur_refl_b02", "sur_refl_b06"]
                ).rename("NDWI")
                stats = ndwi.reduceRegion(
                    reducer=ee.Reducer.median(),
                    geometry=region,
                    scale=500,
                    maxPixels=1e6,
                )
                val = stats.get("NDWI").getInfo()
                if val is not None and -1 <= val <= 1:
                    logger.debug(
                        "NDWI from MODIS MOD09GA: %.3f at (%.4f, %.4f)",
                        val, lat, lon,
                    )
                    return float(val)
        except Exception as modis_exc:
            logger.debug("MODIS NDWI failed: %s", modis_exc)

    except Exception as exc:
        logger.warning("GEE NDWI retrieval error: %s", exc)

    return None


def fetch_lst_gee(
    lat: float,
    lon: float,
    date: str,
) -> Optional[float]:
    """Fetch daytime Land Surface Temperature from MODIS MOD11A1 via GEE.

    The ``LST_Day_1km`` band is stored as scaled integers:
    multiply by **0.02** to get Kelvin, then convert to Celsius.

    Args:
        lat: Latitude in decimal degrees.
        lon: Longitude in decimal degrees.
        date: ISO-format date string (YYYY-MM-DD).

    Returns:
        LST in °C or ``None`` if unavailable.
    """
    if not _init_gee():
        return None

    try:
        import ee  # type: ignore[import-untyped]

        point = ee.Geometry.Point([lon, lat])
        end_date = ee.Date(date)
        start_date = end_date.advance(-8, "day")

        modis_lst = (
            ee.ImageCollection("MODIS/061/MOD11A1")
            .filterBounds(point)
            .filterDate(start_date, end_date)
            .select(["LST_Day_1km"])
        )

        if modis_lst.size().getInfo() == 0:
            logger.debug("No MODIS LST images found for (%.4f, %.4f).", lat, lon)
            return None

        median = modis_lst.median()
        # Scale: DN * 0.02 → Kelvin
        lst_k = median.multiply(0.02)
        stats = lst_k.reduceRegion(
            reducer=ee.Reducer.median(),
            geometry=point.buffer(1000),
            scale=1000,
            maxPixels=1e4,
        )
        val = stats.get("LST_Day_1km").getInfo()
        if val is not None and val > 0:
            lst_c = val - 273.15
            logger.debug(
                "LST from MODIS: %.1f °C at (%.4f, %.4f)", lst_c, lat, lon
            )
            return float(lst_c)

    except Exception as exc:
        logger.warning("GEE LST retrieval error: %s", exc)

    return None


# ---------------------------------------------------------------------------
# ESA CCI land-cover → BurnTrack fuel model mapping
# ---------------------------------------------------------------------------

# ESA CCI land cover class codes → BurnTrack fuel model codes.
# Reference: ESA CCI Land Cover v2.1.1 (2020) user guide.
_ESA_LC_TO_FUEL: Dict[int, str] = {
    # Croplands
    10: "GR1",    # Cropland, rainfed
    11: "GR1",    # Herbaceous cover
    12: "GR4",    # Tree/shrub cover
    20: "GR1",    # Cropland, irrigated
    30: "GR4",    # Mosaic cropland / natural vegetation
    40: "GR4",    # Mosaic natural veg / cropland
    # Forest
    50: "TL3",    # Tree cover, broadleaved, evergreen, closed
    60: "TL3",    # Tree cover, broadleaved, deciduous, closed (>40%)
    61: "TL6",    # Tree cover, broadleaved, deciduous, open (15-40%)
    62: "TL6",    # Tree cover, broadleaved, deciduous, open (15-40%)
    70: "TL1",    # Tree cover, needleleaved, evergreen, closed
    80: "TL1",    # Tree cover, needleleaved, deciduous, closed
    90: "TL3",    # Tree cover, mixed leaf type
    100: "SH2",   # Mosaic tree and shrub / herbaceous
    110: "GR2",   # Mosaic herbaceous / tree and shrub
    # Shrubland
    120: "SH2",   # Shrubland
    121: "SH1",   # Shrubland, evergreen
    122: "SH2",   # Shrubland, deciduous
    # Grassland
    130: "GR2",   # Grassland
    140: "GR1",   # Lichens and mosses
    # Sparse vegetation
    150: "GR1",   # Sparse vegetation (tree/shrub/herb)
    152: "GR1",   # Sparse shrub
    153: "GR1",   # Sparse herbaceous
    # Wetland
    160: "GR4",   # Tree cover, flooded, fresh/brackish
    170: "GR4",   # Tree cover, flooded, saline
    180: "GR2",   # Shrub / herb cover, flooded
    # Urban / bare / water — no fuel model
    190: None,    # Urban areas
    200: None,    # Bare areas
    201: None,    # Consolidated bare
    202: None,    # Unconsolidated bare
    210: None,    # Water
    220: None,    # Permanent snow/ice
    # African-specific fuel models (mapped from Copernicus)
}

# Copernicus Global Land Cover (100 m) discrete classification
# to BurnTrack fuel model
_COPERNICUS_LC_TO_FUEL: Dict[int, str] = {
    0:   None,             # Unknown
    20:  "SH2",            # Shrubs
    30:  "GR2",            # Herbaceous vegetation
    40:  "GR1",            # Cultivated / managed
    50:  "GR4",            # Urban / built-up
    60:  None,             # Bare / sparse vegetation
    70:  None,             # Snow and ice
    80:  None,             # Permanent water
    90:  "GR4",            # Herbaceous wetland
    100: "GR1",            # Moss and lichen
    111: "TL3",            # Closed forest, evergreen needle leaf
    112: "TL3",            # Closed forest, evergreen broad leaf
    113: "TL3",            # Closed forest, deciduous needle leaf
    114: "TL6",            # Closed forest, deciduous broad leaf
    115: "TL3",            # Closed forest, mixed
    116: "TL3",            # Closed forest, not matching
    121: "TL6",            # Open forest, evergreen needle leaf
    122: "TL6",            # Open forest, evergreen broad leaf
    123: "TL6",            # Open forest, deciduous needle leaf
    124: "TL6",            # Open forest, deciduous broad leaf
    125: "TL6",            # Open forest, mixed
    126: "TL6",            # Open forest, not matching
    200: None,             # Oceans, seas
}


def fetch_land_cover_gee(
    lat: float,
    lon: float,
    year: int,
) -> Optional[str]:
    """Fetch land cover from Copernicus 100 m via GEE and map to fuel model.

    Tries Copernicus Global Land Cover first, then falls back to ESA CCI.

    Args:
        lat: Latitude.
        lon: Longitude.
        year: Target year (closest available year is used).

    Returns:
        BurnTrack fuel model code (e.g. ``"GR2"``) or ``None``.
    """
    if not _init_gee():
        return None

    try:
        import ee  # type: ignore[import-untyped]

        point = ee.Geometry.Point([lon, lat])

        # ---- Copernicus 100 m Proba-V ----
        try:
            cop = (
                ee.ImageCollection(
                    "COPERNICUS/Landcover/100m/Proba-V-C3/Global"
                )
                .filterBounds(point)
                .filterDate(f"{year}-01-01", f"{year}-12-31")
                .select(["discrete_classification"])
                .sort("system:time_start", False)  # newest first
                .first()
            )
            val = cop.reduceRegion(
                reducer=ee.Reducer.mode(),
                geometry=point.buffer(100),
                scale=100,
                maxPixels=100,
            ).get("discrete_classification").getInfo()

            if val is not None:
                fuel = _COPERNICUS_LC_TO_FUEL.get(int(val))
                if fuel is not None:
                    logger.debug(
                        "Land cover (Copernicus): class=%d → fuel=%s at "
                        "(%.4f, %.4f)", val, fuel, lat, lon,
                    )
                    return fuel
        except Exception as cop_exc:
            logger.debug("Copernicus land cover failed: %s", cop_exc)

        # ---- ESA CCI fallback ----
        try:
            cci = (
                ee.ImageCollection("ESA/CCI/FireCCI/5_1")
                .filterBounds(point)
                .filterDate(f"{year}-01-01", f"{year}-12-31")
                .select(["LandCover"])
                .first()
            )
            val = cci.reduceRegion(
                reducer=ee.Reducer.mode(),
                geometry=point.buffer(250),
                scale=250,
                maxPixels=100,
            ).get("LandCover").getInfo()

            if val is not None:
                fuel = _ESA_LC_TO_FUEL.get(int(val))
                if fuel is not None:
                    logger.debug(
                        "Land cover (ESA CCI): class=%d → fuel=%s at "
                        "(%.4f, %.4f)", val, fuel, lat, lon,
                    )
                    return fuel
        except Exception as cci_exc:
            logger.debug("ESA CCI land cover failed: %s", cci_exc)

    except Exception as exc:
        logger.warning("GEE land cover retrieval error: %s", exc)

    return None


# ============================================================================
# Offline fallback — climate-based estimation
# ============================================================================

# ---- African biome NDVI climatology ----
#
# Values are calibrated against:
#   - Huete et al. 2002 (MODIS global NDVI seasonal profiles)
#   - Archibald et al. 2010 (Southern African grass curing & fire regimes)
#   - Eva & Lambin 2000 (West African savanna fires & vegetation dynamics)
#   - Giglio et al. 2013 (MODIS fire activity and NDVI relationships)
#
# Each zone defines wet-season months, peak NDVI during wet season,
# and trough NDVI during dry season.

AFRICA_NDVI_CLIMATOLOGY: Dict[str, Dict] = {
    "sahel": {  # 10–15 °N — short, intense wet season
        "lat_range": (10.0, 15.0),
        "hemisphere": "N",
        "wet": (6, 7, 8, 9, 10),
        "ndvi_wet": 0.35,
        "ndvi_dry": 0.12,
    },
    "sudan": {  # 5–10 °N — Sudanian savanna
        "lat_range": (5.0, 10.0),
        "hemisphere": "N",
        "wet": (5, 6, 7, 8, 9, 10),
        "ndvi_wet": 0.45,
        "ndvi_dry": 0.18,
    },
    "guinea": {  # 5–8 °N — Guinea savanna / forest mosaic
        "lat_range": (4.0, 8.0),
        "hemisphere": "N",
        "wet": (4, 5, 6, 7, 8, 9, 10, 11),
        "ndvi_wet": 0.55,
        "ndvi_dry": 0.30,
    },
    "equatorial": {  # ±5 °N/S — Congo basin, high year-round greenness
        "lat_range": (-5.0, 5.0),
        "hemisphere": "E",
        "wet": tuple(range(1, 13)),  # year-round
        "ndvi_wet": 0.60,
        "ndvi_dry": 0.50,
    },
    "miombo": {  # 5–15 °S — Miombo woodland belt
        "lat_range": (-15.0, -5.0),
        "hemisphere": "S",
        "wet": (11, 12, 1, 2, 3, 4),
        "ndvi_wet": 0.50,
        "ndvi_dry": 0.20,
    },
    "highveld": {  # 25–30 °S — South African highveld / grassland
        "lat_range": (-30.0, -25.0),
        "hemisphere": "S",
        "wet": (10, 11, 12, 1, 2, 3),
        "ndvi_wet": 0.45,
        "ndvi_dry": 0.15,
    },
    "fynbos": {  # 33–35 °S — Cape fynbos (winter rainfall)
        "lat_range": (-35.0, -33.0),
        "hemisphere": "S",
        "wet": (5, 6, 7, 8),
        "ndvi_wet": 0.40,
        "ndvi_dry": 0.25,
    },
    "steppe": {  # 30–36 °N — North Africa steppe / Mediterranean
        "lat_range": (30.0, 36.0),
        "hemisphere": "N",
        "wet": (11, 12, 1, 2, 3, 4),
        "ndvi_wet": 0.30,
        "ndvi_dry": 0.10,
    },
    "desert_margin": {  # 15–18 °N — Sahel–Sahara transition
        "lat_range": (15.0, 18.0),
        "hemisphere": "N",
        "wet": (7, 8, 9),
        "ndvi_wet": 0.20,
        "ndvi_dry": 0.05,
    },
    "kalahari": {  # 20–25 °S — Kalahari semi-arid savanna
        "lat_range": (-25.0, -20.0),
        "hemisphere": "S",
        "wet": (11, 12, 1, 2, 3),
        "ndvi_wet": 0.40,
        "ndvi_dry": 0.15,
    },
    "east_africa": {  # 0–10 °S — East African savanna / woodland
        "lat_range": (-10.0, 0.0),
        "hemisphere": "S",
        "wet": (3, 4, 5, 10, 11, 12),  # bimodal
        "ndvi_wet": 0.50,
        "ndvi_dry": 0.25,
    },
}

# Fuel-model-based NDVI adjustments
# Some fuel models imply specific vegetation states.
_FUEL_NDVI_ADJUSTMENT: Dict[str, float] = {
    # Grass models — typically lower canopy, quick cure
    "GR1": -0.05,
    "GR2": 0.00,
    "GR4": 0.05,
    "AF_SAHEL_GRASS": -0.08,
    "AF_GRASSLAND_FERTILE": 0.03,
    # Shrub models — moderate greenness
    "SH1": 0.05,
    "SH2": 0.03,
    "AF_FYNBOS": 0.06,
    "AF_STEPPE": -0.04,
    # Woodland / timber
    "TL1": 0.10,
    "TL3": 0.12,
    "TL6": 0.08,
    "AF_MIOMBO": 0.10,
}


def _classify_climate_zone(lat: float) -> str:
    """Map a latitude to the most appropriate African climate zone."""
    # Sort zones by how well the latitude fits their range
    best_zone = "sudan"  # default fallback
    best_dist = float("inf")

    for zone, params in AFRICA_NDVI_CLIMATOLOGY.items():
        lo, hi = params["lat_range"]
        if lo <= lat <= hi:
            return zone
        # Distance to nearest range edge
        dist = min(abs(lat - lo), abs(lat - hi))
        if dist < best_dist:
            best_dist = dist
            best_zone = zone

    return best_zone


def _interpolate_seasonal_ndvi(
    month: int,
    wet_months: Tuple[int, ...],
    ndvi_wet: float,
    ndvi_dry: float,
) -> float:
    """Interpolate NDVI between wet and dry seasons using a smooth curve.

    Uses a cosine-based interpolation so transitions are gradual, matching
    real phenological dynamics (green-up / senescence are not step functions).
    """
    if len(wet_months) == 12:
        # Year-round wet (e.g. equatorial forest)
        return (ndvi_wet + ndvi_dry) / 2.0

    # Compute a "greenness phase" based on distance to wet season centre.
    wet_centre = np.mean(
        [(m if m >= 6 else m + 12) for m in wet_months]
    )
    if wet_centre > 12:
        wet_centre -= 12

    # Circular distance (months wrap around)
    dist = min(
        abs(month - wet_centre),
        12 - abs(month - wet_centre),
    )
    max_dist = 6.0  # maximum half-cycle

    # Cosine interpolation: 0 at peak wet → 1 at peak dry
    phase = 0.5 * (1 - math.cos(math.pi * dist / max_dist))
    return ndvi_wet * (1 - phase) + ndvi_dry * phase


def estimate_ndvi_offline(
    lat: float,
    lon: float,
    date: str,
    fuel_model_code: str = "",
) -> float:
    """Estimate NDVI from latitude, date, and optional fuel model.

    Uses the ``AFRICA_NDVI_CLIMATOLOGY`` look-up table with smooth
    seasonal interpolation and optional fuel-model-based adjustment.

    Args:
        lat: Latitude (decimal degrees, negative for southern hemisphere).
        lon: Longitude (unused for now; reserved for future refinement).
        date: ISO date string ``YYYY-MM-DD``.
        fuel_model_code: Optional BurnTrack fuel model code for fine-tuning.

    Returns:
        Estimated NDVI in [0, 1] (clipped).
    """
    try:
        month = int(date.split("-")[1])
    except (IndexError, ValueError):
        logger.warning("Could not parse month from date '%s'; using 1.", date)
        month = 1

    zone_name = _classify_climate_zone(lat)
    zone = AFRICA_NDVI_CLIMATOLOGY[zone_name]

    ndvi = _interpolate_seasonal_ndvi(
        month, zone["wet"], zone["ndvi_wet"], zone["ndvi_dry"]
    )

    # Apply fuel-model adjustment
    if fuel_model_code and fuel_model_code in _FUEL_NDVI_ADJUSTMENT:
        ndvi += _FUEL_NDVI_ADJUSTMENT[fuel_model_code]

    # Add small deterministic jitter from lat/lon to avoid perfectly
    # uniform values across the landscape
    jitter = 0.02 * math.sin(lat * 7.3 + lon * 3.7)
    ndvi += jitter

    ndvi = float(np.clip(ndvi, 0.0, 1.0))
    logger.debug(
        "Offline NDVI estimate: %.3f (zone=%s, month=%d, fuel=%s)",
        ndvi, zone_name, month, fuel_model_code,
    )
    return ndvi


def estimate_ndwi_offline(
    lat: float,
    lon: float,
    date: str,
    fuel_model_code: str = "",
) -> float:
    """Estimate NDWI from NDVI via empirical correlation.

    In African grasslands and savannas, the Gao-style NDWI (vegetation
    moisture) is moderately correlated with NDVI.  Literature values
    (Ceccato et al. 2001; Maki et al. 2004) suggest:

        NDWI ≈ 0.50 × NDVI − 0.05  (grasslands)
        NDWI ≈ 0.60 × NDVI − 0.03  (woodlands)

    Args:
        lat: Latitude.
        lon: Longitude.
        date: ISO date string.
        fuel_model_code: Optional fuel model code.

    Returns:
        Estimated NDWI in [-0.3, 0.6] (clipped).
    """
    ndvi = estimate_ndvi_offline(lat, lon, date, fuel_model_code)

    # Use different coefficients for woody vs. herbaceous
    woody_models = {
        "TL1", "TL3", "TL6", "SH1", "SH2", "AF_MIOMBO", "AF_FYNBOS",
    }
    if fuel_model_code in woody_models:
        ndwi = 0.60 * ndvi - 0.03
    else:
        ndwi = 0.50 * ndvi - 0.05

    ndwi = float(np.clip(ndwi, -0.3, 0.6))
    logger.debug(
        "Offline NDWI estimate: %.3f (ndvi=%.3f, fuel=%s)",
        ndwi, ndvi, fuel_model_code,
    )
    return ndwi


def estimate_lst_offline(
    lat: float,
    temp_c: float,
    month: int,
) -> float:
    """Estimate Land Surface Temperature from air temperature.

    LST is typically higher than air temperature by a ΔT that depends
    on vegetation cover density, solar elevation, and soil moisture.

    Empirical ΔT model (after Mildrexler et al. 2011; Wan 2008):

        ΔT_base ≈ 10 °C  (typical for sparse / moderate vegetation)
        ΔT increases by ~5 °C during dry season (bare soil effect)
        ΔT decreases by ~3 °C in forested / green areas
        ΔT increases by ~2 °C at low latitudes (higher solar angle)

    Args:
        lat: Latitude.
        temp_c: Air temperature in °C.
        month: Month (1–12).

    Returns:
        Estimated LST in °C.
    """
    delta_t_base = 10.0

    # Solar angle effect — lower latitudes receive more direct radiation
    abs_lat = abs(lat)
    if abs_lat < 10:
        solar_boost = 3.0
    elif abs_lat < 20:
        solar_boost = 2.0
    elif abs_lat < 30:
        solar_boost = 1.0
    else:
        solar_boost = 0.0

    # Dry-season effect — heuristic based on hemisphere and month.
    # In the Southern Hemisphere, dry season peaks Jun–Sep;
    # in the Northern Hemisphere, dry season peaks Dec–Mar.
    if lat < 0:
        # Southern hemisphere
        dry_peak_months = (6, 7, 8, 9)
    else:
        # Northern hemisphere
        dry_peak_months = (12, 1, 2, 3)

    if month in dry_peak_months:
        dry_boost = 5.0
    elif month in _adjacent_months(dry_peak_months):
        dry_boost = 2.5
    else:
        dry_boost = 0.0

    delta_t = delta_t_base + solar_boost + dry_boost
    lst = temp_c + delta_t

    logger.debug(
        "Offline LST estimate: %.1f °C (air=%.1f, ΔT=%.1f, lat=%.1f, month=%d)",
        lst, temp_c, delta_t, lat, month,
    )
    return float(lst)


def _adjacent_months(months: Tuple[int, ...]) -> Tuple[int, ...]:
    """Return the months immediately adjacent to the given month set."""
    adj = set()
    for m in months:
        before = m - 1 if m > 1 else 12
        after = m + 1 if m < 12 else 1
        if before not in months:
            adj.add(before)
        if after not in months:
            adj.add(after)
    return tuple(sorted(adj))


# ============================================================================
# Unified interface
# ============================================================================

def fetch_ndvi(
    lat: float,
    lon: float,
    date: str,
    fuel_model_code: str = "",
    use_gee: bool = True,
) -> float:
    """Fetch NDVI with GEE primary, offline fallback.

    Args:
        lat: Latitude.
        lon: Longitude.
        date: ISO date string (YYYY-MM-DD).
        fuel_model_code: Optional fuel model code for offline refinement.
        use_gee: Whether to attempt GEE retrieval first.

    Returns:
        NDVI value (always returns a valid estimate).
    """
    if use_gee:
        val = fetch_ndvi_gee(lat, lon, date)
        if val is not None:
            return val
        logger.debug("GEE NDVI unavailable; using offline estimate.")
    return estimate_ndvi_offline(lat, lon, date, fuel_model_code)


def fetch_ndwi(
    lat: float,
    lon: float,
    date: str,
    fuel_model_code: str = "",
    use_gee: bool = True,
) -> float:
    """Fetch NDWI with GEE primary, offline fallback.

    Args:
        lat: Latitude.
        lon: Longitude.
        date: ISO date string (YYYY-MM-DD).
        fuel_model_code: Optional fuel model code for offline refinement.
        use_gee: Whether to attempt GEE retrieval first.

    Returns:
        NDWI value (always returns a valid estimate).
    """
    if use_gee:
        val = fetch_ndwi_gee(lat, lon, date)
        if val is not None:
            return val
        logger.debug("GEE NDWI unavailable; using offline estimate.")
    return estimate_ndwi_offline(lat, lon, date, fuel_model_code)


def fetch_lst(
    lat: float,
    lon: float,
    date: str,
    temp_c: float = 25.0,
    use_gee: bool = True,
) -> float:
    """Fetch LST with GEE primary, offline fallback.

    Args:
        lat: Latitude.
        lon: Longitude.
        date: ISO date string (YYYY-MM-DD).
        temp_c: Air temperature in °C (used by offline fallback).
        use_gee: Whether to attempt GEE retrieval first.

    Returns:
        LST in °C (always returns a valid estimate).
    """
    if use_gee:
        val = fetch_lst_gee(lat, lon, date)
        if val is not None:
            return val
        logger.debug("GEE LST unavailable; using offline estimate.")

    try:
        month = int(date.split("-")[1]) if isinstance(date, str) else date.month
    except (IndexError, ValueError, AttributeError):
        month = 1
    return estimate_lst_offline(lat, temp_c, month)


def fetch_land_cover(
    lat: float,
    lon: float,
    year: int,
    use_gee: bool = True,
) -> Optional[str]:
    """Fetch land cover class and map to fuel model code.

    Args:
        lat: Latitude.
        lon: Longitude.
        year: Target year.
        use_gee: Whether to attempt GEE retrieval.

    Returns:
        BurnTrack fuel model code (e.g. ``"GR2"``) or ``None``.
    """
    if use_gee:
        val = fetch_land_cover_gee(lat, lon, year)
        if val is not None:
            return val
    # No offline fallback for land cover — requires external data
    return None


# ============================================================================
# DataFrame enrichment
# ============================================================================

def enrich_dataframe_with_satellite(
    df: "pd.DataFrame",
    use_gee: bool = True,
    progress: bool = True,
) -> "pd.DataFrame":
    """Enrich a DataFrame with NDVI, NDWI, and LST columns.

    Expects the input DataFrame to contain at least:
        - ``lat`` or ``latitude``: decimal-degree latitude
        - ``lon`` or ``longitude``: decimal-degree longitude
        - ``date`` or ``datetime``: ISO date string or datetime column
        - ``temp_c`` (optional): air temperature for LST fallback

    Optionally:
        - ``fuel_model_code``: used for offline NDVI/NDWI refinement

    New columns added:
        - ``ndvi``, ``ndvi_origin`` ("gee" or "offline")
        - ``ndwi``, ``ndwi_origin``
        - ``lst_c``, ``lst_c_origin``

    Args:
        df: Input DataFrame (modified in-place *and* returned).
        use_gee: Whether to attempt Google Earth Engine retrieval.
        progress: Whether to log progress every 50 rows.

    Returns:
        The enriched DataFrame with satellite-derived columns.
    """
    if not _PANDAS_AVAILABLE:
        raise ImportError(
            "pandas is required for enrich_dataframe_with_satellite"
        )

    df = df.copy()
    n = len(df)

    # Resolve column name variants
    lat_col = "lat" if "lat" in df.columns else "latitude"
    lon_col = "lon" if "lon" in df.columns else "longitude"
    date_col = "date" if "date" in df.columns else "datetime"
    temp_col = "temp_c" if "temp_c" in df.columns else None
    fuel_col = (
        "fuel_model_code" if "fuel_model_code" in df.columns else None
    )

    if lat_col not in df.columns:
        raise ValueError(
            f"DataFrame must have a 'lat' or 'latitude' column; "
            f"found: {list(df.columns)}"
        )
    if lon_col not in df.columns:
        raise ValueError(
            f"DataFrame must have a 'lon' or 'longitude' column; "
            f"found: {list(df.columns)}"
        )
    if date_col not in df.columns:
        raise ValueError(
            f"DataFrame must have a 'date' or 'datetime' column; "
            f"found: {list(df.columns)}"
        )

    # Pre-allocate arrays
    ndvi_vals = np.full(n, np.nan)
    ndwi_vals = np.full(n, np.nan)
    lst_vals = np.full(n, np.nan)
    ndvi_origins = [""] * n
    ndwi_origins = [""] * n
    lst_origins = [""] * n

    # Attempt GEE init once
    gee_ok = _init_gee() if use_gee else False

    for i, (idx, row) in enumerate(df.iterrows()):
        lat = float(row[lat_col])
        lon = float(row[lon_col])
        date_val = str(row[date_col])[:10]  # YYYY-MM-DD
        temp_c = float(row[temp_col]) if temp_col and not pd.isna(row.get(temp_col)) else 25.0
        fuel = str(row[fuel_col]) if fuel_col and not pd.isna(row.get(fuel_col)) else ""

        # ---- NDVI ----
        ndvi = None
        origin = "offline"
        if gee_ok:
            ndvi = fetch_ndvi_gee(lat, lon, date_val)
            if ndvi is not None:
                origin = "gee"
        if ndvi is None:
            ndvi = estimate_ndvi_offline(lat, lon, date_val, fuel)
        ndvi_vals[i] = ndvi
        ndvi_origins[i] = origin

        # ---- NDWI ----
        ndwi = None
        origin = "offline"
        if gee_ok:
            ndwi = fetch_ndwi_gee(lat, lon, date_val)
            if ndwi is not None:
                origin = "gee"
        if ndwi is None:
            ndwi = estimate_ndwi_offline(lat, lon, date_val, fuel)
        ndwi_vals[i] = ndwi
        ndwi_origins[i] = origin

        # ---- LST ----
        lst = None
        origin = "offline"
        if gee_ok:
            lst = fetch_lst_gee(lat, lon, date_val)
            if lst is not None:
                origin = "gee"
        if lst is None:
            try:
                month = int(date_val.split("-")[1])
            except (IndexError, ValueError):
                month = 1
            lst = estimate_lst_offline(lat, temp_c, month)
        lst_vals[i] = lst
        lst_origins[i] = origin

        if progress and (i + 1) % 50 == 0:
            logger.info(
                "Satellite enrichment progress: %d / %d rows (%.0f%%)",
                i + 1, n, 100 * (i + 1) / n,
            )

    df["ndvi"] = ndvi_vals
    df["ndvi_origin"] = ndvi_origins
    df["ndwi"] = ndwi_vals
    df["ndwi_origin"] = ndwi_origins
    df["lst_c"] = lst_vals
    df["lst_c_origin"] = lst_origins

    logger.info(
        "Satellite enrichment complete: %d rows. "
        "NDVI origins: %s. NDWI origins: %s. LST origins: %s.",
        n,
        dict(pd.Series(ndvi_origins).value_counts()),
        dict(pd.Series(ndwi_origins).value_counts()),
        dict(pd.Series(lst_origins).value_counts()),
    )
    return df
