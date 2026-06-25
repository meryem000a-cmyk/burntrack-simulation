"""
burntrack.data — Data acquisition pipeline.

Provides:
    - FIRMS active fire download (NASA VIIRS)
    - Weather download (Open-Meteo + ERA5 CDS)
    - Slope/aspect computation from elevation grids
    - Real dataset pipeline for African fire propagation

NOTE: The synthetic dataset generator was removed on 2026-06-25 (NO-synthetic-data
constraint, see ReadIfAgent.md). All training now uses real observations only.

Convenience aliases:
    download_weather  -> download_openmeteo
    download_era5     -> download_era5_africa
    download_firms    -> download_all_africa_fires
"""

from .weather import (
    download_openmeteo,
    download_era5_africa,
    ERA5Downloader,
    OpenMeteoDownloader,
    fetch_weather_for_points,
    compute_slope_aspect,
    request_with_retry,
    rate_limited_get,
)

from .firms import (
    download_firms_region,
    download_all_africa_fires,
    reconstruct_propagation,
    AFRICA_REGIONS,
    MAX_DISTANCE_M,
    MAX_TIME_DIFF_MIN,
)

from .real_dataset import (
    build_real_dataset,
    compute_rothermel_baseline,
    compute_vpd,
    compute_dfmc,
    compute_wind_mid_flame,
    convert_fuel_to_engine,
)

# Convenience aliases
download_weather = download_openmeteo
download_era5 = download_era5_africa
download_firms = download_all_africa_fires

__all__ = [
    # Weather
    "download_openmeteo",
    "download_era5_africa",
    "download_weather",
    "download_era5",
    "ERA5Downloader",
    "OpenMeteoDownloader",
    "fetch_weather_for_points",
    "compute_slope_aspect",
    "request_with_retry",
    "rate_limited_get",
    # FIRMS
    "download_firms_region",
    "download_all_africa_fires",
    "download_firms",
    "reconstruct_propagation",
    "AFRICA_REGIONS",
    "MAX_DISTANCE_M",
    "MAX_TIME_DIFF_MIN",
    # Real dataset
    "build_real_dataset",
    "compute_rothermel_baseline",
    "compute_vpd",
    "compute_dfmc",
    "compute_wind_mid_flame",
    "convert_fuel_to_engine",
]
