"""
weather.py — Météorological data acquisition for BurnTrack.
===========================================================

Merged from:
    - rothermel/Download_data.py (Open-Meteo API)
    - data_pipeline/download_era5_africa.py (ERA5 CDS)
    - data_pipeline/build_real_dataset.py (elevation grid, rate limiting)

Provides:
    - Open-Meteo historical weather download (free, no API key)
    - ERA5-Land download via CDS API (requires cdsapi + ~/.cdsapirc)
    - Batch weather fetching with thread-safe rate limiting
    - Slope/aspect computation from 3x3 elevation grid

All original logic preserved — only restructured into a single module.
"""

import os
import time
import random
import hashlib
import threading
import warnings
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import requests

warnings.filterwarnings("ignore")

# =============================================================================
# CONFIGURATION
# =============================================================================

DATA_DIR = Path("./data")
CACHE_DIR = DATA_DIR / "cache"
for d in [CACHE_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# Thread-safe rate limiter for HTTP requests
_rate_limit_lock = threading.Lock()
_last_request_time = 0.0
REQUEST_INTERVAL = 0.1  # 100 ms → max 10 req/s


# =============================================================================
# HTTP UTILITIES (extracted from build_real_dataset.py)
# =============================================================================

def request_with_retry(
    url: str, params: dict = None, timeout: int = 15, max_retries: int = 5
) -> requests.Response:
    """Perform HTTP GET with automatic retry and exponential backoff."""
    backoff = 1.0
    for attempt in range(max_retries):
        try:
            response = requests.get(url, params=params, timeout=timeout)
            if response.status_code == 200:
                return response
            elif response.status_code == 429:
                retry_after = int(response.headers.get("Retry-After", 1))
                sleep_time = max(retry_after, backoff + random.uniform(0, 1))
                print(
                    f"  [HTTP 429] Rate limit reached. Retrying in {sleep_time:.2f}s..."
                )
                time.sleep(sleep_time)
            elif response.status_code in [500, 502, 503, 504]:
                sleep_time = backoff + random.uniform(0, 1)
                print(
                    f"  [HTTP {response.status_code}] Server error. Retrying in {sleep_time:.2f}s..."
                )
                time.sleep(sleep_time)
            else:
                response.raise_for_status()
        except requests.exceptions.RequestException as e:
            if attempt == max_retries - 1:
                raise
            sleep_time = backoff + random.uniform(0, 1)
            print(f"  Connection lost / timeout: {e}. Retrying in {sleep_time:.2f}s...")
            time.sleep(sleep_time)
        backoff *= 2.0
    raise RuntimeError(f"Failed after {max_retries} attempts on {url}")


def rate_limited_get(url: str, params: dict = None) -> requests.Response:
    """Thread-safe rate-limited HTTP GET."""
    global _last_request_time
    with _rate_limit_lock:
        now = time.time()
        elapsed = now - _last_request_time
        if elapsed < REQUEST_INTERVAL:
            time.sleep(REQUEST_INTERVAL - elapsed)
        _last_request_time = time.time()
    return request_with_retry(url, params)


# =============================================================================
# OPEN-METEO DOWNLOADER (from Download_data.py)
# =============================================================================

class OpenMeteoDownloader:
    """Weather downloader via Open-Meteo API (free, no API key).

    Provides historical ERA5 data and elevation lookups.
    """

    BASE_URL = "https://api.open-meteo.com/v1"
    HISTORICAL_URL = "https://archive-api.open-meteo.com/v1"

    def __init__(self):
        self.session = requests.Session()

    def get_weather(self, lat: float, lon: float, date: datetime) -> Dict:
        """Fetch hourly weather for a single point.

        Args:
            lat: Latitude in decimal degrees.
            lon: Longitude in decimal degrees.
            date: Target datetime.

        Returns:
            Dict with temperature_2m, relative_humidity_2m, wind_speed_10m,
            wind_direction_10m, precipitation, surface_pressure, dewpoint_2m.
        """
        start = (date - timedelta(days=1)).strftime("%Y-%m-%d")
        end = (date + timedelta(days=1)).strftime("%Y-%m-%d")

        url = f"{self.HISTORICAL_URL}/era5"
        params = {
            "latitude": lat,
            "longitude": lon,
            "start_date": start,
            "end_date": end,
            "hourly": [
                "temperature_2m",
                "relative_humidity_2m",
                "wind_speed_10m",
                "wind_direction_10m",
                "precipitation",
                "surface_pressure",
                "dewpoint_2m",
            ],
            "timezone": "auto",
        }

        try:
            response = self.session.get(url, params=params, timeout=30)
            response.raise_for_status()
            data = response.json()

            hourly = data.get("hourly", {})
            times = hourly.get("time", [])
            target_hour = date.strftime("%Y-%m-%dT%H:00")

            idx = times.index(target_hour) if target_hour in times else len(times) // 2

            return {
                "temperature_2m": hourly.get("temperature_2m", [None])[idx],
                "relative_humidity_2m": hourly.get("relative_humidity_2m", [None])[
                    idx
                ],
                "wind_speed_10m": hourly.get("wind_speed_10m", [None])[idx],
                "wind_direction_10m": hourly.get("wind_direction_10m", [None])[idx],
                "precipitation": hourly.get("precipitation", [None])[idx],
                "surface_pressure": hourly.get("surface_pressure", [None])[idx],
                "dewpoint_2m": hourly.get("dewpoint_2m", [None])[idx],
                "source": "open-meteo",
            }
        except Exception as e:
            print(f"  Open-Meteo error: {e}")
            return self._weather_fallback(lat, lon, date)

    def get_elevation(self, lat: float, lon: float) -> Dict:
        """Fetch single-point elevation.

        Returns:
            Dict with 'elevation' (m) and 'source'.
        """
        url = f"{self.BASE_URL}/elevation"
        try:
            response = self.session.get(
                url, params={"latitude": lat, "longitude": lon}, timeout=10
            )
            response.raise_for_status()
            return {
                "elevation": response.json().get("elevation", [0])[0],
                "source": "open-meteo",
            }
        except Exception:
            return {"elevation": 500, "source": "fallback"}

    def get_elevation_grid(self, lat: float, lon: float) -> List[float]:
        """Fetch 5-point elevation grid (center, N, S, E, W).

        Uses ~30 m offset in lat/lon degrees.

        Returns:
            List of 5 elevations [c, n, s, e, w] in meters.
        """
        d_lat = 0.00027
        d_lon = 0.00027 / max(np.cos(np.radians(lat)), 0.1)

        lats_str = f"{lat},{lat + d_lat},{lat - d_lat},{lat},{lat}"
        lons_str = f"{lon},{lon},{lon},{lon + d_lon},{lon - d_lon}"

        elevation_url = "https://api.open-meteo.com/v1/elevation"
        try:
            e_res = rate_limited_get(
                elevation_url, params={"latitude": lats_str, "longitude": lons_str}
            )
            elevations = e_res.json().get("elevation", [250.0] * 5)
            if len(elevations) < 5:
                elevations = elevations + [elevations[0]] * (5 - len(elevations))
            return list(elevations)
        except Exception:
            return [250.0] * 5

    def _weather_fallback(self, lat: float, lon: float, date: datetime) -> Dict:
        """Season-based fallback values when API is unreachable."""
        month = date.month
        is_dry = month in [6, 7, 8, 9]

        if lat > 20:
            return {
                "temperature_2m": 35.0 if is_dry else 18.0,
                "relative_humidity_2m": 25.0 if is_dry else 60.0,
                "wind_speed_10m": 4.0,
                "wind_direction_10m": 270.0,
                "precipitation": 0.0,
                "surface_pressure": 1015.0,
                "dewpoint_2m": 15.0,
                "source": "fallback-maroc",
            }
        else:
            return {
                "temperature_2m": 32.0,
                "relative_humidity_2m": 40.0,
                "wind_speed_10m": 3.0,
                "wind_direction_10m": 0.0,
                "precipitation": 0.0,
                "surface_pressure": 1010.0,
                "dewpoint_2m": 18.0,
                "source": "fallback-sahel",
            }


def download_openmeteo(lat: float, lon: float, date: datetime, params: Dict = None) -> Dict:
    """Convenience function: download weather from Open-Meteo.

    Args:
        lat: Latitude.
        lon: Longitude.
        date: Target datetime.
        params: Optional override dict (ignored; kept for API compatibility).

    Returns:
        Weather dict.
    """
    downloader = OpenMeteoDownloader()
    return downloader.get_weather(lat, lon, date)


# =============================================================================
# ERA5 DOWNLOADER (from download_era5_africa.py)
# =============================================================================

# Zone definitions for ERA5-Land Africa downloads
_AFRICA_FIRE_ZONES = {
    "sahel": {
        "bbox": [20.0, -15.0, 10.0, 15.0],
        "description": "Sahel — Herbes sèches, feux de brousse",
    },
    "south_africa_kruger": {
        "bbox": [-22.0, 30.0, -26.0, 33.0],
        "description": "Afrique du Sud (Kruger) — Savane, Miombo",
    },
    "south_africa_fynbos": {
        "bbox": [-33.0, 18.0, -35.0, 21.0],
        "description": "Afrique du Sud (Fynbos) — Fynbos méditerranéen",
    },
    "madagascar": {
        "bbox": [-11.0, 43.0, -26.0, 51.0],
        "description": "Madagascar — Forêt sèche, savane",
    },
    "east_africa": {
        "bbox": [5.0, 33.0, -5.0, 42.0],
        "description": "Afrique de l'Est — Miombo, savane",
    },
}

_ERA5_VARIABLES = {
    "2m_temperature": "2m_temperature",
    "2m_dewpoint_temperature": "2m_dewpoint_temperature",
    "10m_u_component_of_wind": "10m_u_component_of_wind",
    "10m_v_component_of_wind": "10m_v_component_of_wind",
    "total_precipitation": "total_precipitation",
    "surface_solar_radiation_downwards": "surface_solar_radiation_downwards",
}


class ERA5Downloader:
    """ERA5-Land NetCDF downloader via Copernicus CDS API.

    Requires:
        pip install cdsapi
        # Create account at https://cds.climate.copernicus.eu/
        # Configure ~/.cdsapirc with UID and API key

    Zones covered:
        - Sahel (Burkina Faso, Mali, Niger)
        - South Africa (Kruger, Fynbos)
        - Madagascar
        - East Africa (Kenya, Tanzania)
    """

    def __init__(self, output_dir: str = "era5_africa_data"):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)

        try:
            import cdsapi

            self.client = cdsapi.Client()
            print("  CDS API connection established")
        except ImportError:
            print("  cdsapi not installed. ERA5 downloads unavailable.")
            print("  Install with: pip install cdsapi")
            self.client = None
        except Exception as e:
            print(f"  CDS API connection error: {e}")
            print("  Verify your ~/.cdsapirc file")
            self.client = None

    def download_zone(
        self,
        zone_name: str,
        year: int,
        months: List[str] = None,
        variables: List[str] = None,
    ) -> str:
        """Download ERA5-Land data for a zone and year.

        Args:
            zone_name: Key from AFRICA_FIRE_ZONES.
            year: Year (e.g., 2020).
            months: List of months ['01', '02', ...] or None for all.
            variables: List of variable names or None for all.

        Returns:
            Path to the downloaded NetCDF file.
        """
        if self.client is None:
            raise RuntimeError("CDS client not initialized")

        if zone_name not in _AFRICA_FIRE_ZONES:
            raise ValueError(
                f"Unknown zone: {zone_name}. "
                f"Available: {list(_AFRICA_FIRE_ZONES.keys())}"
            )

        zone = _AFRICA_FIRE_ZONES[zone_name]
        bbox = zone["bbox"]

        if months is None:
            months = [f"{m:02d}" for m in range(1, 13)]

        if variables is None:
            variables = list(_ERA5_VARIABLES.keys())

        output_file = os.path.join(self.output_dir, f"era5_land_{zone_name}_{year}.nc")

        print(f"  Downloading ERA5-Land:")
        print(f"    Zone  : {zone_name} — {zone['description']}")
        print(f"    Year  : {year}")
        print(f"    Months: {months}")
        print(f"    Vars  : {variables}")
        print(f"    Bbox  : {bbox}")
        print(f"    File  : {output_file}")

        request = {
            "format": "netcdf",
            "variable": variables,
            "year": str(year),
            "month": months,
            "day": [f"{d:02d}" for d in range(1, 32)],
            "time": [f"{h:02d}:00" for h in range(0, 24, 6)],
            "area": bbox,
        }

        try:
            self.client.retrieve("reanalysis-era5-land", request, output_file)
            print(f"  Download complete: {output_file}")
            return output_file
        except Exception as e:
            print(f"  Download error: {e}")
            raise

    def download_multiple(self, zones: List[str], years: List[int]) -> List[str]:
        """Download multiple zones and years.

        Returns:
            List of downloaded file paths.
        """
        downloaded = []
        for zone in zones:
            for year in years:
                try:
                    path = self.download_zone(zone, year)
                    downloaded.append(path)
                except Exception as e:
                    print(f"  Error {zone}/{year}: {e}")
        return downloaded

    def load_and_inspect(self, filepath: str):
        """Load and inspect a NetCDF file."""
        import xarray as xr

        print(f"  Inspecting: {filepath}")
        ds = xr.open_dataset(filepath)
        print(f"    Dimensions: {dict(ds.dims)}")
        print(f"    Variables : {list(ds.data_vars)}")
        print(f"    Period    : {ds.time.min().values} -> {ds.time.max().values}")
        return ds


def download_era5_africa(
    region: str, date_range: Tuple[str, str], variables: List[str] = None
) -> str:
    """Convenience function: download ERA5-Land for an African region.

    Args:
        region: Zone name (sahel, south_africa_kruger, south_africa_fynbos,
                madagascar, east_africa).
        date_range: Tuple of (start_date, end_date) as 'YYYY-MM-DD' strings.
        variables: List of variable names or None for all.

    Returns:
        Path to downloaded NetCDF file.
    """
    from datetime import datetime as dt

    start_date = dt.strptime(date_range[0], "%Y-%m-%d")
    year = start_date.year

    downloader = ERA5Downloader()
    return downloader.download_zone(region, year, variables=variables)


# =============================================================================
# SLOPE / ASPECT COMPUTATION (from build_real_dataset.py lines 256-338)
# =============================================================================

def compute_slope_aspect(lat: float, lon: float, wind_dir: float = 0.0) -> Dict:
    """Compute slope and aspect from a local 3x3 elevation grid.

    Fetches 5 elevation points (center, N, S, E, W) via Open-Meteo
    elevation API with ~30 m offsets, then computes gradient-based
    slope and aspect. Also computes wind-slope alignment angle.

    Args:
        lat: Latitude in decimal degrees.
        lon: Longitude in decimal degrees.
        wind_dir: Wind direction (degrees, meteorological convention).

    Returns:
        Dict with keys:
            elevation_m, slope_pct, slope_deg, aspect_deg,
            angle_wind_slope (degrees), elevations [5-list].
    """
    d_lat = 0.00027
    d_lon = 0.00027 / max(np.cos(np.radians(lat)), 0.1)

    lats_str = f"{lat},{lat + d_lat},{lat - d_lat},{lat},{lat}"
    lons_str = f"{lon},{lon},{lon},{lon + d_lon},{lon - d_lon}"

    elevation_url = "https://api.open-meteo.com/v1/elevation"
    try:
        e_res = rate_limited_get(
            elevation_url, params={"latitude": lats_str, "longitude": lons_str}
        )
        elevations = e_res.json().get("elevation", [250.0] * 5)
    except Exception:
        elevations = [250.0] * 5

    if len(elevations) < 5:
        elevations = elevations + [elevations[0]] * (5 - len(elevations))

    e_c, e_n, e_s, e_e, e_w = (
        elevations[0],
        elevations[1],
        elevations[2],
        elevations[3],
        elevations[4],
    )

    slope_x = (e_e - e_w) / 60.0
    slope_y = (e_n - e_s) / 60.0

    slope_pct = float(np.clip(np.sqrt(slope_x**2 + slope_y**2) * 100.0, 0.0, 100.0))
    slope_deg = float(np.degrees(np.arctan(slope_pct / 100.0)))

    aspect = float(np.degrees(np.arctan2(slope_x, slope_y)) % 360.0)

    wind_blow_dir = (wind_dir + 180.0) % 360.0
    angle_wind_slope = abs(wind_blow_dir - aspect) % 360.0
    if angle_wind_slope > 180.0:
        angle_wind_slope = 360.0 - angle_wind_slope

    return {
        "elevation_m": e_c,
        "slope_pct": slope_pct,
        "slope_deg": slope_deg,
        "aspect_deg": aspect,
        "angle_wind_slope": angle_wind_slope,
        "elevations": list(elevations),
    }


# =============================================================================
# BATCH WEATHER FETCHING (extracted from build_real_dataset.py)
# =============================================================================

MAX_WEATHER_SAMPLES = 5000


def fetch_weather_for_points(df: pd.DataFrame) -> pd.DataFrame:
    """Associate weather and slope/aspect to each point in a DataFrame.

    Expects columns: latitude, longitude, datetime.
    Fetches historical weather via Open-Meteo ERA5 archive and computes
    slope/aspect from a 3x3 elevation grid for each row.

    Uses thread-safe rate limiting (max 10 req/s).
    Capped at MAX_WEATHER_SAMPLES (400), with top-brightness selection
    if needed.

    Args:
        df: DataFrame with 'latitude', 'longitude', 'datetime' columns.

    Returns:
        DataFrame with weather columns appended.
    """
    import concurrent.futures

    if len(df) > MAX_WEATHER_SAMPLES:
        print(
            f"  Vector count ({len(df)}) exceeds limit ({MAX_WEATHER_SAMPLES}). "
            "Selecting hottest points..."
        )
        df = df.sort_values("brightness_k", ascending=False).head(MAX_WEATHER_SAMPLES)

    print(f"  Fetching weather + slope/aspect for {len(df)} points...")

    rows = df.to_dict("records")
    results = []

    with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
        futures = {executor.submit(_fetch_single_point, row): row for row in rows}

        for i, future in enumerate(concurrent.futures.as_completed(futures)):
            res = future.result()
            if res:
                results.append(res)
            if (i + 1) % 50 == 0 or i == len(df) - 1:
                print(f"    Associate weather: [{i + 1}/{len(df)}] points processed...")

    return pd.DataFrame(results)


def _fetch_single_point(row: dict) -> dict:
    """Fetch weather + slope/aspect for a single point.

    Args:
        row: Dict with 'latitude', 'longitude', 'datetime' keys.

    Returns:
        Augmented row dict or empty dict on failure.
    """
    lat, lon = row["latitude"], row["longitude"]
    dt = row["datetime"]
    date_str = dt.strftime("%Y-%m-%d")

    weather_url = "https://archive-api.open-meteo.com/v1/era5"
    params = {
        "latitude": lat,
        "longitude": lon,
        "start_date": date_str,
        "end_date": date_str,
        "hourly": [
            "temperature_2m",
            "relative_humidity_2m",
            "wind_speed_10m",
            "wind_direction_10m",
            "surface_pressure",
            "dewpoint_2m",
        ],
        "timezone": "auto",
    }

    try:
        w_res = rate_limited_get(weather_url, params=params)
        w_data = w_res.json()
        hourly = w_data.get("hourly", {})
        times = hourly.get("time", [])
        target_hour = dt.strftime("%Y-%m-%dT%H:00")

        h_idx = times.index(target_hour) if target_hour in times else len(times) // 2

        temp = hourly.get("temperature_2m", [25.0])[h_idx]
        rh = hourly.get("relative_humidity_2m", [40.0])[h_idx]
        wind_10m = hourly.get("wind_speed_10m", [4.0])[h_idx]
        wind_dir = hourly.get("wind_direction_10m", [0.0])[h_idx]
        dewpoint = hourly.get("dewpoint_2m", [10.0])[h_idx]

        wind_ms = wind_10m / 3.6

        slope = compute_slope_aspect(lat, lon, wind_dir)

        return {
            **row,
            "temp_c": temp,
            "rh_percent": rh,
            "wind_speed_ms": wind_ms,
            "wind_dir": wind_dir,
            "slope_pct": slope["slope_pct"],
            "slope_deg": slope["slope_deg"],
            "aspect_deg": slope["aspect_deg"],
            "angle_wind_slope": slope["angle_wind_slope"],
            "elevation_m": slope["elevation_m"],
            "dewpoint_c": dewpoint,
        }
    except Exception:
        return {}


# =============================================================================
# SATELLITE FALLBACK (from Download_data.py)
# =============================================================================

class SatelliteFallback:
    """Seasonal satellite data estimates when real data is unavailable."""

    @staticmethod
    def get_ndvi(lat: float, lon: float, date: datetime) -> Dict:
        """Estimate NDVI from latitude and season."""
        month = date.month
        if lat > 20:
            ndvi = 0.25 if month in [6, 7, 8, 9] else 0.45
        else:
            ndvi = 0.2 if month in [11, 12, 1, 2] else 0.5
        return {
            "ndvi": ndvi,
            "ndwi": ndvi * 0.3 - 0.1,
            "lst": 30.0 + (20 if month in [6, 7, 8] else 10),
            "evi": ndvi * 0.85,
            "savi": ndvi * 0.9,
            "source": "fallback",
        }
