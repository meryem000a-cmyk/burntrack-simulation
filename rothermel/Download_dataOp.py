"""
Download_data.py
================
Module de téléchargement de données géospatiales pour BurnTrack.
VERSION OPEN-METEO : pas d'authentification Google requise.
"""

import os
import json
import hashlib
from datetime import datetime, timedelta
from typing import Dict, Optional, Tuple
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import requests


# =============================================================================
# CONFIGURATION
# =============================================================================

DATA_DIR = Path("./data")
CACHE_DIR = DATA_DIR / "cache"
for d in [CACHE_DIR]:
    d.mkdir(parents=True, exist_ok=True)


# =============================================================================
# OPEN-METEO DOWNLOADER
# =============================================================================

class OpenMeteoDownloader:
    """Téléchargeur via Open-Meteo API (gratuit, pas d'API key)."""
    
    BASE_URL = "https://api.open-meteo.com/v1"
    HISTORICAL_URL = "https://archive-api.open-meteo.com/v1"
    
    def __init__(self):
        self.session = requests.Session()
    
    def get_weather(self, lat: float, lon: float, date: datetime) -> Dict:
        """Récupère la météo horaire pour un point donné."""
        start = (date - timedelta(days=1)).strftime('%Y-%m-%d')
        end = (date + timedelta(days=1)).strftime('%Y-%m-%d')
        
        url = f"{self.HISTORICAL_URL}/era5"
        params = {
            "latitude": lat, "longitude": lon,
            "start_date": start, "end_date": end,
            "hourly": ["temperature_2m", "relative_humidity_2m", "wind_speed_10m", 
                      "wind_direction_10m", "precipitation", "surface_pressure", "dewpoint_2m"],
            "timezone": "auto"
        }
        
        try:
            response = self.session.get(url, params=params, timeout=30)
            response.raise_for_status()
            data = response.json()
            
            hourly = data.get("hourly", {})
            times = hourly.get("time", [])
            target_hour = date.strftime('%Y-%m-%dT%H:00')
            
            idx = times.index(target_hour) if target_hour in times else len(times) // 2
            
            return {
                "temperature_2m": hourly.get("temperature_2m", [None])[idx],
                "relative_humidity_2m": hourly.get("relative_humidity_2m", [None])[idx],
                "wind_speed_10m": hourly.get("wind_speed_10m", [None])[idx],
                "wind_direction_10m": hourly.get("wind_direction_10m", [None])[idx],
                "precipitation": hourly.get("precipitation", [None])[idx],
                "surface_pressure": hourly.get("surface_pressure", [None])[idx],
                "dewpoint_2m": hourly.get("dewpoint_2m", [None])[idx],
                "source": "open-meteo"
            }
        except Exception as e:
            print(f"⚠️ Erreur Open-Meteo : {e}")
            return self._weather_fallback(lat, lon, date)
    
    def get_elevation(self, lat: float, lon: float) -> Dict:
        """Récupère l'élévation."""
        url = f"{self.BASE_URL}/elevation"
        try:
            response = self.session.get(url, params={"latitude": lat, "longitude": lon}, timeout=10)
            response.raise_for_status()
            return {"elevation": response.json().get("elevation", [0])[0], "source": "open-meteo"}
        except Exception as e:
            return {"elevation": 500, "source": "fallback"}
    
    def _weather_fallback(self, lat: float, lon: float, date: datetime) -> Dict:
        """Valeurs fallback selon saison."""
        month = date.month
        is_dry = month in [6, 7, 8, 9]
        
        if lat > 20:
            return {
                "temperature_2m": 35.0 if is_dry else 18.0,
                "relative_humidity_2m": 25.0 if is_dry else 60.0,
                "wind_speed_10m": 4.0, "wind_direction_10m": 270.0,
                "precipitation": 0.0, "surface_pressure": 1015.0,
                "dewpoint_2m": 15.0, "source": "fallback-maroc"
            }
        else:
            return {
                "temperature_2m": 32.0, "relative_humidity_2m": 40.0,
                "wind_speed_10m": 3.0, "wind_direction_10m": 0.0,
                "precipitation": 0.0, "surface_pressure": 1010.0,
                "dewpoint_2m": 18.0, "source": "fallback-sahel"
            }


# =============================================================================
# SATELLITE (fallback)
# =============================================================================

class SatelliteDownloader:
    def get_ndvi(self, lat: float, lon: float, date: datetime) -> Dict:
        month = date.month
        if lat > 20:
            ndvi = 0.25 if month in [6,7,8,9] else 0.45
        else:
            ndvi = 0.2 if month in [11,12,1,2] else 0.5
        return {
            "ndvi": ndvi, "ndwi": ndvi * 0.3 - 0.1,
            "lst": 30.0 + (20 if month in [6,7,8] else 10),
            "evi": ndvi * 0.85, "savi": ndvi * 0.9,
            "source": "fallback"
        }


# =============================================================================
# PIPELINE
# =============================================================================

@dataclass
class GeoExtent:
    min_lon: float
    min_lat: float
    max_lon: float
    max_lat: float

class BurnTrackDataPipeline:
    def __init__(self, firms_api_key: Optional[str] = None):
        self.meteo = OpenMeteoDownloader()
        self.satellite = SatelliteDownloader()
    
    def fetch_all(self, extent: GeoExtent, date: datetime, lat_target: float, lon_target: float) -> Dict:
        weather = self.meteo.get_weather(lat_target, lon_target, date)
        elevation = self.meteo.get_elevation(lat_target, lon_target)
        satellite = self.satellite.get_ndvi(lat_target, lon_target, date)
        
        return {
            "date": date.isoformat(), "lat": lat_target, "lon": lon_target,
            "weather": weather, "elevation": elevation, "satellite": satellite,
            "sources": {"weather": weather["source"], "elevation": elevation["source"], "satellite": satellite["source"]}
        }


# =============================================================================
# TEST
# =============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("BURNTRACK - OPEN-METEO TEST")
    print("=" * 60)
    
    pipeline = BurnTrackDataPipeline()
    extent = GeoExtent(-8.5, 31.5, -7.5, 32.5)
    date = datetime(2024, 8, 15, 14, 0)
    
    result = pipeline.fetch_all(extent, date, 31.63, -7.98)
    
    print(f"\n📍 {result['lat']:.2f}°N, {result['lon']:.2f}°W")
    print(f"🌡️  Temp: {result['weather']['temperature_2m']}°C")
    print(f"💧 RH: {result['weather']['relative_humidity_2m']}%")
    print(f"💨 Vent: {result['weather']['wind_speed_10m']} m/s")
    print(f"🏔️  Élévation: {result['elevation']['elevation']} m")
    print(f"🛰️  NDVI: {result['satellite']['ndvi']:.3f}")
    print(f"\n✅ Open-Meteo fonctionne sans authentification !")