"""
download_data.py
================
Module de téléchargement et de mise en cache des données 
géospatiales pour le pipeline BurnTrack.

Sources :
- ERA5-Land (ECMWF) : météo horaire via Google Earth Engine
- Copernicus DEM : élévation 30m via GEE
- MODIS NDVI : végétation via GEE
- VIIRS FIRMS : feux actifs via NASA
- Global FCCS : fuel models (téléchargement manuel Pangaea)

Architecture :
1. Cache local (./data/cache/) pour éviter les re-téléchargements
2. Téléchargement GEE via ee.ImageCollection
3. Prétraitement : reprojection, découpage, interpolation temporelle
4. Export au format NetCDF / GeoTIFF / CSV selon le besoin
"""

import os
import json
import pickle
import hashlib
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional, Union
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import requests
from PIL import Image

# =============================================================================
# CONFIGURATION
# =============================================================================

DATA_DIR = Path("./data")
CACHE_DIR = DATA_DIR / "cache"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"

# Création des dossiers
for d in [CACHE_DIR, RAW_DIR, PROCESSED_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# =============================================================================
# CLASSES DE DONNÉES
# =============================================================================

@dataclass
class GeoExtent:
    """Rectangle géographique [min_lon, min_lat, max_lon, max_lat]."""
    min_lon: float
    min_lat: float
    max_lon: float
    max_lat: float
    
    def to_ee_geometry(self) -> dict:
        """Convertit en GeoJSON rectangle pour Earth Engine."""
        return {
            "type": "Polygon",
            "coordinates": [[
                [self.min_lon, self.min_lat],
                [self.max_lon, self.min_lat],
                [self.max_lon, self.max_lat],
                [self.min_lon, self.max_lat],
                [self.min_lon, self.min_lat]
            ]]
        }
    
    def center(self) -> Tuple[float, float]:
        return ((self.min_lon + self.max_lon) / 2, 
                (self.min_lat + self.max_lat) / 2)


@dataclass
class DataRequest:
    """Requête de données avec métadonnées."""
    extent: GeoExtent
    start_date: datetime
    end_date: datetime
    variables: List[str]
    resolution_m: int = 1000  # Résolution cible en mètres


# =============================================================================
# CACHE MANAGER
# =============================================================================

class CacheManager:
    """Gestionnaire de cache basé sur hash de la requête."""
    
    def __init__(self, cache_dir: Path = CACHE_DIR):
        self.cache_dir = cache_dir
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.index_file = cache_dir / "index.json"
        self.index = self._load_index()
    
    def _load_index(self) -> Dict:
        if self.index_file.exists():
            with open(self.index_file, 'r') as f:
                return json.load(f)
        return {}
    
    def _save_index(self):
        with open(self.index_file, 'w') as f:
            json.dump(self.index, f, indent=2, default=str)
    
    def _compute_hash(self, request: DataRequest, source: str) -> str:
        """Hash unique pour une requête + source."""
        key = f"{source}:{request.extent.min_lon}:{request.extent.min_lat}:" \
              f"{request.extent.max_lon}:{request.extent.max_lat}:" \
              f"{request.start_date.isoformat()}:{request.end_date.isoformat()}:" \
              f"{','.join(sorted(request.variables))}:{request.resolution_m}"
        return hashlib.md5(key.encode()).hexdigest()[:16]
    
    def get(self, request: DataRequest, source: str) -> Optional[Path]:
        """Récupère un fichier du cache s'il existe."""
        h = self._compute_hash(request, source)
        if h in self.index and (self.cache_dir / self.index[h]).exists():
            return self.cache_dir / self.index[h]
        return None
    
    def put(self, request: DataRequest, source: str, filepath: Path) -> Path:
        """Stocke un fichier dans le cache."""
        h = self._compute_hash(request, source)
        cache_path = self.cache_dir / f"{source}_{h}.nc"
        
        # Si c'est déjà un NetCDF, on le copie
        if str(filepath) != str(cache_path):
            import shutil
            shutil.copy(filepath, cache_path)
        
        self.index[h] = cache_path.name
        self._save_index()
        return cache_path


# =============================================================================
# EARTH ENGINE CONNECTOR
# =============================================================================

class EarthEngineDownloader:
    """
    Téléchargeur via Google Earth Engine.
    Nécessite l'authentification : earthengine authenticate
    """
    
    def __init__(self):
        self.initialized = False
        self._init_ee()
    
    def _init_ee(self):
        """Initialise Earth Engine."""
        try:
            import ee
            if not self.initialized:
                ee.Initialize()
                self.initialized = True
                self.ee = ee
                print("✅ Earth Engine initialisé")
        except Exception as e:
            print(f"⚠️ Earth Engine non disponible : {e}")
            print("   Installez : pip install earthengine-api")
            print("   Authentifiez : earthengine authenticate")
            self.ee = None
    
    def download_era5_land(self, request: DataRequest, 
                          cache: CacheManager) -> Optional[Path]:
        """
        Télécharge ERA5-Land (température, vent, humidité, précipitations).
        
        Variables disponibles :
        - temperature_2m (°C)
        - u_component_of_wind_10m, v_component_of_wind_10m (m/s)
        - dewpoint_temperature_2m (°C)
        - total_precipitation (m)
        """
        if self.ee is None:
            return None
        
        # Vérification cache
        cached = cache.get(request, "era5_land")
        if cached:
            print(f"📦 ERA5-Land trouvé en cache : {cached}")
            return cached
        
        print(f"⬇️ Téléchargement ERA5-Land ({request.start_date.date()} → {request.end_date.date()})...")
        
        # Sélection des variables
        var_map = {
            'temperature_2m': 'temperature_2m',
            'wind_u_10m': 'u_component_of_wind_10m',
            'wind_v_10m': 'v_component_of_wind_10m',
            'dewpoint_2m': 'dewpoint_temperature_2m',
            'precipitation': 'total_precipitation'
        }
        
        # Construction de la requête GEE
        collection = self.ee.ImageCollection("ECMWF/ERA5_LAND/HOURLY") \
            .filterBounds(self.ee.Geometry.Rectangle([
                request.extent.min_lon, request.extent.min_lat,
                request.extent.max_lon, request.extent.max_lat
            ])) \
            .filterDate(
                request.start_date.strftime('%Y-%m-%d'),
                request.end_date.strftime('%Y-%m-%d')
            )
        
        # Sélection des bandes
        bands = [var_map[v] for v in request.variables if v in var_map]
        if not bands:
            print("❌ Aucune variable ERA5 valide demandée")
            return None
        
        collection = collection.select(bands)
        
        # Moyenne temporelle ou série complète selon la durée
        days = (request.end_date - request.start_date).days
        if days <= 7:
            # Série temporelle complète
            image = collection.toBands()
        else:
            # Moyenne journalière
            image = collection.mean()
        
        # Téléchargement
        url = image.getDownloadURL({
            'scale': request.resolution_m,
            'region': request.extent.to_ee_geometry(),
            'format': 'GEO_TIFF'
        })
        
        # Téléchargement local
        output_path = RAW_DIR / f"era5_land_{request.start_date.strftime('%Y%m%d')}.tif"
        self._download_url(url, output_path)
        
        # Mise en cache
        return cache.put(request, "era5_land", output_path)
    
    def download_copernicus_dem(self, request: DataRequest,
                                 cache: CacheManager) -> Optional[Path]:
        """
        Télécharge Copernicus DEM 30m (élévation, pente, aspect).
        """
        if self.ee is None:
            return None
        
        cached = cache.get(request, "copernicus_dem")
        if cached:
            return cached
        
        print("⬇️ Téléchargement Copernicus DEM...")
        
        dem = self.ee.Image("COPERNICUS/DEM/GLO30") \
            .select('DEM') \
            .clip(self.ee.Geometry.Rectangle([
                request.extent.min_lon, request.extent.min_lat,
                request.extent.max_lon, request.extent.max_lat
            ]))
        
        # Calcul de la pente et de l'aspect
        slope = self.ee.Terrain.slope(dem)
        aspect = self.ee.Terrain.aspect(dem)
        
        combined = dem.addBands(slope).addBands(aspect)
        
        url = combined.getDownloadURL({
            'scale': 30,
            'region': request.extent.to_ee_geometry(),
            'format': 'GEO_TIFF'
        })
        
        output_path = RAW_DIR / f"copernicus_dem_{request.extent.center()[0]:.2f}_{request.extent.center()[1]:.2f}.tif"
        self._download_url(url, output_path)
        
        return cache.put(request, "copernicus_dem", output_path)
    
    def download_modis_ndvi(self, request: DataRequest,
                             cache: CacheManager) -> Optional[Path]:
        """
        Télécharge MODIS NDVI (16 jours, 250m).
        """
        if self.ee is None:
            return None
        
        cached = cache.get(request, "modis_ndvi")
        if cached:
            return cached
        
        print("⬇️ Téléchargement MODIS NDVI...")
        
        # MODIS NDVI 16-day composite
        collection = self.ee.ImageCollection("MODIS/061/MOD13Q1") \
            .filterBounds(self.ee.Geometry.Rectangle([
                request.extent.min_lon, request.extent.min_lat,
                request.extent.max_lon, request.extent.max_lat
            ])) \
            .filterDate(
                request.start_date.strftime('%Y-%m-%d'),
                request.end_date.strftime('%Y-%m-%d')
            ) \
            .select('NDVI')
        
        # NDVI est un ratio × 10000 dans MODIS
        image = collection.mean().multiply(0.0001)
        
        url = image.getDownloadURL({
            'scale': max(request.resolution_m, 250),
            'region': request.extent.to_ee_geometry(),
            'format': 'GEO_TIFF'
        })
        
        output_path = RAW_DIR / f"modis_ndvi_{request.start_date.strftime('%Y%m%d')}.tif"
        self._download_url(url, output_path)
        
        return cache.put(request, "modis_ndvi", output_path)
    
    def _download_url(self, url: str, output_path: Path):
        """Télécharge une URL vers un fichier local."""
        response = requests.get(url, stream=True)
        response.raise_for_status()
        
        with open(output_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        
        print(f"   ✅ Sauvegardé : {output_path}")


# =============================================================================
# NASA FIRMS (VIIRS Active Fires)
# =============================================================================

class FIRMSDownloader:
    """
    Télécharge les feux actifs depuis NASA FIRMS.
    API gratuite : https://firms.modaps.eosdis.nasa.gov/api/
    """
    
    FIRMS_URL = "https://firms.modaps.eosdis.nasa.gov/api/area/csv/{source}/{api_key}/{extent}/{date_range}"
    
    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or os.getenv("NASA_FIRMS_API_KEY", "YOUR_API_KEY")
    
    def download_viirs_fires(self, extent: GeoExtent, 
                            start_date: datetime, 
                            end_date: datetime,
                            cache: CacheManager) -> Optional[pd.DataFrame]:
        """
        Télécharge les détections de feux VIIRS.
        
        Retourne un DataFrame avec :
        - latitude, longitude
        - brightness (température du pixel)
        - confidence (nominal, low, high)
        - acq_date, acq_time
        """
        request = DataRequest(
            extent=extent, start_date=start_date, end_date=end_date,
            variables=["viirs_fires"], resolution_m=375
        )
        
        cached = cache.get(request, "viirs_fires")
        if cached:
            return pd.read_csv(cached)
        
        print(f"⬇️ Téléchargement VIIRS FIRMS...")
        
        # Format extent : W,S,E,N
        extent_str = f"{extent.min_lon},{extent.min_lat},{extent.max_lon},{extent.max_lat}"
        date_str = f"{start_date.strftime('%Y-%m-%d')}:{end_date.strftime('%Y-%m-%d')}"
        
        url = self.FIRMS_URL.format(
            source="VJ1NP1_NRT",  # VIIRS NOAA-20 Near Real-Time
            api_key=self.api_key,
            extent=extent_str,
            date_range=date_str
        )
        
        try:
            df = pd.read_csv(url)
            # Sauvegarde cache
            cache_path = CACHE_DIR / f"viirs_fires_{start_date.strftime('%Y%m%d')}.csv"
            df.to_csv(cache_path, index=False)
            cache.put(request, "viirs_fires", cache_path)
            return df
        except Exception as e:
            print(f"❌ Erreur FIRMS : {e}")
            return None


# =============================================================================
# PRÉTRAITEMENT
# =============================================================================

class DataPreprocessor:
    """
    Prétraite les données brutes pour le pipeline BurnTrack.
    """
    
    def __init__(self):
        pass
    
    def extract_weather_features(self, era5_path: Path, 
                                  lat: float, lon: float) -> Dict:
        """
        Extrait les features météo à un point donné depuis ERA5.
        """
        try:
            import rasterio
            from rasterio.sample import sample_gen
            
            features = {}
            
            with rasterio.open(era5_path) as src:
                # Sample au point (lon, lat)
                for i, band_name in enumerate(src.descriptions):
                    value = next(sample_gen(src, [(lon, lat)], indexes=i+1))
                    features[band_name] = float(value[0])
            
            # Calcul du vent résultant
            if 'u_component_of_wind_10m' in features and 'v_component_of_wind_10m' in features:
                u = features['u_component_of_wind_10m']
                v = features['v_component_of_wind_10m']
                features['wind_speed_10m'] = np.sqrt(u**2 + v**2)
                features['wind_direction_10m'] = (np.degrees(np.arctan2(v, u)) + 360) % 360
            
            # Calcul de l'humidité relative depuis la température et le point de rosée
            if 'temperature_2m' in features and 'dewpoint_temperature_2m' in features:
                T = features['temperature_2m'] - 273.15  # K → °C
                Td = features['dewpoint_temperature_2m'] - 273.15
                # Formule Magnus-Tetens
                es = 0.6108 * np.exp(17.27 * T / (T + 237.3))
                ed = 0.6108 * np.exp(17.27 * Td / (Td + 237.3))
                features['relative_humidity'] = 100 * ed / es
                features['temperature_2m_c'] = T
            
            return features
            
        except ImportError:
            print("⚠️ rasterio non installé : pip install rasterio")
            return {}
    
    def extract_elevation_features(self, dem_path: Path,
                                    lat: float, lon: float) -> Dict:
        """
        Extrait élévation, pente, aspect à un point donné.
        """
        try:
            import rasterio
            
            features = {}
            bands = ['elevation', 'slope', 'aspect']
            
            with rasterio.open(dem_path) as src:
                for i, name in enumerate(bands):
                    if i < src.count:
                        value = next(rasterio.sample.sample_gen(src, [(lon, lat)], indexes=i+1))
                        features[name] = float(value[0])
            
            return features
            
        except ImportError:
            return {}
    
    def extract_ndvi(self, ndvi_path: Path, lat: float, lon: float) -> float:
        """Extrait NDVI à un point donné."""
        try:
            import rasterio
            with rasterio.open(ndvi_path) as src:
                value = next(rasterio.sample.sample_gen(src, [(lon, lat)]))
                return float(value[0])
        except:
            return 0.0


# =============================================================================
# PIPELINE COMPLET
# =============================================================================

class BurnTrackDataPipeline:
    """
    Pipeline unifié de téléchargement et de prétraitement.
    """
    
    def __init__(self, api_key_firms: Optional[str] = None):
        self.cache = CacheManager()
        self.ee = EarthEngineDownloader()
        self.firms = FIRMSDownloader(api_key_firms)
        self.preprocessor = DataPreprocessor()
    
    def fetch_all(self, extent: GeoExtent, date: datetime,
                  lat_target: float, lon_target: float) -> Dict:
        """
        Télécharge toutes les données pour une date et un point donnés.
        
        Returns:
            Dict avec toutes les features prêtes pour le pipeline
        """
        # Fenêtre de 7 jours pour les moyennes
        start = date - timedelta(days=3)
        end = date + timedelta(days=3)
        
        request = DataRequest(
            extent=extent, start_date=start, end_date=end,
            variables=['temperature_2m', 'wind_u_10m', 'wind_v_10m', 
                      'dewpoint_2m', 'precipitation'],
            resolution_m=9000
        )
        
        results = {
            'date': date.isoformat(),
            'lat': lat_target,
            'lon': lon_target,
            'sources': {}
        }
        
        # 1. Météo
        era5_path = self.ee.download_era5_land(request, self.cache)
        if era5_path:
            weather = self.preprocessor.extract_weather_features(
                era5_path, lat_target, lon_target
            )
            results['weather'] = weather
            results['sources']['era5_land'] = str(era5_path)
        
        # 2. Élévation
        dem_request = DataRequest(
            extent=extent, start_date=start, end_date=end,
            variables=['elevation'], resolution_m=30
        )
        dem_path = self.ee.download_copernicus_dem(dem_request, self.cache)
        if dem_path:
            terrain = self.preprocessor.extract_elevation_features(
                dem_path, lat_target, lon_target
            )
            results['terrain'] = terrain
            results['sources']['copernicus_dem'] = str(dem_path)
        
        # 3. NDVI
        ndvi_request = DataRequest(
            extent=extent, start_date=start, end_date=end,
            variables=['ndvi'], resolution_m=250
        )
        ndvi_path = self.ee.download_modis_ndvi(ndvi_request, self.cache)
        if ndvi_path:
            ndvi = self.preprocessor.extract_ndvi(ndvi_path, lat_target, lon_target)
            results['ndvi'] = ndvi
            results['sources']['modis_ndvi'] = str(ndvi_path)
        
        # 4. Feux actifs (dernières 24h)
        fires_df = self.firms.download_viirs_fires(
            extent, date - timedelta(days=1), date, self.cache
        )
        if fires_df is not None and len(fires_df) > 0:
            # Distance au feu le plus proche
            from math import radians, sin, cos, sqrt, atan2
            def haversine(lat1, lon1, lat2, lon2):
                R = 6371000
                phi1, phi2 = radians(lat1), radians(lat2)
                dphi = radians(lat2 - lat1)
                dlambda = radians(lon2 - lon1)
                a = sin(dphi/2)**2 + cos(phi1)*cos(phi2)*sin(dlambda/2)**2
                return 2 * R * atan2(sqrt(a), sqrt(1-a))
            
            distances = fires_df.apply(
                lambda row: haversine(lat_target, lon_target, 
                                     row['latitude'], row['longitude']), 
                axis=1
            )
            results['active_fire_proximity_m'] = float(distances.min())
            results['active_fire_count'] = len(fires_df)
        else:
            results['active_fire_proximity_m'] = None
            results['active_fire_count'] = 0
        
        return results


# =============================================================================
# EXEMPLE D'UTILISATION
# =============================================================================

if __name__ == "__main__":
    print("=" * 70)
    print("BURNTRACK DATA PIPELINE - TEST")
    print("=" * 70)
    
    # Zone de test : Marrakech, Maroc
    extent = GeoExtent(
        min_lon=-8.5, min_lat=31.5,
        max_lon=-7.5, max_lat=32.5
    )
    
    pipeline = BurnTrackDataPipeline()
    
    # Test sans Earth Engine (mode offline)
    print("\n📋 Structure du pipeline :")
    print(f"   Cache : {CACHE_DIR}")
    print(f"   Raw   : {RAW_DIR}")
    print(f"   Processed : {PROCESSED_DIR}")
    
    # Exemple de requête
    date = datetime(2024, 8, 15)  # été marocain
    
    print(f"\n🌍 Zone : Marrakech ({extent.min_lat}°N, {extent.min_lon}°W)")
    print(f"📅 Date : {date.date()}")
    
    print("\n⚠️  Pour exécuter le téléchargement réel :")
    print("   1. pip install earthengine-api rasterio requests")
    print("   2. earthengine authenticate")
    print("   3. Définir NASA_FIRMS_API_KEY (optionnel)")
    print("   4. Relancer le script")
    
    print("\n✅ Module prêt à l'emploi")
