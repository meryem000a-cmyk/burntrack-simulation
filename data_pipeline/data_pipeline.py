"""
data_pipeline.py
================
Pipeline unifié BurnTrack : données géospatiales → features → Rothermel → IA.

Ce module fait le pont entre :
  - download_data.py    (téléchargement ERA5-Land, DEM, NDVI, FIRMS)
  - features_engineering.py (calcul VPD, DFMC, stress_index, etc.)
  - fuel_models.py       (sélection du fuel model par espèce/écosystème)
  - rothermel_engine.py  (calcul ROS, flame_length, etc.)
  - ia_corrector.py      (préparation du vecteur de features pour l'IA)

Usage :
    pipeline = BurnTrackPipeline()
    result = pipeline.run(
        lat=31.63, lon=-7.98, date=datetime(2024, 8, 15),
        species="acacia",  # ou ecosystem="steppe"
        robot_data={"temp_air": 38, "rh": 20, "wind_speed": 4, "slope_deg": 15}
    )
"""

import os
import json
import warnings
from datetime import datetime, timedelta
from typing import Dict, Optional, Tuple, Any
from pathlib import Path

import numpy as np

# =============================================================================
# IMPORTS DES MODULES BURNTRACK (supposés dans le même dossier)
# =============================================================================

try:
    from download_data import (
        BurnTrackDataPipeline, GeoExtent, CacheManager,
        DATA_DIR, CACHE_DIR, RAW_DIR, PROCESSED_DIR
    )
    DOWNLOAD_AVAILABLE = True
except ImportError:
    DOWNLOAD_AVAILABLE = False
    warnings.warn("download_data.py non trouvé — mode offline uniquement")

try:
    from features_engineering import (
        FeaturesEngineering, RobotSensors, WeatherAPI, SatelliteData
    )
    FEATURES_AVAILABLE = True
except ImportError:
    FEATURES_AVAILABLE = False

try:
    from fuel_models import get_fuel_model_by_species, get_fuel_model_by_ecosystem
    FUELS_AVAILABLE = True
except ImportError:
    FUELS_AVAILABLE = False

try:
    from rothermel_engine import BurnTrackRothermel, MoistureInputs, EnvironmentalConditions
    ROTHERMEL_AVAILABLE = True
except ImportError:
    ROTHERMEL_AVAILABLE = False

try:
    from ia_corrector import prepare_features_for_ia, FEATURE_NAMES_IA
    IA_AVAILABLE = True
except ImportError:
    IA_AVAILABLE = False


# =============================================================================
# CONFIGURATION & FALLBACKS
# =============================================================================

class SeasonalFallbacks:
    """
    Valeurs par défaut saisonnières par région quand les données 
    géospatiales ne sont pas disponibles (mode offline).
    """
    
    # Climatologie approximative : {mois: {variable: valeur}}
    MOROCCO_SUMMER = {
        "temp_air": 38.0, "rh": 20.0, "wind_speed": 4.0,
        "temp_2m": 35.0, "wind_10m": 5.0, "wind_gust": 8.0,
        "precip_1h": 0.0, "pressure": 1015.0,
        "ndvi": 0.25, "ndwi": -0.15, "lst": 48.0,
        "surface_temp": 52.0
    }
    
    MOROCCO_WINTER = {
        "temp_air": 15.0, "rh": 65.0, "wind_speed": 3.0,
        "temp_2m": 12.0, "wind_10m": 4.0, "wind_gust": 6.0,
        "precip_1h": 2.0, "pressure": 1020.0,
        "ndvi": 0.45, "ndwi": 0.10, "lst": 22.0,
        "surface_temp": 18.0
    }
    
    SAHEL_DRY = {
        "temp_air": 42.0, "rh": 12.0, "wind_speed": 3.5,
        "temp_2m": 40.0, "wind_10m": 4.5, "wind_gust": 7.0,
        "precip_1h": 0.0, "pressure": 1010.0,
        "ndvi": 0.15, "ndwi": -0.25, "lst": 52.0,
        "surface_temp": 55.0
    }
    
    @classmethod
    def get_fallback(cls, lat: float, lon: float, date: datetime) -> Dict[str, float]:
        """Retourne les valeurs fallback selon la localisation et la saison."""
        month = date.month
        
        # Détection saison sèche / humide approximative
        if lat > 20 and lat < 40 and lon > -20 and lon < 20:
            # Afrique du Nord / Maroc
            if month in [6, 7, 8, 9]:
                return cls.MOROCCO_SUMMER.copy()
            else:
                return cls.MOROCCO_WINTER.copy()
        elif lat < 20 and lat > -10:
            # Sahel / Afrique subsaharienne
            return cls.SAHEL_DRY.copy()
        else:
            # Default tempéré
            return cls.MOROCCO_SUMMER.copy()


# =============================================================================
# PIPELINE PRINCIPAL
# =============================================================================

class BurnTrackPipeline:
    """
    Pipeline unifié BurnTrack.
    
    Orchestration complète :
    1. Téléchargement données géospatiales (si online)
    2. Construction des objets RobotSensors / WeatherAPI / SatelliteData
    3. Calcul des features engineering
    4. Sélection du fuel model (vision ou écosystème)
    5. Calcul Rothermel
    6. Préparation du vecteur IA
    7. Retour structuré
    """
    
    def __init__(self, use_gee: bool = True, firms_api_key: Optional[str] = None):
        """
        Args:
            use_gee: Active le téléchargement via Google Earth Engine
            firms_api_key: Clé API NASA FIRMS (optionnel)
        """
        self.use_gee = use_gee and DOWNLOAD_AVAILABLE
        self.data_downloader = None
        self.features_engineer = FeaturesEngineering() if FEATURES_AVAILABLE else None
        self.cache = {}
        
        if self.use_gee:
            try:
                self.data_downloader = BurnTrackDataPipeline(api_key_firms=firms_api_key)
                print("✅ Pipeline de téléchargement initialisé")
            except Exception as e:
                print(f"⚠️ Impossible d'initialiser GEE : {e}")
                self.use_gee = False
    
    def _fetch_geospatial_data(self, lat: float, lon: float, 
                                date: datetime) -> Dict[str, Any]:
        """Télécharge les données géospatiales ou retourne fallback."""
        if not self.use_gee or self.data_downloader is None:
            return SeasonalFallbacks.get_fallback(lat, lon, date)
        
        # Extent autour du point (0.5° de buffer)
        extent = GeoExtent(
            min_lon=lon - 0.5, min_lat=lat - 0.5,
            max_lon=lon + 0.5, max_lat=lat + 0.5
        )
        
        try:
            raw_data = self.data_downloader.fetch_all(extent, date, lat, lon)
            return self._normalize_raw_data(raw_data)
        except Exception as e:
            warnings.warn(f"Erreur téléchargement GEE : {e}. Fallback activé.")
            return SeasonalFallbacks.get_fallback(lat, lon, date)
    
    def _normalize_raw_data(self, raw: Dict) -> Dict[str, float]:
        """Normalise les données brutes téléchargées en variables standard."""
        result = {}
        
        # Météo depuis ERA5-Land
        if 'weather' in raw:
            w = raw['weather']
            result['temp_air'] = w.get('temperature_2m_c', w.get('temperature_2m', 25))
            result['temp_2m'] = result['temp_air']
            result['wind_10m'] = w.get('wind_speed_10m', 3.0)
            result['wind_gust'] = w.get('wind_speed_10m', 3.0) * 1.5
            result['precip_1h'] = w.get('total_precipitation', 0.0) * 1000  # m → mm
            result['pressure'] = 1013.25  # ERA5-Land nécessite conversion
            result['rh'] = w.get('relative_humidity', 40.0)
            result['wind_speed'] = result['wind_10m']  # fallback si pas robot
        
        # Terrain depuis Copernicus DEM
        if 'terrain' in raw:
            t = raw['terrain']
            result['slope_deg'] = t.get('slope', 5.0)
            result['aspect_deg'] = t.get('aspect', 0.0)
            result['elevation_m'] = t.get('elevation', 500.0)
        
        # Satellite
        result['ndvi'] = raw.get('ndvi', 0.3)
        result['ndwi'] = raw.get('ndwi', 0.0)
        result['lst'] = raw.get('weather', {}).get('skin_temperature', result['temp_air'] + 5)
        result['evi'] = result['ndvi'] * 0.8
        result['savi'] = result['ndvi'] * 0.9
        
        # Capteurs robot (estimés depuis données environnementales)
        result['surface_temp'] = result.get('lst', result['temp_air'] + 8)
        result['co_ppm'] = 0.5
        result['co2_ppm'] = 420.0
        
        # Proximité feu actif
        result['active_fire_proximity_m'] = raw.get('active_fire_proximity_m')
        result['active_fire_count'] = raw.get('active_fire_count', 0)
        
        return result
    
    def _build_inputs(self, data: Dict, robot_override: Optional[Dict] = None) -> Tuple:
        """
        Construit les objets RobotSensors, WeatherAPI, SatelliteData.
        
        Args:
            data: Données normalisées (géospatiales ou fallback)
            robot_override: Données capteurs robot si disponibles
        """
        # Priorité aux données robot si fournies
        if robot_override:
            robot = RobotSensors(
                temp_air=robot_override.get('temp_air', data['temp_air']),
                rh=robot_override.get('rh', data['rh']),
                wind_speed=robot_override.get('wind_speed', data['wind_speed']),
                slope_deg=robot_override.get('slope_deg', data.get('slope_deg', 5.0)),
                aspect_deg=robot_override.get('aspect_deg', data.get('aspect_deg', 0.0)),
                surface_temp=robot_override.get('surface_temp', data.get('surface_temp')),
                co_ppm=robot_override.get('co_ppm', data.get('co_ppm')),
                co2_ppm=robot_override.get('co2_ppm', data.get('co2_ppm'))
            )
        else:
            robot = RobotSensors(
                temp_air=data['temp_air'],
                rh=data['rh'],
                wind_speed=data['wind_speed'],
                slope_deg=data.get('slope_deg', 5.0),
                aspect_deg=data.get('aspect_deg', 0.0),
                surface_temp=data.get('surface_temp'),
                co_ppm=data.get('co_ppm'),
                co2_ppm=data.get('co2_ppm')
            )
        
        weather = WeatherAPI(
            temp_2m=data['temp_2m'],
            wind_10m=data['wind_10m'],
            wind_gust=data.get('wind_gust'),
            precip_1h=data.get('precip_1h') if data.get('precip_1h', 0) > 0 else None,
            pressure=data.get('pressure'),
            dew_point=None  # Calculé plus tard si besoin
        )
        
        satellite = SatelliteData(
            ndvi=data['ndvi'],
            ndwi=data['ndwi'],
            lst=data['lst'],
            evi=data.get('evi'),
            savi=data.get('savi'),
            burned_area=data.get('active_fire_proximity_m')
        )
        
        return robot, weather, satellite
    
    def _select_fuel_model(self, species: Optional[str] = None,
                          ecosystem: Optional[str] = None,
                          default: str = "AF_STEPPE") -> str:
        """Sélectionne le fuel model par espèce ou écosystème."""
        if not FUELS_AVAILABLE:
            return default
        
        if species:
            fm = get_fuel_model_by_species(species)
            if fm:
                return fm.code
        
        if ecosystem:
            fm = get_fuel_model_by_ecosystem(ecosystem)
            if fm:
                return fm.code
        
        return default
    
    def _compute_rothermel(self, fuel_code: str, robot: RobotSensors,
                          weather: WeatherAPI, satellite: SatelliteData) -> Dict:
        """Calcule Rothermel et retourne les résultats + features intermédiaires."""
        if not ROTHERMEL_AVAILABLE:
            return {}
        
        # Estimation des humidités depuis les features
        features = self.features_engineer.compute_all_features(robot, weather, satellite)
        
        # Conversion features → humidités Rothermel
        dfmc = features.get('dfmc', 10.0)
        m_1h = dfmc / 100.0
        m_10h = min(m_1h + 0.02, 0.60)
        m_100h = min(m_1h + 0.04, 0.60)
        
        # Humidité herbes vivantes (estimation depuis RH)
        rh = robot.rh
        m_live_herb = 0.30 + (100.0 - rh) / 200.0
        m_live_woody = 0.60 + (100.0 - rh) / 500.0
        
        # Vent à mi-flamme
        wind_mid = features.get('wind_mid_flame', weather.wind_10m * 0.4)
        
        try:
            predictor = BurnTrackRothermel(fuel_code)
            result = predictor.predict(
                temp_air=robot.temp_air,
                rh=robot.rh,
                wind_speed=wind_mid,
                slope_deg=robot.slope_deg,
                dead_1h_moisture=m_1h,
                dead_10h_moisture=m_10h,
                dead_100h_moisture=m_100h,
                live_herb_moisture=m_live_herb,
                live_woody_moisture=m_live_woody
            )
            return result
        except Exception as e:
            warnings.warn(f"Erreur Rothermel : {e}")
            return {}
    
    def _prepare_ia_vector(self, rothermel_out: Dict, 
                          features: Dict) -> Optional[np.ndarray]:
        """Prépare le vecteur de features pour l'IA correctrice."""
        if not IA_AVAILABLE:
            return None
        
        # Fusion des dicts
        combined = {**features, **rothermel_out}
        
        # Renommage pour correspondre à FEATURE_NAMES_IA
        mapping = {
            'ros_m_min': 'ros_base',
            'I_R_kW_m2': 'I_R',
            'phi_w': 'phi_w',
            'phi_s': 'phi_s',
            'beta': 'beta',
            'temp_air': 'temp_air',
            'rh': 'rh',
            'vpd': 'vpd',
            'dfmc': 'dfmc',
            'wind_speed': 'wind_speed',
            'ndvi': 'ndvi',
            'ndwi': 'ndwi',
            'lst': 'lst',
            'temp_2m': 'temp_2m',
            'wind_10m': 'wind_10m',
            'delta_t_surf_air': 'delta_t_surf_air',
            'wind_ratio': 'wind_ratio',
            'stress_index': 'stress_index',
            'ndvi_anomaly': 'ndvi_anomaly',
            'danger_proxy': 'danger_proxy'
        }
        
        normalized = {}
        for key, ia_name in mapping.items():
            if key in combined:
                normalized[ia_name] = combined[key]
            elif ia_name in combined:
                normalized[ia_name] = combined[ia_name]
            else:
                normalized[ia_name] = 0.0
        
        return prepare_features_for_ia(normalized, normalized)
    
    def run(self,
            lat: float,
            lon: float,
            date: Optional[datetime] = None,
            species: Optional[str] = None,
            ecosystem: Optional[str] = None,
            robot_data: Optional[Dict[str, float]] = None,
            fuel_height: float = 0.3) -> Dict[str, Any]:
        """
        Exécute le pipeline complet BurnTrack.
        
        Args:
            lat, lon: Coordonnées GPS
            date: Date/heure de la prédiction (default: now)
            species: Espèce reconnue par le modèle de vision
            ecosystem: Écosystème (alternative à species)
            robot_data: Données capteurs robot {temp_air, rh, wind_speed, ...}
            fuel_height: Hauteur du lit de combustible (m)
        
        Returns:
            Dict structuré avec :
            - 'status': 'success' | 'partial' | 'fallback'
            - 'location': {lat, lon, date}
            - 'data_source': 'gee' | 'fallback' | 'robot'
            - 'features': toutes les features calculées
            - 'fuel_model': code et nom du fuel model
            - 'rothermel': sortie Rothermel
            - 'ia_vector': vecteur prêt pour l'IA (ou None)
            - 'danger': niveau de danger global
        """
        if date is None:
            date = datetime.now()
        
        # ─── 1. TÉLÉCHARGEMENT DONNÉES ───
        geo_data = self._fetch_geospatial_data(lat, lon, date)
        data_source = "gee" if self.use_gee else "fallback"
        
        # Si données robot fournies, elles priment
        if robot_data:
            geo_data.update(robot_data)
            data_source = "robot"
        
        # ─── 2. FEATURES ENGINEERING ───
        robot, weather, satellite = self._build_inputs(geo_data, robot_data)
        
        if self.features_engineer:
            features = self.features_engineer.compute_all_features(
                robot, weather, satellite, fuel_height=fuel_height
            )
        else:
            features = {}
        
        # ─── 3. SÉLECTION FUEL MODEL ───
        fuel_code = self._select_fuel_model(species, ecosystem)
        fuel_name = fuel_code  # Simplifié, peut être enrichi
        
        # ─── 4. ROTHERMEL ───
        rothermel_result = self._compute_rothermel(
            fuel_code, robot, weather, satellite
        )
        
        # ─── 5. VECTEUR IA ───
        ia_vector = self._prepare_ia_vector(rothermel_result, features)
        
        # ─── 6. DANGER GLOBAL ───
        danger = rothermel_result.get('danger_level', 'INCONNU')
        if features.get('danger_proxy', 0) > 0.8:
            danger = "EXTRÊME"
        
        # ─── 7. ASSEMBLAGE ───
        result = {
            'status': 'success' if rothermel_result else 'partial',
            'location': {
                'lat': round(lat, 6),
                'lon': round(lon, 6),
                'date': date.isoformat(),
                'data_source': data_source
            },
            'fuel_model': {
                'code': fuel_code,
                'name': fuel_name,
                'selected_by': 'species' if species else ('ecosystem' if ecosystem else 'default')
            },
            'features': {k: round(v, 4) if isinstance(v, float) else v 
                        for k, v in features.items()},
            'rothermel': rothermel_result,
            'ia_input_vector': ia_vector.tolist() if ia_vector is not None else None,
            'danger_assessment': danger,
            'raw_geo_data': geo_data if data_source == 'fallback' else None
        }
        
        return result
    
    def run_batch(self,
                  locations: List[Tuple[float, float]],
                  date: Optional[datetime] = None,
                  **kwargs) -> List[Dict]:
        """Exécute le pipeline sur plusieurs points."""
        return [self.run(lat, lon, date, **kwargs) for lat, lon in locations]


# =============================================================================
# EXEMPLE D'UTILISATION
# =============================================================================

if __name__ == "__main__":
    print("=" * 70)
    print("BURNTRACK DATA PIPELINE - TEST D'INTÉGRATION")
    print("=" * 70)
    
    pipeline = BurnTrackPipeline(use_gee=False)  # Mode offline pour le test
    
    # Test 1 : Marrakech, steppe, été
    print("\n🧪 TEST 1 : Marrakech, Steppe, Été")
    print("-" * 50)
    
    result = pipeline.run(
        lat=31.63, lon=-7.98,
        date=datetime(2024, 8, 15, 14, 0),
        species="acacia",
        robot_data={
            "temp_air": 38.0,
            "rh": 20.0,
            "wind_speed": 4.0,
            "slope_deg": 15.0,
            "surface_temp": 52.0
        }
    )
    
    print(f"Statut        : {result['status']}")
    print(f"Source données: {result['location']['data_source']}")
    print(f"Fuel model    : {result['fuel_model']['code']}")
    print(f"Danger        : {result['danger_assessment']}")
    
    if result['rothermel']:
        r = result['rothermel']
        print(f"\nRothermel :")
        print(f"  ROS           : {r.get('ros_m_min', 'N/A')} m/min")
        print(f"  Flame length  : {r.get('flame_length_m', 'N/A')} m")
        print(f"  Intensity     : {r.get('fireline_intensity_kW_m', 'N/A')} kW/m")
        print(f"  Danger level  : {r.get('danger_level', 'N/A')}")
    
    print(f"\nFeatures clés :")
    f = result['features']
    print(f"  VPD           : {f.get('vpd', 'N/A')} kPa")
    print(f"  DFMC          : {f.get('dfmc', 'N/A')} %")
    print(f"  Stress index  : {f.get('stress_index', 'N/A')}")
    print(f"  Danger proxy  : {f.get('danger_proxy', 'N/A')}")
    
    if result['ia_input_vector']:
        print(f"\nVecteur IA    : {len(result['ia_input_vector'])} features")
        print(f"  Premières valeurs : {result['ia_input_vector'][:5]}")
    
    # Test 2 : Mode sans robot (fallback géospatial)
    print("\n" + "=" * 70)
    print("🧪 TEST 2 : Mode sans robot (fallback climatologique)")
    print("-" * 50)
    
    result2 = pipeline.run(
        lat=14.5, lon=-2.5,  # Sahel
        date=datetime(2024, 4, 10),
        ecosystem="sahel_herbeuse"
    )
    
    print(f"Statut        : {result2['status']}")
    print(f"Source données: {result2['location']['data_source']}")
    print(f"Fuel model    : {result2['fuel_model']['code']}")
    print(f"Temp air      : {result2['features'].get('temp_air', 'N/A')} °C")
    print(f"RH            : {result2['features'].get('rh', 'N/A')} %")
    
    # Test 3 : Batch
    print("\n" + "=" * 70)
    print("🧪 TEST 3 : Batch multi-points")
    print("-" * 50)
    
    points = [(31.63, -7.98), (33.5, -5.5), (35.0, -3.0)]
    batch = pipeline.run_batch(points, date=datetime(2024, 8, 15), species="acacia")
    
    for i, res in enumerate(batch):
        print(f"  Point {i+1} ({res['location']['lat']}, {res['location']['lon']}) : "
              f"ROS={res['rothermel'].get('ros_m_min', 'N/A') if res['rothermel'] else 'N/A'} m/min | "
              f"Danger={res['danger_assessment']}")
    
    print("\n" + "=" * 70)
    print("✅ Tests terminés. Pipeline prêt pour l'intégration.")
    print("=" * 70)
