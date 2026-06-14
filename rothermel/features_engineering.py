"""
features_engineering.py
========================
Extraction et calcul des features pour le pipeline BurnTrack.
Ce module calcule les variables dérivées à partir des capteurs robot,
météo et satellite.

Features calculées :
- VPD (Vapor Pressure Deficit)
- DFMC (Dead Fuel Moisture Content)
- delta_t_surf_air (différence température surface/air)
- wind_ratio (ratio vent robot / vent météo)
- stress_index (indice de stress végétal)
- ndvi_anomaly (anomalie NDVI)
- danger_proxy (proxy de danger)
"""

import numpy as np
from dataclasses import dataclass
from typing import Dict, Optional, Tuple
import math


@dataclass
class RobotSensors:
    """
    Données brutes des capteurs du robot.
    
    Attributes:
        temp_air: Température de l'air (°C)
        rh: Humidité relative (%)
        wind_speed: Vitesse du vent mesurée par le robot (m/s)
        slope_deg: Pente du terrain (degrés)
        aspect_deg: Orientation de la pente (degrés, optionnel)
        surface_temp: Température de surface (°C, optionnel, capteur IR)
        co_ppm: Concentration CO (ppm, optionnel, MQ-135)
        co2_ppm: Concentration CO2 (ppm, optionnel, MQ-135)
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
    Données météorologiques provenant d'une API externe.
    
    Attributes:
        temp_2m: Température à 2m (°C)
        wind_10m: Vitesse du vent à 10m (m/s)
        wind_gust: Rafales de vent (m/s, optionnel)
        precip_1h: Précipitations sur 1h (mm, optionnel)
        pressure: Pression atmosphérique (hPa, optionnel)
        dew_point: Point de rosée (°C, optionnel)
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
    Données satellite (Google Earth Engine ou autre).
    
    Attributes:
        ndvi: Indice de végétation [-1, 1]
        ndwi: Indice d'eau [-1, 1]
        lst: Land Surface Temperature (°C)
        evi: Enhanced Vegetation Index [-1, 1] (optionnel)
        savi: Soil Adjusted Vegetation Index [-1, 1] (optionnel)
        burned_area: Aire brûlée récente (m², optionnel)
    """
    ndvi: float
    ndwi: float
    lst: float
    evi: Optional[float] = None
    savi: Optional[float] = None
    burned_area: Optional[float] = None


class FeaturesEngineering:
    """
    Moteur de calcul des features dérivées pour l'IA correctrice.
    """
    
    def __init__(self):
        # Références pour calcul des anomalies (à calibrer avec données historiques)
        self.ndvi_reference_mean = 0.35  # NDVI moyen de référence
        self.ndvi_reference_std = 0.15   # Écart-type de référence
    
    # -------------------------------------------------------------------------
    # FEATURES DE BASE (calculées à partir des capteurs robot)
    # -------------------------------------------------------------------------
    
    def compute_vpd(self, temp_air: float, rh: float) -> float:
        """
        Calcule le déficit de pression de vapeur (VPD) en kPa.
        
        Formule : VPD = e_s × (1 - RH/100)
        où e_s = 0.6108 × exp(17.27 × T / (T + 237.3))
        
        Args:
            temp_air: Température de l'air (°C)
            rh: Humidité relative (%)
        
        Returns:
            VPD en kPa
        """
        es = 0.6108 * np.exp(17.27 * temp_air / (temp_air + 237.3))
        vpd = es * (1.0 - rh / 100.0)
        return max(0.0, vpd)  # VPD ne peut pas être négatif
    
    def compute_dfmc(self, temp_air: float, vpd: float) -> float:
        """
        Calcule le taux d'humidité des combustibles morts (DFMC) en %.
        
        Formule simplifiée basée sur VPD et température.
        Source : adaptation de Simard (1968) et Van Wagner (1977)
        
        Args:
            temp_air: Température de l'air (°C)
            vpd: VPD en kPa
        
        Returns:
            DFMC en %, borné entre 3 et 40%
        """
        dfmc = 30.0 - 2.5 * vpd - 0.1 * temp_air
        return float(np.clip(dfmc, 3.0, 40.0))
    
    def compute_dfmc_precip_adjusted(
        self, 
        dfmc_base: float, 
        precip_1h: Optional[float], 
        hours_since_rain: int = 0
    ) -> float:
        """
        Ajuste le DFMC en fonction des précipitations récentes.
        
        Args:
            dfmc_base: DFMC de base (%)
            precip_1h: Précipitations sur la dernière heure (mm)
            hours_since_rain: Heures écoulées depuis la dernière pluie
        
        Returns:
            DFMC ajusté (%)
        """
        if precip_1h is None or precip_1h <= 0:
            # Séchage progressif après la pluie
            drying_factor = 1.0 - np.exp(-0.1 * hours_since_rain)
            return min(dfmc_base + 10.0 * (1.0 - drying_factor), 40.0)
        
        # Augmentation immédiate avec la pluie
        moisture_increase = min(precip_1h * 5.0, 20.0)
        return min(dfmc_base + moisture_increase, 40.0)
    
    # -------------------------------------------------------------------------
    # FEATURES AVANCÉES (fusion multi-sources)
    # -------------------------------------------------------------------------
    
    def compute_delta_t_surf_air(
        self, 
        surface_temp: Optional[float], 
        temp_air: float
    ) -> float:
        """
        Calcule la différence température surface - température air.
        
        Un delta élevé indique un sol très chaud et sec (stress végétal).
        
        Args:
            surface_temp: Température de surface (°C), None si non disponible
            temp_air: Température de l'air (°C)
        
        Returns:
            Delta T en °C, 0 si surface_temp non disponible
        """
        if surface_temp is None:
            return 0.0
        return surface_temp - temp_air
    
    def compute_wind_ratio(
        self, 
        wind_robot: float, 
        wind_10m: float
    ) -> float:
        """
        Calcule le ratio vent robot / vent météo à 10m.
        
        Permet de détecter les effets locaux (vallée, versant, couvert).
        Valeur proche de 1 = conditions cohérentes.
        Valeur > 1 = accélération locale du vent.
        Valeur < 1 = abri local.
        
        Args:
            wind_robot: Vent mesuré par le robot (m/s)
            wind_10m: Vent météo à 10m (m/s)
        
        Returns:
            Ratio, borné entre 0.1 et 3.0
        """
        if wind_10m < 0.1:
            return 1.0
        ratio = wind_robot / wind_10m
        return float(np.clip(ratio, 0.1, 3.0))
    
    def compute_wind_mid_flame(self, wind_10m: float, fuel_height: float = 0.3) -> float:
        """
        Convertit le vent à 10m en vent à mi-flamme.
        
        Formule : U_mid = U_10m × k × ln((h_fuel/2 + z0) / z0) / ln((10 + z0) / z0)
        où k ≈ 0.4 (constante de von Kármán), z0 ≈ 0.01m (rugosité)
        
        Simplification : U_mid ≈ U_10m × 0.4 pour lit bas (< 0.6m)
        
        Args:
            wind_10m: Vent à 10m (m/s)
            fuel_height: Hauteur du lit de combustible (m)
        
        Returns:
            Vent à mi-flamme (m/s)
        """
        if fuel_height < 0.6:
            # Lit bas : réduction d'environ 60%
            return wind_10m * 0.4
        elif fuel_height < 2.0:
            # Lit moyen : réduction d'environ 40%
            return wind_10m * 0.6
        else:
            # Lit haut (forêt) : quasi plein vent
            return wind_10m * 0.8
    
    def compute_stress_index(
        self, 
        vpd: float, 
        ndvi: float, 
        lst: float, 
        temp_air: float
    ) -> float:
        """
        Calcule un indice de stress végétal combinant VPD, NDVI et LST.
        
        Formule : stress = (VPD / 5) × (1 - NDVI) × (LST - T_air) / 10
        Normalisé entre 0 et 1.
        
        Un stress élevé indique :
        - VPD élevé (air sec)
        - NDVI bas (végétation faible/stressée)
        - LST >> T_air (sur-chauffe du sol)
        
        Args:
            vpd: VPD en kPa
            ndvi: NDVI [-1, 1]
            lst: Land Surface Temperature (°C)
            temp_air: Température air (°C)
        
        Returns:
            Stress index [0, 1]
        """
        vpd_term = min(vpd / 5.0, 2.0)  # Normalisé, plafonné à 2
        ndvi_term = max(0.0, 1.0 - ndvi)  # Inversé : NDVI bas = stress élevé
        temp_term = max(0.0, (lst - temp_air) / 10.0)  # Diff normalisée
        
        stress = vpd_term * ndvi_term * temp_term
        return float(np.clip(stress, 0.0, 1.0))
    
    def compute_ndvi_anomaly(self, ndvi: float) -> float:
        """
        Calcule l'anomalie NDVI par rapport à la moyenne de référence.
        
        Anomalie = (NDVI - moyenne) / écart-type
        
        Anomalie < -1 : végétation anormalement faible (sécheresse, sur-pâturage)
        Anomalie > +1 : végétation anormalement dense
        
        Args:
            ndvi: NDVI actuel [-1, 1]
        
        Returns:
            Anomalie NDVI (z-score)
        """
        if self.ndvi_reference_std == 0:
            return 0.0
        return (ndvi - self.ndvi_reference_mean) / self.ndvi_reference_std
    
    def compute_danger_proxy(
        self, 
        vpd: float, 
        dfmc: float, 
        wind_speed: float, 
        ndvi: float
    ) -> float:
        """
        Calcule un proxy de danger avant même de lancer Rothermel.
        
        Formule empirique combinant les facteurs de risque principaux.
        Utile pour une alerte rapide ou fallback si Rothermel échoue.
        
        Args:
            vpd: VPD en kPa
            dfmc: DFMC en %
            wind_speed: Vent (m/s)
            ndvi: NDVI [-1, 1]
        
        Returns:
            Danger proxy [0, 1], 1 = danger maximal
        """
        # VPD : plus c'est élevé, plus c'est dangereux
        vpd_score = min(vpd / 5.0, 1.0)
        
        # DFMC : inverse (bas = dangereux)
        dfmc_score = max(0.0, 1.0 - dfmc / 25.0)
        
        # Vent : plus c'est élevé, plus c'est dangereux
        wind_score = min(wind_speed / 10.0, 1.0)
        
        # NDVI : inverse (bas = moins de biomasse mais aussi moins d'humidité)
        # On considère que NDVI très bas = risque modéré (herbes sèches)
        # NDVI moyen = risque élevé (combustible sec et abondant)
        # NDVI très haut = risque modéré (trop vert pour brûler vite)
        ndvi_score = 1.0 - abs(ndvi - 0.3) / 0.7
        ndvi_score = max(0.0, ndvi_score)
        
        # Combinaison pondérée
        danger = 0.3 * vpd_score + 0.3 * dfmc_score + 0.25 * wind_score + 0.15 * ndvi_score
        return float(np.clip(danger, 0.0, 1.0))
    
    # -------------------------------------------------------------------------
    # PIPELINE COMPLET
    # -------------------------------------------------------------------------
    
    def compute_all_features(
        self,
        robot: RobotSensors,
        weather: WeatherAPI,
        satellite: SatelliteData,
        fuel_height: float = 0.3
    ) -> Dict[str, float]:
        """
        Calcule toutes les features en une seule passe.
        
        Args:
            robot: Données capteurs robot
            weather: Données météo API
            satellite: Données satellite
            fuel_height: Hauteur du lit de combustible (m) pour conversion vent
        
        Returns:
            Dictionnaire de toutes les features calculées
        """
        # Features de base
        vpd = self.compute_vpd(robot.temp_air, robot.rh)
        dfmc = self.compute_dfmc(robot.temp_air, vpd)
        dfmc_adj = self.compute_dfmc_precip_adjusted(
            dfmc, weather.precip_1h, hours_since_rain=0
        )
        
        # Features avancées
        delta_t = self.compute_delta_t_surf_air(robot.surface_temp, robot.temp_air)
        wind_ratio = self.compute_wind_ratio(robot.wind_speed, weather.wind_10m)
        wind_mid = self.compute_wind_mid_flame(weather.wind_10m, fuel_height)
        stress = self.compute_stress_index(vpd, satellite.ndvi, satellite.lst, robot.temp_air)
        ndvi_anomaly = self.compute_ndvi_anomaly(satellite.ndvi)
        danger_proxy = self.compute_danger_proxy(vpd, dfmc, robot.wind_speed, satellite.ndvi)
        
        return {
            # Features de base
            'temp_air': robot.temp_air,
            'rh': robot.rh,
            'vpd': round(vpd, 3),
            'dfmc': round(dfmc, 3),
            'dfmc_adjusted': round(dfmc_adj, 3),
            'wind_speed': robot.wind_speed,
            'wind_mid_flame': round(wind_mid, 3),
            'slope_deg': robot.slope_deg,
            
            # Features météo
            'temp_2m': weather.temp_2m,
            'wind_10m': weather.wind_10m,
            'wind_gust': weather.wind_gust if weather.wind_gust else 0.0,
            'precip_1h': weather.precip_1h if weather.precip_1h else 0.0,
            'pressure': weather.pressure if weather.pressure else 1013.25,
            
            # Features satellite
            'ndvi': satellite.ndvi,
            'ndwi': satellite.ndwi,
            'lst': satellite.lst,
            'evi': satellite.evi if satellite.evi else satellite.ndvi,
            'savi': satellite.savi if satellite.savi else satellite.ndvi,
            
            # Features fusion
            'delta_t_surf_air': round(delta_t, 3),
            'wind_ratio': round(wind_ratio, 3),
            'stress_index': round(stress, 3),
            'ndvi_anomaly': round(ndvi_anomaly, 3),
            'danger_proxy': round(danger_proxy, 3),
        }


# ============================================================================
# EXEMPLE D'UTILISATION
# ============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("FEATURES ENGINEERING - TEST")
    print("=" * 60)
    
    # Instanciation
    fe = FeaturesEngineering()
    
    # Données de test (conditions estivales marocaines)
    robot = RobotSensors(
        temp_air=38.0,
        rh=20.0,
        wind_speed=4.0,
        slope_deg=15.0,
        surface_temp=52.0,
        co_ppm=0.5,
        co2_ppm=420.0
    )
    
    weather = WeatherAPI(
        temp_2m=35.0,
        wind_10m=5.0,
        wind_gust=8.0,
        precip_1h=0.0,
        pressure=1015.0
    )
    
    satellite = SatelliteData(
        ndvi=0.25,
        ndwi=-0.15,
        lst=48.0,
        evi=0.20,
        savi=0.22
    )
    
    # Calcul de toutes les features
    features = fe.compute_all_features(robot, weather, satellite, fuel_height=0.3)
    
    print("\n--- FEATURES CALCULÉES ---")
    for name, value in features.items():
        print(f"{name:25s}: {value}")
    
    print("\n--- INTERPRÉTATION ---")
    print(f"VPD = {features['vpd']} kPa  → {'Très sec' if features['vpd'] > 3 else 'Sec' if features['vpd'] > 1.5 else 'Modéré'}")
    print(f"DFMC = {features['dfmc']}%  → {'Extrêmement inflammable' if features['dfmc'] < 6 else 'Très inflammable' if features['dfmc'] < 10 else 'Inflammable' if features['dfmc'] < 15 else 'Modéré'}")
    print(f"Stress index = {features['stress_index']}  → {'Stress sévère' if features['stress_index'] > 0.7 else 'Stress modéré' if features['stress_index'] > 0.3 else 'Normal'}")
    print(f"Danger proxy = {features['danger_proxy']}  → {'EXTRÊME' if features['danger_proxy'] > 0.8 else 'ÉLEVÉ' if features['danger_proxy'] > 0.6 else 'MODÉRÉ' if features['danger_proxy'] > 0.4 else 'FAIBLE'}")