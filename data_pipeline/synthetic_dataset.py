"""
synthetic_dataset.py
====================
Générateur de données synthétiques calibrées pour l'entraînement de l'IA
correctrice BurnTrack (Option C — Hybride).

Principe :
- Génère des scénarios réalistes avec des distributions météo calibrées sur
  l'Afrique (Sahel, Afrique du Sud, Madagascar, Fynbos)
- Calcule ROS_base via Rothermel (v2 corrigé)
- Applique des biais documentés dans la littérature = ROS_observé
- Ajoute un bruit résiduel faible (10-15%)

Sources de calibration :
- Govender et al. (2006) — Afrique du Sud (Miombo) : ROS_obs < ROS_Rothermel
- Frost & Robertson (1987) — Afrique du Sud (Fynbos) : ROS_obs > ROS_Rothermel
- Cruz et al. (2015) — Herbes sèches : surestimation ROS
- Savadogo et al. (2014) — Burkina Faso : sous-estimation ROS
- Andrews (2018) — Vent fort : surestimation ROS

Structure du dataset :
    X (features) : [ros_rothermel, temp_c, rh_percent, wind_speed, vpd_kpa,
                    slope_deg, fuel_model_encoded, fuel_moisture]
    y (target)   : correction_factor = ROS_observé / ROS_Rothermel
"""

import numpy as np
import pandas as pd
import json
import warnings
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass

# Suppression des warnings pour la génération
warnings.filterwarnings('ignore')


# =============================================================================
# CONFIGURATION
# =============================================================================

@dataclass
class SyntheticConfig:
    """Configuration du générateur de données synthétiques."""
    n_samples: int = 5000
    random_seed: int = 42
    noise_level: float = 0.12  # Bruit résiduel (12%)

    # Bornes empiriques du facteur de correction
    correction_min: float = 0.3
    correction_max: float = 3.0

    # Output
    output_path: str = 'synthetic_dataset.csv'
    config_path: str = 'synthetic_config.json'


# =============================================================================
# DISTRIBUTIONS MÉTÉO AFRIQUE (calibrées sur ERA5-Land + littérature)
# =============================================================================

AFRICA_CLIMATE_ZONES = {
    'sahel': {
        'temp_mean': 35.0, 'temp_std': 5.0,      # °C
        'rh_mean': 20.0, 'rh_std': 10.0,          # %
        'wind_mean': 4.0, 'wind_std': 2.5,         # m/s
        'slope_mean': 3.0, 'slope_std': 4.0,       # degrés
        'vpd_bias': 2.5,                           # kPa (fort VPD)
        'fuel_models': ['AF_SAHEL_GRASS', 'GR1', 'AF_GRASSLAND_FERTILE'],
        'bias_profile': 'dry_grass_over',            # Surestimation Rothermel
    },
    'south_africa_fynbos': {
        'temp_mean': 25.0, 'temp_std': 6.0,
        'rh_mean': 45.0, 'rh_std': 15.0,
        'wind_mean': 5.5, 'wind_std': 3.0,
        'slope_mean': 15.0, 'slope_std': 10.0,
        'vpd_bias': 1.5,
        'fuel_models': ['AF_FYNBOS'],
        'bias_profile': 'fynbos_under',            # Sous-estimation Rothermel
    },
    'south_africa_miombo': {
        'temp_mean': 28.0, 'temp_std': 5.0,
        'rh_mean': 55.0, 'rh_std': 12.0,
        'wind_mean': 3.0, 'wind_std': 1.5,
        'slope_mean': 5.0, 'slope_std': 6.0,
        'vpd_bias': 1.2,
        'fuel_models': ['AF_MIOMBO'],
        'bias_profile': 'miombo_under',            # Sous-estimation Rothermel
    },
    'madagascar': {
        'temp_mean': 30.0, 'temp_std': 4.0,
        'rh_mean': 65.0, 'rh_std': 15.0,
        'wind_mean': 3.5, 'wind_std': 2.0,
        'slope_mean': 10.0, 'slope_std': 12.0,
        'vpd_bias': 1.0,
        'fuel_models': ['AF_STEPPE', 'AF_GRASSLAND_FERTILE'],
        'bias_profile': 'mixed',
    },
    'burkina': {
        'temp_mean': 33.0, 'temp_std': 4.0,
        'rh_mean': 35.0, 'rh_std': 15.0,
        'wind_mean': 3.0, 'wind_std': 1.5,
        'slope_mean': 2.0, 'slope_std': 3.0,
        'vpd_bias': 2.0,
        'fuel_models': ['AF_SAHEL_GRASS', 'GR1'],
        'bias_profile': 'savanna_under',           # Sous-estimation Rothermel
    },
}


# =============================================================================
# BIAIS DOCUMENTÉS (littérature scientifique)
# =============================================================================

BIAS_PROFILES = {
    'dry_grass_over': {
        # Cruz et al. 2015 : Rothermel surestime ROS en herbes sèches (VPD > 3 kPa)
        # Biais moyen : +15-30% → correction_factor = 0.70-0.85
        'base_factor': 0.78,
        'factor_std': 0.08,
        'conditions': {'vpd_threshold': 3.0, 'rh_max': 25},
    },
    'fynbos_under': {
        # Frost & Robertson 1987 : Rothermel sous-estime en Fynbos
        # Biais moyen : -20-30% → correction_factor = 1.20-1.30
        'base_factor': 1.25,
        'factor_std': 0.06,
        'conditions': {'slope_min': 10},
    },
    'miombo_under': {
        # Govender et al. 2006 : Rothermel sous-estime en Miombo (bois dur)
        # Biais moyen : -5-10% → correction_factor = 1.05-1.10
        'base_factor': 1.07,
        'factor_std': 0.03,
        'conditions': {'rh_min': 40},
    },
    'savanna_under': {
        # Savadogo et al. 2014 : Rothermel sous-estime en savane africaine
        # Biais moyen : -10-20% → correction_factor = 1.10-1.20
        'base_factor': 1.15,
        'factor_std': 0.05,
        'conditions': {'temp_min': 30},
    },
    'mixed': {
        # Conditions mixtes : biais faible
        'base_factor': 1.0,
        'factor_std': 0.10,
        'conditions': {},
    },
    'wind_over': {
        # Andrews 2018 : Rothermel surestime avec vent fort (> 6 m/s)
        # Biais moyen : +10-20% → correction_factor = 0.80-0.90
        'base_factor': 0.85,
        'factor_std': 0.05,
        'conditions': {'wind_min': 6.0},
    },
    'humid_under': {
        # Rothermel 1972 : sous-estime en humidité élevée (RH > 70%)
        # Biais moyen : -10-15% → correction_factor = 1.10-1.15
        'base_factor': 1.12,
        'factor_std': 0.03,
        'conditions': {'rh_min': 70},
    },
}


# =============================================================================
# ENCODAGE DES FUEL MODELS
# =============================================================================

FUEL_MODEL_ENCODING = {
    'AF_STEPPE': 1.0,
    'AF_MIOMBO': 2.0,
    'AF_FYNBOS': 3.0,
    'AF_SAHEL_GRASS': 4.0,
    'GR1': 5.0,
    'AF_GRASSLAND_FERTILE': 6.0,
}

FUEL_MODEL_INV = {v: k for k, v in FUEL_MODEL_ENCODING.items()}


# =============================================================================
# UTILITAIRES
# =============================================================================

def compute_vpd(temp_c: float, rh_percent: float) -> float:
    """Calcule le déficit de pression de vapeur (kPa)."""
    es = 0.6108 * np.exp(17.27 * temp_c / (temp_c + 237.3))
    ea = es * (rh_percent / 100.0)
    return max(0.0, es - ea)


def estimate_fuel_moisture(rh_percent: float, temp_c: float) -> float:
    """Estime l'humidité du combustible (%)."""
    # Similaire à l'estimation dans BurnTrackRothermel
    es = 0.6108 * np.exp(17.27 * temp_c / (temp_c + 237.3))
    vpd = es * (1.0 - rh_percent / 100.0)
    dfmc = np.clip(30.0 - 2.5 * vpd - 0.1 * temp_c, 3.0, 40.0)
    return dfmc


def apply_bias_modifiers(
    base_factor: float,
    temp_c: float,
    rh_percent: float,
    wind_speed: float,
    vpd_kpa: float,
    slope_deg: float,
) -> float:
    """
    Applique des modificateurs de biais selon les conditions.

    Modificateurs documentés :
    - VPD > 3 kPa : surestimation Rothermel → réduction factor
    - Vent > 6 m/s : surestimation Rothermel → réduction factor
    - RH > 70% : sous-estimation Rothermel → augmentation factor
    - Pente > 25° : sous-estimation Rothermel → augmentation factor
    """
    factor = base_factor

    # Modificateur VPD (Cruz 2015)
    if vpd_kpa > 3.0:
        factor *= np.random.uniform(0.92, 0.98)
    elif vpd_kpa > 2.0:
        factor *= np.random.uniform(0.95, 1.0)

    # Modificateur vent (Andrews 2018)
    if wind_speed > 6.0:
        factor *= np.random.uniform(0.90, 0.96)
    elif wind_speed > 4.0:
        factor *= np.random.uniform(0.95, 1.0)

    # Modificateur humidité (Rothermel 1972)
    if rh_percent > 70.0:
        factor *= np.random.uniform(1.05, 1.12)
    elif rh_percent > 50.0:
        factor *= np.random.uniform(1.0, 1.05)

    # Modificateur pente (Alexander 1985)
    if slope_deg > 25.0:
        factor *= np.random.uniform(1.05, 1.15)
    elif slope_deg > 15.0:
        factor *= np.random.uniform(1.0, 1.08)

    return factor


# =============================================================================
# GÉNÉRATEUR PRINCIPAL
# =============================================================================

class SyntheticDatasetGenerator:
    """Générateur de dataset synthétique calibré sur la littérature Afrique."""

    def __init__(self, config: Optional[SyntheticConfig] = None):
        self.config = config or SyntheticConfig()
        np.random.seed(self.config.random_seed)

        # Import différé pour éviter dépendance circulaire
        try:
            from rothermel_engine_v2 import BurnTrackRothermel
            self.rothermel_available = True
        except ImportError:
            try:
                from rothermel_engine import BurnTrackRothermel
                self.rothermel_available = True
            except ImportError:
                self.rothermel_available = False
                warnings.warn(
                    "RothermelEngine non disponible. "
                    "Le dataset sera généré avec ROS estimé (moins précis)."
                )

    def _sample_zone(self) -> Tuple[str, Dict]:
        """Tire aléatoirement une zone climatique d'Afrique."""
        zones = list(AFRICA_CLIMATE_ZONES.keys())
        weights = [3, 2, 2, 2, 2]  # Poids : Sahel plus fréquent
        zone = np.random.choice(zones, p=np.array(weights)/sum(weights))
        return zone, AFRICA_CLIMATE_ZONES[zone]

    def _generate_meteo(self, zone_params: Dict) -> Dict:
        """Génère des conditions météo réalistes pour la zone."""
        temp_c = np.random.normal(zone_params['temp_mean'], zone_params['temp_std'])
        temp_c = np.clip(temp_c, 15.0, 48.0)

        rh_percent = np.random.normal(zone_params['rh_mean'], zone_params['rh_std'])
        rh_percent = np.clip(rh_percent, 5.0, 95.0)

        wind_speed = np.random.normal(zone_params['wind_mean'], zone_params['wind_std'])
        wind_speed = max(0.0, wind_speed)

        slope_deg = np.random.normal(zone_params['slope_mean'], zone_params['slope_std'])
        slope_deg = np.clip(slope_deg, 0.0, 45.0)

        vpd_kpa = compute_vpd(temp_c, rh_percent)

        fuel_moisture = estimate_fuel_moisture(rh_percent, temp_c)

        return {
            'temp_c': temp_c,
            'rh_percent': rh_percent,
            'wind_speed': wind_speed,
            'slope_deg': slope_deg,
            'vpd_kpa': vpd_kpa,
            'fuel_moisture': fuel_moisture,
        }

    def _compute_ros_rothermel(
        self, fuel_model: str, meteo: Dict
    ) -> float:
        """Calcule ROS via Rothermel ou estimation."""
        if self.rothermel_available:
            try:
                from rothermel_engine_v2 import BurnTrackRothermel
                predictor = BurnTrackRothermel(fuel_model)
                result = predictor.predict(
                    temp_air=meteo['temp_c'],
                    rh=meteo['rh_percent'],
                    wind_speed=meteo['wind_speed'],
                    slope_deg=meteo['slope_deg']
                )
                return result['ros_m_min']
            except Exception:
                pass

        # Estimation fallback si Rothermel non disponible
        # ROS estimé = f(temp, RH, vent, pente) — approximation empirique
        base_ros = 0.5 + 0.05 * meteo['temp_c'] - 0.01 * meteo['rh_percent']
        base_ros += 0.3 * meteo['wind_speed'] + 0.02 * meteo['slope_deg']
        base_ros = max(0.1, base_ros)
        return base_ros

    def _apply_bias(self, zone: str, meteo: Dict, ros_rothermel: float) -> float:
        """Applique le biais documenté pour obtenir ROS observé."""
        zone_params = AFRICA_CLIMATE_ZONES[zone]
        bias_profile = zone_params['bias_profile']
        bias_config = BIAS_PROFILES[bias_profile]

        # Facteur de base
        base_factor = bias_config['base_factor']

        # Modificateurs selon conditions
        factor = apply_bias_modifiers(
            base_factor,
            meteo['temp_c'],
            meteo['rh_percent'],
            meteo['wind_speed'],
            meteo['vpd_kpa'],
            meteo['slope_deg'],
        )

        # Bruit résiduel (12%)
        noise = np.random.normal(1.0, self.config.noise_level)
        factor *= noise

        # Borner dans les limites empiriques
        factor = np.clip(
            factor,
            self.config.correction_min,
            self.config.correction_max
        )

        return ros_rothermel * factor

    def generate_sample(self) -> Dict:
        """Génère un échantillon unique."""
        zone, zone_params = self._sample_zone()
        fuel_model = np.random.choice(zone_params['fuel_models'])

        meteo = self._generate_meteo(zone_params)
        ros_rothermel = self._compute_ros_rothermel(fuel_model, meteo)
        ros_observed = self._apply_bias(zone, meteo, ros_rothermel)

        # Facteur de correction = cible pour l'IA
        correction_factor = ros_observed / max(ros_rothermel, 0.01)
        correction_factor = np.clip(
            correction_factor,
            self.config.correction_min,
            self.config.correction_max
        )

        return {
            'ros_rothermel': ros_rothermel,
            'temp_c': meteo['temp_c'],
            'rh_percent': meteo['rh_percent'],
            'wind_speed': meteo['wind_speed'],
            'vpd_kpa': meteo['vpd_kpa'],
            'slope_deg': meteo['slope_deg'],
            'fuel_model_encoded': FUEL_MODEL_ENCODING[fuel_model],
            'fuel_moisture': meteo['fuel_moisture'],
            'correction_factor': correction_factor,
            'ros_observed': ros_observed,
            'zone': zone,
            'fuel_model': fuel_model,
        }

    def generate(self, n_samples: Optional[int] = None) -> pd.DataFrame:
        """Génère le dataset complet."""
        n = n_samples or self.config.n_samples

        print(f"Génération de {n} échantillons synthétiques calibrés...")
        print(f"Zones climatiques : {list(AFRICA_CLIMATE_ZONES.keys())}")
        print(f"Bruit résiduel : {self.config.noise_level*100:.0f}%")
        print(f"Rothermel disponible : {self.rothermel_available}")
        print()

        samples = []
        for i in range(n):
            if (i + 1) % 500 == 0:
                print(f"  Progression : {i+1}/{n} ({(i+1)/n*100:.0f}%)")
            samples.append(self.generate_sample())

        df = pd.DataFrame(samples)

        # Sauvegarde
        df.to_csv(self.config.output_path, index=False)

        # Sauvegarde config
        config_dict = {
            'n_samples': n,
            'random_seed': self.config.random_seed,
            'noise_level': self.config.noise_level,
            'correction_bounds': [self.config.correction_min, self.config.correction_max],
            'zones': list(AFRICA_CLIMATE_ZONES.keys()),
            'bias_profiles': list(BIAS_PROFILES.keys()),
            'fuel_models': list(FUEL_MODEL_ENCODING.keys()),
            'rothermel_available': self.rothermel_available,
        }
        with open(self.config.config_path, 'w') as f:
            json.dump(config_dict, f, indent=2)

        print(f"\n✅ Dataset sauvegardé : {self.config.output_path}")
        print(f"   Config sauvegardée : {self.config.config_path}")

        # Statistiques
        print(f"\n📊 Statistiques du dataset :")
        print(f"   ROS Rothermel : {df['ros_rothermel'].mean():.3f} ± {df['ros_rothermel'].std():.3f} m/min")
        print(f"   ROS Observé   : {df['ros_observed'].mean():.3f} ± {df['ros_observed'].std():.3f} m/min")
        print(f"   Facteur corr. : {df['correction_factor'].mean():.3f} ± {df['correction_factor'].std():.3f}")
        print(f"   Facteur [min,max] : [{df['correction_factor'].min():.3f}, {df['correction_factor'].max():.3f}]")

        print(f"\n📊 Distribution par zone :")
        print(df['zone'].value_counts())

        print(f"\n📊 Distribution par fuel model :")
        print(df['fuel_model'].value_counts())

        return df


# =============================================================================
# POINT D'ENTRÉE
# =============================================================================

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Générateur de dataset synthétique BurnTrack")
    parser.add_argument("--n-samples", type=int, default=5000, help="Nombre d'échantillons")
    parser.add_argument("--seed", type=int, default=42, help="Graine aléatoire")
    parser.add_argument("--noise", type=float, default=0.12, help="Niveau de bruit (0-1)")
    parser.add_argument("--output", type=str, default="synthetic_dataset.csv", help="Fichier de sortie")

    args = parser.parse_args()

    config = SyntheticConfig(
        n_samples=args.n_samples,
        random_seed=args.seed,
        noise_level=args.noise,
        output_path=args.output,
    )

    generator = SyntheticDatasetGenerator(config)
    df = generator.generate()
