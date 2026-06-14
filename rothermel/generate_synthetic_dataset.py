"""
generate_synthetic_dataset.py
=============================
Génère un dataset synthétique pour entraîner le corrector v3.

Données de base (ce que TU peux avoir) :
- Capteurs robot : temp_air, rh, wind_speed, slope_deg
- Météo API : temp_2m, wind_10m, wind_gust, precip_1h
- Satellite : ndvi, ndwi, lst
- Fuel model : depuis fuel_models.py (vision ou mapping écosystème)

Le dataset simule un BIAIS CONTRÔLÉ entre ROS Rothermel et ROS "terrain"
pour que le corrector apprenne à corriger.

Biais simulés réalistes :
1. Vent mal mesuré (±20% d'erreur sur anémomètre)
2. Humidité sous-estimée (capteur robot exposé au soleil)
3. Fuel model mal identifié (vision → mauvais mapping)
4. Effet de canyon/vent local non capturé par Rothermel
5. Pente micro-topographique ≠ pente macro (SRTM)
"""

import numpy as np
import pandas as pd
import sys
import os
import warnings
warnings.filterwarnings('ignore')

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from fuel_models import ALL_FUEL_MODELS, get_fuel_model
from rothermel_engine_v3 import (
    RothermelEngine, FuelModel as EngineFuelModel, 
    MoistureInputs, EnvironmentalConditions
)


# =============================================================================
# CONFIGURATION
# =============================================================================

np.random.seed(42)

# Nombre d'échantillons par fuel model
N_SAMPLES_PER_FUEL = 200

# Biais réalistes (erreurs de terrain)
BIAS_CONFIG = {
    'wind_error_std': 0.15,        # Erreur relative sur vent (15%)
    'moisture_bias': -0.03,        # Biais humidité (sous-estimation)
    'moisture_error_std': 0.05,    # Erreur std humidité
    'slope_error_std': 0.10,       # Erreur relative pente (10%)
    'fuel_loading_error': 0.10,    # Erreur relative charge (10%)
    'local_wind_effect': 0.20,     # Effet local vent non modélisé (±20%)
}


def convert_fuel_to_engine(fuel) -> EngineFuelModel:
    """Convertit un FuelModel de fuel_models.py en FuelModel du moteur."""
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


def compute_vpd(temp_c: float, rh_percent: float) -> float:
    """Déficit de pression de vapeur en kPa."""
    es = 0.6108 * np.exp(17.27 * temp_c / (temp_c + 237.3))
    vpd = es * (1.0 - rh_percent / 100.0)
    return max(0.0, vpd)


def compute_dfmc(temp_c: float, vpd: float) -> float:
    """Dead Fuel Moisture Content en %."""
    dfmc = 30.0 - 2.5 * vpd - 0.1 * temp_c
    return float(np.clip(dfmc, 3.0, 40.0))


def compute_wind_mid_flame(wind_10m: float, fuel_height: float = 0.3) -> float:
    """Vent à mi-flamme depuis vent à 10m."""
    if fuel_height < 0.6:
        return wind_10m * 0.4
    elif fuel_height < 2.0:
        return wind_10m * 0.6
    return wind_10m * 0.8


def generate_sample_for_fuel(fuel_code: str, engine: RothermelEngine) -> dict:
    """
    Génère un échantillon synthétique pour un fuel model donné.
    Retourne un dict avec toutes les features pour le corrector v3.
    """
    fuel = get_fuel_model(fuel_code)
    if fuel is None:
        return None

    fuel_engine = convert_fuel_to_engine(fuel)

    # --- Conditions météo aléatoires réalistes ---
    temp_c = np.random.uniform(5.0, 45.0)           # °C
    rh_percent = np.random.uniform(10.0, 95.0)       # %
    wind_10m = np.random.uniform(0.5, 15.0)        # m/s
    slope_deg = np.random.uniform(0.0, 30.0)        # degrés
    slope_pct = np.tan(np.radians(slope_deg)) * 100.0  # %

    # Humidités des combustibles
    m_1h = np.random.uniform(0.03, 0.25)
    m_10h = m_1h + np.random.uniform(-0.02, 0.05)
    m_10h = np.clip(m_10h, 0.03, 0.30)
    m_100h = m_10h + np.random.uniform(-0.02, 0.05)
    m_100h = np.clip(m_100h, 0.05, 0.35)

    m_live_herb = np.random.uniform(0.5, 1.5)
    m_live_woody = np.random.uniform(0.6, 2.0)

    # Angle vent/pente
    angle = np.random.choice([0, 45, 90, 135, 180])

    # --- Calcul Rothermel (baseline) ---
    moisture = MoistureInputs(
        m_1h=m_1h, m_10h=m_10h, m_100h=m_100h,
        m_live_herb=m_live_herb, m_live_woody=m_live_woody
    )

    wind_mid = compute_wind_mid_flame(wind_10m, fuel.delta)
    conditions = EnvironmentalConditions(
        wind_speed=wind_mid,
        slope_pct=slope_pct,
        angle_wind_slope=angle
    )

    output = engine.compute(fuel_engine, moisture, conditions)
    ros_rothermel = output.ros

    # Si ROS nul ou aberrant, rejeter
    if ros_rothermel <= 0 or ros_rothermel > 50:
        return None

    # --- BIAIS CONTRÔLÉ : simuler ROS "terrain" avec erreurs ---
    # Le ROS terrain est le ROS Rothermel + erreurs systématiques

    # 1. Erreur de vent (anémomètre mal calibré)
    wind_error = np.random.normal(1.0, BIAS_CONFIG['wind_error_std'])
    wind_error = np.clip(wind_error, 0.5, 1.5)

    # 2. Biais humidité (capteur exposé au soleil → sous-estime)
    moisture_bias = BIAS_CONFIG['moisture_bias'] + np.random.normal(0, BIAS_CONFIG['moisture_error_std'])

    # 3. Effet local vent (canyon, versant)
    local_effect = np.random.normal(1.0, BIAS_CONFIG['local_wind_effect'])
    local_effect = np.clip(local_effect, 0.5, 1.5)

    # ROS "terrain" = ROS Rothermel avec biais
    # Formule empirique : ROS_terrain = ROS_Roth × (vent_effect × local_effect) / (1 + moisture_bias_effect)
    vent_effect = wind_error ** 0.5  # Exposant réduit car phi_w n'est pas linéaire
    moisture_effect = 1.0 + 2.0 * max(0, -moisture_bias)  # Sous-estimation humidité = ROS plus élevé

    ros_terrain = ros_rothermel * vent_effect * local_effect * moisture_effect

    # Ajout d'un bruit aléatoire faible
    ros_terrain *= np.random.normal(1.0, 0.05)
    ros_terrain = max(0.1, ros_terrain)

    # Delta = correction additive que le corrector doit apprendre
    delta_ros = ros_terrain - ros_rothermel

    # --- Features satellite (simulées réalistes) ---
    # NDVI corrélé négativement avec température et stress
    ndvi = np.clip(np.random.normal(0.35 - 0.005 * temp_c, 0.1), -0.2, 0.8)
    ndwi = np.clip(np.random.normal(-0.1 - 0.003 * temp_c, 0.15), -0.6, 0.4)
    lst_c = temp_c + np.random.uniform(5.0, 20.0)  # Surface plus chaude que l'air

    # --- Features calculées ---
    vpd = compute_vpd(temp_c, rh_percent)
    dfmc = compute_dfmc(temp_c, vpd)

    # --- Construction du dict de sortie ---
    sample = {
        # Target
        'delta_ros': delta_ros,
        'ros_rothermel': ros_rothermel,
        'ros_terrain': ros_terrain,

        # Features météo/capteurs
        'temp_c': temp_c,
        'rh_percent': rh_percent,
        'wind_speed_ms': wind_mid,
        'wind_10m': wind_10m,
        'vpd_kpa': vpd,
        'slope_deg': slope_deg,
        'slope_pct': slope_pct,
        'angle_wind_slope': angle,

        # Features fuel
        'fuel_model_code': fuel_code,
        'w_total_kg_m2': fuel.w_total,
        'w_dead_kg_m2': fuel.w_dead,
        'w_live_kg_m2': fuel.w_live,
        'delta_m': fuel.delta,
        'sigma_m2_m3': output.__dict__.get('sigma', 0),  # SAV pondérée
        'mx_percent': fuel.mx,
        'h_dead_kj_kg': fuel.h_dead,

        # Features moteur Rothermel
        'phi_w': output.phi_w,
        'phi_s': output.phi_s,
        'phi_eff': output.phi_eff,
        'beta': output.beta,
        'beta_opt': output.beta_opt,
        'beta_ratio': output.beta / output.beta_opt if output.beta_opt > 0 else 0,
        'gamma': output.gamma,
        'eta_M': output.eta_M,
        'eta_S': output.eta_S,
        'I_R_kW_m2': output.reaction_intensity,
        'xi': output.xi,
        'tau_min': output.tau,

        # Features satellite
        'ndvi': ndvi,
        'ndwi': ndwi,
        'lst_c': lst_c,
        'dfmc_percent': dfmc,

        # Humidités (pour info)
        'm_1h': m_1h,
        'm_10h': m_10h,
        'm_100h': m_100h,
        'm_live_herb': m_live_herb,
        'm_live_woody': m_live_woody,
    }

    return sample


def generate_dataset(n_samples_per_fuel: int = N_SAMPLES_PER_FUEL,
                     output_path: str = 'synthetic_dataset.csv') -> pd.DataFrame:
    """
    Génère le dataset synthétique complet.

    Args:
        n_samples_per_fuel : nombre d'échantillons par fuel model
        output_path : chemin du fichier CSV de sortie

    Returns:
        DataFrame pandas avec toutes les features
    """
    engine = RothermelEngine()
    fuel_codes = list(ALL_FUEL_MODELS.keys())

    all_samples = []
    rejected = 0

    print(f"Génération de {n_samples_per_fuel} échantillons × {len(fuel_codes)} fuel models...")
    print(f"Total cible : {n_samples_per_fuel * len(fuel_codes):,}")
    print()

    for i, fuel_code in enumerate(fuel_codes):
        fuel_samples = []
        attempts = 0
        max_attempts = n_samples_per_fuel * 5

        while len(fuel_samples) < n_samples_per_fuel and attempts < max_attempts:
            sample = generate_sample_for_fuel(fuel_code, engine)
            attempts += 1
            if sample is not None:
                fuel_samples.append(sample)
            else:
                rejected += 1

        all_samples.extend(fuel_samples)

        if (i + 1) % 10 == 0 or i == len(fuel_codes) - 1:
            print(f"  [{i+1}/{len(fuel_codes)}] {fuel_code} : {len(fuel_samples)}/{n_samples_per_fuel} OK")

    df = pd.DataFrame(all_samples)

    # Sauvegarde
    df.to_csv(output_path, index=False)

    print()
    print("=" * 60)
    print("DATASET GÉNÉRÉ")
    print("=" * 60)
    print(f"Échantillons valides : {len(df):,}")
    print(f"Échantillons rejetés : {rejected:,}")
    print(f"Fuel models couverts : {df['fuel_model_code'].nunique()}")
    print(f"Fichier sauvegardé : {output_path}")
    print()
    print("--- STATISTIQUES TARGET (delta_ros) ---")
    print(f"  Moyenne : {df['delta_ros'].mean():.3f} m/min")
    print(f"  Médiane : {df['delta_ros'].median():.3f} m/min")
    print(f"  Std     : {df['delta_ros'].std():.3f} m/min")
    print(f"  Min     : {df['delta_ros'].min():.3f} m/min")
    print(f"  Max     : {df['delta_ros'].max():.3f} m/min")
    print()
    print("--- STATISTIQUES ROS ---")
    print(f"  ROS Rothermel moyen : {df['ros_rothermel'].mean():.2f} m/min")
    print(f"  ROS terrain moyen   : {df['ros_terrain'].mean():.2f} m/min")
    print(f"  Biais moyen         : {(df['ros_terrain'] / df['ros_rothermel']).mean():.3f}×")
    print()
    print("--- RÉPARTITION PAR FUEL MODEL ---")
    print(df['fuel_model_code'].value_counts().head(10).to_string())

    return df


# =============================================================================
# SPLIT TRAIN / VAL / TEST
# =============================================================================

def split_dataset(df: pd.DataFrame, 
                  train_ratio: float = 0.7,
                  val_ratio: float = 0.15,
                  output_dir: str = '.') -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Split stratifié par fuel model pour éviter le data leakage.
    """
    from sklearn.model_selection import train_test_split

    # Stratification par fuel model
    train_list, val_list, test_list = [], [], []

    for fuel_code in df['fuel_model_code'].unique():
        fuel_df = df[df['fuel_model_code'] == fuel_code]

        if len(fuel_df) < 10:
            # Trop peu d'échantillons → tout dans train
            train_list.append(fuel_df)
            continue

        # Split 70/15/15
        train_fuel, temp_fuel = train_test_split(fuel_df, train_size=train_ratio, random_state=42)
        val_fuel, test_fuel = train_test_split(temp_fuel, train_size=val_ratio/(1-train_ratio), random_state=42)

        train_list.append(train_fuel)
        val_list.append(val_fuel)
        test_list.append(test_fuel)

    train_df = pd.concat(train_list).reset_index(drop=True)
    val_df = pd.concat(val_list).reset_index(drop=True)
    test_df = pd.concat(test_list).reset_index(drop=True)

    # Sauvegarde
    train_df.to_csv(f'{output_dir}/train.csv', index=False)
    val_df.to_csv(f'{output_dir}/val.csv', index=False)
    test_df.to_csv(f'{output_dir}/test.csv', index=False)

    print()
    print("=" * 60)
    print("SPLIT EFFECTUÉ")
    print("=" * 60)
    print(f"Train : {len(train_df):,} ({len(train_df)/len(df)*100:.1f}%)")
    print(f"Val   : {len(val_df):,} ({len(val_df)/len(df)*100:.1f}%)")
    print(f"Test  : {len(test_df):,} ({len(test_df)/len(df)*100:.1f}%)")
    print()
    print(f"Fichiers sauvegardés dans {output_dir}/")

    return train_df, val_df, test_df


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    print("=" * 70)
    print("GENERATEUR DATASET SYNTHÉTIQUE — BurnTrack Corrector v3")
    print("=" * 70)
    print()

    # Génération
    df = generate_dataset(
        n_samples_per_fuel=N_SAMPLES_PER_FUEL,
        output_path='synthetic_dataset.csv'
    )

    # Split
    train_df, val_df, test_df = split_dataset(df, output_dir='.')

    print()
    print("✅ Dataset prêt pour l'entraînement !")
    print()
    print("Prochaines étapes :")
    print("  1. Remplace synthetic_dataset.csv par tes données terrain quand tu les as")
    print("  2. Lance train_corrector_v3.py pour entraîner le modèle")
    print("  3. Évalue sur test.csv")