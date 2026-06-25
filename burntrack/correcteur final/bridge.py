"""
Pont BurnTrack : Rothermel Engine v3 ↔ MLP Corrector
Intègre le moteur physique avec le correcteur IA

Usage:
    from bridge import BurnTrackPredictor

    predictor = BurnTrackPredictor()

    # Prédiction complète
    result = predictor.predict(
        fuel_id='AF_MIOMBO',
        wind_speed=5.0,
        moisture_1h=0.05,      # fraction (0-1)
        moisture_live=0.5,     # fraction (0-1)
        slope_pct=5.0
    )

    # Résultat:
    #   result['ros_rothermel'] : ROS du moteur physique (m/min)
    #   result['delta_mlp']     : Correction IA (m/min)
    #   result['ros_burntrack']   : ROS corrigée (m/min)
"""

import numpy as np
import pandas as pd
import torch
import json
import pickle
from typing import Union, Dict, Tuple, Optional
from pathlib import Path

# === IMPORTS MOTEUR BURNTRACK ===
import sys

# Ajouter la racine du projet et le dossier local au path
_BRIDGE_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _BRIDGE_DIR.parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))
if str(_BRIDGE_DIR) not in sys.path:
    sys.path.insert(0, str(_BRIDGE_DIR))

from burntrack.engine.rothermel import (
    RothermelEngine, FuelModel as RothermelFuelModel, 
    MoistureInputs, EnvironmentalConditions, RothermelOutput
)
from burntrack.engine.fuel_models import get_fuel_model

from source.model import BurnTrackMLPMinimal, BurnTrackAdvancedCorrector, BurnTrackFTGatedCorrector


# =====================================================================
# PONT ROTHERMEL ↔ MLP
# =====================================================================

class BurnTrackPredictor:
    """
    Prédicteur complet BurnTrack : Rothermel Engine v3 + MLP Corrector / PINN / Gated Tabular.

    Structure additive : ROS_burntrack = ROS_rothermel + delta_mlp
    """

    def __init__(self, 
                 model_path: str = "checkpoints/burntrack_mlp_minimal.pt",
                 scaler_path: str = "scaler.pkl",
                 fuel_encoding_path: str = "fuel_encoding.json"):
        """
        Charge le moteur Rothermel + le correcteur IA.

        Args:
            model_path: Checkpoint du modèle IA
            scaler_path: Scaler des features
            fuel_encoding_path: Target encoding des fuels
        """
        # 1. Moteur Rothermel v3
        self.rothermel = RothermelEngine()

        # Résoudre les chemins par défaut et relatifs relativement au dossier de bridge.py
        BRIDGE_DIR = Path(__file__).resolve().parent
        
        if model_path == "checkpoints/burntrack_mlp_minimal.pt":
            model_path = str(BRIDGE_DIR / "checkpoints" / "burntrack_mlp_minimal.pt")
        elif not Path(model_path).exists() and (BRIDGE_DIR / model_path).exists():
            model_path = str(BRIDGE_DIR / model_path)
            
        if scaler_path == "scaler.pkl":
            scaler_path = str(BRIDGE_DIR / "scaler.pkl")
        elif not Path(scaler_path).exists() and (BRIDGE_DIR / scaler_path).exists():
            scaler_path = str(BRIDGE_DIR / scaler_path)
            
        if fuel_encoding_path == "fuel_encoding.json":
            fuel_encoding_path = str(BRIDGE_DIR / "fuel_encoding.json")
        elif not Path(fuel_encoding_path).exists() and (BRIDGE_DIR / fuel_encoding_path).exists():
            fuel_encoding_path = str(BRIDGE_DIR / fuel_encoding_path)

        # 2. Correcteur MLP minimal
        checkpoint = torch.load(model_path, map_location='cpu', weights_only=False)

        config = checkpoint.get('config', {})
        hidden1 = config.get('hidden1', 64)
        hidden2 = config.get('hidden2', 32)
        dropout = config.get('dropout', 0.2)

        # On ne charge plus que le MLP minimal (les architectures gated/advanced
        # avaient été entraînées avec une fuite de cible et ne sont plus fiables).
        self.mlp = BurnTrackMLPMinimal(
            n_features=checkpoint['n_features'],
            hidden1=hidden1, hidden2=hidden2, dropout=dropout
        )
        self.mlp.load_state_dict(checkpoint['model_state_dict'])
        self.mlp.eval()

        # 3. Scaler et encoding
        with open(scaler_path, 'rb') as f:
            self.scaler = pickle.load(f)

        with open(fuel_encoding_path, 'r') as f:
            self.fuel_encoding = json.load(f)

        self.feature_cols = checkpoint['feature_cols']

        print(f"✅ BurnTrack Predictor chargé")
        print(f"   Rothermel: Engine v3")
        print(f"   Modèle IA: {checkpoint['n_features']} features -> {hidden1} -> {hidden2} -> 1")
        print(f"   Fuels connus: {len(self.fuel_encoding)}")

    def _get_rothermel_prediction(self,
                                   fuel_id: str,
                                   wind_speed: float,
                                   moisture_1h: float,
                                   moisture_live: float,
                                   slope_pct: float,
                                   angle_wind_slope: float = 0.0):
        """
        Appelle le moteur Rothermel v3 pour obtenir la sortie complète.
        """
        # Récupération du FuelModel depuis fuel_models.py
        fuel_model = get_fuel_model(fuel_id)
        if fuel_model is None:
            raise ValueError(f"Fuel '{fuel_id}' inconnu. Vérifie fuel_models.py")

        # Conversion vers le format RothermelFuelModel
        rothermel_fuel = RothermelFuelModel(
            name=fuel_model.code,
            w_1h=fuel_model.w_1h,
            w_10h=fuel_model.w_10h,
            w_100h=fuel_model.w_100h,
            w_live_herb=fuel_model.w_live_herb,
            w_live_woody=fuel_model.w_live_woody,
            sigma_1h=fuel_model.sigma_1h,
            sigma_10h=fuel_model.sigma_10h,
            sigma_100h=fuel_model.sigma_100h,
            sigma_live_herb=fuel_model.sigma_live_herb,
            sigma_live_woody=fuel_model.sigma_live_woody,
            delta=fuel_model.delta,
            mx=fuel_model.mx,
            h_dead=fuel_model.h_dead,
            h_live=fuel_model.h_live,
            st=0.0555,
            se=0.01
        )

        # Construction des humidités
        moisture = MoistureInputs(
            m_1h=moisture_1h,
            m_10h=moisture_1h * 1.5,
            m_100h=moisture_1h * 2.0,
            m_live_herb=moisture_live,
            m_live_woody=moisture_live * 1.2
        )

        # Conditions environnementales
        conditions = EnvironmentalConditions(
            wind_speed=wind_speed,
            slope_pct=slope_pct,
            angle_wind_slope=angle_wind_slope
        )

        # Calcul Rothermel
        output = self.rothermel.compute(rothermel_fuel, moisture, conditions)

        return output, fuel_model

    def _prepare_mlp_features(self,
                               fuel_id: str,
                               wind_speed: float,
                               moisture_1h: float,
                               slope_pct: float,
                               output,
                               fuel_model,
                               temp_c: float = 25.0,
                               aspect_deg: float = 0.0) -> np.ndarray:
        """
        Prépare les features pour le modèle IA (target encoding + normalisation).
        NOTE: 'thermal_proxy' (anciennement construit à partir de la cible) a été
        supprimé pour éviter la fuite d'information au moment de l'inférence.
        """
        # Target encoding du fuel
        fuel_encoded = self.fuel_encoding.get(fuel_id,
                                               np.mean(list(self.fuel_encoding.values())))

        # Thermodynamique
        rh_percent = moisture_1h * 100.0
        es = 0.6108 * np.exp(17.27 * temp_c / (temp_c + 237.3))
        vpd = max(0.0, es * (1.0 - rh_percent / 100.0))
        dfmc = float(np.clip(30.0 - 2.5 * vpd - 0.1 * temp_c, 3.0, 40.0))

        w_1h, w_10h, w_100h = fuel_model.w_1h, fuel_model.w_10h, fuel_model.w_100h
        w_live_herb, w_live_woody = fuel_model.w_live_herb, fuel_model.w_live_woody
        w_dead = w_1h + w_10h + w_100h
        w_live = w_live_herb + w_live_woody
        w_total = w_dead + w_live

        sigma_1h = getattr(fuel_model, "sigma_1h", 0.0)
        sigma_10h = getattr(fuel_model, "sigma_10h", 0.0)
        sigma_100h = getattr(fuel_model, "sigma_100h", 0.0)
        sigma_live_herb = getattr(fuel_model, "sigma_live_herb", 0.0)
        sigma_live_woody = getattr(fuel_model, "sigma_live_woody", 0.0)

        sav_dead = (w_1h * sigma_1h + w_10h * sigma_10h + w_100h * sigma_100h) / w_dead if w_dead > 0 else 0.0
        sav_live = (w_live_herb * sigma_live_herb + w_live_woody * sigma_live_woody) / w_live if w_live > 0 else 0.0
        sigma_m2_m3 = (w_dead * sav_dead + w_live * sav_live) / w_total if w_total > 0 else 0.0

        beta_ratio = output.beta / output.beta_opt if output.beta_opt > 0 else 0.0

        # Construction du vecteur features (sans thermal_proxy — évite la fuite de cible)
        feature_dict = {
            'fuel_encoded': fuel_encoded,
            'w_total_kg_m2': w_total,
            'w_dead_kg_m2': w_dead,
            'w_live_kg_m2': w_live,
            'delta_m': getattr(fuel_model, 'delta', 0.5),
            'sigma_m2_m3': sigma_m2_m3,
            'mx_percent': getattr(fuel_model, 'mx', 20.0),
            'slope': slope_pct,
            'aspect_deg': aspect_deg,
            'wind_speed': wind_speed,
            'humidity': rh_percent,
            'temp_c': temp_c,
            'vpd_kpa': vpd,
            'dfmc_percent': dfmc,
            'ros_rothermel': output.ros,
            'phi_w': output.phi_w,
            'phi_s': output.phi_s,
            'phi_eff': output.phi_eff,
            'beta_ratio': beta_ratio,
            'I_R_kW_m2': output.reaction_intensity,
            'xi': output.xi,
            'tau_min': output.tau,
            'wind_sq': wind_speed ** 2,
            'slope_sq': slope_pct ** 2,
            'wind_slope_inter': wind_speed * slope_pct,
            'wind_hum_ratio': wind_speed / (rh_percent + 1e-5),
            'energy_flux': w_total * wind_speed,
            'roth_sq': output.ros ** 2,
            'roth_wind': output.ros * wind_speed,
            'temp_vpd': temp_c * vpd,
            'brightness_k': 305.0
        }

        # Sélection et ordre des features
        features = np.array([[feature_dict.get(col, 0.0) for col in self.feature_cols]])

        # Normalisation
        features_scaled = self.scaler.transform(features)

        return features_scaled

    def predict(self,
                fuel_id: str,
                wind_speed: float,
                moisture_1h: float,
                moisture_live: float = 0.5,
                slope_pct: float = 0.0,
                angle_wind_slope: float = 0.0,
                return_components: bool = False) -> Union[float, Dict]:
        """
        Prédit la ROS corrigée : Rothermel + Modèle IA (sans fuite de cible).
        """
        # 1. Prédiction Rothermel v3
        output, fuel_model = self._get_rothermel_prediction(
            fuel_id=fuel_id,
            wind_speed=wind_speed,
            moisture_1h=moisture_1h,
            moisture_live=moisture_live,
            slope_pct=slope_pct,
            angle_wind_slope=angle_wind_slope
        )
        ros_r = output.ros

        # 2. Préparation features IA
        features = self._prepare_mlp_features(
            fuel_id=fuel_id,
            wind_speed=wind_speed,
            moisture_1h=moisture_1h,
            slope_pct=slope_pct,
            output=output,
            fuel_model=fuel_model,
        )

        # 3. Prédiction delta par le modèle IA
        with torch.no_grad():
            features_t = torch.tensor(features, dtype=torch.float32)
            delta = self.mlp(features_t).item()

        # 4. Structure additive : ROS_final = ROS_r + delta
        ros_final = ros_r + delta

        # Sécurité physique : ROS ne peut pas être négative
        ros_final = max(0.0, ros_final)

        if return_components:
            fuel_encoded = self.fuel_encoding.get(fuel_id, 
                                                   np.mean(list(self.fuel_encoding.values())))
            return {
                'fuel_id': fuel_id,
                'fuel_encoded': fuel_encoded,
                'wind_speed': wind_speed,
                'moisture_1h': moisture_1h,
                'moisture_live': moisture_live,
                'slope_pct': slope_pct,
                'ros_rothermel': ros_r,
                'delta_mlp': delta,
                'ros_burntrack': ros_final,
                'correction_pct': (delta / ros_r * 100) if ros_r > 0 else 0,
                'bias_type': 'sous-estimation' if delta > 0 else 'sur-estimation'
            }

        return ros_final

    def predict_batch(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Prédit sur un batch de données (DataFrame).
        """
        results = []

        for idx, row in df.iterrows():
            result = self.predict(
                fuel_id=row['fuel_model'],
                wind_speed=row['wind_speed'],
                moisture_1h=row.get('moisture_1h', 0.05),
                moisture_live=row.get('moisture_live', 0.5),
                slope_pct=row.get('slope_pct', 0.0),
                return_components=True
            )
            results.append(result)

        result_df = pd.DataFrame(results)
        return pd.concat([df.reset_index(drop=True), result_df], axis=1)


# =====================================================================
# DÉMONSTRATION
# =====================================================================

def demo():
    """Démonstration du pont Rothermel-MLP."""
    print("=" * 60)
    print("🔥 BURNTRACK - DÉMONSTRATION PONT ROTHERMEL v3 ↔ MLP")
    print("=" * 60)

    # Vérification que les fichiers existent
    checkpoint_path = Path(__file__).resolve().parent / "checkpoints/burntrack_mlp_minimal.pt"
    if not checkpoint_path.exists():
        print(f"\n❌ Modèle MLP non trouvé à {checkpoint_path}. Lancez d'abord:")
        print("   python main.py --train")
        return

    # Initialisation
    predictor = BurnTrackPredictor()

    # Cas test 1: AF_MIOMBO (savane miombo)
    print("\n📊 Cas 1: AF_MIOMBO (savane miombo)")
    print("-" * 40)

    result = predictor.predict(
        fuel_id='AF_MIOMBO',
        wind_speed=5.0,
        moisture_1h=0.05,
        moisture_live=0.5,
        slope_pct=5.0,
        return_components=True
    )

    print(f"Fuel: {result['fuel_id']}")
    print(f"Fuel encoded (biais moyen): {result['fuel_encoded']:+.2f} m/min")
    print(f"Conditions: vent={result['wind_speed']} m/s, "
          f"humidité mort={result['moisture_1h']*100:.0f}%, "
          f"humidité vivant={result['moisture_live']*100:.0f}%, "
          f"pente={result['slope_pct']}%")
    print(f"\nRothermel v3 (physique): {result['ros_rothermel']:.2f} m/min")
    print(f"Delta IA (correction):   {result['delta_mlp']:+.2f} m/min")
    print(f"BurnTrack (corrigé):     {result['ros_burntrack']:.2f} m/min")
    print(f"\nType: {result['bias_type']} ({result['correction_pct']:+.1f}%)")

    # Cas test 2: GR3 (herbe)
    print("\n📊 Cas 2: GR3 (herbe)")
    print("-" * 40)

    result2 = predictor.predict(
        fuel_id='GR3',
        wind_speed=8.0,
        moisture_1h=0.08,
        moisture_live=0.6,
        slope_pct=2.0,
        return_components=True
    )

    print(f"Fuel: {result2['fuel_id']}")
    print(f"Fuel encoded (biais moyen): {result2['fuel_encoded']:+.2f} m/min")
    print(f"Rothermel v3 (physique): {result2['ros_rothermel']:.2f} m/min")
    print(f"Delta IA (correction):   {result2['delta_mlp']:+.2f} m/min")
    print(f"BurnTrack (corrigé):     {result2['ros_burntrack']:.2f} m/min")
    print(f"\nType: {result2['bias_type']} ({result2['correction_pct']:+.1f}%)")

    # Cas test 3: AF_FYNBOS (boisé)
    print("\n📊 Cas 3: AF_FYNBOS (boisé)")
    print("-" * 40)

    result3 = predictor.predict(
        fuel_id='AF_FYNBOS',
        wind_speed=10.0,
        moisture_1h=0.12,
        moisture_live=0.8,
        slope_pct=15.0,
        return_components=True
    )

    print(f"Fuel: {result3['fuel_id']}")
    print(f"Fuel encoded (biais moyen): {result3['fuel_encoded']:+.2f} m/min")
    print(f"Rothermel v3 (physique): {result3['ros_rothermel']:.2f} m/min")
    print(f"Delta IA (correction):   {result3['delta_mlp']:+.2f} m/min")
    print(f"BurnTrack (corrigé):     {result3['ros_burntrack']:.2f} m/min")
    print(f"\nType: {result3['bias_type']} ({result3['correction_pct']:+.1f}%)")

    print("\n" + "=" * 60)
    print("✅ DÉMONSTRATION TERMINÉE")
    print("=" * 60)
    print("\nStructure additive: ROS_burntrack = ROS_rothermel + delta_mlp")
    print("Le MLP corrige les biais systématiques du moteur physique.")


if __name__ == "__main__":
    demo()
