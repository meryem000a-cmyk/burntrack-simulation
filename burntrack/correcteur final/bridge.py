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
from pathlib import Path

# Ajouter la racine du projet au path
_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from burntrack.engine.rothermel import (
    RothermelEngine, FuelModel as RothermelFuelModel, 
    MoistureInputs, EnvironmentalConditions, RothermelOutput
)
from burntrack.engine.fuel_models import get_fuel_model

from source.model import BurnTrackMLPMinimal


# =====================================================================
# PONT ROTHERMEL ↔ MLP
# =====================================================================

class BurnTrackPredictor:
    """
    Prédicteur complet BurnTrack : Rothermel Engine v3 + MLP Corrector.

    Structure additive : ROS_burntrack = ROS_rothermel + delta_mlp
    """

    def __init__(self, 
                 model_path: str = "checkpoints/burntrack_mlp_minimal.pt",
                 scaler_path: str = "scaler.pkl",
                 fuel_encoding_path: str = "fuel_encoding.json"):
        """
        Charge le moteur Rothermel + le correcteur MLP.

        Args:
            model_path: Checkpoint du MLP
            scaler_path: Scaler des features
            fuel_encoding_path: Target encoding des fuels
        """
        # 1. Moteur Rothermel v3
        self.rothermel = RothermelEngine()

        # 2. Correcteur MLP
        checkpoint = torch.load(model_path, map_location='cpu')

        self.mlp = BurnTrackMLPMinimal(
            n_features=checkpoint['n_features'],
            hidden1=64, hidden2=32, dropout=0.2
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
        print(f"   MLP: {checkpoint['n_features']} features -> 64 -> 32 -> 1")
        print(f"   Fuels connus: {len(self.fuel_encoding)}")

    def _get_rothermel_prediction(self,
                                   fuel_id: str,
                                   wind_speed: float,
                                   moisture_1h: float,
                                   moisture_live: float,
                                   slope_pct: float,
                                   angle_wind_slope: float = 0.0) -> float:
        """
        Appelle le moteur Rothermel v3 pour obtenir ROS_r.

        Args:
            fuel_id: Identifiant du fuel africain (ex: 'AF_MIOMBO', 'GR3')
            wind_speed: Vitesse du vent (m/s)
            moisture_1h: Humidité combustible mort 1h (fraction 0-1)
            moisture_live: Humidité combustible vivant (fraction 0-1)
            slope_pct: Pente (%)
            angle_wind_slope: Angle vent/pente (degrés, 0 = même direction)

        Returns:
            ROS_r: Vitesse prédite par Rothermel (m/min)
        """
        # Récupération du FuelModel depuis fuel_models.py
        fuel_model = get_fuel_model(fuel_id)
        if fuel_model is None:
            raise ValueError(f"Fuel '{fuel_id}' inconnu. Vérifie fuel_models.py")

        # Conversion vers le format RothermelFuelModel
        # Note: fuel_models.py et rothermel.py ont des FuelModel légèrement différents
        # On adapte ici
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
            mx=fuel_model.mx,  # fuel_models.py a mx en %, rothermel.py attend % aussi
            h_dead=fuel_model.h_dead,
            h_live=fuel_model.h_live,
            st=0.0555,  # Valeurs par défaut si absentes
            se=0.01
        )

        # Construction des humidités
        moisture = MoistureInputs(
            m_1h=moisture_1h,
            m_10h=moisture_1h * 1.5,  # Approximation standard
            m_100h=moisture_1h * 2.0,  # Approximation standard
            m_live_herb=moisture_live,
            m_live_woody=moisture_live * 1.2  # Approximation standard
        )

        # Conditions environnementales
        conditions = EnvironmentalConditions(
            wind_speed=wind_speed,
            slope_pct=slope_pct,
            angle_wind_slope=angle_wind_slope
        )

        # Calcul Rothermel
        output = self.rothermel.compute(rothermel_fuel, moisture, conditions)

        return output.ros

    def _prepare_mlp_features(self,
                               fuel_id: str,
                               wind_speed: float,
                               moisture_1h: float,
                               slope_pct: float,
                               ros_r: float) -> np.ndarray:
        """
        Prépare les features pour le MLP (target encoding + normalisation).

        Args:
            fuel_id: Fuel model africain
            wind_speed: Vent (m/s)
            moisture_1h: Humidité (fraction)
            slope_pct: Pente (%)
            ros_r: ROS prédite par Rothermel

        Returns:
            Features normalisées (1, n_features)
        """
        # Target encoding du fuel
        fuel_encoded = self.fuel_encoding.get(fuel_id, 
                                               np.mean(list(self.fuel_encoding.values())))

        # Construction du vecteur features
        feature_dict = {
            'fuel_encoded': fuel_encoded,
            'wind_speed': wind_speed,
            'humidity': moisture_1h * 100,  # Conversion fraction -> %
            'slope': slope_pct,
            'ros_rothermel': ros_r
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
        Prédit la ROS corrigée : Rothermel + MLP.

        Args:
            fuel_id: Fuel model africain (ex: 'AF_MIOMBO', 'GR3')
            wind_speed: Vitesse du vent (m/s)
            moisture_1h: Humidité combustible mort 1h (fraction, 0-1)
            moisture_live: Humidité combustible vivant (fraction, 0-1)
            slope_pct: Pente (%)
            angle_wind_slope: Angle vent/pente (degrés)
            return_components: Si True, retourne un dict avec tous les détails

        Returns:
            ROS_burntrack (m/min) ou dict avec ROS_r, delta, ROS_final, etc.
        """
        # 1. Prédiction Rothermel v3
        ros_r = self._get_rothermel_prediction(
            fuel_id=fuel_id,
            wind_speed=wind_speed,
            moisture_1h=moisture_1h,
            moisture_live=moisture_live,
            slope_pct=slope_pct,
            angle_wind_slope=angle_wind_slope
        )

        # 2. Préparation features MLP
        features = self._prepare_mlp_features(
            fuel_id=fuel_id,
            wind_speed=wind_speed,
            moisture_1h=moisture_1h,
            slope_pct=slope_pct,
            ros_r=ros_r
        )

        # 3. Prédiction delta par le MLP
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

        Args:
            df: DataFrame avec colonnes: fuel_model, wind_speed, moisture_1h, slope_pct, ...

        Returns:
            DataFrame enrichi avec ROS_r, delta, ROS_burntrack
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
    if not Path("checkpoints/burntrack_mlp_minimal.pt").exists():
        print("\n❌ Modèle MLP non trouvé. Lancez d'abord:")
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
