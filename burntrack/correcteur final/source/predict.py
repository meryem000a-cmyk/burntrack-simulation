"""
Inférence BurnTrack
Charge un modèle entraîné et prédit la correction delta
"""

import numpy as np
import pandas as pd
import torch
import json
import pickle
from typing import Union, List, Dict


class BurnTrackPredictor:
    """
    Prédicteur pour BurnTrack.

    Usage:
        predictor = BurnTrackPredictor('checkpoints/burntrack_model.pt')
        ros_corrected = predictor.predict(new_data_df)
    """

    def __init__(self, model_path: str, scaler_path: str = 'scaler.pkl', 
                 fuel_encoding_path: str = 'fuel_encoding.json'):
        """
        Args:
            model_path: Chemin vers le checkpoint .pt
            scaler_path: Chemin vers le scaler.pkl
            fuel_encoding_path: Chemin vers fuel_encoding.json
        """
        checkpoint = torch.load(model_path, map_location='cpu')

        from .model import BurnTrackMLPMinimal
        self.model = BurnTrackMLPMinimal(
            n_features=checkpoint['n_features'],
            hidden1=64, hidden2=32, dropout=0.2
        )
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.eval()

        with open(scaler_path, 'rb') as f:
            self.scaler = pickle.load(f)

        with open(fuel_encoding_path, 'r') as f:
            self.fuel_encoding = json.load(f)

        self.feature_cols = checkpoint['feature_cols']
        self.ros_r_col = 'ros_rothermel'

        print(f"✅ Modèle chargé: {model_path}")
        print(f"   Features: {self.feature_cols}")
        print(f"   Fuels connus: {len(self.fuel_encoding)}")

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        """
        Prédit la ROS corrigée pour un DataFrame.

        Args:
            df: DataFrame avec les colonnes nécessaires

        Returns:
            ROS corrigée (ROS_r + delta_pred)
        """
        df = df.copy()
        df['fuel_encoded'] = df['fuel_model'].map(self.fuel_encoding)

        unknown_mask = df['fuel_encoded'].isna()
        if unknown_mask.sum() > 0:
            print(f"⚠️ {unknown_mask.sum()} fuels inconnus -> moyenne globale")
            df.loc[unknown_mask, 'fuel_encoded'] = np.mean(list(self.fuel_encoding.values()))

        X = df[self.feature_cols].values
        X_scaled = self.scaler.transform(X)

        with torch.no_grad():
            X_t = torch.tensor(X_scaled, dtype=torch.float32)
            delta_pred = self.model(X_t).numpy()

        ros_corrected = df[self.ros_r_col].values + delta_pred
        return ros_corrected

    def predict_delta(self, df: pd.DataFrame) -> np.ndarray:
        """Prédit uniquement le delta (correction)."""
        df = df.copy()
        df['fuel_encoded'] = df['fuel_model'].map(self.fuel_encoding)
        df['fuel_encoded'] = df['fuel_encoded'].fillna(np.mean(list(self.fuel_encoding.values())))

        X = df[self.feature_cols].values
        X_scaled = self.scaler.transform(X)

        with torch.no_grad():
            X_t = torch.tensor(X_scaled, dtype=torch.float32)
            delta = self.model(X_t).numpy()

        return delta
