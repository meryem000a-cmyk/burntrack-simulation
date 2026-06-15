import numpy as np
from typing import Dict, Optional
import joblib
import os

from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler

from burntrack.corrector.base import BaseCorrector
from burntrack.corrector.mlp import (
    build_ia_vector,
    encode_fuel_model,
    REQUIRED_FEATURES,
)


class RandomForestCorrector(BaseCorrector):
    """
    Correcteur Random Forest pour delta_ros.

    Charge un modèle entraîné + scaler via joblib.
    Construit le vecteur de features avec build_ia_vector,
    scale, prédit delta_ros.
    """

    def __init__(
        self,
        model_path: Optional[str] = None,
        scaler_path: Optional[str] = None,
    ):
        self.model: Optional[RandomForestRegressor] = None
        self.scaler: Optional[StandardScaler] = None

        self.continuous_features = [f for f in REQUIRED_FEATURES if f != 'fuel_model_code']

        if model_path is not None:
            self.load(model_path, scaler_path)

    def load(self, model_path: str, scaler_path: Optional[str] = None):
        """Charge modèle et scaler depuis des fichiers joblib."""
        self.model = joblib.load(model_path)
        if scaler_path is not None:
            self.scaler = joblib.load(scaler_path)
        return self

    def _build_features(self, features: Dict) -> np.ndarray:
        """Construit le vecteur complet [n_continuous + 1] (scaled + fuel_idx)."""
        build_kwargs = {f: features[f] for f in REQUIRED_FEATURES if f != 'fuel_model_code'}
        build_kwargs['fuel_model_code'] = features['fuel_model_code']
        x_continuous, fuel_idx = build_ia_vector(**build_kwargs)

        x_continuous = x_continuous.reshape(1, -1)
        if self.scaler is not None:
            x_continuous = self.scaler.transform(x_continuous)

        fuel_idx_arr = np.array([[fuel_idx]], dtype=np.float32)
        X = np.hstack([x_continuous, fuel_idx_arr])
        return X

    def predict(self, features: Dict) -> Dict[str, float]:
        """Retourne {'delta_ros': float, 'ros_corrected': float}."""
        ros_rothermel = float(features['ros_rothermel'])
        X = self._build_features(features)
        delta_ros = float(self.model.predict(X)[0])
        return {
            'delta_ros': delta_ros,
            'ros_corrected': ros_rothermel + delta_ros,
        }

    def predict_with_uncertainty(self, features: Dict) -> Dict:
        """
        Retourne delta_ros, ros_corrected et uncertainty
        via l'écart-type des arbres individuels.
        """
        ros_rothermel = float(features['ros_rothermel'])
        X = self._build_features(features)

        tree_preds = np.array([tree.predict(X) for tree in self.model.estimators_])
        delta_ros = float(tree_preds.mean())
        uncertainty = float(tree_preds.std())

        return {
            'delta_ros': delta_ros,
            'ros_corrected': ros_rothermel + delta_ros,
            'uncertainty': uncertainty,
        }
