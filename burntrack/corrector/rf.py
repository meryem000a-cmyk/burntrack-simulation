import numpy as np
from typing import Dict, Optional
import joblib
import os

from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler

from burntrack.corrector.base import BaseCorrector
from burntrack.corrector.features import CorrectorFeatureExtractor, ALL_CONTINUOUS_FEATURES

FUEL_MODEL_ENCODING = {
    'GR1': 0, 'GR2': 1, 'GR3': 2, 'GR4': 3, 'GR5': 4,
    'GR6': 5, 'GR7': 6, 'GR8': 7, 'GR9': 8,
    'GS1': 9, 'GS2': 10, 'GS3': 11, 'GS4': 12,
    'SH1': 13, 'SH2': 14, 'SH3': 15, 'SH4': 16, 'SH5': 17,
    'SH6': 18, 'SH7': 19, 'SH8': 20, 'SH9': 21,
    'AF_STEPPE': 22, 'AF_STEPPE_DENSE': 23, 'AF_ARGAN': 24,
    'AF_CHENE_LIEGE': 25, 'AF_CEDRE': 26, 'AF_MAQUIS': 27,
    'AF_CEREALES': 28, 'AF_PALMIER': 29, 'AF_TAMARIX': 30, 'AF_JUJUBIER': 31,
    'AF_SAHEL_GRASS': 32, 'AF_SAHEL_WOODED': 33,
    'AF_SUDAN_GRASS': 34, 'AF_SUDAN_WOODED': 35,
    'AF_MIOMBO': 36, 'AF_MIOMBO_DENSE': 37, 'AF_MOPANE': 38,
    'AF_ACACIA_SAVANNA': 39, 'AF_GRASSLAND_FERTILE': 40,
    'AF_FYNBOS': 41, 'AF_FYNBOS_YOUNG': 42,
    'AF_BUSHVELD': 43, 'AF_BAOBAB': 44, 'AF_FOREST_DRY': 45,
    'AF_AFROMONTANE': 46, 'AF_MANGROVE': 47,
    'AF_RANGE_DEGRADED': 48, 'AF_RANGE_INTACT': 49,
}


def encode_fuel_model(code: str) -> int:
    return FUEL_MODEL_ENCODING.get(code, 0)


class RandomForestCorrector(BaseCorrector):
    """
    Correcteur Random Forest pour delta_ros.
    """

    def __init__(
        self,
        model_path: Optional[str] = None,
        scaler_path: Optional[str] = None,
    ):
        self.model: Optional[RandomForestRegressor] = None
        self.scaler: Optional[StandardScaler] = None
        self.feature_extractor = CorrectorFeatureExtractor()

        if model_path is not None:
            self.load(model_path, scaler_path)

    def load(self, model_path: str, scaler_path: Optional[str] = None):
        self.model = joblib.load(model_path)
        if scaler_path is not None:
            self.scaler = joblib.load(scaler_path)
        return self

    def _build_features(self, features: Dict) -> np.ndarray:
        extracted = self.feature_extractor.extract_row(features)
        feature_names = self.feature_extractor.get_feature_names()
        x = np.array([[extracted.get(name, 0.0) for name in feature_names]], dtype=np.float64)

        if self.scaler is not None:
            x = self.scaler.transform(x)
        return x

    def predict(self, features: Dict) -> Dict[str, float]:
        ros_rothermel = float(features.get('ros_rothermel', features.get('ros', 0.0)))
        X = self._build_features(features)
        delta_ros = float(self.model.predict(X)[0])
        return {
            'delta_ros': delta_ros,
            'ros_corrected': ros_rothermel + delta_ros,
        }

    def predict_with_uncertainty(self, features: Dict) -> Dict:
        ros_rothermel = float(features.get('ros_rothermel', features.get('ros', 0.0)))
        X = self._build_features(features)

        tree_preds = np.array([tree.predict(X) for tree in self.model.estimators_])
        delta_ros = float(tree_preds.mean())
        uncertainty = float(tree_preds.std())

        return {
            'delta_ros': delta_ros,
            'ros_corrected': ros_rothermel + delta_ros,
            'uncertainty': uncertainty,
        }
