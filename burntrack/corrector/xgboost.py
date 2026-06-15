import numpy as np
from typing import Dict, Optional

from sklearn.preprocessing import StandardScaler

from burntrack.corrector.base import BaseCorrector
from burntrack.corrector.mlp import (
    build_ia_vector,
    encode_fuel_model,
    REQUIRED_FEATURES,
)


class XGBoostCorrector(BaseCorrector):
    """
    Correcteur XGBoost pour delta_ros.

    Construit le vecteur de features avec build_ia_vector,
    scale, prédit delta_ros via XGBoost.
    """

    def __init__(
        self,
        model_path: Optional[str] = None,
        scaler_path: Optional[str] = None,
        n_estimators: int = 500,
        max_depth: int = 12,
        learning_rate: float = 0.05,
    ):
        self.model = None
        self.scaler: Optional[StandardScaler] = None

        self.continuous_features = [f for f in REQUIRED_FEATURES if f != 'fuel_model_code']

        self._model_params = dict(
            n_estimators=n_estimators,
            max_depth=max_depth,
            learning_rate=learning_rate,
        )

        if model_path is not None:
            self.load(model_path, scaler_path)

    def load(self, model_path: str, scaler_path: Optional[str] = None):
        """Charge modèle et scaler depuis des fichiers joblib."""
        import joblib
        self.model = joblib.load(model_path)
        if scaler_path is not None:
            self.scaler = joblib.load(scaler_path)
        return self

    def save(self, model_path: str, scaler_path: Optional[str] = None):
        """Sauvegarde modèle et scaler en joblib."""
        import joblib
        joblib.dump(self.model, model_path)
        if scaler_path is not None and self.scaler is not None:
            joblib.dump(self.scaler, scaler_path)

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
        Estime l'incertitude via la std des prédictions individuelles
        des arbres XGBoost (CV folds ou bagging interne).

        Utilise xgboost.Booster.predict avec output_margin=False
        et l'écart-type des contributions des arbres si disponible.
        """
        ros_rothermel = float(features['ros_rothermel'])
        X = self._build_features(features)

        try:
            booster = self.model.get_booster()
            ntree_limit = 0
            preds = []
            for i in range(1, self.model.n_estimators + 1):
                p = booster.predict(
                    X, iteration_range=(0, i), output_margin=False
                )
                preds.append(p[0])
            preds = np.array(preds)
            delta_ros = float(preds[-1])
            uncertainty = float(preds.std())
        except Exception:
            delta_ros = float(self.model.predict(X)[0])
            uncertainty = 0.0

        return {
            'delta_ros': delta_ros,
            'ros_corrected': ros_rothermel + delta_ros,
            'uncertainty': uncertainty,
        }
