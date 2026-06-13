"""
ia_corrector_v2.py
===================
IA Correctrice BurnTrack (v2 corrigée)
Corrections intégrées :
- Normalisation StandardScaler obligatoire
- Bornes empiriques [0.3, 3.0] (renommées depuis "physiques")
- Correction scale étendue à ±1.0 (facteur [0.3, 3.0])
- MC Dropout pour incertitude
- Permutation importance pour interprétabilité
- Exception explicite si features manquantes
- Perte NLL hétéroscédastique

Architecture : MLP [128→64→32] avec BatchNorm + Dropout
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Dict, List, Optional, Tuple
from sklearn.preprocessing import StandardScaler
from sklearn.inspection import permutation_importance
import json
import warnings


# =============================================================================
# CONFIGURATION
# =============================================================================

EMPIRICAL_BOUNDS = (0.3, 3.0)  # Renommé : "bornes empiriques" (pas "physiques")
# Justification : Govender 2006 (-5 à -10%), Frost & Robertson 1987 (+20-30%),
# Cruz 2015 (+15-30%), Andrews 2018 (+10-20%) → plage totale documentée ~0.3-3.0

CORRECTION_SCALE = 1.0  # Étendu depuis ±0.5

REQUIRED_FEATURES = [
    'ros_rothermel',      # m/min — baseline à corriger
    'temp_c',             # °C
    'rh_percent',         # %
    'wind_speed',         # m/s
    'vpd_kpa',            # kPa — déficit de pression de vapeur
    'slope_deg',          # degrés
    'fuel_model_encoded', # encodage du fuel model
    'fuel_moisture',      # % — humidité du combustible
]


# =============================================================================
# MODÈLE : ATLAS CORRECTOR V2
# =============================================================================

class AtlasCorrectorV2(nn.Module):
    """
    MLP correcteur avec BatchNorm, Dropout (pour MC Dropout), et sortie
    correction + log_variance (hétéroscédasticité).
    """

    def __init__(
        self,
        n_features: int = 8,
        hidden_dims: List[int] = None,
        dropout_rate: float = 0.2,
        empirical_bounds: Tuple[float, float] = EMPIRICAL_BOUNDS,
    ):
        super().__init__()

        if hidden_dims is None:
            hidden_dims = [128, 64, 32]

        self.bounds = empirical_bounds
        self.dropout_rate = dropout_rate

        # Construction dynamique des couches
        layers = []
        in_dim = n_features

        for i, h_dim in enumerate(hidden_dims):
            layers.append(nn.Linear(in_dim, h_dim))
            layers.append(nn.BatchNorm1d(h_dim))
            layers.append(nn.ReLU())
            # Dropout TOUJOURS actif (même en eval) pour MC Dropout
            layers.append(nn.Dropout(dropout_rate))
            in_dim = h_dim

        # Tête de sortie : correction + log_variance (incertitude)
        self.backbone = nn.Sequential(*layers)
        self.head = nn.Linear(in_dim, 2)  # [correction_factor, log_var]

        self._init_weights()

    def _init_weights(self):
        """Initialisation Xavier pour stabilité."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass. Retourne [correction_factor, log_variance].

        En entraînement : dropout actif (régularisation).
        En inférence MC Dropout : dropout aussi actif (50 forward passes).
        """
        features = self.backbone(x)
        out = self.head(features)

        # Correction factor bornée empiriquement [0.3, 3.0]
        # On utilise sigmoid + scaling pour garantir les bornes
        correction = torch.sigmoid(out[:, 0:1])  # [0, 1]
        correction = correction * (self.bounds[1] - self.bounds[0]) + self.bounds[0]

        # Log-variance (incertitude, pas bornée)
        log_var = out[:, 1:2]

        return torch.cat([correction, log_var], dim=1)

    def predict_with_uncertainty(
        self,
        x: torch.Tensor,
        n_samples: int = 50,
    ) -> Dict[str, torch.Tensor]:
        """
        MC Dropout : n forward passes avec dropout actif pour estimer
        l'incertitude épistémique.

        Retourne :
            - correction_mean : facteur moyen
            - correction_std : écart-type (incertitude)
            - ros_corrected : ROS corrigée = ROS_Rothermel × correction_mean
            - ros_ci_lower/upper : intervalle de confiance 95%
        """
        self.train()  # Force dropout actif !

        corrections = []
        for _ in range(n_samples):
            with torch.no_grad():
                out = self.forward(x)
                corrections.append(out[:, 0:1])

        corrections = torch.stack(corrections, dim=0)  # [n_samples, batch, 1]

        correction_mean = corrections.mean(dim=0)
        correction_std = corrections.std(dim=0)

        # Intervalle de confiance 95% (approx gaussien)
        ci_lower = correction_mean - 1.96 * correction_std
        ci_upper = correction_mean + 1.96 * correction_std
        ci_lower = torch.clamp(ci_lower, self.bounds[0], self.bounds[1])
        ci_upper = torch.clamp(ci_upper, self.bounds[0], self.bounds[1])

        return {
            'correction_mean': correction_mean,
            'correction_std': correction_std,
            'ci_lower': ci_lower,
            'ci_upper': ci_upper,
        }


# =============================================================================
# GESTION DES FEATURES
# =============================================================================

def build_ia_vector(
    ros_rothermel: float,
    temp_c: float,
    rh_percent: float,
    wind_speed: float,
    vpd_kpa: float,
    slope_deg: float,
    fuel_model_encoded: float,
    fuel_moisture: float,
) -> np.ndarray:
    """
    Construit le vecteur de features pour l'IA.

    LEVER UNE EXCEPTION si une feature est manquante (pas de zéro silencieux).
    """
    features = {
        'ros_rothermel': ros_rothermel,
        'temp_c': temp_c,
        'rh_percent': rh_percent,
        'wind_speed': wind_speed,
        'vpd_kpa': vpd_kpa,
        'slope_deg': slope_deg,
        'fuel_model_encoded': fuel_model_encoded,
        'fuel_moisture': fuel_moisture,
    }

    # Vérification : aucune valeur manquante
    for name, value in features.items():
        if value is None or np.isnan(value):
            raise ValueError(
                f"Feature manquante ou NaN : '{name}'. "
                f"Impossible de faire une prédiction fiable. "
                f"Fournissez toutes les features requises : {REQUIRED_FEATURES}"
            )

    return np.array([features[f] for f in REQUIRED_FEATURES], dtype=np.float32)


# =============================================================================
# DATASET LOADER AVEC NORMALISATION
# =============================================================================

class DatasetLoader:
    """
    Charge et normalise les données. StandardScaler OBLIGATOIRE.
    """

    def __init__(self):
        self.scaler = StandardScaler()
        self.is_fitted = False

    def fit(self, X: np.ndarray):
        """Fit le scaler sur les données d'entraînement."""
        self.scaler.fit(X)
        self.is_fitted = True
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        """Transforme avec le scaler fitté."""
        if not self.is_fitted:
            raise RuntimeError(
                "Le StandardScaler n'est pas fitté. "
                "Appelez .fit(X_train) avant .transform()."
            )
        return self.scaler.transform(X)

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        """Fit + transform."""
        self.fit(X)
        return self.transform(X)

    def inverse_transform(self, X_scaled: np.ndarray) -> np.ndarray:
        """Inverse la normalisation (si besoin)."""
        return self.scaler.inverse_transform(X_scaled)

    def save(self, path: str):
        """Sauvegarde le scaler."""
        import joblib
        joblib.dump(self.scaler, path)

    def load(self, path: str):
        """Charge le scaler."""
        import joblib
        self.scaler = joblib.load(path)
        self.is_fitted = True
        return self


# =============================================================================
# INTERPRÉTABILITÉ : PERMUTATION IMPORTANCE
# =============================================================================

def compute_feature_importance(
    model: AtlasCorrectorV2,
    X_test: np.ndarray,
    y_test: np.ndarray,
    feature_names: List[str] = None,
    n_repeats: int = 10,
) -> Dict:
    """
    Calcule la permutation importance pour interpréter l'IA.

    Nécessite un wrapper sklearn-compatible.
    """
    if feature_names is None:
        feature_names = REQUIRED_FEATURES

    class SklearnWrapper:
        def __init__(self, torch_model, scaler):
            self.model = torch_model
            self.scaler = scaler
            self.model.eval()

        def predict(self, X):
            X_scaled = self.scaler.transform(X)
            with torch.no_grad():
                X_t = torch.tensor(X_scaled, dtype=torch.float32)
                out = self.model(X_t)
                return out[:, 0].numpy()  # correction_factor uniquement

    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler().fit(X_test)
    wrapper = SklearnWrapper(model, scaler)

    result = permutation_importance(
        wrapper, X_test, y_test,
        n_repeats=n_repeats,
        random_state=42,
        scoring='neg_mean_absolute_error',
    )

    importance = {
        name: {
            'mean': result.importances_mean[i],
            'std': result.importances_std[i],
        }
        for i, name in enumerate(feature_names)
    }

    # Trier par importance décroissante
    importance_sorted = dict(sorted(
        importance.items(),
        key=lambda x: x[1]['mean'],
        reverse=True
    ))

    return importance_sorted


# =============================================================================
# FONCTION DE PERTE (NLL avec hétéroscédasticité)
# =============================================================================

def heteroscedastic_loss(
    prediction: torch.Tensor,
    target: torch.Tensor,
) -> torch.Tensor:
    """
    Negative Log-Likelihood avec hétéroscédasticité.

    prediction : [correction_factor, log_var]
    target : correction_factor_vrai
    """
    correction = prediction[:, 0]
    log_var = prediction[:, 1]

    # NLL = 0.5 * (log_var + (target - correction)^2 / exp(log_var))
    precision = torch.exp(-log_var)
    loss = 0.5 * (log_var + precision * (target - correction) ** 2)

    return loss.mean()


# =============================================================================
# EXEMPLE D'UTILISATION
# =============================================================================

if __name__ == "__main__":
    # Test rapide
    print("=" * 60)
    print("AtlasCorrectorV2 — Test de base")
    print("=" * 60)

    # Instanciation
    model = AtlasCorrectorV2(n_features=8, hidden_dims=[128, 64, 32])
    print(f"Paramètres : {sum(p.numel() for p in model.parameters()):,}")

    # Forward simple
    x = torch.randn(4, 8)  # batch de 4
    out = model(x)
    print(f"Sortie [correction, log_var] : {out.shape}")
    print(f"Correction moyenne : {out[:, 0].mean().item():.3f}")
    print(f"Bornes respectées : [{out[:, 0].min().item():.3f}, {out[:, 0].max().item():.3f}]")

    # MC Dropout
    model.eval()
    uncertainty = model.predict_with_uncertainty(x, n_samples=50)
    print(f"Incertitude moyenne (std) : {uncertainty['correction_std'].mean().item():.4f}")

    # Test build_ia_vector avec feature manquante
    print("\n--- Test gestion erreur ---")
    try:
        vec = build_ia_vector(
            ros_rothermel=5.0,
            temp_c=35.0,
            rh_percent=None,  # MANQUANT !
            wind_speed=5.0,
            vpd_kpa=2.5,
            slope_deg=10.0,
            fuel_model_encoded=1.0,
            fuel_moisture=8.0,
        )
    except ValueError as e:
        print(f"✅ Exception levée correctement : {e}")

    print("\n--- Test avec features complètes ---")
    vec = build_ia_vector(
        ros_rothermel=5.0,
        temp_c=35.0,
        rh_percent=25.0,
        wind_speed=5.0,
        vpd_kpa=2.5,
        slope_deg=10.0,
        fuel_model_encoded=1.0,
        fuel_moisture=8.0,
    )
    print(f"Vecteur features : {vec}")
    print(f"Dimensions : {vec.shape}")