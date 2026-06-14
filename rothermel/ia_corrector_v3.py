"""
ia_corrector_v3.py
==================
IA Correctrice BurnTrack (v3 — corrigée pour Rothermel v3)

Corrections majeures par rapport à v2 :
- Correction ADDITIVE (pas multiplicative) : ROS_corr = ROS_Roth + delta_ros
- Features PHYSIQUES du moteur (phi_w, phi_s, beta, gamma, etc.)
- Architecture élargie [256, 128, 64] avec Dropout standard
- Perte MSE + L2 régularisation (pas NLL — trop bruité sur petits datasets)
- Pas de MC Dropout (trop lent) → Dropout standard + ensemble optionnel
- Compatible fuel_models.py et rothermel_engine_v3.py

Données d'entrée réelles (ce que tu as) :
- Capteurs robot : temp_air, rh, wind_speed, slope_deg
- Météo API : temp_2m, wind_10m, wind_gust, precip_1h
- Satellite : ndvi, ndwi, lst
- Moteur Rothermel : ros, phi_w, phi_s, beta, beta_opt, gamma, eta_M, eta_S, I_R, xi
- Fuel model : code, w_total, delta, sigma, mx
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Dict, List, Optional, Tuple
from sklearn.preprocessing import StandardScaler
import json
import warnings


# =============================================================================
# CONFIGURATION
# =============================================================================

# Features requises — ce que TU peux mesurer/obtenir
REQUIRED_FEATURES = [
    # --- Features météo/capteurs (TU les as) ---
    'temp_c',              # °C — température air (robot ou météo)
    'rh_percent',          # % — humidité relative
    'wind_speed_ms',       # m/s — vent à mi-flamme (ou 10m réduit)
    'vpd_kpa',             # kPa — déficit pression vapeur (calculé)
    'slope_deg',           # degrés — pente
    'slope_pct',           # % — pente (calculée depuis degrés)

    # --- Features fuel (depuis fuel_models.py) ---
    'fuel_model_code',     # str — code du fuel model
    'w_total_kg_m2',       # kg/m² — charge totale
    'w_dead_kg_m2',        # kg/m² — charge morte
    'w_live_kg_m2',        # kg/m² — charge vivante
    'delta_m',             # m — profondeur fuel bed
    'sigma_m2_m3',         # m²/m³ — SAV pondérée
    'mx_percent',          # % — humidité d'extinction
    'h_dead_kj_kg',        # kJ/kg — chaleur morts

    # --- Features moteur Rothermel (sorties intermédiaires) ---
    'ros_rothermel',       # m/min — ROS du moteur (baseline)
    'phi_w',               # — coefficient vent
    'phi_s',               # — coefficient pente
    'phi_eff',             # — coefficient effectif
    'beta',                # — packing ratio
    'beta_opt',            # — packing ratio optimal
    'beta_ratio',          # — beta / beta_opt
    'gamma',               # min⁻¹ — vitesse réaction
    'eta_M',               # — amortissement humidité
    'eta_S',               # — amortissement minéral
    'I_R_kW_m2',           # kW/m² — intensité réaction
    'xi',                  # — coefficient propagation
    'tau_min',             # min — temps résidence

    # --- Features satellite (TU les as) ---
    'ndvi',                # [-1, 1]
    'ndwi',                # [-1, 1]
    'lst_c',               # °C — land surface temp
    'dfmc_percent',        # % — dead fuel moisture content
]

# Paramètres du modèle
HIDDEN_DIMS = [256, 128, 64]
DROPOUT_RATE = 0.3
CORRECTION_SCALE = 5.0   # m/min — plage max de correction additive
L2_LAMBDA = 1e-4         # Régularisation L2


# =============================================================================
# ENCODEUR DE FUEL MODEL
# =============================================================================

FUEL_MODEL_ENCODING = {
    # Behave standard
    'GR1': 0, 'GR2': 1, 'GR3': 2, 'GR4': 3, 'GR5': 4,
    'GR6': 5, 'GR7': 6, 'GR8': 7, 'GR9': 8,
    'GS1': 9, 'GS2': 10, 'GS3': 11, 'GS4': 12,
    'SH1': 13, 'SH2': 14, 'SH3': 15, 'SH4': 16, 'SH5': 17,
    'SH6': 18, 'SH7': 19, 'SH8': 20, 'SH9': 21,
    # Afrique du Nord
    'AF_STEPPE': 22, 'AF_STEPPE_DENSE': 23, 'AF_ARGAN': 24,
    'AF_CHENE_LIEGE': 25, 'AF_CEDRE': 26, 'AF_MAQUIS': 27,
    'AF_CEREALES': 28, 'AF_PALMIER': 29, 'AF_TAMARIX': 30,
    'AF_JUJUBIER': 31,
    # Afrique subsaharienne
    'AF_SAHEL_GRASS': 32, 'AF_SAHEL_WOODED': 33, 'AF_SUDAN_GRASS': 34,
    'AF_SUDAN_WOODED': 35, 'AF_MIOMBO': 36, 'AF_MIOMBO_DENSE': 37,
    'AF_MOPANE': 38, 'AF_ACACIA_SAVANNA': 39, 'AF_GRASSLAND_FERTILE': 40,
    'AF_FYNBOS': 41, 'AF_FYNBOS_YOUNG': 42, 'AF_BUSHVELD': 43,
    'AF_BAOBAB': 44, 'AF_FOREST_DRY': 45, 'AF_AFROMONTANE': 46,
    'AF_MANGROVE': 47, 'AF_RANGE_DEGRADED': 48, 'AF_RANGE_INTACT': 49,
}

N_FUEL_MODELS = len(FUEL_MODEL_ENCODING)


def encode_fuel_model(code: str) -> int:
    """Encode un code fuel model en entier."""
    return FUEL_MODEL_ENCODING.get(code, -1)


def decode_fuel_model(idx: int) -> str:
    """Décode un entier en code fuel model."""
    for code, i in FUEL_MODEL_ENCODING.items():
        if i == idx:
            return code
    return "UNKNOWN"


# =============================================================================
# MODÈLE : ATLAS CORRECTOR V3
# =============================================================================

class AtlasCorrectorV3(nn.Module):
    """
    MLP correcteur avec :
    - Embedding pour fuel model (catégoriel)
    - Couches denses [256, 128, 64] avec BatchNorm + Dropout
    - Sortie additive : delta_ros (m/min) + log_var (incertitude)
    """

    def __init__(
        self,
        n_features: int = len(REQUIRED_FEATURES) - 1,  # -1 car fuel_model_code est embedding
        n_fuel_models: int = N_FUEL_MODELS,
        embedding_dim: int = 16,
        hidden_dims: List[int] = None,
        dropout_rate: float = DROPOUT_RATE,
        correction_scale: float = CORRECTION_SCALE,
    ):
        super().__init__()

        if hidden_dims is None:
            hidden_dims = HIDDEN_DIMS

        self.correction_scale = correction_scale
        self.dropout_rate = dropout_rate

        # Embedding pour le fuel model (catégoriel)
        self.fuel_embedding = nn.Embedding(n_fuel_models, embedding_dim)

        # Construction dynamique des couches
        in_dim = n_features + embedding_dim

        layers = []
        for h_dim in hidden_dims:
            layers.append(nn.Linear(in_dim, h_dim))
            layers.append(nn.BatchNorm1d(h_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout_rate))
            in_dim = h_dim

        self.backbone = nn.Sequential(*layers)

        # Tête de sortie : delta_ros + log_var
        self.head = nn.Linear(in_dim, 2)

        self._init_weights()

    def _init_weights(self):
        """Initialisation Xavier pour stabilité."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Embedding):
                nn.init.normal_(m.weight, mean=0.0, std=0.01)

    def forward(self, x_continuous: torch.Tensor, fuel_idx: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x_continuous: [batch, n_continuous_features] — features numériques
            fuel_idx: [batch] — indices des fuel models (entiers)

        Returns:
            [batch, 2] — [delta_ros, log_var]
        """
        # Embedding du fuel model
        fuel_emb = self.fuel_embedding(fuel_idx)  # [batch, embedding_dim]

        # Concaténation
        features = torch.cat([x_continuous, fuel_emb], dim=1)  # [batch, in_dim]

        # Backbone
        h = self.backbone(features)

        # Tête
        out = self.head(h)

        # delta_ros : borné par tanh × scale
        delta_ros = torch.tanh(out[:, 0:1]) * self.correction_scale

        # log_var : incertitude (pas de borne, mais clipping pour stabilité)
        log_var = torch.clamp(out[:, 1:2], -5.0, 5.0)

        return torch.cat([delta_ros, log_var], dim=1)

    def predict_with_uncertainty(
        self,
        x_continuous: torch.Tensor,
        fuel_idx: torch.Tensor,
        n_samples: int = 30,
    ) -> Dict[str, torch.Tensor]:
        """
        MC Dropout : n forward passes avec dropout actif.

        Retourne :
            - delta_mean : correction moyenne (m/min)
            - delta_std : écart-type (incertitude)
            - ros_corrected : ROS_Roth + delta_mean
            - ros_ci_lower/upper : intervalle de confiance 95%
        """
        # BatchNorm nécessite batch_size > 1 en mode train
        # On duplique le batch si nécessaire
        batch_size = x_continuous.shape[0]
        if batch_size == 1:
            x_continuous = x_continuous.repeat(n_samples, 1)
            fuel_idx = fuel_idx.repeat(n_samples)

            self.train()
            with torch.no_grad():
                out = self.forward(x_continuous, fuel_idx)
                deltas = out[:, 0:1].unsqueeze(0)  # [1, n_samples, 1]
        else:
            self.train()  # Force dropout actif
            deltas = []
            for _ in range(n_samples):
                with torch.no_grad():
                    out = self.forward(x_continuous, fuel_idx)
                    deltas.append(out[:, 0:1])
            deltas = torch.stack(deltas, dim=0)  # [n_samples, batch, 1]

        delta_mean = deltas.mean(dim=0)
        delta_std = deltas.std(dim=0)

        # Intervalle de confiance 95%
        ci_lower = delta_mean - 1.96 * delta_std
        ci_upper = delta_mean + 1.96 * delta_std

        return {
            'delta_mean': delta_mean,
            'delta_std': delta_std,
            'ci_lower': ci_lower,
            'ci_upper': ci_upper,
        }


# =============================================================================
# CONSTRUCTION DU VECTEUR DE FEATURES
# =============================================================================

def build_ia_vector(
    temp_c: float,
    rh_percent: float,
    wind_speed_ms: float,
    vpd_kpa: float,
    slope_deg: float,
    slope_pct: float,
    fuel_model_code: str,
    w_total_kg_m2: float,
    w_dead_kg_m2: float,
    w_live_kg_m2: float,
    delta_m: float,
    sigma_m2_m3: float,
    mx_percent: float,
    h_dead_kj_kg: float,
    ros_rothermel: float,
    phi_w: float,
    phi_s: float,
    phi_eff: float,
    beta: float,
    beta_opt: float,
    gamma: float,
    eta_M: float,
    eta_S: float,
    I_R_kW_m2: float,
    xi: float,
    tau_min: float,
    ndvi: float,
    ndwi: float,
    lst_c: float,
    dfmc_percent: float,
) -> Tuple[np.ndarray, int]:
    """
    Construit le vecteur de features pour l'IA corrector v3.

    Retourne :
        - x_continuous : array de features numériques
        - fuel_idx : index du fuel model pour l'embedding

    Lève une exception si une feature est manquante ou NaN.
    """
    features = {
        'temp_c': temp_c,
        'rh_percent': rh_percent,
        'wind_speed_ms': wind_speed_ms,
        'vpd_kpa': vpd_kpa,
        'slope_deg': slope_deg,
        'slope_pct': slope_pct,
        'w_total_kg_m2': w_total_kg_m2,
        'w_dead_kg_m2': w_dead_kg_m2,
        'w_live_kg_m2': w_live_kg_m2,
        'delta_m': delta_m,
        'sigma_m2_m3': sigma_m2_m3,
        'mx_percent': mx_percent,
        'h_dead_kj_kg': h_dead_kj_kg,
        'ros_rothermel': ros_rothermel,
        'phi_w': phi_w,
        'phi_s': phi_s,
        'phi_eff': phi_eff,
        'beta': beta,
        'beta_opt': beta_opt,
        'beta_ratio': beta / beta_opt if beta_opt > 0 else 0.0,
        'gamma': gamma,
        'eta_M': eta_M,
        'eta_S': eta_S,
        'I_R_kW_m2': I_R_kW_m2,
        'xi': xi,
        'tau_min': tau_min,
        'ndvi': ndvi,
        'ndwi': ndwi,
        'lst_c': lst_c,
        'dfmc_percent': dfmc_percent,
    }

    # Vérification : aucune valeur manquante ou NaN
    for name, value in features.items():
        if value is None or (isinstance(value, float) and np.isnan(value)):
            raise ValueError(
                f"Feature manquante ou NaN : '{name}'. "
                f"Impossible de faire une prédiction fiable. "
                f"Fournissez toutes les features requises."
            )

    x_continuous = np.array([features[f] for f in REQUIRED_FEATURES if f != 'fuel_model_code'], 
                            dtype=np.float32)

    # Vérifier la dimension
    if x_continuous.shape[0] != N_CONTINUOUS_FEATURES:
        raise ValueError(
            f"Dimension mismatch : {x_continuous.shape[0]} features fournies, "
            f"{N_CONTINUOUS_FEATURES} attendues. Vérifiez REQUIRED_FEATURES."
        )

    fuel_idx = encode_fuel_model(fuel_model_code)
    if fuel_idx == -1:
        raise ValueError(
            f"Fuel model '{fuel_model_code}' inconnu. "
            f"Codes valides : {list(FUEL_MODEL_ENCODING.keys())}"
        )

    return x_continuous, fuel_idx


# =============================================================================
# FONCTION DE PERTE
# =============================================================================

def corrector_loss(
    prediction: torch.Tensor,
    target_delta: torch.Tensor,
    l2_lambda: float = L2_LAMBDA,
    model: nn.Module = None,
) -> torch.Tensor:
    """
    Perte combinée : MSE + incertitude hétéroscédastique + L2 régularisation.

    prediction : [delta_ros, log_var]
    target_delta : delta_ros vrai (m/min)
    """
    delta_pred = prediction[:, 0]
    log_var = prediction[:, 1]

    # Perte hétéroscédastique (NLL-like mais plus stable)
    # L = 0.5 * exp(-log_var) * (target - pred)² + 0.5 * log_var
    precision = torch.exp(-log_var)
    nll = 0.5 * (precision * (target_delta - delta_pred) ** 2 + log_var)
    loss = nll.mean()

    # Régularisation L2
    if model is not None and l2_lambda > 0:
        l2_reg = sum(p.pow(2.0).sum() for p in model.parameters())
        loss = loss + l2_lambda * l2_reg

    return loss


# =============================================================================
# DATASET LOADER AVEC NORMALISATION
# =============================================================================

class CorrectorDatasetLoader:
    """
    Charge et normalise les données pour le corrector v3.
    StandardScaler OBLIGATOIRE sur les features continues.
    """

    def __init__(self):
        self.scaler = StandardScaler()
        self.is_fitted = False

    def fit(self, X_continuous: np.ndarray):
        """Fit le scaler sur les données d'entraînement."""
        self.scaler.fit(X_continuous)
        self.is_fitted = True
        return self

    def transform(self, X_continuous: np.ndarray) -> np.ndarray:
        """Transforme avec le scaler fitté."""
        if not self.is_fitted:
            raise RuntimeError(
                "Le StandardScaler n'est pas fitté. "
                "Appelez .fit(X_train) avant .transform()."
            )
        return self.scaler.transform(X_continuous)

    def fit_transform(self, X_continuous: np.ndarray) -> np.ndarray:
        """Fit + transform."""
        self.fit(X_continuous)
        return self.transform(X_continuous)

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
# EXEMPLE D'UTILISATION
# =============================================================================

# Calcul automatique du nombre de features continues
N_CONTINUOUS_FEATURES = len([f for f in REQUIRED_FEATURES if f != 'fuel_model_code'])

if __name__ == "__main__":
    print("=" * 70)
    print("AtlasCorrectorV3 — Test de base")
    print("=" * 70)

    # Instanciation
    n_continuous = N_CONTINUOUS_FEATURES
    model = AtlasCorrectorV3(
        n_features=n_continuous,
        n_fuel_models=N_FUEL_MODELS,
        embedding_dim=16,
        hidden_dims=HIDDEN_DIMS,
        dropout_rate=DROPOUT_RATE,
        correction_scale=CORRECTION_SCALE,
    )
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Paramètres : {n_params:,}")
    print(f"Features continues : {n_continuous}")
    print(f"Fuel models : {N_FUEL_MODELS}")
    print(f"Embedding dim : 16")
    print(f"Architecture : {HIDDEN_DIMS}")

    # Forward simple
    batch_size = 4
    x_cont = torch.randn(batch_size, n_continuous)
    fuel_idx = torch.randint(0, N_FUEL_MODELS, (batch_size,))

    out = model(x_cont, fuel_idx)
    print(f"\nSortie [delta_ros, log_var] : {out.shape}")
    print(f"Delta ROS moyen : {out[:, 0].mean().item():.3f} m/min")
    print(f"Delta ROS range : [{out[:, 0].min().item():.3f}, {out[:, 0].max().item():.3f}] m/min")
    print(f"Log-var moyen : {out[:, 1].mean().item():.3f}")

    # MC Dropout
    model.eval()
    uncertainty = model.predict_with_uncertainty(x_cont, fuel_idx, n_samples=30)
    print(f"\nIncertitude moyenne (std) : {uncertainty['delta_std'].mean().item():.4f} m/min")

    # Test build_ia_vector avec feature manquante
    print("\n--- Test gestion erreur ---")
    try:
        x_cont, f_idx = build_ia_vector(
            temp_c=35.0, rh_percent=25.0, wind_speed_ms=5.0, vpd_kpa=2.5,
            slope_deg=10.0, slope_pct=17.6, fuel_model_code="AF_STEPPE",
            w_total_kg_m2=0.42, w_dead_kg_m2=0.30, w_live_kg_m2=0.12,
            delta_m=0.25, sigma_m2_m3=12000, mx_percent=20, h_dead_kj_kg=18600,
            ros_rothermel=5.0, phi_w=2.5, phi_s=0.5, phi_eff=2.8,
            beta=0.0033, beta_opt=0.0042, gamma=16.0,
            eta_M=0.55, eta_S=0.95, I_R_kW_m2=150, xi=0.5, tau_min=0.5,
            ndvi=0.25, ndwi=-0.15, lst_c=48.0, dfmc_percent=13.0,
        )
        print(f"✅ Vecteur construit : shape={x_cont.shape}, fuel_idx={f_idx}")
    except ValueError as e:
        print(f"❌ {e}")

    # Test fuel model inconnu
    print("\n--- Test fuel model inconnu ---")
    try:
        x_cont, f_idx = build_ia_vector(
            temp_c=35.0, rh_percent=25.0, wind_speed_ms=5.0, vpd_kpa=2.5,
            slope_deg=10.0, slope_pct=17.6, fuel_model_code="UNKNOWN_FUEL",
            w_total_kg_m2=0.42, w_dead_kg_m2=0.30, w_live_kg_m2=0.12,
            delta_m=0.25, sigma_m2_m3=12000, mx_percent=20, h_dead_kj_kg=18600,
            ros_rothermel=5.0, phi_w=2.5, phi_s=0.5, phi_eff=2.8,
            beta=0.0033, beta_opt=0.0042, gamma=16.0,
            eta_M=0.55, eta_S=0.95, I_R_kW_m2=150, xi=0.5, tau_min=0.5,
            ndvi=0.25, ndwi=-0.15, lst_c=48.0, dfmc_percent=13.0,
        )
    except ValueError as e:
        print(f"✅ Exception levée correctement : {e}")