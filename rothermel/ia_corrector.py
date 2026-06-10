"""
ia_corrector.py
=================
Réseau de neurones correcteur pour le pipeline BurnTrack.

Ce module implémente l'IA légère qui corrige les biais résiduels
du modèle de Rothermel en apprenant sur des données terrain.

Architecture : MLP (Multi-Layer Perceptron) avec contraintes physiques.
Entrée : features multi-sources (Rothermel + robot + satellite)
Sortie : facteur de correction ∈ [-0.5, 0.5] et flags de risque

ROS_corrigé = ROS_base × (1 + correction_factor)

Contraintes physiques :
- |correction_factor| < 0.5 (l'IA ne peut pas tout changer)
- ROS_corrigé > 0 (pas de ROS négatif)
- Pénalisation des corrections extrêmes dans la loss

Sources :
- Rothermel, R.C. (1972)
- Cruz et al. (2015) - Correction models for fire spread
"""

import numpy as np
import torch
import torch.nn as nn
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
import warnings


# =============================================================================
# FEATURES ATTENDUES PAR L'IA
# =============================================================================

FEATURE_NAMES_IA: List[str] = [
    # Variables Rothermel (5 features)
    'ros_base',          # ROS calculé par Rothermel (m/min)
    'I_R',               # Intensité de réaction (kW/m²)
    'phi_w',             # Coefficient de vent
    'phi_s',             # Coefficient de pente
    'beta',              # Packing ratio
    
    # Variables robot (5 features)
    'temp_air',          # Température air (°C)
    'rh',                # Humidité relative (%)
    'vpd',               # Déficit pression vapeur (kPa)
    'dfmc',              # Dead Fuel Moisture Content (%)
    'wind_speed',        # Vent robot (m/s)
    
    # Variables satellite (3 features)
    'ndvi',              # NDVI [-1, 1]
    'ndwi',              # NDWI [-1, 1]
    'lst',               # Land Surface Temperature (°C)
    
    # Variables météo API (2 features)
    'temp_2m',           # Température 2m (°C)
    'wind_10m',          # Vent 10m (m/s)
    
    # Features fusion (5 features)
    'delta_t_surf_air',  # Différence T° surface - air (°C)
    'wind_ratio',        # Ratio vent robot / vent météo
    'stress_index',      # Indice de stress végétal [0, 1]
    'ndvi_anomaly',      # Anomalie NDVI (z-score)
    'danger_proxy',      # Proxy de danger [0, 1]
]

N_FEATURES = len(FEATURE_NAMES_IA)


# =============================================================================
# MODÈLE DE NEURONES
# =============================================================================

class AtlasCorrector(nn.Module):
    """
    Réseau de neurones correcteur pour Rothermel.
    
    Architecture :
    - Backbone : MLP avec BatchNorm et Dropout
    - Head correction : sortie ∈ [-0.5, 0.5] via Tanh
    - Head flags : probabilités de risques [0, 1]
    
    ROS_corrigé = ROS_base × (1 + correction_factor)
    """
    
    def __init__(self, n_features: int = N_FEATURES, 
                 hidden_dims: List[int] = [64, 32],
                 dropout: float = 0.2,
                 correction_scale: float = 0.5):
        """
        Args:
            n_features: Nombre de features d'entrée
            hidden_dims: Dimensions des couches cachées
            dropout: Taux de dropout
            correction_scale: Échelle de correction (Tanh × scale)
        """
        super().__init__()
        
        self.n_features = n_features
        self.correction_scale = correction_scale
        
        # Construction du backbone dynamiquement
        layers = []
        prev_dim = n_features
        
        for dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, dim),
                nn.BatchNorm1d(dim),
                nn.ReLU(),
                nn.Dropout(dropout),
            ])
            prev_dim = dim
        
        self.backbone = nn.Sequential(*layers)
        
        # Head de correction : [-1, 1] → [-scale, +scale]
        self.correction_head = nn.Sequential(
            nn.Linear(prev_dim, 16),
            nn.ReLU(),
            nn.Linear(16, 1),
            nn.Tanh()  # [-1, 1]
        )
        
        # Head de flags : probabilités de risques
        self.flag_head = nn.Sequential(
            nn.Linear(prev_dim, 16),
            nn.ReLU(),
            nn.Linear(16, 3),
            nn.Sigmoid()
        )
        
        # Initialisation des poids
        self._init_weights()
    
    def _init_weights(self):
        """Initialisation Xavier pour stabilité."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
    
    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Forward pass.
        
        Args:
            x: Tensor de shape (batch_size, n_features)
        
        Returns:
            Dict avec correction_factor et flags
        """
        features = self.backbone(x)
        
        # Correction factor : [-scale, +scale]
        correction_raw = self.correction_head(features)  # [-1, 1]
        correction = correction_raw * self.correction_scale  # [-0.5, 0.5]
        
        # Flags de risque : [0, 1]
        flags = self.flag_head(features)
        
        return {
            'correction_factor': correction,           # [-0.5, 0.5]
            'flag_crown': flags[:, 0:1],              # P(feu de cime)
            'flag_spotting': flags[:, 1:2],           # P(brandons)
            'flag_extreme': flags[:, 2:3],            # P(comportement extrême)
        }
    
    def predict_ros(self, features: torch.Tensor, ros_base: torch.Tensor) -> torch.Tensor:
        """
        Prédit le ROS corrigé à partir des features et du ROS de base.
        
        Args:
            features: Features d'entrée
            ros_base: ROS calculé par Rothermel
        
        Returns:
            ROS corrigé
        """
        with torch.no_grad():
            out = self.forward(features)
            correction = out['correction_factor']
            ros_corrected = ros_base * (1.0 + correction.squeeze())
            return torch.clamp(ros_corrected, min=0.0)
    
    def count_parameters(self) -> int:
        """Nombre total de paramètres entraînables."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# =============================================================================
# FONCTION DE LOSS PHYSIQUE
# =============================================================================

class PhysicsConstrainedLoss(nn.Module):
    """
    Loss avec contraintes physiques.
    
    Composantes :
    1. MSE entre ROS prédit et ROS observé
    2. Pénalité si correction trop extrême (|correction| > 0.4)
    3. Pénalité si ROS négatif
    4. Pénalité si ratio ROS_corrigé/ROS_base > 3 ou < 0.3
    """
    
    def __init__(self, lambda_extreme: float = 0.1,
                 lambda_negative: float = 1.0,
                 lambda_ratio: float = 0.1):
        super().__init__()
        self.mse = nn.MSELoss()
        self.lambda_extreme = lambda_extreme
        self.lambda_negative = lambda_negative
        self.lambda_ratio = lambda_ratio
    
    def forward(self, pred_ros: torch.Tensor, true_ros: torch.Tensor,
                ros_base: torch.Tensor, correction: torch.Tensor) -> torch.Tensor:
        """
        Calcule la loss physique.
        
        Args:
            pred_ros: ROS prédit par l'IA
            true_ros: ROS observé sur le terrain
            ros_base: ROS calculé par Rothermel
            correction: Facteur de correction
        
        Returns:
            Loss scalaire
        """
        # 1. MSE de base
        loss = self.mse(pred_ros, true_ros)
        
        # 2. Pénalité correction extrême (hors [-0.4, 0.4])
        extreme_penalty = torch.mean(
            torch.relu(torch.abs(correction.squeeze()) - 0.4)
        )
        loss = loss + self.lambda_extreme * extreme_penalty
        
        # 3. Pénalité ROS négatif
        negative_penalty = torch.mean(torch.relu(-pred_ros))
        loss = loss + self.lambda_negative * negative_penalty
        
        # 4. Pénalité ratio extrême
        ratio = pred_ros / (ros_base + 1e-6)
        ratio_penalty = torch.mean(
            torch.relu(ratio - 3.0) + torch.relu(0.3 - ratio)
        )
        loss = loss + self.lambda_ratio * ratio_penalty
        
        return loss


# =============================================================================
# CLASSE D'ENTRAÎNEMENT
# =============================================================================

@dataclass
class TrainingConfig:
    """Configuration d'entraînement."""
    lr: float = 1e-3
    weight_decay: float = 1e-5
    batch_size: int = 32
    epochs: int = 200
    patience: int = 20
    grad_clip: float = 1.0
    val_split: float = 0.2
    random_seed: int = 42


class AtlasTrainer:
    """
    Entraînement de l'IA correctrice.
    
    Gère :
    - Préparation des données (train/val split)
    - Entraînement avec early stopping
    - Sauvegarde du meilleur modèle
    - Historique des métriques
    """
    
    def __init__(self, model: AtlasCorrector, config: Optional[TrainingConfig] = None):
        self.model = model
        self.config = config or TrainingConfig()
        self.criterion = PhysicsConstrainedLoss()
        self.optimizer = torch.optim.Adam(
            model.parameters(),
            lr=self.config.lr,
            weight_decay=self.config.weight_decay
        )
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, patience=10, factor=0.5, verbose=True
        )
        self.history = {
            'train_loss': [],
            'val_loss': [],
            'val_mae': [],
            'best_epoch': 0,
        }
        self.best_val_loss = float('inf')
    
    def prepare_data(self, X: np.ndarray, y: np.ndarray, 
                   ros_base: np.ndarray) -> Tuple:
        """
        Prépare les datasets train/val.
        
        Args:
            X: Features (n_samples, n_features)
            y: ROS observé (n_samples,)
            ros_base: ROS Rothermel (n_samples,)
        
        Returns:
            (train_loader, val_loader)
        """
        from sklearn.model_selection import train_test_split
        
        # Split train/val
        X_train, X_val, y_train, y_val, ros_train, ros_val = train_test_split(
            X, y, ros_base, 
            test_size=self.config.val_split,
            random_state=self.config.random_seed
        )
        
        # Conversion en tensors
        train_ds = torch.utils.data.TensorDataset(
            torch.tensor(X_train, dtype=torch.float32),
            torch.tensor(y_train, dtype=torch.float32),
            torch.tensor(ros_train, dtype=torch.float32)
        )
        val_ds = torch.utils.data.TensorDataset(
            torch.tensor(X_val, dtype=torch.float32),
            torch.tensor(y_val, dtype=torch.float32),
            torch.tensor(ros_val, dtype=torch.float32)
        )
        
        train_loader = torch.utils.data.DataLoader(
            train_ds, batch_size=self.config.batch_size, shuffle=True
        )
        val_loader = torch.utils.data.DataLoader(
            val_ds, batch_size=self.config.batch_size
        )
        
        return train_loader, val_loader
    
    def train_epoch(self, train_loader: torch.utils.data.DataLoader) -> float:
        """Entraîne une époque."""
        self.model.train()
        total_loss = 0.0
        
        for x_batch, y_batch, ros_base_batch in train_loader:
            self.optimizer.zero_grad()
            
            # Forward
            out = self.model(x_batch)
            correction = out['correction_factor'].squeeze()
            
            # ROS prédit
            pred_ros = ros_base_batch * (1.0 + correction)
            pred_ros = torch.clamp(pred_ros, min=0.0)
            
            # Loss
            loss = self.criterion(pred_ros, y_batch, ros_base_batch, correction)
            
            # Backward
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(), self.config.grad_clip
            )
            self.optimizer.step()
            
            total_loss += loss.item()
        
        return total_loss / len(train_loader)
    
    def validate(self, val_loader: torch.utils.data.DataLoader) -> Tuple[float, float]:
        """Valide sur le dataset de validation."""
        self.model.eval()
        total_loss = 0.0
        total_mae = 0.0
        
        with torch.no_grad():
            for x_batch, y_batch, ros_base_batch in val_loader:
                out = self.model(x_batch)
                correction = out['correction_factor'].squeeze()
                
                pred_ros = ros_base_batch * (1.0 + correction)
                pred_ros = torch.clamp(pred_ros, min=0.0)
                
                loss = self.criterion(pred_ros, y_batch, ros_base_batch, correction)
                mae = torch.mean(torch.abs(pred_ros - y_batch))
                
                total_loss += loss.item()
                total_mae += mae.item()
        
        return total_loss / len(val_loader), total_mae / len(val_loader)
    
    def fit(self, X: np.ndarray, y: np.ndarray, ros_base: np.ndarray,
            save_path: str = 'atlas_corrector_best.pt') -> Dict:
        """
        Entraîne le modèle complet.
        
        Args:
            X: Features (n_samples, n_features)
            y: ROS observé (n_samples,)
            ros_base: ROS Rothermel (n_samples,)
            save_path: Chemin de sauvegarde du meilleur modèle
        
        Returns:
            Historique d'entraînement
        """
        train_loader, val_loader = self.prepare_data(X, y, ros_base)
        
        print(f"Entraînement : {len(train_loader.dataset)} échantillons")
        print(f"Validation : {len(val_loader.dataset)} échantillons")
        print(f"Paramètres du modèle : {self.model.count_parameters()}")
        
        patience_counter = 0
        
        for epoch in range(self.config.epochs):
            train_loss = self.train_epoch(train_loader)
            val_loss, val_mae = self.validate(val_loader)
            
            self.history['train_loss'].append(train_loss)
            self.history['val_loss'].append(val_loss)
            self.history['val_mae'].append(val_mae)
            
            self.scheduler.step(val_loss)
            
            # Early stopping
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self.history['best_epoch'] = epoch
                torch.save(self.model.state_dict(), save_path)
                patience_counter = 0
            else:
                patience_counter += 1
            
            if epoch % 10 == 0:
                print(f"Epoch {epoch:3d} | "
                      f"train_loss={train_loss:.4f} | "
                      f"val_loss={val_loss:.4f} | "
                      f"val_mae={val_mae:.4f}")
            
            if patience_counter >= self.config.patience:
                print(f"Early stopping à l'époque {epoch}")
                break
        
        print(f"\nMeilleur modèle (époque {self.history['best_epoch']}) : "
              f"val_loss={self.best_val_loss:.4f}")
        print(f"Modèle sauvegardé : {save_path}")
        
        return self.history
    
    def load_best(self, path: str = 'atlas_corrector_best.pt'):
        """Charge le meilleur modèle sauvegardé."""
        self.model.load_state_dict(torch.load(path))
        self.model.eval()


# =============================================================================
# UTILITAIRES
# =============================================================================

def prepare_features_for_ia(rothermel_output: Dict, 
                            features_engineering: Dict) -> np.ndarray:
    """
    Construit le vecteur de features pour l'IA à partir des sorties
    Rothermel et des features engineering.
    
    Args:
        rothermel_output: Sortie du moteur Rothermel
        features_engineering: Sortie du features engineering
    
    Returns:
        Vecteur de features (n_features,)
    """
    feature_vector = []
    
    for name in FEATURE_NAMES_IA:
        if name in rothermel_output:
            feature_vector.append(rothermel_output[name])
        elif name in features_engineering:
            feature_vector.append(features_engineering[name])
        else:
            warnings.warn(f"Feature '{name}' manquante, valeur 0 utilisée")
            feature_vector.append(0.0)
    
    return np.array(feature_vector, dtype=np.float32)


def create_dummy_dataset(n_samples: int = 1000, 
                         noise_level: float = 0.2) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Crée un dataset dummy pour tester l'entraînement.
    
    Args:
        n_samples: Nombre d'échantillons
        noise_level: Niveau de bruit
    
    Returns:
        (X, y, ros_base)
    """
    np.random.seed(42)
    
    X = np.random.randn(n_samples, N_FEATURES).astype(np.float32)
    
    # ROS base simulé (positif)
    ros_base = np.abs(np.random.randn(n_samples) * 2 + 1).astype(np.float32)
    
    # ROS observé = ROS_base × (1 + bruit) avec bruit ∈ [-0.3, 0.3]
    noise = np.random.randn(n_samples) * noise_level
    noise = np.clip(noise, -0.3, 0.3)
    y = ros_base * (1.0 + noise)
    y = np.maximum(y, 0.1)
    
    return X, y, ros_base


# =============================================================================
# EXEMPLE D'UTILISATION
# =============================================================================

if __name__ == "__main__":
    print("=" * 70)
    print("ATLAS CORRECTOR - TEST")
    print("=" * 70)
    
    # 1. Création du modèle
    model = AtlasCorrector(n_features=N_FEATURES)
    print(f"\nParamètres entraînables : {model.count_parameters()}")
    
    # 2. Test forward
    batch_size = 4
    x_dummy = torch.randn(batch_size, N_FEATURES)
    
    print(f"\nTest forward (batch={batch_size}):")
    out = model(x_dummy)
    
    for key, value in out.items():
        print(f"  {key}: shape={value.shape}, "
              f"min={value.min().item():.3f}, "
              f"max={value.max().item():.3f}")
    
    # 3. Test prédiction ROS
    ros_base = torch.tensor([1.5, 3.0, 0.8, 5.0])
    ros_corrected = model.predict_ros(x_dummy, ros_base)
    print(f"\nROS base:      {ros_base.tolist()}")
    print(f"ROS corrigé:   {ros_corrected.tolist()}")
    
    # 4. Test entraînement avec données dummy
    print("\n" + "=" * 70)
    print("TEST ENTRAÎNEMENT (données dummy)")
    print("=" * 70)
    
    X, y, ros_base_np = create_dummy_dataset(n_samples=500)
    
    trainer = AtlasTrainer(model)
    history = trainer.fit(X, y, ros_base_np, save_path='test_model.pt')
    
    # 5. Évaluation
    model.load_state_dict(torch.load('test_model.pt'))
    model.eval()
    
    with torch.no_grad():
        X_tensor = torch.tensor(X[:10], dtype=torch.float32)
        ros_base_tensor = torch.tensor(ros_base_np[:10], dtype=torch.float32)
        
        out = model(X_tensor)
        correction = out['correction_factor'].squeeze()
        pred_ros = ros_base_tensor * (1.0 + correction)
        
        print(f"\nÉvaluation sur 10 échantillons :")
        print(f"  ROS base moyen : {ros_base_tensor.mean().item():.3f}")
        print(f"  ROS prédit moyen : {pred_ros.mean().item():.3f}")
        print(f"  ROS observé moyen : {y[:10].mean():.3f}")
        print(f"  Correction moyenne : {correction.mean().item():.3f}")