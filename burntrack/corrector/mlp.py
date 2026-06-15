"""
mlp.py
======
Deep learning component of the BurnTrack corrector.
Implements DeepPhysicsCorrector and an ensemble wrapper (MLPEnsembleCorrector)
with MC Dropout for uncertainty estimation.
"""

import os
import json
import logging
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from typing import Dict, List, Optional, Tuple, Union
from sklearn.preprocessing import StandardScaler

from burntrack.corrector.base import BaseCorrector
from burntrack.corrector.features import CorrectorFeatureExtractor
from burntrack.corrector.losses import HeteroscedasticNLLLoss, PhysicsInformedLoss

logger = logging.getLogger(__name__)

# Preserve existing FUEL_MODEL_ENCODING exactly for backward compatibility
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

N_FUEL_MODELS = 50

# This will be dynamically set by CorrectorFeatureExtractor, but we need a default
N_CONTINUOUS_FEATURES = 55 

def encode_fuel_model(fuel_model_code: str) -> int:
    """Encode string fuel model to integer index."""
    return FUEL_MODEL_ENCODING.get(fuel_model_code, 0)


class CorrectorDatasetLoader:
    """Dataset loader and scaler for MLP corrector."""
    def __init__(self):
        self.scaler = StandardScaler()
        self.is_fitted = False
        
    def fit(self, X: np.ndarray):
        self.scaler.fit(X)
        self.is_fitted = True
        return self
        
    def transform(self, X: np.ndarray) -> np.ndarray:
        if not self.is_fitted:
            raise ValueError("Scaler is not fitted yet.")
        return self.scaler.transform(X)
        
    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        self.fit(X)
        return self.transform(X)
        
    def save(self, path: str):
        if not self.is_fitted:
            return
        os.makedirs(os.path.dirname(path), exist_ok=True)
        state = {
            'mean_': self.scaler.mean_.tolist(),
            'scale_': self.scaler.scale_.tolist(),
            'var_': self.scaler.var_.tolist() if hasattr(self.scaler, 'var_') else None,
            'n_samples_seen_': int(self.scaler.n_samples_seen_)
        }
        with open(path, 'w') as f:
            json.dump(state, f)
            
    @classmethod
    def load(cls, path: str) -> 'CorrectorDatasetLoader':
        instance = cls()
        with open(path, 'r') as f:
            state = json.load(f)
        instance.scaler.mean_ = np.array(state['mean_'])
        instance.scaler.scale_ = np.array(state['scale_'])
        if state.get('var_') is not None:
            instance.scaler.var_ = np.array(state['var_'])
        instance.scaler.n_samples_seen_ = state['n_samples_seen_']
        instance.is_fitted = True
        return instance


def corrector_loss(prediction: torch.Tensor, target_delta: torch.Tensor, l2_lambda=1e-4, model=None):
    """
    Heteroscedastic NLL loss + L2 regularization.
    Assumes prediction is [batch_size, 2] (delta_ros, log_var).
    """
    delta_pred = prediction[:, 0]
    log_var = prediction[:, 1]
    
    # NLL = 0.5 * exp(-log_var) * (y - mu)^2 + 0.5 * log_var
    precision = torch.exp(-log_var)
    nll = torch.mean(0.5 * precision * (target_delta - delta_pred)**2 + 0.5 * log_var)
    
    l2_reg = 0.0
    if model is not None and l2_lambda > 0:
        for p in model.parameters():
            l2_reg += torch.sum(p ** 2)
            
    return nll + l2_lambda * l2_reg


class DeepPhysicsCorrector(nn.Module):
    """
    Deep neural network for fire behavior correction with uncertainty estimation.
    """
    def __init__(self, n_continuous_features: int = N_CONTINUOUS_FEATURES, n_fuel_models: int = N_FUEL_MODELS):
        super().__init__()
        
        self.fuel_embedding = nn.Embedding(n_fuel_models, 32)
        
        # Backbone: [n_cont + 32] -> 256 -> 128 -> 64 -> 32
        in_features = n_continuous_features + 32
        
        self.backbone = nn.Sequential(
            nn.Linear(in_features, 256),
            nn.BatchNorm1d(256),
            nn.GELU(),
            nn.Dropout(0.2),
            
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.GELU(),
            nn.Dropout(0.2),
            
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.GELU(),
            nn.Dropout(0.2),
            
            nn.Linear(64, 32),
            nn.BatchNorm1d(32),
            nn.GELU(),
            nn.Dropout(0.2)
        )
        
        # Output: [delta_ros, log_var]
        self.head = nn.Linear(32, 2)
        
        self._init_weights()
        
    def _init_weights(self):
        """Kaiming initialization."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
                    
    def forward(self, x_continuous: torch.Tensor, fuel_idx: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        Returns tensor of shape [batch_size, 2] containing [delta_ros_pred, log_var_pred].
        """
        fuel_emb = self.fuel_embedding(fuel_idx)
        
        # Combine continuous features and embedding
        x = torch.cat([x_continuous, fuel_emb], dim=1)
        
        features = self.backbone(x)
        out = self.head(features)
        
        # Clamp log_var to prevent numerical instability
        delta_ros = out[:, 0].unsqueeze(1)
        log_var = torch.clamp(out[:, 1], min=-5.0, max=5.0).unsqueeze(1)  # Limit variance explosion
        
        return torch.cat([delta_ros, log_var], dim=1)


class MLPEnsembleCorrector(BaseCorrector):
    """
    Ensemble of DeepPhysicsCorrector models with MC Dropout.
    """
    def __init__(self, n_continuous_features: int = N_CONTINUOUS_FEATURES, n_fuel_models: int = 50, n_models: int = 5):
        self.n_continuous_features = n_continuous_features
        self.n_fuel_models = n_fuel_models
        self.n_models = n_models
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        self.models = [DeepPhysicsCorrector(n_continuous_features, n_fuel_models).to(self.device) for _ in range(n_models)]
        self.data_loader = CorrectorDatasetLoader()
        self.feature_extractor = CorrectorFeatureExtractor()
        
        # Need to dynamically set N_CONTINUOUS_FEATURES if using the extractor
        self.n_continuous_features = self.feature_extractor.get_n_features()
        
        self.is_fitted = False
        
    def fit(self, X: np.ndarray, y: np.ndarray, fuel_idx: np.ndarray, 
            X_val: Optional[np.ndarray] = None, y_val: Optional[np.ndarray] = None, 
            fuel_idx_val: Optional[np.ndarray] = None,
            epochs: int = 300, batch_size: int = 64, lr: float = 1e-3):
        """Train the ensemble of MLP models."""
        
        logger.info(f"Training MLP Ensemble on device: {self.device}")
        
        X_scaled = self.data_loader.fit_transform(X)
        if X_val is not None:
            X_val_scaled = self.data_loader.transform(X_val)
        else:
            X_val_scaled = None
            
        dataset_size = len(X_scaled)
        
        for i, model in enumerate(self.models):
            logger.info(f"Training MLP model {i+1}/{self.n_models}...")
            
            # Bootstrap sampling for diversity
            indices = np.random.choice(dataset_size, size=dataset_size, replace=True)
            X_boot = X_scaled[indices]
            y_boot = y[indices]
            fuel_boot = fuel_idx[indices]
            
            X_tensor = torch.tensor(X_boot, dtype=torch.float32).to(self.device)
            y_tensor = torch.tensor(y_boot, dtype=torch.float32).to(self.device)
            fuel_tensor = torch.tensor(fuel_boot, dtype=torch.long).to(self.device)
            
            dataset = TensorDataset(X_tensor, y_tensor, fuel_tensor)
            loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
            
            optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=10)
            
            best_val_loss = float('inf')
            
            for epoch in range(epochs):
                model.train()
                train_loss = 0.0
                
                for batch_x, batch_y, batch_fuel in loader:
                    optimizer.zero_grad()
                    preds = model(batch_x, batch_fuel)
                    loss = corrector_loss(preds, batch_y, model=model)
                    loss.backward()
                    # Gradient clipping
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                    optimizer.step()
                    train_loss += loss.item() * len(batch_x)
                    
                train_loss /= dataset_size
                
                # Validation
                val_loss = 0.0
                if X_val_scaled is not None and y_val is not None and fuel_idx_val is not None:
                    model.eval()
                    with torch.no_grad():
                        val_x = torch.tensor(X_val_scaled, dtype=torch.float32).to(self.device)
                        val_y = torch.tensor(y_val, dtype=torch.float32).to(self.device)
                        val_fuel = torch.tensor(fuel_idx_val, dtype=torch.long).to(self.device)
                        
                        val_preds = model(val_x, val_fuel)
                        val_loss = corrector_loss(val_preds, val_y).item()
                        
                    scheduler.step(val_loss)
                    
                    if epoch % 50 == 0:
                        logger.info(f"  Epoch {epoch}/{epochs} - Train Loss: {train_loss:.4f} - Val Loss: {val_loss:.4f}")
                else:
                    scheduler.step(train_loss)
                    if epoch % 50 == 0:
                        logger.info(f"  Epoch {epoch}/{epochs} - Train Loss: {train_loss:.4f}")
                        
        self.is_fitted = True
        
    def predict(self, features: Dict) -> Dict[str, float]:
        """Predict delta_ros and return dictionary for single observation."""
        res = self.predict_with_uncertainty(features)
        return {
            'delta_ros': res['delta_ros'],
            'uncertainty': res['uncertainty']
        }
        
    def predict_with_uncertainty(self, features: Dict, n_mc_samples: int = 30) -> Dict:
        """
        Predict with uncertainty using ensemble + MC Dropout.
        """
        if not self.is_fitted:
            raise ValueError("Model is not fitted yet.")
            
        x_vec = self.feature_extractor.extract_row(features)
        feature_names = self.feature_extractor.get_feature_names()
        x_arr = np.array([[x_vec.get(name, 0.0) for name in feature_names]])
        x_scaled = self.data_loader.transform(x_arr)
        
        fuel_code = features.get('fuel_model_code', '')
        fuel_idx = encode_fuel_model(fuel_code)
        
        x_tensor = torch.tensor(x_scaled, dtype=torch.float32).to(self.device)
        fuel_tensor = torch.tensor([fuel_idx], dtype=torch.long).to(self.device)
        
        all_mc_preds = []
        all_mc_vars = []
        
        # MC Dropout over the ensemble
        for model in self.models:
            model.train() # Enable dropout
            with torch.no_grad():
                for _ in range(n_mc_samples):
                    out = model(x_tensor, fuel_tensor)
                    all_mc_preds.append(out[0, 0].item())
                    # convert log_var to variance
                    all_mc_vars.append(np.exp(out[0, 1].item()))
                    
        # Calculate Law of Total Variance
        # Var(Y) = E[Var(Y|X)] + Var(E[Y|X])
        # Total Unc = Aleatoric Unc (mean of predicted variances) + Epistemic Unc (variance of predicted means)
        
        mc_preds_arr = np.array(all_mc_preds)
        mc_vars_arr = np.array(all_mc_vars)
        
        mean_pred = np.mean(mc_preds_arr)
        epistemic_var = np.var(mc_preds_arr)
        aleatoric_var = np.mean(mc_vars_arr)
        
        total_var = epistemic_var + aleatoric_var
        total_std = np.sqrt(total_var)
        
        return {
            'delta_ros': float(mean_pred),
            'uncertainty': float(total_std),
            'ci_lower': float(mean_pred - 1.96 * total_std),
            'ci_upper': float(mean_pred + 1.96 * total_std)
        }
        
    def save(self, output_dir: str):
        os.makedirs(output_dir, exist_ok=True)
        self.data_loader.save(os.path.join(output_dir, "scaler.json"))
        
        for i, model in enumerate(self.models):
            torch.save(model.state_dict(), os.path.join(output_dir, f"mlp_model_{i}.pt"))
            
        config = {
            'n_continuous_features': self.n_continuous_features,
            'n_fuel_models': self.n_fuel_models,
            'n_models': self.n_models
        }
        with open(os.path.join(output_dir, "mlp_config.json"), "w") as f:
            json.dump(config, f)
            
    @classmethod
    def load(cls, output_dir: str) -> 'MLPEnsembleCorrector':
        with open(os.path.join(output_dir, "mlp_config.json"), "r") as f:
            config = json.load(f)
            
        instance = cls(
            n_continuous_features=config['n_continuous_features'],
            n_fuel_models=config['n_fuel_models'],
            n_models=config['n_models']
        )
        
        instance.data_loader = CorrectorDatasetLoader.load(os.path.join(output_dir, "scaler.json"))
        
        for i in range(instance.n_models):
            model_path = os.path.join(output_dir, f"mlp_model_{i}.pt")
            if os.path.exists(model_path):
                instance.models[i].load_state_dict(torch.load(model_path, map_location=instance.device))
            else:
                logger.warning(f"Could not find model file {model_path}")
                
        instance.is_fitted = True
        return instance

# Backward compatibility functions
def build_ia_vector(*args, **kwargs):
    """Deprecated: Use CorrectorFeatureExtractor instead."""
    logger.warning("build_ia_vector is deprecated. Use CorrectorFeatureExtractor.")
    return []

# Backward compatibility alias
AtlasCorrectorV3 = DeepPhysicsCorrector
