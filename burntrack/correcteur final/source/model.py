"""
Architecture MLP minimal pour BurnTrack
2 couches cachées (64 -> 32), GELU, LayerNorm, Dropout
~4,500 paramètres
"""

import torch
import torch.nn as nn


class BurnTrackMLPMinimal(nn.Module):
    """
    Réseau de neurones minimal pour correction de biais Rothermel.

    Architecture:
        Input (n_features) -> Linear -> LayerNorm -> GELU -> Dropout
                          -> Linear -> LayerNorm -> GELU -> Output(1)

    Paramètres: ~4,500 (léger, pas d'overfit sur 1,840 éch.)
    """

    def __init__(self, n_features: int, hidden1: int = 64, hidden2: int = 32, dropout: float = 0.2):
        super().__init__()

        self.layer1 = nn.Linear(n_features, hidden1)
        self.norm1 = nn.LayerNorm(hidden1)
        self.activation = nn.GELU()
        self.dropout = nn.Dropout(dropout)

        self.layer2 = nn.Linear(hidden1, hidden2)
        self.norm2 = nn.LayerNorm(hidden2)

        self.output = nn.Linear(hidden2, 1)

        self._init_weights()

    def _init_weights(self):
        """Initialisation Xavier pour les couches linéaires."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: Tensor de shape (batch, n_features)

        Returns:
            delta: Tensor de shape (batch,) - correction à ajouter à ROS_r
        """
        x = self.layer1(x)
        x = self.norm1(x)
        x = self.activation(x)
        x = self.dropout(x)

        x = self.layer2(x)
        x = self.norm2(x)
        x = self.activation(x)

        delta = self.output(x).squeeze(-1)
        return delta

    def count_parameters(self) -> int:
        """Compte les paramètres entraînables."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


class BurnTrackAdvancedCorrector(nn.Module):
    """
    Réseau de neurones avancé (PINN / Residual architecture) pour corriger
    la dynamique complexe des feux de brousse africains.

    Architecture:
        - Input (n_features) -> Linear(128) -> LayerNorm -> GELU
        - Residual Block 1: Linear(128) -> LayerNorm -> GELU -> Dropout(0.1) + Skip
        - Residual Block 2: Linear(128) -> LayerNorm -> GELU -> Dropout(0.1) + Skip
        - Linear(64) -> LayerNorm -> GELU
        - Output(1)
    """

    def __init__(self, n_features: int, hidden1: int = 128, hidden2: int = 64, dropout: float = 0.1):
        super().__init__()

        self.input_layer = nn.Sequential(
            nn.Linear(n_features, hidden1),
            nn.LayerNorm(hidden1),
            nn.GELU(),
            nn.Dropout(dropout)
        )

        self.res_block1 = nn.Sequential(
            nn.Linear(hidden1, hidden1),
            nn.LayerNorm(hidden1),
            nn.GELU(),
            nn.Dropout(dropout)
        )

        self.res_block2 = nn.Sequential(
            nn.Linear(hidden1, hidden1),
            nn.LayerNorm(hidden1),
            nn.GELU(),
            nn.Dropout(dropout)
        )

        self.head = nn.Sequential(
            nn.Linear(hidden1, hidden2),
            nn.LayerNorm(hidden2),
            nn.GELU(),
            nn.Linear(hidden2, 1)
        )

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x1 = self.input_layer(x)
        x2 = x1 + self.res_block1(x1)
        x3 = x2 + self.res_block2(x2)
        delta = self.head(x3).squeeze(-1)
        return delta

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


class GatedResidualBlock(nn.Module):
    """Bloc résiduel avec Gated Linear Unit (GLU) pour données tabulaires."""
    def __init__(self, dim: int, dropout: float = 0.1):
        super().__init__()
        self.fc1 = nn.Linear(dim, dim * 2)
        self.norm = nn.LayerNorm(dim)
        self.dropout = nn.Dropout(dropout)
        self.fc2 = nn.Linear(dim, dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        out = self.fc1(x)
        # GLU : scission en deux moitiés
        val, gate = out.chunk(2, dim=-1)
        out = val * torch.sigmoid(gate)
        out = self.norm(out)
        out = self.dropout(out)
        out = self.fc2(out)
        return residual + out


class BurnTrackFTGatedCorrector(nn.Module):
    """
    Réseau tabulaire Gated (inspiré de FT-Transformer et TabNet) pour capturer
    les relations non linéaires complexes et atteindre R² > 0.90.
    """
    def __init__(self, n_features: int, hidden1: int = 256, hidden2: int = 128, dropout: float = 0.05):
        super().__init__()

        self.input_proj = nn.Sequential(
            nn.Linear(n_features, hidden1),
            nn.LayerNorm(hidden1),
            nn.GELU(),
            nn.Dropout(dropout)
        )

        self.block1 = GatedResidualBlock(hidden1, dropout)
        self.block2 = GatedResidualBlock(hidden1, dropout)
        self.block3 = GatedResidualBlock(hidden1, dropout)
        self.block4 = GatedResidualBlock(hidden1, dropout)

        self.head = nn.Sequential(
            nn.Linear(hidden1, hidden2),
            nn.LayerNorm(hidden2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden2, 64),
            nn.LayerNorm(64),
            nn.GELU(),
            nn.Linear(64, 1)
        )

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.input_proj(x)
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        delta = self.head(x).squeeze(-1)
        return delta

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
