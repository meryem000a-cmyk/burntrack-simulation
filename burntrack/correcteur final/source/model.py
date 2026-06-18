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
