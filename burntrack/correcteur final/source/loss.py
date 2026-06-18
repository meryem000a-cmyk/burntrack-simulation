"""
Loss pondérée pour BurnTrack
Les données réelles comptent plus que les synthétiques
"""

import torch
import torch.nn as nn


class WeightedMSELoss(nn.Module):
    """
    MSE pondérée: les données réelles comptent plus que les synthétiques.

    Objectif: éviter que le modèle n'apprenne les biais de la simulation
    au détriment des mesures terrain.

    Formule:
        loss = mean( w_i * (pred_i - target_i)^2 )
        où w_i = weight_real si donnée réelle, weight_synth sinon
    """

    def __init__(self, weight_real: float = 3.0, weight_synth: float = 1.0):
        super().__init__()
        self.weight_real = weight_real
        self.weight_synth = weight_synth

    def forward(self, pred: torch.Tensor, target: torch.Tensor, is_real: torch.Tensor) -> torch.Tensor:
        """
        Args:
            pred: Prédiction du delta (batch,)
            target: Delta cible (batch,)
            is_real: Flag binaire (batch,) - 1 si donnée réelle, 0 si synthétique

        Returns:
            Loss scalaire
        """
        weights = torch.where(
            is_real.bool(),
            torch.tensor(self.weight_real),
            torch.tensor(self.weight_synth)
        )

        squared_error = (pred - target) ** 2
        loss = (weights * squared_error).mean()
        return loss
