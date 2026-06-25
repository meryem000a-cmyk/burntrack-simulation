"""
Physics-informed loss functions for BurnTrack fire behavior correction.

L_total = L_MSE + λ₁·L_positivity + λ₂·L_ros_bound + λ₃·L_monotonic + λ₄·L_uncertainty

Where:
- L_MSE: Mean squared error on delta_ros
- L_positivity: Penalizes ROS_corrected < 0 (fire cannot have negative spread rate)
- L_ros_bound: Penalizes ROS > 200 m/min (physical maximum for surface fire)
- L_monotonic: Penalizes violation of wind monotonicity (d(ROS)/d(wind) >= 0)
- L_uncertainty: Heteroscedastic NLL for calibrated uncertainty estimation
"""

import torch
import torch.nn.functional as F
import numpy as np
from typing import Dict, Tuple, Optional
import logging

logger = logging.getLogger(__name__)


class PhysicsInformedLoss:
    """
    Composite loss function with physics constraints for fire behavior correction.

    Physics constraints enforce known fire behavior properties:
    1. ROS >= 0: Fire cannot spread backwards (negative ROS)
    2. ROS <= 200 m/min: Physical maximum for surface fires (beyond this = crown fire)
    3. d(ROS)/d(wind) >= 0: Higher wind speed increases ROS
    4. Uncertainty: Predicted uncertainty should be calibrated to actual errors

    Args:
        lambda_pos: Weight for positivity constraint (default: 1.0)
        lambda_bound: Weight for upper bound constraint (default: 0.5)
        lambda_mono: Weight for monotonicity constraint (default: 0.3)
        lambda_unc: Weight for uncertainty/heteroscedastic loss (default: 0.1)
        ros_max: Maximum physical ROS in m/min (default: 200.0)
    """

    def __init__(
        self,
        lambda_pos: float = 1.0,
        lambda_bound: float = 0.5,
        lambda_mono: float = 0.3,
        lambda_unc: float = 0.1,
        ros_max: float = 200.0,
    ):
        self.lambda_pos = lambda_pos
        self.lambda_bound = lambda_bound
        self.lambda_mono = lambda_mono
        self.lambda_unc = lambda_unc
        self.ros_max = ros_max

    # ------------------------------------------------------------------
    # Overridable data-fidelity term
    # ------------------------------------------------------------------

    def _data_loss(
        self, delta_pred: torch.Tensor, delta_true: torch.Tensor
    ) -> torch.Tensor:
        """Compute the data-fidelity loss (MSE by default).

        Subclasses override this to swap in a different loss (e.g. Huber).
        Must return a differentiable scalar tensor.
        """
        return F.mse_loss(delta_pred, delta_true)

    # ------------------------------------------------------------------
    # Physics penalty terms
    # ------------------------------------------------------------------

    @staticmethod
    def _positivity_penalty(ros_corrected: torch.Tensor) -> torch.Tensor:
        """Penalize negative corrected ROS (fire cannot spread backwards)."""
        return torch.mean(torch.relu(-ros_corrected))

    def _bound_penalty(self, ros_corrected: torch.Tensor) -> torch.Tensor:
        """Penalize corrected ROS exceeding the physical maximum."""
        return torch.mean(torch.relu(ros_corrected - self.ros_max))

    @staticmethod
    def _monotonicity_penalty(
        phi_w: torch.Tensor,
        ros_corrected: torch.Tensor,
        n_max_pairs: int = 256,
    ) -> torch.Tensor:
        """Penalize violations of wind-monotonicity: d(ROS)/d(wind) >= 0.

        Uses random pairwise comparisons within the batch for efficiency.
        When phi_w_a > phi_w_b we expect ros_a >= ros_b.

        Args:
            phi_w: Wind coefficients [batch].
            ros_corrected: Corrected ROS values [batch].
            n_max_pairs: Maximum number of random pairs to sample.

        Returns:
            Scalar penalty tensor (differentiable).
        """
        n = len(phi_w)
        n_pairs = min(n * 2, n_max_pairs)
        idx_a = torch.randint(0, n, (n_pairs,), device=phi_w.device)
        idx_b = torch.randint(0, n, (n_pairs,), device=phi_w.device)

        phi_diff = phi_w[idx_a] - phi_w[idx_b]  # positive ⟹ a has more wind
        ros_diff = ros_corrected[idx_a] - ros_corrected[idx_b]

        # Penalty when phi_w_a > phi_w_b but ros_a < ros_b
        violations = torch.relu(phi_diff * (-ros_diff))
        return violations.mean()

    @staticmethod
    def _uncertainty_loss(
        delta_pred: torch.Tensor,
        delta_true: torch.Tensor,
        log_var: torch.Tensor,
    ) -> torch.Tensor:
        """Heteroscedastic Gaussian NLL for calibrated uncertainty.

        NLL = 0.5 * exp(-log_var) * (y - ŷ)² + 0.5 * log_var
        """
        precision = torch.exp(-log_var)
        return torch.mean(
            0.5 * precision * (delta_true - delta_pred) ** 2 + 0.5 * log_var
        )

    # ------------------------------------------------------------------
    # Main entry point
    # ------------------------------------------------------------------

    def __call__(
        self,
        delta_pred: torch.Tensor,
        delta_true: torch.Tensor,
        ros_rothermel: torch.Tensor,
        phi_w: Optional[torch.Tensor] = None,
        log_var: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Compute total physics-informed loss.

        Args:
            delta_pred: Predicted delta_ros corrections [batch].
            delta_true: Ground-truth delta_ros values [batch].
            ros_rothermel: Rothermel baseline ROS values [batch].
            phi_w: Wind coefficients for monotonicity check [batch] (optional).
            log_var: Log-variance for heteroscedastic loss [batch] (optional).

        Returns:
            total_loss: Differentiable scalar tensor.
            loss_components: Dict with individual loss terms for logging.
        """
        # Corrected ROS lives on the computation graph
        ros_corrected = ros_rothermel + delta_pred

        # --- Individual loss terms (all differentiable) ---
        data_loss = self._data_loss(delta_pred, delta_true)
        pos_penalty = self._positivity_penalty(ros_corrected)
        bnd_penalty = self._bound_penalty(ros_corrected)

        # Monotonicity (optional)
        if phi_w is not None and len(phi_w) > 1:
            mono_penalty = self._monotonicity_penalty(phi_w, ros_corrected)
        else:
            mono_penalty = torch.tensor(0.0, device=delta_pred.device)

        # Uncertainty (optional)
        if log_var is not None:
            unc_loss = self._uncertainty_loss(delta_pred, delta_true, log_var)
        else:
            unc_loss = torch.tensor(0.0, device=delta_pred.device)

        # --- Weighted sum ---
        total = (
            data_loss
            + self.lambda_pos * pos_penalty
            + self.lambda_bound * bnd_penalty
            + self.lambda_mono * mono_penalty
            + self.lambda_unc * unc_loss
        )

        # Detached component dict for logging / TensorBoard
        components = {
            "total": total.item(),
            "data": data_loss.item(),
            "positivity": pos_penalty.item(),
            "bound": bnd_penalty.item(),
            "monotonic": mono_penalty.item(),
            "uncertainty": unc_loss.item(),
        }

        return total, components


class HeteroscedasticNLLLoss:
    """Negative log-likelihood loss for heteroscedastic regression.

    Predicts both mean and variance, useful for uncertainty estimation.

        NLL = 0.5 * exp(-log_var) * (y - μ)² + 0.5 * log_var
    """

    def __call__(
        self,
        mu: torch.Tensor,
        log_var: torch.Tensor,
        target: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            mu: Predicted means [batch].
            log_var: Predicted log-variance [batch].
            target: Ground-truth values [batch].

        Returns:
            Scalar NLL loss (differentiable).
        """
        precision = torch.exp(-log_var)
        return torch.mean(0.5 * precision * (target - mu) ** 2 + 0.5 * log_var)


class HuberPhysicsLoss(PhysicsInformedLoss):
    """Physics-informed loss using Huber (smooth L1) instead of MSE.

    More robust to outliers in fire behavior data — large delta_ros errors
    (e.g. from mislabelled satellite detections) are down-weighted compared
    to MSE, while small errors are penalised quadratically.

    The physics constraints are inherited from :class:`PhysicsInformedLoss`.

    Args:
        delta_huber: Huber delta (transition point between L1 and L2).
                     Default 1.0 m/min.
        **kwargs: Forwarded to :class:`PhysicsInformedLoss`.
    """

    def __init__(self, delta_huber: float = 1.0, **kwargs):
        super().__init__(**kwargs)
        self.delta_huber = delta_huber

    def _data_loss(
        self, delta_pred: torch.Tensor, delta_true: torch.Tensor
    ) -> torch.Tensor:
        """Huber / smooth-L1 data-fidelity term (differentiable)."""
        return F.smooth_l1_loss(delta_pred, delta_true, beta=self.delta_huber)
