"""Loss functions for Self-Supervised Spectral Foundation Modeling."""

import torch
import torch.nn as nn
import torch.nn.functional as F


def spectral_angle_mapper(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Compute Spectral Angle Mapper (SAM) loss between predictions and targets.

    SAM measures the spectral angle between two vectors along the last dimension,
    which is invariant to overall scale/intensity and sensitive to spectral shape/peaks.

    Args:
        pred: Predicted tensor of shape (..., channels).
        target: Target tensor of shape (..., channels).

    Returns:
        Scalar tensor containing the mean spectral angle in radians.
    """
    cos_sim = F.cosine_similarity(pred, target, dim=-1)
    # In float32, parallel vectors may have cos_sim ≈ 0.9999999.
    # Because d(acos)/dx ≈ sqrt(2 * dx) near 1, dx=1e-7 yields ~0.00045 rad.
    # Snap values extremely close to 1.0/-1.0 for numerical precision and exact 0 for identical angles.
    cos_sim = torch.where(cos_sim >= 1.0 - 1e-6, torch.ones_like(cos_sim), cos_sim)
    cos_sim = torch.where(cos_sim <= -1.0 + 1e-6, -torch.ones_like(cos_sim), cos_sim)
    clamped_cos = torch.clamp(cos_sim, -1.0, 1.0)
    sam_rad = torch.acos(clamped_cos)
    return sam_rad.mean()


def mse_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Compute Mean Squared Error (MSE) loss.

    Args:
        pred: Predicted tensor.
        target: Target ground-truth tensor.

    Returns:
        Scalar MSE loss.
    """
    return F.mse_loss(pred, target)


class MAELoss(nn.Module):
    """Composite loss function for Masked Spectral Autoencoders.

    Combines Mean Squared Error (for numerical calibration) and Spectral Angle
    Mapper (SAM, for spectral peak/band shape fidelity) computed specifically
    on masked patches.
    """

    def __init__(self, alpha: float = 0.5) -> None:
        """Initialize MAELoss.

        Args:
            alpha: Weighting factor for the SAM loss term (default: 0.5).
        """
        super().__init__()
        self.alpha = alpha

    def forward(
        self, pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor
    ) -> torch.Tensor:
        """Compute the masked composite loss.

        Args:
            pred: Predicted patch tensor of shape (batch, num_patches, patch_size) or (batch, channels).
            target: Target patch tensor of same shape as pred.
            mask: Boolean mask tensor where True indicates a masked (reconstructed) position.

        Returns:
            Scalar composite loss value (MSE + alpha * SAM).
        """
        pred_masked = pred[mask]
        target_masked = target[mask]

        if pred_masked.numel() == 0:
            return torch.tensor(0.0, device=pred.device, requires_grad=True)

        mse = F.mse_loss(pred_masked, target_masked)

        if pred_masked.dim() >= 2:
            sam = spectral_angle_mapper(pred_masked, target_masked)
        else:
            sam = spectral_angle_mapper(pred_masked.unsqueeze(0), target_masked.unsqueeze(0))

        return mse + self.alpha * sam


class DownstreamLoss(nn.Module):
    """Joint loss for multi-gas identification and concentration quantification.

    Computes Binary Cross-Entropy with Logits for multi-label gas presence,
    plus Mean Squared Error for continuous gas concentration estimates.
    """

    def __init__(self) -> None:
        super().__init__()

    def forward(
        self,
        class_pred: torch.Tensor,
        class_target: torch.Tensor,
        conc_pred: torch.Tensor,
        conc_target: torch.Tensor,
    ) -> torch.Tensor:
        """Compute combined downstream loss.

        Args:
            class_pred: Unnormalized logits for multi-label gas detection (batch, num_gases).
            class_target: Binary ground-truth gas labels (batch, num_gases).
            conc_pred: Predicted concentration values (batch, num_gases).
            conc_target: Target ground-truth concentration values (batch, num_gases).

        Returns:
            Scalar combined loss.
        """
        bce = F.binary_cross_entropy_with_logits(class_pred, class_target.float())
        mse = F.mse_loss(conc_pred, conc_target.float())
        return bce + mse
