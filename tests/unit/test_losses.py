"""Unit tests for sfm.models.losses module."""

import math
import torch
from sfm.models.losses import DownstreamLoss, MAELoss, mse_loss, spectral_angle_mapper


def test_mse_loss_zero():
    """Identical tensors must produce zero MSE loss."""
    a = torch.tensor([1.0, 2.0, 3.0])
    loss = mse_loss(a, a)
    torch.testing.assert_close(loss, torch.tensor(0.0), atol=1e-5, rtol=1e-5)


def test_mse_loss_positive():
    """Different tensors must produce strictly positive MSE loss."""
    a = torch.tensor([1.0, 2.0, 3.0])
    b = torch.tensor([1.0, 2.0, 4.0])
    loss = mse_loss(a, b)
    expected = torch.tensor(1.0 / 3.0)
    torch.testing.assert_close(loss, expected, atol=1e-5, rtol=1e-5)


def test_sam_loss_identical():
    """Identical or linearly scaled vectors must produce zero Spectral Angle Mapper loss."""
    a = torch.tensor([[1.0, 2.0, 3.0]])
    b = torch.tensor([[2.0, 4.0, 6.0]])  # Same direction
    loss = spectral_angle_mapper(a, b)
    torch.testing.assert_close(loss, torch.tensor(0.0), atol=1e-4, rtol=1e-4)


def test_sam_loss_orthogonal():
    """Orthogonal vectors must produce SAM loss equal to pi / 2 radians."""
    a = torch.tensor([[1.0, 0.0]])
    b = torch.tensor([[0.0, 1.0]])
    loss = spectral_angle_mapper(a, b)
    expected = torch.tensor(math.pi / 2.0)
    torch.testing.assert_close(loss, expected, atol=1e-4, rtol=1e-4)


def test_mae_loss_combines_mse_and_sam():
    """MAELoss must accurately compute and combine MSE and SAM on masked patches only."""
    # (batch=1, num_patches=2, patch_size=2)
    target = torch.tensor([[[1.0, 0.0], [0.0, 1.0]]])
    pred = torch.tensor([[[1.0, 0.0], [0.0, 2.0]]])  # patch 0 matches; patch 1 has different scale
    # Mask patch 1 only
    mask = torch.tensor([[False, True]])

    loss_fn = MAELoss(alpha=0.5)
    loss = loss_fn(pred, target, mask)

    # Patch 1: pred=[0, 2], target=[0, 1]
    # MSE = (0^2 + 1^2)/2 = 0.5
    # SAM = angle between [0, 2] and [0, 1] = 0.0
    # Expected total = 0.5 + 0.5 * 0.0 = 0.5
    expected = torch.tensor(0.5)
    torch.testing.assert_close(loss, expected, atol=1e-4, rtol=1e-4)


def test_downstream_loss():
    """DownstreamLoss must return sum of BCEWithLogits and MSE."""
    loss_fn = DownstreamLoss()

    class_pred = torch.tensor([[0.0, 0.0]])
    class_target = torch.tensor([[1.0, 0.0]])
    conc_pred = torch.tensor([[0.5, 0.5]])
    conc_target = torch.tensor([[0.5, 0.5]])

    total_loss = loss_fn(class_pred, class_target, conc_pred, conc_target)

    # conc MSE is 0.0
    # BCE for logit 0.0 with target 1.0 is -ln(sigmoid(0)) = ln(2) = 0.693147
    # BCE for logit 0.0 with target 0.0 is -ln(1 - sigmoid(0)) = ln(2) = 0.693147
    # Mean BCE = ln(2) ~ 0.693147
    expected_bce = math.log(2.0)
    torch.testing.assert_close(total_loss, torch.tensor(expected_bce), atol=1e-4, rtol=1e-4)

