"""Unit tests for downstream prediction heads and downstream model."""

import torch
import torch.nn as nn

from sfm.models.heads import DownstreamModel, MultiGasHead
from sfm.models.mae import PatchEmbedding
from sfm.models.transformer import TransformerEncoder


def test_multigas_head_forward():
    """Verify MultiGasHead output shapes and types."""
    batch_size = 4
    input_dim = 64
    hidden_dim = 32
    num_gases = 5

    head = MultiGasHead(input_dim=input_dim, hidden_dim=hidden_dim, num_gases=num_gases, dropout=0.0)
    x = torch.randn(batch_size, input_dim)
    class_logits, conc_preds = head(x)

    assert class_logits.shape == (batch_size, num_gases)
    assert conc_preds.shape == (batch_size, num_gases)
    assert torch.isfinite(class_logits).all()
    assert torch.isfinite(conc_preds).all()


def test_downstream_model_backbone_frozen():
    """Verify that backbone parameters are frozen when freeze_backbone=True."""
    num_channels = 200
    patch_size = 10
    encoder_dim = 32
    num_patches = num_channels // patch_size
    num_gases = 3

    patch_embedding = PatchEmbedding(num_channels, patch_size, encoder_dim)
    pos_embed = nn.Parameter(torch.randn(1, num_patches, encoder_dim) * 0.02)
    encoder = TransformerEncoder(dim=encoder_dim, depth=2, num_heads=2)
    head_config = {"hidden_dim": 16, "dropout": 0.0}

    model = DownstreamModel(
        encoder=encoder,
        patch_embedding=patch_embedding,
        pos_embed=pos_embed,
        head_config=head_config,
        num_gases=num_gases,
        freeze_backbone=True,
    )

    # Check that backbone params do not require grad
    for p in model.encoder.parameters():
        assert not p.requires_grad
    for p in model.patch_embedding.parameters():
        assert not p.requires_grad
    assert not model.pos_embed.requires_grad

    # Check that head parameters require grad
    for p in model.head.parameters():
        assert p.requires_grad


def test_downstream_model_forward():
    """Verify end-to-end forward pass through DownstreamModel."""
    batch_size = 3
    num_channels = 100
    patch_size = 10
    encoder_dim = 32
    num_patches = num_channels // patch_size
    num_gases = 4

    patch_embedding = PatchEmbedding(num_channels, patch_size, encoder_dim)
    pos_embed = nn.Parameter(torch.randn(1, num_patches, encoder_dim) * 0.02)
    encoder = TransformerEncoder(dim=encoder_dim, depth=2, num_heads=2)

    model = DownstreamModel(
        encoder=encoder,
        patch_embedding=patch_embedding,
        pos_embed=pos_embed,
        head_config={"hidden_dim": 16, "dropout": 0.0},
        num_gases=num_gases,
    )

    x = torch.randn(batch_size, num_channels)
    class_logits, conc_preds = model(x)

    assert class_logits.shape == (batch_size, num_gases)
    assert conc_preds.shape == (batch_size, num_gases)

