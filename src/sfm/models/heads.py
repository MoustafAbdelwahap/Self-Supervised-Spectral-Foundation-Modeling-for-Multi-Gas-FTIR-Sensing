"""Downstream task prediction heads for multi-gas identification and quantification.

Provides:
- MultiGasHead: Dual classification (BCE logits) and concentration regression (MSE).
- DownstreamModel: Wraps frozen self-supervised backbone with MultiGasHead.
- create_downstream_model: Factory to instantiate and initialize from pre-trained checkpoints.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, Tuple

import torch
import torch.nn as nn

from sfm.models.mae import PatchEmbedding, SpectralMAE
from sfm.models.transformer import TransformerEncoder
from sfm.training.utils import load_checkpoint

logger = logging.getLogger(__name__)


class MultiGasHead(nn.Module):
    """Dual-head prediction module for simultaneous gas classification and quantification.

    Architecture:
        LayerNorm(input_dim) -> Linear(input_dim, hidden_dim) -> GELU -> Dropout
        -> Classifier: Linear(hidden_dim, num_gases) [unnormalized logits]
        -> Regressor:  Linear(hidden_dim, num_gases) [continuous concentration values]
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        num_gases: int,
        dropout: float = 0.1,
    ) -> None:
        """Initialize MultiGasHead.

        Args:
            input_dim: Feature dimension from pooled backbone encoder.
            hidden_dim: Projection dimension before output branches.
            num_gases: Number of candidate target gases.
            dropout: Dropout probability.
        """
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_gases = num_gases

        self.norm = nn.LayerNorm(input_dim)
        self.proj = nn.Linear(input_dim, hidden_dim)
        self.act = nn.GELU()
        self.dropout = nn.Dropout(dropout)

        # Multi-label classification branch (outputs raw logits for BCEWithLogitsLoss)
        self.classifier = nn.Linear(hidden_dim, num_gases)
        # Multi-target concentration regression branch (raw concentration values for MSELoss)
        self.regressor = nn.Linear(hidden_dim, num_gases)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass through downstream prediction head.

        Args:
            x: Mean-pooled spectral embedding tensor of shape (batch_size, input_dim).

        Returns:
            Tuple of:
            - class_logits: Shape (batch_size, num_gases)
            - conc_preds:   Shape (batch_size, num_gases)
        """
        feat = self.norm(x)
        feat = self.proj(feat)
        feat = self.act(feat)
        feat = self.dropout(feat)

        class_logits = self.classifier(feat)
        conc_preds = self.regressor(feat)
        return class_logits, conc_preds


class DownstreamModel(nn.Module):
    """End-to-end downstream model combining frozen pre-trained backbone and MultiGasHead.

    Processes raw absorption spectra through patch embedding, positional encoding,
    transformer encoder layers, global average pooling across patches, and the
    dual classification/regression prediction head.
    """

    def __init__(
        self,
        encoder: TransformerEncoder,
        patch_embedding: PatchEmbedding,
        pos_embed: nn.Parameter,
        head_config: Dict[str, Any],
        num_gases: int,
        freeze_backbone: bool = True,
    ) -> None:
        """Initialize DownstreamModel.

        Args:
            encoder: Pre-trained TransformerEncoder.
            patch_embedding: Pre-trained PatchEmbedding layer.
            pos_embed: Pre-trained positional embedding parameter tensor.
            head_config: Dictionary containing head hyperparameters (hidden_dim, dropout).
            num_gases: Number of target gases to predict.
            freeze_backbone: Whether to lock backbone weights during fine-tuning.
        """
        super().__init__()
        self.encoder = encoder
        self.patch_embedding = patch_embedding
        self.pos_embed = pos_embed
        self.num_gases = num_gases
        self.freeze_backbone = freeze_backbone
        self.head_config = head_config

        if freeze_backbone:
            for p in self.encoder.parameters():
                p.requires_grad = False
            for p in self.patch_embedding.parameters():
                p.requires_grad = False
            self.pos_embed.requires_grad = False
            logger.info("Frozen pre-trained transformer encoder, patch embedding, and positional embeddings.")

        encoder_dim = getattr(encoder, "dim", encoder.norm.normalized_shape[0])
        hidden_dim = head_config.get("hidden_dim", 128)
        dropout = head_config.get("dropout", 0.1)

        self.head = MultiGasHead(
            input_dim=encoder_dim,
            hidden_dim=hidden_dim,
            num_gases=num_gases,
            dropout=dropout,
        )

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass through downstream model.

        Args:
            x: Input spectra tensor of shape (batch_size, num_channels).

        Returns:
            Tuple of:
            - class_logits: Shape (batch_size, num_gases)
            - conc_preds:   Shape (batch_size, num_gases)
        """
        # 1. Patchify and add positional embeddings
        tokens = self.patch_embedding(x) + self.pos_embed
        # 2. Extract latent representations via encoder
        encoded = self.encoder(tokens)  # (batch_size, num_patches, encoder_dim)
        # 3. Global mean pooling across sequence / patch dimension
        pooled = encoded.mean(dim=1)    # (batch_size, encoder_dim)
        # 4. Predict classification logits and concentration values
        return self.head(pooled)


def create_downstream_model(
    pretrained_checkpoint_path: str,
    head_config: Dict[str, Any],
    num_gases: int,
    device: torch.device,
    freeze_backbone: bool = True,
) -> DownstreamModel:
    """Create a DownstreamModel from a pre-trained MAE or downstream checkpoint.

    Args:
        pretrained_checkpoint_path: Path to .pt checkpoint file.
        head_config: Dictionary containing head configuration (hidden_dim, dropout).
        num_gases: Number of target gases.
        device: Device on which to allocate model tensors.
        freeze_backbone: Whether to freeze the feature extractor backbone.

    Returns:
        Instantiated and weighted DownstreamModel on target device.
    """
    ckpt = load_checkpoint(pretrained_checkpoint_path, device=device)
    state_dict = ckpt.get("model_state_dict", ckpt)
    cfg = ckpt.get("config", {})
    model_cfg = cfg.get("model", {})

    # Check if checkpoint is already a complete DownstreamModel
    is_downstream_ckpt = any(k.startswith("head.") for k in state_dict.keys())

    # Infer model hyperparameters from state_dict if config missing
    if "patch_embed.proj.weight" in state_dict:
        patch_weight = state_dict["patch_embed.proj.weight"]
        encoder_dim = patch_weight.shape[0]
        patch_size = patch_weight.shape[2]
    elif "patch_embedding.proj.weight" in state_dict:
        patch_weight = state_dict["patch_embedding.proj.weight"]
        encoder_dim = patch_weight.shape[0]
        patch_size = patch_weight.shape[2]
    else:
        encoder_dim = model_cfg.get("encoder_dim", 256)
        patch_size = model_cfg.get("patch_size", 10)

    if "pos_embed" in state_dict:
        num_patches = state_dict["pos_embed"].shape[1]
    else:
        num_patches = model_cfg.get("num_channels", 200) // patch_size

    num_channels = num_patches * patch_size

    # Infer encoder depth
    encoder_layers = [
        int(k.split(".")[2])
        for k in state_dict.keys()
        if (k.startswith("encoder.blocks.") or k.startswith("encoder.layers.")) and k.endswith(".norm1.weight")
    ]
    encoder_depth = max(encoder_layers) + 1 if encoder_layers else model_cfg.get("encoder_layers", 6)
    encoder_heads = model_cfg.get("encoder_heads", 8)

    # Instantiate backbone components
    patch_embedding = PatchEmbedding(
        num_channels=num_channels,
        patch_size=patch_size,
        embed_dim=encoder_dim,
    )
    pos_embed = nn.Parameter(torch.randn(1, num_patches, encoder_dim) * 0.02)
    encoder = TransformerEncoder(
        dim=encoder_dim,
        depth=encoder_depth,
        num_heads=encoder_heads,
    )

    if is_downstream_ckpt:
        if "head.classifier.weight" in state_dict:
            num_gases = state_dict["head.classifier.weight"].shape[0]
        if "head.proj.weight" in state_dict:
            head_config = dict(head_config)
            head_config["hidden_dim"] = state_dict["head.proj.weight"].shape[0]
        # Load directly into downstream architecture
        model = DownstreamModel(
            encoder=encoder,
            patch_embedding=patch_embedding,
            pos_embed=pos_embed,
            head_config=head_config,
            num_gases=num_gases,
            freeze_backbone=freeze_backbone,
        )
        model.load_state_dict(state_dict)
    else:
        # Infer decoder dim and depth if present in state_dict
        if "decoder.norm.weight" in state_dict:
            decoder_dim = state_dict["decoder.norm.weight"].shape[0]
        else:
            decoder_dim = model_cfg.get("decoder_dim", 128)

        dec_layers = [
            int(k.split(".")[2])
            for k in state_dict.keys()
            if (k.startswith("decoder.blocks.") or k.startswith("decoder.layers.")) and k.endswith(".norm1.weight")
        ]
        decoder_depth = max(dec_layers) + 1 if dec_layers else model_cfg.get("decoder_layers", 4)
        decoder_heads = model_cfg.get("decoder_heads", 4)

        # Load from pre-trained SpectralMAE
        mae = SpectralMAE(
            num_channels=num_channels,
            patch_size=patch_size,
            encoder_dim=encoder_dim,
            encoder_depth=encoder_depth,
            encoder_heads=encoder_heads,
            decoder_dim=decoder_dim,
            decoder_depth=decoder_depth,
            decoder_heads=decoder_heads,
            mask_ratio=model_cfg.get("mask_ratio", 0.75),
        )
        mae.load_state_dict(state_dict, strict=False)
        model = DownstreamModel(
            encoder=mae.encoder,
            patch_embedding=mae.patch_embed,
            pos_embed=mae.pos_embed,
            head_config=head_config,
            num_gases=num_gases,
            freeze_backbone=freeze_backbone,
        )

    model.to(device)
    logger.info(
        "Created DownstreamModel with %d gases (freeze_backbone=%s) on %s",
        num_gases,
        freeze_backbone,
        device,
    )
    return model
