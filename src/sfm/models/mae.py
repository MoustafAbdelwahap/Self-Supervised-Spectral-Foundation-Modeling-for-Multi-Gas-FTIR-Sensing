"""Masked Spectral Autoencoder (MAE) model for self-supervised FTIR representation learning."""

from typing import Tuple
import torch
import torch.nn as nn
from sfm.models.transformer import TransformerDecoder, TransformerEncoder


class PatchEmbedding(nn.Module):
    """Segment 1D continuous spectra into non-overlapping patch embeddings."""

    def __init__(self, num_channels: int, patch_size: int, embed_dim: int) -> None:
        """Initialize PatchEmbedding.

        Args:
            num_channels: Total spectral channels.
            patch_size: Channel length of each patch.
            embed_dim: Target embedding dimension.
        """
        super().__init__()
        self.num_channels = num_channels
        self.patch_size = patch_size
        self.num_patches = num_channels // patch_size
        self.proj = nn.Conv1d(
            in_channels=1,
            out_channels=embed_dim,
            kernel_size=patch_size,
            stride=patch_size,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Project spectra into patch token sequences.

        Args:
            x: Input spectra tensor of shape (batch_size, num_channels).

        Returns:
            Tokens of shape (batch_size, num_patches, embed_dim).
        """
        # Reshape to (batch_size, 1, num_channels) for 1D convolution
        x_1d = x.unsqueeze(1)
        # Output of conv1d is (batch_size, embed_dim, num_patches)
        tokens = self.proj(x_1d)
        # Transpose to (batch_size, num_patches, embed_dim)
        return tokens.transpose(1, 2)


class SpectralMAE(nn.Module):
    """Self-Supervised Masked Spectral Autoencoder.

    Encodes a sparse subset of visible spectral patches through an asymmetric
    1D Transformer encoder, then reconstructs the full spectrum using a lightweight
    decoder and learnable mask tokens.
    """

    def __init__(
        self,
        num_channels: int = 200,
        patch_size: int = 10,
        encoder_dim: int = 256,
        encoder_depth: int = 6,
        encoder_heads: int = 8,
        decoder_dim: int = 128,
        decoder_depth: int = 4,
        decoder_heads: int = 4,
        mask_ratio: float = 0.75,
    ) -> None:
        """Initialize SpectralMAE.

        Args:
            num_channels: Length of input spectral vector.
            patch_size: Length of each 1D patch.
            encoder_dim: Embedding dimension of encoder.
            encoder_depth: Number of encoder Transformer blocks.
            encoder_heads: Number of attention heads in encoder.
            decoder_dim: Embedding dimension of decoder.
            decoder_depth: Number of decoder Transformer blocks.
            decoder_heads: Number of attention heads in decoder.
            mask_ratio: Fraction of patches to mask during self-supervised learning.
        """
        super().__init__()
        self.num_channels = num_channels
        self.patch_size = patch_size
        self.num_patches = num_channels // patch_size
        self.encoder_dim = encoder_dim
        self.decoder_dim = decoder_dim
        self.mask_ratio = mask_ratio

        # 1. Patch Embedding & Positional Encoding
        self.patch_embed = PatchEmbedding(num_channels, patch_size, encoder_dim)
        self.pos_embed = nn.Parameter(
            torch.randn(1, self.num_patches, encoder_dim) * 0.02
        )

        # 2. Transformer Encoder
        self.encoder = TransformerEncoder(
            dim=encoder_dim,
            depth=encoder_depth,
            num_heads=encoder_heads,
        )

        # 3. Encoder-to-Decoder projection
        self.enc_to_dec = nn.Linear(encoder_dim, decoder_dim)

        # 4. Decoder & Mask Tokens
        self.mask_token = nn.Parameter(torch.randn(1, 1, decoder_dim) * 0.02)
        self.decoder_pos_embed = nn.Parameter(
            torch.randn(1, self.num_patches, decoder_dim) * 0.02
        )
        self.decoder = TransformerDecoder(
            dim=decoder_dim,
            depth=decoder_depth,
            num_heads=decoder_heads,
        )

        # 5. Patch reconstruction head
        self.pred_head = nn.Linear(decoder_dim, patch_size)

    def forward_encoder(
        self, x: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Patch embed, randomly mask, and encode visible patches only."""
        b = x.shape[0]
        tokens = self.patch_embed(x) + self.pos_embed

        # Random masking per sample via uniform shuffling
        noise = torch.rand(b, self.num_patches, device=x.device)
        ids_shuffle = torch.argsort(noise, dim=1)
        ids_restore = torch.argsort(ids_shuffle, dim=1)

        len_keep = self.num_patches - int(self.num_patches * self.mask_ratio)
        ids_keep = ids_shuffle[:, :len_keep]

        # Gather visible tokens
        visible_tokens = torch.gather(
            tokens, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, self.encoder_dim)
        )
        encoded_visible = self.encoder(visible_tokens)

        # Generate boolean mask: True for masked positions
        mask = torch.ones(b, self.num_patches, dtype=torch.bool, device=x.device)
        mask[:, :len_keep] = False
        mask = torch.gather(mask, dim=1, index=ids_restore)

        return encoded_visible, mask, ids_restore

    def forward_decoder(
        self, encoded_visible: torch.Tensor, ids_restore: torch.Tensor
    ) -> torch.Tensor:
        """Combine encoded visible tokens with mask tokens and decode."""
        b = encoded_visible.shape[0]
        dec_visible = self.enc_to_dec(encoded_visible)

        len_keep = encoded_visible.shape[1]
        num_masked = self.num_patches - len_keep

        # Expand mask tokens
        mask_tokens = self.mask_token.repeat(b, num_masked, 1)

        # Recombine and restore spatial order
        dec_all = torch.cat([dec_visible, mask_tokens], dim=1)
        dec_all = torch.gather(
            dec_all, dim=1, index=ids_restore.unsqueeze(-1).repeat(1, 1, self.decoder_dim)
        )

        # Add decoder positional embeddings
        dec_all = dec_all + self.decoder_pos_embed
        decoded = self.decoder(dec_all)
        return self.pred_head(decoded)

    def forward(
        self, x: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """End-to-end forward pass: embed -> mask -> encode -> decode -> reconstruct.

        Args:
            x: Raw input spectra of shape (batch_size, num_channels).

        Returns:
            Tuple of:
            - pred_patches: (batch_size, num_patches, patch_size)
            - original_patches: (batch_size, num_patches, patch_size)
            - mask: (batch_size, num_patches) boolean where True indicates masked
        """
        b = x.shape[0]
        x_in = x[:, : self.num_patches * self.patch_size]
        original_patches = x_in.reshape(b, self.num_patches, self.patch_size)
        encoded_visible, mask, ids_restore = self.forward_encoder(x_in)
        pred_patches = self.forward_decoder(encoded_visible, ids_restore)
        return pred_patches, original_patches, mask

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Extract continuous latent representations of unmasked spectra for downstream tasks.

        Args:
            x: Input spectra tensor of shape (batch_size, num_channels).

        Returns:
            Latent token sequence of shape (batch_size, num_patches, encoder_dim).
        """
        x_in = x[:, : self.num_patches * self.patch_size]
        tokens = self.patch_embed(x_in) + self.pos_embed
        return self.encoder(tokens)

