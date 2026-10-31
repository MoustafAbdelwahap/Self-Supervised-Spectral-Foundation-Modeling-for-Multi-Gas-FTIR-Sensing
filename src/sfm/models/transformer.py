"""1D Transformer encoder and decoder blocks optimized for spectral sequence modeling."""

import torch
import torch.nn as nn


class TransformerBlock(nn.Module):
    """Pre-LayerNorm Transformer Block with Multi-Head Self-Attention and MLP."""

    def __init__(
        self,
        dim: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        dropout: float = 0.0,
    ) -> None:
        """Initialize TransformerBlock.

        Args:
            dim: Token embedding dimension.
            num_heads: Number of parallel self-attention heads.
            mlp_ratio: Ratio of hidden dimension in feed-forward MLP (default: 4.0).
            dropout: Dropout probability (default: 0.0).
        """
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(
            embed_dim=dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.norm2 = nn.LayerNorm(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(dim, mlp_hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_hidden_dim, dim),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through self-attention and MLP with residual skip connections.

        Args:
            x: Input token tensor of shape (batch_size, seq_len, dim).

        Returns:
            Processed tensor of shape (batch_size, seq_len, dim).
        """
        # Multi-head self-attention with pre-LayerNorm
        normed_x = self.norm1(x)
        attn_out, _ = self.attn(normed_x, normed_x, normed_x)
        x = x + attn_out

        # Feed-forward MLP with pre-LayerNorm
        x = x + self.mlp(self.norm2(x))
        return x


class TransformerEncoder(nn.Module):
    """Sequence of TransformerBlocks followed by final LayerNorm."""

    def __init__(
        self,
        dim: int,
        depth: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        dropout: float = 0.0,
    ) -> None:
        """Initialize TransformerEncoder.

        Args:
            dim: Embedding dimension.
            depth: Number of Transformer blocks.
            num_heads: Number of attention heads.
            mlp_ratio: Hidden dimension expansion ratio for MLP.
            dropout: Dropout rate.
        """
        super().__init__()
        self.dim = dim
        self.depth = depth
        self.num_heads = num_heads
        self.blocks = nn.ModuleList(
            [
                TransformerBlock(
                    dim=dim,
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratio,
                    dropout=dropout,
                )
                for _ in range(depth)
            ]
        )
        self.norm = nn.LayerNorm(dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Process input tokens through all encoder layers.

        Args:
            x: Tensor of shape (batch_size, seq_len, dim).

        Returns:
            Encoded representation of shape (batch_size, seq_len, dim).
        """
        for block in self.blocks:
            x = block(x)
        return self.norm(x)


class TransformerDecoder(nn.Module):
    """Lightweight sequence of TransformerBlocks for reconstruction in MAE."""

    def __init__(
        self,
        dim: int,
        depth: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        dropout: float = 0.0,
    ) -> None:
        """Initialize TransformerDecoder.

        Args:
            dim: Decoder token embedding dimension.
            depth: Number of decoder layers.
            num_heads: Number of attention heads.
            mlp_ratio: MLP inner expansion ratio.
            dropout: Dropout rate.
        """
        super().__init__()
        self.dim = dim
        self.depth = depth
        self.num_heads = num_heads
        self.blocks = nn.ModuleList(
            [
                TransformerBlock(
                    dim=dim,
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratio,
                    dropout=dropout,
                )
                for _ in range(depth)
            ]
        )
        self.norm = nn.LayerNorm(dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Process sequence through all decoder layers.

        Args:
            x: Sequence tensor of shape (batch_size, seq_len, dim).

        Returns:
            Decoded tokens of shape (batch_size, seq_len, dim).
        """
        for block in self.blocks:
            x = block(x)
        return self.norm(x)

