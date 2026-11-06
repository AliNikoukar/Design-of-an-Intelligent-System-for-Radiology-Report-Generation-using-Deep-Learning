"""
Perceiver Resampler module for visual token sequence compression.
Compresses dense visual feature sequences (128 patches) into a fixed budget
of learnable latent queries (32 queries) via multi-head cross-attention.
"""

import torch
import torch.nn as nn


class PerceiverResampler(nn.Module):
    """
    Perceiver Resampler:
    Projects visual patch tokens into a lower-dimensional latent sequence
    using learnable query vectors and cross-attention transformer decoder layers.
    """

    def __init__(
        self,
        visual_dim: int = 768,
        qwen_dim: int = 2048,
        num_queries: int = 32,
        num_heads: int = 8,
        num_layers: int = 2,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.num_queries = num_queries
        self.qwen_dim = qwen_dim

        # Input projection & normalization for visual features
        self.input_proj = nn.Linear(visual_dim, qwen_dim)
        self.input_norm = nn.LayerNorm(qwen_dim)

        # Learnable latent queries
        self.queries = nn.Parameter(torch.randn(1, num_queries, qwen_dim) * 0.02)

        # Transformer cross-attention decoder stack
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=qwen_dim,
            nhead=num_heads,
            dim_feedforward=qwen_dim * 2,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)
        self.output_norm = nn.LayerNorm(qwen_dim)

    def forward(self, visual_tokens: torch.Tensor) -> torch.Tensor:
        """
        Args:
            visual_tokens: Tensor of shape (Batch, NumPatches, VisualDim)
        Returns:
            resampled_tokens: Tensor of shape (Batch, NumQueries, QwenDim)
        """
        B = visual_tokens.size(0)
        memory = self.input_norm(self.input_proj(visual_tokens))
        queries = self.queries.expand(B, -1, -1)
        resampled = self.decoder(tgt=queries, memory=memory)
        return self.output_norm(resampled)
