"""
Spatial Attention Pooling Multi-Label Classifier for CheXpert pathologies.
Computes independent 2D attention heatmaps per disease class over the 16x8 spatial grid.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple


class AttentionPoolClassifier(nn.Module):
    """
    Per-class spatial attention pooling classifier for 14 CheXpert conditions.
    
    Given N spatial tokens (e.g. 128 = 16x8 patches), learns an attention projection
    that assigns attention weights across the 2D grid for each pathology class independently.
    """

    def __init__(self, visual_dim: int = 768, num_classes: int = 14, hidden_dim: int = 256):
        super().__init__()
        self.num_classes = num_classes
        self.visual_dim = visual_dim

        # Attention projection: maps visual tokens to per-class attention scores
        self.attn_proj = nn.Sequential(
            nn.Linear(visual_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, num_classes),
        )

        # Class-specific diagnostic prediction heads
        self.heads = nn.ModuleList([
            nn.Sequential(
                nn.Linear(visual_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.GELU(),
                nn.Dropout(0.2),
                nn.Linear(hidden_dim, 1),
            )
            for _ in range(num_classes)
        ])

    def forward(self, spatial_tokens: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            spatial_tokens: Tensor of shape (Batch, NumTokens, VisualDim)
        Returns:
            logits: Tensor of shape (Batch, NumClasses)
            attn_weights: Tensor of shape (Batch, NumTokens, NumClasses)
        """
        # (Batch, NumTokens, NumClasses)
        attn_scores = self.attn_proj(spatial_tokens)
        attn_weights = F.softmax(attn_scores, dim=1)

        # Weighted spatial aggregation: (Batch, NumClasses, VisualDim)
        pooled = torch.einsum("bpd,bpc->bcd", spatial_tokens, attn_weights)

        # Independent classification per class
        logits_list = [self.heads[c](pooled[:, c, :]) for c in range(self.num_classes)]
        logits = torch.cat(logits_list, dim=-1)

        return logits, attn_weights
