"""
Soft Pathology Conditioning Projector.
Maps continuous predicted multi-label pathology probability vectors
into the continuous hidden embedding space of the language decoder.
"""

import torch
import torch.nn as nn


class SoftConditioningProjector(nn.Module):
    """
    MLP projector mapping disease probability vectors (14 classes)
    into a continuous prefix token representation in Qwen's hidden space.
    """

    def __init__(self, num_classes: int = 14, hidden_dim: int = 256, qwen_dim: int = 2048, dropout: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(num_classes, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, qwen_dim),
            nn.LayerNorm(qwen_dim),
        )

    def forward(self, disease_probs: torch.Tensor) -> torch.Tensor:
        """
        Args:
            disease_probs: Tensor of shape (Batch, NumClasses)
        Returns:
            prefix_token: Tensor of shape (Batch, 1, QwenDim)
        """
        token = self.net(disease_probs)
        return token.unsqueeze(1)
