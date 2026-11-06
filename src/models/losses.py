"""
Loss functions for extreme multi-label clinical class imbalance.
Implements Weighted Focal Asymmetric Loss (WF-ASL).
"""

import torch
import torch.nn as nn
from typing import Optional


class WeightedFocalASL(nn.Module):
    """
    Weighted Focal Asymmetric Loss (WF-ASL):
    Combines class-frequency positive weighting with asymmetric margin clipping
    to counter extreme label sparsity in rare chest pathologies.
    
    Positive samples:
        L_+ = - pos_weight * targets * (1 - p)^gamma_pos * log(p)
    Negative samples:
        L_- = - (1 - targets) * (p_neg)^gamma_neg * log(1 - p_neg)
        where p_neg = max(p - clip, 0)
    """

    def __init__(
        self,
        gamma_pos: float = 2.0,
        gamma_neg: float = 4.0,
        clip: float = 0.05,
        eps: float = 1e-7,
        pos_weight: Optional[torch.Tensor] = None,
    ):
        super().__init__()
        self.gamma_pos = gamma_pos
        self.gamma_neg = gamma_neg
        self.clip = clip
        self.eps = eps
        self.register_buffer("pos_weight", pos_weight)

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        logits = logits.float()
        targets = targets.float()
        probs = torch.sigmoid(logits).clamp(min=self.eps, max=1.0 - self.eps)

        targets_pos = targets
        targets_neg = 1.0 - targets

        # Asymmetric negative probability shifting
        probs_neg = probs
        if self.clip is not None and self.clip > 0:
            probs_neg = (probs_neg - self.clip).clamp(min=0.0, max=1.0 - self.eps)

        # Asymmetric focal weighting
        loss_pos = targets_pos * ((1.0 - probs) ** self.gamma_pos) * torch.log(probs)
        loss_neg = targets_neg * (probs_neg ** self.gamma_neg) * torch.log(1.0 - probs_neg)

        if self.pos_weight is not None:
            loss_pos = loss_pos * self.pos_weight

        loss = -(loss_pos + loss_neg)
        return loss.mean()
