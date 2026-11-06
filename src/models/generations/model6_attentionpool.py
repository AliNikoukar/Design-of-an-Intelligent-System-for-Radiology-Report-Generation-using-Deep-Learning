"""
Model Generation 6: AttentionPoolClassifier + Weighted Focal ASL + Qwen-2.5-3B.
Introduced 2D spatial attention pooling for disease localization and Weighted Focal ASL.
Uncovered the 'Cascading Error' bottleneck of discrete prompt condition strings.
"""

import torch
import torch.nn as nn
from typing import Tuple, Optional

from src.models.classifier import AttentionPoolClassifier


class MultiTaskQwenModel6(nn.Module):
    """
    Model 6:
    - Classifier: Per-pathology Spatial Attention Pooling (AttentionPoolClassifier)
    - Loss: Weighted Focal Asymmetric Loss (WF-ASL)
    - Connector: 2-layer LayerNorm MLP (768 -> 2048)
    - Prompt: Discrete pathology text string injection (identified cascading error)
    """

    def __init__(
        self,
        qwen_model: nn.Module,
        visual_dim: int = 768,
        qwen_dim: int = 2048,
        num_patches: int = 128,
        num_classes: int = 14,
    ):
        super().__init__()
        self.qwen = qwen_model
        self.num_patches = num_patches
        self.visual_dim = visual_dim
        self.qwen_dim = qwen_dim
        self.num_classes = num_classes

        self.visual_projection = nn.Sequential(
            nn.Linear(visual_dim, qwen_dim),
            nn.LayerNorm(qwen_dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(qwen_dim, qwen_dim),
            nn.LayerNorm(qwen_dim),
        )

        self.attn_pool_classifier = AttentionPoolClassifier(
            visual_dim=visual_dim,
            num_classes=num_classes,
        )

    def forward(
        self,
        spatial_features: torch.Tensor,
        prompt_input_ids: torch.Tensor,
        report_input_ids: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        cls_logits, attn_weights = self.attn_pool_classifier(spatial_features)
        vis_embeds = self.visual_projection(spatial_features)

        if report_input_ids is not None:
            text_ids = torch.cat([prompt_input_ids, report_input_ids], dim=1)
            text_embeds = self.qwen.get_input_embeddings()(text_ids)
            full_embeds = torch.cat([vis_embeds, text_embeds], dim=1)

            prefix_len = self.num_patches + prompt_input_ids.size(1)
            target_ids = torch.full(
                full_embeds.shape[:2], fill_value=-100, dtype=torch.long, device=full_embeds.device
            )
            target_ids[:, prefix_len:] = report_input_ids

            out = self.qwen(inputs_embeds=full_embeds, labels=target_ids)
            return cls_logits, attn_weights, out.loss
        else:
            prompt_embeds = self.qwen.get_input_embeddings()(prompt_input_ids)
            full_embeds = torch.cat([vis_embeds, prompt_embeds], dim=1)
            return cls_logits, attn_weights, None
