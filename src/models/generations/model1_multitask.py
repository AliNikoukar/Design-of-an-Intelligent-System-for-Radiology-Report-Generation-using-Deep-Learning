"""
Model Generation 1: DenseNet-121 + Multitask Gradient Alignment + DistilGPT-2.
Co-trains a 14-class CheXpert classification head alongside the language decoder
to force the visual encoder to preserve pathological features.
"""

import torch
import torch.nn as nn
from typing import Tuple


class MultiTaskDenseNetDistilGPT2(nn.Module):
    """
    Model 1:
    - Shared Visual Encoder: DenseNet-121 (49 patches of dim 1024)
    - Multitask Branch A (Clinical): 14-class pathology classification head
    - Multitask Branch B (Text): Visual prefix projection to DistilGPT-2
    - Joint Loss: L_joint = L_LM + lambda_cls * L_BCE
    """

    def __init__(
        self,
        gpt2_decoder: nn.Module,
        encoder_dim: int = 1024,
        gpt2_dim: int = 768,
        num_classes: int = 14,
        num_patches: int = 49,
        pad_token_id: int = 0,
    ):
        super().__init__()
        self.decoder = gpt2_decoder
        self.encoder_dim = encoder_dim
        self.gpt2_dim = gpt2_dim
        self.num_classes = num_classes
        self.num_patches = num_patches
        self.pad_token_id = pad_token_id

        # 14-class CheXpert diagnostic head
        self.classifier_head = nn.Sequential(
            nn.Linear(encoder_dim, 256),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(256, num_classes),
        )

        # Visual connector for transformer
        self.visual_projection = nn.Sequential(
            nn.Linear(encoder_dim, gpt2_dim),
            nn.GELU(),
            nn.Linear(gpt2_dim, gpt2_dim),
            nn.LayerNorm(gpt2_dim),
        )

    def forward(
        self,
        spatial_tokens: torch.Tensor,
        mean_features: torch.Tensor,
        input_ids: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            spatial_tokens: (Batch, 49, 1024)
            mean_features: (Batch, 1024)
            input_ids: (Batch, SeqLen)
        Returns:
            text_logits: (Batch, SeqLen - 1, VocabSize)
            cls_logits: (Batch, 14)
        """
        batch_size, seq_len = input_ids.shape

        # 1. Pathology classification logits
        cls_logits = self.classifier_head(mean_features)

        # 2. Text generation logits
        visual_embeds = self.visual_projection(spatial_tokens)
        text_embeds = self.decoder.transformer.wte(input_ids)
        inputs_embeds = torch.cat([visual_embeds, text_embeds], dim=1)

        vis_pos_ids = torch.arange(0, self.num_patches, dtype=torch.long, device=input_ids.device).unsqueeze(0).repeat(batch_size, 1)
        text_pos_ids = torch.arange(0, seq_len, dtype=torch.long, device=input_ids.device).unsqueeze(0).repeat(batch_size, 1)
        position_ids = torch.cat([vis_pos_ids, text_pos_ids], dim=1)

        vis_attn = torch.ones((batch_size, self.num_patches), device=input_ids.device)
        text_attn = (input_ids != self.pad_token_id).float()
        attention_mask = torch.cat([vis_attn, text_attn], dim=1)

        outputs = self.decoder(
            inputs_embeds=inputs_embeds,
            position_ids=position_ids,
            attention_mask=attention_mask,
        )
        text_logits = outputs.logits[:, self.num_patches:-1, :].contiguous()

        return text_logits, cls_logits
