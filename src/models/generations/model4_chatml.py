"""
Model Generation 4: Qwen-2.5-3B + 4-bit QLoRA + ChatML Prompt Structure.
Identified the critical 'Template Collapse' / linguistic stagnation failure mode
caused by conversational template overhead and autoregressive mode collapse.
"""

import torch
import torch.nn as nn
from typing import Tuple, Optional


class MultiTaskQwenModel4(nn.Module):
    """
    Model 4:
    - Backbone: Qwen-2.5-3B-Instruct quantized in 4-bit NormalFloat (NF4) via QLoRA (r=32, alpha=64)
    - Visual Input: 128 patch tokens from RAD-DINO
    - Visual Projection: 2-layer MLP (768 -> 2048 -> 2048) with GELU
    - Attention Mask: 4D causal prefix attention mask (bidirectional on patches, causal on text)
    - Prompt Format: Conversational ChatML (<|im_start|>system...<|im_start|>user...)
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
            nn.GELU(),
            nn.Linear(qwen_dim, qwen_dim),
        )

        self.classifier = nn.Sequential(
            nn.Linear(visual_dim, 512),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(512, num_classes),
        )

    def build_4d_prefix_attention_mask(
        self,
        batch_size: int,
        seq_len: int,
        device: torch.device,
        dtype: torch.dtype,
        text_pad_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Constructs a 4D causal prefix attention mask:
        - Visual patches have full bidirectional cross-attention with each other
        - Text tokens attend to all visual patches and strictly past text tokens (causal)
        """
        total_len = self.num_patches + seq_len
        mask = torch.full((total_len, total_len), -10000.0, dtype=dtype, device=device)

        # Bidirectional visual prefix
        mask[:self.num_patches, :self.num_patches] = 0.0

        # Causal text decoding
        for i in range(self.num_patches, total_len):
            mask[i, :i + 1] = 0.0

        mask_4d = mask.unsqueeze(0).unsqueeze(0).expand(batch_size, 1, total_len, total_len).clone()
        if text_pad_mask is not None:
            # Mask out padding tokens
            pad_expanded = (~text_pad_mask).unsqueeze(1).unsqueeze(2)
            mask_4d[:, :, :, self.num_patches:].masked_fill_(pad_expanded, -10000.0)

        return mask_4d

    def forward(
        self,
        spatial_features: torch.Tensor,
        prompt_input_ids: torch.Tensor,
        report_input_ids: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Args:
            spatial_features: (Batch, 128, 768)
            prompt_input_ids: (Batch, PromptLen) ChatML tokens
            report_input_ids: (Batch, ReportLen) target report tokens
        Returns:
            cls_logits: (Batch, 14)
            lm_loss: scalar language modeling loss
        """
        # Pathology classification on mean spatial features
        mean_feat = spatial_features.mean(dim=1)
        cls_logits = self.classifier(mean_feat)

        # Project visual patches
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
            return cls_logits, out.loss
        else:
            prompt_embeds = self.qwen.get_input_embeddings()(prompt_input_ids)
            full_embeds = torch.cat([vis_embeds, prompt_embeds], dim=1)
            return cls_logits, None
