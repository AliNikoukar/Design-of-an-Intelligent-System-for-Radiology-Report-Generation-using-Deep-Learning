"""
Baseline Architecture C: DenseNet-121 + Visual Prefix Projection + DistilGPT-2 Decoder.
Represents the modern lightweight Transformer-decoder baseline with prefix visual tokens.
"""

import torch
import torch.nn as nn
from typing import Optional, List


class BaselineDenseNetDistilGPT2(nn.Module):
    """
    Baseline C:
    - Visual Encoder: DenseNet-121 spatial patch tokens (49 patches of dim 1024)
    - Visual Connector: 2-layer MLP with GELU and LayerNorm (1024 -> 768)
    - Language Decoder: Pre-trained DistilGPT-2 (6 transformer blocks, 768 dim)
    - Multimodal Assembly: [49 Visual Prefix Tokens, Text Tokens]
    """

    def __init__(
        self,
        gpt2_decoder: nn.Module,
        encoder_dim: int = 1024,
        gpt2_dim: int = 768,
        num_patches: int = 49,
        pad_token_id: int = 0,
    ):
        super().__init__()
        self.decoder = gpt2_decoder
        self.encoder_dim = encoder_dim
        self.gpt2_dim = gpt2_dim
        self.num_patches = num_patches
        self.pad_token_id = pad_token_id

        # Non-linear visual projection layer
        self.visual_projection = nn.Sequential(
            nn.Linear(encoder_dim, gpt2_dim),
            nn.GELU(),
            nn.Linear(gpt2_dim, gpt2_dim),
            nn.LayerNorm(gpt2_dim),
        )

    def forward(
        self,
        spatial_tokens: torch.Tensor,
        input_ids: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            spatial_tokens: (Batch, NumPatches, EncoderDim) e.g. (B, 49, 1024)
            input_ids: (Batch, SeqLen)
        Returns:
            logits: Next-token logits over the text segment (Batch, SeqLen - 1, VocabSize)
        """
        batch_size, seq_len = input_ids.shape
        visual_embeds = self.visual_projection(spatial_tokens)  # (B, 49, 768)
        text_embeds = self.decoder.transformer.wte(input_ids)  # (B, SeqLen, 768)

        inputs_embeds = torch.cat([visual_embeds, text_embeds], dim=1)  # (B, 49 + SeqLen, 768)

        # Explicit position IDs to avoid sequence length collisions
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
        logits = outputs.logits  # (B, 49 + SeqLen, VocabSize)
        text_logits = logits[:, self.num_patches:-1, :].contiguous()
        return text_logits
