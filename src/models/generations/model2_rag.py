"""
Model Generation 2: Visual-RAG Architecture + Multi-Encoder Ablation.
Retrieves medically analogous exemplar reports from a patient-stratified FAISS index
and injects them as prompt guidance into the language decoder.
"""

import torch
import torch.nn as nn
from typing import Tuple, Optional


class MultiTaskRAGDenseNetDistilGPT2(nn.Module):
    """
    Model 2:
    - Visual Encoder: Ablated across DenseNet-121, BioViL, BioMedCLIP, CX-Former, RAD-DINO
    - Retrieval: Visual-RAG using FAISS FlatIP index on training cohort visual embeddings
    - Multitask Branch A: 14-class CheXpert classification head
    - Multitask Branch B: Visual Prefix + Retrieved RAG Exemplar Text Prompt
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

        self.classifier_head = nn.Sequential(
            nn.Linear(encoder_dim, 256),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(256, num_classes),
        )

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
            spatial_tokens: (Batch, NumPatches, EncoderDim)
            mean_features: (Batch, EncoderDim)
            input_ids: (Batch, SeqLen) containing concatenated [<bos>, <rag>, retrieved_tokens, target_tokens]
        Returns:
            text_logits: (Batch, SeqLen - 1, VocabSize)
            cls_logits: (Batch, 14)
        """
        batch_size, seq_len = input_ids.shape

        cls_logits = self.classifier_head(mean_features)
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
