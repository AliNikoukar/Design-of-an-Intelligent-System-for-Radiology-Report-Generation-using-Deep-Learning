"""
Model 7 Flagship Multimodal Architecture for Radiology Report Generation.
Integrates RAD-DINO spatial features, Perceiver Resampler, AttentionPool classifier,
Soft Pathology Conditioning, and Qwen-2.5-3B-Instruct adapted via LoRA.
"""

import torch
import torch.nn as nn
from typing import Dict, Any, Tuple, Optional, List

from src.models.perceiver import PerceiverResampler
from src.models.classifier import AttentionPoolClassifier
from src.models.projector import SoftConditioningProjector
from src.data.constants import CHEXPERT_PATHOLOGIES, DEFAULT_OPTIMAL_THRESHOLDS


class MultiTaskModel7(nn.Module):
    """
    Model 7 Architecture:
    - Input: 2D spatial features from frozen RAD-DINO (128 patches of dim 768)
    - Classifier Branch: Spatial Attention Pooling -> 14 CheXpert logits
    - Compression Branch: Perceiver Resampler -> 32 latent queries of dim 2048
    - Conditioning Branch: Continuous MLP projection of predicted probabilities -> 1 token
    - Language Decoder: Qwen-2.5-3B-Instruct adapted with LoRA (rank 32, alpha 64)
    """

    def __init__(
        self,
        qwen_model: nn.Module,
        visual_dim: int = 768,
        qwen_dim: int = 2048,
        num_queries: int = 32,
        num_classes: int = 14,
    ):
        super().__init__()
        self.qwen = qwen_model
        self.visual_dim = visual_dim
        self.qwen_dim = qwen_dim
        self.num_queries = num_queries
        self.num_classes = num_classes

        # Visual compression module
        self.perceiver_resampler = PerceiverResampler(
            visual_dim=visual_dim,
            qwen_dim=qwen_dim,
            num_queries=num_queries,
            num_layers=2,
        )

        # Spatial attention pathology classifier
        self.attn_pool_classifier = AttentionPoolClassifier(
            visual_dim=visual_dim,
            num_classes=num_classes,
        )

        # Continuous soft pathology conditioning projector
        self.cls_embed_proj = SoftConditioningProjector(
            num_classes=num_classes,
            hidden_dim=256,
            qwen_dim=qwen_dim,
        )

    def forward(
        self,
        spatial_features: torch.Tensor,
        prompt_input_ids: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
        report_input_ids: Optional[torch.Tensor] = None,
    ) -> Dict[str, Any]:
        """
        Forward pass for joint multitask training.
        """
        # Step 1: Predict pathology logits
        cls_logits, attn_weights = self.attn_pool_classifier(spatial_features)
        cls_probs = torch.sigmoid(cls_logits.float())

        # Step 2: Extract visual queries and continuous pathology token
        visual_queries = self.perceiver_resampler(spatial_features)
        cls_token = self.cls_embed_proj(cls_probs)

        # Step 3: Embed text prompts
        prompt_embeds = self.qwen.get_input_embeddings()(prompt_input_ids)

        if visual_queries.dtype != prompt_embeds.dtype:
            visual_queries = visual_queries.to(prompt_embeds.dtype)
        if cls_token.dtype != prompt_embeds.dtype:
            cls_token = cls_token.to(prompt_embeds.dtype)

        # Multimodal prefix: [32 visual queries, 1 soft pathology token, prompt tokens]
        prefix_embeds = torch.cat([visual_queries, cls_token], dim=1)
        inputs_embeds = torch.cat([prefix_embeds, prompt_embeds], dim=1)

        outputs = {
            "cls_logits": cls_logits,
            "attn_weights": attn_weights,
            "inputs_embeds": inputs_embeds,
        }

        # Step 4: Causal language modeling loss if training labels are provided
        if report_input_ids is not None:
            # Concatenate report tokens to full sequence
            report_embeds = self.qwen.get_input_embeddings()(report_input_ids)
            full_embeds = torch.cat([inputs_embeds, report_embeds], dim=1)

            # Mask out prefix tokens in loss computation
            prefix_len = inputs_embeds.size(1)
            target_ids = torch.full(
                full_embeds.shape[:2], fill_value=-100, dtype=torch.long, device=full_embeds.device
            )
            target_ids[:, prefix_len:] = report_input_ids

            lm_out = self.qwen(inputs_embeds=full_embeds, labels=target_ids)
            outputs["lm_loss"] = lm_out.loss
            outputs["logits"] = lm_out.logits

        return outputs

    @torch.inference_mode()
    def generate_report(
        self,
        spatial_tokens: torch.Tensor,
        tokenizer: Any,
        optimal_thresholds: Optional[Dict[str, float]] = None,
        reference_report: Optional[str] = None,
        max_new_tokens: int = 150,
        num_beams: int = 4,
        device: torch.device = torch.device("cpu"),
    ) -> Tuple[str, List[float], Any, str]:
        """
        Inference routine using deterministic beam search decoding.
        """
        self.eval()
        if spatial_tokens.dim() == 2:
            spatial_tokens = spatial_tokens.unsqueeze(0)

        spatial_tokens = spatial_tokens.to(device)
        thresholds = optimal_thresholds or DEFAULT_OPTIMAL_THRESHOLDS

        # 1. Disease prediction
        cls_logits, attn_weights = self.attn_pool_classifier(spatial_tokens.float())
        cls_probs_t = torch.sigmoid(cls_logits.float())
        cls_probs = cls_probs_t.squeeze(0).cpu().numpy().tolist()

        active_findings = [
            c for i, c in enumerate(CHEXPERT_PATHOLOGIES)
            if cls_probs[i] >= thresholds.get(c, 0.5)
        ]
        pathology_str = ", ".join(active_findings) if active_findings else "No acute cardiopulmonary abnormality"

        # 2. Build completion prompt
        ref_str = reference_report if reference_report is not None else "None."
        prompt_str = f"Pathology Assessment: {pathology_str}\nReference: {ref_str}\nImpression:"
        prompt_enc = tokenizer(prompt_str, return_tensors="pt").to(device)
        prompt_ids = prompt_enc["input_ids"]
        prompt_embeds = self.qwen.get_input_embeddings()(prompt_ids)

        # 3. Multimodal prefix
        visual_queries = self.perceiver_resampler(spatial_tokens.float())
        cls_token = self.cls_embed_proj(cls_probs_t)

        if visual_queries.dtype != prompt_embeds.dtype:
            visual_queries = visual_queries.to(prompt_embeds.dtype)
        if cls_token.dtype != prompt_embeds.dtype:
            cls_token = cls_token.to(prompt_embeds.dtype)

        prefix_embeds = torch.cat([visual_queries, cls_token], dim=1)
        inputs_embeds = torch.cat([prefix_embeds, prompt_embeds], dim=1)
        attn_mask = torch.ones(inputs_embeds.shape[:2], dtype=torch.long, device=device)

        # 4. Beam search generation
        gen_out = self.qwen.generate(
            inputs_embeds=inputs_embeds,
            attention_mask=attn_mask,
            max_new_tokens=max_new_tokens,
            num_beams=num_beams,
            early_stopping=True,
            repetition_penalty=1.05,
            no_repeat_ngram_size=3,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.pad_token_id,
            use_cache=True,
        )

        if gen_out.size(1) > inputs_embeds.size(1):
            gen_tokens = gen_out[0, inputs_embeds.size(1):]
        else:
            gen_tokens = gen_out[0]

        pred_text = tokenizer.decode(gen_tokens, skip_special_tokens=True).strip()
        if pred_text.lower().startswith("impression:"):
            pred_text = pred_text[len("impression:"):].strip()

        return pred_text, cls_probs, attn_weights.squeeze(0).cpu().numpy(), pathology_str
