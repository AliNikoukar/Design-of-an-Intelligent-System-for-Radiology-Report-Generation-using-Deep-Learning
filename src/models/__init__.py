"""
Unified Model Registry for all 3 Baselines and 7 Architecture Generations.
"""

from typing import Any, Dict

from src.models.classifier import AttentionPoolClassifier
from src.models.perceiver import PerceiverResampler
from src.models.projector import SoftConditioningProjector
from src.models.losses import WeightedFocalASL
from src.models.model7 import MultiTaskModel7

# Baselines
from src.models.baselines.baseline_a import BaselineDenseNetLSTM
from src.models.baselines.baseline_b import BaselineDenseNetAttentionLSTM, BahdanauAttention
from src.models.baselines.baseline_c import BaselineDenseNetDistilGPT2

# Generations
from src.models.generations.model1_multitask import MultiTaskDenseNetDistilGPT2
from src.models.generations.model2_rag import MultiTaskRAGDenseNetDistilGPT2
from src.models.generations.model4_chatml import MultiTaskQwenModel4
from src.models.generations.model5_completion import MultiTaskQwenModel5
from src.models.generations.model6_attentionpool import MultiTaskQwenModel6

MODEL_REGISTRY: Dict[str, Any] = {
    # Baselines
    "baseline_a": BaselineDenseNetLSTM,
    "baseline_b": BaselineDenseNetAttentionLSTM,
    "baseline_c": BaselineDenseNetDistilGPT2,
    # Architectural Generations
    "model1": MultiTaskDenseNetDistilGPT2,
    "model2": MultiTaskRAGDenseNetDistilGPT2,
    "model4": MultiTaskQwenModel4,
    "model5": MultiTaskQwenModel5,
    "model6": MultiTaskQwenModel6,
    "model7": MultiTaskModel7,
}


def build_model(model_name: str, **kwargs) -> Any:
    """
    Factory function to instantiate any baseline or model generation by identifier.

    Available names:
        - 'baseline_a' : DenseNet-121 + Linear Projection + LSTM
        - 'baseline_b' : DenseNet-121 + Bahdanau Spatial Attention + LSTM
        - 'baseline_c' : DenseNet-121 + 2-layer MLP Visual Prefix + DistilGPT-2
        - 'model1'     : Multi-task Co-trained CheXpert Head + DistilGPT-2
        - 'model2'     : Visual-RAG (FAISS retriever) + DistilGPT-2
        - 'model4'     : Qwen-2.5-3B + QLoRA + ChatML prompt + 4D Causal Prefix Mask
        - 'model5'     : Qwen-2.5-3B + Direct Completion Prompt + LayerNorm Connector + ASL
        - 'model6'     : Qwen-2.5-3B + AttentionPoolClassifier + WF-ASL
        - 'model7'     : Flagship RAD-DINO + Perceiver Resampler + Soft Pathology Conditioning
    """
    key = model_name.lower().strip()
    if key not in MODEL_REGISTRY:
        raise ValueError(
            f"Unknown model architecture '{model_name}'. Available options: {list(MODEL_REGISTRY.keys())}"
        )
    return MODEL_REGISTRY[key](**kwargs)


__all__ = [
    # Factory & Registry
    "build_model",
    "MODEL_REGISTRY",
    # Components
    "AttentionPoolClassifier",
    "PerceiverResampler",
    "SoftConditioningProjector",
    "WeightedFocalASL",
    "BahdanauAttention",
    # Baselines
    "BaselineDenseNetLSTM",
    "BaselineDenseNetAttentionLSTM",
    "BaselineDenseNetDistilGPT2",
    # Generations
    "MultiTaskDenseNetDistilGPT2",
    "MultiTaskRAGDenseNetDistilGPT2",
    "MultiTaskQwenModel4",
    "MultiTaskQwenModel5",
    "MultiTaskQwenModel6",
    "MultiTaskModel7",
]
