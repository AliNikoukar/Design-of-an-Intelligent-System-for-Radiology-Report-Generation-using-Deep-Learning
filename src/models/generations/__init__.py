from src.models.generations.model1_multitask import MultiTaskDenseNetDistilGPT2
from src.models.generations.model2_rag import MultiTaskRAGDenseNetDistilGPT2
from src.models.generations.model4_chatml import MultiTaskQwenModel4
from src.models.generations.model5_completion import MultiTaskQwenModel5
from src.models.generations.model6_attentionpool import MultiTaskQwenModel6
from src.models.model7 import MultiTaskModel7

__all__ = [
    "MultiTaskDenseNetDistilGPT2",
    "MultiTaskRAGDenseNetDistilGPT2",
    "MultiTaskQwenModel4",
    "MultiTaskQwenModel5",
    "MultiTaskQwenModel6",
    "MultiTaskModel7",
]
