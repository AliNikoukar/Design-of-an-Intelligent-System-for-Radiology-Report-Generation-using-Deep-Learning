"""
PyTorch Datasets for multi-modal radiology report generation,
supporting memory-mapped 2D visual features and multi-label pathology targets.
"""

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from typing import Optional, Dict, Any, List

from src.data.constants import CHEXPERT_PATHOLOGIES


class MIMICCXRFeaturesDataset(Dataset):
    """
    Dataset loader for precomputed 2D spatial visual features (16x8 = 128 patches)
    and paired clinical impression targets.
    """

    def __init__(
        self,
        df: pd.DataFrame,
        features_spatial_path: str,
        tokenizer: Any,
        max_seq_len: int = 256,
        max_rag_tokens: int = 60,
        is_train: bool = True,
        use_rag: bool = False,
    ):
        self.df = df.reset_index(drop=True)
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len
        self.max_rag_tokens = max_rag_tokens
        self.is_train = is_train
        self.use_rag = use_rag

        # Infer shape from memmap file size (128 patches * 768 dim * 2 bytes for float16)
        bytes_per_sample = 128 * 768 * 2
        import os
        total_bytes = os.path.getsize(features_spatial_path)
        self.num_samples = min(len(self.df), total_bytes // bytes_per_sample)

        self.spatial_mmap = np.memmap(
            features_spatial_path,
            dtype="float16",
            mode="r",
            shape=(self.num_samples, 128, 768),
        )

        # Pre-extract multi-label binary ground truth
        available_cols = [c for c in CHEXPERT_PATHOLOGIES if c in self.df.columns]
        if len(available_cols) == len(CHEXPERT_PATHOLOGIES):
            self.labels = self.df[CHEXPERT_PATHOLOGIES].values[:self.num_samples].astype(np.float32)
        else:
            self.labels = np.zeros((self.num_samples, len(CHEXPERT_PATHOLOGIES)), dtype=np.float32)

    def __len__(self) -> int:
        return self.num_samples

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        row = self.df.iloc[idx]
        spatial_feats = torch.from_numpy(np.array(self.spatial_mmap[idx])).float()
        label_vec = torch.tensor(self.labels[idx], dtype=torch.float32)

        # Handle retrieved exemplar report (Visual-RAG)
        if self.use_rag and "retrieved_report" in row and pd.notna(row["retrieved_report"]):
            rag_text = str(row["retrieved_report"]).strip()
            rag_tokens = self.tokenizer.encode(rag_text, add_special_tokens=False)[:self.max_rag_tokens]
            rag_clean = self.tokenizer.decode(rag_tokens, skip_special_tokens=True)
        else:
            rag_clean = "None."

        # Active findings prompt construction
        active = [c for c, l in zip(CHEXPERT_PATHOLOGIES, label_vec) if l == 1.0]
        findings_str = ", ".join(active) if active else "No acute cardiopulmonary abnormality"

        prompt_str = f"Pathology Assessment: {findings_str}\nReference: {rag_clean}\nImpression:"
        report_str = str(row["report"]).strip() if pd.notna(row.get("report")) else ""

        return {
            "index": idx,
            "spatial_features": spatial_feats,
            "labels": label_vec,
            "prompt_text": prompt_str,
            "ground_truth_report": report_str,
            "study_id": str(row.get("study_id", idx)),
        }
