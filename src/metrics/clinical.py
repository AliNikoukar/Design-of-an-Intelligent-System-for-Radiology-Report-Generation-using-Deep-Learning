"""
Clinical diagnostic evaluation for radiology report generation.
Implements rule-based concept extraction with a 35-character bidirectional NegEx window.
Computes Macro-F1 (All-14 vs Core-10), Micro-F1, per-class Precision, Recall, and F1.
"""

import re
from typing import List, Dict, Any, Tuple
import numpy as np
from sklearn.metrics import precision_score, recall_score, f1_score

from src.data.constants import (
    CHEXPERT_PATHOLOGIES,
    CORE_10_PATHOLOGIES,
    CLINICAL_KEYWORDS,
    NEGATION_PREFIX_PATTERN,
    NEGATION_SUFFIX_PATTERN,
)


class NegExClinicalEvaluator:
    """
    Rule-based clinical presence detector utilizing regular expressions
    and bidirectional negation scope analysis.
    """

    def __init__(self, window_size: int = 50):
        self.window_size = window_size
        self.keywords = CLINICAL_KEYWORDS
        self.prefix_pattern = NEGATION_PREFIX_PATTERN
        self.suffix_pattern = NEGATION_SUFFIX_PATTERN

    def detect_presence(self, text: str, pathology: str) -> int:
        """
        Detects whether a given pathology is affirmed (positive finding) in the text.
        """
        text_lower = str(text).lower()
        terms = self.keywords.get(pathology, [])

        for kw in terms:
            matches = [m.start() for m in re.finditer(re.escape(kw), text_lower)]
            for match_idx in matches:
                kw_len = len(kw)
                prefix_window = text_lower[max(0, match_idx - self.window_size):match_idx]
                suffix_window = text_lower[match_idx + kw_len:min(len(text_lower), match_idx + kw_len + self.window_size)]

                is_negated_prefix = bool(self.prefix_pattern.search(prefix_window))
                is_negated_suffix = bool(self.suffix_pattern.search(suffix_window))

                if not is_negated_prefix and not is_negated_suffix:
                    return 1
        return 0

    def evaluate_cohort(
        self,
        candidate_reports: List[str],
        reference_reports: List[str],
    ) -> Dict[str, Any]:
        """
        Evaluates a cohort of generated reports against ground truth references.
        """
        n_samples = len(candidate_reports)
        n_classes = len(CHEXPERT_PATHOLOGIES)

        y_pred = np.zeros((n_samples, n_classes), dtype=np.int32)
        y_true = np.zeros((n_samples, n_classes), dtype=np.int32)

        for i, (cand, ref) in enumerate(zip(candidate_reports, reference_reports)):
            for j, path in enumerate(CHEXPERT_PATHOLOGIES):
                y_pred[i, j] = self.detect_presence(cand, path)
                y_true[i, j] = self.detect_presence(ref, path)

        # Per-class metrics
        per_class_f1 = {}
        per_class_prec = {}
        per_class_rec = {}

        for j, path in enumerate(CHEXPERT_PATHOLOGIES):
            f1 = f1_score(y_true[:, j], y_pred[:, j], zero_division=0)
            p = precision_score(y_true[:, j], y_pred[:, j], zero_division=0)
            r = recall_score(y_true[:, j], y_pred[:, j], zero_division=0)

            per_class_f1[path] = float(f1)
            per_class_prec[path] = float(p)
            per_class_rec[path] = float(r)

        # Macro-F1 across All-14
        macro_f1_all14 = float(np.mean(list(per_class_f1.values())))

        # Macro-F1 across Core-10 (excluding 4 ultra-rare classes)
        core_10_f1_vals = [per_class_f1[p] for p in CORE_10_PATHOLOGIES]
        macro_f1_core10 = float(np.mean(core_10_f1_vals))

        # Micro-F1 pooled across all labels
        micro_f1 = float(f1_score(y_true.ravel(), y_pred.ravel(), zero_division=0))
        micro_prec = float(precision_score(y_true.ravel(), y_pred.ravel(), zero_division=0))
        micro_rec = float(recall_score(y_true.ravel(), y_pred.ravel(), zero_division=0))

        return {
            "Macro_F1_All14": macro_f1_all14,
            "Macro_F1_Core10": macro_f1_core10,
            "Micro_F1": micro_f1,
            "Micro_Precision": micro_prec,
            "Micro_Recall": micro_rec,
            "Per_Class_F1": per_class_f1,
            "Per_Class_Precision": per_class_prec,
            "Per_Class_Recall": per_class_rec,
            "y_pred": y_pred,
            "y_true": y_true,
        }
