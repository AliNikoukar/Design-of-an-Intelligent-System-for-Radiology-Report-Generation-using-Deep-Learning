"""
Natural Language Processing (NLP) evaluation metrics for generated medical reports.
Computes BLEU-1 to BLEU-4, ROUGE-L, and lexical diversity (Distinct-1 & Distinct-2).
"""

from typing import List, Dict, Any
import numpy as np


def compute_nlp_metrics(
    candidate_reports: List[str],
    reference_reports: List[str],
) -> Dict[str, float]:
    """
    Computes standard NLP benchmarks between candidate and reference texts.

    Args:
        candidate_reports: List of generated impression strings.
        reference_reports: List of ground-truth radiologist impression strings.

    Returns:
        Dict mapping metric names to computed floating point values.
    """
    assert len(candidate_reports) == len(reference_reports), "Length mismatch between candidates and references."

    try:
        from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
        from rouge_score import rouge_scorer
    except ImportError as e:
        raise ImportError(
            f"Required NLP metric libraries missing: {e}. "
            "Please install them via: pip install nltk rouge-score"
        )

    smoother = SmoothingFunction().method1
    scorer = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=True)

    b1_list, b2_list, b3_list, b4_list = [], [], [], []
    rL_list = []
    sample_d1_list, sample_d2_list = [], []
    lengths_list = []
    all_unigrams, all_bigrams = [], []

    for cand, ref in zip(candidate_reports, reference_reports):
        cand_tokens = str(cand).strip().lower().split()
        ref_tokens = str(ref).strip().lower().split()

        # Token length
        l = len(cand_tokens)
        lengths_list.append(l)

        # Sentence BLEU
        b1 = sentence_bleu([ref_tokens], cand_tokens, weights=(1.0, 0, 0, 0), smoothing_function=smoother)
        b2 = sentence_bleu([ref_tokens], cand_tokens, weights=(0.5, 0.5, 0, 0), smoothing_function=smoother)
        b3 = sentence_bleu([ref_tokens], cand_tokens, weights=(0.333, 0.333, 0.333, 0), smoothing_function=smoother)
        b4 = sentence_bleu([ref_tokens], cand_tokens, weights=(0.25, 0.25, 0.25, 0.25), smoothing_function=smoother)

        b1_list.append(b1)
        b2_list.append(b2)
        b3_list.append(b3)
        b4_list.append(b4)

        # ROUGE-L
        r = scorer.score(" ".join(ref_tokens), " ".join(cand_tokens))
        rL_list.append(r["rougeL"].fmeasure)

        # Sample-level Distinct
        if l >= 1:
            sample_d1_list.append(len(set(cand_tokens)) / l)
            all_unigrams.extend(cand_tokens)
        else:
            sample_d1_list.append(1.0)

        if l >= 2:
            bigrams = [f"{cand_tokens[i]} {cand_tokens[i+1]}" for i in range(l - 1)]
            sample_d2_list.append(len(set(bigrams)) / len(bigrams))
            all_bigrams.extend(bigrams)
        else:
            sample_d2_list.append(1.0)

    # Corpus-level Distinct
    corpus_d1 = len(set(all_unigrams)) / max(len(all_unigrams), 1)
    corpus_d2 = len(set(all_bigrams)) / max(len(all_bigrams), 1)

    return {
        "BLEU-1": float(np.mean(b1_list)),
        "BLEU-2": float(np.mean(b2_list)),
        "BLEU-3": float(np.mean(b3_list)),
        "BLEU-4": float(np.mean(b4_list)),
        "ROUGE-L": float(np.mean(rL_list)),
        "Sample_Distinct_1": float(np.mean(sample_d1_list)),
        "Sample_Distinct_2": float(np.mean(sample_d2_list)),
        "Corpus_Distinct_1": float(corpus_d1),
        "Corpus_Distinct_2": float(corpus_d2),
        "Mean_Generated_Length": float(np.mean(lengths_list)),
    }
