import unittest
import numpy as np

try:
    import nltk
    from rouge_score import rouge_scorer
    HAS_NLP_DEPS = True
except ImportError:
    HAS_NLP_DEPS = False

from src.metrics.clinical import NegExClinicalEvaluator
from src.metrics.bootstrap import compute_bootstrap_ci


class TestMetrics(unittest.TestCase):
    @unittest.skipUnless(HAS_NLP_DEPS, "nltk or rouge_score not installed in current environment")
    def test_nlp_metrics_exact_match(self):
        from src.metrics.nlp import compute_nlp_metrics
        predictions = ["No acute cardiopulmonary abnormality."]
        references = ["No acute cardiopulmonary abnormality."]
        metrics = compute_nlp_metrics(predictions, references)
        self.assertAlmostEqual(metrics["BLEU-1"], 1.0, places=2)
        self.assertAlmostEqual(metrics["BLEU-4"], 1.0, places=2)
        self.assertAlmostEqual(metrics["ROUGE-L"], 1.0, places=2)
        self.assertAlmostEqual(metrics["bleu_1"], 1.0, places=2)
        self.assertAlmostEqual(metrics["bleu_4"], 1.0, places=2)
        self.assertAlmostEqual(metrics["rouge_l"], 1.0, places=2)

    @unittest.skipUnless(HAS_NLP_DEPS, "nltk or rouge_score not installed in current environment")
    def test_nlp_metrics_partial_match(self):
        from src.metrics.nlp import compute_nlp_metrics
        predictions = ["Lungs are clear. No pneumothorax."]
        references = ["Lungs are clear. Mild cardiomegaly without pneumothorax."]
        metrics = compute_nlp_metrics(predictions, references)
        self.assertTrue(0.0 < metrics["BLEU-1"] < 1.0)
        self.assertTrue(0.0 < metrics["ROUGE-L"] < 1.0)
        self.assertTrue(0.0 < metrics["bleu_1"] < 1.0)
        self.assertTrue(0.0 < metrics["rouge_l"] < 1.0)
        self.assertGreater(metrics["distinct_1"], 0.0)
        self.assertGreater(metrics["distinct_2"], 0.0)
        self.assertGreater(metrics["Sample_Distinct_1"], 0.0)
        self.assertGreater(metrics["Sample_Distinct_2"], 0.0)

    def test_clinical_evaluator_negex(self):
        evaluator = NegExClinicalEvaluator()
        sample_text = (
            "Lungs are clear with no focal consolidation, pneumothorax, or pleural effusion. "
            "Moderate cardiomegaly is noted."
        )
        
        # Positive finding
        self.assertEqual(evaluator.detect_presence(sample_text, "Cardiomegaly"), 1)
        # Explicitly negated findings
        self.assertEqual(evaluator.detect_presence(sample_text, "Consolidation"), 0)
        self.assertEqual(evaluator.detect_presence(sample_text, "Pneumothorax"), 0)
        self.assertEqual(evaluator.detect_presence(sample_text, "Pleural Effusion"), 0)

    def test_bootstrap_ci(self):
        rng = np.random.RandomState(42)
        scores = rng.normal(loc=0.5, scale=0.05, size=200)
        mean, lower, upper = compute_bootstrap_ci(scores, n_bootstraps=500, confidence_level=0.95, seed=42)
        self.assertTrue(lower <= mean <= upper)
        self.assertTrue(0.45 <= mean <= 0.55)


if __name__ == "__main__":
    unittest.main()
