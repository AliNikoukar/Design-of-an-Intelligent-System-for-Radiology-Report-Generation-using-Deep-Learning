import unittest
from src.data.constants import (
    CHEXPERT_PATHOLOGIES,
    CORE_10_PATHOLOGIES,
    RARE_4_PATHOLOGIES,
    DEFAULT_OPTIMAL_THRESHOLDS,
)


class TestDataConstants(unittest.TestCase):
    def test_chexpert_constants(self):
        self.assertEqual(len(CHEXPERT_PATHOLOGIES), 14)
        for key in ["Atelectasis", "Cardiomegaly", "Consolidation", "Edema", "Pleural Effusion", "Pneumonia", "Pneumothorax"]:
            self.assertIn(key, CHEXPERT_PATHOLOGIES)

    def test_core10_subset(self):
        self.assertEqual(len(CORE_10_PATHOLOGIES), 10)
        for p in CORE_10_PATHOLOGIES:
            self.assertIn(p, CHEXPERT_PATHOLOGIES)

    def test_rare4_subset(self):
        self.assertEqual(len(RARE_4_PATHOLOGIES), 4)
        for p in RARE_4_PATHOLOGIES:
            self.assertIn(p, CHEXPERT_PATHOLOGIES)

    def test_optimal_thresholds(self):
        self.assertEqual(len(DEFAULT_OPTIMAL_THRESHOLDS), 14)
        for k, v in DEFAULT_OPTIMAL_THRESHOLDS.items():
            self.assertIn(k, CHEXPERT_PATHOLOGIES)
            self.assertTrue(0.0 <= v <= 1.0)


if __name__ == "__main__":
    unittest.main()
