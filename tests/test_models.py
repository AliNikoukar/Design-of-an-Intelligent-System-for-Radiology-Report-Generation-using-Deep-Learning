import unittest
import torch
from src.models.perceiver import PerceiverResampler
from src.models.projector import SoftConditioningProjector
from src.models.losses import WeightedFocalASL
from src.models.classifier import AttentionPoolClassifier


class TestModels(unittest.TestCase):
    def test_perceiver_resampler_shapes(self):
        B, num_patches, visual_dim = 2, 128, 768
        num_queries, qwen_dim = 32, 2048
        
        perceiver = PerceiverResampler(
            visual_dim=visual_dim,
            qwen_dim=qwen_dim,
            num_queries=num_queries,
            num_heads=4,
            num_layers=1,
        )
        
        x = torch.randn(B, num_patches, visual_dim)
        out = perceiver(x)
        self.assertEqual(out.shape, (B, num_queries, qwen_dim))

    def test_soft_conditioning_projector_shapes(self):
        B, num_classes, qwen_dim = 3, 14, 2048
        projector = SoftConditioningProjector(num_classes=num_classes, hidden_dim=64, qwen_dim=qwen_dim)
        
        disease_probs = torch.rand(B, num_classes)
        out = projector(disease_probs)
        self.assertEqual(out.shape, (B, 1, qwen_dim))

    def test_attention_pool_classifier(self):
        B, num_patches, visual_dim, num_classes = 2, 64, 768, 14
        classifier = AttentionPoolClassifier(visual_dim=visual_dim, num_classes=num_classes)
        
        x = torch.randn(B, num_patches, visual_dim)
        logits, attn_weights = classifier(x)
        self.assertEqual(logits.shape, (B, num_classes))
        self.assertEqual(attn_weights.shape, (B, num_patches, num_classes))
        # Check attention weights across patches sum to 1 per class
        self.assertTrue(torch.allclose(attn_weights.sum(dim=1), torch.ones(B, num_classes), atol=1e-5))

    def test_asl_loss_computation(self):
        criterion = WeightedFocalASL(gamma_pos=0.0, gamma_neg=4.0, clip=0.05)
        logits = torch.randn(4, 14)
        targets = torch.randint(0, 2, (4, 14)).float()
        
        loss = criterion(logits, targets)
        self.assertEqual(loss.ndim, 0)
        self.assertFalse(torch.isnan(loss))
        self.assertGreaterEqual(loss.item(), 0.0)


if __name__ == "__main__":
    unittest.main()
