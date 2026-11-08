import importlib.util
import unittest

import numpy as np


@unittest.skipUnless(importlib.util.find_spec("torch"), "Install requirements-train.txt to test U-Net")
class EvaluateUNetTests(unittest.TestCase):
    def test_prediction_stitches_tiles_and_uses_probability_threshold(self):
        import torch

        from evaluate_unet import predict_mask

        class RedProbability(torch.nn.Module):
            def forward(self, inputs):
                return inputs[:, 3:4].clamp(0.01, 0.99).logit()

        before = np.zeros((512, 512, 3), dtype=np.uint8)
        after = before.copy()
        after[254:260, 254:259, 0] = 102  # Probability 0.4, spanning all four tiles.
        after[400, 305, 0] = 200
        reference = np.zeros((512, 512), dtype=bool)

        model = RedProbability()
        lower = predict_mask(model, before, after, reference, 0.35, 3, "cpu")
        fixed = predict_mask(model, before, after, reference, 0.50, 3, "cpu")
        self.assertEqual(lower.dtype, np.bool_)
        np.testing.assert_array_equal(lower, after[:, :, 0] > 0)
        self.assertEqual(fixed.sum(), 1)
        self.assertTrue(fixed[400, 305])


if __name__ == "__main__":
    unittest.main()
