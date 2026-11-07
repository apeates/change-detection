import importlib.util
import tempfile
import unittest
from pathlib import Path

import numpy as np


@unittest.skipUnless(importlib.util.find_spec("torch"), "Install requirements-train.txt to test U-Net")
class UNetTrainingTests(unittest.TestCase):
    def test_output_matches_input_spatial_shape(self):
        import torch

        from early_fusion_unet import EarlyFusionUNet

        model = EarlyFusionUNet().eval()
        with torch.inference_mode():
            output = model(torch.zeros(1, 6, 256, 256))
        self.assertEqual(tuple(output.shape), (1, 1, 256, 256))
        self.assertTrue(torch.isfinite(output).all().item())

    def test_training_updates_model_and_checkpoint_restores_prediction(self):
        import torch

        from early_fusion_unet import EarlyFusionUNet
        from levir_tiles import PairTile
        from train_unet import load_model, save_checkpoint, train_epoch

        torch.manual_seed(42)
        model = EarlyFusionUNet()
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        images = np.zeros((6, 32, 32), dtype=np.float32)
        mask = np.zeros((1, 32, 32), dtype=np.float32)
        images[3:, 8:24, 8:24] = 1
        mask[:, 8:24, 8:24] = 1
        tile = PairTile(x=0, y=0, inputs=images, mask=mask)

        losses = [train_epoch(model, optimizer, [[tile]], "cpu") for _ in range(12)]
        self.assertLess(losses[-1], losses[0])
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "checkpoint.pt"
            save_checkpoint(path, model, optimizer, epoch=12, seed=42)
            inputs = torch.from_numpy(images).unsqueeze(0)
            with torch.inference_mode():
                expected = model.eval()(inputs)
                actual = load_model(path)(inputs)
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)


if __name__ == "__main__":
    unittest.main()
