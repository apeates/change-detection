import csv
import importlib.util
import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image


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

    def test_sweep_reuses_inference_and_pools_counts_including_blank_masks(self):
        import torch

        from evaluate_unet import SWEEP_THRESHOLDS, compare_validation, evaluate

        class RedProbability(torch.nn.Module):
            def __init__(self, channel):
                super().__init__()
                self.channel = channel
                self.calls = 0

            def forward(self, inputs):
                self.calls += 1
                return inputs[:, self.channel:self.channel + 1].clamp(0.01, 0.99).logit()

        before = np.zeros((256, 256, 3), dtype=np.uint8)
        after = before.copy()
        after[0, 0, 0] = 102
        after[0, 0, 1] = 102
        after_changed = after.copy()
        after_changed[0, :2, 1] = 153
        blank = np.zeros((256, 256), dtype=np.uint8)
        changed = blank.copy()
        changed[0, :2] = 255

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            split = root / "val"
            output = root / "report"
            for folder in ("A", "B", "label"):
                (split / folder).mkdir(parents=True)
            for name, image, label in (("blank.png", after, blank), ("changed.png", after_changed, changed)):
                for folder, pixels in (("A", before), ("B", image), ("label", label)):
                    Image.fromarray(pixels).save(split / folder / name)

            model = RedProbability(channel=3)
            summary = evaluate(model, split, 0.35, 2, "cpu", output, "best.pt", sweep=True)
            with (output / "threshold_sweep.csv").open(newline="") as file:
                sweep = {float(row["threshold"]): row for row in csv.DictReader(file)}
            with (output / "per_image.csv").open(newline="") as file:
                rows = list(csv.DictReader(file))
            self.assertEqual(len(rows), 2)
            self.assertEqual(rows[0]["fp"], "1")
            self.assertTrue((output / "false_alarms.png").is_file())
            self.assertTrue((output / "missed_changes.png").is_file())
            self.assertTrue((output / "best_detection.png").is_file())
            with self.assertRaisesRegex(ValueError, "validation split"):
                evaluate(model, root / "test", 0.35, 2, "cpu", output, "best.pt", sweep=True)

            candidate = root / "candidate"
            evaluate(RedProbability(channel=4), split, 0.35, 2, "cpu", candidate, "best.pt", sweep=True)
            self.assertTrue((candidate / "per_image_best.csv").is_file())
            comparison = compare_validation(output, candidate)
            fixed = comparison["fixed_threshold"]
            best = comparison["best_threshold"]
            self.assertAlmostEqual(fixed["f1_delta"], 0.3)
            self.assertEqual(best["baseline_threshold"], 0.35)
            self.assertEqual(best["candidate_threshold"], 0.50)
            self.assertEqual(best["f1_delta"], 0.5)
            self.assertEqual(best["count_delta"], {"tp": 1, "fp": -1, "fn": -1})
            self.assertEqual(best["f1_delta_ci95"], [0.0, 0.5])

        self.assertEqual(model.calls, 2)
        self.assertEqual(tuple(sweep), SWEEP_THRESHOLDS)
        self.assertEqual(summary["counts"], {"tp": 1, "fp": 1, "fn": 1})
        self.assertEqual(summary["metrics"]["F1"], 0.5)
        self.assertEqual(summary["masks_without_change"], 1)
        self.assertEqual(summary["blank_mask_fp_rate"], 1 / blank.size)
        self.assertEqual(summary["threshold_sweep"]["best_threshold"], 0.35)
        self.assertEqual(float(sweep[0.35]["F1"]), summary["metrics"]["F1"])
        self.assertEqual((sweep[0.50]["tp"], sweep[0.50]["fp"], sweep[0.50]["fn"]), ("0", "0", "2"))


if __name__ == "__main__":
    unittest.main()
