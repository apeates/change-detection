import csv
import importlib.util
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np


@unittest.skipUnless(importlib.util.find_spec("torch"), "Install requirements-train.txt to test U-Net")
class FinalEvaluationTests(unittest.TestCase):
    def test_sensitivity_reuses_the_full_test_predictions(self):
        from evaluate_test import EARLY_FUSION, SIAMESE, evaluate_test

        samples = {
            "test_1.png": ([0, 1], [0, 2], [0, 1]),
            "test_2.png": ([], [2], []),
            "test_9.png": ([0], [], [0]),
            "test_19.png": ([], [0, 1], [0]),
        }
        selection = {
            "pixel_difference": {"threshold": 30},
            EARLY_FUSION: {"threshold": 0.35},
            SIAMESE: {"threshold": 0.50},
        }

        def load_sample(split, sample):
            changed, early, siamese = samples[sample]
            before = np.zeros((1024, 1024, 3), dtype=np.uint8)
            after = before.copy()
            reference = np.zeros((1024, 1024), dtype=bool)
            reference[0, changed] = True
            after[0, early, 0] = 255
            after[0, siamese, 1] = 255
            return before, after, reference

        def predict(channel, before, after, reference, threshold, batch_size, device):
            self.assertEqual(threshold, (0.35, 0.50)[channel])
            return after[:, :, channel] > 0

        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            with patch("evaluate_test.TEST_SAMPLES", frozenset(samples)), \
                 patch("evaluate_test.list_samples", return_value=list(samples)), \
                 patch("evaluate_test.load_pair", side_effect=load_sample), \
                 patch("evaluate_test.predict_mask", side_effect=predict) as inference:
                summary = evaluate_test(output / "test", {EARLY_FUSION: 0, SIAMESE: 1},
                                        selection, 2, "cpu", output)

            self.assertEqual(inference.call_count, 8)
            self.assertEqual(summary["excluded_from_sensitivity"], ["test_19.png", "test_9.png"])
            full, sensitivity = (summary["subsets"][name] for name in ("full", "sensitivity"))
            self.assertEqual((full["samples"], sensitivity["samples"]), (4, 2))
            self.assertEqual(full["methods"][EARLY_FUSION]["counts"], {"tp": 1, "fp": 4, "fn": 2})
            self.assertEqual(sensitivity["methods"][EARLY_FUSION]["counts"], {"tp": 1, "fp": 2, "fn": 1})
            self.assertEqual(full["methods"][SIAMESE]["counts"], {"tp": 3, "fp": 1, "fn": 0})
            self.assertEqual(sensitivity["methods"][SIAMESE]["metrics"]["F1"], 1.0)
            self.assertTrue((output / "false_alarms.png").is_file())
            self.assertTrue((output / "missed_changes.png").is_file())
            with (output / "per_image.csv").open(newline="") as file:
                rows = list(csv.DictReader(file))
            self.assertEqual(len(rows), 12)
            self.assertEqual(sum(row["included_in_sensitivity"] == "True" for row in rows), 6)
            with (output / "test_results.csv").open(newline="") as file:
                self.assertEqual(len(list(csv.DictReader(file))), 6)

    def test_frozen_models_accept_legacy_early_fusion_and_reject_a_different_epoch(self):
        import torch

        from early_fusion_unet import UNET_MODELS
        from evaluate_test import EARLY_FUSION, FROZEN_MODELS, SIAMESE, load_frozen_models

        with tempfile.TemporaryDirectory() as directory:
            runs = Path(directory)
            for method, (run, epoch, _) in FROZEN_MODELS.items():
                path = runs / run / "full" / "best.pt"
                path.parent.mkdir(parents=True)
                checkpoint = {"epoch": epoch, "model_state": UNET_MODELS[method]().state_dict()}
                if method == SIAMESE:
                    checkpoint["architecture"] = method
                torch.save(checkpoint, path)

            models, selection = load_frozen_models(runs, "cpu")
            self.assertEqual(models[EARLY_FUSION].architecture, EARLY_FUSION)
            self.assertEqual(selection[SIAMESE]["epoch"], 30)
            self.assertEqual(len(selection[SIAMESE]["checkpoint_sha256"]), 64)
            path = Path(selection[SIAMESE]["checkpoint"])
            checkpoint = torch.load(path, map_location="cpu", weights_only=True)
            checkpoint["epoch"] = 29
            torch.save(checkpoint, path)
            with self.assertRaisesRegex(ValueError, "Expected siamese_unet from epoch 30"):
                load_frozen_models(runs, "cpu")


if __name__ == "__main__":
    unittest.main()
