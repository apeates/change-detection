import importlib.util
import unittest
from unittest.mock import patch

import numpy as np


@unittest.skipUnless(importlib.util.find_spec("torch"), "Install requirements-train.txt to test U-Net")
class MixedTrainingTests(unittest.TestCase):
    def test_batches_balance_sources_and_seek_across_corpus_cycles(self):
        from mixed_data import LEVIR, S2LOOKING, Split, mixed_batches

        datasets = {
            LEVIR: {"train": Split(["l1", "l2"], lambda sample: (sample,))},
            S2LOOKING: {"train": Split(["s1", "s2", "s3"], lambda sample: (sample,))},
        }
        with patch("mixed_data.TILES_PER_PAIR", 2), patch(
                "mixed_data.iter_tiles", side_effect=lambda sample: [f"{sample}-0", f"{sample}-1"]):
            complete = list(mixed_batches(datasets, seed=42, start=0, steps=19))
            resumed = list(mixed_batches(datasets, seed=42, start=7, steps=12))

        self.assertEqual(resumed, complete[7:])
        self.assertTrue(all(lev.startswith("l") and s2.startswith("s") for lev, s2 in complete))
        self.assertEqual(len(set(lev for lev, _ in complete[:4])), 4)
        self.assertEqual(len(set(s2 for _, s2 in complete[:6])), 6)

    def test_selection_enforces_levir_limit_and_breaks_ties(self):
        from train_mixed_unet import select_epoch

        def score(epoch, levir, s2looking):
            return {"epoch": epoch, "levir_val_f1": levir, "s2looking_val_f1": s2looking}

        history = [score(1, 0.76, 0.40), score(2, 0.78, 0.30), score(3, 0.79, 0.30),
                   score(4, 0.79, 0.30), score(5, 0.80, 0.20)]
        self.assertEqual(select_epoch(history, 0.79)["epoch"], 3)

        no_eligible = [score(1, 0.76, 0.40), score(2, 0.77, 0.30), score(3, 0.77, 0.35),
                       score(4, 0.77, 0.35)]
        self.assertEqual(select_epoch(no_eligible, 0.79)["epoch"], 3)

    def test_audit_detects_decoded_duplicates_between_sources_and_splits(self):
        from levir_data import DatasetError
        from mixed_data import LEVIR, S2LOOKING, Split, audit_splits

        def reader(value):
            image = np.full((2, 2, 3), value, dtype=np.uint8)
            return lambda sample: (image, image, np.zeros((2, 2), dtype=bool))

        datasets = {
            LEVIR: {"train": Split(["a.png"], reader(1))},
            S2LOOKING: {"val": Split(["b.png"], reader(1))},
        }
        with patch("mixed_data.IMAGE_SHAPE", (2, 2)):
            with self.assertRaisesRegex(DatasetError, r"LEVIR-CD/train/a.png.*S2Looking/val/b.png"):
                audit_splits(datasets)

    def test_paired_comparison_excludes_the_same_images_and_rejects_different_labels(self):
        from train_mixed_unet import compare_predictions

        reference = {"a": [2, 1, 2], "b": [0, 3, 1]}
        mixed = {"a": [3, 0, 1], "b": [1, 0, 0]}
        result = compare_predictions(reference, mixed, excluded={"b"})
        self.assertEqual(result["samples"], 1)
        self.assertEqual(result["mixed"]["counts"], {"tp": 3, "fp": 0, "fn": 1})
        self.assertAlmostEqual(result["f1_delta"], 6 / 7 - 4 / 7)
        self.assertTrue(np.allclose(result["f1_delta_ci95"], [2 / 7, 2 / 7]))
        with self.assertRaisesRegex(ValueError, "same number of positive pixels"):
            compare_predictions(reference, {**mixed, "a": [3, 0, 0]})


if __name__ == "__main__":
    unittest.main()
