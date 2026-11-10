import importlib.util
import io
import json
import os
import subprocess
import sys
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from unittest.mock import Mock, patch

import numpy as np


class ColabMixedLoggingTests(unittest.TestCase):
    def test_notebook_keeps_live_output_and_error_in_drive_log(self):
        notebook = Path(__file__).resolve().parents[1] / "notebooks" / "train_unet_colab.ipynb"
        cells = json.loads(notebook.read_text(encoding="utf-8"))["cells"]
        smoke = next(cell for cell in cells if cell["id"] == "mixed-smoke")
        with tempfile.TemporaryDirectory() as directory, redirect_stdout(io.StringIO()) as console:
            root = Path(directory)
            data = root / "data"
            for split in ("train", "val", "test"):
                (data / split).mkdir(parents=True)
            (root / "train_mixed_unet.py").write_text(
                "import sys\n"
                "print('Checking data')\n"
                "if 'train' in sys.argv:\n"
                "    print('Failed to read sample.png', file=sys.stderr)\n"
                "    sys.exit(2)\n", encoding="utf-8")
            namespace = {"DATA_ROOT": data, "REPO_DIR": root, "RUNS_ROOT": root / "drive",
                         "get_s2looking_archive": lambda: root / "S2Looking.zip", "REVISION": "test",
                         "os": os, "sys": sys, "subprocess": subprocess}
            exec("".join(smoke["source"]), namespace)
            with self.assertRaisesRegex(RuntimeError, "Failed to read sample.png"):
                namespace["run_mixed"]("train")

            logs = namespace["RUNS_ROOT"] / "siamese_v1_mixed_e30"
            self.assertIn("Çıkış kodu: 0", (logs / "smoke.log").read_text(encoding="utf-8"))
            self.assertIn("Failed to read sample.png", console.getvalue())
            log = (logs / "train.log").read_text(encoding="utf-8")
            self.assertIn("Checking data\nFailed to read sample.png", log)
            self.assertIn("Çıkış kodu: 2", log)

            broken_log = Mock()
            broken_log.write.side_effect = [None, OSError("simulated Drive write failure")]
            with patch.object(Path, "open", return_value=broken_log):
                with self.assertRaisesRegex(RuntimeError, "Failed to read sample.png"):
                    namespace["run_mixed"]("train")
            self.assertIn("simulated Drive write failure", console.getvalue())
            broken_log.close.assert_called_once()

            with patch.object(Path, "open", side_effect=OSError("simulated Drive open failure")):
                namespace["run_mixed"]("smoke")
            self.assertIn("simulated Drive open failure", console.getvalue())


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
