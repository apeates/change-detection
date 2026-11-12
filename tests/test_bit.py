import csv
import importlib.util
import io
import json
import tempfile
import unittest
from contextlib import redirect_stdout
from functools import partial
from pathlib import Path
from unittest.mock import patch

import numpy as np
from PIL import Image


@unittest.skipUnless(importlib.util.find_spec("torch"), "Install requirements-train.txt to test BIT")
class BITTests(unittest.TestCase):
    def test_bit_resume_matches_uninterrupted_training(self):
        import torch

        import train_unet
        from bit import BIT
        from levir_tiles import iter_tiles

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            rng = np.random.default_rng(42)
            before = rng.integers(0, 256, (64, 64, 3), dtype=np.uint8)
            after = np.flip(before, axis=0).copy()
            label = np.full((64, 64), 255, dtype=np.uint8)
            for split in ("train", "val"):
                for folder, array in (("A", before), ("B", after), ("label", label)):
                    path = root / split / folder
                    path.mkdir(parents=True)
                    Image.fromarray(array).save(path / "sample.png")

            def run(output, *extra):
                args = ["train_unet.py", "--model", "bit", "--data-root", str(root),
                        "--output-dir", str(output), "--epochs", "2", "--lr-schedule", "linear",
                        "--smoke-tiles", "4", *extra]
                with patch("sys.argv", args), redirect_stdout(io.StringIO()) as output_text:
                    train_unet.main()
                    return output_text.getvalue()

            original = train_unet.train_epoch

            def interrupt_second_epoch(*args, **kwargs):
                if (root / "resumed" / "last.pt").exists():
                    raise KeyboardInterrupt
                return original(*args, **kwargs)

            with patch("train_unet.iter_tiles", partial(iter_tiles, tile_size=32)), \
                    patch("train_unet.save_smoke_preview", return_value=0):
                run(root / "continuous")
                with patch("train_unet.train_epoch", side_effect=interrupt_second_epoch), self.assertRaises(KeyboardInterrupt):
                    run(root / "resumed")
                self.assertIn("Resuming from epoch 1", run(root / "resumed", "--resume"))

            uninterrupted = torch.load(root / "continuous" / "last.pt", weights_only=True)
            resumed_path = root / "resumed" / "last.pt"
            resumed = torch.load(resumed_path, weights_only=True)
            self.assertEqual([row["optimizer_steps"] for row in resumed["history"]], [2, 4])
            for name, parameter in resumed["model_state"].items():
                torch.testing.assert_close(parameter, uninterrupted["model_state"][name], rtol=0, atol=0)
            torch.manual_seed(42)
            self.assertFalse(torch.equal(BIT().classifier[-1].weight, resumed["model_state"]["classifier.3.weight"]))
            with torch.inference_mode():
                output = train_unet.load_model(resumed_path)(torch.zeros(1, 6, 32, 48))
            self.assertEqual(tuple(output.shape), (1, 1, 32, 48))
            self.assertTrue(torch.isfinite(output).all())

    def test_test_selection_rejects_changed_checkpoint_or_threshold(self):
        from evaluate_bit import BIT_UPDATES, checkpoint_hash, read_bit_selection

        with tempfile.TemporaryDirectory() as directory:
            run = Path(directory)
            (run / "full").mkdir()
            validation = run / "validation"
            validation.mkdir()
            checkpoint = run / "full" / "best.pt"
            checkpoint.write_bytes(b"chosen checkpoint")
            (validation / "summary.json").write_text(json.dumps(
                {"threshold_sweep": {"best_threshold": 0.65, "best_f1": 0.8}}))
            selection_file = validation / "selection.json"
            selection = {"architecture": "bit", "split": "val", "samples": 64, "threshold": 0.65,
                         "validation_f1": 0.8, "optimizer_steps": BIT_UPDATES,
                         "checkpoint_sha256": checkpoint_hash(checkpoint)}
            selection_file.write_text(json.dumps(selection))
            self.assertEqual(read_bit_selection(run), (checkpoint, selection))
            selection_file.write_text(json.dumps({**selection, "threshold": 0.35}))
            with self.assertRaisesRegex(ValueError, "frozen validation"):
                read_bit_selection(run)
            selection_file.write_text(json.dumps(selection))
            checkpoint.write_bytes(b"different checkpoint")
            with self.assertRaisesRegex(ValueError, "matching checkpoint"):
                read_bit_selection(run)

    def test_classical_thresholds_and_shift_preserve_valid_pixels(self):
        from benchmark_levir import cva_threshold, otsu_threshold, shift_right

        self.assertEqual(otsu_threshold(np.array([5, 0, 0, 5])), 0)
        self.assertEqual(otsu_threshold(np.array([0, 4, 0, 0])), 1)
        self.assertEqual(cva_threshold(np.array([0, 0, 3]), np.array([2, 3, 0])), 1)
        image = np.arange(12).reshape(1, 4, 3)
        np.testing.assert_array_equal(shift_right(image, 2)[:, 2:], image[:, :2])

    def test_controlled_reports_cover_five_methods_and_resource_deltas(self):
        from benchmark_levir import CVA, OTSU, SHIFTS, write_reports
        from evaluate_test import EARLY_FUSION, PIXEL_METHOD, SIAMESE

        thresholds = {OTSU: 23, CVA: 61}
        selection = {method: {"threshold": threshold} for method, threshold in
                     ((PIXEL_METHOD, 30), (EARLY_FUSION, 0.35), (SIAMESE, 0.50))}
        methods = (PIXEL_METHOD, OTSU, CVA, EARLY_FUSION, SIAMESE)
        totals = {shift: {method: {"tp": 20 - shift, "fp": shift, "fn": shift} for method in methods}
                  for shift in SHIFTS}
        classical = {method: totals[0][method] for method in thresholds}
        resources = {
            "device": "test GPU", "sample": "val_1.png", "tiles": 16, "batch_size": 2,
            "warmup_passes": 3, "timed_passes": 10,
            "models": {
                EARLY_FUSION: {"parameters": 100, "median_forward_ms_per_pair": 3, "peak_allocated_bytes": 2**20},
                SIAMESE: {"parameters": 200, "median_forward_ms_per_pair": 4.5, "peak_allocated_bytes": 3 * 2**20},
            },
        }
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            write_reports(output, totals, classical, thresholds, resources, selection)
            with (output / "robustness.csv").open() as file:
                rows = list(csv.DictReader(file))
            self.assertEqual([(int(row["shift_px"]), row["method"]) for row in rows],
                             [(shift, method) for shift in SHIFTS for method in methods])
            with (output / "classical_test.csv").open() as file:
                self.assertEqual([row["method"] for row in csv.DictReader(file)], [OTSU, CVA])
            protocol = json.loads((output / "protocol.json").read_text())
            self.assertEqual((protocol["validation_pairs"], protocol["test_pairs"]), (64, 128))
            self.assertEqual(protocol["thresholds"], {**thresholds, PIXEL_METHOD: 30, EARLY_FUSION: 0.35, SIAMESE: 0.50})
            self.assertTrue((output / "resources.json").is_file())
            self.assertGreater((output / "robustness.png").stat().st_size, 0)
            summary = (output / "summary.md").read_text(encoding="utf-8")
            for value in ("-20.00", "+1.50 ms/çift", "+2.00 MiB", "12 tamamlanmış tur"):
                self.assertIn(value, summary)
