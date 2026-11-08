import csv
import importlib.util
import io
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from unittest.mock import patch

import numpy as np
from PIL import Image


@unittest.skipUnless(importlib.util.find_spec("torch"), "Install requirements-train.txt to test U-Net")
class UNetTrainingTests(unittest.TestCase):
    def test_learning_rate_schedule_endpoints(self):
        from train_unet import epoch_learning_rate

        self.assertEqual(epoch_learning_rate(9), 0.001)
        values = [epoch_learning_rate(epoch, 10) for epoch in range(1, 11)]
        self.assertEqual((values[0], values[-1]), (0.001, 0.0001))
        for first, second in zip(values, values[1:]):
            self.assertAlmostEqual(first - second, 0.0001)

    def write_pair(self, root, split, name, after, mask):
        base = root / split
        before = np.zeros_like(after)
        for folder, pixels in (("A", before), ("B", after), ("label", mask)):
            (base / folder).mkdir(parents=True, exist_ok=True)
            Image.fromarray(pixels).save(base / folder / name)

    def test_output_matches_input_spatial_shape(self):
        import torch

        from early_fusion_unet import EarlyFusionUNet

        model = EarlyFusionUNet().eval()
        with torch.inference_mode():
            output = model(torch.zeros(1, 6, 256, 256))
        self.assertEqual(tuple(output.shape), (1, 1, 256, 256))
        self.assertTrue(torch.isfinite(output).all().item())

    def test_siamese_prediction_is_symmetric_and_survives_checkpoint_reload(self):
        import torch

        from early_fusion_unet import SiameseUNet
        from train_unet import load_model, save_checkpoint

        torch.manual_seed(42)
        model = SiameseUNet().eval()
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        inputs = torch.rand(2, 6, 32, 48)
        swapped = torch.cat((inputs[:, 3:], inputs[:, :3]), dim=1)
        with tempfile.TemporaryDirectory() as directory, torch.inference_mode():
            path = Path(directory) / "checkpoint.pt"
            save_checkpoint(path, model, optimizer, epoch=1, seed=42)
            restored = load_model(path)
            expected = model(inputs)
            self.assertEqual(expected.shape, (2, 1, 32, 48))
            self.assertTrue(torch.isfinite(expected).all().item())
            self.assertIsInstance(restored, SiameseUNet)
            torch.testing.assert_close(model(swapped), expected, rtol=0, atol=0)
            torch.testing.assert_close(restored(inputs), expected, rtol=0, atol=0)

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

    def test_validation_pools_pixels_and_counts_false_alarms(self):
        import torch

        from train_unet import validate_epoch

        class RedDifference(torch.nn.Module):
            def forward(self, inputs):
                return inputs[:, 3:4] - inputs[:, :1] - 0.5

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            after = np.zeros((256, 256, 3), dtype=np.uint8)
            after[0, :2, 0] = 255
            label = np.zeros((256, 256), dtype=np.uint8)
            label[0, [0, 2]] = 255
            self.write_pair(root, "val", "changed.png", after, label)
            self.write_pair(root, "val", "unchanged.png", after, np.zeros_like(label))
            scores = validate_epoch(RedDifference(), root / "val", ["changed.png", "unchanged.png"], 2, "cpu")

        self.assertEqual(scores, {"Precision": 0.25, "Recall": 0.5, "F1": 1 / 3, "IoU": 0.2})

    def test_resume_keeps_history_and_uses_best_validation_epoch(self):
        import torch

        from train_unet import main

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            run = root / "resumed"
            uninterrupted = root / "uninterrupted"
            for split, x in (("train", 48), ("val", 96)):
                after = np.zeros((256, 256, 3), dtype=np.uint8)
                label = np.zeros((256, 256), dtype=np.uint8)
                after[64:128, x:x + 64] = 255
                label[64:128, x:x + 64] = 255
                self.write_pair(root, split, f"{split}.png", after, label)

            def run_trainer(output, *args):
                argv = ["train_unet.py", "--data-root", str(root), "--output-dir", str(output), *args]
                with patch("sys.argv", argv), redirect_stdout(io.StringIO()) as stdout:
                    main()
                    return stdout.getvalue()

            run_trainer(run, "--epochs", "1")
            legacy = torch.load(run / "last.pt", map_location="cpu", weights_only=True)
            legacy.pop("architecture")
            legacy["config"].pop("augment")
            torch.save(legacy, run / "last.pt")
            with self.assertRaises(SystemExit), patch("sys.stderr", io.StringIO()) as stderr:
                run_trainer(run, "--epochs", "2", "--resume", "--model", "siamese_unet")
            self.assertIn("Checkpoint uses early_fusion_unet", stderr.getvalue())
            with self.assertRaises(SystemExit), patch("sys.stderr", io.StringIO()) as stderr:
                run_trainer(run, "--epochs", "2", "--resume", "--augment")
            self.assertIn("run settings changed", stderr.getvalue())
            (run / "history.csv").unlink()
            (run / "best.pt").unlink()
            self.assertIn("Resuming from epoch 1", run_trainer(run, "--epochs", "2", "--resume"))
            run_trainer(uninterrupted, "--epochs", "2")

            def checkpoint(path):
                return torch.load(path, map_location="cpu", weights_only=True)

            with (run / "history.csv").open(newline="", encoding="utf-8") as file:
                rows = list(csv.DictReader(file))
            self.assertEqual([int(row["epoch"]) for row in rows], [1, 2])
            best_epoch = int(max(rows, key=lambda row: float(row["val_f1"]))["epoch"])
            self.assertEqual(checkpoint(run / "best.pt")["epoch"], best_epoch)
            resumed = checkpoint(run / "last.pt")
            expected = checkpoint(uninterrupted / "last.pt")
            self.assertEqual(resumed["epoch"], 2)
            self.assertEqual(resumed["history"], expected["history"])
            torch.testing.assert_close(resumed["model_state"], expected["model_state"])

            linear_run = root / "linear"
            from train_unet import train_epoch

            with patch("train_unet.save_history", side_effect=RuntimeError("Interrupted")):
                with self.assertRaises(SystemExit), patch("sys.stderr", io.StringIO()):
                    run_trainer(linear_run, "--epochs", "2", "--lr-schedule", "linear", "--augment")
            with patch("train_unet.train_epoch", wraps=train_epoch) as resumed_train:
                run_trainer(linear_run, "--epochs", "2", "--lr-schedule", "linear", "--augment", "--resume")
                self.assertEqual(resumed_train.call_count, 1)
            full_augmented = root / "full_augmented"
            run_trainer(full_augmented, "--epochs", "2", "--lr-schedule", "linear", "--augment")
            linear = checkpoint(linear_run / "last.pt")
            expected = checkpoint(full_augmented / "last.pt")
            self.assertTrue(linear["config"]["augment"])
            self.assertEqual([row["learning_rate"] for row in linear["history"]], [0.001, 0.0001])
            self.assertEqual(linear["optimizer_state"]["param_groups"][0]["lr"], 0.0001)
            self.assertEqual(linear["history"], expected["history"])
            torch.testing.assert_close(linear["model_state"], expected["model_state"], rtol=0, atol=0)
            (linear_run / "history.csv").unlink()
            with patch("train_unet.train_epoch") as already_complete:
                run_trainer(linear_run, "--epochs", "2", "--lr-schedule", "linear", "--augment", "--resume")
                already_complete.assert_not_called()
            self.assertTrue((linear_run / "history.csv").is_file())


if __name__ == "__main__":
    unittest.main()
