"""Measure the frozen LEVIR-CD Siamese U-Net on the official S2Looking test split."""

import argparse
import csv
import json
import tempfile
import zipfile
import zlib
from pathlib import Path, PurePosixPath

import numpy as np
import torch
from PIL import Image

from evaluate_test import SIAMESE, load_frozen_model
from evaluate_unet import COUNT_FIELDS, evaluate, resampled_f1
from levir_data import DatasetError


TEST_PAIR_COUNT = 1000
IMAGE_SHAPE = (1024, 1024)
ARCHIVE_ID = "1YAUrUp3QtZ6VMM1nSuqOjveIVO9ADOY1"


class S2LookingSplit:
    """Index the author's PNG files directly inside the ZIP."""

    def __init__(self, archive, split, expected):
        self.archive = archive
        self.entries = {group: {} for group in ("image1", "image2", "label", "label1", "label2")}
        for entry in archive.infolist():
            path = PurePosixPath(entry.filename)
            if (entry.is_dir() or len(path.parts) < 3 or path.parts[-3].lower() != split
                    or path.suffix.lower() != ".png" or path.name.startswith(".") or "__MACOSX" in path.parts):
                continue
            group = path.parent.name.lower()
            if group not in self.entries:
                continue
            if path.name in self.entries[group]:
                raise DatasetError(f"Duplicate S2Looking {split} file: {group}/{path.name}.")
            self.entries[group][path.name] = entry

        self.labels = ("label",) if self.entries["label"] else ("label1", "label2")
        groups = ("image1", "image2", *self.labels)
        names = set.union(*(set(self.entries[group]) for group in groups))
        missing = [f"{group}/{name}" for group in groups for name in sorted(names - self.entries[group].keys())]
        if len(names) != expected or missing:
            raise DatasetError(f"Expected {expected} matched S2Looking {split} pairs with binary labels, "
                               f"found {len(names)} names. Missing: {', '.join(missing) or 'none listed'}.")
        self.samples = sorted(names)

    def read_image(self, group, sample, mode):
        entry = self.entries[group][sample]
        try:
            with self.archive.open(entry) as file, Image.open(file) as image:
                pixels = np.array(image.convert(mode))
        except (OSError, ValueError, EOFError, zipfile.BadZipFile, zlib.error) as error:
            raise DatasetError(f"Cannot read S2Looking ZIP member {entry.filename}: {error}") from error
        if pixels.shape[:2] != IMAGE_SHAPE:
            raise DatasetError(f"{group}/{sample}: expected 1024×1024 pixels.")
        return pixels

    def load_pair(self, sample):
        before = self.read_image("image1", sample, "RGB")
        after = self.read_image("image2", sample, "RGB")
        reference = np.zeros(IMAGE_SHAPE, dtype=bool)
        for group in self.labels:
            mask = self.read_image(group, sample, "L")
            if np.any((mask != 0) & (mask != 1) & (mask != 255)):
                raise DatasetError(f"{group}/{sample}: expected binary labels encoded as 0/1 or 0/255.")
            reference |= mask != 0
        return before, after, reference


class S2LookingTest(S2LookingSplit):
    def __init__(self, archive):
        super().__init__(archive, "test", TEST_PAIR_COUNT)


def evaluate_transfer(archive, runs_root, output, batch_size, device, revision=None):
    # The source model and threshold were selected on LEVIR-CD validation and used for its final test.
    model, selection = load_frozen_model(runs_root, SIAMESE, device)
    source_path = runs_root / "levir_final_test" / "summary.json"
    source = json.loads(source_path.read_text(encoding="utf-8"))
    frozen = source["selection"][SIAMESE]
    if any(selection[key] != frozen[key] for key in ("checkpoint_sha256", "epoch", "threshold")):
        raise ValueError("The checkpoint and threshold must match the final LEVIR-CD test report.")

    test = S2LookingTest(archive)
    summary = evaluate(model, Path("test"), selection["threshold"], batch_size, device, output,
                       Path(selection["checkpoint"]).name, dataset="S2Looking",
                       samples=test.samples, read_sample=test.load_pair)
    with (output / "per_image.csv").open(newline="", encoding="utf-8") as file:
        counts = np.array([[int(row[field]) for field in COUNT_FIELDS] for row in csv.DictReader(file)])
    indices = np.random.default_rng(42).integers(len(counts), size=(5000, len(counts)), dtype=np.int32)
    f1_interval = np.quantile(resampled_f1(counts, indices), (0.025, 0.975))
    levir_f1 = source["subsets"]["full"]["methods"][SIAMESE]["metrics"]["F1"]
    summary.update({
        "protocol": "zero_shot_transfer",
        "target": "new_or_demolished_building",
        "target_mask_folders": list(test.labels),
        "archive_id": ARCHIVE_ID,
        "revision": revision,
        "device": torch.cuda.get_device_name() if device == "cuda" else device,
        "batch_size": batch_size,
        "selection": selection,
        "f1_ci95": f1_interval.tolist(),
        "positive_pixel_rate": (summary["counts"]["tp"] + summary["counts"]["fn"])
        / (summary["samples"] * np.prod(IMAGE_SHAPE)),
        "levir_cd_test_f1": levir_f1,
        "f1_difference_from_levir_cd": summary["metrics"]["F1"] - levir_f1,
    })
    (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--runs-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--revision")
    args = parser.parse_args()
    if args.batch_size < 1:
        parser.error("Batch size must be positive.")
    if args.device == "cuda" and not torch.cuda.is_available():
        parser.error("CUDA is unavailable. Enable a GPU runtime.")
    if args.output_dir.exists():
        parser.error(f"Output already exists: {args.output_dir}. Choose a new output directory.")
    if args.device == "cpu":
        torch.set_num_threads(min(4, torch.get_num_threads()))
    try:
        with zipfile.ZipFile(args.archive) as archive:
            args.output_dir.parent.mkdir(parents=True, exist_ok=True)
            with tempfile.TemporaryDirectory(dir=args.output_dir.parent) as directory:
                staged = Path(directory) / "report"
                evaluate_transfer(archive, args.runs_root, staged, args.batch_size, args.device, args.revision)
                staged.rename(args.output_dir)
    except (OSError, ValueError, RuntimeError, zipfile.BadZipFile) as error:
        parser.error(str(error))
    print(f"Saved: {args.output_dir}")


if __name__ == "__main__":
    main()
