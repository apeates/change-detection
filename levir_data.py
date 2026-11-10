"""Read and validate paired LEVIR-CD images and reference masks."""

import argparse
from pathlib import Path

import numpy as np
from PIL import Image


DEFAULT_DATA_ROOT = Path(__file__).resolve().parent / "data" / "levir-cd"
PAIR_FOLDERS = ("A", "B", "label")


class DatasetError(ValueError):
    """A dataset file is missing, unreadable, or incompatible with its pair."""


def list_samples(split: Path) -> list[str]:
    """Return sorted PNG names, rejecting any unpaired file in the split."""
    files_by_folder = {}
    for folder in PAIR_FOLDERS:
        directory = split / folder
        if not directory.is_dir():
            raise DatasetError(f"Missing dataset folder: {directory}")
        files_by_folder[folder] = {
            path.name for path in directory.iterdir()
            if path.is_file() and path.suffix.lower() == ".png"
        }

    samples = sorted(set.union(*files_by_folder.values()))
    if not samples:
        raise DatasetError(f"No PNG samples found in {split}")
    unmatched = []
    for sample in samples:
        missing = [folder for folder, names in files_by_folder.items() if sample not in names]
        if missing:
            unmatched.append(f"{sample}: missing file in {', '.join(missing)}")
    if unmatched:
        raise DatasetError(f"{split}: {'; '.join(unmatched)}")
    return samples


def _read_image(path: Path, mode: str) -> np.ndarray:
    try:
        with Image.open(path) as image:
            return np.array(image.convert(mode))
    except FileNotFoundError as error:
        raise DatasetError(f"Missing file: {path}") from error
    except (OSError, ValueError) as error:
        raise DatasetError(f"Cannot read {path}: {error}") from error


def load_pair(split: Path, sample: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Load two uint8 RGB arrays and a boolean mask with matching spatial dimensions."""
    if Path(sample).name != sample or Path(sample).suffix.lower() != ".png":
        raise DatasetError(f"Expected a PNG filename, got {sample!r}")

    before = _read_image(split / "A" / sample, "RGB")
    after = _read_image(split / "B" / sample, "RGB")
    label = _read_image(split / "label" / sample, "L")
    if before.shape[:2] != after.shape[:2] or before.shape[:2] != label.shape:
        sizes = [f"{image.shape[1]}x{image.shape[0]}" for image in (before, after, label)]
        raise DatasetError(f"{sample}: dimension mismatch (A={sizes[0]}, B={sizes[1]}, label={sizes[2]})")
    return before, after, label >= 128


def audit_split(split: Path) -> tuple[int, int]:
    """Return the number of valid samples and masks without change pixels."""
    samples = list_samples(split)
    empty_masks = 0
    for sample in samples:
        _, _, mask = load_pair(split, sample)
        if not mask.any():
            empty_masks += 1
    return len(samples), empty_masks


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--split", choices=("train", "val", "test"), default="val")
    args = parser.parse_args()
    try:
        valid_samples, empty_masks = audit_split(args.data_root / args.split)
    except DatasetError as error:
        parser.error(str(error))
    print(f"Split: {args.split}")
    print(f"Valid image pairs: {valid_samples}")
    print(f"Masks without change: {empty_masks}")


if __name__ == "__main__":
    main()
