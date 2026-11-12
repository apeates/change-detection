"""Freeze the BIT checkpoint and threshold using official LEVIR validation only."""

import argparse
import hashlib
import json
import tempfile
from pathlib import Path

import torch

from bit import BIT
from evaluate_unet import SWEEP_THRESHOLDS, evaluate
from levir_data import DEFAULT_DATA_ROOT, official_samples, split_fingerprint
from train_unet import best_row, load_model


BIT_EPOCHS = 30
BIT_STEPS_PER_EPOCH = 3560
BIT_UPDATES = BIT_EPOCHS * BIT_STEPS_PER_EPOCH


def checkpoint_hash(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def completed_bit_checkpoint(run):
    full = run / "full"
    last = torch.load(full / "last.pt", map_location="cpu", weights_only=True)
    config = last["config"]
    history = last["history"]
    if (last["architecture"] != BIT.architecture or last["epoch"] != BIT_EPOCHS
            or config["schedule_epochs"] != BIT_EPOCHS or config["batch_size"] != 2
            or config["seed"] != 42 or config["augment"] or config["smoke_tiles"] is not None
            or [row["epoch"] for row in history] != list(range(1, BIT_EPOCHS + 1))
            or any(row["optimizer_steps"] != row["epoch"] * BIT_STEPS_PER_EPOCH for row in history)):
        raise ValueError("Finish the full BIT protocol before locking the threshold.")
    path = full / "best.pt"
    best = torch.load(path, map_location="cpu", weights_only=True)
    if (best["architecture"] != BIT.architecture or best["epoch"] != best_row(history)["epoch"]
            or best["config"] != config):
        raise ValueError("The best checkpoint does not match the completed run.")
    return path, best["epoch"], config


def read_bit_selection(run):
    validation = run / "validation"
    selection = json.loads((validation / "selection.json").read_text())
    sweep = json.loads((validation / "summary.json").read_text())["threshold_sweep"]
    checkpoint = run / "full" / "best.pt"
    if (selection["architecture"] != BIT.architecture or selection["split"] != "val"
            or selection["samples"] != 64 or selection["optimizer_steps"] != BIT_UPDATES
            or selection["threshold"] not in SWEEP_THRESHOLDS
            or selection["threshold"] != sweep["best_threshold"]
            or selection["validation_f1"] != sweep["best_f1"]
            or selection["checkpoint_sha256"] != checkpoint_hash(checkpoint)):
        raise ValueError("BIT test requires the frozen validation selection and matching checkpoint.")
    return checkpoint, selection


def freeze_validation(run, split, device):
    output = run / "validation"
    if output.exists():
        return read_bit_selection(run)[1]

    path, epoch, config = completed_bit_checkpoint(run)
    samples = official_samples(split)
    fingerprint = split_fingerprint(split, samples)
    if fingerprint != config["fingerprints"]["val"]:
        raise ValueError("Validation data differs from the training checkpoint.")
    sha256 = checkpoint_hash(path)
    model = load_model(path, device)
    with tempfile.TemporaryDirectory(dir=run) as temporary:
        staged = Path(temporary) / "validation"
        report = evaluate(model, split, 0.50, 2, device, staged, path.name, sweep=True, samples=samples)
        selection = {
            "architecture": BIT.architecture, "variant": config["variant"], "split": split.name,
            "samples": len(samples), "checkpoint": str(path), "checkpoint_sha256": sha256,
            "epoch": epoch, "threshold": report["threshold_sweep"]["best_threshold"],
            "validation_f1": report["threshold_sweep"]["best_f1"], "seed": config["seed"],
            "validation_fingerprint": fingerprint, "optimizer_steps": BIT_UPDATES, "revision": config["revision"],
        }
        (staged / "selection.json").write_text(json.dumps(selection, indent=2) + "\n")
        staged.rename(output)
    return selection


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    args = parser.parse_args()
    if args.device == "cuda" and not torch.cuda.is_available():
        parser.error("CUDA is unavailable. Enable a GPU runtime.")
    if args.device == "cpu":
        torch.set_num_threads(min(4, torch.get_num_threads()))
    print(json.dumps(freeze_validation(args.run, args.data_root / "val", args.device), indent=2))


if __name__ == "__main__":
    main()
