"""Compare equal-budget Siamese U-Net runs on two building-change datasets."""

import argparse
import csv
import hashlib
import json
import sys
import tempfile
import zipfile
from importlib.metadata import version
from pathlib import Path

import numpy as np
import torch

from baseline_diff import metrics_from_counts
from early_fusion_unet import SiameseUNet
from evaluate_test import SENSITIVITY_EXCLUSIONS, SIAMESE, load_frozen_model
from evaluate_unet import COUNT_FIELDS, evaluate, resampled_f1
from levir_data import DEFAULT_DATA_ROOT
from levir_tiles import TILE_SIZE
from mixed_data import LEVIR, LEVIR_COUNTS, S2LOOKING, TILES_PER_PAIR, audit_splits, mixed_batches, open_splits
from train_unet import (START_LR, copy_checkpoint, epoch_learning_rate, load_model, save_checkpoint, save_history,
                        train_epoch, validate_epoch)


RUN_NAME = "siamese_v1_mixed_e30"
EPOCHS = 30
BATCH_SIZE = 2
SEED = 42
STEPS_PER_EPOCH = LEVIR_COUNTS["train"] * TILES_PER_PAIR // BATCH_SIZE
THRESHOLD = 0.50
MAX_LEVIR_LOSS = 0.01
MIN_S2LOOKING_GAIN = 0.05
HISTORY_FIELDS = ("epoch", "learning_rate", "train_loss", "levir_val_f1", "s2looking_val_f1",
                  "optimizer_steps", "levir_tiles", "s2looking_tiles")
LIMITATIONS = [
    "Single training seed. Image bootstrap intervals do not measure seed variation or geographic dependence.",
    "Official test sets are preserved. LEVIR-CD has two documented partial train/test footprint overlaps.",
    "S2Looking pairs sharing an exact decoded RGB image across splits are kept in test before val before train. "
    "S2Looking coordinates are unavailable for a footprint audit.",
]


def checkpoint_sha256(path):
    with path.open("rb") as file:
        return hashlib.file_digest(file, "sha256").hexdigest()


def write_json(path, value):
    temporary = path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def select_epoch(history, levir_reference):
    eligible = [row for row in history if row["levir_val_f1"] >= levir_reference - MAX_LEVIR_LOSS]
    if eligible:
        return max(eligible, key=lambda row: (row["s2looking_val_f1"], row["levir_val_f1"], -row["epoch"]))
    return max(history, key=lambda row: (row["levir_val_f1"], row["s2looking_val_f1"], -row["epoch"]))


def validation_f1(model, datasets, device, limit=None):
    scores = {}
    for name, splits in datasets.items():
        data = splits["val"]
        samples = data.samples[:limit]
        print(f"Validating {name}: {len(samples)} pairs.", flush=True)
        scores[name] = validate_epoch(model, Path("val"), samples, BATCH_SIZE, device,
                                      read_sample=data.read_sample)["F1"]
    return scores


def training_batches(datasets, epoch):
    start = (epoch - 1) * STEPS_PER_EPOCH
    batches = mixed_batches(datasets, SEED, start, STEPS_PER_EPOCH)
    for step, batch in enumerate(batches, 1):
        yield batch
        if step % 500 == 0 or step == STEPS_PER_EPOCH:
            print(f"Epoch {epoch}/{EPOCHS} | {step}/{STEPS_PER_EPOCH} training updates completed.", flush=True)


def load_reference(runs_root, device):
    model, selection = load_frozen_model(runs_root, SIAMESE, device)
    report = json.loads((runs_root / "levir_final_test" / "summary.json").read_text(encoding="utf-8"))
    frozen = report["selection"][SIAMESE]
    if selection["threshold"] != THRESHOLD or any(
            selection[key] != frozen[key] for key in ("checkpoint_sha256", "epoch", "threshold")):
        raise ValueError("The reference checkpoint must match the frozen LEVIR-CD test report.")
    return model, selection


def read_counts(path):
    with path.open(newline="", encoding="utf-8") as file:
        rows = list(csv.DictReader(file))
    counts = {row["sample"]: [int(row[field]) for field in COUNT_FIELDS] for row in rows}
    if not counts or len(counts) != len(rows):
        raise ValueError(f"Expected unique image names in {path}.")
    return counts


def compare_predictions(reference, mixed, excluded=()):
    if reference.keys() != mixed.keys():
        raise ValueError("The paired evaluation must contain the same image names for both models.")
    names = sorted(reference.keys() - set(excluded))
    before = np.array([reference[name] for name in names], dtype=np.int64)
    after = np.array([mixed[name] for name in names], dtype=np.int64)
    if not np.array_equal(before[:, 0] + before[:, 2], after[:, 0] + after[:, 2]):
        raise ValueError("The models must be scored against the same number of positive pixels per image.")
    indices = np.random.default_rng(SEED).integers(len(names), size=(5000, len(names)), dtype=np.int32)
    result = {"samples": len(names)}
    f1_draws = {}
    for role, counts in (("reference", before), ("mixed", after)):
        totals = dict(zip(COUNT_FIELDS, counts.sum(axis=0).tolist(), strict=True))
        f1_draws[role] = resampled_f1(counts, indices)
        result[role] = {
            "counts": totals,
            "metrics": metrics_from_counts(**totals),
            "f1_ci95": np.quantile(f1_draws[role], (0.025, 0.975)).tolist(),
        }
    result["f1_delta"] = result["mixed"]["metrics"]["F1"] - result["reference"]["metrics"]["F1"]
    result["f1_delta_ci95"] = np.quantile(f1_draws["mixed"] - f1_draws["reference"], (0.025, 0.975)).tolist()
    return result


def compare_split(datasets, models, split, device, output, selection, revision):
    if output.exists():
        report = json.loads((output / "comparison.json").read_text(encoding="utf-8"))
        if report["split"] != split or report["selection"] != selection:
            raise ValueError(f"A different comparison already exists at {output}.")
        return report
    with tempfile.TemporaryDirectory(dir=output.parent) as directory:
        staged = Path(directory) / output.name
        comparisons = {}
        for name, domain in datasets.items():
            data = domain[split]
            counts = {}
            for role, model in models.items():
                print(f"Comparing {name}/{split}: {role} on {len(data.samples)} pairs.", flush=True)
                report_dir = staged / name / role
                evaluate(model, Path(split), THRESHOLD, BATCH_SIZE, device, report_dir, "best.pt",
                         dataset=name, samples=data.samples, read_sample=data.read_sample)
                counts[role] = read_counts(report_dir / "per_image.csv")
            subset = "without_shared_images" if name == S2LOOKING and split == "val" else "official"
            comparisons[name] = {subset: compare_predictions(counts["reference"], counts["mixed"])}
            if name == LEVIR and split == "test":
                sensitivity = compare_predictions(counts["reference"], counts["mixed"], SENSITIVITY_EXCLUSIONS)
                sensitivity["excluded"] = sorted(SENSITIVITY_EXCLUSIONS)
                comparisons[name]["without_overlaps"] = sensitivity
        report = {"split": split, "threshold": THRESHOLD, "revision": revision, "selection": selection,
                  "datasets": comparisons, "limitations": LIMITATIONS}
        write_json(staged / "comparison.json", report)
        staged.rename(output)
    return report


def training_state(datasets, output, manifest, reference, reference_selection, device, revision):
    last = output / "last.pt"
    saved = None
    if last.is_file():
        print(f"Reading resume checkpoint {last}.", flush=True)
        saved = torch.load(last, map_location="cpu", weights_only=True)
        baseline = saved["config"]["reference_validation"]
    else:
        print("Measuring the frozen reference on both validation splits.", flush=True)
        baseline = validation_f1(reference, datasets, device)
    config = {
        "revision": revision,
        "seed": SEED,
        "epochs": EPOCHS,
        "steps_per_epoch": STEPS_PER_EPOCH,
        "batch_size": BATCH_SIZE,
        "tiles_per_batch": {LEVIR: 1, S2LOOKING: 1},
        "tile_size": TILE_SIZE,
        "threshold": THRESHOLD,
        "optimizer": "Adam",
        "loss": "BCE+Dice",
        "learning_rate": [START_LR, epoch_learning_rate(EPOCHS, EPOCHS)],
        "augment": False,
        "reference_sha256": reference_selection["checkpoint_sha256"],
        "reference_validation": baseline,
        "data_sha256": {domain: {split: entry["decoded_sha256"] for split, entry in splits.items()}
                        for domain, splits in manifest.items()},
    }
    if saved is not None:
        if saved["config"] != config or saved["architecture"] != SIAMESE:
            raise ValueError("Resume requires the original data, reference checkpoint and training settings.")
        history = saved["history"]
        if not 0 < saved["epoch"] <= EPOCHS or [row["epoch"] for row in history] != list(range(1, saved["epoch"] + 1)):
            raise ValueError("Invalid epoch history in last.pt.")
        for row in history:
            expected = row["epoch"] * STEPS_PER_EPOCH
            if any(row[field] != expected for field in ("optimizer_steps", "levir_tiles", "s2looking_tiles")):
                raise ValueError("Invalid update or tile counts in last.pt.")
    else:
        history = []
    return config, saved, history, baseline


def train_mixed(datasets, manifest, reference, reference_selection, output, device, revision):
    config, saved, history, baseline = training_state(
        datasets, output, manifest, reference, reference_selection, device, revision)
    write_json(output / "split_manifest.json", manifest)
    write_json(output / "baseline_validation.json", baseline)
    runtime_file = output / "runtime.json"
    runtime = {"python": sys.version.split()[0], "torch": torch.__version__,
               "numpy": version("numpy"), "pillow": version("Pillow"),
               "cuda": torch.version.cuda,
               "device": torch.cuda.get_device_name() if device == "cuda" else device}
    environments = json.loads(runtime_file.read_text(encoding="utf-8")) if runtime_file.exists() else []
    if runtime not in environments:
        write_json(runtime_file, [*environments, runtime])

    torch.manual_seed(SEED)
    model = SiameseUNet().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=START_LR)
    last, best = output / "last.pt", output / "best.pt"
    if saved is not None:
        model.load_state_dict(saved["model_state"])
        optimizer.load_state_dict(saved["optimizer_state"])
        if select_epoch(history, baseline[LEVIR])["epoch"] == saved["epoch"]:
            copy_checkpoint(last, best)
        elif not best.is_file():
            raise ValueError("Missing the selected checkpoint from an earlier epoch.")
        save_history(output / "history.csv", history, fieldnames=HISTORY_FIELDS)
        print(f"Resuming from completed epoch {saved['epoch']}.", flush=True)

    for epoch in range(len(history) + 1, EPOCHS + 1):
        rate = epoch_learning_rate(epoch, EPOCHS)
        for group in optimizer.param_groups:
            group["lr"] = rate
        print(f"Epoch {epoch}/{EPOCHS} | training at learning rate {rate:.6g}.", flush=True)
        loss = train_epoch(model, optimizer, training_batches(datasets, epoch), device)
        scores = validation_f1(model, datasets, device)
        updates = epoch * STEPS_PER_EPOCH
        history.append({"epoch": epoch, "learning_rate": rate, "train_loss": loss,
                        "levir_val_f1": scores[LEVIR], "s2looking_val_f1": scores[S2LOOKING],
                        "optimizer_steps": updates, "levir_tiles": updates, "s2looking_tiles": updates})
        print(f"Epoch {epoch}/{EPOCHS} | saving checkpoint to {last}.", flush=True)
        save_checkpoint(last, model, optimizer, epoch, SEED, history=history, config=config)
        if select_epoch(history, baseline[LEVIR])["epoch"] == epoch:
            copy_checkpoint(last, best)
        save_history(output / "history.csv", history, fieldnames=HISTORY_FIELDS)
        print(f"Epoch {epoch}/{EPOCHS} | updates {updates} | "
              f"LEVIR val F1 {scores[LEVIR]:.2%} | S2Looking val F1 {scores[S2LOOKING]:.2%}", flush=True)

    selected = select_epoch(history, baseline[LEVIR])
    best_state = torch.load(best, map_location="cpu", weights_only=True)
    if best_state["epoch"] != selected["epoch"] or best_state["config"] != config:
        raise ValueError("The selected checkpoint does not match the completed training history.")
    targets = {LEVIR: baseline[LEVIR] - MAX_LEVIR_LOSS, S2LOOKING: baseline[S2LOOKING] + MIN_S2LOOKING_GAIN}
    scores = {LEVIR: selected["levir_val_f1"], S2LOOKING: selected["s2looking_val_f1"]}
    success = {domain: scores[domain] >= target for domain, target in targets.items()}
    selection = {
        "threshold": THRESHOLD,
        "reference": reference_selection,
        "mixed": {"checkpoint": str(best), "checkpoint_sha256": checkpoint_sha256(best), "epoch": selected["epoch"]},
        "reference_validation_f1": baseline,
        "mixed_validation_f1": scores,
        "validation_targets": targets,
        "success": success,
        "passed_both_targets": all(success.values()),
        "optimizer_steps": EPOCHS * STEPS_PER_EPOCH,
        "data_sha256": config["data_sha256"],
    }
    models = {"reference": reference, "mixed": load_model(best, device)}
    compare_split(datasets, models, "val", device, output / "validation", selection, revision)
    write_json(output / "selection.json", selection)
    print(json.dumps(selection, indent=2))


def test_mixed(datasets, manifest, reference, reference_selection, output, device, revision):
    selection = json.loads((output / "selection.json").read_text(encoding="utf-8"))
    best = output / "best.pt"
    fingerprints = {domain: {split: entry["decoded_sha256"] for split, entry in splits.items()}
                    for domain, splits in manifest.items()}
    if (selection["threshold"] != THRESHOLD or selection["optimizer_steps"] != EPOCHS * STEPS_PER_EPOCH
            or selection["reference"]["checkpoint_sha256"] != reference_selection["checkpoint_sha256"]
            or selection["mixed"]["checkpoint_sha256"] != checkpoint_sha256(best)
            or selection["data_sha256"] != fingerprints):
        raise ValueError("Test requires the frozen checkpoint identities, data and threshold.")
    models = {"reference": reference, "mixed": load_model(best, device)}
    report = compare_split(datasets, models, "test", device, output / "test", selection, revision)
    print(json.dumps(report["datasets"], indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", choices=("smoke", "train", "test"), required=True)
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--s2looking-archive", type=Path, required=True)
    parser.add_argument("--runs-root", type=Path, required=True)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--revision")
    args = parser.parse_args()
    if args.device == "cuda" and not torch.cuda.is_available():
        parser.error("CUDA is unavailable. Enable a GPU runtime.")
    if args.device == "cpu":
        torch.set_num_threads(min(4, torch.get_num_threads()))
    output = args.runs_root / RUN_NAME
    if args.phase == "test" and not (output / "selection.json").is_file():
        parser.error("Complete mixed training and freeze the selection before evaluating test.")
    print(f"Indexing LEVIR-CD at {args.data_root} and S2Looking at {args.s2looking_archive}.", flush=True)
    with zipfile.ZipFile(args.s2looking_archive) as archive:
        datasets = open_splits(args.data_root, archive)
        if args.phase == "smoke":
            torch.manual_seed(SEED)
            model = SiameseUNet().to(args.device)
            optimizer = torch.optim.Adam(model.parameters(), lr=START_LR)
            loss = train_epoch(model, optimizer, mixed_batches(datasets, SEED, 0, 8), args.device)
            print(f"Smoke | 8 mixed updates | loss {loss:.4f} | val F1 {validation_f1(model, datasets, args.device, 1)}")
            return
        print(f"Checking the frozen reference under {args.runs_root}.", flush=True)
        reference, reference_selection = load_reference(args.runs_root, args.device)
        datasets, manifest = audit_splits(datasets)
        output.mkdir(parents=True, exist_ok=True)
        run = train_mixed if args.phase == "train" else test_mixed
        run(datasets, manifest, reference, reference_selection, output, args.device, args.revision)


if __name__ == "__main__":
    main()
