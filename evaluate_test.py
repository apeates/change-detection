"""Evaluate the three frozen LEVIR-CD methods on the official test split."""

import argparse
import csv
import hashlib
import json
import tempfile
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

from baseline_diff import change_counts, metrics_from_counts, predict_change
from evaluate_baseline import CSV_FIELDS, ERROR_COLORS, select_examples
from evaluate_unet import COUNT_FIELDS, predict_mask, resampled_f1
from levir_data import DEFAULT_DATA_ROOT, DatasetError, list_samples, load_pair
from train_unet import load_model


PIXEL_METHOD = "pixel_difference"
EARLY_FUSION = "early_fusion_unet"
SIAMESE = "siamese_unet"
METHOD_LABELS = {
    PIXEL_METHOD: "Pixel difference",
    EARLY_FUSION: "Early Fusion U-Net",
    SIAMESE: "Siamese U-Net",
}
PIXEL_THRESHOLD = 30
# Best checkpoints and thresholds were chosen on validation before opening test.
FROZEN_MODELS = {
    EARLY_FUSION: ("early_fusion_v2_linear", 9, 0.35),
    SIAMESE: ("siamese_v1_linear_e30", 30, 0.50),
}
TEST_SAMPLES = frozenset(f"test_{index}.png" for index in range(1, 129))
# These two test footprints partially overlap train. See docs/levir_split_audit.md.
SENSITIVITY_EXCLUSIONS = frozenset(("test_9.png", "test_19.png"))


def load_frozen_model(runs_root, method, device):
    run, epoch, threshold = FROZEN_MODELS[method]
    path = runs_root / run / "full" / "best.pt"
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    model = load_model(path, device)
    if model.architecture != method or checkpoint["epoch"] != epoch:
        raise ValueError(f"Expected {method} from epoch {epoch} at {path}.")
    with path.open("rb") as file:
        sha256 = hashlib.file_digest(file, "sha256").hexdigest()
    return model, {
        "checkpoint": str(path),
        "checkpoint_sha256": sha256,
        "epoch": epoch,
        "threshold": threshold,
    }


def load_frozen_models(runs_root, device):
    models = {}
    selection = {PIXEL_METHOD: {"threshold": PIXEL_THRESHOLD}}
    for method in FROZEN_MODELS:
        models[method], selection[method] = load_frozen_model(runs_root, method, device)
    return models, selection


def summarize_subset(rows_by_method, excluded=()):
    sample_count = sum(row["sample"] not in excluded for row in rows_by_method[PIXEL_METHOD])
    indices = np.random.default_rng(42).integers(sample_count, size=(5000, sample_count))
    methods = {}
    f1_draws = {}
    for method, rows in rows_by_method.items():
        included = [row for row in rows if row["sample"] not in excluded]
        counts = np.array([[row[field] for field in COUNT_FIELDS] for row in included], dtype=np.int64)
        totals = dict(zip(COUNT_FIELDS, counts.sum(axis=0).tolist(), strict=True))
        blank_pixels = sum(row["blank_mask_pixels"] for row in included)
        blank_fp = sum(row["fp"] for row in included if row["blank_mask_pixels"])
        f1_draws[method] = resampled_f1(counts, indices)
        methods[method] = {
            "counts": totals,
            "metrics": metrics_from_counts(**totals),
            "f1_ci95": np.quantile(f1_draws[method], (0.025, 0.975)).tolist(),
            "blank_mask_fp_rate": blank_fp / blank_pixels if blank_pixels else None,
        }
    paired_delta = f1_draws[SIAMESE] - f1_draws[EARLY_FUSION]
    return {
        "samples": sample_count,
        "methods": methods,
        "siamese_vs_early_fusion_f1_delta": methods[SIAMESE]["metrics"]["F1"] - methods[EARLY_FUSION]["metrics"]["F1"],
        "siamese_vs_early_fusion_f1_delta_ci95": np.quantile(paired_delta, (0.025, 0.975)).tolist(),
    }


def save_comparison(path, sample, before, after, reference, predictions, selection):
    fig, axes = plt.subplots(2, 3, figsize=(13, 8), layout="constrained")
    for ax in axes.flat:
        ax.set_axis_off()
    for ax, image, title in zip(axes[0], (before, after, reference), ("T1", "T2", "Reference mask"), strict=True):
        ax.imshow(image, cmap="gray", vmin=0, vmax=1, interpolation="nearest")
        ax.set_title(title)
    for ax, (method, prediction) in zip(axes[1], predictions.items(), strict=True):
        errors = np.zeros((*reference.shape, 3), dtype=np.uint8)
        errors[prediction & reference] = ERROR_COLORS["Correct change"]
        errors[prediction & ~reference] = ERROR_COLORS["False alarm"]
        errors[~prediction & reference] = ERROR_COLORS["Missed change"]
        ax.imshow(errors, interpolation="nearest")
        counts = change_counts(prediction, reference)
        f1 = metrics_from_counts(**counts)["F1"]
        threshold = selection[method]["threshold"]
        ax.set_title(f"{METHOD_LABELS[method]} ({threshold:g})\n"
                     f"F1 {f1:.2%} | FP {counts['fp']:,} | FN {counts['fn']:,}")
    fig.suptitle(f"LEVIR-CD test | {sample}\nGreen = correct change, orange = false alarm, purple = missed change")
    fig.savefig(path, dpi=140)
    plt.close(fig)


def write_table(path, fields, rows):
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def evaluate_test(split, models, selection, batch_size, device, output, revision=None):
    samples = list_samples(split)
    if split.name != "test" or set(samples) != TEST_SAMPLES:
        raise DatasetError("Expected the official test split with test_1.png through test_128.png.")

    rows = {method: [] for method in METHOD_LABELS}
    packed = {}
    for sample in samples:
        before, after, reference = load_pair(split, sample)
        if reference.shape != (1024, 1024):
            raise DatasetError(f"{sample}: expected a full-resolution 1024×1024 test pair.")
        predictions = {PIXEL_METHOD: predict_change(before, after, PIXEL_THRESHOLD)}
        for method, model in models.items():
            predictions[method] = predict_mask(model, before, after, reference,
                                               selection[method]["threshold"], batch_size, device)
        blank_pixels = 0 if reference.any() else reference.size
        for method, prediction in predictions.items():
            counts = change_counts(prediction, reference)
            rows[method].append({"sample": sample, **counts, **metrics_from_counts(**counts),
                                 "blank_mask_pixels": blank_pixels})
        packed[sample] = {method: np.packbits(prediction, axis=-1) for method, prediction in predictions.items()}

    subsets = {"full": summarize_subset(rows), "sensitivity": summarize_subset(rows, SENSITIVITY_EXCLUSIONS)}
    write_table(output / "per_image.csv", ("method", *CSV_FIELDS, "blank_mask_pixels", "included_in_sensitivity"),
                ({"method": method, **row, "included_in_sensitivity": row["sample"] not in SENSITIVITY_EXCLUSIONS}
                 for method, method_rows in rows.items() for row in method_rows))
    write_table(output / "test_results.csv",
                ("subset", "samples", "method", "threshold", *CSV_FIELDS[1:], "blank_mask_fp_rate"),
                ({"subset": subset, "samples": result["samples"], "method": method,
                  "threshold": selection[method]["threshold"], **score["counts"], **score["metrics"],
                  "blank_mask_fp_rate": score["blank_mask_fp_rate"]}
                 for subset, result in subsets.items() for method, score in result["methods"].items()))

    examples = {}
    for name, row in select_examples(rows[EARLY_FUSION]).items():
        if row is not None:
            sample = row["sample"]
            before, after, reference = load_pair(split, sample)
            predictions = {method: np.unpackbits(mask, axis=-1, count=reference.shape[1]).astype(bool)
                           for method, mask in packed[sample].items()}
            figure = f"{name}.png"
            save_comparison(output / figure, sample, before, after, reference, predictions, selection)
            examples[name] = {"sample": sample, "figure": figure}

    summary = {
        "split": "test",
        "aggregation": "pooled_change_pixel_counts",
        "revision": revision,
        "device": torch.cuda.get_device_name() if device == "cuda" else device,
        "batch_size": batch_size,
        "method_labels": METHOD_LABELS,
        "selection": selection,
        "excluded_from_sensitivity": sorted(SENSITIVITY_EXCLUSIONS),
        "subsets": subsets,
        "examples": examples,
    }
    (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
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
        models, selection = load_frozen_models(args.runs_root, args.device)
        args.output_dir.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(dir=args.output_dir.parent) as directory:
            staged = Path(directory) / "report"
            staged.mkdir()
            evaluate_test(args.data_root / "test", models, selection, args.batch_size,
                          args.device, staged, revision=args.revision)
            staged.rename(args.output_dir)
    except (OSError, ValueError, RuntimeError) as error:
        parser.error(str(error))
    print(f"Saved: {args.output_dir}")


if __name__ == "__main__":
    main()
