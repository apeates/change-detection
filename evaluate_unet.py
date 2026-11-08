"""Evaluate a saved U-Net on LEVIR-CD validation images and visualize errors."""

import argparse
import csv
import json
from collections import Counter
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import numpy as np
import torch

from baseline_diff import change_counts, metrics_from_counts
from evaluate_baseline import CSV_FIELDS, ERROR_COLORS, select_examples
from levir_data import DEFAULT_DATA_ROOT, list_samples, load_pair
from levir_tiles import TILE_SIZE, iter_tiles
from train_unet import batch_tiles, load_model


SWEEP_THRESHOLDS = (0.15, 0.25, 0.35, 0.50, 0.65, 0.75, 0.85)
SWEEP_FIELDS = ("threshold", *CSV_FIELDS[1:], "blank_mask_fp_pixels", "blank_mask_fp_rate")


@torch.inference_mode()
def predict_logits(model, before, after, reference, batch_size, device):
    """Run each tile once and stitch its scores before applying any threshold."""
    prediction = np.empty(reference.shape, dtype=np.float32)
    for batch in batch_tiles(iter_tiles(before, after, reference), batch_size):
        inputs = torch.from_numpy(np.stack([tile.inputs for tile in batch])).to(device)
        logits = model(inputs).cpu().numpy()
        for tile, scores in zip(batch, logits, strict=True):
            prediction[tile.y:tile.y + TILE_SIZE, tile.x:tile.x + TILE_SIZE] = scores[0]
    return prediction


def predict_mask(model, before, after, reference, threshold, batch_size, device):
    limit = torch.tensor(threshold, device=device).logit().item()
    return predict_logits(model, before, after, reference, batch_size, device) > limit


def save_panel(output, sample, before, after, reference, prediction, counts, threshold):
    errors = np.zeros((*reference.shape, 3), dtype=np.uint8)
    errors[prediction & reference] = ERROR_COLORS["Correct change"]
    errors[prediction & ~reference] = ERROR_COLORS["False alarm"]
    errors[~prediction & reference] = ERROR_COLORS["Missed change"]

    fig, axes = plt.subplots(2, 3, figsize=(12, 8), layout="constrained")
    for ax in axes.flat:
        ax.set_axis_off()
    for ax, pixels, title in (
        (axes[0, 0], before, "Before (T1)"),
        (axes[0, 1], after, "After (T2)"),
        (axes[1, 0], reference, "Reference mask"),
        (axes[1, 1], prediction, "Predicted mask"),
        (axes[1, 2], errors, "Pixel errors"),
    ):
        ax.imshow(pixels, cmap="gray", vmin=0, vmax=1, interpolation="nearest")
        ax.set_title(title)

    metrics = metrics_from_counts(**counts)
    key = axes[0, 2]
    key.text(0.06, 0.92, f"F1   {metrics['F1']:.2%}\nIoU {metrics['IoU']:.2%}",
             fontsize=17, va="top", transform=key.transAxes)
    key.legend(handles=[
        Patch(color=tuple(channel / 255 for channel in ERROR_COLORS[label]),
              label=f"{label}: {counts[field]:,}")
        for label, field in (("Correct change", "tp"), ("False alarm", "fp"), ("Missed change", "fn"))
    ], loc="center left", bbox_to_anchor=(0.03, 0.43), frameon=False)
    fig.suptitle(f"LEVIR-CD validation | {sample} | threshold {threshold:g}")
    fig.savefig(output, dpi=140)
    plt.close(fig)


def evaluate(model, split, threshold, batch_size, device, output, checkpoint_name, *, sweep=False):
    if sweep and split.name != "val":
        raise ValueError("Choose thresholds on the validation split.")

    thresholds = sorted(set((*SWEEP_THRESHOLDS, threshold))) if sweep else [threshold]
    limits = {value: torch.tensor(value, device=device).logit().item() for value in thresholds}
    totals = {value: Counter(tp=0, fp=0, fn=0) for value in thresholds}
    blank_false_alarms = Counter()
    blank_masks = 0
    blank_pixels = 0
    rows = []
    packed_predictions = {}
    model.eval()
    for sample in list_samples(split):
        before, after, reference = load_pair(split, sample)
        logits = predict_logits(model, before, after, reference, batch_size, device)
        blank = not reference.any()
        if blank:
            blank_masks += 1
            blank_pixels += reference.size
        for value, limit in limits.items():
            prediction = logits > limit
            counts = change_counts(prediction, reference)
            totals[value].update(counts)
            if blank:
                blank_false_alarms[value] += counts["fp"]
            if value == threshold:
                rows.append({"sample": sample, **counts, **metrics_from_counts(**counts)})
                # Keep the masks compact so rendering examples needs no second inference.
                packed_predictions[sample] = np.packbits(prediction, axis=-1)

    scores = {
        value: {
            "threshold": value,
            **totals[value],
            **metrics_from_counts(**totals[value]),
            "blank_mask_fp_pixels": blank_false_alarms[value],
            "blank_mask_fp_rate": blank_false_alarms[value] / blank_pixels if blank_pixels else None,
        }
        for value in thresholds
    }

    output.mkdir(parents=True, exist_ok=True)
    with (output / "per_image.csv").open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=CSV_FIELDS)
        writer.writeheader()
        writer.writerows(rows)

    chosen = select_examples(rows)
    chosen["best_detection"] = max((row for row in rows if row["tp"] > 0),
                                   key=lambda row: row["F1"], default=None)
    examples = {}
    for kind, row in chosen.items():
        if row is None:
            continue
        sample = row["sample"]
        before, after, reference = load_pair(split, sample)
        prediction = np.unpackbits(packed_predictions[sample], axis=-1, count=reference.shape[1]).astype(bool)
        figure = f"{kind}.png"
        counts = {field: row[field] for field in ("tp", "fp", "fn")}
        save_panel(output / figure, sample, before, after, reference, prediction, counts, threshold)
        examples[kind] = {"sample": sample, "figure": figure}

    summary = {
        "method": "early_fusion_unet",
        "split": "val",
        "checkpoint": checkpoint_name,
        "threshold": threshold,
        "samples": len(rows),
        "masks_without_change": blank_masks,
        "blank_mask_pixels": blank_pixels,
        "blank_mask_fp_pixels": blank_false_alarms[threshold],
        "blank_mask_fp_rate": scores[threshold]["blank_mask_fp_rate"],
        "aggregation": "pooled_change_pixel_counts",
        "counts": dict(totals[threshold]),
        "metrics": metrics_from_counts(**totals[threshold]),
        "examples": examples,
    }
    if sweep:
        sweep_rows = [scores[value] for value in SWEEP_THRESHOLDS]
        with (output / "threshold_sweep.csv").open("w", newline="", encoding="utf-8") as file:
            writer = csv.DictWriter(file, fieldnames=SWEEP_FIELDS)
            writer.writeheader()
            writer.writerows(sweep_rows)
        best = max(sweep_rows, key=lambda row: (row["F1"], -abs(row["threshold"] - 0.50), row["threshold"]))
        summary["threshold_sweep"] = {
            "file": "threshold_sweep.csv",
            "best_threshold": best["threshold"],
            "best_f1": best["F1"],
        }
    (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--threshold", type=float, default=0.35)
    parser.add_argument("--sweep", action="store_true", help="Score seven validation thresholds in one pass.")
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--device", choices=("cpu", "cuda", "mps"), default="cpu")
    args = parser.parse_args()
    if not 0 < args.threshold < 1 or args.batch_size < 1:
        parser.error("Threshold must be between 0 and 1, and batch size must be positive.")
    if not args.checkpoint.is_file():
        parser.error(f"Checkpoint not found: {args.checkpoint}")
    if args.device == "cuda" and not torch.cuda.is_available():
        parser.error("CUDA is unavailable. Enable a GPU runtime.")
    if args.device == "mps" and not torch.backends.mps.is_available():
        parser.error("MPS is unavailable. Use --device cpu.")
    if args.device == "cpu":
        torch.set_num_threads(min(4, torch.get_num_threads()))

    try:
        model = load_model(args.checkpoint, args.device)
        summary = evaluate(model, args.data_root / "val", args.threshold, args.batch_size,
                           args.device, args.output_dir, args.checkpoint.name, sweep=args.sweep)
    except (OSError, ValueError, RuntimeError) as error:
        parser.error(str(error))
    print(json.dumps(summary, indent=2))
    print(f"Saved: {args.output_dir}")


if __name__ == "__main__":
    main()
