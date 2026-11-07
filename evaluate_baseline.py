"""Evaluate the pixel difference baseline on an entire LEVIR-CD split."""

import argparse
import csv
import json
from collections import Counter
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import numpy as np

from baseline_diff import change_counts, metrics_from_counts, predict_change
from levir_data import DEFAULT_DATA_ROOT, DatasetError, list_samples, load_pair


DEFAULT_OUTPUT_ROOT = Path(__file__).resolve().parent / "outputs"
CSV_FIELDS = ("sample", "tp", "fp", "fn", "Precision", "Recall", "F1", "IoU")
ERROR_COLORS = {
    "Correct change": (0, 191, 125),
    "False alarm": (255, 167, 61),
    "Missed change": (166, 136, 255),
}


def evaluate_split(split, threshold):
    """Evaluate one pair at a time and pool counts before computing overall scores."""
    rows = []
    totals = Counter(tp=0, fp=0, fn=0)
    for sample in list_samples(split):
        before, after, reference = load_pair(split, sample)
        prediction = predict_change(before, after, threshold)
        counts = change_counts(prediction, reference)
        totals.update(counts)
        rows.append({"sample": sample, **counts, **metrics_from_counts(**counts)})
    return rows, dict(totals)


def select_examples(rows):
    """Select the largest false alarm on a blank mask and the most missed pixels."""
    false_alarms = [row for row in rows if row["tp"] + row["fn"] == 0 and row["fp"] > 0]
    missed_changes = [row for row in rows if row["fn"] > 0]
    return {
        "false_alarms": max(false_alarms, key=lambda row: row["fp"], default=None),
        "missed_changes": max(missed_changes, key=lambda row: row["fn"], default=None),
    }


def save_example(split, row, threshold, output):
    before, after, reference = load_pair(split, row["sample"])
    prediction = predict_change(before, after, threshold)
    errors = np.zeros((*reference.shape, 3), dtype=np.uint8)
    errors[prediction & reference] = ERROR_COLORS["Correct change"]
    errors[prediction & ~reference] = ERROR_COLORS["False alarm"]
    errors[~prediction & reference] = ERROR_COLORS["Missed change"]

    fig, axes = plt.subplots(2, 2, figsize=(8, 8), layout="constrained")
    for ax, image, title in zip(
        axes.flat,
        (before, after, reference, errors),
        ("Before (T1)", "After (T2)", "Reference mask", "Prediction vs. reference"),
    ):
        ax.imshow(image, cmap="gray", vmin=0, vmax=1, interpolation="nearest")
        ax.set_title(title)
        ax.set_axis_off()
    axes[1, 1].legend(
        handles=[
            Patch(color=tuple(channel / 255 for channel in color), label=label)
            for label, color in ERROR_COLORS.items()
        ],
        loc="upper center", bbox_to_anchor=(0.5, -0.02), ncols=3, fontsize=8,
    )
    fig.suptitle(
        f"{row['sample']} | Threshold {threshold:g}\n"
        f"TP {row['tp']:,} | FP {row['fp']:,} | FN {row['fn']:,}"
    )
    fig.savefig(output, dpi=140)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--split", choices=("train", "val", "test"), default="val")
    parser.add_argument("--threshold", type=float, default=30)
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()
    if not 0 <= args.threshold <= 255:
        parser.error("Threshold must be between 0 and 255.")

    split = args.data_root / args.split
    output = args.output_dir or DEFAULT_OUTPUT_ROOT / f"baseline_{args.split}_t{args.threshold:g}"
    try:
        rows, totals = evaluate_split(split, args.threshold)
    except DatasetError as error:
        parser.error(str(error))

    output.mkdir(parents=True, exist_ok=True)
    with (output / "per_image.csv").open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=CSV_FIELDS)
        writer.writeheader()
        writer.writerows(rows)

    metrics = metrics_from_counts(**totals)
    examples = {}
    for name, row in select_examples(rows).items():
        if row is not None:
            filename = f"{name}.png"
            save_example(split, row, args.threshold, output / filename)
            examples[name] = {"sample": row["sample"], "figure": filename}

    summary = {
        "method": "mean_absolute_rgb_difference",
        "split": args.split,
        "threshold": args.threshold,
        "samples": len(rows),
        "masks_without_change": sum(row["tp"] + row["fn"] == 0 for row in rows),
        "aggregation": "pooled_change_pixel_counts",
        "counts": totals,
        "metrics": metrics,
        "examples": examples,
    }
    (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(f"Split: {args.split} | Threshold: {args.threshold:g} | Samples: {len(rows)}")
    for name, value in metrics.items():
        print(f"{name}: {value:.4f}")
    print(f"Saved: {output}")


if __name__ == "__main__":
    main()
