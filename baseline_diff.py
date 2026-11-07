"""Detect candidate changes using the mean absolute RGB difference."""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from levir_data import DEFAULT_DATA_ROOT, DatasetError, load_pair


PROJECT_ROOT = Path(__file__).resolve().parent


def predict_change(before, after, threshold):
    # Convert before subtracting: unsigned RGB values would wrap around.
    delta = np.subtract(before, after, dtype=np.float32)
    return np.abs(delta).mean(axis=-1) > threshold


def change_metrics(prediction, reference):
    tp = np.count_nonzero(prediction & reference)
    fp = np.count_nonzero(prediction & ~reference)
    fn = np.count_nonzero(~prediction & reference)

    def divide(numerator, denominator):
        return numerator / denominator if denominator else 0.0

    return {
        "Precision": divide(tp, tp + fp),
        "Recall": divide(tp, tp + fn),
        "F1": divide(2 * tp, 2 * tp + fp + fn),
        "IoU": divide(tp, tp + fp + fn),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("sample", nargs="?", default="val_1.png")
    parser.add_argument("--threshold", type=float, default=30, help="RGB difference threshold (0–255).")
    parser.add_argument("--no-show", action="store_true", help="Save without opening a plot window.")
    args = parser.parse_args()
    if not 0 <= args.threshold <= 255:
        parser.error("Threshold must be between 0 and 255.")

    try:
        before, after, reference = load_pair(DEFAULT_DATA_ROOT / "val", args.sample)
    except DatasetError as error:
        parser.error(str(error))

    prediction = predict_change(before, after, args.threshold)
    metrics = change_metrics(prediction, reference)
    print(f"Sample: {args.sample} | Threshold: {args.threshold:g}")
    for name, value in metrics.items():
        print(f"{name}: {value:.4f}")

    fig, axes = plt.subplots(2, 2, figsize=(9, 9), layout="constrained")
    panels = [(before, "Before (T1)"), (after, "After (T2)"),
              (reference, "Reference change mask"), (prediction, "Pixel difference prediction")]
    for ax, (image, title) in zip(axes.flat, panels):
        if image.ndim == 2:
            ax.imshow(image, cmap="gray", vmin=0, vmax=1, interpolation="nearest")
        else:
            ax.imshow(image)
        ax.set_title(title)
        ax.set_axis_off()
    fig.suptitle(f"{args.sample} | Threshold {args.threshold:g} | F1 {metrics['F1']:.3f} | IoU {metrics['IoU']:.3f}")

    output = PROJECT_ROOT / "outputs" / f"{Path(args.sample).stem}_diff_t{args.threshold:g}.png"
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=150)
    print(f"Saved: {output.relative_to(PROJECT_ROOT)}")
    if not args.no_show:
        plt.show()
    plt.close(fig)


if __name__ == "__main__":
    main()
