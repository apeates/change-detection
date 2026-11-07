"""Inspect two LEVIR-CD images and their reference change mask."""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
from levir_data import DEFAULT_DATA_ROOT, DatasetError, load_pair


PROJECT_ROOT = Path(__file__).resolve().parent


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("sample", nargs="?", default="val_1.png")
    parser.add_argument("--no-show", action="store_true", help="Save without opening a plot window.")
    args = parser.parse_args()

    try:
        before, after, reference = load_pair(DEFAULT_DATA_ROOT / "val", args.sample)
    except DatasetError as error:
        parser.error(str(error))

    panels = [(before, "Before (T1)"), (after, "After (T2)"), (reference, "Reference change mask")]
    fig, axes = plt.subplots(1, 3, figsize=(12, 4.5), layout="constrained")
    for ax, (image, title) in zip(axes, panels):
        height, width = image.shape[:2]
        print(f"{title}: {width} x {height}")
        if image.ndim == 2:
            ax.imshow(image, cmap="gray", vmin=0, vmax=1, interpolation="nearest")
        else:
            ax.imshow(image)
        ax.set_title(title)
        ax.set_axis_off()

    fig.suptitle(f"LEVIR-CD | {args.sample}")
    output = PROJECT_ROOT / "outputs" / f"{Path(args.sample).stem}_panel.png"
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=150)
    print(f"Saved: {output.relative_to(PROJECT_ROOT)}")

    if not args.no_show:
        plt.show()
    plt.close(fig)


if __name__ == "__main__":
    main()
