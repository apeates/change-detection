"""Inspect two LEVIR-CD images and their reference change mask."""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
from PIL import Image


PROJECT_ROOT = Path(__file__).resolve().parent


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("sample", nargs="?", default="val_1.png")
    parser.add_argument("--no-show", action="store_true", help="Save without opening a plot window.")
    args = parser.parse_args()

    split = PROJECT_ROOT / "data" / "levir-cd" / "val"
    panels = [
        ("A", "Before (T1)", "RGB"),
        ("B", "After (T2)", "RGB"),
        ("label", "Reference change mask", "L"),
    ]
    images = []
    for folder, title, mode in panels:
        path = split / folder / args.sample
        if not path.is_file():
            parser.error(f"Missing image: {path}")
        with Image.open(path) as source:
            image = source.convert(mode)
        print(f"{title}: {image.width} x {image.height}")
        images.append(image)

    if len({image.size for image in images}) != 1:
        parser.error("Both images and the mask must have the same dimensions.")

    fig, axes = plt.subplots(1, 3, figsize=(12, 4.5), layout="constrained")
    for ax, image, (_, title, mode) in zip(axes, images, panels):
        if mode == "L":
            ax.imshow(image, cmap="gray", vmin=0, vmax=255, interpolation="nearest")
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
