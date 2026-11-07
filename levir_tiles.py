"""Read aligned LEVIR-CD tiles and save a preview of their coordinates."""

import argparse
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from levir_data import DEFAULT_DATA_ROOT, load_pair


TILE_SIZE = 256
OUTPUT_ROOT = Path(__file__).resolve().parent / "outputs"


@dataclass(frozen=True)
class PairTile:
    x: int
    y: int
    inputs: np.ndarray
    mask: np.ndarray


def iter_tiles(before, after, mask, tile_size=TILE_SIZE) -> Iterator[PairTile]:
    """Yield non-overlapping tiles in row order, with channel-first float32 arrays."""
    if tile_size <= 0:
        raise ValueError("Tile size must be positive.")
    if before.ndim != 3 or before.shape[-1] != 3 or after.shape != before.shape:
        raise ValueError("Both RGB images must have the same height and width.")
    height, width = before.shape[:2]
    if mask.shape != (height, width):
        raise ValueError("The mask must match the image height and width.")
    if not height or not width or height % tile_size or width % tile_size:
        raise ValueError(f"Image dimensions {width}x{height} must be multiples of {tile_size}.")

    for y in range(0, height, tile_size):
        for x in range(0, width, tile_size):
            region = np.s_[y:y + tile_size, x:x + tile_size]
            inputs = np.concatenate((before[region], after[region]), axis=-1)
            inputs = np.ascontiguousarray(inputs.transpose(2, 0, 1), dtype=np.float32)
            inputs /= 255.0
            target = mask[region][None, :, :].astype(np.float32)
            yield PairTile(x=x, y=y, inputs=inputs, mask=target)


def save_preview(after, tile, tile_size, sample, output):
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle

    height, width = after.shape[:2]
    fig, axes = plt.subplots(2, 2, figsize=(8, 8), layout="constrained")
    overview = axes[0, 0]
    overview.imshow(after, extent=(0, width, height, 0))
    overview.vlines(range(tile_size, width, tile_size), 0, height, colors="white", alpha=0.65)
    overview.hlines(range(tile_size, height, tile_size), 0, width, colors="white", alpha=0.65)
    overview.add_patch(Rectangle(
        (tile.x, tile.y), tile_size, tile_size, fill=False, edgecolor="#ffd84a", linewidth=3,
    ))
    overview.set_title("After image · selected tile")

    panels = (
        (tile.inputs[:3].transpose(1, 2, 0), "Before tile (T1)"),
        (tile.inputs[3:].transpose(1, 2, 0), "After tile (T2)"),
        (tile.mask[0], "Reference mask tile"),
    )
    for ax, (image, title) in zip(list(axes.flat)[1:], panels):
        ax.imshow(image, cmap="gray", vmin=0, vmax=1, interpolation="nearest")
        ax.set_title(title)
    for ax in axes.flat:
        ax.set_axis_off()
    fig.suptitle(f"{sample} | {tile_size} x {tile_size} | x={tile.x}, y={tile.y}")
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=140)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("sample", nargs="?", default="val_1.png")
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--split", choices=("train", "val", "test"), default="val")
    args = parser.parse_args()

    try:
        before, after, mask = load_pair(args.data_root / args.split, args.sample)
        selected = max(iter_tiles(before, after, mask), key=lambda tile: np.count_nonzero(tile.mask))
    except ValueError as error:
        parser.error(str(error))

    height, width = mask.shape
    tile_count = (height // TILE_SIZE) * (width // TILE_SIZE)
    output = OUTPUT_ROOT / "tiles" / args.split / f"{Path(args.sample).stem}_tiles.png"
    save_preview(after, selected, TILE_SIZE, args.sample, output)
    print(f"Sample: {args.sample} | Split: {args.split} | Tiles: {tile_count}")
    print(f"Selected tile: x={selected.x}, y={selected.y}")
    print(f"Input: {selected.inputs.shape} | Mask: {selected.mask.shape}")
    print(f"Saved: {output}")


if __name__ == "__main__":
    main()
