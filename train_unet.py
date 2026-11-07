"""Train Early Fusion U-Net on LEVIR-CD's train split."""

import argparse
import csv
import random
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

from baseline_diff import change_metrics
from early_fusion_unet import EarlyFusionUNet
from levir_data import DEFAULT_DATA_ROOT, DatasetError, list_samples, load_pair
from levir_tiles import TILE_SIZE, iter_tiles


RUNS_ROOT = Path(__file__).resolve().parent / "runs"


def segmentation_loss(logits, masks):
    probabilities = logits.sigmoid()
    dims = (1, 2, 3)
    overlap = (probabilities * masks).sum(dim=dims)
    dice = (2 * overlap + 1) / (probabilities.sum(dim=dims) + masks.sum(dim=dims) + 1)
    return F.binary_cross_entropy_with_logits(logits, masks) + (1 - dice.mean())


def find_smoke_tiles(split, samples, count):
    """Collect a few positive tiles for a repeatable training sanity check."""
    selected = []
    for sample in samples:
        for tile in iter_tiles(*load_pair(split, sample)):
            if tile.mask.any():
                selected.append((sample, tile))
                if len(selected) == count:
                    return selected
    raise DatasetError(f"Requested {count} positive train tiles, found {len(selected)} in {split}")


def shuffled_train_tiles(split, samples, rng):
    names = list(samples)
    rng.shuffle(names)
    for sample in names:
        tiles = list(iter_tiles(*load_pair(split, sample)))
        rng.shuffle(tiles)
        yield from tiles


def batch_tiles(tiles, batch_size):
    batch = []
    for tile in tiles:
        batch.append(tile)
        if len(batch) == batch_size:
            yield batch
            batch = []
    if batch:
        yield batch


def train_epoch(model, optimizer, batches, device):
    model.train()
    total_loss = 0.0
    tile_count = 0
    for batch in batches:
        inputs = torch.from_numpy(np.stack([tile.inputs for tile in batch])).to(device)
        masks = torch.from_numpy(np.stack([tile.mask for tile in batch])).to(device)
        optimizer.zero_grad(set_to_none=True)
        loss = segmentation_loss(model(inputs), masks)
        if not torch.isfinite(loss):
            raise RuntimeError("Training loss is not finite.")
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * len(batch)
        tile_count += len(batch)
    if not tile_count:
        raise ValueError("No training tiles were loaded.")
    return total_loss / tile_count


def save_checkpoint(path, model, optimizer, epoch, seed):
    torch.save({
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "epoch": epoch,
        "seed": seed,
        "tile_size": TILE_SIZE,
    }, path)


def load_model(path, device="cpu"):
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    model = EarlyFusionUNet()
    model.load_state_dict(checkpoint["model_state"])
    return model.to(device).eval()


def save_smoke_preview(model, sample, tile, output):
    import matplotlib.pyplot as plt

    with torch.inference_mode():
        inputs = torch.from_numpy(tile.inputs).unsqueeze(0).to(next(model.parameters()).device)
        predicted = model(inputs)[0, 0].cpu().numpy() > 0
    reference = tile.mask[0].astype(bool)
    metrics = change_metrics(predicted, reference)
    panels = (
        (tile.inputs[:3].transpose(1, 2, 0), "Before (T1)"),
        (tile.inputs[3:].transpose(1, 2, 0), "After (T2)"),
        (reference, "Reference mask"),
        (predicted, "U-Net prediction"),
    )
    fig, axes = plt.subplots(2, 2, figsize=(7, 7), layout="constrained")
    for ax, (image, title) in zip(axes.flat, panels):
        ax.imshow(image, cmap="gray", vmin=0, vmax=1, interpolation="nearest")
        ax.set_title(title)
        ax.set_axis_off()
    fig.suptitle(f"Training tile: {sample} ({tile.x}, {tile.y}) | F1 {metrics['F1']:.3f}")
    fig.savefig(output, dpi=120)
    plt.close(fig)
    return metrics["F1"]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--device", choices=("cpu", "mps", "cuda"), default="cpu")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--smoke-tiles", type=int, help="Repeat only this many positive train tiles.")
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()
    if args.epochs < 1 or args.batch_size < 1 or (args.smoke_tiles is not None and args.smoke_tiles < 1):
        parser.error("Epochs, batch size, and smoke tile count must be positive.")
    if args.device == "cuda" and not torch.cuda.is_available():
        parser.error("CUDA is not available in this environment.")
    if args.device == "mps" and not torch.backends.mps.is_available():
        parser.error("MPS is not available in this environment.")
    if args.device == "cpu":
        torch.set_num_threads(min(4, torch.get_num_threads()))

    split = args.data_root / "train"
    try:
        samples = list_samples(split)
        smoke = find_smoke_tiles(split, samples, args.smoke_tiles) if args.smoke_tiles else None
    except ValueError as error:
        parser.error(str(error))

    torch.manual_seed(args.seed)
    rng = random.Random(args.seed)
    model = EarlyFusionUNet().to(args.device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    output = args.output_dir or RUNS_ROOT / ("unet_smoke" if smoke else "unet")
    output.mkdir(parents=True, exist_ok=True)
    if smoke:
        print("Smoke tiles: " + ", ".join(f"{sample} ({tile.x}, {tile.y})" for sample, tile in smoke))
    with (output / "history.csv").open("w", encoding="utf-8", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(("epoch", "train_loss"))
        try:
            for epoch in range(1, args.epochs + 1):
                tiles = (tile for _, tile in smoke) if smoke else shuffled_train_tiles(split, samples, rng)
                loss = train_epoch(model, optimizer, batch_tiles(tiles, args.batch_size), args.device)
                writer.writerow((epoch, loss))
                file.flush()
                print(f"Epoch {epoch}/{args.epochs} | Train loss: {loss:.6f}")
        except (ValueError, RuntimeError) as error:
            parser.error(str(error))

    checkpoint = output / "last.pt"
    save_checkpoint(checkpoint, model, optimizer, args.epochs, args.seed)
    print(f"Saved: {checkpoint}")
    if smoke:
        sample, tile = smoke[0]
        restored = load_model(checkpoint, args.device)
        score = save_smoke_preview(restored, sample, tile, output / "training_tile.png")
        print(f"Reloaded checkpoint | F1 on one training tile: {score:.4f}")


if __name__ == "__main__":
    main()
