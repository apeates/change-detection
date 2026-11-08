"""Train on LEVIR-CD's train split and select checkpoints on val."""

import argparse
import csv
import random
import shutil
from collections import Counter
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

from baseline_diff import change_counts, change_metrics, metrics_from_counts
from early_fusion_unet import UNET_MODELS, EarlyFusionUNet
from levir_data import DEFAULT_DATA_ROOT, DatasetError, list_samples, load_pair
from levir_tiles import TILE_SIZE, iter_tiles, orient_tile


RUNS_ROOT = Path(__file__).resolve().parent / "runs"
START_LR = 1e-3
END_LR = 1e-4
HISTORY_FIELDS = ("epoch", "learning_rate", "train_loss", "val_precision", "val_recall", "val_f1", "val_iou")


def epoch_learning_rate(epoch, schedule_epochs=None):
    if schedule_epochs is None or schedule_epochs == 1 or epoch == 1:
        return START_LR
    if epoch == schedule_epochs:
        return END_LR
    return START_LR + (END_LR - START_LR) * (epoch - 1) / (schedule_epochs - 1)


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


@torch.inference_mode()
def validate_epoch(model, split, samples, batch_size, device):
    model.eval()
    tiles = (tile for sample in samples for tile in iter_tiles(*load_pair(split, sample)))
    totals = Counter(tp=0, fp=0, fn=0)
    for batch in batch_tiles(tiles, batch_size):
        inputs = torch.from_numpy(np.stack([tile.inputs for tile in batch])).to(device)
        predictions = (model(inputs) > 0).cpu().numpy()
        references = np.stack([tile.mask for tile in batch]).astype(bool)
        totals.update(change_counts(predictions, references))
    return metrics_from_counts(**totals)


def save_checkpoint(path, model, optimizer, epoch, seed, *, history=None, config=None):
    temporary = path.with_suffix(".pt.tmp")
    torch.save({
        "architecture": model.architecture,
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "epoch": epoch,
        "seed": seed,
        "tile_size": TILE_SIZE,
        "history": history,
        "config": config,
    }, temporary)
    temporary.replace(path)


def copy_checkpoint(source, destination):
    temporary = destination.with_suffix(".pt.tmp")
    shutil.copyfile(source, temporary)
    temporary.replace(destination)


def save_history(path, history):
    temporary = path.with_suffix(".csv.tmp")
    with temporary.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=HISTORY_FIELDS)
        writer.writeheader()
        writer.writerows(history)
    temporary.replace(path)


def best_row(history):
    # max keeps the first epoch in case of a tie.
    return max(history, key=lambda row: row["val_f1"])


def load_model(path, device="cpu"):
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    architecture = checkpoint.get("architecture", EarlyFusionUNet.architecture)
    if architecture not in UNET_MODELS:
        raise ValueError(f"Unknown U-Net architecture in {path}: {architecture}")
    model = UNET_MODELS[architecture]()
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
    parser.add_argument("--model", choices=tuple(UNET_MODELS), default=EarlyFusionUNet.architecture)
    parser.add_argument("--device", choices=("cpu", "mps", "cuda"), default="cpu")
    parser.add_argument("--epochs", type=int, default=5, help="Total target epochs, including any resumed epochs.")
    parser.add_argument("--lr-schedule", choices=("constant", "linear"), default="constant",
                        help="Linear mode decreases Adam's learning rate from 0.001 to 0.0001 over the run.")
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--augment", action="store_true",
                        help="Randomly rotate or reflect aligned training tiles.")
    parser.add_argument("--smoke-tiles", type=int, help="Repeat this many positive train tiles and validate one image.")
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--resume", action="store_true", help="Continue from last.pt in the output directory.")
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
    val_split = args.data_root / "val"
    try:
        samples = list_samples(split)
        val_samples = list_samples(val_split)
        smoke = find_smoke_tiles(split, samples, args.smoke_tiles) if args.smoke_tiles else None
    except ValueError as error:
        parser.error(str(error))
    if smoke:
        val_samples = val_samples[:1]

    torch.manual_seed(args.seed)
    model = UNET_MODELS[args.model]().to(args.device)
    optimizer = torch.optim.Adam(model.parameters(), lr=START_LR)
    run_name = "unet" if args.model == EarlyFusionUNet.architecture else args.model
    output = args.output_dir or RUNS_ROOT / (f"{run_name}_smoke" if smoke else run_name)
    output.mkdir(parents=True, exist_ok=True)
    checkpoint = output / "last.pt"
    best_checkpoint = output / "best.pt"
    history_path = output / "history.csv"
    schedule_epochs = args.epochs if args.lr_schedule == "linear" else None
    config = {
        "seed": args.seed,
        "batch_size": args.batch_size,
        "smoke_tiles": args.smoke_tiles,
        "train_samples": len(samples),
        "val_samples": len(val_samples),
        "schedule_epochs": schedule_epochs,
        "augment": args.augment,
    }
    history = []
    first_epoch = 1
    if args.resume:
        if not checkpoint.is_file():
            parser.error(f"No checkpoint to resume: {checkpoint}")
        saved = torch.load(checkpoint, map_location="cpu", weights_only=True)
        saved_architecture = saved.get("architecture", EarlyFusionUNet.architecture)
        if saved_architecture != args.model:
            parser.error(f"Checkpoint uses {saved_architecture}. Use that model or a new output directory.")
        saved_config = saved.get("config")
        if (not saved_config or {"schedule_epochs": None, "augment": False, **saved_config} != config
                or not saved.get("history")):
            parser.error("Checkpoint is from an older trainer or the run settings changed. Use the original settings or a new output directory.")
        history = saved["history"]
        for row in history:
            row.setdefault("learning_rate", epoch_learning_rate(row["epoch"], schedule_epochs))
        first_epoch = saved["epoch"] + 1
        if saved["epoch"] > args.epochs:
            parser.error(f"Run already completed {saved['epoch']} epochs. Set a larger --epochs to continue.")
        model.load_state_dict(saved["model_state"])
        optimizer.load_state_dict(saved["optimizer_state"])
        if best_row(history)["epoch"] == saved["epoch"]:
            copy_checkpoint(checkpoint, best_checkpoint)
        elif not best_checkpoint.is_file():
            parser.error(f"Missing best checkpoint from an earlier epoch: {best_checkpoint}")
        save_history(history_path, history)
        print(f"Resuming from epoch {saved['epoch']}.")
    elif any(path.exists() for path in (checkpoint, best_checkpoint, history_path)):
        parser.error(f"Run files already exist in {output}. Use --resume or a new --output-dir.")

    print(f"Model: {args.model} | Augmentation: {'on' if args.augment else 'off'}")
    print(f"Validation images: {len(val_samples)}" + (" (smoke check)" if smoke else ""))
    if smoke:
        print("Smoke tiles: " + ", ".join(f"{sample} ({tile.x}, {tile.y})" for sample, tile in smoke))
    try:
        for epoch in range(first_epoch, args.epochs + 1):
            rng = random.Random(args.seed + epoch)
            tiles = (tile for _, tile in smoke) if smoke else shuffled_train_tiles(split, samples, rng)
            if args.augment:
                # Keep the baseline's tile order and replay the same transforms after a resume.
                # Reorienting both dates and the mask preserves the building-change label.
                augmentation_rng = random.Random(f"{args.seed}:{epoch}:augmentation")
                tiles = (orient_tile(tile, augmentation_rng.randrange(8)) for tile in tiles)
            learning_rate = epoch_learning_rate(epoch, schedule_epochs)
            for group in optimizer.param_groups:
                group["lr"] = learning_rate
            loss = train_epoch(model, optimizer, batch_tiles(tiles, args.batch_size), args.device)
            val_metrics = validate_epoch(model, val_split, val_samples, args.batch_size, args.device)
            history.append({
                "epoch": epoch,
                "learning_rate": learning_rate,
                "train_loss": loss,
                **{f"val_{name.lower()}": score for name, score in val_metrics.items()},
            })
            save_checkpoint(checkpoint, model, optimizer, epoch, args.seed, history=history, config=config)
            if best_row(history)["epoch"] == epoch:
                copy_checkpoint(checkpoint, best_checkpoint)
            save_history(history_path, history)
            print(f"Epoch {epoch}/{args.epochs} | LR: {learning_rate:.6g} | Train loss: {loss:.6f} | "
                  f"Val F1: {val_metrics['F1']:.4f} | Val IoU: {val_metrics['IoU']:.4f}")
    except (ValueError, RuntimeError) as error:
        parser.error(str(error))

    best = best_row(history)
    print(f"Last: {checkpoint}")
    print(f"Best: {best_checkpoint} | epoch {best['epoch']} | Val F1 {best['val_f1']:.4f}")
    if smoke:
        sample, tile = smoke[0]
        restored = load_model(checkpoint, args.device)
        score = save_smoke_preview(restored, sample, tile, output / "training_tile.png")
        print(f"Reloaded checkpoint | F1 on one training tile: {score:.4f}")


if __name__ == "__main__":
    main()
