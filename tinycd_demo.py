"""Load TinyCD offline and predict a full-resolution change mask on CPU."""

import argparse
import hashlib
import json
from pathlib import Path
import platform
import shutil
import sys
from tempfile import NamedTemporaryFile
from time import perf_counter

import torch
import numpy as np
from PIL import Image

from levir_data import DEFAULT_DATA_ROOT
from levir_tiles import TILE_SIZE, tile_regions


ARTIFACTS = Path(__file__).resolve().parent / "artifacts" / "tinycd"
CHECKPOINT_NAME = "epoch_40-cae2a604.pt"
CHECKPOINT_SHA256 = "dc5eca2b1ed9fa0a37fdf23993f111b9c6b06e71bb33bbcd1e6d5f3bdf64ba2a"
EPOCH = 40
SCENE_SIZE = 1024
BATCH_SIZE = 8
THRESHOLD = 0.5
MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32).reshape(1, 3, 1, 1)
STD = np.array([0.229, 0.224, 0.225], dtype=np.float32).reshape(1, 3, 1, 1)
DEMO_SCENES = ("val_56.png", "val_39.png")
TEST_DEMO_SCENES = ("test_102.png", "test_18.png", "test_108.png")
UPSTREAM_COMMIT = "1696827ffb7473c448591aaaa4434d4d4934a1ba"
MODEL_SOURCE_HASHES = {
    "models/__init__.py": "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855",
    "models/change_classifier.py": "b9ec63a09859afe48617f4ffcb2a71990b4627bad082e39cc71e24e5ba291eab",
    "models/layers.py": "5fcfef498f68b53e069722d3d98dc6019f44fca2f59dc926d4aeabed9802c7fc",
}


def verify_sha256(stream, path, expected):
    actual = hashlib.file_digest(stream, "sha256").hexdigest()
    if actual != expected:
        raise ValueError(f"{path}: SHA-256 {actual} (expected {expected})")
    return actual


def install_checkpoint(source):
    target = ARTIFACTS / CHECKPOINT_NAME
    if source.resolve() == target:
        return target
    with source.open("rb") as incoming:
        verify_sha256(incoming, source, CHECKPOINT_SHA256)
        incoming.seek(0)
        ARTIFACTS.mkdir(parents=True, exist_ok=True)
        with NamedTemporaryFile(dir=ARTIFACTS, delete=False) as outgoing:
            pending = Path(outgoing.name)
            try:
                shutil.copyfileobj(incoming, outgoing)
                outgoing.flush()
                with pending.open("rb") as copied:
                    verify_sha256(copied, pending, CHECKPOINT_SHA256)
                pending.replace(target)
            finally:
                pending.unlink(missing_ok=True)
    return target


def model_class():
    root = ARTIFACTS / f"tinycd-{UPSTREAM_COMMIT}"
    for name, expected in MODEL_SOURCE_HASHES.items():
        path = root / name
        with path.open("rb") as stream:
            verify_sha256(stream, path, expected)
        module = name.removesuffix(".py").removesuffix("/__init__").replace("/", ".")
        if module in sys.modules and Path(getattr(sys.modules[module], "__file__", "") or "").resolve() != path:
            raise RuntimeError(f"{module} is already imported from another location.")
    sys.path.insert(0, str(root))
    try:
        from models.change_classifier import ChangeClassifier

        return ChangeClassifier
    finally:
        sys.path.remove(str(root))


def load_model(checkpoint=None):
    path = checkpoint if checkpoint is not None else ARTIFACTS / CHECKPOINT_NAME
    with path.open("rb") as stream:
        digest = verify_sha256(stream, path, CHECKPOINT_SHA256)
        stream.seek(0)
        saved = torch.load(stream, map_location="cpu", weights_only=True)
    if saved.get("epoch") != EPOCH:
        raise ValueError(f"{path}: expected epoch {EPOCH}.")
    model = model_class()(bkbn_name="efficientnet_b4", pretrained=False,
                          output_layer_bkbn="3", freeze_backbone=False)
    model.load_state_dict(saved["model"], strict=True)
    model = model.to(device="cpu", dtype=torch.float32).eval()
    return model, {"epoch": EPOCH, "sha256": digest, "path": str(path)}


def normalized_batch(image, regions):
    batch = np.stack([image[region].transpose(2, 0, 1) for region in regions]).astype(np.float32)
    batch /= np.float32(255)
    return torch.from_numpy(np.ascontiguousarray((batch - MEAN) / STD))


@torch.inference_mode()
def predict_probabilities(model, before, after):
    """Predict from two aligned RGB arrays. Return a 1024×1024 float32 probability map."""
    shape = (SCENE_SIZE, SCENE_SIZE, 3)
    if any(image.shape != shape or image.dtype != np.uint8 for image in (before, after)):
        raise ValueError("Expected two aligned 1024×1024 uint8 RGB images.")
    model.eval()
    regions = [region for _, _, region in tile_regions(SCENE_SIZE, SCENE_SIZE)]
    probabilities_map = np.empty(shape[:2], dtype=np.float32)
    for start in range(0, len(regions), BATCH_SIZE):
        batch = regions[start:start + BATCH_SIZE]
        probabilities = model(normalized_batch(before, batch), normalized_batch(after, batch))
        if (probabilities.shape != (len(batch), 1, TILE_SIZE, TILE_SIZE)
                or not torch.isfinite(probabilities).all()):
            raise RuntimeError("TinyCD returned invalid probabilities.")
        for region, tile in zip(batch, probabilities[:, 0].numpy(), strict=True):
            probabilities_map[region] = tile
    return probabilities_map


@torch.inference_mode()
def predict_scene(model, before, after, threshold=THRESHOLD):
    """Predict from two aligned RGB arrays. Return a 1024×1024 boolean mask."""
    return predict_probabilities(model, before, after) > threshold


def count_regions(mask, min_pixels=16):
    """Count 8-connected foreground regions using fast row run-length union-find."""
    if mask.ndim != 2:
        raise ValueError("Expected a 2D boolean mask.")
    mask = np.asarray(mask, dtype=bool)
    if not mask.any():
        return {"count": 0, "total_pixels": 0, "filtered_pixels": 0}

    parent = []
    sizes = []

    def find(node):
        while parent[node] != node:
            parent[node] = parent[parent[node]]
            node = parent[node]
        return node

    def union(a, b):
        ra, rb = find(a), find(b)
        if ra == rb:
            return ra
        if sizes[ra] < sizes[rb]:
            ra, rb = rb, ra
        parent[rb] = ra
        sizes[ra] += sizes[rb]
        return ra

    padded = np.pad(mask, ((0, 0), (1, 1)), constant_values=False)
    diff = np.diff(padded.astype(np.int8), axis=1)
    prev_runs = []
    for row_index in range(mask.shape[0]):
        starts = np.flatnonzero(diff[row_index] == 1)
        if not len(starts):
            prev_runs = []
            continue
        ends = np.flatnonzero(diff[row_index] == -1)
        curr_runs = []
        prev_idx = 0
        prev_len = len(prev_runs)
        for start, end in zip(starts.tolist(), ends.tolist()):
            node = len(parent)
            parent.append(node)
            sizes.append(end - start)
            while prev_idx < prev_len and prev_runs[prev_idx][1] < start:
                prev_idx += 1
            scan = prev_idx
            while scan < prev_len and prev_runs[scan][0] <= end:
                node = union(node, prev_runs[scan][2])
                scan += 1
            curr_runs.append((start, end, node))
        prev_runs = curr_runs

    roots = {find(index) for index in range(len(parent))}
    kept = [sizes[root] for root in roots if sizes[root] >= min_pixels]
    return {
        "count": len(kept),
        "total_pixels": int(mask.sum()),
        "filtered_pixels": int(sum(kept)),
    }


def load_demo_image(path, mode="RGB"):
    with Image.open(path) as image:
        return np.asarray(image.convert(mode))


def check_validation_scenes(model, metadata, load_seconds):
    from baseline_diff import change_counts, metrics_from_counts

    split = DEFAULT_DATA_ROOT / "val"
    output = ARTIFACTS / "stage2"
    output.mkdir(parents=True, exist_ok=True)
    rows = {}
    for sample in DEMO_SCENES:
        before = load_demo_image(split / "A" / sample)
        after = load_demo_image(split / "B" / sample)
        started = perf_counter()
        prediction = predict_scene(model, before, after)
        elapsed = perf_counter() - started
        reference = load_demo_image(split / "label" / sample, "L") >= 128
        if reference.shape != prediction.shape:
            raise ValueError(f"{sample}: reference mask dimensions do not match.")
        counts = change_counts(prediction, reference)
        mask_path = output / f"{Path(sample).stem}_mask.png"
        Image.fromarray(prediction.astype(np.uint8) * 255).save(mask_path)
        rows[sample] = {
            **counts, **metrics_from_counts(**counts), "prediction_seconds": elapsed,
            "mask": str(mask_path),
        }
        print(f"{sample} · {elapsed:.3f} s · {counts}", flush=True)
    (output / "summary.json").write_text(json.dumps({
        "model": metadata, "split": split.name, "threshold": THRESHOLD, "decision": "p > 0.5",
        "tile_size": TILE_SIZE, "batch_size": BATCH_SIZE, "device": "cpu", "dtype": "float32",
        "torch_threads": torch.get_num_threads(), "model_load_seconds": load_seconds, "scenes": rows,
    }, indent=2) + "\n", encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--install-checkpoint", type=Path, metavar="PATH",
                        help="Verify a downloaded checkpoint and copy it into artifacts/tinycd.")
    parser.add_argument("--validate-scenes", action="store_true",
                        help="Predict and score the two fixed validation demo scenes.")
    args = parser.parse_args()
    try:
        if args.install_checkpoint is not None:
            install_checkpoint(args.install_checkpoint)
        started = perf_counter()
        model, metadata = load_model()
        load_seconds = perf_counter() - started
        if args.validate_scenes:
            check_validation_scenes(model, metadata, load_seconds)
    except (OSError, ValueError, RuntimeError) as error:
        parser.exit(1, f"{error}\n")
    print(f"Python {platform.python_version()} · PyTorch {torch.__version__}")
    print(f"TinyCD epoch {metadata['epoch']} · CPU float32 · {load_seconds:.3f} s")
    print(f"Checkpoint SHA-256 {metadata['sha256']}")
    print(metadata["path"])


if __name__ == "__main__":
    main()
