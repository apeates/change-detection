"""Train/validation patch indexing and deterministic TinyCD preprocessing."""

import hashlib
import json
import random
from pathlib import Path

import cv2
import numpy as np
from PIL import Image
import torch
from torch.utils.data import DataLoader, Dataset

from levir_data import PAIR_FOLDERS, official_samples
from levir_tiles import TILE_SIZE


SEED = 42
BATCH_SIZE = 8
SCENE_SIZE = 1024
COORDINATES = tuple(
    (x, y) for y in range(0, SCENE_SIZE, TILE_SIZE) for x in range(0, SCENE_SIZE, TILE_SIZE)
)
SPLITS = ("train", "val")
MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def compact_json(value):
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"), allow_nan=False).encode("utf-8")


def file_hash(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def hash_json(value):
    return hashlib.sha256(compact_json(value)).hexdigest()


def seed_for(*parts):
    digest = hashlib.sha256(compact_json([SEED, *parts])).digest()
    return int.from_bytes(digest[:8], "big")


def rng_for(*parts):
    return np.random.Generator(np.random.PCG64(seed_for(*parts)))


def atomic_json(path, value):
    tmp = path.with_suffix(path.suffix + ".part")
    tmp.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
                   encoding="utf-8")
    tmp.replace(path)


def inventory(root):
    """Hash only official train and validation source files."""
    result = {}
    for split in SPLITS:
        directory = root / split
        result[split] = {
            name: [file_hash(directory / folder / name) for folder in PAIR_FOLDERS]
            for name in official_samples(directory)
        }
    return result


def decode_scene(directory, name):
    frames = []
    for folder in PAIR_FOLDERS:
        with Image.open(directory / folder / name) as image:
            if image.size != (SCENE_SIZE, SCENE_SIZE):
                raise ValueError(f"{directory.name}/{folder}/{name}: expected 1024×1024, got {image.size}")
            mode = "L" if folder == "label" else "RGB"
            frames.append(np.asarray(image.convert(mode)))
    before, after, label = frames
    codes = np.unique(label).tolist()
    if not set(codes) <= {0, 1, 255} or {1, 255} <= set(codes):
        raise ValueError(f"{directory.name}/label/{name}: unsupported mask values {codes}")
    packed = np.empty((7, SCENE_SIZE, SCENE_SIZE), dtype=np.uint8)
    packed[:3], packed[3:6], packed[6] = before.transpose(2, 0, 1), after.transpose(2, 0, 1), label > 0
    return packed, codes


def summarize_split(scenes):
    patch_counts = [count for scene in scenes.values() for count in scene["change_pixels_per_patch"]]
    num_pixels = TILE_SIZE ** 2
    return {
        "pairs": len(scenes),
        "patches": len(patch_counts),
        "label_codes": sorted({code for scene in scenes.values() for code in scene["label_codes"]}),
        "positive_pixel_fraction": sum(patch_counts) / (len(patch_counts) * num_pixels),
        "empty_patch_fraction": sum(count == 0 for count in patch_counts) / len(patch_counts),
        "dense_patch_fraction": sum(count * 100 >= 95 * num_pixels for count in patch_counts) / len(patch_counts),
        "rgb_min_a_b": np.min([row["rgb_min_a_b"] for row in scenes.values()], axis=0).tolist(),
        "rgb_max_a_b": np.max([row["rgb_max_a_b"] for row in scenes.values()], axis=0).tolist(),
        "scenes": scenes,
    }


def prepare_cache(data_root, cache_root, source, build_id):
    """Cache raw, unaugmented pixels per scene on the local disk."""
    key = hash_json([build_id, source])
    cache = cache_root / key
    cache.mkdir(parents=True, exist_ok=True)
    audit_path = cache / "audit.json"
    if audit_path.is_file():
        audit = json.loads(audit_path.read_text())
        print(f"Using verified source identity for existing local cache {key[:12]}", flush=True)
        return cache, audit

    fingerprints = {}
    audit = {}
    decoded_identity = {}
    for split, samples in source.items():
        staging = cache / split
        staging.mkdir(exist_ok=True)
        scenes = {}
        for name in samples:
            packed, codes = decode_scene(data_root / split, name)
            if [file_hash(data_root / split / folder / name) for folder in PAIR_FOLDERS] != samples[name]:
                raise RuntimeError(f"Source changed during caching: {split}/{name}")
            pair_hash = hashlib.sha256(packed[:6]).hexdigest()
            other_split, other_name = fingerprints.setdefault(pair_hash, (split, name))
            if other_split != split:
                raise ValueError(f"Duplicate image pair across splits: {other_name}, {name}")
            if other_name != name:
                raise ValueError(f"Duplicate image pair within {split}: {other_name}, {name}")
            target = staging / f"{name}.npy"
            with target.with_suffix(".part").open("wb") as stream:
                np.save(stream, packed, allow_pickle=False)
            target.with_suffix(".part").replace(target)
            labels = packed[6]
            counts = [
                int(labels[y:y + TILE_SIZE, x:x + TILE_SIZE].sum()) for x, y in COORDINATES
            ]
            scenes[name] = {
                "positive_pixel_fraction": float(labels.mean()),
                "label_codes": codes,
                "change_pixels_per_patch": counts,
                "rgb_min_a_b": packed[:6].min(axis=(1, 2)).tolist(),
                "rgb_max_a_b": packed[:6].max(axis=(1, 2)).tolist(),
            }
            decoded_identity[f"{split}/{name}"] = hashlib.sha256(packed).hexdigest()
            if len(scenes) % 50 == 0:
                print(f"Audit/cache {split}: {len(scenes)}/{len(samples)}", flush=True)
        audit[split] = summarize_split(scenes)
    audit["decoded_identity"] = decoded_identity
    atomic_json(audit_path, audit)
    return cache, audit


def rgb_photo(rgb, rng):
    if rng.random() < 0.5:
        alpha = np.float32(rng.uniform(0.8, 1.2))
        beta = np.float32(rng.uniform(-0.2, 0.2))
        rgb = np.clip(rgb * alpha + beta, 0.0, 1.0)
    if rng.random() < 0.5:
        kernel = int(rng.choice((3, 5)))
        rgb = cv2.GaussianBlur(rgb, (kernel, kernel), sigmaX=0, sigmaY=0,
                               borderType=cv2.BORDER_REFLECT_101)
    return rgb


def rotate_pair(before, after, label, angle):
    center = ((TILE_SIZE - 1) / 2, (TILE_SIZE - 1) / 2)
    matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
    size = (TILE_SIZE, TILE_SIZE)
    border = cv2.BORDER_REFLECT_101
    return (
        cv2.warpAffine(before, matrix, size, flags=cv2.INTER_LINEAR, borderMode=border),
        cv2.warpAffine(after, matrix, size, flags=cv2.INTER_LINEAR, borderMode=border),
        cv2.warpAffine(label, matrix, size, flags=cv2.INTER_NEAREST, borderMode=border),
    )


def transform(tile, epoch, name, x, y):
    before = tile[:3].transpose(1, 2, 0).astype(np.float32) / np.float32(255)
    after = tile[3:6].transpose(1, 2, 0).astype(np.float32) / np.float32(255)
    label = np.array(tile[6], copy=True)
    if epoch:
        geometry = rng_for(epoch, name, x, y, "geometry")
        if geometry.random() < 0.5:
            axes = ((1,), (0,), (0, 1))[int(geometry.integers(3))]
            before, after, label = (
                np.ascontiguousarray(np.flip(array, axis=axes)) for array in (before, after, label)
            )
        if geometry.random() < 0.5:
            before, after, label = rotate_pair(before, after, label, geometry.uniform(-5, 5))
        before = rgb_photo(before, rng_for(epoch, name, x, y, "photo_a"))
        after = rgb_photo(after, rng_for(epoch, name, x, y, "photo_b"))
    return {
        "a": torch.from_numpy(np.ascontiguousarray(((before - MEAN) / STD).transpose(2, 0, 1))),
        "b": torch.from_numpy(np.ascontiguousarray(((after - MEAN) / STD).transpose(2, 0, 1))),
        "label": torch.from_numpy(label[None].astype(np.float32)),
        "scene": name,
        "id": f"{name}:{x}:{y}",
    }


class PatchDataset(Dataset):
    def __init__(self, cache, split, samples, epoch=0):
        if split not in SPLITS or (split == "train" and epoch < 1) or (split == "val" and epoch != 0):
            raise ValueError("Expected a positive train epoch or deterministic validation.")
        self.path = Path(cache) / split
        self.entries = tuple((name, x, y) for name in samples for x, y in COORDINATES)
        self.epoch = epoch

    def __len__(self):
        return len(self.entries)

    def __getitem__(self, index):
        name, x, y = self.entries[index]
        scene = np.load(self.path / f"{name}.npy", mmap_mode="r", allow_pickle=False)
        tile = scene[:, y:y + TILE_SIZE, x:x + TILE_SIZE]
        return transform(tile, self.epoch, name, x, y)


def initialize_worker(_worker):
    worker_seed = torch.initial_seed()
    random.seed(worker_seed)
    np.random.seed(worker_seed % 2**32)
    cv2.setNumThreads(1)
    cv2.ocl.setUseOpenCL(False)


def epoch_order(size, epoch):
    return rng_for(epoch, "sampler").permutation(size).tolist()


def make_loader(dataset, workers=2):
    epoch = dataset.epoch
    order = epoch_order(len(dataset), epoch) if epoch else range(len(dataset))
    loader_seed = torch.Generator().manual_seed(seed_for(epoch, "loader"))
    parallel = {"multiprocessing_context": "spawn", "prefetch_factor": 2} if workers else {}
    return DataLoader(
        dataset, batch_size=BATCH_SIZE, sampler=order, num_workers=workers,
        persistent_workers=False, worker_init_fn=initialize_worker, generator=loader_seed,
        pin_memory=False, **parallel,
    )


def batch_signature(batch, lr):
    signatures = {}
    for name in ("a", "b", "label"):
        array = batch[name].detach().cpu().contiguous().numpy()
        digest = hashlib.sha256(compact_json([array.dtype.str, list(array.shape)]))
        digest.update(memoryview(array).cast("B"))
        signatures[name] = digest.hexdigest()
    return {"ids": list(batch["id"]), "lr": lr, "sha256": signatures}
