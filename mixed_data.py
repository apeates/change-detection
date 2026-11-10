"""Audit official splits and stream balanced, reproducible mixed batches."""

import hashlib
import random
from collections.abc import Callable
from dataclasses import dataclass
from functools import partial
from itertools import islice

from evaluate_s2looking import IMAGE_SHAPE, S2LookingSplit
from levir_data import DatasetError, list_samples, load_pair
from levir_tiles import TILE_SIZE, iter_tiles


LEVIR = "LEVIR-CD"
S2LOOKING = "S2Looking"
LEVIR_COUNTS = {"train": 445, "val": 64, "test": 128}
S2LOOKING_COUNTS = {"train": 3500, "val": 500, "test": 1000}
TILES_PER_PAIR = (IMAGE_SHAPE[0] // TILE_SIZE) * (IMAGE_SHAPE[1] // TILE_SIZE)


@dataclass(frozen=True)
class Split:
    samples: list[str]
    read_sample: Callable


def open_splits(data_root, archive):
    datasets = {LEVIR: {}, S2LOOKING: {}}
    for split, count in LEVIR_COUNTS.items():
        path = data_root / split
        samples = list_samples(path)
        expected = {f"{split}_{i}.png" for i in range(1, count + 1)}
        missing = sorted(expected - set(samples))
        unexpected = sorted(set(samples) - expected)
        if missing or unexpected:
            raise DatasetError(f"Unexpected LEVIR-CD {split} split. Missing: {missing}. Extra: {unexpected}.")
        datasets[LEVIR][split] = Split(samples, partial(load_pair, path))
        s2 = S2LookingSplit(archive, split, S2LOOKING_COUNTS[split])
        datasets[S2LOOKING][split] = Split(s2.samples, s2.load_pair)
    return datasets


def audit_splits(datasets):
    """Check decoded RGB duplicates across split roles, including between datasets."""
    manifest = {}
    images_by_hash = {}
    invalid = []
    for domain, splits in datasets.items():
        manifest[domain] = {}
        for split, data in splits.items():
            signature = hashlib.sha256()
            positives, blank_masks = 0, 0
            for sample in data.samples:
                try:
                    before, after, mask = data.read_sample(sample)
                    if mask.shape != IMAGE_SHAPE or mask.dtype.kind != "b":
                        raise DatasetError("Expected a 1024×1024 binary mask.")
                except (OSError, ValueError) as error:
                    invalid.append(f"{domain}/{split}/{sample}: {error}")
                    continue
                signature.update(sample.encode() + b"\0")
                for date, image in (("T1", before), ("T2", after)):
                    digest = hashlib.sha256(image).digest()
                    signature.update(digest)
                    matches = images_by_hash.setdefault(digest, {})
                    matches.setdefault(split, []).append(f"{domain}/{split}/{sample}/{date}")
                signature.update(hashlib.sha256(mask).digest())
                positive = int(mask.sum())
                positives += positive
                blank_masks += positive == 0
            manifest[domain][split] = {
                "samples": data.samples,
                "decoded_sha256": signature.hexdigest(),
                "change_pixels": positives,
                "empty_masks": blank_masks,
            }
            print(f"Audited {domain}/{split}: {len(data.samples)} pairs.", flush=True)

    duplicates = [" = ".join(sample for group in matches.values() for sample in group)
                  for matches in images_by_hash.values() if len(matches) > 1]
    if invalid or duplicates:
        raise DatasetError("\n".join([*invalid, *(f"Cross-split RGB duplicate: {row}" for row in duplicates)]))
    return manifest


def tile_stream(data, domain, seed, start=0):
    """Seek by tile offset into independently shuffled, complete corpus passes."""
    cycle_length = len(data.samples) * TILES_PER_PAIR
    cycle, within_cycle = divmod(start, cycle_length)
    while True:
        samples = list(data.samples)
        random.Random(f"{seed}:{domain}:{cycle}").shuffle(samples)
        first_pair, first_tile = divmod(within_cycle, TILES_PER_PAIR)
        for index in range(first_pair, len(samples)):
            sample = samples[index]
            tiles = list(iter_tiles(*data.read_sample(sample)))
            if len(tiles) != TILES_PER_PAIR:
                raise DatasetError(f"{domain}/{sample}: expected {TILES_PER_PAIR} tiles.")
            random.Random(f"{seed}:{domain}:{cycle}:{sample}").shuffle(tiles)
            yield from tiles[first_tile:]
            first_tile = 0
        cycle += 1
        within_cycle = 0


def mixed_batches(datasets, seed, start, steps):
    streams = (tile_stream(datasets[domain]["train"], domain, seed, start) for domain in (LEVIR, S2LOOKING))
    return islice(zip(*streams), steps)
