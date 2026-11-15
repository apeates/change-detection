"""Train the pinned TinyCD author model on LEVIR-CD train and validation."""

import argparse
from contextlib import contextmanager
from datetime import datetime, timezone
import hashlib
from importlib.metadata import version
import itertools
import json
import os
from pathlib import Path
import random
import shutil
import signal
import subprocess
import sys
from time import perf_counter
from urllib.request import urlopen
from uuid import uuid4

import cv2
import numpy as np
import torch
from torch import nn

from baseline_diff import metrics_from_counts
from tinycd_data import (
    BATCH_SIZE, COORDINATES, SEED, TILE_SIZE, PatchDataset, atomic_json, batch_signature,
    file_hash, hash_json, inventory, make_loader, prepare_cache, rng_for,
)


PILOT_EPOCHS = 10
TOTAL_EPOCHS = 40
PROBE_EPOCH = 9
PROBE_BATCHES = 3
RESTART_REQUIRED = 75
LR = 0.00356799066427741
WEIGHT_DECAY = 0.009449677083344786
THRESHOLD = 0.5
UPSTREAM = "1696827ffb7473c448591aaaa4434d4d4934a1ba"
UPSTREAM_FILES = {
    "change_classifier.py": "b9ec63a09859afe48617f4ffcb2a71990b4627bad082e39cc71e24e5ba291eab",
    "layers.py": "5fcfef498f68b53e069722d3d98dc6019f44fca2f59dc926d4aeabed9802c7fc",
}
IMAGENET_URL = "https://download.pytorch.org/models/efficientnet_b4_rwightman-23ab8bcd.pth"
FOOTPRINT_URL = ("https://raw.githubusercontent.com/justchenhao/STANet/master/"
                 "LEVIR_CD_spatial_info/LEVIR_CD_name_coords.geojson")
FOOTPRINT_HASH = "1f0d80046d85658be5091f2be0ffe746491889ee07524e8c4c9585b4aded0d64"
IMPLEMENTATION_FILES = ("train_tinycd.py", "tinycd_data.py", "levir_data.py",
                        "levir_tiles.py", "baseline_diff.py", "requirements-tinycd.txt",
                        "requirements-train.txt", "requirements.txt")


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def download_verified(url, target, sha256, limit):
    import tempfile

    target.parent.mkdir(parents=True, exist_ok=True)
    if target.is_file():
        if not file_hash(target).startswith(sha256):
            raise ValueError(f"Cached artifact SHA-256 mismatch: {target}")
        return
    handle = tempfile.NamedTemporaryFile(dir=target.parent, delete=False)
    temporary = Path(handle.name)
    try:
        digest = hashlib.sha256()
        size = 0
        with handle, urlopen(url, timeout=60) as response:
            while block := response.read(1024 * 1024):
                size += len(block)
                if size > limit:
                    raise ValueError(f"Artifact exceeds expected size limit: {url}")
                digest.update(block)
                handle.write(block)
        if not digest.hexdigest().startswith(sha256):
            raise ValueError(f"Artifact SHA-256 mismatch: {url}")
        temporary.replace(target)
    finally:
        temporary.unlink(missing_ok=True)


def model_class(work_dir):
    root = work_dir / f"tinycd-{UPSTREAM}"
    for name, digest in UPSTREAM_FILES.items():
        url = f"https://raw.githubusercontent.com/AndreaCodegoni/Tiny_model_4_CD/{UPSTREAM}/models/{name}"
        download_verified(url, root / "models" / name, digest, limit=100_000)
    (root / "models" / "__init__.py").touch()
    sys.path.insert(0, str(root))
    try:
        from models.change_classifier import ChangeClassifier
        if Path(sys.modules[ChangeClassifier.__module__].__file__).resolve() != (
            root / "models" / "change_classifier.py"
        ).resolve():
            raise RuntimeError("A different models package is already imported.")
        return ChangeClassifier
    finally:
        sys.path.remove(str(root))


def imagenet_source(work_dir):
    path = work_dir / "initialization" / IMAGENET_URL.rsplit("/", 1)[-1]
    download_verified(IMAGENET_URL, path, "23ab8bcd", limit=150_000_000)
    identity = {
        "name": "EfficientNet_B4_Weights.IMAGENET1K_V1",
        "url": IMAGENET_URL,
        "sha256": file_hash(path),
        "bytes": path.stat().st_size,
    }
    return path, identity


def build_model(model_cls, device, initial_weights=None):
    model = model_cls(bkbn_name="efficientnet_b4", pretrained=False,
                      output_layer_bkbn="3", freeze_backbone=False)
    if initial_weights is not None:
        full = torch.load(initial_weights, map_location="cpu", weights_only=True)
        backbone_state = {key: full[f"features.{key}"] for key in model._backbone.state_dict()}
        model._backbone.load_state_dict(backbone_state, strict=True)
    return model.to(device=device, dtype=torch.float32)


def seed_training():
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)


def collect_rng():
    kind, state, pos, cached, gaussian = np.random.get_state()
    return {
        "python": random.getstate(),
        "numpy": (kind, state.tolist(), pos, cached, gaussian),
        "torch": torch.get_rng_state(),
        "cuda": torch.cuda.get_rng_state_all(),
    }


def restore_rng(rng):
    random.setstate(rng["python"])
    kind, state, pos, cached, gaussian = rng["numpy"]
    np.random.set_state((kind, np.array(state, dtype=np.uint32), pos, cached, gaussian))
    torch.set_rng_state(rng["torch"])
    torch.cuda.set_rng_state_all(rng["cuda"])


def runtime_environment():
    driver = subprocess.check_output(
        ["nvidia-smi", "--query-gpu=name,driver_version", "--format=csv,noheader"],
        text=True, timeout=15,
    ).strip()
    return {
        "python": sys.version.split()[0],
        **{package: version(package) for package in
           ("torch", "torchvision", "numpy", "Pillow", "opencv-python-headless", "matplotlib")},
        "cuda": torch.version.cuda,
        "cudnn": torch.backends.cudnn.version(),
        "device": torch.cuda.get_device_name(0),
        "driver": driver,
    }


def footprint_audit(work_dir, source):
    path = work_dir / "footprints.geojson"
    download_verified(FOOTPRINT_URL, path, FOOTPRINT_HASH, limit=1_000_000)
    names = set(source["train"]) | set(source["val"])
    boxes = {}
    for feature in json.loads(path.read_text())["features"]:
        name = feature["properties"]["name"]
        if name in names:
            x, y = zip(*feature["geometry"]["coordinates"])
            if name in boxes:
                raise ValueError(f"Duplicate footprint: {name}")
            boxes[name] = (min(x), min(y), max(x), max(y))
    if boxes.keys() != names:
        raise ValueError(f"Missing train/val footprints: {sorted(names - boxes.keys())}")

    def separation(a, b):
        return min(a[2], b[2]) - max(a[0], b[0]), min(a[3], b[3]) - max(a[1], b[1])

    overlap = []
    boundaries = 0
    for a in source["train"]:
        for b in source["val"]:
            dx, dy = separation(boxes[a], boxes[b])
            if dx > 0 and dy > 0:
                overlap.append((a, b))
            elif dx >= 0 and dy >= 0:
                boundaries += 1
    if overlap:
        raise ValueError(f"Train/validation footprints overlap: {overlap}")

    # Keep edge-touching validation scenes in the same resampling unit.
    pending = set(source["val"])
    groups = []
    while pending:
        group = [min(pending)]
        pending.remove(group[0])
        for name in group:
            neighbors = {other for other in pending if min(separation(boxes[name], boxes[other])) >= 0}
            pending.difference_update(neighbors)
            group.extend(sorted(neighbors))
        groups.append(group)
    return {
        "source_sha256": FOOTPRINT_HASH,
        "train_val_overlap": overlap,
        "train_val_touches": boundaries,
        "val_footprint_groups": groups,
        "official_test_scope": "Known partial train/test footprint overlap; see docs/levir_split_audit.md.",
    }


def save_verified_checkpoint(output, local, checkpoint, history, previous):
    epoch = checkpoint["epoch"]
    filename = f"epoch_{epoch:02d}-{uuid4().hex[:8]}.pt"
    tmp = local / filename
    target = output / "checkpoints" / filename
    incoming = target.with_suffix(".part")
    try:
        torch.save(checkpoint, tmp)
        digest = file_hash(tmp)
        shutil.copyfile(tmp, incoming)
        if file_hash(incoming) != digest:
            raise OSError("Checkpoint hash mismatch after copying to Drive.")
        incoming.replace(target)
    finally:
        tmp.unlink(missing_ok=True)
        incoming.unlink(missing_ok=True)
    latest = {"epoch": epoch, "file": filename, "sha256": digest}
    row = history[-1]
    if previous is None or row["val"]["F1"] > previous["best"]["val_f1"]:
        best = {**latest, "val_f1": row["val"]["F1"]}
    else:
        best = previous["best"]
    pointer = {"latest": latest, "best": best}
    atomic_json(output / "epochs" / f"epoch_{epoch:02d}.json", row)
    atomic_json(output / "checkpoints.json", pointer)
    return pointer


def verified_checkpoint_path(output, record):
    filename = record["file"]
    if Path(filename).name != filename or not filename.endswith(".pt"):
        raise ValueError("Invalid checkpoint pointer.")
    path = output / "checkpoints" / filename
    if file_hash(path) != record["sha256"]:
        raise ValueError(f"Checkpoint SHA-256 mismatch: {filename}")
    return path


def read_checkpoint(output, pointer, identity):
    record = pointer["latest"]
    path = verified_checkpoint_path(output, record)
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    if checkpoint["identity"] != identity or checkpoint["epoch"] != record["epoch"]:
        raise ValueError("Resume checkpoint configuration/data identity mismatch.")
    return checkpoint


def gpu_start(device):
    torch.cuda.synchronize(device)
    torch.cuda.reset_peak_memory_stats(device)
    return perf_counter()


def gpu_end(device, started):
    torch.cuda.synchronize(device)
    return {
        "seconds": perf_counter() - started,
        "peak_allocated_bytes": torch.cuda.max_memory_allocated(device),
        "peak_reserved_bytes": torch.cuda.max_memory_reserved(device),
    }


def run_batch(model, loss_fn, batch, device):
    before = batch["a"].to(device)
    after = batch["b"].to(device)
    label = batch["label"].to(device)
    probabilities = model(before, after)
    loss = loss_fn(probabilities, label)
    if not torch.isfinite(loss):
        raise RuntimeError("Non-finite BCE loss.")
    return probabilities, label, loss


def finite_tensors(tensors):
    return bool(torch.stack([tensor.detach().isfinite().all() for tensor in tensors]).all())


def train_steps(model, optimizer, loss_fn, batches, device, *, check_updates=False):
    model.train()
    steps, seen, total = 0, 0, 0.0
    started = gpu_start(device)
    for batch in batches:
        optimizer.zero_grad(set_to_none=True)
        _, labels, loss = run_batch(model, loss_fn, batch, device)
        loss.backward()
        params = [param for param in model.parameters() if param.grad is not None]
        if not params or not finite_tensors(param.grad for param in params):
            raise RuntimeError("Missing or non-finite gradients.")
        before = [p.detach().clone() for p in params] if check_updates else ()
        optimizer.step()
        if check_updates or steps % 100 == 0:
            if not finite_tensors(model.parameters()):
                raise RuntimeError("Non-finite model parameters.")
        if check_updates and not any(not torch.equal(old, new) for old, new in zip(before, params)):
            raise RuntimeError("Optimizer left all parameters unchanged.")
        steps += 1
        seen += len(labels)
        total += loss.item() * len(labels)
        del labels, loss, _
        if steps % 100 == 0:
            print(f"Train updates: {steps}", flush=True)
    if not finite_tensors(model.parameters()) or not seen:
        raise RuntimeError("No training samples or non-finite model parameters.")
    return {"loss": total / seen, "steps": steps, "samples": seen, **gpu_end(device, started)}


@torch.inference_mode()
def validate(model, loss_fn, loader, device):
    model.eval()
    per_scene = {}
    hits, positives, total_loss = [], [], 0.0
    started = gpu_start(device)
    for batch in loader:
        probabilities, labels, loss = run_batch(model, loss_fn, batch, device)
        predicted, reference = probabilities > THRESHOLD, labels.bool()
        reduce = (1, 2, 3)
        counts = torch.stack([
            (predicted & reference).sum(reduce),
            (predicted & ~reference).sum(reduce),
            (~predicted & reference).sum(reduce),
        ], dim=1).cpu().tolist()
        hits.extend(predicted.sum(reduce).cpu().tolist())
        positives.extend(reference.sum(reduce).cpu().tolist())
        total_loss += loss.item() * len(labels)
        for scene, count in zip(batch["scene"], counts):
            previous = per_scene.setdefault(scene, [0, 0, 0])
            for index, value in enumerate(count):
                previous[index] += value
    if len(hits) != len(loader.dataset):
        raise RuntimeError("Validation did not cover the entire split.")
    tp, fp, fn = map(sum, zip(*per_scene.values()))
    if tp + fp != sum(hits) or tp + fn != sum(positives):
        raise RuntimeError("Inconsistent validation counts.")
    return {
        "loss": total_loss / len(hits),
        "tp": tp, "fp": fp, "fn": fn,
        **metrics_from_counts(tp, fp, fn),
        "pred_positive_fraction": sum(hits) / (len(hits) * TILE_SIZE**2),
        "label_positive_fraction": sum(positives) / (len(hits) * TILE_SIZE**2),
        "pred_empty_patch_fraction": hits.count(0) / len(hits),
        "label_empty_patch_fraction": positives.count(0) / len(hits),
        "pred_dense_patch_fraction": sum(i * 100 >= 95 * TILE_SIZE**2 for i in hits) / len(hits),
        "label_dense_patch_fraction": sum(i * 100 >= 95 * TILE_SIZE**2 for i in positives) / len(hits),
        "per_scene_tp_fp_fn": per_scene,
        **gpu_end(device, started),
    }


def plot_history(output, history):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    epochs = [row["epoch"] for row in history]
    fig, axes = plt.subplots(3, 1, figsize=(9, 9), sharex=True, layout="constrained")
    for split in ("train", "val"):
        axes[0].plot(epochs, [row[split]["loss"] for row in history], label=split)
    for metric in ("Precision", "Recall", "F1"):
        axes[1].plot(epochs, [row["val"][metric] for row in history], label=metric)
    for label, suffix in (("Boş patch", "empty_patch_fraction"), ("Yoğun patch", "dense_patch_fraction")):
        axes[2].plot(epochs, [row["val"][f"pred_{suffix}"] for row in history], label=f"{label} · tahmin")
        axes[2].plot(epochs, [row["val"][f"label_{suffix}"] for row in history], "--", label=f"{label} · etiket")
    for ax, title in zip(axes, ("BCE", "Validation değişim metrikleri", "Validation tahmin davranışı")):
        ax.set_title(title)
        ax.legend()
    axes[-1].set_xlabel("Epoch")
    stage = "pilot" if epochs[-1] == PILOT_EPOCHS else "final"
    fig.savefig(output / f"{stage}_curves.png", dpi=140)
    plt.close(fig)


def summarize(output, history, pointer, groups):
    best = pointer["best"]
    verified_checkpoint_path(output, best)
    plot_history(output, history)
    counts = history[best["epoch"] - 1]["val"]["per_scene_tp_fp_fn"]
    group_counts = np.array([np.sum([counts[name] for name in group], axis=0) for group in groups])
    uncertainty = {"group_count": len(groups), "method": "touching-footprint group bootstrap",
                   "status": "insufficient groups" if len(groups) < 10 else "measured"}
    if len(groups) >= 10:
        generator = rng_for("validation_bootstrap")
        draws = generator.integers(len(groups), size=(2000, len(groups)))
        sums = group_counts[draws].sum(axis=1)
        score = [metrics_from_counts(*row)["F1"] for row in sums]
        uncertainty["descriptive_val_f1_95_interval"] = np.quantile(score, [0.025, 0.975]).tolist()
    stage = "pilot" if history[-1]["epoch"] == PILOT_EPOCHS else "final"
    atomic_json(output / f"{stage}_summary.json", {
        "checkpoint": best, "uncertainty": uncertainty, "epochs_completed": history[-1]["epoch"],
        "continue_review": ("Review loss and validation precision/recall/F1 curves, empty/dense predictions "
                            "relative to target distribution, time, memory, and resume probe."),
        "history": history,
    })


@contextmanager
def run_lock(output):
    output.mkdir(parents=True, exist_ok=True)
    lock = output / ".writer"
    try:
        lock.mkdir()
    except FileExistsError as error:
        raise RuntimeError(f"Run locked: {lock}. Remove only after confirming its writer has stopped.") from error
    try:
        (lock / "owner.json").write_text(json.dumps({"pid": os.getpid(), "started_utc": utc_now()}))
        yield
    finally:
        shutil.rmtree(lock)


def data_context(args, code_id, work_dir):
    source = inventory(args.data_root)
    footprint = footprint_audit(work_dir, source)
    cache, audit = prepare_cache(args.data_root, args.cache_dir, source, code_id)
    manifest = {
        "source_png_sha256": source,
        "coordinate_order_xy": COORDINATES,
        "tile_size": TILE_SIZE,
        "decoded_sha256": audit["decoded_identity"],
        "footprint_audit": footprint,
    }
    return cache, source, audit, manifest


def run(args):
    cv2.setNumThreads(1)
    cv2.ocl.setUseOpenCL(False)
    root = Path(__file__).resolve().parent
    implementation = {name: file_hash(root / name) for name in IMPLEMENTATION_FILES}
    work_dir = args.cache_dir / "artifacts"
    if args.mode == "audit":
        _, _, audit, manifest = data_context(args, hash_json(implementation), work_dir)
        with run_lock(args.output_dir):
            atomic_json(args.output_dir / "data_audit.json", audit)
            atomic_json(args.output_dir / "data_manifest.json", manifest)
        return 0

    if not torch.cuda.is_available() or torch.cuda.get_device_name(0) not in {"Tesla T4", "NVIDIA T4", "T4"}:
        raise RuntimeError("Pilot and final training require a Tesla T4 Colab runtime.")
    if tuple(map(int, torch.__version__.split(".")[:2])) < (2, 6):
        raise RuntimeError("PyTorch 2.6 or later is required for restricted checkpoint loading.")
    if (np.__version__, version("opencv-python-headless")) != ("2.4.6", "4.10.0.84"):
        raise RuntimeError("Install the exact NumPy/OpenCV versions in requirements-tinycd.txt.")
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False
    device = torch.device("cuda:0")
    environment = runtime_environment()

    with run_lock(args.output_dir):
        return run_locked(args, implementation, work_dir, environment, device)


def run_locked(args, implementation, work_dir, environment, device):
    output = args.output_dir
    checkpoint_index = output / "checkpoints.json"
    pointer = json.loads(checkpoint_index.read_text()) if checkpoint_index.is_file() else None
    completed = pointer["latest"]["epoch"] if pointer else 0
    if args.mode == "continue" and completed < PILOT_EPOCHS:
        raise RuntimeError("Finish the approved 10-epoch pilot before continuing.")
    if args.mode == "continue" and not (output / "resume_probe.json").is_file():
        raise RuntimeError("The pilot's epoch 10 resume verification is missing.")
    end = PILOT_EPOCHS if args.mode == "pilot" else TOTAL_EPOCHS

    cache, source, audit, data_manifest = data_context(args, hash_json(implementation), work_dir)
    config = {
        "upstream_commit": UPSTREAM, "upstream_source_sha256": UPSTREAM_FILES,
        "project_revision": args.revision,
        "implementation": implementation, "environment": environment, "seed": SEED,
        "total_epochs": TOTAL_EPOCHS, "pilot_epochs": PILOT_EPOCHS, "batch_size": BATCH_SIZE,
        "bce": "sigmoid probabilities", "optimizer": "AdamW",
        "lr": LR, "weight_decay": WEIGHT_DECAY, "amsgrad": False,
        "scheduler": {"type": "CosineAnnealingLR", "T_max": TOTAL_EPOCHS, "eta_min": 0},
        "threshold": THRESHOLD, "decision": "strict greater than",
        "augmentation": "Project OpenCV/NumPy implementation; upstream equivalence unmeasured",
        "rotation": {"degrees": [-5, 5], "border": "BORDER_REFLECT_101",
                     "image_interpolation": "INTER_LINEAR", "label_interpolation": "INTER_NEAREST"},
        "loader": {"num_workers": 2, "context": "spawn", "persistent_workers": False, "prefetch_factor": 2},
    }
    data_id, config_id = hash_json(data_manifest), hash_json(config)
    run_file = output / "run.json"
    initial_weights = None
    if run_file.is_file():
        recorded = json.loads(run_file.read_text())
        if (config_id, data_id) != (recorded["config_sha256"], recorded["data_sha256"]):
            raise ValueError("Configuration, runtime, or train/val data changed since this run started.")
        saved_initialization = recorded["initialization"]
    else:
        if pointer:
            raise ValueError("Checkpoint pointer exists without its run manifest.")
        initial_weights, saved_initialization = imagenet_source(work_dir)
        saved_initialization["recorded_utc"] = utc_now()
        atomic_json(output / "data_manifest.json", data_manifest)
        atomic_json(output / "data_audit.json", audit)
        atomic_json(run_file, {"config": config, "config_sha256": config_id, "data_sha256": data_id,
                               "initialization": saved_initialization})

    if not pointer and initial_weights is None:
        initial_weights, checked = imagenet_source(work_dir)
        if checked["sha256"] != saved_initialization["sha256"]:
            raise ValueError("ImageNet initialization changed since run creation.")
    identity = [config_id, data_id, saved_initialization["sha256"]]
    if completed >= end:
        checkpoint = read_checkpoint(output, pointer, identity)
        if completed == end:
            summarize(output, checkpoint["history"], pointer, data_manifest["footprint_audit"]["val_footprint_groups"])
        print(f"{args.mode}: epoch {completed} already stored. {output}", flush=True)
        return 0

    (output / "epochs").mkdir(exist_ok=True)
    (output / "checkpoints").mkdir(exist_ok=True)
    model_cls = model_class(work_dir)

    def train_loader(epoch):
        return make_loader(PatchDataset(cache, "train", source["train"], epoch))

    def optimizer_for(model):
        return torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY, amsgrad=False)

    loss_fn = nn.BCELoss()
    if pointer is None:
        control_dir = output / "preflight" / uuid4().hex
        control_dir.mkdir(parents=True)
        seed_training()
        started = utc_now()
        try:
            control = build_model(model_cls, device, initial_weights)
            control_optimizer = optimizer_for(control)
            batches = itertools.islice(train_loader(1), PROBE_BATCHES)
            result = train_steps(control, control_optimizer, loss_fn, batches, device, check_updates=True)
            if result["steps"] != PROBE_BATCHES or result["samples"] != PROBE_BATCHES * BATCH_SIZE:
                raise RuntimeError("Preflight did not receive three full batches.")
            atomic_json(control_dir / "result.json", {"started_utc": started, "finished_utc": utc_now(),
                                                     "status": "passed", **result})
            del batches, control_optimizer, control
            torch.cuda.empty_cache()
        except Exception as error:
            atomic_json(control_dir / "result.json", {"started_utc": started, "finished_utc": utc_now(),
                                                     "status": "failed", "error": repr(error)})
            raise

    # The preflight model, optimizer and RNG state are discarded.
    seed_training()
    model = build_model(model_cls, device, initial_weights)
    optimizer = optimizer_for(model)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=TOTAL_EPOCHS, eta_min=0)
    history = []
    probe = None
    if pointer:
        checkpoint = read_checkpoint(output, pointer, identity)
        model.load_state_dict(checkpoint["model"], strict=True)
        optimizer.load_state_dict(checkpoint["optimizer"])
        scheduler.load_state_dict(checkpoint["scheduler"])
        restore_rng(checkpoint["rng"])
        history = checkpoint["history"]
        probe = checkpoint["resume_probe"]
        del checkpoint
        if scheduler.last_epoch != completed:
            raise RuntimeError("Scheduler state does not match the saved epoch.")
        if completed == PROBE_EPOCH and not probe:
            raise RuntimeError("Epoch 9 checkpoint is missing its required batch probe.")

    val_loader = make_loader(PatchDataset(cache, "val", source["val"]))
    for epoch in range(completed + 1, end + 1):
        started = utc_now()
        lr = optimizer.param_groups[0]["lr"]
        batches = iter(train_loader(epoch))
        if epoch == PROBE_EPOCH + 1:
            first = list(itertools.islice(batches, PROBE_BATCHES))
            actual = [batch_signature(batch, lr) for batch in first]
            if probe != actual or len(first) != PROBE_BATCHES:
                raise RuntimeError("Resume probe mismatch in LR, example order or transformed A/B/label tensors.")
            atomic_json(output / "resume_probe.json", {"epoch": epoch, "matched": True, "batches": actual})
            batches = itertools.chain(first, batches)

        train = train_steps(model, optimizer, loss_fn, batches, device)
        if train["samples"] != 7120 or train["steps"] != 890:
            raise RuntimeError("Train did not cover all 7,120 patches.")
        val = validate(model, loss_fn, val_loader, device)
        scheduler.step()
        next_lr = optimizer.param_groups[0]["lr"]
        history.append({"epoch": epoch, "started_utc": started, "finished_utc": utc_now(),
                        "lr": lr, "next_lr": next_lr, "train": train, "val": val})
        probe = None
        if epoch == PROBE_EPOCH:
            probe = [batch_signature(batch, next_lr) for batch in
                     itertools.islice(train_loader(epoch + 1), PROBE_BATCHES)]
            if len(probe) != PROBE_BATCHES:
                raise RuntimeError("Could not prepare three reference batches.")
        snapshot = {
            "epoch": epoch, "identity": identity, "model": model.state_dict(),
            "optimizer": optimizer.state_dict(), "scheduler": scheduler.state_dict(),
            "rng": collect_rng(), "history": history, "resume_probe": probe,
            "previous_best": pointer["best"] if pointer else None,
        }
        pointer = save_verified_checkpoint(output, work_dir, snapshot, history, pointer)
        print(f"Epoch {epoch}/{end} stored. Validation F1 {val['F1']:.5f}", flush=True)
        if epoch == PROBE_EPOCH:
            print("Restart this process with the same run path for the epoch 10 probe.", flush=True)
            return RESTART_REQUIRED
    summarize(output, history, pointer, data_manifest["footprint_audit"]["val_footprint_groups"])
    return 0


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", required=True, choices=("audit", "pilot", "continue"))
    parser.add_argument("--revision", help="Git commit checked out by the Colab notebook.")
    for name in ("data-root", "cache-dir", "output-dir"):
        parser.add_argument("--" + name, required=True, type=Path)
    args = parser.parse_args()
    args.data_root, args.cache_dir, args.output_dir = (
        path.resolve() for path in (args.data_root, args.cache_dir, args.output_dir)
    )

    def stop(_signum, _frame):
        raise KeyboardInterrupt("Training process received SIGTERM.")

    signal.signal(signal.SIGTERM, stop)
    return run(args)


if __name__ == "__main__":
    sys.exit(main())
