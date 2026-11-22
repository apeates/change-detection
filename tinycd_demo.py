"""Open the frozen TinyCD demo checkpoint on CPU using local model sources."""

import argparse
import hashlib
from pathlib import Path
import platform
import shutil
import sys
from tempfile import NamedTemporaryFile
from time import perf_counter

import torch


ARTIFACTS = Path(__file__).resolve().parent / "artifacts" / "tinycd"
CHECKPOINT_NAME = "epoch_40-cae2a604.pt"
CHECKPOINT_SHA256 = "dc5eca2b1ed9fa0a37fdf23993f111b9c6b06e71bb33bbcd1e6d5f3bdf64ba2a"
EPOCH = 40
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


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--install-checkpoint", type=Path, metavar="PATH",
                        help="Verify a downloaded checkpoint and copy it into artifacts/tinycd.")
    args = parser.parse_args()
    try:
        if args.install_checkpoint is not None:
            install_checkpoint(args.install_checkpoint)
        started = perf_counter()
        _, metadata = load_model()
    except (OSError, ValueError, RuntimeError) as error:
        parser.exit(1, f"{error}\n")
    print(f"Python {platform.python_version()} · PyTorch {torch.__version__}")
    print(f"TinyCD epoch {metadata['epoch']} · CPU float32 · {perf_counter() - started:.3f} s")
    print(f"Checkpoint SHA-256 {metadata['sha256']}")
    print(metadata["path"])


if __name__ == "__main__":
    main()
