#!/usr/bin/env python3
"""Prepare model files on the persistent RunPod volume.

Downloads are idempotent and guarded by an advisory file lock. Large files stay
under /workspace; the image itself contains no model weights.
"""

from __future__ import annotations

import argparse
import fcntl
import json
import os
import tempfile
from contextlib import contextmanager
from pathlib import Path

from huggingface_hub import hf_hub_download, snapshot_download
from safetensors import safe_open

WORKSPACE = Path(os.getenv("WORKSPACE", "/workspace"))
MODEL_DIR = Path(os.getenv("MODEL_DIR", str(WORKSPACE / "models")))
HF_TOKEN = os.getenv("HF_TOKEN") or None
FORCE = os.getenv("FORCE_DOWNLOAD", "0").lower() in {"1", "true", "yes", "on"}

ANIMA_REPO = os.getenv("ANIMA_REPO", "circlestone-labs/Anima")
ANIMA_REVISION = os.getenv(
    "ANIMA_REVISION", "38371ccdbd541b114a9ddbc6b77cb31dae2c4027"
)
ANIMA_FILE = os.getenv(
    "ANIMA_FILE", "split_files/diffusion_models/anima-preview.safetensors"
)
QWEN_REPO = os.getenv("QWEN_REPO", "Qwen/Qwen3-0.6B-Base")
QWEN_REVISION = os.getenv(
    "QWEN_REVISION", "11214f7f3465775dcce23c3752ecea5a42ee0ddc"
)
T5_REPO = os.getenv("T5_REPO", "google/t5-v1_1-xxl")
T5_REVISION = os.getenv(
    "T5_REVISION", "3db67ab1af984cf10548a73467f0e5bca2aaaeb2"
)


@contextmanager
def download_lock():
    lock_dir = WORKSPACE / ".locks"
    lock_dir.mkdir(parents=True, exist_ok=True)
    with (lock_dir / "anima-model-download.lock").open("w") as handle:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
        yield


def validate_nonempty(path: Path) -> None:
    if not path.is_file() or path.stat().st_size <= 0:
        raise RuntimeError(f"Missing or empty file: {path}")


def validate_safetensors(path: Path) -> None:
    validate_nonempty(path)
    with safe_open(path, framework="pt", device="cpu") as handle:
        if not list(handle.keys()):
            raise RuntimeError(f"No tensors found in: {path}")


def atomic_symlink(target: Path, link: Path) -> None:
    target = target.resolve(strict=True)
    link.parent.mkdir(parents=True, exist_ok=True)

    if link.is_symlink():
        try:
            if link.resolve(strict=True) == target:
                print(f"[skip] {link}")
                return
        except FileNotFoundError:
            pass
        link.unlink()
    elif link.exists():
        if link.is_file() and link.stat().st_size > 0 and not FORCE:
            print(f"[skip:user-file] {link}")
            return
        if link.is_dir():
            raise RuntimeError(f"Expected a file but found a directory: {link}")
        link.unlink()

    temporary = link.with_name(f".{link.name}.tmp-{os.getpid()}")
    temporary.unlink(missing_ok=True)
    temporary.symlink_to(target)
    os.replace(temporary, link)
    print(f"[ready] {link}")


def prepare_models() -> None:
    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    anima_store = MODEL_DIR / ".anima-source"
    transformer_source = Path(
        hf_hub_download(
            repo_id=ANIMA_REPO,
            filename=ANIMA_FILE,
            revision=ANIMA_REVISION,
            local_dir=anima_store,
            force_download=FORCE,
            token=HF_TOKEN,
        )
    )
    vae_source = Path(
        hf_hub_download(
            repo_id=ANIMA_REPO,
            filename="split_files/vae/qwen_image_vae.safetensors",
            revision=ANIMA_REVISION,
            local_dir=anima_store,
            force_download=FORCE,
            token=HF_TOKEN,
        )
    )
    validate_safetensors(transformer_source)
    validate_safetensors(vae_source)

    atomic_symlink(transformer_source, MODEL_DIR / "transformers" / "anima.safetensors")
    atomic_symlink(vae_source, MODEL_DIR / "vae" / "qwen_image_vae.safetensors")

    qwen_dir = Path(
        snapshot_download(
            repo_id=QWEN_REPO,
            revision=QWEN_REVISION,
            local_dir=MODEL_DIR / "text_encoders",
            allow_patterns=[
                "config.json",
                "generation_config.json",
                "model.safetensors",
                "tokenizer.json",
                "tokenizer_config.json",
                "vocab.json",
                "merges.txt",
            ],
            force_download=FORCE,
            token=HF_TOKEN,
        )
    )
    t5_dir = Path(
        snapshot_download(
            repo_id=T5_REPO,
            revision=T5_REVISION,
            local_dir=MODEL_DIR / "t5_tokenizer",
            allow_patterns=[
                "spiece.model",
                "tokenizer_config.json",
                "special_tokens_map.json",
            ],
            force_download=FORCE,
            token=HF_TOKEN,
        )
    )

    required = [
        qwen_dir / "config.json",
        qwen_dir / "model.safetensors",
        qwen_dir / "tokenizer_config.json",
        t5_dir / "spiece.model",
        t5_dir / "tokenizer_config.json",
        t5_dir / "special_tokens_map.json",
    ]
    for path in required:
        validate_nonempty(path)
    validate_safetensors(qwen_dir / "model.safetensors")

    marker = MODEL_DIR / ".model-sources.json"
    marker_tmp = marker.with_suffix(".json.tmp")
    marker_tmp.write_text(
        json.dumps(
            {
                "anima": {"repo": ANIMA_REPO, "revision": ANIMA_REVISION, "file": ANIMA_FILE},
                "qwen": {"repo": QWEN_REPO, "revision": QWEN_REVISION},
                "t5": {"repo": T5_REPO, "revision": T5_REVISION},
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    os.replace(marker_tmp, marker)
    print(f"Model preparation complete: {MODEL_DIR}")


def self_test() -> None:
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        source = root / "source.bin"
        source.write_bytes(b"test\n")
        link = root / "nested" / "model.bin"
        atomic_symlink(source, link)
        atomic_symlink(source, link)
        if link.read_bytes() != b"test\n":
            raise RuntimeError("atomic symlink self-test failed")
    print("download_models self-test: OK")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if args.self_test:
        self_test()
        return
    with download_lock():
        prepare_models()


if __name__ == "__main__":
    main()
