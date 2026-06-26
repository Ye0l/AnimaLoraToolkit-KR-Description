#!/usr/bin/env python3
from __future__ import annotations

import argparse
import fcntl
import os
import shutil
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
ANIMA_REVISION = os.getenv("ANIMA_REVISION", "main")
ANIMA_FILE = os.getenv("ANIMA_FILE", "split_files/diffusion_models/anima-preview.safetensors")
QWEN_REPO = os.getenv("QWEN_REPO", "Qwen/Qwen3-0.6B-Base")
QWEN_REVISION = os.getenv("QWEN_REVISION", "main")
T5_REPO = os.getenv("T5_REPO", "google/t5-v1_1-xxl")
T5_REVISION = os.getenv("T5_REVISION", "main")

TRANSFORMER = MODEL_DIR / "transformers" / "anima-preview.safetensors"
VAE = MODEL_DIR / "vae" / "qwen_image_vae.safetensors"
QWEN = MODEL_DIR / "text_encoders"
T5 = MODEL_DIR / "t5_tokenizer"
STAGING = MODEL_DIR / ".downloads"


@contextmanager
def locked():
    lock_dir = WORKSPACE / ".locks"
    lock_dir.mkdir(parents=True, exist_ok=True)
    with (lock_dir / "model-download.lock").open("w") as handle:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
        yield


def validate_file(path: Path) -> None:
    if not path.is_file() or path.stat().st_size <= 0:
        raise RuntimeError(f"missing or empty file: {path}")


def validate_safetensors(path: Path) -> None:
    validate_file(path)
    with safe_open(path, framework="pt", device="cpu") as handle:
        if not list(handle.keys()):
            raise RuntimeError(f"no tensors found: {path}")


def valid(path: Path, validator) -> bool:
    try:
        validator(path)
        return True
    except Exception:
        return False


def install_file(repo: str, revision: str, remote: str, destination: Path) -> None:
    if not FORCE and valid(destination, validate_safetensors):
        print(f"[skip] {destination}")
        return

    stage = STAGING / repo.replace("/", "--")
    source = Path(hf_hub_download(
        repo_id=repo,
        filename=remote,
        revision=revision,
        local_dir=stage,
        force_download=FORCE,
        token=HF_TOKEN,
    ))
    validate_safetensors(source)

    destination.parent.mkdir(parents=True, exist_ok=True)
    partial = destination.with_name(f".{destination.name}.part")
    partial.unlink(missing_ok=True)
    try:
        os.replace(source, partial)
    except OSError:
        shutil.copy2(source, partial)
    validate_safetensors(partial)
    os.replace(partial, destination)
    print(f"[ready] {destination}")


def validate_qwen(path: Path) -> None:
    for name in ("config.json", "model.safetensors", "tokenizer_config.json"):
        validate_file(path / name)
    validate_safetensors(path / "model.safetensors")
    if not (path / "tokenizer.json").is_file() and not ((path / "vocab.json").is_file() and (path / "merges.txt").is_file()):
        raise RuntimeError(f"incomplete Qwen tokenizer: {path}")


def validate_t5(path: Path) -> None:
    for name in ("spiece.model", "tokenizer_config.json", "special_tokens_map.json"):
        validate_file(path / name)


def install_snapshot(repo: str, revision: str, destination: Path, patterns: list[str], validator) -> None:
    if not FORCE:
        try:
            validator(destination)
            print(f"[skip] {destination}")
            return
        except Exception:
            pass
    snapshot_download(
        repo_id=repo,
        revision=revision,
        local_dir=destination,
        allow_patterns=patterns,
        force_download=FORCE,
        token=HF_TOKEN,
    )
    validator(destination)
    print(f"[ready] {destination}")


def prepare() -> None:
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    install_file(ANIMA_REPO, ANIMA_REVISION, ANIMA_FILE, TRANSFORMER)
    install_file(ANIMA_REPO, ANIMA_REVISION, "split_files/vae/qwen_image_vae.safetensors", VAE)
    install_snapshot(QWEN_REPO, QWEN_REVISION, QWEN, [
        "config.json", "generation_config.json", "model.safetensors",
        "tokenizer.json", "tokenizer_config.json", "vocab.json", "merges.txt",
    ], validate_qwen)
    install_snapshot(T5_REPO, T5_REVISION, T5, [
        "spiece.model", "tokenizer_config.json", "special_tokens_map.json",
    ], validate_t5)
    print(f"Model preparation complete: {MODEL_DIR}")


def self_test() -> None:
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "test.bin"
        path.write_bytes(b"ok\n")
        validate_file(path)
    print("download_models_direct self-test: OK")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if args.self_test:
        self_test()
        return
    with locked():
        prepare()


if __name__ == "__main__":
    main()
