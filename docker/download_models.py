#!/usr/bin/env python3
"""Prepare model files on the persistent RunPod volume safely and idempotently."""

from __future__ import annotations

import argparse
import fcntl
import os
import tempfile
from contextlib import contextmanager
from pathlib import Path
from typing import Callable

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


def is_valid_file(path: Path) -> bool:
    try:
        validate_nonempty(path)
        return True
    except Exception:
        return False


def is_valid_safetensors(path: Path) -> bool:
    try:
        validate_safetensors(path)
        return True
    except Exception:
        return False


def force_attempts() -> tuple[bool, ...]:
    return (True,) if FORCE else (False, True)


def download_file_validated(
    *, repo_id: str, filename: str, revision: str, local_dir: Path
) -> Path:
    last_error: Exception | None = None
    for force_download in force_attempts():
        path = Path(
            hf_hub_download(
                repo_id=repo_id,
                filename=filename,
                revision=revision,
                local_dir=local_dir,
                force_download=force_download,
                token=HF_TOKEN,
            )
        )
        try:
            validate_safetensors(path)
            return path
        except Exception as exc:
            last_error = exc
            print(f"WARNING: invalid cached download, retrying: {path}: {exc}")
            path.unlink(missing_ok=True)
    raise RuntimeError(f"Could not obtain a valid file: {repo_id}:{filename}: {last_error}")


def snapshot_validated(
    *,
    repo_id: str,
    revision: str,
    local_dir: Path,
    allow_patterns: list[str],
    validator: Callable[[Path], None],
) -> Path:
    last_error: Exception | None = None
    for force_download in force_attempts():
        path = Path(
            snapshot_download(
                repo_id=repo_id,
                revision=revision,
                local_dir=local_dir,
                allow_patterns=allow_patterns,
                force_download=force_download,
                token=HF_TOKEN,
            )
        )
        try:
            validator(path)
            return path
        except Exception as exc:
            last_error = exc
            print(f"WARNING: invalid local snapshot, retrying with force: {path}: {exc}")
    raise RuntimeError(f"Could not obtain a valid snapshot: {repo_id}: {last_error}")


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
        if link.is_dir():
            raise RuntimeError(f"Expected a file but found a directory: {link}")
        link.unlink()

    temporary = link.with_name(f".{link.name}.tmp-{os.getpid()}")
    temporary.unlink(missing_ok=True)
    temporary.symlink_to(target)
    os.replace(temporary, link)
    print(f"[ready] {link}")


def prepare_anima() -> None:
    transformer = MODEL_DIR / "transformers" / "anima.safetensors"
    vae = MODEL_DIR / "vae" / "qwen_image_vae.safetensors"
    need_transformer = FORCE or not is_valid_safetensors(transformer)
    need_vae = FORCE or not is_valid_safetensors(vae)

    if not need_transformer:
        print(f"[skip] {transformer}")
    if not need_vae:
        print(f"[skip] {vae}")
    if not need_transformer and not need_vae:
        return

    anima_store = MODEL_DIR / ".anima-source"
    if need_transformer:
        source = download_file_validated(
            repo_id=ANIMA_REPO,
            filename=ANIMA_FILE,
            revision=ANIMA_REVISION,
            local_dir=anima_store,
        )
        atomic_symlink(source, transformer)

    if need_vae:
        source = download_file_validated(
            repo_id=ANIMA_REPO,
            filename="split_files/vae/qwen_image_vae.safetensors",
            revision=ANIMA_REVISION,
            local_dir=anima_store,
        )
        atomic_symlink(source, vae)


def validate_qwen(target: Path) -> None:
    for name in ("config.json", "model.safetensors", "tokenizer_config.json"):
        validate_nonempty(target / name)
    validate_safetensors(target / "model.safetensors")
    tokenizer_ok = is_valid_file(target / "tokenizer.json") or (
        is_valid_file(target / "vocab.json") and is_valid_file(target / "merges.txt")
    )
    if not tokenizer_ok:
        raise RuntimeError(f"Qwen tokenizer files are incomplete under {target}")


def prepare_qwen() -> None:
    target = MODEL_DIR / "text_encoders"
    if not FORCE:
        try:
            validate_qwen(target)
            print(f"[skip] {target}")
            return
        except Exception:
            pass

    snapshot_validated(
        repo_id=QWEN_REPO,
        revision=QWEN_REVISION,
        local_dir=target,
        allow_patterns=[
            "config.json",
            "generation_config.json",
            "model.safetensors",
            "tokenizer.json",
            "tokenizer_config.json",
            "vocab.json",
            "merges.txt",
        ],
        validator=validate_qwen,
    )
    print(f"[ready] {target}")


def validate_t5(target: Path) -> None:
    for name in ("spiece.model", "tokenizer_config.json", "special_tokens_map.json"):
        validate_nonempty(target / name)


def prepare_t5() -> None:
    target = MODEL_DIR / "t5_tokenizer"
    if not FORCE:
        try:
            validate_t5(target)
            print(f"[skip] {target}")
            return
        except Exception:
            pass

    snapshot_validated(
        repo_id=T5_REPO,
        revision=T5_REVISION,
        local_dir=target,
        allow_patterns=[
            "spiece.model",
            "tokenizer_config.json",
            "special_tokens_map.json",
        ],
        validator=validate_t5,
    )
    print(f"[ready] {target}")


def prepare_models() -> None:
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    prepare_anima()
    prepare_qwen()
    prepare_t5()
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

        damaged = root / "damaged.bin"
        damaged.write_bytes(b"damaged")
        atomic_symlink(source, damaged)
        if not damaged.is_symlink() or damaged.read_bytes() != b"test\n":
            raise RuntimeError("damaged destination replacement self-test failed")
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
