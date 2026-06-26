#!/usr/bin/env python3
"""Fail before GPU allocation when a RunPod training input is incomplete."""

from __future__ import annotations

import os
import sys
from pathlib import Path

import torch
import yaml
from PIL import Image
from safetensors import safe_open

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}


def fail(message: str) -> None:
    raise SystemExit(f"Preflight failed: {message}")


def require_nonempty(path: Path, label: str) -> None:
    if not path.is_file() or path.stat().st_size <= 0:
        fail(f"missing or empty {label}: {path}")


def validate_safetensors(path: Path, label: str) -> list[str]:
    require_nonempty(path, label)
    try:
        with safe_open(path, framework="pt", device="cpu") as handle:
            keys = list(handle.keys())
    except Exception as exc:
        fail(f"invalid {label}: {path}: {exc}")
    if not keys:
        fail(f"no tensors found in {label}: {path}")
    return keys


def load_config(config_path: Path) -> dict:
    if not config_path.is_file():
        fail(f"config file not found: {config_path}")
    try:
        config = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    except Exception as exc:
        fail(f"invalid YAML: {config_path}: {exc}")
    if not isinstance(config, dict):
        fail("the YAML root must be a mapping")
    return config


def validate_config(config: dict) -> None:
    required = (
        "transformer_path",
        "vae_path",
        "text_encoder_path",
        "t5_tokenizer_path",
        "data_dir",
        "output_dir",
    )
    missing = [key for key in required if not config.get(key)]
    if missing:
        fail(f"missing config keys: {', '.join(missing)}")

    for key in ("batch_size", "grad_accum", "epochs", "resolution", "lora_rank"):
        try:
            value = int(config.get(key, 0))
        except (TypeError, ValueError):
            fail(f"{key} is not an integer: {config.get(key)!r}")
        if value <= 0:
            fail(f"{key} must be greater than 0, got {value}")

    resolution = int(config["resolution"])
    if resolution % 64:
        fail(f"resolution must be divisible by 64, got {resolution}")

    try:
        learning_rate = float(config.get("learning_rate", 0))
        tag_dropout = float(config.get("tag_dropout", 0))
    except (TypeError, ValueError) as exc:
        fail(f"invalid numeric option: {exc}")
    if learning_rate <= 0:
        fail(f"learning_rate must be greater than 0, got {learning_rate}")
    if not 0 <= tag_dropout <= 1:
        fail(f"tag_dropout must be between 0 and 1, got {tag_dropout}")


def validate_models(config: dict) -> None:
    transformer = Path(config["transformer_path"])
    vae = Path(config["vae_path"])
    transformer_keys = validate_safetensors(transformer, "Anima transformer")
    validate_safetensors(vae, "VAE")
    if not any(key.endswith("x_embedder.proj.1.weight") for key in transformer_keys):
        fail(f"checkpoint does not look like a complete Anima transformer: {transformer}")

    qwen = Path(config["text_encoder_path"])
    for name in ("config.json", "model.safetensors", "tokenizer_config.json"):
        require_nonempty(qwen / name, f"Qwen {name}")
    validate_safetensors(qwen / "model.safetensors", "Qwen model")
    has_tokenizer_json = (qwen / "tokenizer.json").is_file()
    has_bpe_pair = (qwen / "vocab.json").is_file() and (qwen / "merges.txt").is_file()
    if not has_tokenizer_json and not has_bpe_pair:
        fail(f"Qwen tokenizer files are incomplete under {qwen}")

    t5 = Path(config["t5_tokenizer_path"])
    for name in ("spiece.model", "tokenizer_config.json", "special_tokens_map.json"):
        require_nonempty(t5 / name, f"T5 tokenizer {name}")


def caption_for(image: Path, prefer_json: bool) -> Path | None:
    if prefer_json and image.with_suffix(".json").is_file():
        return image.with_suffix(".json")
    for suffix in (".txt", ".caption"):
        candidate = image.with_suffix(suffix)
        if candidate.is_file():
            return candidate
    return None


def validate_dataset(config: dict) -> tuple[int, int]:
    data_dir = Path(config["data_dir"])
    if not data_dir.is_dir():
        fail(f"dataset directory not found: {data_dir}")

    images = sorted(path for path in data_dir.rglob("*") if path.suffix.lower() in IMAGE_EXTENSIONS)
    if not images:
        fail(f"no supported images found under {data_dir}")

    pairs: list[tuple[Path, Path]] = []
    missing_captions: list[Path] = []
    prefer_json = bool(config.get("prefer_json", True))
    for image in images:
        caption = caption_for(image, prefer_json)
        if caption is None:
            missing_captions.append(image)
        else:
            pairs.append((image, caption))

    if not pairs:
        fail(f"no image/caption pairs found under {data_dir}")

    corrupt: list[str] = []
    for image, caption in pairs:
        try:
            with Image.open(image) as opened:
                opened.verify()
        except Exception as exc:
            corrupt.append(f"{image}: {exc}")
        try:
            text = caption.read_text(encoding="utf-8")
            if not text.strip():
                corrupt.append(f"{caption}: empty caption")
        except Exception as exc:
            corrupt.append(f"{caption}: {exc}")
    if corrupt:
        preview = "\n".join(corrupt[:10])
        fail(f"invalid dataset files ({len(corrupt)}):\n{preview}")

    repeats = max(1, int(config.get("repeats", 1)))
    effective_samples = len(pairs) * repeats
    batch_size = int(config["batch_size"])
    grad_accum = int(config["grad_accum"])
    if effective_samples < batch_size * grad_accum:
        fail(
            "dataset cannot produce one optimizer step: "
            f"samples={effective_samples}, batch_size={batch_size}, grad_accum={grad_accum}"
        )

    if missing_captions:
        print(
            f"WARNING: {len(missing_captions)} image(s) have no matching caption and will be skipped.",
            file=sys.stderr,
        )
    return len(pairs), len(missing_captions)


def validate_runtime(config: dict) -> None:
    if os.getenv("PREFLIGHT_REQUIRE_CUDA", "1").lower() not in {"0", "false", "no", "off"}:
        if not torch.cuda.is_available():
            fail("CUDA is not available inside the container")
        if config.get("mixed_precision") == "bf16" and not torch.cuda.is_bf16_supported():
            fail("the selected GPU does not support bf16")

    output_dir = Path(config["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    probe = output_dir / f".write-test-{os.getpid()}"
    try:
        probe.write_text("ok\n", encoding="utf-8")
    except Exception as exc:
        fail(f"output directory is not writable: {output_dir}: {exc}")
    finally:
        probe.unlink(missing_ok=True)


def main() -> None:
    config_path = Path(
        sys.argv[1] if len(sys.argv) > 1 else "/opt/AnimaLoraToolkit/config/runpod-docker.yaml"
    )
    config = load_config(config_path)
    validate_config(config)
    validate_models(config)
    pair_count, skipped = validate_dataset(config)
    validate_runtime(config)
    print(
        "Training preflight OK: "
        f"pairs={pair_count}, skipped={skipped}, "
        f"batch={config['batch_size']}x{config['grad_accum']}, resolution={config['resolution']}"
    )


if __name__ == "__main__":
    main()
