# syntax=docker/dockerfile:1.7

ARG CUDA_IMAGE=nvidia/cuda:12.8.1-cudnn-runtime-ubuntu24.04
FROM ${CUDA_IMAGE}

ARG PYTHON_VERSION=3.12
ARG TORCH_VERSION=2.8.0
ARG TORCHVISION_VERSION=0.23.0

LABEL org.opencontainers.image.title="AnimaLoraToolkit RunPod"
LABEL org.opencontainers.image.description="RunPod-optimized Anima LoRA trainer; model weights are downloaded at runtime"
LABEL org.opencontainers.image.source="https://github.com/Ye0l/AnimaLoraToolkit-KR-Description"

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    VIRTUAL_ENV=/opt/venv \
    PATH=/opt/venv/bin:${PATH} \
    APP_DIR=/opt/AnimaLoraToolkit \
    WORKSPACE=/workspace \
    MODEL_DIR=/workspace/models \
    DATA_DIR=/workspace/dataset \
    OUTPUT_DIR=/workspace/output \
    UPLOAD_DIR=/workspace/dataset \
    SSH_ENABLED=1 \
    WEBUI_ENABLED=1 \
    WEBUI_PORT=7860 \
    WEBUI_USER=runpod \
    HF_HOME=/workspace/.cache/huggingface \
    HF_HUB_CACHE=/workspace/.cache/huggingface/hub \
    HF_XET_CACHE=/workspace/.cache/huggingface/xet \
    HF_XET_HIGH_PERFORMANCE=1 \
    HF_HUB_DISABLE_TELEMETRY=1 \
    HF_HUB_DISABLE_PROGRESS_BARS=1 \
    XDG_CACHE_HOME=/workspace/.cache \
    TORCH_HOME=/workspace/.cache/torch \
    TORCHINDUCTOR_CACHE_DIR=/workspace/.cache/torch/inductor \
    PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    CUDA_MODULE_LOADING=LAZY \
    TOKENIZERS_PARALLELISM=false \
    AUTO_DOWNLOAD_MODELS=1

RUN apt-get update && apt-get install -y --no-install-recommends \
        ca-certificates \
        curl \
        git \
        less \
        libgl1 \
        libglib2.0-0 \
        nano \
        openssh-server \
        procps \
        python${PYTHON_VERSION} \
        python${PYTHON_VERSION}-venv \
        tini \
        tmux \
        util-linux \
    && rm -rf /var/lib/apt/lists/* \
    && python${PYTHON_VERSION} -m venv ${VIRTUAL_ENV} \
    && mkdir -p /run/sshd /root/.ssh /etc/ssh/sshd_config.d \
    && chmod 700 /root/.ssh \
    && printf '%s\n' \
        'PermitRootLogin prohibit-password' \
        'PasswordAuthentication no' \
        'KbdInteractiveAuthentication no' \
        'PubkeyAuthentication yes' \
        > /etc/ssh/sshd_config.d/99-runpod.conf

WORKDIR ${APP_DIR}
COPY requirements.txt /tmp/requirements.txt

RUN python -m pip install --no-cache-dir --upgrade pip setuptools wheel \
    && python -m pip install --no-cache-dir \
        torch==${TORCH_VERSION} \
        torchvision==${TORCHVISION_VERSION} \
        --index-url https://download.pytorch.org/whl/cu128 \
    && grep -vE '^(torch|torchvision|pillow-jxlpy)([<>=!~].*)?$' /tmp/requirements.txt > /tmp/requirements.runpod.txt \
    && python -m pip install --no-cache-dir -r /tmp/requirements.runpod.txt \
    && python -m pip install --no-cache-dir \
        'huggingface_hub>=0.34.0' \
        hf-xet \
        protobuf \
        sentencepiece \
        tiktoken \
        uploadserver==6.0.1 \
    && rm -f /tmp/requirements.txt /tmp/requirements.runpod.txt

COPY . ${APP_DIR}

RUN <<'SETUP'
set -eu

cat > /usr/local/bin/download-anima-models <<'SCRIPT'
#!/usr/bin/env bash
set -Eeuo pipefail
umask 022

MODEL_DIR="${MODEL_DIR:-/workspace/models}"
WORKSPACE="${WORKSPACE:-/workspace}"
ANIMA_REPO="${ANIMA_REPO:-circlestone-labs/Anima}"
ANIMA_REVISION="${ANIMA_REVISION:-main}"
ANIMA_FILE="${ANIMA_FILE:-split_files/diffusion_models/anima-preview.safetensors}"

mkdir -p \
    "$MODEL_DIR/transformers" \
    "$MODEL_DIR/vae" \
    "$MODEL_DIR/text_encoders" \
    "$MODEL_DIR/t5_tokenizer" \
    "$WORKSPACE/.locks"

# Prevent boot-time auto-download and train-runpod from modifying the same files.
exec 9>"$WORKSPACE/.locks/anima-model-download.lock"
flock 9

validate_file() {
    local path="$1"
    [[ -f "$path" && -s "$path" ]] || return 1

    case "$path" in
        *.json)
            python - "$path" <<'PY'
import json
import sys
with open(sys.argv[1], "r", encoding="utf-8") as handle:
    json.load(handle)
PY
            ;;
        *.safetensors)
            python - "$path" <<'PY'
import sys
from safetensors import safe_open
with safe_open(sys.argv[1], framework="pt", device="cpu") as handle:
    if not list(handle.keys()):
        raise SystemExit("safetensors file has no tensors")
PY
            ;;
    esac
}

materialize_cached_file() {
    local cached="$1"
    local destination="$2"

    validate_file "$cached" || {
        echo "ERROR: Downloaded cache file is invalid: $cached" >&2
        return 1
    }

    if [[ -e "$destination" && "$cached" -ef "$destination" ]]; then
        validate_file "$destination"
        echo "[ready] $destination"
        return
    fi

    mkdir -p "$(dirname "$destination")"
    local temporary="${destination}.part.$$.${RANDOM}"
    rm -f -- "$temporary"

    if ! cp --reflink=auto --preserve=mode,timestamps -- "$cached" "$temporary"; then
        rm -f -- "$temporary"
        return 1
    fi

    if ! validate_file "$temporary"; then
        echo "ERROR: Temporary model file failed validation: $temporary" >&2
        rm -f -- "$temporary"
        return 1
    fi

    mv -fT -- "$temporary" "$destination"
}

fetch_file() {
    local repo="$1"
    local revision="$2"
    local remote_file="$3"
    local destination="$4"
    local source_spec="${repo}@${revision}:${remote_file}"
    local source_marker="${destination}.source"
    local force="${FORCE_DOWNLOAD:-0}"
    local recorded_source=""

    if [[ -f "$source_marker" ]]; then
        IFS= read -r recorded_source < "$source_marker" || true
    fi

    # No marker means a user-managed file. Preserve it unless force download is requested.
    if validate_file "$destination" && [[ "$force" != "1" ]] \
        && { [[ -z "$recorded_source" ]] || [[ "$recorded_source" == "$source_spec" ]]; }; then
        echo "[skip] $destination"
        return
    fi

    echo "[download] $source_spec"
    local cached
    cached="$(python - "$repo" "$remote_file" "$revision" "$force" <<'PY'
import os
import sys
from huggingface_hub import hf_hub_download

repo_id, filename, revision, force = sys.argv[1:5]
path = hf_hub_download(
    repo_id=repo_id,
    filename=filename,
    revision=revision,
    force_download=(force == "1"),
    token=os.environ.get("HF_TOKEN") or None,
)
print(path)
PY
)"

    materialize_cached_file "$cached" "$destination"

    local marker_tmp="${source_marker}.part.$$"
    printf '%s\n' "$source_spec" > "$marker_tmp"
    mv -fT -- "$marker_tmp" "$source_marker"
    echo "[ready] $destination"
}

run_self_test() {
    local test_dir
    test_dir="$(mktemp -d)"
    local source="$test_dir/source.bin"
    local destination="$test_dir/destination.bin"
    local same_file="$test_dir/same.bin"
    local empty_file="$test_dir/empty.bin"

    printf 'model-download-self-test\n' > "$source"
    materialize_cached_file "$source" "$destination"
    cmp -s "$source" "$destination"

    ln "$source" "$same_file"
    materialize_cached_file "$source" "$same_file"
    cmp -s "$source" "$same_file"

    : > "$empty_file"
    materialize_cached_file "$source" "$empty_file"
    cmp -s "$source" "$empty_file"

    rm -rf -- "$test_dir"
    echo "download-anima-models self-test: OK"
}

if [[ "${1:-}" == "--self-test" ]]; then
    run_self_test
    exit 0
fi

fetch_file "$ANIMA_REPO" "$ANIMA_REVISION" "$ANIMA_FILE" \
    "$MODEL_DIR/transformers/anima.safetensors"
fetch_file "$ANIMA_REPO" "$ANIMA_REVISION" \
    "split_files/vae/qwen_image_vae.safetensors" \
    "$MODEL_DIR/vae/qwen_image_vae.safetensors"

for file in config.json generation_config.json model.safetensors tokenizer.json tokenizer_config.json vocab.json merges.txt; do
    fetch_file "${QWEN_REPO:-Qwen/Qwen3-0.6B-Base}" "${QWEN_REVISION:-main}" "$file" \
        "$MODEL_DIR/text_encoders/$file"
done

for file in spiece.model tokenizer_config.json special_tokens_map.json; do
    fetch_file "${T5_REPO:-google/t5-v1_1-xxl}" "${T5_REVISION:-main}" "$file" \
        "$MODEL_DIR/t5_tokenizer/$file"
done

echo "Model preparation complete: $MODEL_DIR"
SCRIPT

cat > /usr/local/bin/validate-anima-training <<'SCRIPT'
#!/usr/bin/env python3
from __future__ import annotations

import sys
from pathlib import Path

import yaml
from safetensors import safe_open

config_path = Path(sys.argv[1] if len(sys.argv) > 1 else "/opt/AnimaLoraToolkit/config/runpod-docker.yaml")
if not config_path.is_file():
    raise SystemExit(f"Config file not found: {config_path}")

with config_path.open("r", encoding="utf-8") as handle:
    config = yaml.safe_load(handle) or {}

required = ("transformer_path", "vae_path", "text_encoder_path", "t5_tokenizer_path", "data_dir", "output_dir")
missing_keys = [key for key in required if not config.get(key)]
if missing_keys:
    raise SystemExit(f"Missing config keys: {', '.join(missing_keys)}")

for key in ("batch_size", "grad_accum", "epochs", "resolution"):
    value = int(config.get(key, 0))
    if value <= 0:
        raise SystemExit(f"{key} must be greater than 0, got {value}")

resolution = int(config["resolution"])
if resolution % 64:
    raise SystemExit(f"resolution must be divisible by 64, got {resolution}")

for key in ("transformer_path", "vae_path"):
    path = Path(config[key])
    if not path.is_file() or path.stat().st_size == 0:
        raise SystemExit(f"Missing or empty model file: {path}")
    with safe_open(path, framework="pt", device="cpu") as handle:
        keys = list(handle.keys())
        if not keys:
            raise SystemExit(f"No tensors found in {path}")
        if key == "transformer_path" and not any(name.endswith("x_embedder.proj.1.weight") for name in keys):
            raise SystemExit(f"The selected transformer does not look like a complete Anima checkpoint: {path}")

qwen = Path(config["text_encoder_path"])
for name in ("config.json", "model.safetensors", "tokenizer_config.json"):
    path = qwen / name
    if not path.is_file() or path.stat().st_size == 0:
        raise SystemExit(f"Missing Qwen file: {path}")
if not (qwen / "tokenizer.json").is_file() and not ((qwen / "vocab.json").is_file() and (qwen / "merges.txt").is_file()):
    raise SystemExit(f"Missing Qwen tokenizer files under {qwen}")

for name in ("spiece.model", "tokenizer_config.json", "special_tokens_map.json"):
    path = Path(config["t5_tokenizer_path"]) / name
    if not path.is_file() or path.stat().st_size == 0:
        raise SystemExit(f"Missing T5 tokenizer file: {path}")

data_dir = Path(config["data_dir"])
if not data_dir.is_dir():
    raise SystemExit(f"Dataset directory not found: {data_dir}")

extensions = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
images = [path for path in data_dir.rglob("*") if path.suffix.lower() in extensions]
paired = [
    path for path in images
    if path.with_suffix(".txt").is_file()
    or path.with_suffix(".caption").is_file()
    or (bool(config.get("prefer_json", True)) and path.with_suffix(".json").is_file())
]
if not paired:
    raise SystemExit(f"No image/caption pairs found under {data_dir}")
if len(paired) != len(images):
    print(f"WARNING: {len(images) - len(paired)} image(s) have no matching caption and will be skipped.", file=sys.stderr)

effective_samples = len(paired) * max(1, int(config.get("repeats", 1)))
effective_batch = int(config["batch_size"]) * int(config["grad_accum"])
if effective_samples < effective_batch:
    raise SystemExit(
        f"Dataset is too small for one optimizer step: samples={effective_samples}, effective_batch={effective_batch}"
    )

Path(config["output_dir"]).mkdir(parents=True, exist_ok=True)
print(
    f"Training preflight OK: pairs={len(paired)}, repeats={config.get('repeats', 1)}, "
    f"batch={config['batch_size']}x{config['grad_accum']}"
)
SCRIPT

cat > /usr/local/bin/train-runpod <<'SCRIPT'
#!/usr/bin/env bash
set -Eeuo pipefail

cd "${APP_DIR:-/opt/AnimaLoraToolkit}"
mkdir -p "${OUTPUT_DIR:-/workspace/output}" "${WORKSPACE:-/workspace}/logs"

if [[ "${DOWNLOAD_MODELS_ON_TRAIN:-1}" == "1" ]]; then
    download-anima-models
fi

config="${TRAIN_CONFIG:-/opt/AnimaLoraToolkit/config/runpod-docker.yaml}"
validate-anima-training "$config"

log_file="${TRAIN_LOG:-/workspace/logs/anima-training-$(date +%Y%m%d-%H%M%S).log}"
command=(python -u anima_train.py --config "$config" --no-monitor "$@")
printf -v quoted '%q ' "${command[@]}"

# script creates a pseudo-TTY, so Rich progress remains visible while the same
# output is also persisted to a log file. Do not pipe this command through tee.
exec script -q -f -e -c "$quoted" "$log_file"
SCRIPT

cat > /usr/local/bin/runpod-self-test <<'SCRIPT'
#!/usr/bin/env bash
set -Eeuo pipefail

bash -n /usr/local/bin/download-anima-models
bash -n /usr/local/bin/train-runpod
bash -n /usr/local/bin/runpod-entrypoint
python -m py_compile /opt/AnimaLoraToolkit/anima_train.py
python -m uploadserver --help >/dev/null
/usr/sbin/sshd -t
download-anima-models --self-test
python - <<'PY'
import yaml
from pathlib import Path
path = Path("/opt/AnimaLoraToolkit/config/runpod-docker.yaml")
config = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
required = {"transformer_path", "vae_path", "text_encoder_path", "t5_tokenizer_path", "data_dir", "output_dir"}
missing = sorted(required - config.keys())
if missing:
    raise SystemExit(f"Missing RunPod config keys: {missing}
")
PY
echo "RunPod image self-test: OK"
SCRIPT

cat > /usr/local/bin/runpod-entrypoint <<'SCRIPT'
#!/usr/bin/env bash
set -Eeuo pipefail

workspace="${WORKSPACE:-/workspace}"
upload_dir="${UPLOAD_DIR:-/workspace/dataset}"
webui_port="${WEBUI_PORT:-7860}"

mkdir -p \
    "${MODEL_DIR:-/workspace/models}" \
    "${DATA_DIR:-/workspace/dataset}" \
    "${OUTPUT_DIR:-/workspace/output}" \
    "$upload_dir" \
    "$workspace/logs" \
    "${HF_HOME:-/workspace/.cache/huggingface}" \
    "${TORCH_HOME:-/workspace/.cache/torch}" \
    /run/sshd \
    /root/.ssh

if [[ "${SSH_ENABLED:-1}" == "1" ]]; then
    chmod 700 /root/.ssh
    ssh_key="${SSH_PUBLIC_KEY:-${PUBLIC_KEY:-}}"
    if [[ -n "$ssh_key" ]]; then
        printf '%s\n' "$ssh_key" > /root/.ssh/authorized_keys
        chmod 600 /root/.ssh/authorized_keys
    else
        echo "WARNING: SSH_PUBLIC_KEY/PUBLIC_KEY is empty; SSH key login may not work."
    fi
    ssh-keygen -A
    /usr/sbin/sshd -t
    /usr/sbin/sshd
    echo "SSH server started on container port 22."
fi

if [[ "${WEBUI_ENABLED:-1}" == "1" ]]; then
    webui=(
        python -m uploadserver
        --bind 0.0.0.0
        --directory "$upload_dir"
        --theme dark
        --allow-replace
    )

    if [[ -n "${WEBUI_PASSWORD:-}" ]]; then
        webui+=(--basic-auth "${WEBUI_USER:-runpod}:${WEBUI_PASSWORD}")
    else
        echo "WARNING: Upload Web UI has no password. Set WEBUI_PASSWORD in the RunPod template."
    fi

    webui+=("$webui_port")
    "${webui[@]}" > "$workspace/logs/uploadserver.log" 2>&1 &
    webui_pid=$!

    webui_ready=0
    for _ in $(seq 1 30); do
        if ! kill -0 "$webui_pid" 2>/dev/null; then
            break
        fi
        status="$(curl -sS -o /dev/null -w '%{http_code}' "http://127.0.0.1:${webui_port}/upload" || true)"
        if [[ "$status" == "200" || "$status" == "401" ]]; then
            webui_ready=1
            break
        fi
        sleep 0.2
    done

    if [[ "$webui_ready" == "1" ]]; then
        echo "Upload Web UI started on port $webui_port; upload page: /upload"
        echo "Upload destination: $upload_dir"
    else
        echo "WARNING: Upload Web UI failed to start. Log follows:" >&2
        tail -n 50 "$workspace/logs/uploadserver.log" >&2 || true
    fi
fi

if [[ "${AUTO_DOWNLOAD_MODELS:-1}" == "1" ]]; then
    if ! download-anima-models; then
        echo "WARNING: Automatic model download failed. The container, SSH, and Upload Web UI will remain running." >&2
    fi
fi

if [[ "$#" -eq 0 ]]; then
    set -- sleep infinity
fi
exec "$@"
SCRIPT

cat > /opt/AnimaLoraToolkit/config/runpod-docker.yaml <<'YAML'
transformer_path: "/workspace/models/transformers/anima.safetensors"
vae_path: "/workspace/models/vae/qwen_image_vae.safetensors"
text_encoder_path: "/workspace/models/text_encoders"
t5_tokenizer_path: "/workspace/models/t5_tokenizer"

data_dir: "/workspace/dataset"
resolution: 1024
repeats: 2
prefer_json: false
shuffle_caption: true
keep_tokens: 1
flip_augment: false
tag_dropout: 0.05
cache_latents: true

lora_type: "lora"
lora_rank: 32
lora_alpha: 32.0
lokr_factor: 8

epochs: 10
max_steps: 0
batch_size: 2
grad_accum: 2
learning_rate: 0.0001
mixed_precision: "bf16"
grad_checkpoint: true
xformers: false
num_workers: 0

output_dir: "/workspace/output"
output_name: "my-character-anima"
save_every: 0
save_every_steps: 200
save_state_every: 1000
resume_lora: ""
resume_state: ""
seed: 42

sample_every: 0
sample_steps: 0
sample_prompt: "newest, safe, 1girl, YOUR_TRIGGER, solo, portrait, looking at viewer"
sample_prompts: []
sample_cfg_scale: 4.0
sample_negative_prompt: ""
sample_width: 1024
sample_height: 1024
sample_seed: 42
sample_infer_steps: 25
sample_sampler_name: "er_sde"
sample_scheduler: "simple"

loss_curve_steps: 100
no_progress: false
log_every: 10
no_monitor: true
no_browser: true
YAML

chmod +x \
    /usr/local/bin/download-anima-models \
    /usr/local/bin/validate-anima-training \
    /usr/local/bin/train-runpod \
    /usr/local/bin/runpod-self-test \
    /usr/local/bin/runpod-entrypoint

/usr/local/bin/runpod-self-test
SETUP

EXPOSE 22 7860
VOLUME ["/workspace"]
WORKDIR ${APP_DIR}
ENTRYPOINT ["/usr/bin/tini", "--", "/usr/local/bin/runpod-entrypoint"]
CMD ["sleep", "infinity"]
