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
        'PubkeyAuthentication yes' \
        > /etc/ssh/sshd_config.d/99-runpod.conf

WORKDIR ${APP_DIR}
COPY requirements.txt /tmp/requirements.txt

RUN python -m pip install --no-cache-dir --upgrade pip setuptools wheel \
    && python -m pip install --no-cache-dir \
        torch==${TORCH_VERSION} \
        torchvision==${TORCHVISION_VERSION} \
        --index-url https://download.pytorch.org/whl/cu128 \
    && grep -vE '^pillow-jxlpy([<>=!~].*)?$' /tmp/requirements.txt > /tmp/requirements.runpod.txt \
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
set -euo pipefail

MODEL_DIR="${MODEL_DIR:-/workspace/models}"
ANIMA_REPO="${ANIMA_REPO:-circlestone-labs/Anima}"
ANIMA_REVISION="${ANIMA_REVISION:-main}"
ANIMA_FILE="${ANIMA_FILE:-split_files/diffusion_models/anima-preview.safetensors}"

mkdir -p "$MODEL_DIR/transformers" "$MODEL_DIR/vae" "$MODEL_DIR/text_encoders" "$MODEL_DIR/t5_tokenizer"

fetch_file() {
    local repo="$1"
    local revision="$2"
    local remote_file="$3"
    local destination="$4"

    if [[ -s "$destination" ]]; then
        echo "[skip] $destination"
        return
    fi

    mkdir -p "$(dirname "$destination")"
    echo "[download] ${repo}@${revision}:${remote_file}"
    local cached
    cached="$(hf download "$repo" "$remote_file" --revision "$revision" --quiet)"

    if [[ -e "$destination" && "$cached" -ef "$destination" ]]; then
        echo "[ready] $destination"
        return
    fi

    local temporary="${destination}.part.$$"
    rm -f "$temporary"
    ln "$cached" "$temporary" 2>/dev/null || cp "$cached" "$temporary"

    if [[ -e "$destination" && "$temporary" -ef "$destination" ]]; then
        rm -f "$temporary"
        echo "[ready] $destination"
        return
    fi

    mv -f "$temporary" "$destination"
}

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

cat > /usr/local/bin/train-runpod <<'SCRIPT'
#!/usr/bin/env bash
set -euo pipefail

cd "${APP_DIR:-/opt/AnimaLoraToolkit}"
mkdir -p "${OUTPUT_DIR:-/workspace/output}" "${WORKSPACE:-/workspace}/logs"

if [[ "${DOWNLOAD_MODELS_ON_TRAIN:-1}" == "1" ]]; then
    download-anima-models
fi

config="${TRAIN_CONFIG:-/opt/AnimaLoraToolkit/config/runpod-docker.yaml}"
log_file="${TRAIN_LOG:-/workspace/logs/anima-training-$(date +%Y%m%d-%H%M%S).log}"
command=(python -u anima_train.py --config "$config" --no-monitor "$@")
printf -v quoted '%q ' "${command[@]}"

# script creates a pseudo-TTY, so Rich progress remains visible while the same
# output is also persisted to a log file. Do not pipe this command through tee.
exec script -q -f -e -c "$quoted" "$log_file"
SCRIPT

cat > /usr/local/bin/runpod-entrypoint <<'SCRIPT'
#!/usr/bin/env bash
set -euo pipefail

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
    if [[ -n "${PUBLIC_KEY:-}" ]]; then
        printf '%s\n' "$PUBLIC_KEY" > /root/.ssh/authorized_keys
        chmod 600 /root/.ssh/authorized_keys
    else
        echo "WARNING: PUBLIC_KEY is empty; SSH key login may not work. Add an SSH key to the RunPod account/template."
    fi
    ssh-keygen -A
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
    echo "Upload Web UI started on port $webui_port; upload page: /upload"
    echo "Upload destination: $upload_dir"
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

chmod +x /usr/local/bin/download-anima-models /usr/local/bin/train-runpod /usr/local/bin/runpod-entrypoint
SETUP

EXPOSE 22 7860
VOLUME ["/workspace"]
WORKDIR ${APP_DIR}
ENTRYPOINT ["/usr/bin/tini", "--", "/usr/local/bin/runpod-entrypoint"]
CMD ["sleep", "infinity"]
