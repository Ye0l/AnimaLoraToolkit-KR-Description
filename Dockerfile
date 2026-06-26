# syntax=docker/dockerfile:1.7

ARG CUDA_IMAGE=nvidia/cuda:12.8.1-cudnn-runtime-ubuntu24.04
FROM ${CUDA_IMAGE}

ARG PYTHON_VERSION=3.12
ARG TORCH_VERSION=2.8.0
ARG TORCHVISION_VERSION=0.23.0

LABEL org.opencontainers.image.title="AnimaLoraToolkit RunPod"
LABEL org.opencontainers.image.description="RunPod-optimized Anima LoRA training image; model weights are downloaded at runtime"
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
        procps \
        python${PYTHON_VERSION} \
        python${PYTHON_VERSION}-venv \
        tini \
        tmux \
        util-linux \
    && rm -rf /var/lib/apt/lists/* \
    && python${PYTHON_VERSION} -m venv ${VIRTUAL_ENV}

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
    && rm -f /tmp/requirements.txt /tmp/requirements.runpod.txt

COPY . ${APP_DIR}

RUN chmod +x \
        ${APP_DIR}/scripts/runpod-entrypoint.sh \
        ${APP_DIR}/scripts/train-runpod.sh \
    && ln -s ${APP_DIR}/scripts/train-runpod.sh /usr/local/bin/train-runpod \
    && ln -s ${APP_DIR}/scripts/download_models.py /usr/local/bin/download-anima-models

VOLUME ["/workspace"]

WORKDIR ${APP_DIR}
ENTRYPOINT ["/usr/bin/tini", "--", "/opt/AnimaLoraToolkit/scripts/runpod-entrypoint.sh"]
CMD ["sleep", "infinity"]
