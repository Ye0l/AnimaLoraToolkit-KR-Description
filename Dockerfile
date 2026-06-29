# syntax=docker/dockerfile:1.7

ARG CUDA_IMAGE=nvidia/cuda:12.8.1-cudnn-runtime-ubuntu24.04
FROM ${CUDA_IMAGE}

ARG PYTHON_VERSION=3.12
ARG TORCH_VERSION=2.8.0
ARG TORCHVISION_VERSION=0.23.0

LABEL org.opencontainers.image.title="AnimaLoraToolkit RunPod"
LABEL org.opencontainers.image.description="RunPod-optimized Anima LoRA trainer"
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
    JUPYTER_ENABLED=1 \
    JUPYTER_PORT=8888 \
    AUTO_DOWNLOAD_MODELS=1 \
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
    TOKENIZERS_PARALLELISM=false

RUN apt-get update && apt-get install -y --no-install-recommends \
        btop \
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
        wget \
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
    && grep -vE '^(torch|torchvision|pillow-jxlpy)([<>=!~].*)?$' \
        /tmp/requirements.txt > /tmp/requirements.runpod.txt \
    && python -m pip install --no-cache-dir -r /tmp/requirements.runpod.txt \
    && python -m pip install --no-cache-dir --upgrade \
        'huggingface_hub>=0.36.0,<1.0' \
        hf-xet \
        protobuf \
        sentencepiece \
        tiktoken \
        uploadserver==6.0.1 \
        jupyterlab \
    && rm -f /tmp/requirements.txt /tmp/requirements.runpod.txt

COPY . ${APP_DIR}

RUN cat > ${APP_DIR}/config/runpod-docker.yaml <<'YAML'
transformer_path: "/workspace/models/transformers/anima-preview.safetensors"
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
train_llm_adapter: true
timestep_shift: 3.0
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

RUN cat >> /root/.bashrc <<'BASHRC'

# AnimaLoraToolkit: activate the training venv and alias btop's UTF-8 flag.
source /opt/venv/bin/activate
alias btop='btop --utf-force'

# Auto-attach (or create) a tmux session for interactive SSH logins only,
# so the CI smoke test's non-interactive "ssh ... true" check is unaffected.
if [[ $- == *i* ]] && [[ -z "${TMUX:-}" ]] && [[ -t 0 ]]; then
    exec tmux -u new-session -A -s main
fi
BASHRC

RUN printf '%s\n' \
        '[[ -f ~/.bashrc ]] && source ~/.bashrc' \
        > /root/.bash_profile

RUN chmod +x ${APP_DIR}/docker/*.sh ${APP_DIR}/docker/*.py \
    && ln -sf ${APP_DIR}/docker/download_models_direct.py /usr/local/bin/download-anima-models \
    && ln -sf ${APP_DIR}/docker/preflight.py /usr/local/bin/validate-anima-training \
    && ln -sf ${APP_DIR}/docker/train-runpod.sh /usr/local/bin/train-runpod \
    && ${APP_DIR}/docker/runpod-self-test.sh

EXPOSE 22 7860 8888
VOLUME ["/workspace"]
WORKDIR ${APP_DIR}
ENTRYPOINT ["/usr/bin/tini", "--", "/opt/AnimaLoraToolkit/docker/entrypoint.sh"]
CMD ["sleep", "infinity"]
