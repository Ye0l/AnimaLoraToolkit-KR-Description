#!/usr/bin/env bash
set -Eeuo pipefail

workspace="${WORKSPACE:-/workspace}"
upload_dir="${UPLOAD_DIR:-/workspace/dataset}"
webui_port="${WEBUI_PORT:-7860}"

mkdir -p \
    "$workspace/logs" \
    "$workspace/.locks" \
    "$upload_dir" \
    "${MODEL_DIR:-/workspace/models}" \
    "${DATA_DIR:-/workspace/dataset}" \
    "${OUTPUT_DIR:-/workspace/output}" \
    "${HF_HOME:-/workspace/.cache/huggingface}" \
    "${TORCH_HOME:-/workspace/.cache/torch}" \
    /run/sshd \
    /root/.ssh

start_ssh() {
    [[ "${SSH_ENABLED:-1}" == "1" ]] || return 0

    chmod 700 /root/.ssh
    local ssh_key="${SSH_PUBLIC_KEY:-${PUBLIC_KEY:-}}"
    if [[ -n "$ssh_key" ]]; then
        printf '%s\n' "$ssh_key" > /root/.ssh/authorized_keys
        chmod 600 /root/.ssh/authorized_keys
    else
        echo "WARNING: SSH_PUBLIC_KEY/PUBLIC_KEY is empty; SSH key login will not work." >&2
    fi

    ssh-keygen -A >/dev/null
    /usr/sbin/sshd -t
    /usr/sbin/sshd
    echo "SSH server started on container port 22."
}

start_webui() {
    [[ "${WEBUI_ENABLED:-1}" == "1" ]] || return 0

    local -a command=(
        python -m uploadserver
        --bind 0.0.0.0
        --directory "$upload_dir"
        --theme dark
        --allow-replace
    )
    if [[ -n "${WEBUI_PASSWORD:-}" ]]; then
        command+=(--basic-auth "${WEBUI_USER:-runpod}:${WEBUI_PASSWORD}")
    else
        echo "WARNING: Upload Web UI has no password. Set WEBUI_PASSWORD." >&2
    fi
    command+=("$webui_port")

    "${command[@]}" > "$workspace/logs/uploadserver.log" 2>&1 &
    local webui_pid=$!
    local status=""

    for _ in $(seq 1 40); do
        if ! kill -0 "$webui_pid" 2>/dev/null; then
            break
        fi
        status="$(curl -sS -o /dev/null -w '%{http_code}' \
            "http://127.0.0.1:${webui_port}/upload" || true)"
        if [[ "$status" == "200" || "$status" == "401" ]]; then
            echo "Upload Web UI started: port=$webui_port path=/upload directory=$upload_dir"
            return 0
        fi
        sleep 0.25
    done

    echo "WARNING: Upload Web UI failed to become ready. Log follows:" >&2
    tail -n 80 "$workspace/logs/uploadserver.log" >&2 || true
    return 0
}

start_ssh
start_webui

if [[ "${AUTO_DOWNLOAD_MODELS:-1}" == "1" ]]; then
    if ! /usr/local/bin/download-anima-models; then
        echo "WARNING: Automatic model preparation failed; SSH and Web UI remain available." >&2
    fi
fi

user_config="$workspace/my_character.yaml"
if [[ ! -f "$user_config" ]]; then
    cp "${APP_DIR:-/opt/AnimaLoraToolkit}/config/runpod-docker.yaml" "$user_config"
    echo "Copied default training config to $user_config"
fi

if [[ "${AUTO_VALIDATE:-1}" == "1" ]]; then
    if /usr/local/bin/validate-anima-training "$user_config"; then
        echo "Preflight validation passed for $user_config. Training was not started automatically."
    else
        echo "NOTE: Preflight validation did not pass yet (e.g. dataset not uploaded). Upload your dataset and edit $user_config, then run: validate-anima-training $user_config" >&2
    fi
fi

if [[ "$#" -eq 0 ]]; then
    set -- sleep infinity
fi
exec "$@"
