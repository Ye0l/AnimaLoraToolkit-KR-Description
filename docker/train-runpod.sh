#!/usr/bin/env bash
set -Eeuo pipefail

app_dir="${APP_DIR:-/opt/AnimaLoraToolkit}"
workspace="${WORKSPACE:-/workspace}"
config="${TRAIN_CONFIG:-$app_dir/config/runpod-docker.yaml}"
log_file="${TRAIN_LOG:-$workspace/logs/anima-training-$(date +%Y%m%d-%H%M%S).log}"

cd "$app_dir"
mkdir -p "${OUTPUT_DIR:-/workspace/output}" "$workspace/logs"

if [[ "${DOWNLOAD_MODELS_ON_TRAIN:-1}" == "1" ]]; then
    /usr/local/bin/download-anima-models
fi

/usr/local/bin/validate-anima-training "$config"

command=(python -u anima_train.py --config "$config" --no-monitor "$@")
printf -v quoted '%q ' "${command[@]}"

# A pseudo-TTY keeps Rich live progress usable while script(1) persists output.
exec script -q -f -e -c "$quoted" "$log_file"
