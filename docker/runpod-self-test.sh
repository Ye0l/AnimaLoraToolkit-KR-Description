#!/usr/bin/env bash
set -Eeuo pipefail

app_dir="${APP_DIR:-/opt/AnimaLoraToolkit}"

bash -n "$app_dir/docker/entrypoint.sh"
bash -n "$app_dir/docker/train-runpod.sh"
python -m py_compile \
    "$app_dir/anima_train.py" \
    "$app_dir/train_config_tui.py" \
    "$app_dir/docker/download_models_direct.py" \
    "$app_dir/docker/preflight.py"
python -m pip check
python -m uploadserver --help >/dev/null
/usr/sbin/sshd -t
python "$app_dir/docker/download_models_direct.py" --self-test

python - "$app_dir/config/runpod-docker.yaml" <<'PY'
from pathlib import Path
import sys
import yaml

path = Path(sys.argv[1])
config = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
required = {
    "transformer_path",
    "vae_path",
    "text_encoder_path",
    "t5_tokenizer_path",
    "data_dir",
    "output_dir",
}
missing = sorted(required - config.keys())
if missing:
    raise SystemExit(f"Missing RunPod config keys: {missing}")
PY

echo "RunPod image self-test: OK"
