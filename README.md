# AnimaLoraToolkit RunPod 한국어 가이드

이 저장소는 `Moeblack/AnimaLoraToolkit` 기반의 Anima LoRA 학습 도구와 RunPod용 Docker 이미지를 제공합니다.

이 문서는 **현재 `master` 브랜치의 Dockerfile과 `docker/` 스크립트 기준**입니다. 예전 수동 설치 방식과 현재 Docker 실행 방식을 섞지 않습니다.

## 현재 구성

- Ubuntu 24.04
- Python 3.12
- PyTorch 2.8.0
- CUDA 12.8
- SSH 서버
- 데이터셋 업로드 Web UI
- Jupyter Lab
- 시작 시 모델 자동 다운로드
- 학습 전 모델·데이터셋·CUDA 검사
- Rich 진행 화면과 로그 동시 저장

컨테이너 이미지는 다음 경로로 빌드됩니다.

```text
ghcr.io/ye0l/animaloratoolkit-kr-description:latest
```

GitHub Actions의 `Build RunPod image` 작업이 성공한 뒤 생성된 `latest`를 사용해야 합니다.

---

## 디렉터리 구조

```text
/opt/AnimaLoraToolkit/                 Docker 이미지에 포함된 프로그램 코드
├── anima_train.py
├── config/runpod-docker.yaml          이미지 기본 학습 설정
└── docker/                            RunPod 실행 스크립트

/workspace/                            RunPod 영구 볼륨
├── models/                            시작 시 받은 모델
│   ├── transformers/
│   │   └── anima-preview.safetensors
│   ├── vae/
│   │   └── qwen_image_vae.safetensors
│   ├── text_encoders/                 Qwen3-0.6B-Base
│   ├── t5_tokenizer/                  T5 v1.1 XXL tokenizer
│   └── .downloads/                    다운로드 중 사용하는 임시 공간
├── dataset/                           학습 이미지와 캡션
├── output/                            LoRA, 샘플, 학습 상태
├── logs/                              학습 및 업로드 서버 로그
└── my_character.yaml                  사용자가 복사해 수정하는 설정 예시
```

`/opt`는 이미지 코드이고 `/workspace`는 영구 데이터입니다. 팟을 중지하거나 이미지를 교체할 때 보존해야 하는 모델, 데이터셋, 결과물은 모두 `/workspace`에 둡니다.

---

# 1. RunPod Template 설정

## 이미지

```text
ghcr.io/ye0l/animaloratoolkit-kr-description:latest
```

## 포트

RunPod Template에서 다음 포트를 노출합니다.

```text
TCP  22
HTTP 7860
HTTP 8888
```

- `22`: SSH
- `7860`: 파일 업로드 Web UI
- `8888`: Jupyter Lab

## 권장 환경 변수

```text
WEBUI_USER=runpod
WEBUI_PASSWORD=충분히_긴_비밀번호
JUPYTER_TOKEN=충분히_긴_토큰
```

RunPod 계정에 SSH 공개키를 등록하면 보통 `PUBLIC_KEY` 환경 변수가 자동으로 전달됩니다. 직접 넣을 때는 다음 변수도 지원합니다.

```text
SSH_PUBLIC_KEY=ssh-ed25519 AAAA...
```

기본 동작을 바꾸는 변수:

```text
AUTO_DOWNLOAD_MODELS=1
SSH_ENABLED=1
WEBUI_ENABLED=1
WEBUI_PORT=7860
UPLOAD_DIR=/workspace/dataset
JUPYTER_ENABLED=1
JUPYTER_PORT=8888
```

`JUPYTER_TOKEN`을 지정하지 않으면 컨테이너가 시작할 때마다 무작위 토큰을 생성하고 시작 로그에 접속 URL과 함께 출력합니다. RunPod의 외부 포트 매핑을 통해 다음과 같이 접속합니다.

```text
https://<RunPod에서 매핑된 호스트:포트>/lab?token=<로그에 출력된 토큰>
```

Jupyter Lab을 끄려면:

```text
JUPYTER_ENABLED=0
```

모델 자동 다운로드를 끄려면:

```text
AUTO_DOWNLOAD_MODELS=0
```

---

# 2. 컨테이너 시작 동작

컨테이너는 다음 순서로 시작합니다.

```text
/workspace 디렉터리 생성
→ SSH 서버 시작
→ 업로드 Web UI 시작
→ Jupyter Lab 시작
→ 모델 존재 여부와 무결성 검사
→ 없거나 손상된 모델 다운로드
→ sleep infinity로 컨테이너 유지
```

학습은 자동으로 시작하지 않습니다.

시작 로그에서 다음과 비슷한 메시지를 확인할 수 있습니다.

```text
SSH server started on container port 22.
Upload Web UI started: port=7860 path=/upload directory=/workspace/dataset
Jupyter Lab started: port=8888 token=<생성된 토큰>
Access URL: http://<host>:8888/lab?token=<생성된 토큰>
Model preparation complete: /workspace/models
```

모델 다운로드가 실패해도 SSH, Web UI, Jupyter Lab은 계속 실행됩니다.

---

# 3. 모델 다운로드

기본적으로 다음 Hugging Face 저장소를 사용합니다.

```text
circlestone-labs/Anima
Qwen/Qwen3-0.6B-Base
google/t5-v1_1-xxl
```

Anima에서 받는 파일:

```text
split_files/diffusion_models/anima-preview.safetensors
split_files/vae/qwen_image_vae.safetensors
```

최종 저장 위치:

```text
/workspace/models/transformers/anima-preview.safetensors
/workspace/models/vae/qwen_image_vae.safetensors
/workspace/models/text_encoders/
/workspace/models/t5_tokenizer/
```

현재 다운로더는 최종 모델 경로에 일반 파일을 저장합니다. 최종 Anima와 VAE 파일을 Hugging Face 캐시로 연결하는 심볼릭 링크 방식은 사용하지 않습니다.

수동 실행:

```bash
download-anima-models
```

정상 파일이 이미 있으면 다음처럼 건너뜁니다.

```text
[skip] /workspace/models/transformers/anima-preview.safetensors
[skip] /workspace/models/vae/qwen_image_vae.safetensors
[skip] /workspace/models/text_encoders
[skip] /workspace/models/t5_tokenizer
```

다시 받으려면 한 번만 환경 변수를 붙입니다.

```bash
FORCE_DOWNLOAD=1 download-anima-models
```

기존 이미지에서 생성된 깨진 링크가 남아 있으면 확인 후 삭제할 수 있습니다.

```bash
find -L /workspace/models -type l -print
find -L /workspace/models -type l -delete
```

다운로드 저장소를 바꾸는 환경 변수:

```text
ANIMA_REPO
ANIMA_REVISION
ANIMA_FILE
QWEN_REPO
QWEN_REVISION
T5_REPO
T5_REVISION
HF_TOKEN
```

---

# 4. 데이터셋 업로드

브라우저에서 다음 주소를 엽니다.

```text
https://<POD_ID>-7860.proxy.runpod.net/upload
```

기본 업로드 위치:

```text
/workspace/dataset
```

이미지와 캡션의 파일명이 같아야 합니다.

```text
/workspace/dataset/
├── 001.png
├── 001.txt
├── 002.webp
├── 002.txt
└── ...
```

기본 설정은 TXT 캡션을 사용합니다.

```text
my_trigger, 1girl, solo, long hair, looking at viewer
```

지원 이미지 확장자:

```text
.jpg .jpeg .png .webp .bmp
```

지원 캡션:

```text
.txt
.caption
.json  # prefer_json: true일 때
```

업로드 서버 로그:

```text
/workspace/logs/uploadserver.log
```

---

# 5. SSH 접속

RunPod의 Connect 화면에 표시된 주소와 외부 포트를 사용합니다.

```bash
ssh root@<PUBLIC_IP> -p <EXTERNAL_SSH_PORT>
```

컨테이너 내부 SSH 포트는 `22`입니다. RunPod가 외부 포트를 별도로 할당할 수 있으므로 무조건 `-p 22`를 쓰면 안 됩니다.

SSH로 접속하면 자동으로 `/opt/venv`가 활성화되고 `tmux` 세션(`main`)에 붙습니다(연결이 끊겨도 작업이 유지됩니다). `btop`은 `--utf-force`가 alias로 적용되어 있어 그냥 `btop`만 입력하면 됩니다.

SSH 상태 확인:

```bash
pgrep -a sshd
ss -lntp | grep ':22'
```

공개키 확인:

```bash
cat /root/.ssh/authorized_keys
```

---

# 6. 학습 설정 만들기

이미지 기본 설정은 다음 위치에 있습니다.

```text
/opt/AnimaLoraToolkit/config/runpod-docker.yaml
```

컨테이너가 처음 시작될 때 `/workspace/my_character.yaml`이 없으면 자동으로 복사되고, 모델 다운로드 후 검사(`validate-anima-training`)까지 자동 실행됩니다(학습은 자동으로 시작되지 않습니다). 데이터셋 업로드 전이라면 이 검사는 실패하는 게 정상이니 무시하고 데이터셋을 올린 뒤 직접 다시 실행하면 됩니다. 직접 복사하려면:

```bash
cp /opt/AnimaLoraToolkit/config/runpod-docker.yaml \
   /workspace/my_character.yaml
```

최소한 다음 항목을 수정합니다.

```yaml
output_name: "my-character-anima"
sample_prompt: "newest, safe, 1girl, my_trigger, solo, portrait, looking at viewer"
```

기본 모델 경로는 다음과 일치해야 합니다.

```yaml
transformer_path: "/workspace/models/transformers/anima-preview.safetensors"
vae_path: "/workspace/models/vae/qwen_image_vae.safetensors"
text_encoder_path: "/workspace/models/text_encoders"
t5_tokenizer_path: "/workspace/models/t5_tokenizer"
```

기본 데이터 및 출력 경로:

```yaml
data_dir: "/workspace/dataset"
output_dir: "/workspace/output"
```

RTX 3090 24GB에서 사용한 기본 학습값:

```yaml
resolution: 1024
repeats: 2
batch_size: 2
grad_accum: 2
mixed_precision: "bf16"
grad_checkpoint: true
cache_latents: true
lora_type: "lora"
lora_rank: 32
lora_alpha: 32.0
epochs: 10
save_every_steps: 200
```

---

# 7. 학습 전 검사

설정과 데이터가 준비되면 직접 검사할 수 있습니다.

```bash
validate-anima-training /workspace/my_character.yaml
```

검사 항목:

- YAML 필수 키와 숫자 범위
- 해상도가 64의 배수인지
- Anima, VAE, Qwen safetensors가 열리는지
- Qwen과 T5 tokenizer가 로컬에서 로드되는지
- CUDA와 BF16 지원 여부
- 이미지 파일 손상 여부
- 이미지와 캡션 짝
- 빈 캡션과 잘못된 JSON
- 출력 디렉터리 쓰기 가능 여부

정상 예시:

```text
Training preflight OK: pairs=224, skipped=0, batch=2x2, resolution=1024
```

---

# 8. 학습 실행

```bash
TRAIN_CONFIG=/workspace/my_character.yaml train-runpod
```

`train-runpod`은 다음을 자동 수행합니다.

```text
모델 확인 및 누락 파일 다운로드
→ 학습 전 검사
→ anima_train.py 실행
→ 설정 확인/수정 TUI 표시 (주요 설정값 검토)
→ Rich 진행 화면 표시
→ 동일 출력을 로그에 저장
```

## 설정 확인/수정 TUI

학습이 실제로 시작되기 전에 주요 설정값을 표로 보여주고 그 자리에서 수정할 수 있습니다.

- **번호 입력** → 해당 설정 수정 (원하는 순서로 왔다갔다 수정 가능)
- **Enter 또는 `s`** → 현재 값으로 학습 시작
- **`e`** → 현재 적용값을 YAML로 export
- **`q`** → 취소

기본값은 `my_character.yaml`(및 CLI 인자)에서 읽은 값이며, 화면에 각 설정의 짧은 설명이 함께 표시됩니다. 학습이 시작되면 최종 적용값이 항상 `/workspace/output/applied_config_<시각>.yaml`로 자동 저장되어 재현이 가능합니다.

확인 없이 바로 시작하려면 `-y`(또는 `--yes`)를 붙입니다.

```bash
TRAIN_CONFIG=/workspace/my_character.yaml train-runpod -y
```

주요 설정 외에 이번에 노출된 항목:

- `timestep_shift` (기본 3.0): 학습 노이즈 분포 편향. 낮출수록(예: 1.0) 세부 디테일 비중↑.
- `train_llm_adapter` (기본 true): llm_adapter에도 LoRA를 넣을지. `false`로 두면 "트리거를 넣으면 구도가 통째로 딸려오는" 현상 완화에 도움될 수 있습니다.
- `tag_dropout`: 이제 TXT 캡션에서도 실제로 동작합니다(이전에는 JSON 전용).

로그 위치:

```text
/workspace/logs/anima-training-YYYYMMDD-HHMMSS.log
```

결과 위치:

```text
/workspace/output
```

`train-runpod` 실행 시 모델 확인을 생략하려면:

```bash
DOWNLOAD_MODELS_ON_TRAIN=0 \
TRAIN_CONFIG=/workspace/my_character.yaml \
train-runpod
```

Rich 진행 화면을 유지하려면 `tee`로 다시 파이프하지 마십시오. `train-runpod`이 내부에서 pseudo-TTY와 로그 저장을 처리합니다.

---

# 9. 재시작 시 동작

컨테이너를 재시작하면:

- `/workspace`의 모델, 데이터셋, 출력, 로그는 유지됩니다.
- SSH와 Web UI가 다시 시작됩니다.
- 모델 파일을 검사합니다.
- 정상 모델은 다시 받지 않습니다.
- 손상되거나 비어 있는 모델은 다시 받습니다.
- 학습은 자동 재개되지 않습니다.

학습 상태를 저장하고 재개하려면 YAML의 다음 값을 사용합니다.

```yaml
save_state_every: 1000
resume_state: "/workspace/output/<state-file>.pt"
```

---

# 10. 자주 쓰는 확인 명령

모델 파일:

```bash
find /workspace/models -maxdepth 2 -type f -printf '%12s  %p\n' | sort -n
```

깨진 링크:

```bash
find -L /workspace/models -type l -print
```

GPU 프로세스:

```bash
nvidia-smi
nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv
```

실행 서비스:

```bash
pgrep -a -f 'sshd|uploadserver|anima_train'
```

업로드 UI 로그:

```bash
tail -n 100 /workspace/logs/uploadserver.log
```

최신 학습 로그:

```bash
ls -1t /workspace/logs/anima-training-*.log | head -n 1
```

---

## 핵심 명령 요약

```bash
# 모델 준비
download-anima-models

# 설정 복사
cp /opt/AnimaLoraToolkit/config/runpod-docker.yaml \
   /workspace/my_character.yaml

# 사전 검사
validate-anima-training /workspace/my_character.yaml

# 학습
TRAIN_CONFIG=/workspace/my_character.yaml train-runpod
```

현재 Docker 실행 기준에서 사용하는 파일은 다음입니다.

```text
Dockerfile
docker/entrypoint.sh
docker/download_models_direct.py
docker/preflight.py
docker/train-runpod.sh
docker/runpod-self-test.sh
.github/workflows/docker-runpod.yml
```
