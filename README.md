# AnimaLoraToolkit 한국어 가이드

이 저장소는 [Moeblack/AnimaLoraToolkit](https://github.com/Moeblack/AnimaLoraToolkit)을 기반으로 한 포크입니다.

원본 코드의 중국어 README 대신, **RunPod에서 Anima 단일 캐릭터 LoRA를 실제로 학습하면서 확인한 설치 과정과 오류 해결 방법**을 한국어로 정리했습니다.

> 이 문서는 2026-06-26 기준으로 작성되었습니다. 원본 프로젝트가 업데이트되면 일부 설정명이나 의존성이 달라질 수 있습니다.

---

## 검증한 환경

- RunPod
- NVIDIA RTX 3090 24GB
- Ubuntu 24.04 계열 컨테이너
- Python 3.12
- PyTorch 2.8.0 + CUDA 12.8
- 데이터셋 224장
- 모든 이미지 1024×1024 PNG
- 이미지와 같은 이름의 TXT 캡션 사용
- 단일 캐릭터 LoRA
- LoRA rank 32

실제 측정값은 대략 다음과 같았습니다.

- `batch_size: 1`, `grad_accum: 4`
- 약 `0.16 it/s`
- VRAM 약 11GB 사용
- GPU 사용률 100%

VRAM 사용량이 낮아 보여도 `cache_latents`, BF16, gradient checkpointing, LoRA 학습 구조 때문에 정상일 수 있습니다.

---

## ComfyUI가 필요한가?

필요하지 않습니다.

AnimaLoraToolkit은 독립적으로 학습할 수 있습니다. ComfyUI는 학습이 끝난 `.safetensors` LoRA를 나중에 테스트하거나 사용할 때만 필요합니다.

---

## 전체 과정

1. RunPod 생성
2. 저장소 clone
3. Python 가상환경 및 의존성 설치
4. Anima, VAE, Qwen3, T5 tokenizer 준비
5. 이미지와 TXT 캡션 준비
6. 데이터셋과 모델 검증
7. YAML 설정 작성
8. smoke test
9. 본 학습
10. 결과 LoRA 확인

---

# 1. 저장소 설치

```bash
cd /workspace

git clone https://github.com/Ye0l/AnimaLoraToolkit-KR-Description.git
cd AnimaLoraToolkit-KR-Description
```

RunPod의 CUDA PyTorch를 그대로 쓰기 위해 `--system-site-packages`를 사용합니다.

```bash
python -m venv .venv --system-site-packages
source .venv/bin/activate

python -m pip install --upgrade pip setuptools wheel
```

## `pillow-jxlpy` 설치 오류 우회

현재 `requirements.txt`에는 다음 항목이 있습니다.

```text
pillow-jxlpy>=0.9.0
```

일부 Python 3.12 Linux 환경에서는 다음 오류가 발생합니다.

```text
ERROR: Could not find a version that satisfies the requirement pillow-jxlpy>=0.9.0
ERROR: No matching distribution found for pillow-jxlpy>=0.9.0
```

PNG, JPG, WebP만 학습한다면 JPEG XL 지원은 필요하지 않으므로 해당 줄을 제외하고 설치할 수 있습니다.

```bash
grep -v 'pillow-jxlpy' requirements.txt > requirements.runpod.txt
python -m pip install -r requirements.runpod.txt
```

T5 tokenizer 검증에 필요한 패키지도 설치합니다.

```bash
python -m pip install \
  sentencepiece \
  tiktoken \
  protobuf \
  huggingface_hub
```

설치 상태 확인:

```bash
python -m pip check

python - <<'PY'
import torch
import sentencepiece
import tiktoken
import google.protobuf

print("torch:", torch.__version__)
print("torch CUDA:", torch.version.cuda)
print("CUDA available:", torch.cuda.is_available())
print("sentencepiece:", sentencepiece.__version__)
print("tiktoken:", tiktoken.__version__)
print("protobuf:", google.protobuf.__version__)

if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))
PY
```

정상이라면 다음이 포함되어야 합니다.

```text
CUDA available: True
GPU: NVIDIA GeForce RTX 3090
```

---

# 2. 모델 파일 준비

필요한 구성은 다음과 같습니다.

```text
models/
├── transformers/
│   └── anima.safetensors
├── vae/
│   └── qwen_image_vae.safetensors
├── text_encoders/
│   ├── config.json
│   ├── model.safetensors
│   ├── tokenizer.json
│   ├── tokenizer_config.json
│   ├── vocab.json
│   └── merges.txt
└── t5_tokenizer/
    ├── spiece.model
    ├── tokenizer_config.json
    └── special_tokens_map.json
```

## Anima와 VAE

다음 저장소에서 받습니다.

- [circlestone-labs/Anima](https://huggingface.co/circlestone-labs/Anima)

예시 경로:

```text
models/transformers/anima.safetensors
models/vae/qwen_image_vae.safetensors
```

Anima 파일명은 반드시 `anima-preview.safetensors`일 필요가 없습니다.

파일명이 `anima.safetensors`라면 설정 파일도 다음처럼 맞추면 됩니다.

```yaml
transformer_path: "models/transformers/anima.safetensors"
```

## Qwen3-0.6B-Base

Qwen 모델은 ComfyUI용 단일 text encoder 파일이 아니라 Hugging Face 디렉터리 형식이 필요합니다.

```bash
mkdir -p models/text_encoders

hf download Qwen/Qwen3-0.6B-Base \
  model.safetensors \
  config.json \
  tokenizer.json \
  tokenizer_config.json \
  vocab.json \
  merges.txt \
  --local-dir models/text_encoders
```

Qwen3 공식 저장소에는 `special_tokens_map.json`이 없을 수 있습니다. 없어도 `AutoTokenizer.from_pretrained()`가 정상적으로 로드된다면 문제없습니다.

다른 모델의 `special_tokens_map.json`을 임의로 복사하지 마십시오.

## T5 tokenizer

RunPod에서는 Hugging Face 공식 주소를 사용합니다.

```bash
python download_tokenizers.py --no-mirror
```

이 스크립트는 tokenizer 파일을 내려받습니다. Anima, VAE, Qwen3 모델 가중치는 별도로 준비해야 합니다.

---

# 3. 모델 검증

```bash
source .venv/bin/activate
python validate_local_models.py
```

정상 결과:

```text
[PASS] T5 Tokenizer
[PASS] Qwen Tokenizer
[PASS] Qwen Model
[PASS] Encode Workflow
```

## T5 tokenizer 오류

### `sentencepiece`가 없다는 오류

```text
SentencePieceExtractor requires the SentencePiece library
```

해결:

```bash
python -m pip install sentencepiece tiktoken
```

### `protobuf`가 없다는 오류

```text
SentencePieceExtractor requires the protobuf library
```

해결:

```bash
python -m pip install protobuf
```

이때 함께 나타나는 다음 오류는 `protobuf` 경로가 실패한 뒤 tiktoken fallback이 바이너리 `spiece.model`을 잘못 읽어서 발생한 2차 오류입니다.

```text
Error parsing line ... spiece.model
```

`protobuf`, `sentencepiece`, `tiktoken`을 모두 설치한 뒤 다시 검증하면 됩니다.

---

# 4. 데이터셋 준비

예시:

```text
/workspace/anima-dataset/
├── 001.png
├── 001.txt
├── 002.png
├── 002.txt
└── ...
```

이미지와 캡션 파일의 stem이 같아야 합니다.

```text
001.png ↔ 001.txt
```

## TXT 캡션 예시

```text
my_trigger, 1girl, solo, long hair, black hair, blue eyes, looking at viewer
```

트리거 태그를 모든 캡션의 첫 번째에 두면 다음 설정을 사용할 수 있습니다.

```yaml
shuffle_caption: true
keep_tokens: 1
```

캡션이 다음처럼 시작한다면:

```text
safe, 1girl, my_trigger, solo, ...
```

앞의 세 태그를 고정하려면:

```yaml
keep_tokens: 3
```

## 품질 태그

`best quality`, `masterpiece`, `score_7` 같은 품질 태그는 필수가 아닙니다.

단일 캐릭터 데이터가 이미 일정한 품질로 선별되어 있다면 품질 태그 없이 학습해도 됩니다. 잘못된 품질 태그를 모든 이미지에 일괄 삽입하는 것보다 생략하는 편이 낫습니다.

---

# 5. 데이터셋 검증

이미지와 TXT 수 확인:

```bash
cd /workspace

echo "PNG:"
find anima-dataset -maxdepth 1 -type f -iname '*.png' | wc -l

echo "TXT:"
find anima-dataset -maxdepth 1 -type f -iname '*.txt' | wc -l
```

파일 대응 확인:

```bash
python - <<'PY'
from pathlib import Path

root = Path("/workspace/anima-dataset")

images = {p.stem for p in root.glob("*.png")}
captions = {p.stem for p in root.glob("*.txt")}

print("PNG:", len(images))
print("TXT:", len(captions))
print("캡션 누락:", sorted(images - captions))
print("이미지 누락:", sorted(captions - images))

empty = [
    str(p) for p in root.glob("*.txt")
    if not p.read_text(encoding="utf-8").strip()
]
print("빈 캡션:", empty)
PY
```

정상 예시:

```text
PNG: 224
TXT: 224
캡션 누락: []
이미지 누락: []
빈 캡션: []
```

이미지 무결성과 해상도 확인:

```bash
python - <<'PY'
from pathlib import Path
from PIL import Image

root = Path("/workspace/anima-dataset")
errors = []
sizes = []

for path in root.glob("*.png"):
    try:
        with Image.open(path) as image:
            image.verify()

        with Image.open(path) as image:
            sizes.append((image.width, image.height, path.name))
    except Exception as e:
        errors.append((path.name, str(e)))

print("valid:", len(sizes))
print("errors:", len(errors))

for item in errors:
    print("ERROR:", item)

if sizes:
    widths = [x[0] for x in sizes]
    heights = [x[1] for x in sizes]

    print("width range:", min(widths), "~", max(widths))
    print("height range:", min(heights), "~", max(heights))

    small = [x for x in sizes if min(x[0], x[1]) < 512]
    print("short side below 512:", len(small))
PY
```

이 문서에서 사용한 데이터셋 결과:

```text
valid: 224
errors: 0
width range: 1024 ~ 1024
height range: 1024 ~ 1024
short side below 512: 0
```

---

# 6. RTX 3090용 설정 파일

다음 파일을 생성합니다.

```bash
cd /workspace/AnimaLoraToolkit-KR-Description
source .venv/bin/activate

cat > config/my_character.yaml <<'YAML'
# 모델
transformer_path: "models/transformers/anima.safetensors"
vae_path: "models/vae/qwen_image_vae.safetensors"
text_encoder_path: "models/text_encoders"
t5_tokenizer_path: "models/t5_tokenizer"

# 데이터
# 실제 데이터셋 경로로 변경
data_dir: "/workspace/anima-dataset"
resolution: 1024

# 224장 기준
repeats: 2

# TXT 캡션
prefer_json: false
shuffle_caption: true

# 첫 번째 태그가 트리거일 때 1
keep_tokens: 1

# 좌우 비대칭 특성이 있을 수 있으므로 기본 비활성
flip_augment: false

tag_dropout: 0.05
cache_latents: true

# LoRA
lora_type: "lora"
lora_rank: 32
lora_alpha: 32.0
lokr_factor: 8

# 학습
epochs: 10
max_steps: 0

batch_size: 1
grad_accum: 4
learning_rate: 1.0e-4

mixed_precision: "bf16"
grad_checkpoint: true
xformers: false
num_workers: 0

# 출력
output_dir: "/workspace/anima-output"
output_name: "my-character-anima"

save_every: 0
save_every_steps: 200
save_state_every: 1000

seed: 42

# 중간 샘플은 우선 비활성
sample_every: 0
sample_steps: 0

sample_infer_steps: 25
sample_cfg_scale: 4.0
sample_sampler_name: "er_sde"
sample_scheduler: "simple"
sample_width: 0
sample_height: 0
sample_seed: 42
sample_negative_prompt: ""

# YOUR_TRIGGER를 실제 트리거로 변경
sample_prompt: "newest, safe, 1girl, YOUR_TRIGGER, solo, portrait, looking at viewer"

# 진행 표시
loss_curve_steps: 100
no_progress: false
log_every: 10
YAML
```

트리거와 출력명을 변경합니다.

```bash
sed -i 's/YOUR_TRIGGER/my_trigger/g' config/my_character.yaml
sed -i 's/my-character-anima/my_trigger-anima/g' config/my_character.yaml
```

경로 검사:

```bash
python - <<'PY'
from pathlib import Path
import yaml

config_path = Path("config/my_character.yaml")
config = yaml.safe_load(config_path.read_text(encoding="utf-8"))

for key in [
    "transformer_path",
    "vae_path",
    "text_encoder_path",
    "t5_tokenizer_path",
    "data_dir",
]:
    path = Path(config[key])
    if not path.is_absolute():
        path = Path.cwd() / path

    print(f"{key:22} {'OK' if path.exists() else 'MISSING'}  {path}")
PY
```

모든 항목이 `OK`여야 합니다.

---

# 7. 예상 step 수

현재 예시 설정:

```text
224장 × repeats 2 × epochs 10 ÷ effective batch 4
= 약 1120 optimizer step
```

```text
effective batch = batch_size × grad_accum
```

현재 설정:

```text
1 × 4 = 4
```

`batch_size: 2`, `grad_accum: 2`로 바꿔도 effective batch는 동일하므로 총 step 수는 그대로입니다.

```text
2 × 2 = 4
```

---

# 8. Smoke test

본 학습 전에 1 epoch만 실행해 모델 로딩, latent cache, backward, 저장이 정상인지 확인합니다.

```bash
cp config/my_character.yaml config/smoke_test.yaml

python - <<'PY'
from pathlib import Path
import yaml

path = Path("config/smoke_test.yaml")
config = yaml.safe_load(path.read_text(encoding="utf-8"))

config["epochs"] = 1
config["repeats"] = 1
config["output_dir"] = "/workspace/anima-smoke-output"
config["output_name"] = "smoke-test"
config["save_every"] = 1
config["save_every_steps"] = 0
config["save_state_every"] = 0
config["sample_every"] = 0
config["sample_steps"] = 0

path.write_text(
    yaml.safe_dump(config, sort_keys=False, allow_unicode=True),
    encoding="utf-8",
)
PY
```

실행:

```bash
python anima_train.py \
  --config config/smoke_test.yaml \
  --no-monitor
```

정상적인 주요 로그:

```text
Transformer 로딩
VAE 로딩
텍스트 인코더 로딩
LoRA 주입
데이터셋 인식
latent cache 확인 또는 생성
학습 loop 진입
safetensors 저장
```

---

# 9. 본 학습

RunPod 터미널 연결이 끊겨도 유지하려면 `tmux`를 사용합니다.

```bash
apt-get update
apt-get install -y tmux

tmux new -s anima
```

`tmux` 안에서 실행:

```bash
cd /workspace/AnimaLoraToolkit-KR-Description
source .venv/bin/activate

mkdir -p /workspace/anima-output

python anima_train.py \
  --config config/my_character.yaml \
  --no-monitor
```

세션 분리:

```text
Ctrl+B
D
```

다시 접속:

```bash
tmux attach -t anima
```

---

# 10. 진행률이 보이지 않는 문제

다음처럼 `tee`를 사용하면 tqdm 진행 표시가 보이지 않거나 같은 줄 갱신이 깨질 수 있습니다.

```bash
python anima_train.py \
  --config config/my_character.yaml \
  --no-monitor \
  2>&1 | tee /workspace/anima-output/training.log
```

설정 파일은 다음이어야 합니다.

```yaml
no_progress: false
log_every: 10
```

진행률을 가장 확실하게 보려면 `tee` 없이 실행합니다.

```bash
python anima_train.py \
  --config config/my_character.yaml \
  --no-monitor
```

로그도 남기고 TTY 진행 표시도 유지하려면 `script`를 사용할 수 있습니다.

```bash
script -q -f \
  /workspace/anima-output/training.log \
  -c 'python anima_train.py --config config/my_character.yaml --no-monitor'
```

이미 실행 중이고 진행률이 보이지 않을 때는 별도 터미널에서 GPU 상태를 확인합니다.

```bash
watch -n 1 nvidia-smi
```

프로세스 상태:

```bash
ps -eo pid,stat,pcpu,pmem,etime,cmd | grep '[a]nima_train.py'
```

출력 파일 확인:

```bash
watch -n 10 'find /workspace/anima-output -maxdepth 1 -type f -printf "%TY-%Tm-%Td %TH:%TM:%TS %f\n" | sort'
```

GPU 사용률이 계속 90~100%이고 프로세스가 살아 있다면 학습 중일 가능성이 높습니다.

---

# 11. 설정 파일을 수정했는데 적용되지 않는 경우

YAML은 프로세스 시작 시 한 번만 읽습니다.

이미 실행 중인 상태에서 `config/my_character.yaml`을 수정해도 현재 프로세스에는 반영되지 않습니다.

실행 중인 명령 확인:

```bash
ps -ef | grep '[a]nima_train.py'
```

실제 설정 확인:

```bash
grep -E '^(batch_size|grad_accum|grad_checkpoint|repeats|epochs):' \
  config/my_character.yaml
```

변경값을 적용하려면 현재 학습을 안전하게 중단한 뒤 다시 실행해야 합니다.

---

# 12. VRAM과 batch 설정

RTX 3090에서 다음 설정은 안전한 출발점입니다.

```yaml
batch_size: 1
grad_accum: 4
grad_checkpoint: true
cache_latents: true
mixed_precision: "bf16"
```

VRAM이 남는다면 다음 구성을 시험할 수 있습니다.

```yaml
batch_size: 2
grad_accum: 2
```

effective batch는 동일합니다.

```text
기존: 1 × 4 = 4
변경: 2 × 2 = 4
```

총 optimizer step 수가 같아도 정상입니다.

배치를 바꿨을 때는 `it/s`만 비교하지 말고 **한 epoch가 실제로 끝나는 시간**을 비교해야 합니다.

더 많은 VRAM을 활용하고 싶다면 별도로 다음 설정도 시험할 수 있습니다.

```yaml
grad_checkpoint: false
```

다만 VRAM 사용량이 크게 증가할 수 있으므로 smoke test로 먼저 확인해야 합니다.

---

# 13. 예상 속도

RTX 3090, 1024×1024, rank 32, batch 1, grad accumulation 4, gradient checkpointing 활성 상태에서 실제로 약 다음 속도가 확인되었습니다.

```text
0.16 it/s
약 6.25초 / optimizer step
```

총 1120 step이라면 순수 학습 시간은 대략 2시간 전후입니다.

모델 로딩, 캐시, 저장 시간을 포함하면 조금 더 걸릴 수 있습니다.

---

# 14. 출력 파일

예시:

```text
/workspace/anima-output/
├── my_trigger-anima_step200.safetensors
├── my_trigger-anima_step400.safetensors
├── my_trigger-anima_step600.safetensors
├── my_trigger-anima_step800.safetensors
├── my_trigger-anima_step1000.safetensors
├── my_trigger-anima.safetensors
└── training_state_step1000.pt
```

- `*_stepN.safetensors`: 중간 LoRA
- 최종 `.safetensors`: 최종 LoRA
- `training_state_stepN.pt`: optimizer와 random state를 포함한 복구용 상태

최종 epoch가 항상 가장 좋은 결과는 아닙니다. 같은 seed와 프롬프트로 중간 체크포인트도 비교하는 것이 좋습니다.

---

# 15. 중단 및 재개

## LoRA 가중치만 이어서 학습

```yaml
resume_lora: "/workspace/anima-output/my_trigger-anima_step1000.safetensors"
```

optimizer 상태는 초기화됩니다.

## 전체 학습 상태 복구

```yaml
resume_state: "/workspace/anima-output/training_state_step1000.pt"
```

optimizer, random state, epoch, step, loss history를 복구합니다.

---

# 16. 자주 발생한 문제

## `pillow-jxlpy`를 찾을 수 없음

```bash
grep -v 'pillow-jxlpy' requirements.txt > requirements.runpod.txt
python -m pip install -r requirements.runpod.txt
```

## T5 tokenizer가 sentencepiece를 찾지 못함

```bash
python -m pip install sentencepiece tiktoken protobuf
```

## Qwen `special_tokens_map.json`이 없음

Qwen3-0.6B-Base에서는 없어도 정상일 수 있습니다. 실제 tokenizer 검증이 PASS라면 임의로 만들 필요가 없습니다.

## Anima 모델이 MISSING으로 나옴

실제 파일명과 YAML 경로를 맞춥니다.

```yaml
transformer_path: "models/transformers/anima.safetensors"
```

## GPU는 100%인데 진행 로그가 없음

`tqdm` 출력이 `tee`나 비-TTY 환경에서 보이지 않는 경우가 있습니다. GPU 사용률, 프로세스 상태, 중간 저장 파일을 확인합니다.

## VRAM을 11GB 정도만 사용함

다음 기능 때문에 정상일 수 있습니다.

- LoRA 파라미터만 학습
- BF16
- gradient checkpointing
- latent cache
- batch size 1

VRAM을 꽉 채우는 것이 목표가 아니라, GPU 연산기가 계속 사용되는지가 더 중요합니다.

## OOM

```yaml
batch_size: 1
grad_accum: 4
grad_checkpoint: true
cache_latents: true
mixed_precision: "bf16"
sample_every: 0
sample_steps: 0
```

그래도 부족하면:

```yaml
resolution: 768
```

---

# 원본 프로젝트

- 원본 저장소: [Moeblack/AnimaLoraToolkit](https://github.com/Moeblack/AnimaLoraToolkit)
- Anima 모델: [circlestone-labs/Anima](https://huggingface.co/circlestone-labs/Anima)
- Qwen3-0.6B-Base: [Qwen/Qwen3-0.6B-Base](https://huggingface.co/Qwen/Qwen3-0.6B-Base)
- T5 tokenizer: [google/t5-v1_1-xxl](https://huggingface.co/google/t5-v1_1-xxl)

코드의 라이선스와 저작권은 원본 저장소의 조건을 따릅니다.
