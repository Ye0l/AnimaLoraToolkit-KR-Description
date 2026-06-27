#!/usr/bin/env python3
"""Interactive, menu-based review/edit screen for the key training settings.

Shown right before training starts (unless --yes is passed or stdin is not a TTY).
The defaults are whatever is already in the merged config (YAML + CLI), so the
user only changes what they want. The menu is non-linear: pick any row by number,
edit it, come back, edit another, in any order, then start. The final, effective
values can be exported to a reusable YAML file.
"""

from __future__ import annotations

import sys
from datetime import datetime
from pathlib import Path

# ----------------------------------------------------------------------------
# Field catalogue. Each field maps a YAML key <-> argparse attribute and carries
# a one-line Korean explanation shown in the menu.
# kind: "int" | "float" | "bool" | "str" | one of a tuple of choices
# ----------------------------------------------------------------------------

FIELD_GROUPS = [
    ("학습 (Training)", [
        ("learning_rate", "lr", "float", "학습률. 작은 데이터셋(수십 장)엔 1e-4가 높을 수 있음. 구도가 너무 빨리 굳으면 3e-5~5e-5로 낮춰보세요."),
        ("epochs", "epochs", "int", "전체 데이터셋을 몇 바퀴 돌지. 총 스텝 = 이미지수×repeats×epochs / (batch_size×grad_accum)."),
        ("batch_size", "batch_size", "int", "한 번에 처리할 이미지 수. VRAM에 직접 영향. 16GB(4080S)면 1~2 권장."),
        ("grad_accum", "grad_accum", "int", "그래디언트 누적 횟수. 유효 배치 = batch_size×grad_accum (VRAM은 거의 안 늘고 안정성↑)."),
        ("resolution", "resolution", "int", "학습 해상도(64의 배수). 1024 기본, OOM이면 768로."),
        ("repeats", "repeats", "int", "같은 이미지를 한 epoch에서 몇 번 노출할지. 새 데이터가 아니라 반복. 헷갈리면 1로 두고 epochs로 조절."),
        ("seed", "seed", "int", "랜덤 시드. 재현성용."),
    ]),
    ("LoRA", [
        ("lora_type", "lora_type", ("lora", "lokr"), "어댑터 종류. lora=일반, lokr=LyCORIS LoKr(더 작고 표현 다름)."),
        ("lora_rank", "lora_rank", "int", "랭크(용량). 32면 캐릭터+화풍에 충분한 편. 더 올린다고 항상 좋진 않음."),
        ("lora_alpha", "lora_alpha", "float", "스케일. 이 구현에서 LoRA 출력 배율 = alpha/rank. rank와 같게 두면 배율 1.0."),
        ("lokr_factor", "lokr_factor", "int", "LoKr 분해 factor. lora_type=lokr일 때만 의미 있음."),
    ]),
    ("캡션 / 증강 (Caption)", [
        ("shuffle_caption", "shuffle_caption", "bool", "태그 순서 섞기. Danbooru식 태그엔 ON, 자연어 문장 캡션엔 OFF."),
        ("keep_tokens", "keep_tokens", "int", "앞쪽 N개 태그는 섞지 않고 고정(보통 트리거 워드). 첫 태그가 트리거면 1."),
        ("tag_dropout", "tag_dropout", "float", "태그를 확률적으로 일부 드롭(0~1). 과적합/구도고착 완화. 이제 TXT 캡션에도 적용됩니다."),
        ("flip_augment", "flip_augment", "bool", "좌우반전 증강. 주의: cache_latents와 같이 쓰면 캐시 시점에 한 번만 결정됨."),
        ("cache_latents", "cache_latents", "bool", "VAE latent를 미리 캐싱해 속도↑/VRAM↓. 해상도·전처리 바꾸면 .npz 캐시를 지워야 함."),
    ]),
    ("고급 (Advanced)", [
        ("timestep_shift", "timestep_shift", "float", "학습 노이즈 분포 편향. 3.0=구조/구도 위주(기본), 1.0=치우침 없음(세부 디테일 비중↑)."),
        ("train_llm_adapter", "train_llm_adapter", "bool", "llm_adapter에도 LoRA를 넣을지. OFF로 두면 '트리거 넣으면 구도 통째로 딸려옴' 증상 완화에 도움될 수 있음."),
        ("grad_checkpoint", "grad_checkpoint", "bool", "그래디언트 체크포인트. VRAM 크게 절약, 속도 약간↓. 16GB면 ON 권장."),
        ("mixed_precision", "mixed_precision", ("bf16", "fp32"), "혼합정밀도. 최신 GPU는 bf16 권장."),
    ]),
    ("저장 / 샘플 (Save & Sample)", [
        ("save_every_steps", "save_every_steps", "int", "N스텝마다 LoRA 저장(0=epoch 기준만). 중간 결과 비교하려면 100~200."),
        ("save_state_every", "save_state_every", "int", "N스텝마다 전체 학습상태 저장(.pt, 이어학습/재시작용). 0=비활성."),
        ("save_every", "save_every", "int", "N epoch마다 저장(0=마지막에만)."),
        ("sample_steps", "sample_steps", "int", "N스텝마다 샘플 이미지 생성(0=비활성)."),
        ("sample_every", "sample_every", "int", "N epoch마다 샘플 이미지 생성(0=비활성)."),
        ("sample_prompt", "sample_prompt", "str", "샘플 생성 프롬프트. 트리거 워드 포함 권장."),
    ]),
    ("출력 (Output)", [
        ("output_name", "output_name", "str", "결과 파일 이름 접두사."),
    ]),
]

# Full key->attr map used for export (mirror of apply_yaml_config), so the
# exported YAML is a complete, reusable config — not just the edited rows.
EXPORT_MAP = {
    "transformer_path": "transformer", "vae_path": "vae",
    "text_encoder_path": "qwen", "t5_tokenizer_path": "t5_tokenizer",
    "data_dir": "data_dir", "resolution": "resolution", "repeats": "repeats",
    "shuffle_caption": "shuffle_caption", "keep_tokens": "keep_tokens",
    "flip_augment": "flip_augment", "tag_dropout": "tag_dropout",
    "prefer_json": "prefer_json", "cache_latents": "cache_latents",
    "lora_type": "lora_type", "lora_rank": "lora_rank", "lora_alpha": "lora_alpha",
    "lokr_factor": "lokr_factor", "resume_lora": "resume_lora",
    "train_llm_adapter": "train_llm_adapter", "timestep_shift": "timestep_shift",
    "epochs": "epochs", "max_steps": "max_steps", "batch_size": "batch_size",
    "grad_accum": "grad_accum", "learning_rate": "lr",
    "mixed_precision": "mixed_precision", "grad_checkpoint": "grad_checkpoint",
    "xformers": "xformers", "num_workers": "num_workers",
    "output_dir": "output_dir", "output_name": "output_name",
    "save_every": "save_every", "save_every_steps": "save_every_steps",
    "save_state_every": "save_state_every", "resume_state": "resume_state",
    "seed": "seed", "sample_every": "sample_every", "sample_steps": "sample_steps",
    "sample_prompt": "sample_prompt", "sample_cfg_scale": "sample_cfg_scale",
    "sample_negative_prompt": "sample_negative_prompt",
    "sample_width": "sample_width", "sample_height": "sample_height",
    "sample_seed": "sample_seed", "sample_infer_steps": "sample_infer_steps",
    "sample_sampler_name": "sample_sampler_name", "sample_scheduler": "sample_scheduler",
    "loss_curve_steps": "loss_curve_steps", "log_every": "log_every",
}


def _flat_fields():
    rows = []
    for _group, fields in FIELD_GROUPS:
        for f in fields:
            rows.append(f)
    return rows


def _coerce(kind, raw, current):
    raw = raw.strip()
    if raw == "":
        return current, None
    if kind == "int":
        try:
            return int(raw), None
        except ValueError:
            return current, "정수를 입력하세요."
    if kind == "float":
        try:
            return float(raw), None
        except ValueError:
            return current, "숫자를 입력하세요."
    if kind == "bool":
        low = raw.lower()
        if low in ("y", "yes", "true", "1", "on", "t"):
            return True, None
        if low in ("n", "no", "false", "0", "off", "f"):
            return False, None
        return current, "y/n 으로 입력하세요."
    if isinstance(kind, tuple):
        if raw in kind:
            return raw, None
        return current, f"{', '.join(kind)} 중 하나를 입력하세요."
    return raw, None  # str


def export_config(args, path: Path) -> Path:
    import yaml
    data = {}
    for ykey, attr in EXPORT_MAP.items():
        if hasattr(args, attr):
            val = getattr(args, attr)
            if val is None:
                continue
            data[ykey] = val
    # sample_prompts list is handled separately if present
    if getattr(args, "sample_prompts", None):
        data["sample_prompts"] = args.sample_prompts
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        yaml.safe_dump(data, f, allow_unicode=True, sort_keys=False)
    return path


# ----------------------------------------------------------------------------
# Rendering
# ----------------------------------------------------------------------------

def _fmt(val):
    if isinstance(val, bool):
        return "ON" if val else "OFF"
    return str(val)


def _render_rich(args):
    from rich.console import Console
    from rich.table import Table
    console = Console()
    console.clear()
    console.rule("[bold cyan]학습 설정 확인/수정[/bold cyan]")
    idx = 1
    index_map = {}
    for group, fields in FIELD_GROUPS:
        table = Table(show_header=True, header_style="bold", expand=True, padding=(0, 1))
        table.add_column("#", width=3, justify="right", style="dim")
        table.add_column("설정", width=20, style="cyan")
        table.add_column("값", width=16, style="bold green")
        table.add_column("설명", overflow="fold")
        for ykey, attr, kind, helptext in fields:
            cur = getattr(args, attr, None)
            table.add_row(str(idx), ykey, _fmt(cur), helptext)
            index_map[idx] = (ykey, attr, kind, helptext)
            idx += 1
        console.print(f"\n[bold yellow]{group}[/bold yellow]")
        console.print(table)
    console.print(
        "\n[bold]번호[/bold]=수정  ·  [bold]s[/bold] 또는 Enter=학습 시작  ·  "
        "[bold]e[/bold]=설정 export  ·  [bold]q[/bold]=취소"
    )
    return index_map


def _render_plain(args):
    print("\n==== 학습 설정 확인/수정 ====")
    idx = 1
    index_map = {}
    for group, fields in FIELD_GROUPS:
        print(f"\n[{group}]")
        for ykey, attr, kind, helptext in fields:
            cur = getattr(args, attr, None)
            print(f"  {idx:>2}. {ykey:<20} = {_fmt(cur):<16} | {helptext}")
            index_map[idx] = (ykey, attr, kind, helptext)
            idx += 1
    print("\n번호=수정 · s 또는 Enter=학습 시작 · e=설정 export · q=취소")
    return index_map


def run_config_tui(args):
    """Show the menu, mutate `args` in place, return (args, started: bool).

    Returns started=False if the user chose to cancel.
    """
    use_rich = True
    try:
        import rich  # noqa: F401
    except Exception:
        use_rich = False

    while True:
        index_map = _render_rich(args) if use_rich else _render_plain(args)
        try:
            choice = input("\n> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n취소되었습니다.")
            return args, False

        low = choice.lower()
        if low in ("", "s", "start"):
            return args, True
        if low in ("q", "quit", "exit"):
            print("취소되었습니다.")
            return args, False
        if low in ("e", "export"):
            _do_export(args)
            input("\n계속하려면 Enter...")
            continue

        if not choice.isdigit() or int(choice) not in index_map:
            print("올바른 번호 또는 명령(s/e/q)을 입력하세요.")
            input("계속하려면 Enter...")
            continue

        ykey, attr, kind, helptext = index_map[int(choice)]
        cur = getattr(args, attr, None)
        print(f"\n[{ykey}] {helptext}")
        if isinstance(kind, tuple):
            prompt = f"선택지 {{{', '.join(kind)}}} (현재: {cur}, 빈칸=유지) > "
        elif kind == "bool":
            prompt = f"y/n (현재: {_fmt(cur)}, 빈칸=유지) > "
        else:
            prompt = f"새 값 (현재: {cur}, 빈칸=유지) > "
        try:
            raw = input(prompt)
        except (EOFError, KeyboardInterrupt):
            continue
        new_val, err = _coerce(kind, raw, cur)
        if err:
            print(f"  ! {err} 변경하지 않았습니다.")
            input("계속하려면 Enter...")
            continue
        setattr(args, attr, new_val)


def _do_export(args):
    out_dir = Path(getattr(args, "output_dir", ".") or ".")
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    path = out_dir / f"applied_config_{stamp}.yaml"
    try:
        export_config(args, path)
        print(f"설정을 저장했습니다: {path}")
        print(f"다음에 그대로 쓰려면: TRAIN_CONFIG={path} train-runpod -y")
    except Exception as exc:
        print(f"export 실패: {exc}")


def maybe_run_tui(args):
    """Entry point used by anima_train.py. Honours --yes and non-TTY.

    Always writes an applied_config snapshot when training proceeds, so every
    run is reproducible.
    """
    interactive = (not getattr(args, "yes", False)) and sys.stdin.isatty() and sys.stdout.isatty()
    started = True
    if interactive:
        args, started = run_config_tui(args)
        if not started:
            sys.exit(0)
    # Always snapshot the effective config (both -y and interactive paths).
    try:
        out_dir = Path(getattr(args, "output_dir", ".") or ".")
        stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        snap = out_dir / f"applied_config_{stamp}.yaml"
        export_config(args, snap)
        print(f"[config] 적용된 설정 저장: {snap}")
    except Exception as exc:
        print(f"[config] 설정 스냅샷 저장 실패(무시): {exc}")
    return args
