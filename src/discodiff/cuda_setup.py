"""CUDA diagnostics and env-gated performance toggles (Linux + NVIDIA; defaults unchanged)."""

from __future__ import annotations

import os
import subprocess
import sys


def _truthy(env_val: str) -> bool:
    return env_val.strip().lower() in ("1", "true", "yes", "on")


def warn_if_non_linux_platform() -> None:
    """Supported matrix is Linux only; warn on other OS unless DISCO_ALLOW_NON_LINUX is set."""
    if sys.platform == "linux":
        return
    if _truthy(os.environ.get("DISCO_ALLOW_NON_LINUX", "")):
        return
    print(
        "[discodiff] Supported platform is Linux only; running on "
        f"{sys.platform!r} is untested. Set DISCO_ALLOW_NON_LINUX=1 to hide this warning.",
        file=sys.stderr,
        flush=True,
    )


def log_cuda_device(device) -> None:
    import torch

    if device.type != "cuda":
        print("[CUDA] Using CPU (CUDA not selected or unavailable).", flush=True)
        return

    idx = device.index if device.index is not None else 0
    props = torch.cuda.get_device_properties(idx)
    print(
        f"[CUDA] device {idx}: {props.name}  compute {props.major}.{props.minor}  "
        f"VRAM {props.total_memory // (1024 ** 2)} MiB",
        flush=True,
    )
    cudnn_ver: int | str | None
    try:
        cudnn_ver = torch.backends.cudnn.version()
    except Exception:
        cudnn_ver = "n/a"
    print(
        f"[CUDA] torch {torch.__version__}  built with CUDA {torch.version.cuda}  cudnn={cudnn_ver}",
        flush=True,
    )
    if not torch.backends.cudnn.enabled:
        print("[CUDA] cuDNN is disabled (e.g. A100 workaround or manual); convs use fallback.", flush=True)

    _log_nvidia_smi_driver(torch.version.cuda)


def _log_nvidia_smi_driver(torch_cuda_build: str | None) -> None:
    try:
        r = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=driver_version",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            timeout=8,
        )
    except (FileNotFoundError, subprocess.TimeoutExpired, OSError):
        return
    if r.returncode != 0 or not (r.stdout or "").strip():
        return
    drv = (r.stdout.strip().splitlines() or [""])[0].strip()
    suffix = ""
    if torch_cuda_build:
        suffix = f" — ensure this driver supports PyTorch’s CUDA {torch_cuda_build} builds (see pytorch.org)"
    print(f"[CUDA] nvidia-smi driver: {drv}{suffix}", flush=True)


def apply_env_tf32(device) -> None:
    import torch

    if device.type != "cuda":
        return
    if not _truthy(os.environ.get("DISCO_ALLOW_TF32", "")):
        return
    major, _minor = torch.cuda.get_device_capability(device)
    if major < 8:
        print("[CUDA] DISCO_ALLOW_TF32 set but TF32 requires Ampere (8.x) or newer; skipping.", flush=True)
        return
    torch.backends.cuda.matmul.allow_tf32 = True
    if torch.backends.cudnn.enabled:
        torch.backends.cudnn.allow_tf32 = True
    print("[CUDA] DISCO_ALLOW_TF32=1: TF32 enabled for matmul/cuDNN (Ampere+).", flush=True)


def use_cudnn_benchmark_mode() -> bool:
    """Throughput-oriented cuDNN autotune; tiny non-determinism vs strict repro."""
    return _truthy(os.environ.get("DISCO_CUDNN_BENCHMARK", ""))


def format_cuda_oom_hint() -> str:
    return (
        "\n[CUDA OOM] Try: smaller --width/--height, fewer CLIP models in main.py, lower --cutn / "
        "--cutn-batches, export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True, or set USE_CPU = True "
        "near the top of src/discodiff/main.py.\n"
    )
