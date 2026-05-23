"""Legacy CUDA helper wrappers.

Prefer src/platform/device.py for new device and platform logic.
"""

from __future__ import annotations

import os
import sys

from .device import log_device_selection, resolve_runtime_device, warn_if_unsupported_platform


def _truthy(env_val: str) -> bool:
    return env_val.strip().lower() in ("1", "true", "yes", "on")


def warn_if_non_linux_platform() -> None:
    """Backward-compatible alias for platform warning."""
    warn_if_unsupported_platform()


def log_cuda_device(device) -> None:
    """Backward-compatible logger for an explicitly selected torch device."""
    import torch

    if device.type != "cuda":
        print("[CUDA] Using CPU (CUDA not selected or unavailable).", flush=True)
        return
    idx = device.index if device.index is not None else 0
    selection = resolve_runtime_device(f"cuda:{idx}")
    log_device_selection(selection)


def apply_env_tf32(device) -> None:
    """Backward-compatible TF32 env toggle."""
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
        "near the top of src/run.py.\n"
    )
