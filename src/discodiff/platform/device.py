"""OS-aware device selection and backend defaults for diffusion runtime."""

from __future__ import annotations

import os
import platform
import subprocess
import sys
from dataclasses import dataclass


def _truthy(env_val: str) -> bool:
    return env_val.strip().lower() in ("1", "true", "yes", "on")


def warn_if_unsupported_platform() -> None:
    """Warn for non-Linux platforms unless explicitly allowed."""
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


@dataclass(frozen=True)
class DeviceSelection:
    requested: str
    torch_device: str
    accelerator: str
    os_name: str
    rtx_capable: bool
    gpu_name: str | None
    warnings: tuple[str, ...]


def _is_rtx_name(name: str) -> bool:
    lower = name.lower()
    return "rtx" in lower


def _first_rtx_index(torch) -> int | None:
    if not torch.cuda.is_available():
        return None
    for idx in range(torch.cuda.device_count()):
        if _is_rtx_name(torch.cuda.get_device_name(idx)):
            return idx
    return None


def _parse_cuda_index(requested: str) -> int:
    if ":" not in requested:
        return 0
    try:
        return int(requested.split(":", 1)[1])
    except ValueError:
        return 0


def resolve_runtime_device(requested: str, *, use_cpu: bool = False) -> DeviceSelection:
    """Resolve a runtime device with consistent cross-OS behavior.

    Supported requests: auto, rtx, cpu, mps, cuda, cuda:N.
    """
    import torch

    req = (requested or "auto").strip().lower()
    os_name = platform.system().lower()
    warnings: list[str] = []

    cuda_available = torch.cuda.is_available()
    mps_available = bool(getattr(torch.backends, "mps", None) and torch.backends.mps.is_available())

    torch_device = "cpu"
    accelerator = "cpu"

    if use_cpu:
        req = "cpu"

    if req in ("", "auto"):
        if cuda_available:
            rtx_idx = _first_rtx_index(torch)
            idx = 0 if rtx_idx is None else rtx_idx
            torch_device = f"cuda:{idx}"
            accelerator = "cuda"
        elif mps_available:
            torch_device = "mps"
            accelerator = "mps"
    elif req == "rtx":
        if cuda_available:
            rtx_idx = _first_rtx_index(torch)
            if rtx_idx is None:
                warnings.append("Requested RTX device but no RTX-named GPU was found; using cuda:0.")
                torch_device = "cuda:0"
            else:
                torch_device = f"cuda:{rtx_idx}"
            accelerator = "cuda"
        else:
            warnings.append("Requested RTX device but CUDA is unavailable; using CPU.")
    elif req.startswith("cuda"):
        if cuda_available:
            idx = _parse_cuda_index(req)
            if idx < 0 or idx >= torch.cuda.device_count():
                warnings.append(f"Requested {req} is out of range; using cuda:0.")
                idx = 0
            torch_device = f"cuda:{idx}"
            accelerator = "cuda"
        else:
            warnings.append(f"Requested {req} but CUDA is unavailable; using CPU.")
    elif req == "mps":
        if mps_available:
            torch_device = "mps"
            accelerator = "mps"
        else:
            warnings.append("Requested mps but Apple MPS is unavailable; using CPU.")
    elif req == "cpu":
        torch_device = "cpu"
        accelerator = "cpu"
    else:
        warnings.append(f"Unknown device request {requested!r}; using auto.")
        if cuda_available:
            torch_device = "cuda:0"
            accelerator = "cuda"
        elif mps_available:
            torch_device = "mps"
            accelerator = "mps"

    gpu_name: str | None = None
    rtx_capable = False
    if accelerator == "cuda":
        idx = _parse_cuda_index(torch_device)
        gpu_name = torch.cuda.get_device_name(idx)
        rtx_capable = _is_rtx_name(gpu_name)

    return DeviceSelection(
        requested=requested,
        torch_device=torch_device,
        accelerator=accelerator,
        os_name=os_name,
        rtx_capable=rtx_capable,
        gpu_name=gpu_name,
        warnings=tuple(warnings),
    )


def apply_backend_defaults(selection: DeviceSelection, *, profile: str = "default") -> None:
    """Apply safe backend defaults by OS/accelerator and optional profile."""
    import torch

    profile_l = (profile or "default").strip().lower()
    rtx_profile = profile_l in ("rtx", "rtx-safe", "rtx-fast")

    if selection.accelerator == "cuda":
        os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

        cudnn_benchmark = rtx_profile
        if "DISCO_CUDNN_BENCHMARK" in os.environ:
            cudnn_benchmark = _truthy(os.environ["DISCO_CUDNN_BENCHMARK"])
        os.environ["DISCO_CUDNN_BENCHMARK"] = "1" if cudnn_benchmark else "0"

        allow_tf32 = rtx_profile and selection.rtx_capable
        if "DISCO_ALLOW_TF32" in os.environ:
            allow_tf32 = _truthy(os.environ["DISCO_ALLOW_TF32"])

        if allow_tf32:
            idx = _parse_cuda_index(selection.torch_device)
            major, _minor = torch.cuda.get_device_capability(idx)
            if major >= 8:
                torch.backends.cuda.matmul.allow_tf32 = True
                if torch.backends.cudnn.enabled:
                    torch.backends.cudnn.allow_tf32 = True

    if selection.accelerator == "mps":
        os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")


def log_device_selection(selection: DeviceSelection) -> None:
    """Emit a concise diagnostic summary for selected runtime device."""
    import torch

    for msg in selection.warnings:
        print(f"[device] {msg}", flush=True)

    print(
        f"[device] requested={selection.requested!r} selected={selection.torch_device} "
        f"accelerator={selection.accelerator} os={selection.os_name} rtx={selection.rtx_capable}",
        flush=True,
    )

    if selection.accelerator != "cuda":
        return

    idx = _parse_cuda_index(selection.torch_device)
    props = torch.cuda.get_device_properties(idx)
    print(
        f"[CUDA] device {idx}: {props.name}  compute {props.major}.{props.minor}  "
        f"VRAM {props.total_memory // (1024 ** 2)} MiB",
        flush=True,
    )
    try:
        cudnn_ver = torch.backends.cudnn.version()
    except Exception:
        cudnn_ver = "n/a"
    print(
        f"[CUDA] torch {torch.__version__}  built with CUDA {torch.version.cuda}  cudnn={cudnn_ver}",
        flush=True,
    )

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
    if torch.version.cuda:
        suffix = f" - ensure this driver supports PyTorch CUDA {torch.version.cuda} builds"
    print(f"[CUDA] nvidia-smi driver: {drv}{suffix}", flush=True)
