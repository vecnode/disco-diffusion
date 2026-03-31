"""OS / GPU helpers (Linux notice, CUDA logging, optional TF32 and cuDNN benchmark)."""

from .cuda import (
    apply_env_tf32,
    format_cuda_oom_hint,
    log_cuda_device,
    use_cudnn_benchmark_mode,
    warn_if_non_linux_platform,
)

__all__ = [
    "apply_env_tf32",
    "format_cuda_oom_hint",
    "log_cuda_device",
    "use_cudnn_benchmark_mode",
    "warn_if_non_linux_platform",
]
