"""OS / GPU helpers and runtime device abstraction."""

from .cuda import (
    format_cuda_oom_hint,
    use_cudnn_benchmark_mode,
)
from .device import (
    DeviceSelection,
    apply_backend_defaults,
    log_device_selection,
    resolve_runtime_device,
    warn_if_unsupported_platform,
)

__all__ = [
    "DeviceSelection",
    "apply_backend_defaults",
    "format_cuda_oom_hint",
    "log_device_selection",
    "resolve_runtime_device",
    "use_cudnn_benchmark_mode",
    "warn_if_unsupported_platform",
]
