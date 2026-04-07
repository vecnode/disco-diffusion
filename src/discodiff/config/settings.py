"""Centralized run settings and environment-variable overrides.

This is a small first step for issue #2: one typed object for top-level runtime
config, loaded once before the diffusion run starts.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, Mapping, Optional

_ALLOWED_GENERATION_MODES = {"None", "2D", "3D", "Video Input"}


@dataclass(frozen=True)
class RunConfig:
    """Top-level configuration for a diffusion run."""

    output_dir: Path
    device: str = "auto"
    seed: Optional[int] = None
    generation_mode: str = "None"
    profile: str = "default"
    runtime_values: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_env(cls, root_path: str | Path) -> "RunConfig":
        """Build config from environment variables with safe defaults."""
        root = Path(root_path)
        output_dir = Path(os.environ.get("DISCO_OUTPUT_DIR", root / "output"))
        device = os.environ.get("DISCO_DEVICE", "auto").strip() or "auto"
        profile = os.environ.get("DISCO_PROFILE", "default").strip() or "default"

        generation_mode = os.environ.get("DISCO_GENERATION_MODE", "None").strip() or "None"
        if generation_mode not in _ALLOWED_GENERATION_MODES:
            generation_mode = "None"

        seed_raw = os.environ.get("DISCO_SEED", "").strip()
        seed: Optional[int] = None
        if seed_raw:
            try:
                seed = int(seed_raw)
            except ValueError:
                seed = None

        return cls(
            output_dir=output_dir,
            device=device,
            seed=seed,
            generation_mode=generation_mode,
            profile=profile,
        )

    def with_runtime_values(self, values: Mapping[str, Any]) -> "RunConfig":
        """Attach a snapshot of runtime values used for legacy args building."""
        return replace(self, runtime_values=dict(values))


def apply_runtime_overrides(
    cfg: RunConfig,
    *,
    device: str | None = None,
    seed: int | None = None,
    generation_mode: str | None = None,
) -> RunConfig:
    """Return an updated config with runtime-resolved fields."""
    return replace(
        cfg,
        device=cfg.device if device is None else device,
        seed=cfg.seed if seed is None else seed,
        generation_mode=cfg.generation_mode if generation_mode is None else generation_mode,
    )
