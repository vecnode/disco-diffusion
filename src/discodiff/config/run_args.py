"""Assemble ``args`` SimpleNamespace from centralized :class:`RunConfig`."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

from .keyframes import split_prompts
from .settings import RunConfig


_DIRECT_KEYS: tuple[str, ...] = (
    "batchNum",
    "batch_name",
    "steps",
    "init_image",
    "init_scale",
    "skip_steps",
    "side_x",
    "side_y",
    "key_frames",
    "start_frame",
    "near_plane",
    "far_plane",
    "fov",
    "padding_mode",
    "sampling_mode",
    "angle_series",
    "zoom_series",
    "translation_x_series",
    "translation_y_series",
    "translation_z_series",
    "rotation_3d_x_series",
    "rotation_3d_y_series",
    "rotation_3d_z_series",
    "calc_frames_skip_steps",
    "text_prompts",
)


def build_run_args_namespace(cfg: RunConfig) -> SimpleNamespace:
    """Mirror legacy ``args = { ... }; SimpleNamespace(**args)`` from RunConfig runtime state."""
    ns = cfg.runtime_values
    args: dict[str, Any] = {
        key: ns[key] for key in _DIRECT_KEYS
    }
    args.update(
        {
            "prompts_series": split_prompts(ns["text_prompts"], ns["max_frames"])
            if ns["text_prompts"]
            else None,
            "seed": ns["seed"],
            "n_batches": 1,
            "max_frames": ns["max_frames"],
        }
    )

    return SimpleNamespace(**args)
