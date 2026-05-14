"""Assemble ``args`` SimpleNamespace from centralized :class:`RunConfig`."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

from .keyframes import split_prompts
from .settings import RunConfig


_DIRECT_KEYS: tuple[str, ...] = (
    "batchNum",
    "display_rate",
    "batch_size",
    "batch_name",
    "steps",
    "diffusion_sampling_mode",
    "width_height",
    "clip_guidance_scale",
    "tv_scale",
    "range_scale",
    "sat_scale",
    "cutn_batches",
    "init_image",
    "init_scale",
    "skip_steps",
    "side_x",
    "side_y",
    "timestep_respacing",
    "diffusion_steps",
    "key_frames",
    "interp_spline",
    "start_frame",
    "angle",
    "zoom",
    "translation_x",
    "translation_y",
    "translation_z",
    "rotation_3d_x",
    "rotation_3d_y",
    "rotation_3d_z",
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
    "frames_scale",
    "skip_step_ratio",
    "calc_frames_skip_steps",
    "text_prompts",
    "image_prompts",
    "intermediate_saves",
    "intermediates_in_subfolder",
    "steps_per_checkpoint",
    "perlin_init",
    "perlin_mode",
    "set_seed",
    "eta",
    "clamp_grad",
    "clamp_max",
    "skip_augs",
    "randomize_class",
    "clip_denoised",
    "fuzzy_prompt",
    "rand_mag",
    "turbo_mode",
    "turbo_steps",
    "turbo_preroll",
    "use_vertical_symmetry",
    "use_horizontal_symmetry",
    "transformation_percent",
)


def build_run_args_namespace(cfg: RunConfig) -> SimpleNamespace:
    """Mirror legacy ``args = { ... }; SimpleNamespace(**args)`` from RunConfig runtime state."""
    ns = cfg.runtime_values
    animation_mode = str(cfg.generation_mode)
    args: dict[str, Any] = {
        key: ns[key] for key in _DIRECT_KEYS
    }
    args.update(
        {
            "prompts_series": split_prompts(ns["text_prompts"], ns["max_frames"])
            if ns["text_prompts"]
            else None,
            "image_prompts_series": split_prompts(ns["image_prompts"], ns["max_frames"])
            if ns["image_prompts"]
            else None,
            "seed": ns["seed"],
            "n_batches": ns["n_batches"] if animation_mode == "None" else 1,
            "animation_mode": animation_mode,
            "max_frames": ns["max_frames"] if animation_mode != "None" else 1,
            "cut_overview": eval(ns["cut_overview"]),
            "cut_innercut": eval(ns["cut_innercut"]),
            "cut_ic_pow": eval(ns["cut_ic_pow"]),
            "cut_icgray_p": eval(ns["cut_icgray_p"]),
        }
    )

    return SimpleNamespace(**args)
