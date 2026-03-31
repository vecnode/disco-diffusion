"""Guided-diffusion progressive sampling entrypoints (DDIM / PLMS)."""

from __future__ import annotations

from typing import Any


def timestep_after_skip(diffusion: Any, skip_steps: int) -> int:
    return diffusion.num_timesteps - skip_steps - 1


def iter_clip_guided_samples(
    *,
    diffusion_sampling_mode: str,
    diffusion: Any,
    model: Any,
    batch_size: int,
    side_y: int,
    side_x: int,
    clip_denoised: bool,
    cond_fn: Any,
    skip_timesteps: int,
    init_image: Any,
    randomize_class: bool,
    eta: float,
    symmetry_transformation_fn: Any,
    transformation_percent: Any,
) -> Any:
    """Return the progressive sample iterator from guided-diffusion (same kwargs as ``main.do_run``)."""
    if diffusion_sampling_mode == "ddim":
        return diffusion.ddim_sample_loop_progressive(
            model,
            (batch_size, 3, side_y, side_x),
            clip_denoised=clip_denoised,
            model_kwargs={},
            cond_fn=cond_fn,
            progress=True,
            skip_timesteps=skip_timesteps,
            init_image=init_image,
            randomize_class=randomize_class,
            eta=eta,
            transformation_fn=symmetry_transformation_fn,
            transformation_percent=transformation_percent,
        )
    return diffusion.plms_sample_loop_progressive(
        model,
        (batch_size, 3, side_y, side_x),
        clip_denoised=clip_denoised,
        model_kwargs={},
        cond_fn=cond_fn,
        progress=True,
        skip_timesteps=skip_timesteps,
        init_image=init_image,
        randomize_class=randomize_class,
        order=2,
    )
