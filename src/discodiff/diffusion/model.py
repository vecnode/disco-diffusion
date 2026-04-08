"""Load primary UNet + diffusion, and progressive sampling entrypoints (DDIM / PLMS)."""

from __future__ import annotations

from typing import Any, Callable, Tuple

import gc
import torch


def load_primary_diffusion_model(
    *,
    model_config: dict[str, Any],
    diffusion_model: str,
    custom_path: str,
    model_path: str,
    diff_model_map: dict[str, Any],
    device: torch.device,
    create_model_and_diffusion: Callable[..., Tuple[Any, Any]],
    get_model_filename: Callable[[str, dict[str, Any]], str],
) -> tuple[Any, Any]:
    """Create UNet + diffusion, load weights, apply fp16 and partial requires_grad."""
    model, diffusion = create_model_and_diffusion(**model_config)
    if diffusion_model == "custom":
        model.load_state_dict(torch.load(custom_path, map_location="cpu"))
    else:
        model.load_state_dict(
            torch.load(
                f"{model_path}/{get_model_filename(diffusion_model, diff_model_map)}",
                map_location="cpu",
            )
        )
    model.requires_grad_(False).eval().to(device)
    for name, param in model.named_parameters():
        if "qkv" in name or "norm" in name or "proj" in name:
            param.requires_grad_()
    if model_config["use_fp16"]:
        model.convert_to_fp16()

    gc.collect()
    torch.cuda.empty_cache()
    return model, diffusion


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
    """Return the progressive sample iterator from guided-diffusion."""
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
