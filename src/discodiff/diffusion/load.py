"""Load primary UNet + diffusion schedule (checkpoint, device, fp16, trainable projector norms)."""

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
    """Create UNet + diffusion, load weights, apply fp16 and partial requires_grad (unchanged behavior)."""
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
