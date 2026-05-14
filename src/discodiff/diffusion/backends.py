"""Backend-agnostic diffusion interfaces for pixel and latent generation."""

from __future__ import annotations

import os
import re
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image

@dataclass
class BackendPromptState:
    """Normalized prompt state consumed by backend generate()."""

    prompt_text: str
    seed: int | None
    size: tuple[int, int]


class DiffusionBackend(ABC):
    """Common backend contract used by the animation loop."""

    @abstractmethod
    def prepare(self, prompt: list[str], seed: int | None, size: tuple[int, int]) -> BackendPromptState:
        raise NotImplementedError

    @abstractmethod
    def generate(
        self,
        *,
        init_image: Any,
        strength_or_skip: float,
        steps: int,
        guidance_scale: float,
        extra_guidance_state: dict[str, Any],
    ) -> Any:
        raise NotImplementedError


class LatentDiffusionBackend(DiffusionBackend):
    """Diffusers img2img backend used by 3D_latent mode."""

    def __init__(
        self,
        *,
        device: torch.device,
        model_id: str | None = None,
        models_root: str | None = None,
    ) -> None:
        self.device = device
        self.model_id = model_id or os.environ.get("DISCO_LATENT_MODEL", "runwayml/stable-diffusion-v1-5")
        self.models_root = Path(models_root or os.path.join(os.getcwd(), "models"))
        self.local_model_dir = Path(
            os.environ.get("DISCO_LATENT_MODEL_DIR", "").strip()
            or self.models_root / "latent" / self._sanitize_repo_id(self.model_id)
        )
        self._pipe = None
        self._txt2img_pipe = None

    @staticmethod
    def _sanitize_repo_id(repo_id: str) -> str:
        return re.sub(r"[^a-zA-Z0-9._-]+", "_", repo_id)

    def _download_to_local_dir(self) -> None:
        self.local_model_dir.mkdir(parents=True, exist_ok=True)
        print(f"[3D_latent] Downloading model '{self.model_id}' to {self.local_model_dir}", flush=True)

        try:
            from huggingface_hub import snapshot_download

            snapshot_download(repo_id=self.model_id, local_dir=str(self.local_model_dir), local_dir_use_symlinks=False)
        except Exception as exc:
            raise RuntimeError(
                f"Failed to download latent model '{self.model_id}' into {self.local_model_dir}: {exc}"
            ) from exc

    def _ensure_pipe(self) -> None:
        if self._pipe is not None:
            return
        try:
            from diffusers import StableDiffusionImg2ImgPipeline
        except Exception as exc:  # pragma: no cover - import behavior depends on environment
            raise RuntimeError(
                "3D_latent requires the diffusers stack. Install with: uv add diffusers transformers accelerate"
            ) from exc

        dtype = torch.float16 if self.device.type == "cuda" else torch.float32

        if not (self.local_model_dir / "model_index.json").exists():
            self._download_to_local_dir()

        pipe = StableDiffusionImg2ImgPipeline.from_pretrained(str(self.local_model_dir), torch_dtype=dtype)
        pipe = pipe.to(self.device)
        pipe.set_progress_bar_config(disable=True)
        self._pipe = pipe

    def _ensure_txt2img_pipe(self) -> None:
        if self._txt2img_pipe is not None:
            return
        try:
            from diffusers import StableDiffusionPipeline
        except Exception as exc:  # pragma: no cover - import behavior depends on environment
            raise RuntimeError(
                "3D_latent first-frame txt2img requires StableDiffusionPipeline in diffusers"
            ) from exc

        dtype = torch.float16 if self.device.type == "cuda" else torch.float32

        if not (self.local_model_dir / "model_index.json").exists():
            self._download_to_local_dir()

        pipe = StableDiffusionPipeline.from_pretrained(str(self.local_model_dir), torch_dtype=dtype)
        pipe = pipe.to(self.device)
        pipe.set_progress_bar_config(disable=True)
        self._txt2img_pipe = pipe

    def prepare(self, prompt: list[str], seed: int | None, size: tuple[int, int]) -> BackendPromptState:
        prompt_text = ", ".join([p.split(":", 1)[0] for p in prompt]) if prompt else ""
        return BackendPromptState(prompt_text=prompt_text, seed=seed, size=size)

    def generate(
        self,
        *,
        init_image: Any,
        strength_or_skip: float,
        steps: int,
        guidance_scale: float,
        extra_guidance_state: dict[str, Any],
    ) -> Image.Image:
        self._ensure_pipe()

        if not isinstance(init_image, Image.Image):
            raise RuntimeError("3D_latent expects an init PIL image for img2img generation")

        prompt_state: BackendPromptState = extra_guidance_state["prompt_state"]
        if not prompt_state.prompt_text:
            raise RuntimeError("3D_latent requires at least one text prompt")

        init_color_reset = float(extra_guidance_state.get("latent_color_reset", 0.0) or 0.0)
        init_color_reset = max(0.0, min(0.35, init_color_reset))
        if init_color_reset > 0:
            image_array = np.array(init_image.resize(prompt_state.size, Image.LANCZOS)).astype("float32") / 255.0
            if init_color_reset > 0:
                luminance = np.dot(image_array[..., :3], np.array([0.299, 0.587, 0.114], dtype="float32"))
                neutral = np.repeat(luminance[..., None], 3, axis=2)
                image_array = image_array * (1.0 - init_color_reset) + neutral * init_color_reset
            init_image = Image.fromarray((np.clip(image_array, 0.0, 1.0) * 255.0).astype("uint8"))

        strength = float(strength_or_skip)
        if strength > 1.0:
            strength = strength / max(float(steps), 1.0)
        strength = max(0.05, min(0.95, strength))

        # CLIP guidance scales in this repo are often large (hundreds/thousands).
        cfg_scale = float(guidance_scale)
        if cfg_scale > 30.0:
            cfg_scale = cfg_scale / 100.0
        cfg_scale = max(1.0, min(20.0, cfg_scale))

        generator = None
        if prompt_state.seed is not None:
            generator = torch.Generator(device=self.device).manual_seed(int(prompt_state.seed))

        result = self._pipe(
            prompt=prompt_state.prompt_text,
            image=init_image.resize(prompt_state.size, Image.LANCZOS),
            strength=strength,
            num_inference_steps=max(1, int(steps)),
            guidance_scale=cfg_scale,
            generator=generator,
        )
        return result.images[0]

    def generate_first_frame(
        self,
        *,
        prompt_state: BackendPromptState,
        steps: int,
        guidance_scale: float,
    ) -> Image.Image:
        self._ensure_txt2img_pipe()

        if not prompt_state.prompt_text:
            raise RuntimeError("3D_latent requires at least one text prompt")

        cfg_scale = float(guidance_scale)
        if cfg_scale > 30.0:
            cfg_scale = cfg_scale / 100.0
        cfg_scale = max(1.0, min(20.0, cfg_scale))

        generator = None
        if prompt_state.seed is not None:
            generator = torch.Generator(device=self.device).manual_seed(int(prompt_state.seed))

        out = self._txt2img_pipe(
            prompt=prompt_state.prompt_text,
            num_inference_steps=max(1, int(steps)),
            guidance_scale=cfg_scale,
            height=prompt_state.size[1],
            width=prompt_state.size[0],
            generator=generator,
        )
        return out.images[0]
