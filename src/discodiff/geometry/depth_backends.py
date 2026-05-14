"""Depth estimation backends for 3D reprojection."""

from __future__ import annotations

import os
import re
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch
from PIL import Image


class DepthBackend(ABC):
    depth_contrast: float = 1.0

    @abstractmethod
    def predict_depth(self, image: Image.Image, target_size: tuple[int, int]) -> np.ndarray:
        raise NotImplementedError


class MarigoldDepthBackend(DepthBackend):
    def __init__(
        self,
        *,
        device: torch.device,
        models_root: str | None = None,
        model_id: str | None = None,
    ) -> None:
        self.device = device
        self.models_root = Path(models_root or os.path.join(os.getcwd(), "models"))
        self.model_id = model_id or os.environ.get("DISCO_MARIGOLD_MODEL", "prs-eth/marigold-depth-lcm-v1-0")
        self.depth_contrast = float(os.environ.get("DISCO_MARIGOLD_DEPTH_CONTRAST", "1.35"))
        self.invert_depth = os.environ.get("DISCO_MARIGOLD_INVERT_DEPTH", "0").strip() in {"1", "true", "True"}
        self.local_model_dir = Path(
            os.environ.get("DISCO_MARIGOLD_MODEL_DIR", "").strip()
            or self.models_root / "depth" / self._sanitize_repo_id(self.model_id)
        )
        self._pipe = None

    @staticmethod
    def _sanitize_repo_id(repo_id: str) -> str:
        return re.sub(r"[^a-zA-Z0-9._-]+", "_", repo_id)

    def _download_to_local_dir(self) -> None:
        self.local_model_dir.mkdir(parents=True, exist_ok=True)
        print(f"[depth:marigold] Downloading model '{self.model_id}' to {self.local_model_dir}", flush=True)
        try:
            from huggingface_hub import snapshot_download

            snapshot_download(repo_id=self.model_id, local_dir=str(self.local_model_dir), local_dir_use_symlinks=False)
        except Exception as exc:
            raise RuntimeError(
                f"Failed to download Marigold model '{self.model_id}' into {self.local_model_dir}: {exc}"
            ) from exc

    def _ensure_pipe(self) -> None:
        if self._pipe is not None:
            return
        try:
            from diffusers import MarigoldDepthPipeline
        except Exception as exc:
            raise RuntimeError(
                "Marigold depth backend requires diffusers with Marigold support. "
                "Run `uv sync` and ensure diffusers is up to date."
            ) from exc

        if not (self.local_model_dir / "model_index.json").exists():
            self._download_to_local_dir()

        dtype = torch.float16 if self.device.type == "cuda" else torch.float32
        pipe = MarigoldDepthPipeline.from_pretrained(str(self.local_model_dir), torch_dtype=dtype)
        pipe = pipe.to(self.device)
        pipe.set_progress_bar_config(disable=True)
        self._pipe = pipe

    @staticmethod
    def _extract_depth_array(result: Any) -> np.ndarray:
        for attr in ("depth_np", "prediction", "predicted_depth", "depth"):
            if hasattr(result, attr):
                value = getattr(result, attr)
                if isinstance(value, list) and value:
                    value = value[0]
                if isinstance(value, torch.Tensor):
                    value = value.detach().float().cpu().numpy()
                arr = np.asarray(value, dtype=np.float32)
                return np.squeeze(arr)
        raise RuntimeError("Marigold returned an unsupported output format; expected depth array field")

    def predict_depth(self, image: Image.Image, target_size: tuple[int, int]) -> np.ndarray:
        self._ensure_pipe()
        result = self._pipe(image, num_inference_steps=4, ensemble_size=1)
        depth = self._extract_depth_array(result)
        if self.invert_depth:
            depth = 1.0 / np.clip(depth, 1e-6, None)
        h = int(target_size[1])
        w = int(target_size[0])
        if depth.ndim != 2 or depth.shape != (h, w):
            depth = cv2.resize(depth, (w, h), interpolation=cv2.INTER_CUBIC)
        return np.nan_to_num(depth, nan=0.0, posinf=0.0, neginf=0.0)
