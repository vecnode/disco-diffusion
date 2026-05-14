"""Argument parsing for ``disco.py`` (optional overrides; omitted flags keep ``main.py`` defaults)."""

from __future__ import annotations

import argparse
import json
from typing import Any


def _int_key_mapping(raw: dict) -> dict:
    """JSON object keys are strings; prompt maps use int frame indices."""
    out = {}
    for k, v in raw.items():
        if isinstance(k, int):
            out[k] = v
        else:
            sk = str(k)
            out[int(sk) if sk.isdigit() else sk] = v
    return out


def _optional_json_dict(path: str) -> dict:
    """Accept either an inline JSON string or a path to a JSON file."""
    stripped = path.strip()
    if stripped.startswith("{"):
        data = json.loads(stripped)
    else:
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
    if not isinstance(data, dict):
        raise ValueError(f"Expected JSON object, got: {type(data)}")
    return _int_key_mapping(data)


def _positive_int(s: str) -> int:
    v = int(s)
    if v < 1:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return v


def parse_disco_argv(argv: list[str] | None) -> dict[str, Any]:
    """Return only keys the user set on the command line (no entries → no overrides)."""
    p = argparse.ArgumentParser(
        prog="disco.py",
        description="Disco Diffusion launcher; omit flags to keep defaults from src/discodiff/main.py.",
    )
    p.add_argument("--init-image", type=str, dest="init_image", help="URL/path, or empty for None")
    p.add_argument("--init-scale", type=float, dest="init_scale")
    p.add_argument("--skip-steps", type=int, dest="skip_steps")
    p.add_argument(
        "--steps",
        type=_positive_int,
        dest="steps",
        metavar="N",
        help="DDIM step count (timestep_respacing becomes ddim{N}; same as notebook steps)",
    )
    p.add_argument("--perlin-init", action=argparse.BooleanOptionalAction, dest="perlin_init")
    p.add_argument("--perlin-mode", type=str, dest="perlin_mode", choices=("gray", "color", "mixed"))
    p.add_argument(
        "--set-seed",
        type=str,
        dest="set_seed",
        metavar="STR_OR_INT",
        help="Literal random_seed or an integer string",
    )
    p.add_argument(
        "--text-prompts-json",
        type=str,
        dest="text_prompts_json",
        metavar="PATH",
        help="JSON object: frame index → list of prompt strings",
    )
    p.add_argument(
        "--width",
        type=int,
        dest="output_width",
        metavar="PX",
        help="Output width (use with --height; multiples of 64 work best)",
    )
    p.add_argument(
        "--height",
        type=int,
        dest="output_height",
        metavar="PX",
        help="Output height (use with --width)",
    )
    p.add_argument("--max-frames", type=_positive_int, dest="max_frames", metavar="N")
    p.add_argument("--translation-x", type=str, dest="translation_x", metavar="KEYFRAMES")
    p.add_argument("--translation-y", type=str, dest="translation_y", metavar="KEYFRAMES")
    p.add_argument("--translation-z", type=str, dest="translation_z", metavar="KEYFRAMES")
    p.add_argument("--rotation-3d-x", type=str, dest="rotation_3d_x", metavar="KEYFRAMES")
    p.add_argument("--rotation-3d-y", type=str, dest="rotation_3d_y", metavar="KEYFRAMES")
    p.add_argument("--rotation-3d-z", type=str, dest="rotation_3d_z", metavar="KEYFRAMES")
    p.add_argument("--near-plane", type=float, dest="near_plane")
    p.add_argument("--far-plane", type=float, dest="far_plane")
    p.add_argument("--fov", type=float, dest="fov")
    p.add_argument("--padding-mode", type=str, dest="padding_mode")
    p.add_argument("--sampling-mode", type=str, dest="sampling_mode")
    p.add_argument(
        "--depth-backend",
        type=str,
        dest="depth_backend",
        choices=("marigold",),
        help="Depth estimator for 3D reprojection.",
    )
    p.add_argument("--turbo-mode", action=argparse.BooleanOptionalAction, dest="turbo_mode")
    p.add_argument("--turbo-steps", type=str, dest="turbo_steps", metavar="N")
    p.add_argument("--turbo-preroll", type=_positive_int, dest="turbo_preroll", metavar="N")
    p.add_argument("--frames-scale", type=int, dest="frames_scale")
    p.add_argument("--frames-skip-steps", type=str, dest="frames_skip_steps", metavar="PERCENT")
    p.add_argument(
        "--latent-first-frame",
        type=str,
        dest="latent_first_frame_strategy",
        choices=("txt2img", "black"),
        help="How to seed frame 0 in 3D_latent when no init image is provided.",
    )
    p.add_argument(
        "--latent-strength",
        type=float,
        dest="latent_strength",
        help="Override img2img strength in 3D_latent (0..1). Lower values improve temporal coherence.",
    )
    p.add_argument(
        "--latent-temporal-blend",
        type=float,
        dest="latent_temporal_blend",
        help="Blend fraction of warped previous frame into each 3D_latent frame (0..1).",
    )
    p.add_argument(
        "--latent-novelty-strength",
        type=float,
        dest="latent_novelty_strength",
        help="Extra img2img strength ramped in over the run to add novelty without abrupt motion.",
    )
    p.add_argument(
        "--latent-color-reset",
        type=float,
        dest="latent_color_reset",
        help="Blend conditioning images toward neutral color balance to prevent saturation drift.",
    )
    p.add_argument(
        "--device",
        type=str,
        dest="device",
        metavar="DEV",
        help="Runtime device selector: auto, rtx, cpu, cuda, cuda:N, or mps.",
    )
    p.add_argument(
        "--profile",
        type=str,
        dest="profile",
        metavar="NAME",
        help="Runtime profile for backend defaults (for example: default, rtx, rtx-safe, rtx-fast).",
    )

    ns = p.parse_args(argv)
    ov: dict[str, Any] = {}

    for key in (
        "init_scale",
        "skip_steps",
        "steps",
        "perlin_mode",
        "max_frames",
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
        "depth_backend",
        "turbo_mode",
        "turbo_steps",
        "turbo_preroll",
        "frames_scale",
        "frames_skip_steps",
        "latent_first_frame_strategy",
        "latent_strength",
        "latent_temporal_blend",
        "latent_novelty_strength",
        "latent_color_reset",
        "perlin_init",
    ):
        v = getattr(ns, key)
        if v is not None:
            ov[key] = v

    if ns.set_seed is not None:
        ov["set_seed"] = ns.set_seed

    if ns.init_image is not None:
        ov["init_image"] = None if str(ns.init_image).strip() == "" else ns.init_image

    if ns.text_prompts_json is not None:
        ov["text_prompts"] = _optional_json_dict(ns.text_prompts_json)

    if ns.output_width is not None or ns.output_height is not None:
        if ns.output_width is None or ns.output_height is None:
            p.error("--width and --height must be given together")
        ov["width_height"] = [ns.output_width, ns.output_height]

    if ns.device is not None:
        ov["device"] = ns.device

    if ns.profile is not None:
        ov["profile"] = ns.profile

    return ov
