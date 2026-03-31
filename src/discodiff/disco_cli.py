"""CLI for `disco.py`: optional overrides only (omit a flag to keep `main.py` defaults)."""

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
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise ValueError(f"Expected JSON object in {path}")
    return _int_key_mapping(data)


def _parse_transformation_percent(s: str) -> list[float]:
    s = s.strip()
    if s.startswith("["):
        v = json.loads(s)
        if not isinstance(v, list):
            raise ValueError("transformation_percent JSON must be a list")
        return [float(x) for x in v]
    return [float(x.strip()) for x in s.split(",") if x.strip()]


def parse_disco_argv(argv: list[str] | None) -> dict[str, Any]:
    """Return only keys the user set on the command line (no entries → no overrides)."""
    p = argparse.ArgumentParser(
        prog="disco.py",
        description="Disco Diffusion launcher; omit flags to keep defaults from src/discodiff/main.py.",
    )
    p.add_argument("--clip-guidance-scale", type=float, dest="clip_guidance_scale")
    p.add_argument("--tv-scale", type=float, dest="tv_scale")
    p.add_argument("--range-scale", type=float, dest="range_scale")
    p.add_argument("--sat-scale", type=float, dest="sat_scale")
    p.add_argument("--cutn", type=int, dest="cutn")
    p.add_argument("--cutn-batches", type=int, dest="cutn_batches")
    p.add_argument("--init-image", type=str, dest="init_image", help="URL/path, or empty for None")
    p.add_argument("--init-scale", type=float, dest="init_scale")
    p.add_argument("--skip-steps", type=int, dest="skip_steps")
    p.add_argument("--perlin-init", action=argparse.BooleanOptionalAction, dest="perlin_init")
    p.add_argument("--perlin-mode", type=str, dest="perlin_mode", choices=("gray", "color", "mixed"))
    p.add_argument("--skip-augs", action=argparse.BooleanOptionalAction, dest="skip_augs")
    p.add_argument("--randomize-class", action=argparse.BooleanOptionalAction, dest="randomize_class")
    p.add_argument("--clip-denoised", action=argparse.BooleanOptionalAction, dest="clip_denoised")
    p.add_argument("--clamp-grad", action=argparse.BooleanOptionalAction, dest="clamp_grad")
    p.add_argument(
        "--set-seed",
        type=str,
        dest="set_seed",
        metavar="STR_OR_INT",
        help="Literal random_seed or an integer string",
    )
    p.add_argument("--fuzzy-prompt", action=argparse.BooleanOptionalAction, dest="fuzzy_prompt")
    p.add_argument("--rand-mag", type=float, dest="rand_mag")
    p.add_argument("--eta", type=float, dest="eta")
    p.add_argument(
        "--use-vertical-symmetry", action=argparse.BooleanOptionalAction, dest="use_vertical_symmetry"
    )
    p.add_argument(
        "--use-horizontal-symmetry",
        action=argparse.BooleanOptionalAction,
        dest="use_horizontal_symmetry",
    )
    p.add_argument(
        "--transformation-percent",
        type=str,
        dest="transformation_percent",
        help='JSON list e.g. [0.09] or comma floats e.g. "0.09"',
    )
    p.add_argument("--video-init-flow-warp", action=argparse.BooleanOptionalAction, dest="video_init_flow_warp")
    p.add_argument("--video-init-flow-blend", type=float, dest="video_init_flow_blend")
    p.add_argument(
        "--video-init-check-consistency",
        action=argparse.BooleanOptionalAction,
        dest="video_init_check_consistency",
    )
    p.add_argument(
        "--text-prompts-json",
        type=str,
        dest="text_prompts_json",
        metavar="PATH",
        help="JSON object: frame index → list of prompt strings",
    )
    p.add_argument(
        "--image-prompts-json",
        type=str,
        dest="image_prompts_json",
        metavar="PATH",
        help="JSON object: frame index → list of image prompt strings",
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

    ns = p.parse_args(argv)
    ov: dict[str, Any] = {}

    for key in (
        "clip_guidance_scale",
        "tv_scale",
        "range_scale",
        "sat_scale",
        "cutn",
        "cutn_batches",
        "init_scale",
        "skip_steps",
        "perlin_mode",
        "rand_mag",
        "eta",
        "video_init_flow_blend",
        "perlin_init",
        "skip_augs",
        "randomize_class",
        "clip_denoised",
        "clamp_grad",
        "fuzzy_prompt",
        "use_vertical_symmetry",
        "use_horizontal_symmetry",
        "video_init_flow_warp",
        "video_init_check_consistency",
    ):
        v = getattr(ns, key)
        if v is not None:
            ov[key] = v

    if ns.set_seed is not None:
        ov["set_seed"] = ns.set_seed

    if ns.init_image is not None:
        ov["init_image"] = None if str(ns.init_image).strip() == "" else ns.init_image

    if ns.transformation_percent is not None:
        ov["transformation_percent"] = _parse_transformation_percent(ns.transformation_percent)

    if ns.text_prompts_json is not None:
        ov["text_prompts"] = _optional_json_dict(ns.text_prompts_json)

    if ns.image_prompts_json is not None:
        ov["image_prompts"] = _optional_json_dict(ns.image_prompts_json)

    if ns.output_width is not None or ns.output_height is not None:
        if ns.output_width is None or ns.output_height is None:
            p.error("--width and --height must be given together")
        ov["width_height"] = [ns.output_width, ns.output_height]

    return ov
