"""Assemble ``args`` SimpleNamespace from legacy ``main()`` locals."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any, Mapping

from .keyframes import split_prompts


def build_run_args_namespace(ns: Mapping[str, Any]) -> SimpleNamespace:
    """Mirror ``main()``'s original ``args = { ... }; SimpleNamespace(**args)`` block."""
    animation_mode = ns["GENERATION_MODE"]
    args: dict[str, Any] = {
        "batchNum": ns["batchNum"],
        "prompts_series": split_prompts(ns["text_prompts"], ns["max_frames"])
        if ns["text_prompts"]
        else None,
        "image_prompts_series": split_prompts(ns["image_prompts"], ns["max_frames"])
        if ns["image_prompts"]
        else None,
        "seed": ns["seed"],
        "display_rate": ns["display_rate"],
        "n_batches": ns["n_batches"] if animation_mode == "None" else 1,
        "batch_size": ns["batch_size"],
        "batch_name": ns["batch_name"],
        "steps": ns["steps"],
        "diffusion_sampling_mode": ns["diffusion_sampling_mode"],
        "width_height": ns["width_height"],
        "clip_guidance_scale": ns["clip_guidance_scale"],
        "tv_scale": ns["tv_scale"],
        "range_scale": ns["range_scale"],
        "sat_scale": ns["sat_scale"],
        "cutn_batches": ns["cutn_batches"],
        "init_image": ns["init_image"],
        "init_scale": ns["init_scale"],
        "skip_steps": ns["skip_steps"],
        "side_x": ns["side_x"],
        "side_y": ns["side_y"],
        "timestep_respacing": ns["timestep_respacing"],
        "diffusion_steps": ns["diffusion_steps"],
        "animation_mode": animation_mode,
        "video_init_path": ns["video_init_path"],
        "extract_nth_frame": ns["extract_nth_frame"],
        "video_init_seed_continuity": ns["video_init_seed_continuity"],
        "key_frames": ns["key_frames"],
        "max_frames": ns["max_frames"] if animation_mode != "None" else 1,
        "interp_spline": ns["interp_spline"],
        "start_frame": ns["start_frame"],
        "angle": ns["angle"],
        "zoom": ns["zoom"],
        "translation_x": ns["translation_x"],
        "translation_y": ns["translation_y"],
        "translation_z": ns["translation_z"],
        "rotation_3d_x": ns["rotation_3d_x"],
        "rotation_3d_y": ns["rotation_3d_y"],
        "rotation_3d_z": ns["rotation_3d_z"],
        "midas_depth_model": ns["midas_depth_model"],
        "midas_weight": ns["midas_weight"],
        "near_plane": ns["near_plane"],
        "far_plane": ns["far_plane"],
        "fov": ns["fov"],
        "padding_mode": ns["padding_mode"],
        "sampling_mode": ns["sampling_mode"],
        "angle_series": ns["angle_series"],
        "zoom_series": ns["zoom_series"],
        "translation_x_series": ns["translation_x_series"],
        "translation_y_series": ns["translation_y_series"],
        "translation_z_series": ns["translation_z_series"],
        "rotation_3d_x_series": ns["rotation_3d_x_series"],
        "rotation_3d_y_series": ns["rotation_3d_y_series"],
        "rotation_3d_z_series": ns["rotation_3d_z_series"],
        "frames_scale": ns["frames_scale"],
        "skip_step_ratio": ns["skip_step_ratio"],
        "calc_frames_skip_steps": ns["calc_frames_skip_steps"],
        "text_prompts": ns["text_prompts"],
        "image_prompts": ns["image_prompts"],
        "cut_overview": eval(ns["cut_overview"]),
        "cut_innercut": eval(ns["cut_innercut"]),
        "cut_ic_pow": eval(ns["cut_ic_pow"]),
        "cut_icgray_p": eval(ns["cut_icgray_p"]),
        "intermediate_saves": ns["intermediate_saves"],
        "intermediates_in_subfolder": ns["intermediates_in_subfolder"],
        "steps_per_checkpoint": ns["steps_per_checkpoint"],
        "perlin_init": ns["perlin_init"],
        "perlin_mode": ns["perlin_mode"],
        "set_seed": ns["set_seed"],
        "eta": ns["eta"],
        "clamp_grad": ns["clamp_grad"],
        "clamp_max": ns["clamp_max"],
        "skip_augs": ns["skip_augs"],
        "randomize_class": ns["randomize_class"],
        "clip_denoised": ns["clip_denoised"],
        "fuzzy_prompt": ns["fuzzy_prompt"],
        "rand_mag": ns["rand_mag"],
        "turbo_mode": ns["turbo_mode"],
        "turbo_steps": ns["turbo_steps"],
        "turbo_preroll": ns["turbo_preroll"],
        "use_vertical_symmetry": ns["use_vertical_symmetry"],
        "use_horizontal_symmetry": ns["use_horizontal_symmetry"],
        "transformation_percent": ns["transformation_percent"],
        "video_init_steps": ns["video_init_steps"],
        "video_init_clip_guidance_scale": ns["video_init_clip_guidance_scale"],
        "video_init_tv_scale": ns["video_init_tv_scale"],
        "video_init_range_scale": ns["video_init_range_scale"],
        "video_init_sat_scale": ns["video_init_sat_scale"],
        "video_init_cutn_batches": ns["video_init_cutn_batches"],
        "video_init_skip_steps": ns["video_init_skip_steps"],
        "video_init_frames_scale": ns["video_init_frames_scale"],
        "video_init_frames_skip_steps": ns["video_init_frames_skip_steps"],
        "video_init_flow_warp": ns["video_init_flow_warp"],
        "video_init_flow_blend": ns["video_init_flow_blend"],
        "video_init_check_consistency": ns["video_init_check_consistency"],
        "video_init_blend_mode": ns["video_init_blend_mode"],
    }

    if animation_mode == "Video Input":
        args["steps"] = args["video_init_steps"]
        args["clip_guidance_scale"] = args["video_init_clip_guidance_scale"]
        args["tv_scale"] = args["video_init_tv_scale"]
        args["range_scale"] = args["video_init_range_scale"]
        args["sat_scale"] = args["video_init_sat_scale"]
        args["cutn_batches"] = args["video_init_cutn_batches"]
        args["skip_steps"] = args["video_init_skip_steps"]
        args["frames_scale"] = args["video_init_frames_scale"]
        args["frames_skip_steps"] = args["video_init_frames_skip_steps"]

    return SimpleNamespace(**args)
