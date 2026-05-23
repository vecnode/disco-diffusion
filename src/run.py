"""Diffusion application runtime (notebook-style script body; launched via repo `main.py`)."""
from __future__ import annotations

def main(cli_overrides: dict | None = None) -> None:
    import os
    import sys

    print("[run] Runtime starting.", flush=True)

    from platform.device import warn_if_unsupported_platform

    warn_if_unsupported_platform()

    import shutil

    from assets import (
        createPath,
        fetch,
    )

    # If running locally, there's a good chance your env will need this in order to not crash upon np.matmul() or similar operations.
    os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'


    ROOT_PATH = os.getcwd()

    USE_CPU = False

    PROJECT_DIR = os.path.abspath(os.getcwd())
    from config import RunConfig, apply_runtime_overrides

    run_config = RunConfig.from_env(ROOT_PATH)
    if cli_overrides:
        run_config = apply_runtime_overrides(
            run_config,
            device=cli_overrides.get("device", run_config.device),
            profile=cli_overrides.get("profile", run_config.profile),
        )


    batch_name = 'example'
    steps = 100 # [25,50,100,150,250,500,1000]

    latent_guidance_scale = 750

    init_image = None
    init_scale = 1000
    skip_steps = 10

    key_frames = True
    max_frames = 10000
    interp_spline = 'Linear' # ['Linear','Quadratic','Cubic']
    angle = "0:(0)"
    zoom = "0: (1), 10: (1.05)"
    translation_x = "0: (0)"
    translation_y = "0: (0)"
    translation_z = "0: (10.0)"
    rotation_3d_x = "0: (0)"
    rotation_3d_y = "0: (0)"
    rotation_3d_z = "0: (0)"
    near_plane = 200
    far_plane = 1000
    fov = 60
    padding_mode = 'border'
    sampling_mode = 'bicubic'

    turbo_mode = True
    turbo_steps = "3" # ["2","3","4","5","6","10"]
    turbo_preroll = 24 # frames

    frames_scale = 1500
    frames_skip_steps = '60%'

    perlin_init = False
    perlin_mode = 'mixed' # ['mixed', 'color', 'gray']
    set_seed = 'random_seed'

    depth_backend = 'marigold'
    latent_first_frame_strategy = 'txt2img'  # ['txt2img', 'black']
    latent_strength = None  # Optional float in [0,1]
    latent_temporal_blend = 0.18  # Blend warped previous frame into latent output
    latent_novelty_strength = 0.08  # Extra img2img strength ramped in over the run
    latent_color_reset = 0.14  # Pull conditioning images toward neutral color balance




    inputDirPath = f'{ROOT_PATH}/input'
    createPath(inputDirPath)
    outputDirPath = str(run_config.output_dir)
    createPath(outputDirPath)

    model_path = f'{ROOT_PATH}/models'
    createPath(model_path)



    from diffusion import LatentDiffusionBackend

    # Package helpers (geometry.warp, config.keyframes) — no upstream clone.
    sys.path.append(PROJECT_DIR)

    import torch
    import torchvision.transforms.functional as TF

    import cv2
    import gc
    import math
    from PIL import Image
    from glob import glob
    import json
    from tqdm import tqdm
    import numpy as np
    import random
    import warnings

    os.chdir(PROJECT_DIR)
    warnings.filterwarnings("ignore", category=UserWarning)

    from platform.device import apply_backend_defaults, log_device_selection, resolve_runtime_device

    device_selection = resolve_runtime_device(run_config.device, use_cpu=USE_CPU)
    DEVICE = torch.device(device_selection.torch_device)
    apply_backend_defaults(device_selection, profile=run_config.profile)
    log_device_selection(device_selection)

    run_config = apply_runtime_overrides(run_config, device=device_selection.torch_device)
    print('Using device:', DEVICE)
    device = DEVICE # At least one of the modules expects this name..

    if not USE_CPU and DEVICE.type == 'cuda':
        if torch.cuda.get_device_capability(DEVICE) == (8, 0):  # A100 fix thanks to Emad
            print('Disabling CUDNN for A100 gpu', file=sys.stderr)
            torch.backends.cudnn.enabled = False

    def init_marigold_depth_backend():
        print("Initializing Marigold depth backend")
        backend = MarigoldDepthBackend(device=DEVICE, models_root=model_path)
        print("Marigold depth backend initialized.")
        return backend




    from geometry import MarigoldDepthBackend
    from geometry import py3d_tools as p3dT
    from geometry import warp as dxf

    from image import noise as _noise
    stop_on_next_loop = False  # Make sure GPU memory doesn't get corrupted from cancelling the run mid-way through, allow a full frame to complete
    TRANSLATION_SCALE = 1.0/200.0
    FORWARD_TRANSLATION_DAMPING = 0.5
    stabilization_warmup_frames = max(24, int(turbo_preroll))
    effective_turbo_preroll = stabilization_warmup_frames

    def _smoothstep(value: float) -> float:
        value = max(0.0, min(1.0, value))
        return value * value * (3.0 - 2.0 * value)

    def _stabilization_progress(frame_index: int) -> float:
        if stabilization_warmup_frames <= 0:
            return 1.0
        return _smoothstep(frame_index / float(stabilization_warmup_frames))


    def do_3d_step(img_filepath, frame_num, depth_backend_instance):
        warmup_progress = _stabilization_progress(frame_num)
        motion_scale = 0.2 + 0.8 * warmup_progress

        if args.key_frames:
            translation_x = args.translation_x_series[frame_num]
            translation_y = args.translation_y_series[frame_num]
            translation_z = args.translation_z_series[frame_num]
            rotation_3d_x = args.rotation_3d_x_series[frame_num]
            rotation_3d_y = args.rotation_3d_y_series[frame_num]
            rotation_3d_z = args.rotation_3d_z_series[frame_num]
            # Silently read keyframe values

        translate_xyz = [
            -translation_x * TRANSLATION_SCALE * motion_scale,
            translation_y * TRANSLATION_SCALE * motion_scale,
            -translation_z * TRANSLATION_SCALE * motion_scale * FORWARD_TRANSLATION_DAMPING,
        ]
        rotate_xyz_degrees = [
            rotation_3d_x * motion_scale,
            rotation_3d_y * motion_scale,
            rotation_3d_z * motion_scale,
        ]
        rotate_xyz = [math.radians(rotate_xyz_degrees[0]), math.radians(rotate_xyz_degrees[1]), math.radians(rotate_xyz_degrees[2])]
        rot_mat = p3dT.euler_angles_to_matrix(torch.tensor(rotate_xyz, device=device), "XYZ").unsqueeze(0)
        next_step_pil = dxf.transform_image_3d(img_filepath, depth_backend_instance, DEVICE,
                               rot_mat, translate_xyz, args.near_plane, args.far_plane,
                               args.fov, padding_mode=args.padding_mode,
                               sampling_mode=args.sampling_mode,
                               debug_dir=debug3dFolder, frame_num=frame_num)
        return next_step_pil

    def do_run():
            from platform.cuda import use_cudnn_benchmark_mode

            _cudnn_benchmark = use_cudnn_benchmark_mode()
            seed = args.seed
            last_rendered_frame = None
            print(range(args.start_frame, args.max_frames))

            depth_backend_instance = None
            selected_depth_backend = "marigold"
            depth_backend_instance = init_marigold_depth_backend()
            print(f"[3D] depth backend: {selected_depth_backend}")
            if turbo_mode:
                print(
                    f"[turbo] steps={args.steps} turbo_steps={int(turbo_steps)} "
                    f"frames_skip_steps={frames_skip_steps} calc_skip_steps={args.calc_frames_skip_steps} "
                    f"effective_diffusion_steps={args.steps - args.calc_frames_skip_steps}"
                )
            for frame_num in range(args.start_frame, args.max_frames):
                if stop_on_next_loop:
                  break

                # Print Frame progress for each frame
                batchBar = tqdm(range(args.max_frames), desc ="Frames")
                batchBar.n = frame_num
                batchBar.refresh()


                if args.init_image in ['', 'none', 'None', 'NONE']:
                  init_image = None
                else:
                  init_image = args.init_image
                init_scale = args.init_scale
                skip_steps = args.skip_steps

                if frame_num > 0:
                    seed += 1
                    if resume_run and frame_num == start_frame:
                        img_filepath = batchFolder + f"/{batch_name}({batchNum})_{start_frame-1:04}.png"
                        if turbo_mode and frame_num > effective_turbo_preroll:
                            shutil.copyfile(img_filepath, old_frame_scaled_path)
                    else:
                        img_filepath = prev_frame_path

                    if not os.path.exists(img_filepath):
                        if last_rendered_frame is not None:
                            last_rendered_frame.save(img_filepath)
                        else:
                            fallback_frame_path = batchFolder + f"/{batch_name}({batchNum})_{frame_num-1:04}.png"
                            if os.path.exists(fallback_frame_path):
                                shutil.copyfile(fallback_frame_path, img_filepath)
                            else:
                                raise FileNotFoundError(
                                    f"Missing prior frame for 3D warp: {img_filepath} (fallback: {fallback_frame_path})"
                                )

                    next_step_pil = do_3d_step(img_filepath, frame_num, depth_backend_instance)
                    next_step_pil.save(prev_frame_scaled_path)

                    ### Turbo mode - skip some diffusions, use 3d morph for clarity and to save time
                    if turbo_mode:
                        if frame_num == effective_turbo_preroll:  # start tracking oldframe
                            next_step_pil.save(old_frame_scaled_path)  # stash for later blending
                        elif frame_num > effective_turbo_preroll:
                            if not os.path.exists(old_frame_scaled_path):
                                # Bootstrap turbo old-frame state if missing (fresh run / resumed run without cache).
                                next_step_pil.save(old_frame_scaled_path)
                            # set up 2 warped image sequences, old & new, to blend toward new diff image
                            # Reuse the newly warped frame in latent mode to avoid a second AdaBins+reprojection pass.
                            old_frame = next_step_pil
                            old_frame.save(old_frame_scaled_path)
                            if frame_num % int(turbo_steps) != 0:
                                print('turbo skip this frame: skipping diffusion step')
                                filename = f'{args.batch_name}({args.batchNum})_{frame_num:04}.png'
                                blend_factor = ((frame_num % int(turbo_steps)) + 1) / int(turbo_steps)
                                print('turbo skip this frame: saving blended frame')
                                newWarpedImg = cv2.imread(prev_frame_scaled_path)  # this is already updated..
                                oldWarpedImg = cv2.imread(old_frame_scaled_path)
                                blendedImage = cv2.addWeighted(newWarpedImg, blend_factor, oldWarpedImg, 1 - blend_factor, 0.0)
                                cv2.imwrite(f'{batchFolder}/{filename}', blendedImage)
                                next_step_pil.save(f'{img_filepath}')  # save it also as prev_frame to feed next iteration
                                continue
                            else:
                                oldWarpedImg = cv2.imread(prev_frame_scaled_path)
                                cv2.imwrite(old_frame_scaled_path, oldWarpedImg)  # swap in for blending later
                                print('diffuse this frame')

                    init_image = prev_frame_scaled_path
                    warmup_progress = _stabilization_progress(frame_num)
                    skip_steps = min(
                            args.steps - 1,
                            int(round(args.calc_frames_skip_steps + (10.0 * (1.0 - warmup_progress))))
                    )

                if seed is not None:
                    np.random.seed(seed)
                    random.seed(seed)
                    torch.manual_seed(seed)
                    if device.type == 'cuda':
                        torch.cuda.manual_seed_all(seed)
                    if _cudnn_benchmark and torch.backends.cudnn.enabled:
                        torch.backends.cudnn.benchmark = True
                        torch.backends.cudnn.deterministic = False
                    else:
                        torch.backends.cudnn.benchmark = False
                        torch.backends.cudnn.deterministic = True

                if args.prompts_series is not None and frame_num >= len(args.prompts_series):
                  frame_prompt = args.prompts_series[-1]
                elif args.prompts_series is not None:
                  frame_prompt = args.prompts_series[frame_num]
                else:
                  frame_prompt = []

                # Only print at keyframes with new prompts
                if frame_num == 0 or (args.prompts_series is not None and frame_num < len(args.prompts_series) and args.prompts_series[frame_num] != args.prompts_series[frame_num - 1]):
                  if frame_prompt:
                    print(f'Frame {frame_num} Text Prompt: {frame_prompt}')

                init_pil = None
                if init_image is not None:
                    init_pil = Image.open(fetch(init_image)).convert('RGB')
                    init_pil = init_pil.resize((args.side_x, args.side_y), Image.LANCZOS)
                elif latent_first_frame_strategy == "black":
                    init_pil = Image.new("RGB", (args.side_x, args.side_y), color=(0, 0, 0))
                for i in range(args.n_batches):
                    print('')
                    gc.collect()
                    torch.cuda.empty_cache()
                    total_steps = 1
                    save_num = f'{frame_num:04}'
                    filename = f'{args.batch_name}({args.batchNum})_{save_num}.png'
                    prompt_state = latent_backend.prepare(frame_prompt, seed, (args.side_x, args.side_y))
                    if frame_num == 0 and init_image is None and latent_first_frame_strategy == "txt2img":
                        image = latent_backend.generate_first_frame(
                            prompt_state=prompt_state,
                            steps=args.steps,
                            guidance_scale=latent_guidance_scale,
                        )
                    else:
                        if init_pil is None:
                            init_pil = Image.new("RGB", (args.side_x, args.side_y), color=(0, 0, 0))
                        if latent_strength is not None:
                            strength = max(0.05, min(0.95, float(latent_strength)))
                        else:
                            # More conservative default than skip/steps to reduce inter-frame drift.
                            ratio = float(skip_steps) / max(float(args.steps), 1.0)
                            strength = max(0.05, min(0.95, ratio * 0.65))
                        if latent_novelty_strength:
                            run_progress = _smoothstep(frame_num / float(max(args.max_frames - 1, 1)))
                            strength = max(0.05, min(0.95, strength + float(latent_novelty_strength) * run_progress))
                        if latent_color_reset:
                            color_reset = max(0.0, min(0.85, float(latent_color_reset) * (0.35 + 0.65 * _smoothstep(frame_num / float(max(args.max_frames - 1, 1))))))
                        else:
                            color_reset = 0.0
                        image = latent_backend.generate(
                            init_image=init_pil,
                            strength_or_skip=strength,
                            steps=args.steps,
                            guidance_scale=latent_guidance_scale,
                            extra_guidance_state={
                                "prompt_state": prompt_state,
                                "latent_color_reset": color_reset,
                            },
                        )
                    if frame_num > 0 and init_pil is not None and latent_temporal_blend > 0:
                        keep = max(0.0, min(1.0, float(latent_temporal_blend)))
                        keep *= max(0.0, (1.0 - run_progress) ** 3)
                        image = Image.blend(image, init_pil, keep)
                    image.save(progress_path)
                    if frame_num == 0:
                        save_settings()
                    image.save(prev_frame_path)
                    image.save(f'{batchFolder}/{filename}')
                    last_rendered_frame = image.copy()
                    next_init_pil = image.copy()
                    if turbo_mode and frame_num > 0:
                        blend_factor = 1.0 / int(turbo_steps)
                        newFrame = cv2.imread(prev_frame_path)
                        prev_frame_warped = cv2.imread(prev_frame_scaled_path)
                        blendedImage = cv2.addWeighted(newFrame, blend_factor, prev_frame_warped, (1-blend_factor), 0.0)
                        cv2.imwrite(f'{batchFolder}/{filename}', blendedImage)
                        next_init_pil = Image.fromarray(cv2.cvtColor(blendedImage, cv2.COLOR_BGR2RGB))
                    init_pil = next_init_pil
                    continue




    def save_settings():
        setting_list = {
          'text_prompts': text_prompts,
                    'latent_guidance_scale': latent_guidance_scale,
          'max_frames': max_frames,
          'interp_spline': interp_spline,
          # 'rotation_per_frame': rotation_per_frame,
          'init_image': init_image,
          'init_scale': init_scale,
          'skip_steps': skip_steps,
          # 'zoom_per_frame': zoom_per_frame,
          'frames_scale': frames_scale,
          'frames_skip_steps': frames_skip_steps,
          'perlin_init': perlin_init,
          'perlin_mode': perlin_mode,
          'seed': seed,
          'width': width_height[0],
          'height': width_height[1],
          'steps': steps,
          'key_frames': key_frames,
          'max_frames': max_frames,
          'angle': angle,
          'zoom': zoom,
          'translation_x': translation_x,
          'translation_y': translation_y,
          'translation_z': translation_z,
          'rotation_3d_x': rotation_3d_x,
          'rotation_3d_y': rotation_3d_y,
          'rotation_3d_z': rotation_3d_z,
          'near_plane': near_plane,
          'far_plane': far_plane,
          'fov': fov,
          'padding_mode': padding_mode,
          'sampling_mode': sampling_mode,
          'turbo_mode':turbo_mode,
          'turbo_steps':turbo_steps,
          'turbo_preroll':turbo_preroll,
          'depth_backend': depth_backend,
          'latent_first_frame_strategy': latent_first_frame_strategy,
          'latent_strength': latent_strength,
          'latent_temporal_blend': latent_temporal_blend,
          'latent_novelty_strength': latent_novelty_strength,
          'latent_color_reset': latent_color_reset,
        }
        with open(f"{batchFolder}/{batch_name}({batchNum})_settings.txt", "w+", encoding="utf-8") as f:   #save settings
            json.dump(setting_list, f, ensure_ascii=False, indent=4)


    width_height = [1024, 576]


    side_x = (width_height[0]//64)*64;
    side_y = (width_height[1]//64)*64;
    if side_x != width_height[0] or side_y != width_height[1]:
        print(f'Changing output size to {side_x}x{side_y}. Dimensions must by multiples of 64.')


    # Make folder for batch
    batchFolder = f'{outputDirPath}/{batch_name}'
    createPath(batchFolder)

    runtimeFolder = f"{batchFolder}/runtime"
    createPath(runtimeFolder)
    debug3dFolder = f"{batchFolder}/3d"
    createPath(debug3dFolder)
    progress_path = f"{runtimeFolder}/progress.png"
    prev_frame_path = f"{runtimeFolder}/prevFrame.png"
    prev_frame_scaled_path = f"{runtimeFolder}/prevFrameScaled.png"
    old_frame_scaled_path = f"{runtimeFolder}/oldFrameScaled.png"

    # Default text prompts can be overridden via CLI JSON.
    text_prompts = {
        0: ["A beautiful painting of a singular lighthouse, shining its light across a tumultuous sea of blood by greg rutkowski and thomas kinkade, Trending on artstation.", "yellow color scheme"],
        100: ["This set of prompts start at frame 100", "This prompt has weight five:5"],
    }

    def _apply_cli_overrides(ov: dict | None) -> None:
        nonlocal latent_guidance_scale
        nonlocal init_image, init_scale, skip_steps, perlin_init, perlin_mode
        nonlocal set_seed
        nonlocal text_prompts
        nonlocal width_height, side_x, side_y, steps, max_frames
        nonlocal translation_x, translation_y, translation_z
        nonlocal rotation_3d_x, rotation_3d_y, rotation_3d_z, near_plane, far_plane, fov
        nonlocal padding_mode, sampling_mode
        nonlocal depth_backend, latent_first_frame_strategy, latent_strength, latent_temporal_blend
        nonlocal latent_novelty_strength, latent_color_reset
        nonlocal turbo_mode, turbo_steps, turbo_preroll, frames_scale, frames_skip_steps
        nonlocal zoom

        if not ov:
            zoom = "0: (1)"
            translation_z = "0: (1.5)"
            return

        if "init_image" in ov:
            init_image = ov["init_image"]
        if "init_scale" in ov:
            init_scale = ov["init_scale"]
        if "skip_steps" in ov:
            skip_steps = ov["skip_steps"]
        if "steps" in ov:
            steps = ov["steps"]
        if "perlin_init" in ov:
            perlin_init = ov["perlin_init"]
        if "perlin_mode" in ov:
            perlin_mode = ov["perlin_mode"]
        if "set_seed" in ov:
            set_seed = ov["set_seed"]
        if "max_frames" in ov:
            max_frames = ov["max_frames"]
        if "translation_x" in ov:
            translation_x = ov["translation_x"]
        if "translation_y" in ov:
            translation_y = ov["translation_y"]
        if "translation_z" in ov:
            translation_z = ov["translation_z"]
        if "rotation_3d_x" in ov:
            rotation_3d_x = ov["rotation_3d_x"]
        if "rotation_3d_y" in ov:
            rotation_3d_y = ov["rotation_3d_y"]
        if "rotation_3d_z" in ov:
            rotation_3d_z = ov["rotation_3d_z"]
        if "near_plane" in ov:
            near_plane = ov["near_plane"]
        if "far_plane" in ov:
            far_plane = ov["far_plane"]
        if "fov" in ov:
            fov = ov["fov"]
        if "padding_mode" in ov:
            padding_mode = ov["padding_mode"]
        if "sampling_mode" in ov:
            sampling_mode = ov["sampling_mode"]
        if "depth_backend" in ov:
            depth_backend = "marigold"
        if "latent_first_frame_strategy" in ov:
            latent_first_frame_strategy = str(ov["latent_first_frame_strategy"]).lower()
        if "latent_strength" in ov:
            latent_strength = ov["latent_strength"]
        if "latent_temporal_blend" in ov:
            latent_temporal_blend = ov["latent_temporal_blend"]
        if "latent_novelty_strength" in ov:
            latent_novelty_strength = ov["latent_novelty_strength"]
        if "latent_color_reset" in ov:
            latent_color_reset = ov["latent_color_reset"]
        if "turbo_mode" in ov:
            turbo_mode = ov["turbo_mode"]
        if "turbo_steps" in ov:
            turbo_steps = ov["turbo_steps"]
        if "turbo_preroll" in ov:
            turbo_preroll = ov["turbo_preroll"]
        if "frames_scale" in ov:
            frames_scale = ov["frames_scale"]
        if "frames_skip_steps" in ov:
            frames_skip_steps = ov["frames_skip_steps"]
        if "text_prompts" in ov:
            text_prompts = ov["text_prompts"]
        if "width_height" in ov:
            width_height = [ov["width_height"][0], ov["width_height"][1]]
            side_x = (width_height[0] // 64) * 64
            side_y = (width_height[1] // 64) * 64
            if side_x != width_height[0] or side_y != width_height[1]:
                print(
                    f"Changing output size to {side_x}x{side_y}. Dimensions must be multiples of 64."
                )

        zoom = "0: (1)"
        translation_z = "0: (1.5)"

    _apply_cli_overrides(cli_overrides)



    """
    ### Animation Settings
    """

    # 2D Animation Settings:**
    # `zoom` is a multiplier of dimensions, 1 is no zoom.
    # All rotations are provided in degrees.

    from config.keyframes import get_inbetweens, parse_key_frames


    if key_frames:
        try:
            angle_series = get_inbetweens(parse_key_frames(angle), max_frames, interp_spline)
        except RuntimeError as e:
            print(
                "WARNING: You have selected to use key frames, but you have not "
                "formatted `angle` correctly for key frames.\n"
                "Attempting to interpret `angle` as "
                f'"0: ({angle})"\n'
                "Please read the instructions to find out how to use key frames "
                "correctly.\n"
            )
            angle = f"0: ({angle})"
            angle_series = get_inbetweens(parse_key_frames(angle), max_frames, interp_spline)

        try:
            zoom_series = get_inbetweens(parse_key_frames(zoom), max_frames, interp_spline)
        except RuntimeError as e:
            print(
                "WARNING: You have selected to use key frames, but you have not "
                "formatted `zoom` correctly for key frames.\n"
                "Attempting to interpret `zoom` as "
                f'"0: ({zoom})"\n'
                "Please read the instructions to find out how to use key frames "
                "correctly.\n"
            )
            zoom = f"0: ({zoom})"
            zoom_series = get_inbetweens(parse_key_frames(zoom), max_frames, interp_spline)

        try:
            translation_x_series = get_inbetweens(parse_key_frames(translation_x), max_frames, interp_spline)
        except RuntimeError as e:
            print(
                "WARNING: You have selected to use key frames, but you have not "
                "formatted `translation_x` correctly for key frames.\n"
                "Attempting to interpret `translation_x` as "
                f'"0: ({translation_x})"\n'
                "Please read the instructions to find out how to use key frames "
                "correctly.\n"
            )
            translation_x = f"0: ({translation_x})"
            translation_x_series = get_inbetweens(parse_key_frames(translation_x), max_frames, interp_spline)

        try:
            translation_y_series = get_inbetweens(parse_key_frames(translation_y), max_frames, interp_spline)
        except RuntimeError as e:
            print(
                "WARNING: You have selected to use key frames, but you have not "
                "formatted `translation_y` correctly for key frames.\n"
                "Attempting to interpret `translation_y` as "
                f'"0: ({translation_y})"\n'
                "Please read the instructions to find out how to use key frames "
                "correctly.\n"
            )
            translation_y = f"0: ({translation_y})"
            translation_y_series = get_inbetweens(parse_key_frames(translation_y), max_frames, interp_spline)

        try:
            translation_z_series = get_inbetweens(parse_key_frames(translation_z), max_frames, interp_spline)
        except RuntimeError as e:
            print(
                "WARNING: You have selected to use key frames, but you have not "
                "formatted `translation_z` correctly for key frames.\n"
                "Attempting to interpret `translation_z` as "
                f'"0: ({translation_z})"\n'
                "Please read the instructions to find out how to use key frames "
                "correctly.\n"
            )
            translation_z = f"0: ({translation_z})"
            translation_z_series = get_inbetweens(parse_key_frames(translation_z), max_frames, interp_spline)

        try:
            rotation_3d_x_series = get_inbetweens(parse_key_frames(rotation_3d_x), max_frames, interp_spline)
        except RuntimeError as e:
            print(
                "WARNING: You have selected to use key frames, but you have not "
                "formatted `rotation_3d_x` correctly for key frames.\n"
                "Attempting to interpret `rotation_3d_x` as "
                f'"0: ({rotation_3d_x})"\n'
                "Please read the instructions to find out how to use key frames "
                "correctly.\n"
            )
            rotation_3d_x = f"0: ({rotation_3d_x})"
            rotation_3d_x_series = get_inbetweens(parse_key_frames(rotation_3d_x), max_frames, interp_spline)

        try:
            rotation_3d_y_series = get_inbetweens(parse_key_frames(rotation_3d_y), max_frames, interp_spline)
        except RuntimeError as e:
            print(
                "WARNING: You have selected to use key frames, but you have not "
                "formatted `rotation_3d_y` correctly for key frames.\n"
                "Attempting to interpret `rotation_3d_y` as "
                f'"0: ({rotation_3d_y})"\n'
                "Please read the instructions to find out how to use key frames "
                "correctly.\n"
            )
            rotation_3d_y = f"0: ({rotation_3d_y})"
            rotation_3d_y_series = get_inbetweens(parse_key_frames(rotation_3d_y), max_frames, interp_spline)

        try:
            rotation_3d_z_series = get_inbetweens(parse_key_frames(rotation_3d_z), max_frames, interp_spline)
        except RuntimeError as e:
            print(
                "WARNING: You have selected to use key frames, but you have not "
                "formatted `rotation_3d_z` correctly for key frames.\n"
                "Attempting to interpret `rotation_3d_z` as "
                f'"0: ({rotation_3d_z})"\n'
                "Please read the instructions to find out how to use key frames "
                "correctly.\n"
            )
            rotation_3d_z = f"0: ({rotation_3d_z})"
            rotation_3d_z_series = get_inbetweens(parse_key_frames(rotation_3d_z), max_frames, interp_spline)

    else:
        angle = float(angle)
        zoom = float(zoom)
        translation_x = float(translation_x)
        translation_y = float(translation_y)
        translation_z = float(translation_z)
        rotation_3d_x = float(rotation_3d_x)
        rotation_3d_y = float(rotation_3d_y)
        rotation_3d_z = float(rotation_3d_z)



    """
    ### Extra Settings
        Partial Saves and runtime controls
    """


    intermediate_saves = 0
    intermediates_in_subfolder = True
    # Intermediate steps will save a copy at your specified intervals. You can either format it as a single integer or a list of specific steps

    # A value of `2` will save a copy at 33% and 66%. 0 will save none.

    # A value of `[5, 9, 34, 45]` will save at steps 5, 9, 34, and 45. (Make sure to include the brackets)


    if type(intermediate_saves) is not list:
        if intermediate_saves:
            steps_per_checkpoint = math.floor((steps - skip_steps - 1) // (intermediate_saves+1))
            steps_per_checkpoint = steps_per_checkpoint if steps_per_checkpoint > 0 else 1
            print(f'Will save every {steps_per_checkpoint} steps')
        else:
            steps_per_checkpoint = steps+10
    else:
        steps_per_checkpoint = None

    partialFolder = None
    if intermediates_in_subfolder is True:
        partialFolder = f'{batchFolder}/partials'
        createPath(partialFolder)




    def move_files(start_num, end_num, old_folder, new_folder):
        for i in range(start_num, end_num):
            old_file = old_folder + f'/{batch_name}({batchNum})_{i:04}.png'
            new_file = new_folder + f'/{batch_name}({batchNum})_{i:04}.png'
            os.rename(old_file, new_file)

    resume_run = False
    run_to_resume = 'latest'
    resume_from_frame = 'latest'
    retain_overwritten_frames = False
    if retain_overwritten_frames:
        retainFolder = f'{batchFolder}/retained'
        createPath(retainFolder)


    skip_step_ratio = int(frames_skip_steps.rstrip("%")) / 100
    calc_frames_skip_steps = math.floor(steps * skip_step_ratio)

    if steps <= calc_frames_skip_steps:
        sys.exit("ERROR: You can't skip more steps than your total steps")

    if resume_run:
        if run_to_resume == 'latest':
            try:
                batchNum
            except:
                batchNum = len(glob(f"{batchFolder}/{batch_name}(*)_settings.txt"))-1
        else:
            batchNum = int(run_to_resume)
        if resume_from_frame == 'latest':
            start_frame = len(glob(batchFolder+f"/{batch_name}({batchNum})_*.png"))
        else:
            start_frame = int(resume_from_frame)+1
            if retain_overwritten_frames is True:
                existing_frames = len(glob(batchFolder+f"/{batch_name}({batchNum})_*.png"))
                frames_to_save = existing_frames - start_frame
                print(f'Moving {frames_to_save} frames to the Retained folder')
                move_files(start_frame, existing_frames, batchFolder, retainFolder)
    else:
        start_frame = 0
        batchNum = len(glob(batchFolder+"/*.txt"))
        while os.path.isfile(f"{batchFolder}/{batch_name}({batchNum})_settings.txt") or os.path.isfile(f"{batchFolder}/{batch_name}-{batchNum}_settings.txt"):
            batchNum += 1

    print(f'Starting Run: {batch_name}({batchNum}) at frame {start_frame}')

    if set_seed == 'random_seed':
        random.seed()
        seed = random.randint(0, 2**32)
        print(f'Using random seed: {seed}')
    else:
        seed = int(set_seed)

    run_config = apply_runtime_overrides(
        run_config,
        generation_mode="3D_latent",
        seed=seed,
    )
    print(
        f"[config] profile={run_config.profile} mode={run_config.generation_mode} "
        f"device={run_config.device} output_dir={run_config.output_dir} seed={run_config.seed}",
        flush=True,
    )

    from config import build_run_args_namespace

    run_config = run_config.with_runtime_values(locals())
    args = build_run_args_namespace(run_config)

    print('Prepping model')
    latent_backend = LatentDiffusionBackend(device=device, models_root=model_path)

    import run as _main_pub

    _main_pub.do_run = do_run

    try:
        do_run()
    except KeyboardInterrupt:
        pass
    except RuntimeError as exc:
        from platform.cuda import format_cuda_oom_hint

        if DEVICE.type == "cuda" and "out of memory" in str(exc).lower():
            print(format_cuda_oom_hint(), file=sys.stderr)
        raise
    finally:
        print('Seed used:', seed)
        gc.collect()
        torch.cuda.empty_cache()




    from image.ffmpeg_utils import encode_numbered_png_sequence_h264

    # Create video
    # Video file will save in the same folder as your images.

    latest_run = batchNum

    folder = batch_name
    run = latest_run
    final_frame = 'final_frame'


    init_frame = 1
    last_frame = final_frame
    fps = 25

    if last_frame == 'final_frame':
        last_frame = len(glob(batchFolder+f"/{folder}({run})_*.png"))
        print(f'Total frames: {last_frame}')

    image_path = f"{outputDirPath}/{folder}/{folder}({run})_%04d.png"
    filepath = f"{outputDirPath}/{folder}/{folder}({run}).mp4"


    encode_numbered_png_sequence_h264(
        cwd=f"{batchFolder}",
        image_sequence_pattern=image_path,
        output_path=filepath,
        fps=fps,
        start_number=init_frame,
        frames_v=last_frame + 1,
    )

    # Create per-channel 3D debug videos if the 3d folder exists and has frames
    if os.path.isdir(debug3dFolder):
        _3d_channels = ["warped", "source", "depth_blended", "depth_backend", "depth_midas", "flow_field"]
        for ch in _3d_channels:
            pattern = os.path.join(debug3dFolder, f"frame_%04d_{ch}.png")
            # Count matching frames to see if this channel was produced
            n_ch = len(glob(os.path.join(debug3dFolder, f"frame_*_{ch}.png")))
            if n_ch == 0:
                continue
            out_mp4 = os.path.join(debug3dFolder, f"{folder}({run})_3d_{ch}.mp4")
            encode_numbered_png_sequence_h264(
                cwd=debug3dFolder,
                image_sequence_pattern=pattern,
                output_path=out_mp4,
                fps=fps,
                start_number=0,
                frames_v=n_ch,
            )
            print(f"[3D] {ch} video → {out_mp4}")


if __name__ == "__main__":
    main()

