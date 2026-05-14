"""Diffusion application runtime (notebook-style script body; launched via repo `disco.py`)."""
from __future__ import annotations

def main(cli_overrides: dict | None = None) -> None:
    import subprocess
    import os
    import sys

    print("[discodiff] Runtime starting.", flush=True)

    from .platform.device import warn_if_unsupported_platform

    warn_if_unsupported_platform()

    import pathlib
    import shutil

    from .assets import (
        createPath,
        download_model,
        fetch,
        get_model_filename,
        gitclone,
        wget,
    )

    # If running locally, there's a good chance your env will need this in order to not crash upon np.matmul() or similar operations.
    os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'


    ROOT_PATH = os.getcwd()

    USE_CPU = False 

    PROJECT_DIR = os.path.abspath(os.getcwd())
    from .config import RunConfig, apply_runtime_overrides

    run_config = RunConfig.from_env(ROOT_PATH)
    if cli_overrides:
        run_config = apply_runtime_overrides(
            run_config,
            device=cli_overrides.get("device", run_config.device),
            profile=cli_overrides.get("profile", run_config.profile),
        )

    # System-impacting runtime configuration (consolidated)
    USE_ADABINS = True

    USE_SECONDARY_DIFFUSION_MODEL = True

    diffusion_model = "512x512_diffusion_uncond_finetune_008100"
    diffusion_sampling_mode = 'ddim' # ['plms','ddim']
    custom_path = 'xyz/ddpm/ema_0.9999_058000.pt'
    check_model_SHA = False

    use_checkpoint = True
    ViTB32 = True
    ViTB16 = True
    ViTL14 = False
    ViTL14_336px = False
    RN101 = False
    RN50 = True
    RN50x4 = False
    RN50x16 = False
    RN50x64 = False

    batch_name = 'example'
    steps = 100 # [25,50,100,150,250,500,1000]
    width_height_for_512x512_models = [512, 256] # [1280, 768]
    width_height_for_256x256_models = [512, 448]

    clip_guidance_scale = 750
    tv_scale = 150
    range_scale = 150
    sat_scale = 0
    cutn_batches = 4
    cutn = 16
    skip_augs = False

    video_init_steps = 100 # [25,50,100,150,250,500,1000]
    video_init_clip_guidance_scale = 1000
    video_init_tv_scale = 0.1
    video_init_range_scale = 150
    video_init_sat_scale = 300
    video_init_cutn_batches = 4
    video_init_skip_steps = 50

    init_image = None
    init_scale = 1000
    skip_steps = 10

    GENERATION_MODE = run_config.generation_mode  # Literal: "None" | "2D" | "3D" | "3D_latent" | "Video Input"

    video_init_path = "init.mp4"
    extract_nth_frame = 2
    persistent_frame_output_in_batch_folder = True
    video_init_seed_continuity = False
    video_init_flow_warp = True
    video_init_flow_blend = 0.999 # 0 - take next frame, 1 - take prev warped frame
    video_init_check_consistency = False
    video_init_blend_mode = "optical flow" # ['None', 'linear', 'optical flow']

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

    video_init_frames_scale = 15000
    video_init_frames_skip_steps = '70%'

    perlin_init = False
    perlin_mode = 'mixed' # ['mixed', 'color', 'gray']
    set_seed = 'random_seed'
    eta = 0.8
    clamp_grad = True
    clamp_max = 0.05

    randomize_class = True
    clip_denoised = False
    fuzzy_prompt = False
    rand_mag = 0.05

    use_vertical_symmetry = False
    use_horizontal_symmetry = False
    transformation_percent = [0.09]

    display_rate = 20
    n_batches = 50

    if GENERATION_MODE == 'Video Input':
        steps = video_init_steps



    inputDirPath = f'{ROOT_PATH}/input'
    createPath(inputDirPath)
    outputDirPath = str(run_config.output_dir)
    createPath(outputDirPath)

    model_path = f'{ROOT_PATH}/models'
    createPath(model_path)



    try:
        from CLIP import clip
    except ImportError:
        if not os.path.exists("CLIP"):
            gitclone("https://github.com/openai/CLIP")
        sys.path.append(f'{PROJECT_DIR}/CLIP')
        from CLIP import clip

    # guided-diffusion is vendored into discodiff as an internal module.
    from .guided_diffusion.script_util import create_model_and_diffusion, model_and_diffusion_defaults

    from .guidance.clip_cuts import MakeCutouts, MakeCutoutsDango, range_loss, spherical_dist_loss, tv_loss
    from .diffusion import GuidedDiffusionBackend, LatentDiffusionBackend, timestep_after_skip

    # Package helpers (geometry.warp, config.keyframes) — no upstream clone.
    sys.path.append(PROJECT_DIR)

    import torch
    from torch import nn
    from torch.nn import functional as F

    import torchvision.transforms as T
    import torchvision.transforms.functional as TF

    from contextlib import nullcontext
    from dataclasses import dataclass
    from functools import partial
    import cv2
    import gc
    import math
    import lpips
    from PIL import Image
    from glob import glob
    import json
    from tqdm import tqdm, trange
    from datetime import datetime
    import numpy as np
    import random
    import warnings

    os.chdir(PROJECT_DIR)
    warnings.filterwarnings("ignore", category=UserWarning)

    # AdaBins — single mirror: Hugging Face `deforum/AdaBins` (see README if wget fails).
    _ADABINS_NYU_URL = "https://huggingface.co/deforum/AdaBins/resolve/main/AdaBins_nyu.pt"
    if USE_ADABINS:
        try:
            from infer import InferenceHelper
        except:
            if not os.path.exists("AdaBins"):
                gitclone("https://github.com/shariqfarooq123/AdaBins.git")
            _adabins_pt = f'{PROJECT_DIR}/models/AdaBins_nyu.pt'
            if not os.path.exists(_adabins_pt):
                createPath(f'{PROJECT_DIR}/models')
                print(f"Downloading AdaBins_nyu.pt from Hugging Face")
                _r = subprocess.run(
                    ['wget', '-O', _adabins_pt, _ADABINS_NYU_URL],
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                )
                _ok = (
                    _r.returncode == 0
                    and os.path.exists(_adabins_pt)
                    and os.path.getsize(_adabins_pt) > 100_000_000
                )
                if not _ok:
                    if os.path.exists(_adabins_pt):
                        try:
                            os.remove(_adabins_pt)
                        except OSError:
                            pass
                    warnings.warn(
                        "AdaBins_nyu.pt could not be downloaded automatically. Get it from the Hugging Face "
                        "repository and save it as:\n"
                        f"  {_adabins_pt}\n"
                        "Direct file: https://huggingface.co/deforum/AdaBins/resolve/main/AdaBins_nyu.pt\n"
                        "Repo browser: https://huggingface.co/deforum/AdaBins\n"
                        "To disable AdaBins depth helpers, set USE_ADABINS = False near the top of src/discodiff/main.py.",
                        RuntimeWarning,
                        stacklevel=1,
                    )
                    raise RuntimeError(
                        "AdaBins_nyu.pt missing after download failure; see warning above or set USE_ADABINS = False."
                    )
            sys.path.append(f'{PROJECT_DIR}/AdaBins')
        try:
            from infer import InferenceHelper
        except ModuleNotFoundError as exc:
            raise RuntimeError(
                "AdaBins requires additional Python packages that are not installed "
                f"(missing: {exc.name}). Run `uv sync` and retry."
            ) from exc
        MAX_ADABINS_AREA = 500000

    from .platform.device import apply_backend_defaults, log_device_selection, resolve_runtime_device

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

    def init_adabins_depth_helper():
        print("Initializing AdaBins depth helper")
        adabins_helper = InferenceHelper(dataset='nyu', device=DEVICE)
        print("AdaBins depth helper initialized.")
        return adabins_helper




    from .geometry import py3d_tools as p3dT
    from .geometry import warp as dxf

    from .image import noise as _noise


    def parse_prompt(prompt):
        if prompt.startswith('http://') or prompt.startswith('https://'):
            vals = prompt.rsplit(':', 2)
            vals = [vals[0] + ':' + vals[1], *vals[2:]]
        else:
            vals = prompt.rsplit(':', 1)
        vals = vals + ['', '1'][len(vals):]
        return vals[0], float(vals[1])

    stop_on_next_loop = False  # Make sure GPU memory doesn't get corrupted from cancelling the run mid-way through, allow a full frame to complete
    TRANSLATION_SCALE = 1.0/200.0
    stabilization_warmup_frames = max(24, int(turbo_preroll))
    effective_turbo_preroll = stabilization_warmup_frames

    def _smoothstep(value: float) -> float:
        value = max(0.0, min(1.0, value))
        return value * value * (3.0 - 2.0 * value)

    def _stabilization_progress(frame_index: int) -> float:
        if stabilization_warmup_frames <= 0:
            return 1.0
        return _smoothstep(frame_index / float(stabilization_warmup_frames))


    def do_3d_step(img_filepath, frame_num, adabins_helper):
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
            -translation_z * TRANSLATION_SCALE * motion_scale,
        ]
        rotate_xyz_degrees = [
            rotation_3d_x * motion_scale,
            rotation_3d_y * motion_scale,
            rotation_3d_z * motion_scale,
        ]
        rotate_xyz = [math.radians(rotate_xyz_degrees[0]), math.radians(rotate_xyz_degrees[1]), math.radians(rotate_xyz_degrees[2])]
        rot_mat = p3dT.euler_angles_to_matrix(torch.tensor(rotate_xyz, device=device), "XYZ").unsqueeze(0)
        next_step_pil = dxf.transform_image_3d(img_filepath, adabins_helper, DEVICE,
                               rot_mat, translate_xyz, args.near_plane, args.far_plane,
                               args.fov, padding_mode=args.padding_mode,
                               sampling_mode=args.sampling_mode,
                               debug_dir=debug3dFolder, frame_num=frame_num)
        return next_step_pil

    def symmetry_transformation_fn(x):
        if args.use_horizontal_symmetry:
            [n, c, h, w] = x.size()
            x = torch.concat((x[:, :, :, :w//2], torch.flip(x[:, :, :, :w//2], [-1])), -1)
            print("horizontal symmetry applied")
        if args.use_vertical_symmetry:
            [n, c, h, w] = x.size()
            x = torch.concat((x[:, :, :h//2, :], torch.flip(x[:, :, :h//2, :], [-2])), -2)
            print("vertical symmetry applied")
        return x

    def do_run():
      from .platform.cuda import use_cudnn_benchmark_mode

      _cudnn_benchmark = use_cudnn_benchmark_mode()
      seed = args.seed
      print(range(args.start_frame, args.max_frames))

      adabins_helper = None
      if args.animation_mode in ("3D", "3D_latent"):
          adabins_helper = init_adabins_depth_helper()
      if args.animation_mode in ("3D", "3D_latent") and turbo_mode:
          print(
              f"[turbo] steps={args.steps} turbo_steps={int(turbo_steps)} "
              f"frames_skip_steps={frames_skip_steps} calc_skip_steps={args.calc_frames_skip_steps} "
              f"effective_diffusion_steps={args.steps - args.calc_frames_skip_steps}"
          )
      for frame_num in range(args.start_frame, args.max_frames):
          if stop_on_next_loop:
            break

          # Print Frame progress if animation mode is on
          if args.animation_mode != "None":
            batchBar = tqdm(range(args.max_frames), desc ="Frames")
            batchBar.n = frame_num
            batchBar.refresh()


          # Inits if not video frames
          if args.animation_mode != "Video Input":
            if args.init_image in ['','none', 'None', 'NONE']:
              init_image = None
            else:
              init_image = args.init_image
            init_scale = args.init_scale
            skip_steps = args.skip_steps

          if args.animation_mode == "2D":
            if args.key_frames:
              angle = args.angle_series[frame_num]
              zoom = args.zoom_series[frame_num]
              translation_x = args.translation_x_series[frame_num]
              translation_y = args.translation_y_series[frame_num]
              print(
                  f'angle: {angle}',
                  f'zoom: {zoom}',
                  f'translation_x: {translation_x}',
                  f'translation_y: {translation_y}',
              )

            if frame_num > 0:
              seed += 1
              if resume_run and frame_num == start_frame:
                img_0 = cv2.imread(batchFolder+f"/{batch_name}({batchNum})_{start_frame-1:04}.png")
              else:
                img_0 = cv2.imread(prev_frame_path)
              center = (1*img_0.shape[1]//2, 1*img_0.shape[0]//2)
              trans_mat = np.float32(
                  [[1, 0, translation_x],
                  [0, 1, translation_y]]
              )
              rot_mat = cv2.getRotationMatrix2D( center, angle, zoom )
              trans_mat = np.vstack([trans_mat, [0,0,1]])
              rot_mat = np.vstack([rot_mat, [0,0,1]])
              transformation_matrix = np.matmul(rot_mat, trans_mat)
              img_0 = cv2.warpPerspective(
                  img_0,
                  transformation_matrix,
                  (img_0.shape[1], img_0.shape[0]),
                  borderMode=cv2.BORDER_WRAP
              )

              cv2.imwrite(prev_frame_scaled_path, img_0)
              init_image = prev_frame_scaled_path
              init_scale = args.frames_scale
              skip_steps = args.calc_frames_skip_steps

          if args.animation_mode in ("3D", "3D_latent"):
                        if frame_num > 0:
                            seed += 1
                            if resume_run and frame_num == start_frame:
                                img_filepath = batchFolder + f"/{batch_name}({batchNum})_{start_frame-1:04}.png"
                                if turbo_mode and frame_num > effective_turbo_preroll:
                                    shutil.copyfile(img_filepath, old_frame_scaled_path)
                            else:
                                img_filepath = prev_frame_path

                            next_step_pil = do_3d_step(img_filepath, frame_num, adabins_helper)
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
                                    if args.animation_mode == "3D_latent":
                                        # Reuse the newly warped frame in latent mode to avoid a second AdaBins+reprojection pass.
                                        old_frame = next_step_pil
                                    else:
                                        old_frame = do_3d_step(old_frame_scaled_path, frame_num, adabins_helper)
                                    old_frame.save(old_frame_scaled_path)
                                    if frame_num % int(turbo_steps) != 0:
                                        print('turbo skip this frame: skipping clip diffusion steps')
                                        filename = f'{args.batch_name}({args.batchNum})_{frame_num:04}.png'
                                        blend_factor = ((frame_num % int(turbo_steps)) + 1) / int(turbo_steps)
                                        print('turbo skip this frame: skipping clip diffusion steps and saving blended frame')
                                        newWarpedImg = cv2.imread(prev_frame_scaled_path)  # this is already updated..
                                        oldWarpedImg = cv2.imread(old_frame_scaled_path)
                                        blendedImage = cv2.addWeighted(newWarpedImg, blend_factor, oldWarpedImg, 1 - blend_factor, 0.0)
                                        cv2.imwrite(f'{batchFolder}/{filename}', blendedImage)
                                        next_step_pil.save(f'{img_filepath}')  # save it also as prev_frame to feed next iteration
                                        continue
                                    else:
                                        # if not a skip frame, will run diffusion and need to blend.
                                        oldWarpedImg = cv2.imread(prev_frame_scaled_path)
                                        cv2.imwrite(old_frame_scaled_path, oldWarpedImg)  # swap in for blending later
                                        print('clip/diff this frame - generate clip diff image')

                            init_image = prev_frame_scaled_path
                            warmup_progress = _stabilization_progress(frame_num)
                            init_scale = int(round(args.frames_scale * (1.15 - 0.15 * warmup_progress)))
                            skip_steps = min(
                                    args.steps - 1,
                                    int(round(args.calc_frames_skip_steps + (10.0 * (1.0 - warmup_progress))))
                            )

          if  args.animation_mode == "Video Input":
            init_scale = args.video_init_frames_scale
            skip_steps = args.calc_frames_skip_steps
            if not video_init_seed_continuity:
              seed += 1
            if video_init_flow_warp:
              if frame_num == 0: 
                skip_steps = args.video_init_skip_steps
                init_image = f'{videoFramesFolder}/{frame_num+1:04}.jpg'
              if frame_num > 0: 
                prev = Image.open(batchFolder+f"/{batch_name}({batchNum})_{frame_num-1:04}.png")

                frame1_path = f'{videoFramesFolder}/{frame_num:04}.jpg'
                frame2 = Image.open(f'{videoFramesFolder}/{frame_num+1:04}.jpg')
                flo_path = f"/{flo_folder}/{frame1_path.split('/')[-1]}.npy"

                init_image = 'warped.png'
                print(video_init_flow_blend)
                weights_path = None
                if video_init_check_consistency:
                    # TBD
                    pass

                warp(prev, frame2, flo_path, blend=video_init_flow_blend, weights_path=weights_path).save(init_image)

            else:
              init_image = f'{videoFramesFolder}/{frame_num+1:04}.jpg'


          loss_values = []

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

          target_embeds, weights = [], []

          if args.prompts_series is not None and frame_num >= len(args.prompts_series):
            frame_prompt = args.prompts_series[-1]
          elif args.prompts_series is not None:
            frame_prompt = args.prompts_series[frame_num]
          else:
            frame_prompt = []

          if args.image_prompts_series is not None and frame_num >= len(args.image_prompts_series):
            image_prompt = args.image_prompts_series[-1]
          elif args.image_prompts_series is not None:
            image_prompt = args.image_prompts_series[frame_num]
          else:
            image_prompt = []

          # Only print at keyframes with new prompts
          if frame_num == 0 or (args.prompts_series is not None and frame_num < len(args.prompts_series) and args.prompts_series[frame_num] != args.prompts_series[frame_num - 1]):
            if frame_prompt:
              print(f'Frame {frame_num} Text Prompt: {frame_prompt}')
          if frame_num == 0 or (args.image_prompts_series is not None and frame_num < len(args.image_prompts_series) and args.image_prompts_series[frame_num] != args.image_prompts_series[frame_num - 1]):
            if image_prompt:
              print(f'Frame {frame_num} Image Prompt: {image_prompt}')

          model_stats = []
          for clip_model in clip_models:
                model_stat = {"clip_model":None,"target_embeds":[],"make_cutouts":None,"weights":[]}
                model_stat["clip_model"] = clip_model

                for prompt in frame_prompt:
                    txt, weight = parse_prompt(prompt)
                    txt = clip_model.encode_text(clip.tokenize(prompt).to(device)).float()

                    if args.fuzzy_prompt:
                        for i in range(25):
                            model_stat["target_embeds"].append((txt + torch.randn(txt.shape, device=device, dtype=txt.dtype) * args.rand_mag).clamp(0,1))
                            model_stat["weights"].append(weight)
                    else:
                        model_stat["target_embeds"].append(txt)
                        model_stat["weights"].append(weight)

                if image_prompt:
                  model_stat["make_cutouts"] = MakeCutouts(clip_model.visual.input_resolution, cutn, skip_augs=skip_augs) 
                  for prompt in image_prompt:
                      path, weight = parse_prompt(prompt)
                      img = Image.open(fetch(path)).convert('RGB')
                      img = TF.resize(img, min(side_x, side_y, *img.size), T.InterpolationMode.LANCZOS)
                      batch = model_stat["make_cutouts"](TF.to_tensor(img).to(device).unsqueeze(0).mul(2).sub(1))
                      embed = clip_model.encode_image(normalize(batch)).float()
                      if fuzzy_prompt:
                          for i in range(25):
                              model_stat["target_embeds"].append((embed + torch.randn(embed.shape, device=device, dtype=embed.dtype) * rand_mag).clamp(0,1))
                              weights.extend([weight / cutn] * cutn)
                      else:
                          model_stat["target_embeds"].append(embed)
                          model_stat["weights"].extend([weight / cutn] * cutn)

                model_stat["target_embeds"] = torch.cat(model_stat["target_embeds"])
                model_stat["weights"] = torch.tensor(model_stat["weights"], device=device)
                if model_stat["weights"].sum().abs() < 1e-3:
                    raise RuntimeError('The weights must not sum to 0.')
                model_stat["weights"] /= model_stat["weights"].sum().abs()
                model_stats.append(model_stat)

          init = None
          init_pil = None
          if init_image is not None:
              init_pil = Image.open(fetch(init_image)).convert('RGB')
              init_pil = init_pil.resize((args.side_x, args.side_y), Image.LANCZOS)
              if args.animation_mode != "3D_latent":
                  init = TF.to_tensor(init_pil).to(device).unsqueeze(0).mul(2).sub(1)
          elif args.animation_mode == "3D_latent":
              init_pil = Image.new("RGB", (args.side_x, args.side_y), color=(0, 0, 0))

          if args.perlin_init and args.animation_mode != "3D_latent":
              if args.perlin_mode == 'color':
                  init = _noise.create_perlin_noise([1.5**-i*0.5 for i in range(12)], 1, 1, False, side_x, side_y, device)
                  init2 = _noise.create_perlin_noise([1.5**-i*0.5 for i in range(8)], 4, 4, False, side_x, side_y, device)
              elif args.perlin_mode == 'gray':
                init = _noise.create_perlin_noise([1.5**-i*0.5 for i in range(12)], 1, 1, True, side_x, side_y, device)
                init2 = _noise.create_perlin_noise([1.5**-i*0.5 for i in range(8)], 4, 4, True, side_x, side_y, device)
              else:
                init = _noise.create_perlin_noise([1.5**-i*0.5 for i in range(12)], 1, 1, False, side_x, side_y, device)
                init2 = _noise.create_perlin_noise([1.5**-i*0.5 for i in range(8)], 4, 4, True, side_x, side_y, device)
              # init = TF.to_tensor(init).add(TF.to_tensor(init2)).div(2).to(device)
              init = TF.to_tensor(init).add(TF.to_tensor(init2)).div(2).to(device).unsqueeze(0).mul(2).sub(1)
              del init2

          cur_t = None

          def cond_fn(x, t, y=None):
              with torch.enable_grad():
                  x_is_NaN = False
                  x = x.detach().requires_grad_()
                  n = x.shape[0]
                  if USE_SECONDARY_DIFFUSION_MODEL is True:
                    alpha = torch.tensor(diffusion.sqrt_alphas_cumprod[cur_t], device=device, dtype=torch.float32)
                    sigma = torch.tensor(diffusion.sqrt_one_minus_alphas_cumprod[cur_t], device=device, dtype=torch.float32)
                    cosine_t = alpha_sigma_to_t(alpha, sigma)
                    out = secondary_model(x, cosine_t[None].repeat([n])).pred
                    fac = diffusion.sqrt_one_minus_alphas_cumprod[cur_t]
                    x_in = out * fac + x * (1 - fac)
                    x_in_grad = torch.zeros_like(x_in)
                  else:
                    my_t = torch.ones([n], device=device, dtype=torch.long) * cur_t
                    out = diffusion.p_mean_variance(model, x, my_t, clip_denoised=False, model_kwargs={'y': y})
                    fac = diffusion.sqrt_one_minus_alphas_cumprod[cur_t]
                    x_in = out['pred_xstart'] * fac + x * (1 - fac)
                    x_in_grad = torch.zeros_like(x_in)
                  for model_stat in model_stats:
                    for i in range(args.cutn_batches):
                        t_int = int(t.item())+1 # errors on last step without +1, need to find source
                        # when using SLIP Base model the dimensions need to be hard coded to avoid AttributeError: 'VisionTransformer' object has no attribute 'input_resolution'
                        try:
                            input_resolution=model_stat["clip_model"].visual.input_resolution
                        except:
                            input_resolution=224

                        cuts = MakeCutoutsDango(
                                input_resolution,
                                Overview= args.cut_overview[1000-t_int], 
                                InnerCrop = args.cut_innercut[1000-t_int],
                                IC_Size_Pow=args.cut_ic_pow[1000-t_int],
                                IC_Grey_P = args.cut_icgray_p[1000-t_int],
                                animation_mode=args.animation_mode,
                                skip_augs=skip_augs,
                                )
                        clip_in = normalize(cuts(x_in.add(1).div(2)))
                        image_embeds = model_stat["clip_model"].encode_image(clip_in).float()
                        dists = spherical_dist_loss(image_embeds.unsqueeze(1), model_stat["target_embeds"].unsqueeze(0))
                        dists = dists.view([args.cut_overview[1000-t_int]+args.cut_innercut[1000-t_int], n, -1])
                        losses = dists.mul(model_stat["weights"]).sum(2).mean(0)
                        loss_values.append(losses.sum().item()) # log loss, probably shouldn't do per cutn_batch
                        x_in_grad += torch.autograd.grad(losses.sum() * clip_guidance_scale, x_in)[0] / cutn_batches
                  tv_losses = tv_loss(x_in)
                  if USE_SECONDARY_DIFFUSION_MODEL is True:
                    range_losses = range_loss(out)
                  else:
                    range_losses = range_loss(out['pred_xstart'])
                  sat_losses = torch.abs(x_in - x_in.clamp(min=-1,max=1)).mean()
                  loss = tv_losses.sum() * tv_scale + range_losses.sum() * range_scale + sat_losses.sum() * sat_scale
                  if init is not None and init_scale:
                      init_losses = lpips_model(x_in, init)
                      loss = loss + init_losses.sum() * init_scale
                  x_in_grad += torch.autograd.grad(loss, x_in)[0]
                  if torch.isnan(x_in_grad).any()==False:
                      grad = -torch.autograd.grad(x_in, x, x_in_grad)[0]
                  else:
                    # print("NaN'd")
                    x_is_NaN = True
                    grad = torch.zeros_like(x)
              if args.clamp_grad and x_is_NaN == False:
                  magnitude = grad.square().mean().sqrt()
                  return grad * magnitude.clamp(max=args.clamp_max) / magnitude  #min=-0.02, min=-clamp_max, 
              return grad

          image_display = nullcontext()
          for i in range(args.n_batches):
              if args.animation_mode == 'None':
                batchBar = tqdm(range(args.n_batches), desc ="Batches")
                batchBar.n = i
                batchBar.refresh()
              print('')
              gc.collect()
              torch.cuda.empty_cache()
              is_latent_backend = args.animation_mode == "3D_latent"
              if is_latent_backend:
                  cur_t = -1
                  total_steps = 1
              else:
                  cur_t = timestep_after_skip(diffusion, skip_steps)
                  total_steps = cur_t

              if perlin_init and not is_latent_backend:
                  init = _noise.regen_perlin(perlin_mode, side_x, side_y, device, batch_size)

              if is_latent_backend:
                  save_num = f'{frame_num:04}' if GENERATION_MODE != "None" else i
                  filename = f'{args.batch_name}({args.batchNum})_{save_num}.png'
                  prompt_state = latent_backend.prepare(frame_prompt, seed, (args.side_x, args.side_y))
                  strength = float(skip_steps) / max(float(args.steps), 1.0)
                  image = latent_backend.generate(
                      init_image=init_pil,
                      strength_or_skip=strength,
                      steps=args.steps,
                      guidance_scale=clip_guidance_scale,
                      extra_guidance_state={"prompt_state": prompt_state},
                  )
                  image.save(progress_path)
                  if frame_num == 0:
                      save_settings()
                  if args.animation_mode != "None":
                      image.save(prev_frame_path)
                  image.save(f'{batchFolder}/{filename}')
                  if args.animation_mode in ("3D", "3D_latent"):
                      if turbo_mode and frame_num > 0:
                          blend_factor = 1.0 / int(turbo_steps)
                          newFrame = cv2.imread(prev_frame_path)
                          prev_frame_warped = cv2.imread(prev_frame_scaled_path)
                          blendedImage = cv2.addWeighted(newFrame, blend_factor, prev_frame_warped, (1-blend_factor), 0.0)
                          cv2.imwrite(f'{batchFolder}/{filename}', blendedImage)
                  continue

              samples = guided_backend.generate(
                  init_image=init,
                  strength_or_skip=skip_steps,
                  steps=args.steps,
                  guidance_scale=clip_guidance_scale,
                  extra_guidance_state={"cond_fn": cond_fn},
              )


              for j, sample in enumerate(samples):    
                cur_t -= 1
                intermediateStep = False
                if args.steps_per_checkpoint is not None:
                    if j % args.steps_per_checkpoint == 0 and j > 0:
                      intermediateStep = True
                elif j in args.intermediate_saves:
                  intermediateStep = True
                with image_display:
                  if j % args.display_rate == 0 or cur_t == -1 or intermediateStep == True:
                      for k, image in enumerate(sample['pred_xstart']):
                          # tqdm.write(f'Batch {i}, step {j}, output {k}:')
                          current_time = datetime.now().strftime('%y%m%d-%H%M%S_%f')
                          percent = math.ceil(j/total_steps*100)
                          if args.n_batches > 0:
                            #if intermediates are saved to the subfolder, don't append a step or percentage to the name
                            if cur_t == -1 and args.intermediates_in_subfolder is True:
                              save_num = f'{frame_num:04}' if GENERATION_MODE != "None" else i
                              filename = f'{args.batch_name}({args.batchNum})_{save_num}.png'
                            else:
                              #If we're working with percentages, append it
                              if args.steps_per_checkpoint is not None:
                                filename = f'{args.batch_name}({args.batchNum})_{i:04}-{percent:02}%.png'
                              # Or else, iIf we're working with specific steps, append those
                              else:
                                filename = f'{args.batch_name}({args.batchNum})_{i:04}-{j:03}.png'
                          image = TF.to_pil_image(image.add(1).div(2).clamp(0, 1))
                          if j % args.display_rate == 0 or cur_t == -1:
                            image.save(progress_path)
                          if args.steps_per_checkpoint is not None:
                            if j % args.steps_per_checkpoint == 0 and j > 0:
                              if args.intermediates_in_subfolder is True:
                                image.save(f'{partialFolder}/{filename}')
                              else:
                                image.save(f'{batchFolder}/{filename}')
                          else:
                            if j in args.intermediate_saves:
                              if args.intermediates_in_subfolder is True:
                                image.save(f'{partialFolder}/{filename}')
                              else:
                                image.save(f'{batchFolder}/{filename}')
                          if cur_t == -1:
                            if frame_num == 0:
                              save_settings()
                            if args.animation_mode != "None":
                              image.save(prev_frame_path)
                            image.save(f'{batchFolder}/{filename}')
                            if args.animation_mode in ("3D", "3D_latent"):
                                # Match notebook behavior: blend only for the saved output frame,
                                # but keep prev_frame_path as the newly diffused frame.
                                if turbo_mode and frame_num > 0:
                                    blend_factor = 1.0 / int(turbo_steps)
                                    newFrame = cv2.imread(prev_frame_path) # This is already updated..
                                    prev_frame_warped = cv2.imread(prev_frame_scaled_path)
                                    blendedImage = cv2.addWeighted(newFrame, blend_factor, prev_frame_warped, (1-blend_factor), 0.0)
                                    cv2.imwrite(f'{batchFolder}/{filename}', blendedImage)
                                else:
                                    image.save(f'{batchFolder}/{filename}')




    def save_settings():
        setting_list = {
          'text_prompts': text_prompts,
          'image_prompts': image_prompts,
          'clip_guidance_scale': clip_guidance_scale,
          'tv_scale': tv_scale,
          'range_scale': range_scale,
          'sat_scale': sat_scale,
          # 'cutn': cutn,
          'cutn_batches': cutn_batches,
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
          'skip_augs': skip_augs,
          'randomize_class': randomize_class,
          'clip_denoised': clip_denoised,
          'clamp_grad': clamp_grad,
          'clamp_max': clamp_max,
          'seed': seed,
          'fuzzy_prompt': fuzzy_prompt,
          'rand_mag': rand_mag,
          'eta': eta,
          'width': width_height[0],
          'height': width_height[1],
          'diffusion_model': diffusion_model,
          'use_secondary_model': USE_SECONDARY_DIFFUSION_MODEL,
          'steps': steps,
          'diffusion_steps': diffusion_steps,
          'diffusion_sampling_mode': diffusion_sampling_mode,
          'ViTB32': ViTB32,
          'ViTB16': ViTB16,
          'ViTL14': ViTL14,
          'ViTL14_336px': ViTL14_336px,
          'RN101': RN101,
          'RN50': RN50,
          'RN50x4': RN50x4,
          'RN50x16': RN50x16,
          'RN50x64': RN50x64,
          'cut_overview': str(cut_overview),
          'cut_innercut': str(cut_innercut),
          'cut_ic_pow': str(cut_ic_pow),
          'cut_icgray_p': str(cut_icgray_p),
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
          'video_init_path':video_init_path,
          'extract_nth_frame':extract_nth_frame,
          'video_init_seed_continuity': video_init_seed_continuity,
          'turbo_mode':turbo_mode,
          'turbo_steps':turbo_steps,
          'turbo_preroll':turbo_preroll,
          'use_horizontal_symmetry':use_horizontal_symmetry,
          'use_vertical_symmetry':use_vertical_symmetry,
          'transformation_percent':transformation_percent,
          #video init settings
          'video_init_steps': video_init_steps,
          'video_init_clip_guidance_scale': video_init_clip_guidance_scale,
          'video_init_tv_scale': video_init_tv_scale,
          'video_init_range_scale': video_init_range_scale,
          'video_init_sat_scale': video_init_sat_scale,
          'video_init_cutn_batches': video_init_cutn_batches,
          'video_init_skip_steps': video_init_skip_steps,
          'video_init_frames_scale': video_init_frames_scale,
          'video_init_frames_skip_steps': video_init_frames_skip_steps,
          #warp settings
          'video_init_flow_warp':video_init_flow_warp,
          'video_init_flow_blend':video_init_flow_blend,
          'video_init_check_consistency':video_init_check_consistency,
          'video_init_blend_mode':video_init_blend_mode
        }
        with open(f"{batchFolder}/{batch_name}({batchNum})_settings.txt", "w+", encoding="utf-8") as f:   #save settings
            json.dump(setting_list, f, ensure_ascii=False, indent=4)





    def append_dims(x, n):
        return x[(Ellipsis, *(None,) * (n - x.ndim))]


    def expand_to_planes(x, shape):
        return append_dims(x, len(shape)).repeat([1, 1, *shape[2:]])


    def alpha_sigma_to_t(alpha, sigma):
        return torch.atan2(sigma, alpha) * 2 / math.pi


    def t_to_alpha_sigma(t):
        return torch.cos(t * math.pi / 2), torch.sin(t * math.pi / 2)


    @dataclass
    class DiffusionOutput:
        v: torch.Tensor
        pred: torch.Tensor
        eps: torch.Tensor


    class ConvBlock(nn.Sequential):
        def __init__(self, c_in, c_out):
            super().__init__(
                nn.Conv2d(c_in, c_out, 3, padding=1),
                nn.ReLU(inplace=True),
            )


    class SkipBlock(nn.Module):
        def __init__(self, main, skip=None):
            super().__init__()
            self.main = nn.Sequential(*main)
            self.skip = skip if skip else nn.Identity()

        def forward(self, input):
            return torch.cat([self.main(input), self.skip(input)], dim=1)


    class FourierFeatures(nn.Module):
        def __init__(self, in_features, out_features, std=1.):
            super().__init__()
            assert out_features % 2 == 0
            self.weight = nn.Parameter(torch.randn([out_features // 2, in_features]) * std)

        def forward(self, input):
            f = 2 * math.pi * input @ self.weight.T
            return torch.cat([f.cos(), f.sin()], dim=-1)


    class SecondaryDiffusionImageNet(nn.Module):
        def __init__(self):
            super().__init__()
            c = 64  # The base channel count

            self.timestep_embed = FourierFeatures(1, 16)

            self.net = nn.Sequential(
                ConvBlock(3 + 16, c),
                ConvBlock(c, c),
                SkipBlock([
                    nn.AvgPool2d(2),
                    ConvBlock(c, c * 2),
                    ConvBlock(c * 2, c * 2),
                    SkipBlock([
                        nn.AvgPool2d(2),
                        ConvBlock(c * 2, c * 4),
                        ConvBlock(c * 4, c * 4),
                        SkipBlock([
                            nn.AvgPool2d(2),
                            ConvBlock(c * 4, c * 8),
                            ConvBlock(c * 8, c * 4),
                            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
                        ]),
                        ConvBlock(c * 8, c * 4),
                        ConvBlock(c * 4, c * 2),
                        nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
                    ]),
                    ConvBlock(c * 4, c * 2),
                    ConvBlock(c * 2, c),
                    nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
                ]),
                ConvBlock(c * 2, c),
                nn.Conv2d(c, 3, 3, padding=1),
            )

        def forward(self, input, t):
            timestep_embed = expand_to_planes(self.timestep_embed(t[:, None]), input.shape)
            v = self.net(torch.cat([input, timestep_embed], dim=1))
            alphas, sigmas = map(partial(append_dims, n=v.ndim), t_to_alpha_sigma(t))
            pred = input * alphas - v * sigmas
            eps = input * sigmas + v * alphas
            return DiffusionOutput(v, pred, eps)


    class SecondaryDiffusionImageNet2(nn.Module):
        def __init__(self):
            super().__init__()
            c = 64  # The base channel count
            cs = [c, c * 2, c * 2, c * 4, c * 4, c * 8]

            self.timestep_embed = FourierFeatures(1, 16)
            self.down = nn.AvgPool2d(2)
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)

            self.net = nn.Sequential(
                ConvBlock(3 + 16, cs[0]),
                ConvBlock(cs[0], cs[0]),
                SkipBlock([
                    self.down,
                    ConvBlock(cs[0], cs[1]),
                    ConvBlock(cs[1], cs[1]),
                    SkipBlock([
                        self.down,
                        ConvBlock(cs[1], cs[2]),
                        ConvBlock(cs[2], cs[2]),
                        SkipBlock([
                            self.down,
                            ConvBlock(cs[2], cs[3]),
                            ConvBlock(cs[3], cs[3]),
                            SkipBlock([
                                self.down,
                                ConvBlock(cs[3], cs[4]),
                                ConvBlock(cs[4], cs[4]),
                                SkipBlock([
                                    self.down,
                                    ConvBlock(cs[4], cs[5]),
                                    ConvBlock(cs[5], cs[5]),
                                    ConvBlock(cs[5], cs[5]),
                                    ConvBlock(cs[5], cs[4]),
                                    self.up,
                                ]),
                                ConvBlock(cs[4] * 2, cs[4]),
                                ConvBlock(cs[4], cs[3]),
                                self.up,
                            ]),
                            ConvBlock(cs[3] * 2, cs[3]),
                            ConvBlock(cs[3], cs[2]),
                            self.up,
                        ]),
                        ConvBlock(cs[2] * 2, cs[2]),
                        ConvBlock(cs[2], cs[1]),
                        self.up,
                    ]),
                    ConvBlock(cs[1] * 2, cs[1]),
                    ConvBlock(cs[1], cs[0]),
                    self.up,
                ]),
                ConvBlock(cs[0] * 2, cs[0]),
                nn.Conv2d(cs[0], 3, 3, padding=1),
            )

        def forward(self, input, t):
            timestep_embed = expand_to_planes(self.timestep_embed(t[:, None]), input.shape)
            v = self.net(torch.cat([input, timestep_embed], dim=1))
            alphas, sigmas = map(partial(append_dims, n=v.ndim), t_to_alpha_sigma(t))
            pred = input * alphas - v * sigmas
            eps = input * sigmas + v * alphas
            return DiffusionOutput(v, pred, eps)



    """
    # 2. Diffusion and CLIP model settings
    """


    # Models Settings (note: For pixel art, the best is pixelartdiffusion_expanded)
    # ["256x256_diffusion_uncond", "512x512_diffusion_uncond_finetune_008100", "portrait_generator_v001", "pixelartdiffusion_expanded", "pixel_art_diffusion_hard_256", "pixel_art_diffusion_soft_256", "pixelartdiffusion4k", "watercolordiffusion_2", "watercolordiffusion", "PulpSciFiDiffusion", "custom"]

    diff_model_map = {
        '256x256_diffusion_uncond': { 'downloaded': False, 'sha': 'a37c32fffd316cd494cf3f35b339936debdc1576dad13fe57c42399a5dbc78b1', 'uri_list': ['https://openaipublic.blob.core.windows.net/diffusion/jul-2021/256x256_diffusion_uncond.pt', 'https://www.dropbox.com/s/9tqnqo930mpnpcn/256x256_diffusion_uncond.pt'] },
        '512x512_diffusion_uncond_finetune_008100': { 'downloaded': False, 'sha': '9c111ab89e214862b76e1fa6a1b3f1d329b1a88281885943d2cdbe357ad57648', 'uri_list': ['https://huggingface.co/lowlevelware/512x512_diffusion_unconditional_ImageNet/resolve/main/512x512_diffusion_uncond_finetune_008100.pt', 'https://the-eye.eu/public/AI/models/512x512_diffusion_unconditional_ImageNet/512x512_diffusion_uncond_finetune_008100.pt'] },
        'portrait_generator_v001': { 'downloaded': False, 'sha': 'b7e8c747af880d4480b6707006f1ace000b058dd0eac5bb13558ba3752d9b5b9', 'uri_list': ['https://huggingface.co/felipe3dartist/portrait_generator_v001/resolve/main/portrait_generator_v001_ema_0.9999_1MM.pt'] },
        'pixelartdiffusion_expanded': { 'downloaded': False, 'sha': 'a73b40556634034bf43b5a716b531b46fb1ab890634d854f5bcbbef56838739a', 'uri_list': ['https://huggingface.co/KaliYuga/PADexpanded/resolve/main/PADexpanded.pt'] },
        'pixel_art_diffusion_hard_256': { 'downloaded': False, 'sha': 'be4a9de943ec06eef32c65a1008c60ad017723a4d35dc13169c66bb322234161', 'uri_list': ['https://huggingface.co/KaliYuga/pixel_art_diffusion_hard_256/resolve/main/pixel_art_diffusion_hard_256.pt'] },
        'pixel_art_diffusion_soft_256': { 'downloaded': False, 'sha': 'd321590e46b679bf6def1f1914b47c89e762c76f19ab3e3392c8ca07c791039c', 'uri_list': ['https://huggingface.co/KaliYuga/pixel_art_diffusion_soft_256/resolve/main/pixel_art_diffusion_soft_256.pt'] },
        'pixelartdiffusion4k': { 'downloaded': False, 'sha': 'a1ba4f13f6dabb72b1064f15d8ae504d98d6192ad343572cc416deda7cccac30', 'uri_list': ['https://huggingface.co/KaliYuga/pixelartdiffusion4k/resolve/main/pixelartdiffusion4k.pt'] },
        'watercolordiffusion_2': { 'downloaded': False, 'sha': '49c281b6092c61c49b0f1f8da93af9b94be7e0c20c71e662e2aa26fee0e4b1a9', 'uri_list': ['https://huggingface.co/KaliYuga/watercolordiffusion_2/resolve/main/watercolordiffusion_2.pt'] },
        'watercolordiffusion': { 'downloaded': False, 'sha': 'a3e6522f0c8f278f90788298d66383b11ac763dd5e0d62f8252c962c23950bd6', 'uri_list': ['https://huggingface.co/KaliYuga/watercolordiffusion/resolve/main/watercolordiffusion.pt'] },
        'PulpSciFiDiffusion': { 'downloaded': False, 'sha': 'b79e62613b9f50b8a3173e5f61f0320c7dbb16efad42a92ec94d014f6e17337f', 'uri_list': ['https://huggingface.co/KaliYuga/PulpSciFiDiffusion/resolve/main/PulpSciFiDiffusion.pt'] },
        'secondary': { 'downloaded': False, 'sha': '983e3de6f95c88c81b2ca7ebb2c217933be1973b1ff058776b970f901584613a', 'uri_list': ['https://huggingface.co/spaces/huggi/secondary_model_imagenet_2.pth/resolve/main/secondary_model_imagenet_2.pth', 'https://the-eye.eu/public/AI/models/v-diffusion/secondary_model_imagenet_2.pth', 'https://ipfs.pollinations.ai/ipfs/bafybeibaawhhk7fhyhvmm7x24zwwkeuocuizbqbcg5nqx64jq42j75rdiy/secondary_model_imagenet_2.pth'] },
    }

    kaliyuga_pixel_art_model_names = ['pixelartdiffusion_expanded', 'pixel_art_diffusion_hard_256', 'pixel_art_diffusion_soft_256', 'pixelartdiffusion4k', 'PulpSciFiDiffusion']
    kaliyuga_watercolor_model_names = ['watercolordiffusion', 'watercolordiffusion_2']
    kaliyuga_pulpscifi_model_names = ['PulpSciFiDiffusion']
    diffusion_models_256x256_list = ['256x256_diffusion_uncond'] + kaliyuga_pixel_art_model_names + kaliyuga_watercolor_model_names + kaliyuga_pulpscifi_model_names

    # Download the diffusion model(s)
    download_model(diffusion_model, diff_model_map, model_path, check_model_SHA)

    if USE_SECONDARY_DIFFUSION_MODEL:
        download_model('secondary', diff_model_map, model_path, check_model_SHA)


    model_config = model_and_diffusion_defaults()
    if diffusion_model == '512x512_diffusion_uncond_finetune_008100':
        model_config.update({
            'attention_resolutions': '32, 16, 8',
            'class_cond': False,
            'diffusion_steps': 1000, #No need to edit this, it is taken care of later.
            'rescale_timesteps': True,
            'timestep_respacing': 250, #No need to edit this, it is taken care of later.
            'image_size': 512,
            'learn_sigma': True,
            'noise_schedule': 'linear',
            'num_channels': 256,
            'num_head_channels': 64,
            'num_res_blocks': 2,
            'resblock_updown': True,
            'use_checkpoint': use_checkpoint,
            'use_fp16': not USE_CPU,
            'use_scale_shift_norm': True,
        })
    elif diffusion_model == '256x256_diffusion_uncond':
        model_config.update({
            'attention_resolutions': '32, 16, 8',
            'class_cond': False,
            'diffusion_steps': 1000, #No need to edit this, it is taken care of later.
            'rescale_timesteps': True,
            'timestep_respacing': 250, #No need to edit this, it is taken care of later.
            'image_size': 256,
            'learn_sigma': True,
            'noise_schedule': 'linear',
            'num_channels': 256,
            'num_head_channels': 64,
            'num_res_blocks': 2,
            'resblock_updown': True,
            'use_checkpoint': use_checkpoint,
            'use_fp16': not USE_CPU,
            'use_scale_shift_norm': True,
        })
    elif diffusion_model == 'portrait_generator_v001':
        model_config.update({
            'attention_resolutions': '32, 16, 8',
            'class_cond': False,
            'diffusion_steps': 1000,
            'rescale_timesteps': True,
            'image_size': 512,
            'learn_sigma': True,
            'noise_schedule': 'linear',
            'num_channels': 128,
            'num_heads': 4,
            'num_res_blocks': 2,
            'resblock_updown': True,
            'use_checkpoint': use_checkpoint,
            'use_fp16': True,
            'use_scale_shift_norm': True,
        })
    else:  # E.g. A model finetuned by KaliYuga
        model_config.update({
              'attention_resolutions': '16',
              'class_cond': False,
              'diffusion_steps': 1000,
              'rescale_timesteps': True,
              'timestep_respacing': 'ddim100',
              'image_size': 256,
              'learn_sigma': True,
              'noise_schedule': 'linear',
              'num_channels': 128,
              'num_heads': 1,
              'num_res_blocks': 2,
              'use_checkpoint': use_checkpoint,
              'use_fp16': True,
              'use_scale_shift_norm': False,
          })

    model_default = model_config['image_size']

    if USE_SECONDARY_DIFFUSION_MODEL:
        secondary_model = SecondaryDiffusionImageNet2()
        secondary_model.load_state_dict(torch.load(f'{model_path}/secondary_model_imagenet_2.pth', map_location='cpu'))
        secondary_model.eval().requires_grad_(False).to(device)

    clip_models = []
    if ViTB32: clip_models.append(clip.load('ViT-B/32', jit=False)[0].eval().requires_grad_(False).to(device))
    if ViTB16: clip_models.append(clip.load('ViT-B/16', jit=False)[0].eval().requires_grad_(False).to(device))
    if ViTL14: clip_models.append(clip.load('ViT-L/14', jit=False)[0].eval().requires_grad_(False).to(device))
    if ViTL14_336px: clip_models.append(clip.load('ViT-L/14@336px', jit=False)[0].eval().requires_grad_(False).to(device))
    if RN50: clip_models.append(clip.load('RN50', jit=False)[0].eval().requires_grad_(False).to(device))
    if RN50x4: clip_models.append(clip.load('RN50x4', jit=False)[0].eval().requires_grad_(False).to(device))
    if RN50x16: clip_models.append(clip.load('RN50x16', jit=False)[0].eval().requires_grad_(False).to(device))
    if RN50x64: clip_models.append(clip.load('RN50x64', jit=False)[0].eval().requires_grad_(False).to(device))
    if RN101: clip_models.append(clip.load('RN101', jit=False)[0].eval().requires_grad_(False).to(device))

    normalize = T.Normalize(mean=[0.48145466, 0.4578275, 0.40821073], std=[0.26862954, 0.26130258, 0.27577711])
    lpips_model = lpips.LPIPS(net='vgg').to(device)




    if diffusion_model == 'custom':
      model_config.update({
              'attention_resolutions': '16',
              'class_cond': False,
              'diffusion_steps': 1000,
              'rescale_timesteps': True,
              'timestep_respacing': 'ddim100',
              'image_size': 256,
              'learn_sigma': True,
              'noise_schedule': 'linear',
              'num_channels': 128,
              'num_heads': 1,
              'num_res_blocks': 2,
              'use_checkpoint': use_checkpoint,
              'use_fp16': True,
              'use_scale_shift_norm': False,
          })


    width_height = width_height_for_256x256_models if diffusion_model in diffusion_models_256x256_list else width_height_for_512x512_models


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

    # Note: If using a pixelart diffusion model, try adding "#pixelart" to the end of the prompt for a stronger effect.
    text_prompts = {
        0: ["A beautiful painting of a singular lighthouse, shining its light across a tumultuous sea of blood by greg rutkowski and thomas kinkade, Trending on artstation.", "yellow color scheme"],
        100: ["This set of prompts start at frame 100", "This prompt has weight five:5"],
    }

    image_prompts = {
        # 0:['ImagePromptsWorkButArentVeryGood.png:2',],
    }

    def _apply_cli_overrides(ov: dict | None) -> None:
        nonlocal clip_guidance_scale, tv_scale, range_scale, sat_scale, cutn, cutn_batches
        nonlocal init_image, init_scale, skip_steps, perlin_init, perlin_mode
        nonlocal skip_augs, randomize_class, clip_denoised, clamp_grad, clamp_max, set_seed
        nonlocal fuzzy_prompt, rand_mag, eta, use_vertical_symmetry, use_horizontal_symmetry
        nonlocal transformation_percent, video_init_flow_warp, video_init_flow_blend
        nonlocal video_init_check_consistency, text_prompts, image_prompts
        nonlocal width_height, side_x, side_y, steps, GENERATION_MODE, max_frames
        nonlocal translation_x, translation_y, translation_z
        nonlocal rotation_3d_x, rotation_3d_y, rotation_3d_z, near_plane, far_plane, fov
        nonlocal padding_mode, sampling_mode
        nonlocal turbo_mode, turbo_steps, turbo_preroll, frames_scale, frames_skip_steps
        nonlocal video_init_frames_scale, video_init_frames_skip_steps, zoom

        if not ov:
            if GENERATION_MODE in ("3D", "3D_latent"):
                zoom = "0: (1)"
                translation_z = "0: (1.5)"
            return

        if "GENERATION_MODE" in ov:
            GENERATION_MODE = ov["GENERATION_MODE"]
        if "clip_guidance_scale" in ov:
            clip_guidance_scale = ov["clip_guidance_scale"]
        if "tv_scale" in ov:
            tv_scale = ov["tv_scale"]
        if "range_scale" in ov:
            range_scale = ov["range_scale"]
        if "sat_scale" in ov:
            sat_scale = ov["sat_scale"]
        if "cutn" in ov:
            cutn = ov["cutn"]
        if "cutn_batches" in ov:
            cutn_batches = ov["cutn_batches"]
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
        if "skip_augs" in ov:
            skip_augs = ov["skip_augs"]
        if "randomize_class" in ov:
            randomize_class = ov["randomize_class"]
        if "clip_denoised" in ov:
            clip_denoised = ov["clip_denoised"]
        if "clamp_grad" in ov:
            clamp_grad = ov["clamp_grad"]
        if "clamp_max" in ov:
            clamp_max = ov["clamp_max"]
        if "set_seed" in ov:
            set_seed = ov["set_seed"]
        if "fuzzy_prompt" in ov:
            fuzzy_prompt = ov["fuzzy_prompt"]
        if "rand_mag" in ov:
            rand_mag = ov["rand_mag"]
        if "eta" in ov:
            eta = ov["eta"]
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
        if "video_init_frames_scale" in ov:
            video_init_frames_scale = ov["video_init_frames_scale"]
        if "video_init_frames_skip_steps" in ov:
            video_init_frames_skip_steps = ov["video_init_frames_skip_steps"]
        if "use_vertical_symmetry" in ov:
            use_vertical_symmetry = ov["use_vertical_symmetry"]
        if "use_horizontal_symmetry" in ov:
            use_horizontal_symmetry = ov["use_horizontal_symmetry"]
        if "transformation_percent" in ov:
            transformation_percent = ov["transformation_percent"]
        if "video_init_flow_warp" in ov:
            video_init_flow_warp = ov["video_init_flow_warp"]
        if "video_init_flow_blend" in ov:
            video_init_flow_blend = ov["video_init_flow_blend"]
        if "video_init_check_consistency" in ov:
            video_init_check_consistency = ov["video_init_check_consistency"]
        if "text_prompts" in ov:
            text_prompts = ov["text_prompts"]
        if "image_prompts" in ov:
            image_prompts = ov["image_prompts"]
        if "width_height" in ov:
            width_height = [ov["width_height"][0], ov["width_height"][1]]
            side_x = (width_height[0] // 64) * 64
            side_y = (width_height[1] // 64) * 64
            if side_x != width_height[0] or side_y != width_height[1]:
                print(
                    f"Changing output size to {side_x}x{side_y}. Dimensions must be multiples of 64."
                )

        if GENERATION_MODE in ("3D", "3D_latent"):
            zoom = "0: (1)"
            translation_z = "0: (1.5)"

    _apply_cli_overrides(cli_overrides)



    """
    ### Animation Settings
    """


    # Video Input Settings
    # Call optical flow from video frames and warp prev frame with flow
    if GENERATION_MODE == "Video Input":
        if persistent_frame_output_in_batch_folder:
            videoFramesFolder = f'{batchFolder}/videoFrames'
        else:
            videoFramesFolder = f'{ROOT_PATH}/videoFrames'
        createPath(videoFramesFolder)
        print(f"Exporting Video Frames (1 every {extract_nth_frame})...")
        try:
            for f in pathlib.Path(f'{videoFramesFolder}').glob('*.jpg'):
                f.unlink()
        except:
            print('')
        vf = f'select=not(mod(n\\,{extract_nth_frame}))'
        if os.path.exists(video_init_path):
            subprocess.run(['ffmpeg', '-i', f'{video_init_path}', '-vf', f'{vf}', '-vsync', 'vfr', '-q:v', '2', '-loglevel', 'error', '-stats', f'{videoFramesFolder}/%04d.jpg'], stdout=subprocess.PIPE).stdout.decode('utf-8')
        else: 
            print(f'\nWARNING!\n\nVideo not found: {video_init_path}.\nPlease check your video path.\n')




    # 2D Animation Settings:**
    # `zoom` is a multiplier of dimensions, 1 is no zoom.
    # All rotations are provided in degrees.
    if GENERATION_MODE == "Video Input":
        max_frames = len(glob(f'{videoFramesFolder}/*.jpg'))

    # Insist that turbo be used only with 3D anim.
    if turbo_mode and GENERATION_MODE not in ('3D', '3D_latent'):
        print('Turbo mode only available with 3D animations. Disabling Turbo.')
        turbo_mode = False

    from .config.keyframes import get_inbetweens, parse_key_frames


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




    force_download = False
    if GENERATION_MODE == 'Video Input':
        try:
            from raft import RAFT
        except:
            if not os.path.exists(os.path.join(PROJECT_DIR, 'RAFT')):
                gitclone('https://github.com/princeton-vl/RAFT', os.path.join(PROJECT_DIR, 'RAFT'))
            sys.path.append(f'{PROJECT_DIR}/RAFT')

        if (not (os.path.exists(f'{ROOT_PATH}/RAFT/models'))) or force_download:
            createPath(f'{ROOT_PATH}/RAFT')
            os.chdir(f'{ROOT_PATH}/RAFT')
            sub_p_res = subprocess.run(['bash', f'{PROJECT_DIR}/RAFT/download_models.sh'], stdout=subprocess.PIPE).stdout.decode('utf-8')
            print(sub_p_res)
            os.chdir(PROJECT_DIR)



    # Define optical flow functions for Video input animation mode only

    if GENERATION_MODE == 'Video Input':
        in_path = videoFramesFolder
        flo_folder = f'{in_path}/out_flo_fwd'
        path = f'{PROJECT_DIR}/RAFT/core'
        import sys
        sys.path.append(f'{PROJECT_DIR}/RAFT/core')
        os.chdir(f'{PROJECT_DIR}/RAFT/core')
        
        print(os.getcwd())
        print("Renaming RAFT core's utils.utils to raftutils.utils (to avoid a naming conflict with AdaBins)")
        
        if not os.path.exists(f'{PROJECT_DIR}/RAFT/core/raftutils'):
            os.rename(f'{PROJECT_DIR}/RAFT/core/utils', f'{PROJECT_DIR}/RAFT/core/raftutils')
            sub_p_res = subprocess.run(['sed', '-i', 's/from utils.utils/from raftutils.utils/g', f'{PROJECT_DIR}/RAFT/core/corr.py'], stdout=subprocess.PIPE).stdout.decode('utf-8')
            sub_p_res = subprocess.run(['sed', '-i', 's/from utils.utils/from raftutils.utils/g', f'{PROJECT_DIR}/RAFT/core/raft.py'], stdout=subprocess.PIPE).stdout.decode('utf-8')

        from raftutils.utils import InputPadder
        from raft import RAFT
        import argparse

        args2 = argparse.Namespace()
        args2.small = False
        args2.mixed_precision = True


        TAG_CHAR = np.array([202021.25], np.float32)

        def writeFlow(filename,uv,v=None):
            """ 
            https://github.com/NVIDIA/flownet2-pytorch/blob/master/utils/flow_utils.py
            Copyright 2017 NVIDIA CORPORATION

            Licensed under the Apache License, Version 2.0 (the "License");
            you may not use this file except in compliance with the License.
            You may obtain a copy of the License at

                http://www.apache.org/licenses/LICENSE-2.0

            Unless required by applicable law or agreed to in writing, software
            distributed under the License is distributed on an "AS IS" BASIS,
            WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
            See the License for the specific language governing permissions and
            limitations under the License.

            Write optical flow to file.

            If v is None, uv is assumed to contain both u and v channels,
            stacked in depth.
            Original code by Deqing Sun, adapted from Daniel Scharstein.
            """
            nBands = 2

            if v is None:
                assert(uv.ndim == 3)
                assert(uv.shape[2] == 2)
                u = uv[:,:,0]
                v = uv[:,:,1]
            else:
                u = uv

            assert(u.shape == v.shape)
            height,width = u.shape
            f = open(filename,'wb')
            # write the header
            f.write(TAG_CHAR)
            np.array(width).astype(np.int32).tofile(f)
            np.array(height).astype(np.int32).tofile(f)
            # arrange into matrix form
            tmp = np.zeros((height, width*nBands))
            tmp[:,np.arange(width)*2] = u
            tmp[:,np.arange(width)*2 + 1] = v
            tmp.astype(np.float32).tofile(f)
            f.close()

        def load_img(img, size):
            img = Image.open(img).convert('RGB').resize(size)
            return torch.from_numpy(np.array(img)).permute(2,0,1).float()[None,...].to(device)

        def get_flow(frame1, frame2, model, iters=20):
            padder = InputPadder(frame1.shape)
            frame1, frame2 = padder.pad(frame1, frame2)
            _, flow12 = model(frame1, frame2, iters=iters, test_mode=True)
            flow12 = flow12[0].permute(1, 2, 0).detach().cpu().numpy()

            return flow12

        def warp_flow(img, flow):
            h, w = flow.shape[:2]
            flow = flow.copy()
            flow[:, :, 0] += np.arange(w)
            flow[:, :, 1] += np.arange(h)[:, np.newaxis]
            res = cv2.remap(img, flow, None, cv2.INTER_LINEAR)
            return res

        def makeEven(_x):
            return _x if (_x % 2 == 0) else _x+1

        def fit(img,maxsize=512):
            maxdim = max(*img.size)
            if maxdim>maxsize:
                # if True:
                ratio = maxsize/maxdim
                x,y = img.size
                size = (makeEven(int(x*ratio)),makeEven(int(y*ratio))) 
                img = img.resize(size)
            return img

        def warp(frame1, frame2, flo_path, blend=0.5, weights_path=None):
            flow21 = np.load(flo_path)
            frame1pil = np.array(frame1.convert('RGB').resize((flow21.shape[1],flow21.shape[0])))
            frame1_warped21 = warp_flow(frame1pil, flow21)
            # frame2pil = frame1pil
            frame2pil = np.array(frame2.convert('RGB').resize((flow21.shape[1],flow21.shape[0])))

            if weights_path:
                # TBD
                pass
            else:
                blended_w = frame2pil*(1-blend) + frame1_warped21*(blend)

            return  Image.fromarray(blended_w.astype('uint8'))

        in_path = videoFramesFolder
        flo_folder = f'{in_path}/out_flo_fwd'

        temp_flo = in_path+'/temp_flo'
        flo_fwd_folder = in_path+'/out_flo_fwd'
        # TBD flow backwards!
        os.chdir(PROJECT_DIR)


    # Generate optical flow and consistency maps
    # Run once per init video

    if GENERATION_MODE == "Video Input":
        import gc

        force_flow_generation = False
        in_path = videoFramesFolder
        flo_folder = f'{in_path}/out_flo_fwd'

        if not video_init_flow_warp:
            print('video_init_flow_warp not set, skipping')

        if (GENERATION_MODE == 'Video Input') and (video_init_flow_warp):
            flows = glob(flo_folder+'/*.*')
            if (len(flows)>0) and not force_flow_generation:
                print(f'Skipping flow generation:\nFound {len(flows)} existing flow files in current working folder: {flo_folder}.\nIf you wish to generate new flow files, check force_flow_generation and run this cell again.')

            if (len(flows)==0) or force_flow_generation:
                frames = sorted(glob(in_path+'/*.*'));
                if len(frames)<2: 
                    print(f'WARNING!\nCannot create flow maps: Found {len(frames)} frames extracted from your video input.\nPlease check your video path.')
                if len(frames)>=2:

                    raft_model = torch.nn.DataParallel(RAFT(args2))
                    raft_model.load_state_dict(torch.load(f'{ROOT_PATH}/RAFT/models/raft-things.pth'))
                    raft_model = raft_model.module.to(device).eval()

                    for f in pathlib.Path(f'{flo_fwd_folder}').glob('*.*'):
                        f.unlink()

                    temp_flo = in_path+'/temp_flo'
                    flo_fwd_folder = in_path+'/out_flo_fwd'

                    createPath(flo_fwd_folder)
                    createPath(temp_flo)

                    # TBD Call out to a consistency checker?

                    framecount = 0
                    for frame1, frame2 in tqdm(zip(frames[:-1], frames[1:]), total=len(frames)-1):

                        out_flow21_fn = f"{flo_fwd_folder}/{frame1.split('/')[-1]}"

                        frame1 = load_img(frame1, width_height)
                        frame2 = load_img(frame2, width_height)

                        flow21 = get_flow(frame2, frame1, raft_model)
                        np.save(out_flow21_fn, flow21)

                        if video_init_check_consistency:
                            # TBD
                            pass

                    del raft_model 
                    gc.collect()


    """
    ### Extra Settings
     Partial Saves, Advanced Settings, Cutn Scheduling
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

    # There are a few extra advanced settings available if you double click this cell.
    # Perlin init will replace your init, so uncheck if using one.



    # Cutn Scheduling:
    # Format: `[40]*400+[20]*600` = 40 cuts for the first 400 /1000 steps, then 20 for the last 600/1000

    # cut_overview and cut_innercut are cumulative for total cutn on any given step. Overview cuts see the entire image and are good for early structure, innercuts are your standard cutn.

    cut_overview = "[12]*400+[4]*600"
    cut_innercut = "[4]*400+[12]*600"
    cut_ic_pow = "[1]*1000"
    cut_icgray_p = "[0.2]*400+[0]*600"

    # KaliYuga model settings. Refer to [cut_ic_pow](https://ezcharts.miraheze.org/wiki/Category:Cut_ic_pow) as a guide. Values between 1 and 100 all work.
    pad_or_pulp_cut_overview = "[15]*100+[15]*100+[12]*100+[12]*100+[6]*100+[4]*100+[2]*200+[0]*200" 
    pad_or_pulp_cut_innercut = "[1]*100+[1]*100+[4]*100+[4]*100+[8]*100+[8]*100+[10]*200+[10]*200"
    pad_or_pulp_cut_ic_pow = "[12]*300+[12]*100+[12]*50+[12]*50+[10]*100+[10]*100+[10]*300"
    pad_or_pulp_cut_icgray_p = "[0.87]*100+[0.78]*50+[0.73]*50+[0.64]*60+[0.56]*40+[0.50]*50+[0.33]*100+[0.19]*150+[0]*400"

    watercolor_cut_overview = "[14]*200+[12]*200+[4]*400+[0]*200"
    watercolor_cut_innercut = "[2]*200+[4]*200+[12]*400+[12]*200"
    watercolor_cut_ic_pow = "[12]*300+[12]*100+[12]*50+[12]*50+[10]*100+[10]*100+[10]*300"
    watercolor_cut_icgray_p = "[0.7]*100+[0.6]*100+[0.45]*100+[0.3]*100+[0]*600"

    if (diffusion_model in kaliyuga_pixel_art_model_names) or (diffusion_model in kaliyuga_pulpscifi_model_names):
        cut_overview = pad_or_pulp_cut_overview
        cut_innercut = pad_or_pulp_cut_innercut
        cut_ic_pow = pad_or_pulp_cut_ic_pow
        cut_icgray_p = pad_or_pulp_cut_icgray_p
    elif diffusion_model in kaliyuga_watercolor_model_names:
        cut_overview = watercolor_cut_overview
        cut_innercut = watercolor_cut_innercut
        cut_ic_pow = watercolor_cut_ic_pow
        cut_icgray_p = watercolor_cut_icgray_p


    # Transformation Settings


    """
    ### Prompts
    When GENERATION_MODE is "None", only the first prompt set is used. For "2D" or video modes, prompts advance per frame and hold on the last defined frame.
    """

    """
    # 4. Diffuse!
    """

    # Do the Run!
    # `n_batches` ignored with animation modes.

    #Update Model Settings
    from .diffusion import load_primary_diffusion_model
    from .guided_diffusion.script_util import diffusion_steps_count, timestep_respacing_ddim

    timestep_respacing = timestep_respacing_ddim(steps)
    diffusion_steps = diffusion_steps_count(steps)
    model_config.update({
        'timestep_respacing': timestep_respacing,
        'diffusion_steps': diffusion_steps,
    })

    batch_size = 1 

    def move_files(start_num, end_num, old_folder, new_folder):
        for i in range(start_num, end_num):
            old_file = old_folder + f'/{batch_name}({batchNum})_{i:04}.png'
            new_file = new_folder + f'/{batch_name}({batchNum})_{i:04}.png'
            os.rename(old_file, new_file)

    def _exit_if_video_input_assets_missing() -> None:
        if GENERATION_MODE != "Video Input":
            return
        frames_chk = sorted(glob(in_path + "/*.*"))
        if len(frames_chk) == 0:
            sys.exit(
                "ERROR: 0 frames found.\nPlease check your video input path and rerun the video settings cell."
            )
        flows_chk = glob(flo_folder + "/*.*")
        if len(flows_chk) == 0 and video_init_flow_warp:
            sys.exit("ERROR: 0 flow files found.\nPlease rerun the flow generation cell.")



    resume_run = False 
    run_to_resume = 'latest'
    resume_from_frame = 'latest'
    retain_overwritten_frames = False
    if retain_overwritten_frames:
        retainFolder = f'{batchFolder}/retained'
        createPath(retainFolder)


    skip_step_ratio = int(frames_skip_steps.rstrip("%")) / 100
    calc_frames_skip_steps = math.floor(steps * skip_step_ratio)

    _exit_if_video_input_assets_missing()

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
            if GENERATION_MODE not in ('3D', '3D_latent') and turbo_mode == True and start_frame > turbo_preroll and start_frame % int(turbo_steps) != 0:
                start_frame = start_frame - (start_frame % int(turbo_steps))
        else:
            start_frame = int(resume_from_frame)+1
            if GENERATION_MODE not in ('3D', '3D_latent') and turbo_mode == True and start_frame > turbo_preroll and start_frame % int(turbo_steps) != 0:
                start_frame = start_frame - (start_frame % int(turbo_steps))
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
        generation_mode=GENERATION_MODE,
        seed=seed,
    )
    print(
        f"[config] profile={run_config.profile} mode={run_config.generation_mode} "
        f"device={run_config.device} output_dir={run_config.output_dir} seed={run_config.seed}",
        flush=True,
    )

    from .config import build_run_args_namespace

    run_config = run_config.with_runtime_values(locals())
    args = build_run_args_namespace(run_config)

    print('Prepping model')

    model, diffusion = load_primary_diffusion_model(
        model_config=model_config,
        diffusion_model=diffusion_model,
        custom_path=custom_path,
        model_path=model_path,
        diff_model_map=diff_model_map,
        device=device,
        create_model_and_diffusion=create_model_and_diffusion,
        get_model_filename=get_model_filename,
    )

    guided_backend = GuidedDiffusionBackend(
        diffusion_sampling_mode=diffusion_sampling_mode,
        diffusion=diffusion,
        model=model,
        batch_size=batch_size,
        side_y=side_y,
        side_x=side_x,
        clip_denoised=clip_denoised,
        randomize_class=randomize_class,
        eta=eta,
        symmetry_transformation_fn=symmetry_transformation_fn,
        transformation_percent=transformation_percent,
    )
    latent_backend = LatentDiffusionBackend(device=device, models_root=model_path)

    from . import main as _main_pub

    _main_pub.do_run = do_run

    try:
        do_run()
    except KeyboardInterrupt:
        pass
    except RuntimeError as exc:
        from .platform.cuda import format_cuda_oom_hint

        if DEVICE.type == "cuda" and "out of memory" in str(exc).lower():
            print(format_cuda_oom_hint(), file=sys.stderr)
        raise
    finally:
        print('Seed used:', seed)
        gc.collect()
        torch.cuda.empty_cache()




    from .image.ffmpeg_utils import encode_numbered_png_sequence_h264

    # Create video
    # Video file will save in the same folder as your images.

    _exit_if_video_input_assets_missing()

    blend =  0.5
    video_init_check_consistency = False

    latest_run = batchNum

    folder = batch_name
    run = latest_run
    final_frame = 'final_frame'


    init_frame = 1
    last_frame = final_frame
    fps = 12

    if last_frame == 'final_frame':
        last_frame = len(glob(batchFolder+f"/{folder}({run})_*.png"))
        print(f'Total frames: {last_frame}')

    image_path = f"{outputDirPath}/{folder}/{folder}({run})_%04d.png"
    filepath = f"{outputDirPath}/{folder}/{folder}({run}).mp4"

    if (video_init_blend_mode == 'optical flow') and (GENERATION_MODE == 'Video Input'):
        image_path = f"{outputDirPath}/{folder}/flow/{folder}({run})_%04d.png"
        filepath = f"{outputDirPath}/{folder}/{folder}({run})_flow.mp4"
        if last_frame == 'final_frame':
            last_frame = len(glob(batchFolder+f"/flow/{folder}({run})_*.png"))
        flo_out = batchFolder+f"/flow"
        createPath(flo_out)
        frames_in = sorted(glob(batchFolder+f"/{folder}({run})_*.png"))
        shutil.copy(frames_in[0], flo_out)
        for i in trange(init_frame, min(len(frames_in), last_frame)):
            frame1_path = frames_in[i-1]
            frame2_path = frames_in[i]

            frame1 = Image.open(frame1_path)
            frame2 = Image.open(frame2_path)
            frame1_stem = f"{(int(frame1_path.split('/')[-1].split('_')[-1][:-4])+1):04}.jpg"
            flo_path = f"/{flo_folder}/{frame1_stem}.npy"
            weights_path = None
            if video_init_check_consistency:
                # TBD
                pass
            warp(frame1, frame2, flo_path, blend=blend, weights_path=weights_path).save(batchFolder+f"/flow/{folder}({run})_{i:04}.png")
    if video_init_blend_mode == 'linear':
        image_path = f"{outputDirPath}/{folder}/blend/{folder}({run})_%04d.png"
        filepath = f"{outputDirPath}/{folder}/{folder}({run})_blend.mp4"
        if last_frame == 'final_frame':
            last_frame = len(glob(batchFolder+f"/blend/{folder}({run})_*.png"))
        blend_out = batchFolder+f"/blend"
        createPath(blend_out)
        frames_in = glob(batchFolder+f"/{folder}({run})_*.png")
        shutil.copy(frames_in[0], blend_out)
        for i in trange(1, len(frames_in)):
            frame1_path = frames_in[i-1]
            frame2_path = frames_in[i]

            frame1 = Image.open(frame1_path)
            frame2 = Image.open(frame2_path)

            frame = Image.fromarray((np.array(frame1)*(1-blend) + np.array(frame2)*(blend)).astype('uint8')).save(batchFolder+f"/blend/{folder}({run})_{i:04}.png")


    encode_numbered_png_sequence_h264(
        cwd=f"{batchFolder}",
        image_sequence_pattern=image_path,
        output_path=filepath,
        fps=fps,
        start_number=init_frame,
        frames_v=last_frame + 1,
    )

    # Create per-channel 3D debug videos if the 3d folder exists and has frames
    if GENERATION_MODE in ("3D", "3D_latent") and os.path.isdir(debug3dFolder):
        _3d_channels = ["warped", "source", "depth_blended", "depth_midas", "depth_adabins", "flow_field"]
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

