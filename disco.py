

"""

Setting | Description | Default
--- | --- | ---
**Your vision:**
`text_prompts` | A description of what you'd like the machine to generate. Think of it like writing the caption below your image on a website. | N/A
`image_prompts` | Think of these images more as a description of their contents. | N/A
**Image quality:**
`clip_guidance_scale`  | Controls how much the image should look like the prompt. | 1000
`tv_scale` | Controls the smoothness of the final output. | 150
`range_scale` | Controls how far out of range RGB values are allowed to be. | 150
`sat_scale` | Controls how much saturation is allowed. From nshepperd's JAX notebook. | 0
`cutn` | Controls how many crops to take from the image. | 16
`cutn_batches` | Accumulate CLIP gradient from multiple batches of cuts. | 2
**Init settings:**
`init_image` | URL or local path | None
`init_scale` | This enhances the effect of the init image, a good value is 1000 | 0
`skip_steps` | Controls the starting point along the diffusion timesteps | 0
`perlin_init` | Option to start with random perlin noise | False
`perlin_mode` | ('gray', 'color') | 'mixed'
**Advanced:**
`skip_augs` | Controls whether to skip torchvision augmentations | False
`randomize_class` | Controls whether the imagenet class is randomly changed each iteration | True
`clip_denoised` | Determines whether CLIP discriminates a noisy or denoised image | False
`clamp_grad` | Experimental: Using adaptive clip grad in the cond_fn | True
`seed`  | Choose a random seed and print it at end of run for reproduction | random_seed
`fuzzy_prompt` | Controls whether to add multiple noisy prompts to the prompt losses | False
`rand_mag` | Controls the magnitude of the random noise | 0.1
`eta` | DDIM hyperparameter | 0.5
`use_vertical_symmetry` | Enforce symmetry over x axis of the image on [`tr_st`*`steps` for `tr_st` in `transformation_steps`] steps of the diffusion process | False
`use_horizontal_symmetry` | Enforce symmetry over y axis of the image on [`tr_st`*`steps` for `tr_st` in `transformation_steps`] steps of the diffusion process | False
`transformation_steps` | Steps (expressed in percentages) in which the symmetry is enforced | [0.01]
`video_init_flow_warp` | Flow warp enabled | True
`video_init_flow_blend` | 0 - you get raw input, 1 - you get warped diffused previous frame  | 0.999
`video_init_check_consistency` | TBD check forward-backward flow consistency (uncheck unless there are too many warping artifacts) | False


**Model settings**
---

Setting | Description | Default
--- | --- | ---
**Diffusion:**
`timestep_respacing` | Modify this value to decrease the number of timesteps. | ddim100
`diffusion_steps` || 1000
**Diffusion:**
`clip_models` | Models of CLIP to load. Typically the more, the better but they all come at a hefty VRAM cost. | ViT-B/32, ViT-B/16, RN50x4
"""


import os
import sys

_SRC_ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "src")
if _SRC_ROOT not in sys.path:
    sys.path.insert(0, _SRC_ROOT)

print("[disco.py] Starting runtime.", flush=True)

from discodiff.cli import parse_disco_argv
from discodiff.main import main

if __name__ == "__main__":
    main(cli_overrides=parse_disco_argv(sys.argv[1:]))
