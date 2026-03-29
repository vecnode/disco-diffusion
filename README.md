# Disco Diffusion

A modern refactor of disco-diffusion (ongoing).

## Reproduce

```sh
# Create local env and run
python3 -m venv venv
source venv/bin/activate
python3 disco.py
```

## Models

`disco.py` clones missing **code** into the project directory and downloads **weights** into `models/` (and a few other paths). Defaults assume the 512×512 OpenAI-class UNet plus secondary + CLIP + MiDaS.

### Mandatory


| Block/Model                | What                                                                        | Where                                                                                        |
| -------------------------- | --------------------------------------------------------------------------- | -------------------------------------------------------------------------------------------- |
| **guided-diffusion**       | OpenAI DDPM/ADM implementation (`create_model_and_diffusion`)               | Repo cloned as `guided-diffusion/`                                                           |
| **CLIP**                   | OpenAI CLIP (used via `CLIP.clip`)                                          | Repo cloned as `CLIP/`; ViT/RN **weights** load on first use (cached under `~/.cache`/torch) |
| **ResizeRight**            | Resizing helper                                                             | Repo cloned as `ResizeRight/`                                                                |
| **pytorch3d-lite**         | `py3d_tools` for 3D transforms                                              | Repo cloned as `pytorch3d-lite/`                                                             |
| **MiDaS**                  | Depth backbone for 3D / `disco_xform_utils`                                 | Repo cloned as `MiDaS/`; default `**dpt_large-midas-2f21e586.pt`** → `models/`               |
| **disco_xform_utils**      | 3D warp helpers                                                             | `disco_xform_utils.py` from `alembics/disco-diffusion` if missing                            |
| **Primary diffusion UNet** | Main noise model (default `**512x512_diffusion_uncond_finetune_008100`**)   | `512x512_diffusion_uncond_finetune_008100.pt` → `models/`                                    |
| **Secondary model**        | Smaller imagenet-conditioned helper (`use_secondary_model=True` by default) | `secondary_model_imagenet_2.pth` → `models/`                                                 |


Also required: **PyTorch**, **torchvision**, and Python deps installed by the script (e.g. `lpips`, `timm`, `opencv-python`, `pandas`, …).

### Optional


| Block/Model                         | When                                                     | Notes                                                                                                                                                                                                            |
| ----------------------------------- | -------------------------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **AdaBins**                         | `USE_ADABINS=True` (default)                             | Repo `AdaBins/`; weights `**AdaBins_nyu.pt`** → `pretrained/` (downloaded from [deforum/AdaBins](https://huggingface.co/deforum/AdaBins) only; if `wget` fails, download manually or set `USE_ADABINS=False`).   |
| **open_clip**                       | Any OpenCLIP flag enabled (e.g. LAION checkpoints)       | Repo `open_clip/`; weights fetched by `open_clip` when a model is first built                                                                                                                                    |
| **Alternate diffusion checkpoints** | Set `diffusion_model` to another key in `diff_model_map` | e.g. **256×256** OpenAI base, **portrait_generator_v001**, **pixel** / **watercolor** / **PulpSciFi** finetunes from Hugging Face (see `disco.py` → `diff_model_map`) — each is a separate `.pt` under `models/` |
| **RAFT**                            | `animation_mode == 'Video Input'`                        | Repo `RAFT/`; `raft-things.pth` (and related) via `RAFT/download_models.sh`                                                                                                                                      |
| **Custom UNet**                     | `diffusion_model == 'custom'`                            | Your `custom_path` checkpoint                                                                                                                                                                                    |
| **Extra MiDaS variants**            | `midas_depth_model` other than `dpt_large`               | Additional `.pt` files in `models/` per MiDaS/DPT naming in `disco.py`                                                                                                                                           |


