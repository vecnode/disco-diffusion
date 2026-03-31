# disco-diffusion

A modern refactor of disco-diffusion (ongoing).

## Reproduce

```sh
# Create local env and run
python3 -m venv venv
source venv/bin/activate
python3 disco.py
# equivalent:
cd src && python3 -m discodiff.main
```

```sh
# 1) Baseline — square 512×512, prompt via stdin JSON
python3 disco.py --width 512 --height 512 --text-prompts-json /dev/stdin <<'EOF'
{"0": ["trees and a beautiful field of the mountain"]}
EOF
```

```sh
# 2) Same resolution and prompt, fixed seed for reproducibility
python3 disco.py --width 512 --height 512 --set-seed 42 \
  --text-prompts-json /dev/stdin <<'EOF'
{"0": ["trees and a beautiful field of the mountain"]}
EOF
```

```sh
# 3) Same resolution and prompt, stronger CLIP adherence and lower DDIM eta
python3 disco.py --width 512 --height 512 --clip-guidance-scale 6500 --eta 0.5 \
  --text-prompts-json /dev/stdin <<'EOF'
{"0": ["trees and a beautiful field of the mountain"]}
EOF
```

## Platform

**Linux only.** This repo is maintained and expected to run on **glibc-based Linux** with a working **NVIDIA stack** (proprietary driver so `nvidia-smi` reports your GPU). Other OSes are out of scope; if you run elsewhere, you may see a stderr warning — set `DISCO_ALLOW_NON_LINUX=1` to suppress it (still unsupported).

## GPU (NVIDIA on Linux)

**Optional environment variables** (defaults unchanged):

| Variable | Purpose |
|----------|---------|
| `CUDA_VISIBLE_DEVICES` | Restrict visible GPUs (e.g. `0`). |
| `PYTORCH_CUDA_ALLOC_CONF` | e.g. `expandable_segments:True` to reduce fragmentation OOMs. |
| `DISCO_ALLOW_TF32` | `1` on **Ampere+** enables TF32 for faster matmul / cuDNN (small numeric differences). |
| `DISCO_CUDNN_BENCHMARK` | `1` enables cuDNN autotune for throughput (less strict determinism). |

CPU runs: set `USE_CPU = True` near the top of `src/discodiff/main.py`. On CUDA OOM, a short hint is printed to stderr.

## Libraries and models

**Application layout:** `**src/discodiff/`** as the `**discodiff**` package. Root `**disco.py**` prepends `src/` to `sys.path` and calls `**discodiff.main.main()**`, which still behaves like the original notebook: **pip** installs, **git clones** for missing trees, **weights** into `models/` (and paths below).

`**src/discodiff/` modules**

- `**main.py`** — Environment setup, third-party clones, CLIP / diffusion / MiDaS, user settings, sampling loop, optional ffmpeg video pass.
- `**config.py`** — Builds run `args` as a `SimpleNamespace` from the legacy local-variable layout.
- `**pipeline.py`** — Primary UNet + diffusion instance: checkpoint load, device, fp16 / grad flags.
- `**run.py`** — Invokes the diffusion loop wired through the `main` module.
- `**main_utils.py`** — Git clone, wget, fetch, checkpoint download, paths.
- `**diffusion_utils.py`** — Keyframes and prompt series (`split_prompts`, etc.).
- `**noise.py`** — Perlin initialization.
- `**main_xform_utils.py`** — 3D warping / depth (MiDaS; optional AdaBins when enabled).
- `**cuda_setup.py`** — Linux platform notice; CUDA startup logging; optional TF32 / cuDNN benchmark via env vars.

### Mandatory third-party code (cloned beside the project if missing)

- **guided-diffusion** — OpenAI DDPM/ADM (`create_model_and_diffusion`). Location: `./guided-diffusion/`.
- **CLIP** — `CLIP.clip` text/image encoders. Location: `./CLIP/`; ViT/RN weights often under `~/.cache/torch`.
- **ResizeRight** — `resize_right`. Location: `./ResizeRight/`.
- **pytorch3d-lite** — `py3d_tools` (3D). Location: `./pytorch3d-lite/`.
- **MiDaS** — Depth for 3D / warp. Location: `./MiDaS/`.

### Mandatory weights (default configuration)

- **Primary UNet** (default `512x512_diffusion_uncond_finetune_008100`) — `models/*.pt` per `diff_model_map`.
- **Secondary model** (when enabled) — `secondary_model_imagenet_2.pth` → `models/`.
- **MiDaS DPT Large** (default) — e.g. `dpt_large-midas-2f21e586.pt` → `models/`.

You also need **PyTorch**, **torchvision**, and dependencies the script installs (e.g. `lpips`, `timm`, `opencv-python`, `pandas`).

### Optional third-party / weights

- **AdaBins** — Enable with `**USE_ADABINS = True`** in `src/discodiff/main.py` (default `**False**`). `./AdaBins/`; `AdaBins_nyu.pt` → `pretrained/` ([deforum/AdaBins](https://huggingface.co/deforum/AdaBins)); env `**MAIN_USE_ADABINS**` follows that flag.
- **open_clip** — When OpenCLIP options are enabled in `main.py`. `./open_clip/`; weights when the model is first built.
- **Other diffusion checkpoints** — `diffusion_model` keys in `diff_model_map` in `main.py`; extra `.pt` files under `models/`.
- **RAFT** — When `animation_mode == 'Video Input'`. `./RAFT/`; `raft-things.pth` (see `RAFT/download_models.sh`).
- **Custom UNet** — When `diffusion_model == 'custom'`; path in `custom_path`.
- **Other MiDaS checkpoints** — When `midas_depth_model` ≠ `dpt_large`; matching `.pt` in `models/` per `main.py`.
