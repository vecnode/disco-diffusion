# disco-diffusion

A modern refactor of disco-diffusion (ongoing).

Specific support for RTX Graphics Cards, tests under 24Gb.

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
python3 disco.py --width 1024 --height 576 --set-seed 42 --steps 250 \
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

`**src/discodiff/` layout** (top-level `**main.py**` is the notebook-style runtime.)

- `**main.py`** — Active entry body: environment, clones, CLIP / diffusion / MiDaS, settings, `do_run` sampling loop.
- `**app/entrypoint.py`** — `**discodiff.run()**` → delegates to `**main.main()**` (also exported from `**discodiff.__init__**`).
- `**cli/parser.py`** — `disco.py` flags → override dict (`**discodiff.cli.parse_disco_argv**`).
- `**config/run_args.py`** — `**build_run_args_namespace**`; `**config/keyframes.py`** — `split_prompts`, keyframe parsing; `**config/defaults.py`** reserved for future defaults.
- `**diffusion/**` — `**load.py**` (UNet load), `**schedules.py**` (DDIM string / step count), `**sampling.py`** (placeholder for loop extraction).
- `**guidance/clip_cuts.py`** — Placeholder for CLIP cutouts / `cond_fn` extraction.
- `**platform/cuda.py`** — Linux notice, CUDA logging, TF32 / cuDNN env toggles.
- `**assets/**` — `**downloads.py**` (git, wget, fetch, checkpoint download), `**paths.py**` (`createPath`).
- `**image/**` — `**resize.py**`, `**noise.py**` (Perlin).
- `**geometry/warp.py`** — 3D / depth warp (MiDaS; optional AdaBins).

### Mandatory third-party code (cloned beside the project if missing)

- **guided-diffusion** — OpenAI DDPM/ADM (`create_model_and_diffusion`). Location: `./guided-diffusion/`.
- **CLIP** — `CLIP.clip` text/image encoders. Location: `./CLIP/`; ViT/RN weights often under `~/.cache/torch`.
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
