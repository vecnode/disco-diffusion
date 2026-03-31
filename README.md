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

#### Text-to-image 

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

#### Text-to-video (2D)

```sh
# Example A — single resolution, two prompts (holds last prompt after frame 30)
python3 disco.py --generation-mode 2D --width 512 --height 512 --set-seed 42 --steps 100 \
  --text-prompts-json /dev/stdin <<'EOF'
{"0": ["establishing shot of a coastal lighthouse at dawn, atmospheric"], "30": ["same scene, golden hour, warm light on the cliffs"]}
EOF
```

```sh
# Example B — higher spatial resolution, explicit multi-step schedule (adjust `max_frames` in main when producing longer sequences)
python3 disco.py --generation-mode 2D --width 1024 --height 576 --set-seed 42 --steps 250 \
  --text-prompts-json /dev/stdin <<'EOF'
{"0": ["wide landscape, mountains and a field, misty morning"], "15": ["camera slowly dollying forward, same environment, sharper detail"]}
EOF
```

## Platform

**Linux tested.** This repo is maintained and expected to run on **glibc-based Linux** with a working **NVIDIA stack** (proprietary driver so `nvidia-smi` reports your GPU). Other OSes are currently out of scope; if you run elsewhere, you may see a stderr warning — set `DISCO_ALLOW_NON_LINUX=1` to suppress it (still unsupported).

## GPU (NVIDIA on Linux)

**Optional environment variables** (defaults unchanged):

| Variable | Purpose |
|----------|---------|
| `CUDA_VISIBLE_DEVICES` | Restrict visible GPUs (e.g. `0`). |
| `PYTORCH_CUDA_ALLOC_CONF` | e.g. `expandable_segments:True` to reduce fragmentation OOMs. |
| `DISCO_ALLOW_TF32` | `1` on **Ampere+** enables TF32 for faster matmul / cuDNN (small numeric differences). |
| `DISCO_CUDNN_BENCHMARK` | `1` enables cuDNN autotune for throughput (less strict determinism). |

CPU runs: set `USE_CPU = True` near the top of `src/discodiff/main.py`. On CUDA OOM, a short hint is printed to stderr.

## Libraries

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

