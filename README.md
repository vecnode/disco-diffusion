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

#### Text-to-video (3D)

```sh
# Example A — short 3D smoke run (few steps; good for wiring/paths validation)
python3 disco.py --generation-mode 3D --width 512 --height 512 --set-seed 42 --steps 50 \
  --text-prompts-json /dev/stdin <<'EOF'
{"0": ["cinematic coastal lighthouse, dusk fog, volumetric light, wide angle"]}
EOF
```

```sh
# Example B — longer 3D run (more steps; keep the prompt stable while camera motion comes from keyframes in main.py)
python3 disco.py --generation-mode 3D --width 1024 --height 576 --set-seed 42 --steps 250 \
  --text-prompts-json /dev/stdin <<'EOF'
{"0": ["cinematic coastal lighthouse, dusk fog, volumetric light, wide angle"]}
EOF
```

## Platform

**Linux tested.** This repo is maintained and expected to run on **glibc-based Linux** with a working **NVIDIA stack** (proprietary driver so `nvidia-smi` reports your GPU). Other OSes are currently out of scope; if you run elsewhere, you may see a stderr warning - set `DISCO_ALLOW_NON_LINUX=1` to suppress it (still unsupported).

## RunConfig

Top-level runtime settings are now centralized in `RunConfig` at `src/discodiff/config/settings.py`.

Setting | Env var | Default | Notes
--- | --- | --- | ---
`output_dir` | `DISCO_OUTPUT_DIR` | `<repo>/output` | Output root for generated assets.
`device` | `DISCO_DEVICE` | `auto` | `auto` resolves to CUDA when available, else CPU.
`seed` | `DISCO_SEED` | `None` | Optional integer; runtime still supports random seed behavior.
`generation_mode` | `DISCO_GENERATION_MODE` | `None` | One of `None`, `2D`, `3D`, `Video Input`.
`profile` | `DISCO_PROFILE` | `default` | Reserved label for future preset/profile behavior.
