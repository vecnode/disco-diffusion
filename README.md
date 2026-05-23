# Disco Diffusion 2

3D warp state management + latent img2img generation per frame.

Specific support for RTX Ampere, tests under 24Gb and CUDA 12.8.

## Reproduce

```sh
# Sync environment from lockfile and run
uv sync
uv run main.py
```

## Text-to-video (3D_latent)

```sh
# Latent
uv run main.py --width 1024 --height 576 \
  --max-frames 120 --set-seed 42 --steps 100 \
  --latent-first-frame txt2img \
  --latent-strength 0.5 --latent-temporal-blend 0.2 \
  --latent-novelty-strength 0.03 --latent-color-reset 0.06 \
  --text-prompts-json /dev/stdin \
  <<'EOF'
{"0": ["a person in the city, dusk fog, volumetric light, wide angle"], "40": ["a person in the sea, volumetric light, wide angle"]}
EOF

# Make video
ffmpeg -framerate 25 -pattern_type glob -i "./output/example/render/*.png" -c:v libx264 -preset slow -crf 12 -pix_fmt yuv444p -movflags +faststart "./output/example/render_25fps.mp4"
```

### Device Selection

You can let the runtime auto-select or force a device explicitly.

```sh
# Auto-select (prefers RTX CUDA when available)
uv run main.py --device auto

# Explicit device requests
uv run main.py --device rtx
uv run main.py --device cuda:0
uv run main.py --device cpu
```
