# mixed-diffusion

Under heavy development. 

Specific support for RTX Ampere, tests under 24Gb.

## Reproduce

```sh
# Sync environment from lockfile and run
uv sync
uv run disco.py
# equivalent:
uv run python -m discodiff.main
```

## Text-to-video (3D_latent)

```sh

# Latent

# Optional overrides:
#   --depth-backend marigold|adabins
#   --latent-first-frame txt2img|black
#   --latent-strength 0.25            # lower = more temporal coherence
#   --latent-temporal-blend 0.20      # blend warped prev frame into output
#   --latent-novelty-strength 0.03     # slowly increases novelty over time
#   --latent-color-reset 0.06          # suppress saturation drift over long runs
#   DISCO_MARIGOLD_MODEL=prs-eth/marigold-depth-lcm-v1-0
#   DISCO_MARIGOLD_MODEL_DIR=/absolute/path/to/local/marigold
#   DISCO_MARIGOLD_DEPTH_CONTRAST=1.35  # increase perceived depth parallax
#   DISCO_MARIGOLD_INVERT_DEPTH=0       # set to 1 only if scene depth is reversed

uv run disco.py --width 1024 --height 576 \
  --max-frames 120 --set-seed 42 --steps 100 \
  --latent-first-frame txt2img \
  --latent-strength 0.5 --latent-temporal-blend 0.2 \
  --latent-novelty-strength 0.03 --latent-color-reset 0.06 \
  --text-prompts-json /dev/stdin \
  <<'EOF'
{"0": ["cinematic coastal lighthouse, dusk fog, volumetric light, wide angle"], "40": ["the sea, volumetric light, wide angle"]}
EOF

# Make video

ffmpeg -framerate 25 -pattern_type glob -i "./output/example/render/*.png" -c:v libx264 -preset slow -crf 12 -pix_fmt yuv444p -movflags +faststart "./output/example/render_25fps.mp4"

```



### Device Selection

You can let the runtime auto-select or force a device explicitly.

```sh
# Auto-select (prefers RTX CUDA when available)
uv run disco.py --device auto

# Explicit device requests
uv run disco.py --device rtx
uv run disco.py --device cuda:0
uv run disco.py --device cpu
```
