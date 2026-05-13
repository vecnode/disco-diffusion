# Analysis scripts

## 3D generation trace

Use `trace_3d_signal.py` to inspect frame-to-frame consistency, detect noise spikes,
and correlate instability with intermediate 3D artifacts (`source`, `warped`, `depth_*`, `flow_field`).

### Run

From repository root:

```sh
uv run python analysis/trace_3d_signal.py output/example
```

Custom output folder:

```sh
uv run python analysis/trace_3d_signal.py output/example --out-dir analysis/example_trace
```

Quick pass (first 60 frames only):

```sh
uv run python analysis/trace_3d_signal.py output/example --max-frames 60
```

### Outputs

By default, artifacts are written under `<run_dir>/analysis_trace/`:

- `metrics.csv`: per-frame signals (`mad`, `rmse`, `corr`, `hf_delta`, stage metrics, noise score)
- `summary.json`: aggregate stats, detected noisy frames, stable frames, stage/noise correlations
- `report.md`: concise interpretation for this run
- `plots/temporal_signals.png`: main temporal difference graph
- `plots/noise_score.png`: composite instability graph
- `plots/stage_signals.png`: z-scored intermediate stage signals (if `3d/frame_*` files exist)

Plots are rendered with Matplotlib and written as PNG files.

### Interpreting graphs

- High `MAD` and `RMSE` means stronger visual change from one frame to the next.
- High `1-Corr` means lower temporal consistency.
- High composite `noise_score` marks candidate flicker/noise events.
- Stage signal peaks that align with `noise_score` spikes indicate where instability is likely introduced:
	- `stage_source_mad` spikes: source frame changes are already strong
	- `stage_warped_mad` spikes: geometry/warp likely contributes
	- `stage_depth_*_mad` spikes: depth changes likely contribute
	- `stage_flow_field_strength` spikes: optical flow magnitude changes
