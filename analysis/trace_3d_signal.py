#!/usr/bin/env python3
"""Trace and quantify temporal behavior in Disco Diffusion 3D runs.

This script analyzes a generation output folder and writes:
- metrics.csv: per-frame signals
- summary.json: aggregate stats and detected events
- report.md: short interpretation
- plots/*.png: line-plot images rendered with Matplotlib
"""

from __future__ import annotations

import argparse
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import cv2
import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt


@dataclass
class RunContext:
	run_dir: Path
	settings_path: Optional[Path]
	main_frames: List[Path]
	stages: Dict[str, Dict[int, Path]]


def parse_args() -> argparse.Namespace:
	parser = argparse.ArgumentParser(
		description=(
			"Analyze Disco Diffusion 3D output and trace frame consistency/noise signals"
		)
	)
	parser.add_argument(
		"run_dir",
		type=Path,
		help="Path to a run folder (e.g. output/example)",
	)
	parser.add_argument(
		"--out-dir",
		type=Path,
		default=None,
		help="Directory for analysis artifacts (default: <run_dir>/analysis_trace)",
	)
	parser.add_argument(
		"--max-frames",
		type=int,
		default=None,
		help="Optional cap for processed frames (for quick runs)",
	)
	parser.add_argument(
		"--resize-width",
		type=int,
		default=480,
		help="Resize width used for metric computation (default: 480)",
	)
	return parser.parse_args()


def _is_main_frame(path: Path) -> bool:
	name = path.name
	if not name.lower().endswith(".png"):
		return False
	if "_settings" in name:
		return False
	if name.startswith("frame_"):
		return False
	m = re.search(r"_(\d{4})\.png$", name)
	return m is not None


def _main_frame_index(path: Path) -> int:
	m = re.search(r"_(\d{4})\.png$", path.name)
	if m is None:
		return -1
	return int(m.group(1))


def _parse_stage_file(path: Path) -> Tuple[Optional[int], Optional[str]]:
	m = re.match(r"frame_(\d{4})_(.+)\.png$", path.name)
	if m is None:
		return None, None
	idx = int(m.group(1))
	stage = m.group(2)
	return idx, stage


def discover_run(run_dir: Path, max_frames: Optional[int]) -> RunContext:
	if not run_dir.exists() or not run_dir.is_dir():
		raise FileNotFoundError(f"Run directory not found: {run_dir}")

	settings_candidates = sorted(run_dir.glob("*_settings.txt"))
	settings_path = settings_candidates[0] if settings_candidates else None

	main_frames = sorted(
		[p for p in run_dir.glob("*.png") if _is_main_frame(p)], key=_main_frame_index
	)
	if max_frames is not None and max_frames > 0:
		main_frames = main_frames[:max_frames]

	stages: Dict[str, Dict[int, Path]] = {}
	stage_dir = run_dir / "3d"
	if stage_dir.exists() and stage_dir.is_dir():
		for p in stage_dir.glob("frame_*.png"):
			idx, stage = _parse_stage_file(p)
			if idx is None or stage is None:
				continue
			if max_frames is not None and idx >= max_frames:
				continue
			stages.setdefault(stage, {})[idx] = p

	if not main_frames:
		raise RuntimeError(
			f"No main frames found in {run_dir}. Expected files like *_0000.png"
		)

	return RunContext(
		run_dir=run_dir,
		settings_path=settings_path,
		main_frames=main_frames,
		stages=stages,
	)


def read_image(path: Path, target_width: int) -> np.ndarray:
	img = cv2.imread(str(path), cv2.IMREAD_COLOR)
	if img is None:
		raise RuntimeError(f"Could not read image: {path}")

	h, w = img.shape[:2]
	if target_width > 0 and w != target_width:
		scale = target_width / float(w)
		target_height = max(8, int(h * scale))
		img = cv2.resize(img, (target_width, target_height), interpolation=cv2.INTER_AREA)
	return img


def grayscale(img: np.ndarray) -> np.ndarray:
	return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.float32)


def norm_corr(a: np.ndarray, b: np.ndarray) -> float:
	a_f = a.reshape(-1)
	b_f = b.reshape(-1)
	a_c = a_f - a_f.mean()
	b_c = b_f - b_f.mean()
	den = float(np.linalg.norm(a_c) * np.linalg.norm(b_c))
	if den < 1e-8:
		return 0.0
	return float(np.dot(a_c, b_c) / den)


def edge_energy(gray: np.ndarray) -> float:
	gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
	gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
	mag = cv2.magnitude(gx, gy)
	return float(np.mean(mag))


def high_freq_energy(gray: np.ndarray) -> float:
	blur = cv2.GaussianBlur(gray, (0, 0), sigmaX=1.2, sigmaY=1.2)
	hp = gray - blur
	return float(np.mean(np.abs(hp)))


def diff_metrics(a: np.ndarray, b: np.ndarray) -> Dict[str, float]:
	a_g = grayscale(a)
	b_g = grayscale(b)
	absdiff = np.abs(b_g - a_g)
	return {
		"mad": float(np.mean(absdiff)),
		"rmse": float(np.sqrt(np.mean((b_g - a_g) ** 2))),
		"corr": norm_corr(a_g, b_g),
		"edge_delta": abs(edge_energy(b_g) - edge_energy(a_g)),
		"hf_delta": abs(high_freq_energy(b_g) - high_freq_energy(a_g)),
		"mean_luma": float(np.mean(b_g)),
	}


def read_settings(path: Optional[Path]) -> Dict[str, object]:
	if path is None:
		return {}
	try:
		return json.loads(path.read_text(encoding="utf-8"))
	except Exception:
		return {}


def robust_z(values: Sequence[float]) -> np.ndarray:
	arr = np.asarray(values, dtype=np.float32)
	med = np.median(arr)
	mad = np.median(np.abs(arr - med))
	scale = 1.4826 * mad
	if scale < 1e-8:
		return np.zeros_like(arr)
	return (arr - med) / scale


def compute_main_frame_metrics(ctx: RunContext, resize_width: int) -> pd.DataFrame:
	rows: List[Dict[str, float]] = []

	prev_img = read_image(ctx.main_frames[0], resize_width)
	frame0 = _main_frame_index(ctx.main_frames[0])
	rows.append(
		{
			"frame": float(frame0),
			"mad": 0.0,
			"rmse": 0.0,
			"corr": 1.0,
			"edge_delta": 0.0,
			"hf_delta": 0.0,
			"mean_luma": float(np.mean(grayscale(prev_img))),
		}
	)

	for path in ctx.main_frames[1:]:
		idx = _main_frame_index(path)
		cur = read_image(path, resize_width)
		m = diff_metrics(prev_img, cur)
		m["frame"] = float(idx)
		rows.append(m)
		prev_img = cur

	df = pd.DataFrame(rows).sort_values("frame").reset_index(drop=True)
	return df


def stage_change_metric(stage_files: Dict[int, Path], resize_width: int) -> pd.Series:
	if not stage_files:
		return pd.Series(dtype=np.float32)

	ordered = sorted(stage_files.items(), key=lambda kv: kv[0])
	out: Dict[int, float] = {}

	prev_idx, prev_path = ordered[0]
	prev = read_image(prev_path, resize_width)
	out[prev_idx] = 0.0
	for idx, path in ordered[1:]:
		cur = read_image(path, resize_width)
		out[idx] = diff_metrics(prev, cur)["mad"]
		prev = cur
	return pd.Series(out, name="mad")


def flow_strength(stage_files: Dict[int, Path], resize_width: int) -> pd.Series:
	if not stage_files:
		return pd.Series(dtype=np.float32)
	out: Dict[int, float] = {}
	for idx, path in stage_files.items():
		img = read_image(path, resize_width)
		gray = grayscale(img)
		out[idx] = float(np.mean(gray))
	return pd.Series(out, name="flow_strength")


def correlate_cols(df: pd.DataFrame, target: str, cols: Sequence[str]) -> Dict[str, float]:
	corr: Dict[str, float] = {}
	target_arr = df[target].to_numpy(dtype=np.float32)
	for c in cols:
		if c not in df.columns:
			continue
		vals = df[c].to_numpy(dtype=np.float32)
		if np.std(vals) < 1e-8 or np.std(target_arr) < 1e-8:
			corr[c] = 0.0
		else:
			corr[c] = float(np.corrcoef(target_arr, vals)[0, 1])
	return corr


def draw_line_plot(
	x: np.ndarray,
	series: Sequence[Tuple[str, np.ndarray, Tuple[int, int, int]]],
	title: str,
	out_file: Path,
) -> None:
	plt.style.use("seaborn-v0_8-whitegrid")
	fig, ax = plt.subplots(figsize=(14, 7.8), dpi=120)

	for label, arr, color in series:
		# Existing palette values are in 0-255 BGR order; convert to Matplotlib RGB.
		b, g, r = color
		rgb = (r / 255.0, g / 255.0, b / 255.0)
		ax.plot(x, arr, label=label, linewidth=2.0, color=rgb)

	ax.set_title(title, fontsize=16)
	ax.set_xlabel("Frame index", fontsize=12)
	ax.set_ylabel("Signal value", fontsize=12)
	ax.tick_params(axis="both", labelsize=10)
	ax.legend(loc="best", frameon=True)
	ax.margins(x=0.01)
	fig.tight_layout()
	fig.savefig(out_file, dpi=160)
	plt.close(fig)


def write_report(
	report_path: Path,
	settings: Dict[str, object],
	df: pd.DataFrame,
	noisy_frames: List[int],
	stable_frames: List[int],
	corr: Dict[str, float],
) -> None:
	lines: List[str] = []
	lines.append("# 3D Generation Trace Report")
	lines.append("")
	lines.append("## Run Summary")
	lines.append("")
	lines.append(f"- Frames analyzed: {len(df)}")
	lines.append(f"- MAD mean: {df['mad'].mean():.4f}")
	lines.append(f"- MAD std: {df['mad'].std(ddof=0):.4f}")
	lines.append(f"- Correlation mean: {df['corr'].mean():.4f}")
	lines.append(f"- High-frequency delta mean: {df['hf_delta'].mean():.4f}")
	if "steps" in settings:
		lines.append(f"- Diffusion steps: {settings.get('steps')}")
	if "turbo_mode" in settings:
		lines.append(f"- Turbo mode: {settings.get('turbo_mode')}")
	if "turbo_steps" in settings:
		lines.append(f"- Turbo steps: {settings.get('turbo_steps')}")
	if "midas_weight" in settings:
		lines.append(f"- MiDaS weight: {settings.get('midas_weight')}")
	if "frames_scale" in settings:
		lines.append(f"- Frame consistency scale: {settings.get('frames_scale')}")

	lines.append("")
	lines.append("## Interpreting Signals")
	lines.append("")
	lines.append("- Higher `mad` and `rmse` indicate stronger frame-to-frame change.")
	lines.append("- Lower `corr` indicates weaker temporal consistency.")
	lines.append("- Higher `hf_delta` often reflects texture flicker/noise changes.")
	lines.append("- Stage deltas (`source`, `warped`, `depth_*`) can show where variance enters.")

	lines.append("")
	lines.append("## Detected Events")
	lines.append("")
	if noisy_frames:
		lines.append(f"- Candidate noisy frames (MAD outliers): {noisy_frames}")
	else:
		lines.append("- Candidate noisy frames: none detected")
	if stable_frames:
		lines.append(f"- Most stable frames (high corr / low MAD): {stable_frames}")
	else:
		lines.append("- Most stable frames: none detected")

	lines.append("")
	lines.append("## Correlation With Main Instability")
	lines.append("")
	if corr:
		for k, v in sorted(corr.items(), key=lambda kv: abs(kv[1]), reverse=True):
			lines.append(f"- {k}: {v:.4f}")
	else:
		lines.append("- No stage metrics were available for correlation.")

	report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
	args = parse_args()
	run_dir = args.run_dir.resolve()
	out_dir = (args.out_dir or (run_dir / "analysis_trace")).resolve()
	plots_dir = out_dir / "plots"
	out_dir.mkdir(parents=True, exist_ok=True)
	plots_dir.mkdir(parents=True, exist_ok=True)

	ctx = discover_run(run_dir, max_frames=args.max_frames)
	settings = read_settings(ctx.settings_path)
	df = compute_main_frame_metrics(ctx, resize_width=args.resize_width)

	# Normalize core instability signals to compare on one plot.
	for col in ("mad", "rmse", "hf_delta", "edge_delta"):
		df[f"z_{col}"] = robust_z(df[col].values)
	df["z_inv_corr"] = robust_z((1.0 - df["corr"]).values)

	# Stage-specific per-frame variation signals.
	for stage, stage_files in ctx.stages.items():
		if stage == "flow_field":
			s = flow_strength(stage_files, resize_width=args.resize_width)
			df[f"stage_{stage}_strength"] = df["frame"].map(s).fillna(0.0)
			df[f"z_stage_{stage}_strength"] = robust_z(df[f"stage_{stage}_strength"].values)
		else:
			s = stage_change_metric(stage_files, resize_width=args.resize_width)
			df[f"stage_{stage}_mad"] = df["frame"].map(s).fillna(0.0)
			df[f"z_stage_{stage}_mad"] = robust_z(df[f"stage_{stage}_mad"].values)

	noise_score = (
		0.40 * df["z_mad"]
		+ 0.25 * df["z_inv_corr"]
		+ 0.20 * df["z_hf_delta"]
		+ 0.15 * df["z_edge_delta"]
	)
	df["noise_score"] = noise_score

	# Detect notable changes.
	noisy_mask = df["noise_score"] > 2.0
	noisy_frames = [int(v) for v in df.loc[noisy_mask, "frame"].tolist()]

	stable_mask = (df["corr"] > df["corr"].quantile(0.90)) & (
		df["mad"] < df["mad"].quantile(0.25)
	)
	stable_frames = [int(v) for v in df.loc[stable_mask, "frame"].tolist()]

	stage_cols = [
		c
		for c in df.columns
		if c.startswith("stage_") and (c.endswith("_mad") or c.endswith("_strength"))
	]
	corr = correlate_cols(df, target="noise_score", cols=stage_cols)

	metrics_csv = out_dir / "metrics.csv"
	summary_json = out_dir / "summary.json"
	report_md = out_dir / "report.md"

	df.to_csv(metrics_csv, index=False)

	summary = {
		"run_dir": str(run_dir),
		"frame_count": int(len(df)),
		"settings_file": str(ctx.settings_path) if ctx.settings_path else None,
		"signals": {
			"mad_mean": float(df["mad"].mean()),
			"mad_std": float(df["mad"].std(ddof=0)),
			"corr_mean": float(df["corr"].mean()),
			"noise_score_mean": float(df["noise_score"].mean()),
			"noise_score_std": float(df["noise_score"].std(ddof=0)),
		},
		"noisy_frames": noisy_frames,
		"stable_frames": stable_frames,
		"stage_correlation_with_noise": corr,
	}
	summary_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")

	write_report(
		report_path=report_md,
		settings=settings,
		df=df,
		noisy_frames=noisy_frames,
		stable_frames=stable_frames,
		corr=corr,
	)

	x = df["frame"].to_numpy(dtype=np.float32)
	draw_line_plot(
		x,
		[
			("MAD", df["mad"].to_numpy(dtype=np.float32), (54, 100, 240)),
			("RMSE", df["rmse"].to_numpy(dtype=np.float32), (38, 166, 91)),
			("1-Corr", (1.0 - df["corr"]).to_numpy(dtype=np.float32), (214, 103, 23)),
		],
		"Temporal Difference Signals (main frames)",
		plots_dir / "temporal_signals.png",
	)

	draw_line_plot(
		x,
		[
			("Noise score", df["noise_score"].to_numpy(dtype=np.float32), (53, 53, 196)),
			("z(MAD)", df["z_mad"].to_numpy(dtype=np.float32), (189, 33, 84)),
			("z(1-Corr)", df["z_inv_corr"].to_numpy(dtype=np.float32), (214, 103, 23)),
		],
		"Composite Instability Score",
		plots_dir / "noise_score.png",
	)

	z_stage_cols = [c for c in df.columns if c.startswith("z_stage_")]
	if z_stage_cols:
		palette = [
			(54, 100, 240),
			(38, 166, 91),
			(214, 103, 23),
			(189, 33, 84),
			(117, 117, 117),
			(0, 150, 136),
			(149, 117, 205),
		]
		series = []
		for i, col in enumerate(sorted(z_stage_cols)):
			label = col.replace("z_stage_", "")
			series.append((label, df[col].to_numpy(dtype=np.float32), palette[i % len(palette)]))
		draw_line_plot(
			x,
			series,
			"Intermediate Stage Signals (z-scored)",
			plots_dir / "stage_signals.png",
		)

	print("Trace analysis completed")
	print(f"Run dir: {run_dir}")
	print(f"Artifacts: {out_dir}")
	print(f"- {metrics_csv.name}")
	print(f"- {summary_json.name}")
	print(f"- {report_md.name}")
	print(f"- plots/{'temporal_signals.png'}")
	print(f"- plots/{'noise_score.png'}")
	if (plots_dir / "stage_signals.png").exists():
		print(f"- plots/{'stage_signals.png'}")


if __name__ == "__main__":
	main()
