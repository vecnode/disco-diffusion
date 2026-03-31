"""FFmpeg helpers for frame sequences."""

from __future__ import annotations

import subprocess


def encode_numbered_png_sequence_h264(
    *,
    cwd: str,
    image_sequence_pattern: str,
    output_path: str,
    fps: float = 12.0,
    start_number: int = 1,
    frames_v: int,
    crf: str = "17",
    preset: str = "veryslow",
) -> None:
    """Assemble a numbered PNG sequence into an H.264 MP4 (yuv420p), matching the notebook ffmpeg recipe."""
    cmd = [
        "ffmpeg",
        "-y",
        "-vcodec",
        "png",
        "-r",
        str(fps),
        "-start_number",
        str(start_number),
        "-i",
        image_sequence_pattern,
        "-frames:v",
        str(frames_v),
        "-c:v",
        "libx264",
        "-vf",
        f"fps={fps}",
        "-pix_fmt",
        "yuv420p",
        "-crf",
        crf,
        "-preset",
        preset,
        output_path,
    ]
    process = subprocess.Popen(cmd, cwd=cwd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    _, stderr = process.communicate()
    if process.returncode != 0:
        print(stderr)
        raise RuntimeError(stderr)
    print("The video is ready and saved to the images folder")
