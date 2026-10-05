"""Pure helpers: frame math, long-shot splitting, ffmpeg concat. No torch."""

from __future__ import annotations

import math
import random
import subprocess
from dataclasses import dataclass
from pathlib import Path

from PIL import Image

AUDIO_XFADE_S = 0.05


class InvalidInput(ValueError):
    """Unusable request input (bad image URL...). Surfaces as `invalid_input: ...`."""


class GenerationOOM(RuntimeError):
    """CUDA OOM during a segment; the loaded pipeline stays usable."""


@dataclass
class SegmentJob:
    prompt: str
    num_frames: int
    width: int
    height: int
    first: Image.Image | None
    last: Image.Image | None
    seed: int
    out_path: str


def frames_for(duration_s: float, fps: int, mult: int = 8, offset: int = 1) -> int:
    return max(1, round(duration_s * fps / mult)) * mult + offset


def round_size(w: int, h: int, m: int = 32) -> tuple[int, int]:
    return (w // m) * m, (h // m) * m


def pick_seed(seed: int | None) -> int:
    return seed if seed is not None else random.randint(0, 2**31 - 1)


def split_duration(duration_s: float, max_native_s: float) -> list[float]:
    """Equal parts, each <= max_native_s (no tiny tail segment)."""
    n = max(1, math.ceil(duration_s / max_native_s - 1e-9))
    return [duration_s / n] * n


def last_frame(video: str, out_png: str) -> str:
    subprocess.run(
        ["ffmpeg", "-y", "-loglevel", "error", "-sseof", "-1", "-i", video,
         "-update", "1", "-q:v", "1", out_png],
        check=True,
    )  # fmt: skip
    return out_png


def concat_cmd(parts: list[str], out: str, overlap_frames: int, audio: bool) -> list[str]:
    """Join segments: drop the duplicated first frame(s) of every later segment (it is the
    previous segment's last frame) and crossfade audio at the seams."""
    cmd = ["ffmpeg", "-y", "-loglevel", "error"]
    for p in parts:
        cmd += ["-i", p]
    vf = []
    for i in range(len(parts)):
        trim = f"trim=start_frame={overlap_frames}," if i else ""
        vf.append(f"[{i}:v]{trim}setpts=PTS-STARTPTS[v{i}]")
    fc = vf + ["".join(f"[v{i}]" for i in range(len(parts))) + f"concat=n={len(parts)}:v=1[v]"]
    maps = ["-map", "[v]"]
    if audio:
        prev = "0:a"
        for i in range(1, len(parts)):
            fc.append(f"[{prev}][{i}:a]acrossfade=d={AUDIO_XFADE_S}[a{i}]")
            prev = f"a{i}"
        maps += ["-map", f"[{prev}]" if len(parts) > 1 else "0:a"]
    return [*cmd, "-filter_complex", ";".join(fc), *maps,
            "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "16", *(["-c:a", "aac"] if audio else []),
            out]  # fmt: skip


def concat(parts: list[str], out: str, overlap_frames: int, audio: bool) -> str:
    if len(parts) == 1:
        Path(out).write_bytes(Path(parts[0]).read_bytes())
        return out
    subprocess.run(concat_cmd(parts, out, overlap_frames, audio), check=True)
    return out
