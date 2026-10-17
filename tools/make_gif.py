"""Turn a recorded mp4 into a small, README-friendly GIF.

Two-pass ffmpeg (palettegen + paletteuse) with dithering switched off:
dithering is what makes NES footage explode in size, because every flat
colour area turns into noise that the LZW pass cannot compress.

    python tools/make_gif.py outputs/v1-3_player_sound.mp4 \
        --out docs/images/1-3_best.gif --fps 8 --scale 384 --colors 48
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys

import imageio_ffmpeg


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("src")
    ap.add_argument("--out", required=True)
    ap.add_argument("--fps", type=int, default=8)
    ap.add_argument("--scale", type=int, default=384)
    ap.add_argument("--colors", type=int, default=48)
    args = ap.parse_args()

    exe = imageio_ffmpeg.get_ffmpeg_exe()
    os.makedirs(os.path.dirname(os.path.abspath(args.out)) or ".",
                exist_ok=True)
    vf = (f"fps={args.fps},scale={args.scale}:-1:flags=lanczos,"
          f"split[s0][s1];[s0]palettegen=max_colors={args.colors}[p];"
          f"[s1][p]paletteuse=dither=none")
    cmd = [exe, "-y", "-i", args.src, "-vf", vf, "-loop", "0", args.out]
    subprocess.run(cmd, check=True,
                   stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT)
    mb = os.path.getsize(args.out) / 1e6
    print(f"wrote {args.out}  {mb:.1f} MB", file=sys.stderr)


if __name__ == "__main__":
    main()
