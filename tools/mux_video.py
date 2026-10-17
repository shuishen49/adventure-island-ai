"""Mux a silent recording with the jsnes-recovered NES soundtrack.

    python tools/mux_video.py video.mp4 game_audio.wav out.mp4 [--trim]

The video is written at the true NTSC rate (60.0988 fps) so the two
timelines already line up; ffmpeg only has to pair them.  `--trim` cuts
the audio to the video length (default: cut to the shorter of the two).
"""
from __future__ import annotations

import os
import subprocess
import sys

import imageio_ffmpeg


def main() -> None:
    video = sys.argv[1]
    audio = sys.argv[2]
    out = sys.argv[3]
    exe = imageio_ffmpeg.get_ffmpeg_exe()
    cmd = [exe, "-y", "-i", video, "-i", audio,
           "-map", "0:v:0", "-map", "1:a:0",
           "-c:v", "copy", "-c:a", "aac", "-b:a", "192k",
           "-shortest", out]
    print(" ".join(cmd))
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        print(r.stdout[-2000:])
        print(r.stderr[-2000:])
        sys.exit(1)
    print(f"{out}  {os.path.getsize(out) / 1e6:.1f} MB")


if __name__ == "__main__":
    main()
