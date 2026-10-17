"""Cluster the 32 stages by how far a fixed action script actually gets.

First-frame pixel diff only compares the spawn terrain.  Two stages can
share an opening screen and still diverge later, so this drives every
stage with the SAME deterministic action script and records the real world
progress (``CameraTracker.progress`` -- NOT the on-screen x, which the
camera pins near 96 and is therefore useless as a fingerprint).

Stages whose whole progress trace matches frame-for-frame are the same
stage as far as training is concerned: a policy that clears one clears the
other, so the 32-row curriculum in plan.md may collapse to fewer rows.

usage:
    python tools/cluster_stages.py
"""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
sys.path.insert(0, _ROOT)

from cynes import NES, NES_INPUT_A, NES_INPUT_RIGHT  # noqa: E402

import ram_map as R  # noqa: E402
from env import ROM  # noqa: E402

SAVE_DIR = os.path.join(_ROOT, "saves", "stages")
STAGES = [f"{a}-{r}" for a in range(1, 9) for r in range(1, 5)]

# a fixed script: mostly right, a jump every 3rd beat
PATTERN = [NES_INPUT_RIGHT, NES_INPUT_RIGHT,
           NES_INPUT_RIGHT | NES_INPUT_A,
           NES_INPUT_RIGHT, NES_INPUT_RIGHT,
           NES_INPUT_RIGHT | NES_INPUT_A]


def trace(stage: str, frames: int = 900, beat: int = 20):
    nes = NES(ROM)
    with open(os.path.join(SAVE_DIR, f"stage_{stage}.state"), "rb") as fh:
        nes.load(np.frombuffer(fh.read(), dtype=np.uint8).copy())
    cam = R.CameraTracker(nes)
    out = []
    death_at = None
    for f in range(frames):
        nes.controller = PATTERN[(f // beat) % len(PATTERN)]
        nes.step()
        nes.controller = 0
        out.append(cam.progress(nes))
        if death_at is None and nes[R.DEATH] != 0 \
                and nes[R.STAGE_COMPLETE] == 0:
            death_at = f
    del nes
    return np.asarray(out), death_at


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--frames", type=int, default=900)
    args = ap.parse_args()

    traces, deaths = {}, {}
    for s in STAGES:
        t, d = trace(s, args.frames)
        traces[s], deaths[s] = t, d
        print(f"  {s}: max_progress={t.max():7.1f}px  died_at="
              f"{'--' if d is None else str(d) + 'f'}")

    clusters: list[list[str]] = []
    for s in STAGES:
        for c in clusters:
            rep = traces[c[0]]
            t = traces[s]
            n = min(len(rep), len(t))
            if n and np.abs(rep[:n] - t[:n]).max() < 1.0 \
                    and deaths[s] == deaths[c[0]]:
                c.append(s)
                break
        else:
            clusters.append([s])

    print(f"\n=== {len(clusters)} distinct stages out of {len(STAGES)} ===")
    for c in clusters:
        print("  " + " = ".join(c))


if __name__ == "__main__":
    main()
