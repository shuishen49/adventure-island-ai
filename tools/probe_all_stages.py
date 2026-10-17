"""Scan all 8 areas x 4 rounds = 32 stages by warping $0037/$0038.

Why this matters: plan.md lists "the warp only reaches the first few
stages" as the single biggest risk of phase 0 -- if it failed, training
would have to be STRICTLY SEQUENTIAL (clear stage N to even get a save
state for stage N+1), roughly doubling the wall-clock cost.

Finding (2026-10-17): the stage index is ``area * 4 + round``.  Warping
(area=0, round=4) and (area=1, round=0) produce byte-identical stages
(same spawn, same brightness), and round only has 4 valid values -- a
round of 4..7 rolls into the next area.  So writing both registers at
stage init reaches all 32 stages directly.

usage:
    python tools/probe_all_stages.py            # all 32, + diff matrix
    python tools/probe_all_stages.py --diff-only
"""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np
from PIL import Image

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
sys.path.insert(0, _ROOT)

from cynes import NES, NES_INPUT_START  # noqa: E402

import ram_map as R  # noqa: E402
from env import ROM, control_live, wait_for_picture  # noqa: E402

SHOT_DIR = os.path.join(_ROOT, "outputs", "probe_stages")
SAVE_DIR = os.path.join(_ROOT, "saves", "stages")


def warp_to(area: int, rnd: int):
    nes = NES(ROM)
    nes.step(frames=300)                       # title screen
    nes.controller = NES_INPUT_START
    for _ in range(90):
        nes[R.AREA] = area                     # hammer both through init
        nes[R.ROUND] = rnd
        nes.step()
    nes.controller = 0
    for _ in range(600):
        if control_live(nes):
            break
        for _ in range(10):
            nes.controller = 0
            nes.step()
    else:
        del nes
        return None, "never got control"
    wait_for_picture(nes)
    if not control_live(nes):
        del nes
        return None, "lost control during fade-in"
    return nes, None


def smoke(nes, frames: int = 240):
    """Hold RIGHT for `frames`; does a random-ish warm-up kill us?"""
    st = nes.save()
    nes.controller = 0x01
    for _ in range(frames):
        nes.step()
    nes.controller = 0
    dead = nes[R.DEATH] != 0 and nes[R.STAGE_COMPLETE] == 0
    px = R.px(R.read_obj_world_x(nes, R.PLAYER_SLOT))
    nes.load(st)
    return dead, px


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--areas", type=int, default=8)
    ap.add_argument("--rounds", type=int, default=4)
    ap.add_argument("--no-save", action="store_true",
                    help="do not dump per-stage save states")
    args = ap.parse_args()
    os.makedirs(SHOT_DIR, exist_ok=True)
    os.makedirs(SAVE_DIR, exist_ok=True)

    table = []
    frames = {}
    for a in range(args.areas):
        for r in range(args.rounds):
            label = f"{a + 1}-{r + 1}"
            nes, err = warp_to(a, r)
            if nes is None:
                print(f"{label:>6}  FAILED: {err}")
                table.append((label, None))
                continue
            arr = np.asarray(nes.step(), dtype=np.uint8).copy()
            px = (R.px(R.read_obj_world_x(nes, R.PLAYER_SLOT)),
                  R.px(R.read_obj_world_y(nes, R.PLAYER_SLOT)))
            dead, x_after = smoke(nes)
            Image.fromarray(arr, mode="RGB").resize(
                (512, 480), Image.NEAREST).save(
                    os.path.join(SHOT_DIR, f"stage_{label}.png"))
            if not args.no_save:
                with open(os.path.join(SAVE_DIR,
                                       f"stage_{label}.state"), "wb") as fh:
                    fh.write(nes.save())
            frames[label] = arr
            print(f"{label:>6}  area={nes[R.AREA]} round={nes[R.ROUND]} "
                  f"spawn=({px[0]:.0f},{px[1]:.0f}) bright={arr.mean():5.1f} "
                  f"live=yes  warmup_dead={'YES' if dead else 'no':>3} "
                  f"x_after_240f={x_after:.0f}")
            del nes

    # pairwise pixel diff: proves the 32 warps really are 32 DIFFERENT
    # stages and not the same map re-skinned
    labels = list(frames)
    print("\n=== pairwise pixel diff (%% of pixels differing >24) ===")
    dupes = []
    for i, l1 in enumerate(labels):
        row = []
        for l2 in labels:
            d = np.abs(frames[l1].astype(np.int16)
                       - frames[l2].astype(np.int16)).max(axis=2)
            row.append(float((d > 24).mean()))
        print(f"{l1:>6} " + " ".join(f"{v * 100:4.0f}" for v in row))
        for j, v in enumerate(row):
            if j > i and v < 0.02:
                dupes.append((l1, labels[j], v))
    if dupes:
        print("\n!! near-identical stage pairs (possible warp failure):")
        for l1, l2, v in dupes:
            print(f"   {l1} == {l2}  ({v * 100:.2f}% differ)")
    else:
        print("\nall %d stages are pixel-wise distinct" % len(labels))


if __name__ == "__main__":
    main()
