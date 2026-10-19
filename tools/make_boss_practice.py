"""Snapshot a 1-4 state where the boss is still FAR, for fight practice.

The first attempt (a copy of stage_1-4_bossface.state, boss at +99 px) was
unusable: env.reset() plays up to WARMUP_MAX frames of "run right" before
the episode starts, which closed the gap to 28 px and the agent was dead on
its very first decision.  The boss walks toward you, so the snapshot has to
be taken BEFORE it closes in.

Measured: in saves/stages/stage_1-4_safe5400.state the boss (id 198) first
appears at dx = 135 px (frame 165 of a run-right script) and then patrols
between about 18 and 81 px forever.  So we snapshot the moment it appears,
with Higgins on the ground.

    python tools/make_boss_practice.py --out saves/stages/stage_1-4boss.state
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
sys.path.insert(0, _ROOT)

import ram_map as R                                        # noqa: E402
from env import (ROM, NES_INPUT_A, NES_INPUT_RIGHT,        # noqa: E402
                 load_state_file)
from cynes import NES                                      # noqa: E402

ID, XL, YL = R.OBJ.id, R.OBJ.x_lo, R.OBJ.y_lo
HEAD_ID = R.BOSS_HEAD_ID
GROUND_Y = 138          # screen y; standing on the floor is 144


def head_dx(nes) -> int | None:
    px = int(nes[XL])
    for s in range(1, 16):
        if int(nes[ID + R.OBJ.STRIDE * s]) == HEAD_ID:
            dx = int(nes[XL + R.OBJ.STRIDE * s]) - px
            return dx - 256 if dx > 128 else dx
    return None


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--from-state", default=os.path.join(
        _ROOT, "saves", "stages", "stage_1-4_safe5400.state"))
    ap.add_argument("--out", default=os.path.join(
        _ROOT, "saves", "stages", "stage_1-4boss.state"))
    ap.add_argument("--min-dx", type=int, default=100)
    ap.add_argument("--max-frames", type=int, default=3000)
    args = ap.parse_args()

    nes = NES(ROM)
    nes.load(load_state_file(args.from_state))
    best = None
    # phase 1: run right until the boss shows up at a safe distance
    for f in range(args.max_frames):
        nes[R.STAMINA] = 10
        nes[R.DEATH] = 0                      # survive the run-up only
        nes.controller = (NES_INPUT_RIGHT     # the snapshot itself is clean
                          | (NES_INPUT_A if f % 24 < 4 else 0))
        nes.step()
        nes.controller = 0
        dx = head_dx(nes)
        if dx is not None and dx >= args.min_dx:
            print(f"  f{f}: boss dx={dx} -- now letting Higgins land",
                  flush=True)
            break
    else:
        print("the boss never showed up at that distance")
        return
    # phase 2: hands off until he is on the floor again
    for _ in range(120):
        nes[R.STAMINA] = 10
        nes[R.DEATH] = 0
        nes.controller = 0
        nes.step()
        nes.controller = 0
        if int(nes[YL]) >= GROUND_Y:
            break
    dx = head_dx(nes)
    py = int(nes[YL])
    if dx is None or dx < 60:
        print(f"  gave up: dx={dx} y={py}")
        return
    best = (dx, bytes(nes.save()))
    print(f"  snapshot: boss dx={dx}  player y={py}", flush=True)

    dx, blob = best
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "wb") as fh:
        fh.write(blob)
    print(f"saved {args.out}: boss {dx} px ahead")

    # --- how much does the env warm-up eat? --------------------------
    nes.load(load_state_file(args.out))
    for n in (0, 60, 120, 180, 240):
        nes2 = NES(ROM)
        nes2.load(load_state_file(args.out))
        rng = np.random.default_rng(0)
        for _ in range(n):
            nes2.controller = int(rng.choice(
                (0x00, NES_INPUT_RIGHT, NES_INPUT_RIGHT | NES_INPUT_A)))
            nes2.step()
        nes2.controller = 0
        print(f"  after a {n:3d}-frame warm-up: boss dx={head_dx(nes2)}")


if __name__ == "__main__":
    main()
