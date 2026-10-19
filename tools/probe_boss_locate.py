"""Locate the 1-4 boss by its HUD bar, in PIXELS, along the whole stage.

I cannot eyeball screenshots, so "where is the boss" has to be answered
numerically.  The boss HP bar is drawn with background tiles on the dark
HUD strip at the top of the frame (rows 36..46, x 75..150) -- 11 bright
blocks when the fight starts.  Scanning that strip while a cheat runner
(no death, no starvation, axe always held) walks / climbs through 1-4
tells us exactly at which progress value a boss is on screen.

Also logs $0011 and dumps the full object list at every bar sighting.

    python tools/probe_boss_locate.py
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
from env import (AdventureIslandEnv, NES_INPUT_RIGHT,      # noqa: E402
                 NES_INPUT_A, NES_INPUT_B)


def bar_len(frame: np.ndarray) -> int:
    """number of bright blocks on the HUD strip (boss HP bar)."""
    strip = frame[36:46].astype(np.int16).mean(axis=2)
    cols = (strip > 120)[:, 75:150].any(axis=0)
    runs, in_run = 0, False
    for c in cols:
        if c and not in_run:
            runs += 1
            in_run = True
        elif not c:
            in_run = False
    return runs


def objects(nes):
    out = []
    for s in range(1, 16):
        oid = int(nes[R.OBJ.id + R.OBJ.STRIDE * s])
        if oid == 0:
            continue
        out.append((s, oid, round(R.px(R.read_obj_world_x(nes, s)), 1),
                    round(R.px(R.read_obj_world_y(nes, s)), 1)))
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-frames", type=int, default=60 * 60 * 10)
    ap.add_argument("--no-throw", action="store_true")
    args = ap.parse_args()

    env = AdventureIslandEnv(start_with_axe=True, start_round=3,
                             stage_index=3)
    env.reset(seed=0)
    nes = env._nes
    cam = env._cam

    best, stall = 0.0, 0
    seen_bar_at = []
    prev_hp = int(nes[0x0011])
    prev_bar = 0
    hist = []

    for f in range(args.max_frames):
        nes[R.DEATH] = 0
        nes[R.STAMINA] = 10
        nes[R.WEAPON] = 1
        ctl = NES_INPUT_RIGHT
        if f % 26 < 4:
            ctl |= NES_INPUT_A
        if not args.no_throw and f % 40 == 0:
            ctl |= NES_INPUT_B
        nes.controller = ctl
        arr = np.asarray(nes.step(), dtype=np.uint8)
        if f % 4 == 3:
            env._push_frame(arr)
        p = cam.progress(nes)
        if p > best + 0.5:
            best, stall = p, 0
        else:
            stall += 1

        if f % 10 == 0:
            b = bar_len(arr)
            hp = int(nes[0x0011])
            hist.append((f, p, b, hp))
            if b >= 6 and prev_bar < 6:
                seen_bar_at.append((f, p, b))
                print(f"  ** 血条出现 f={f:6d} progress={p:7.1f} "
                      f"格数={b}  $0011={hp}", flush=True)
                print(f"     物体: {objects(nes)}", flush=True)
            if hp != prev_hp:
                print(f"  $0011 {prev_hp} -> {hp}  f={f:6d} "
                      f"progress={p:7.1f}  血条={b}", flush=True)
                prev_hp = hp
            prev_bar = b

        if stall > 45 * 60:
            print(f"  卡住 {best:.0f} px @ f={f}", flush=True)
            break

    print(f"\n最远 {best:.0f} px")
    print("血条(bright blocks)时间序列抽样:")
    for (f, p, b, hp) in hist[::20]:
        print(f"  f={f:6d} progress={p:7.1f} bar={b:2d} $0011={hp}")


if __name__ == "__main__":
    main()
