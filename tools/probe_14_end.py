"""How long IS stage 1-4, and where does the boss actually stand?

The "boss at 3592 px" figure came from recon scripts that *stalled* there —
ppo_14c now runs to 3760..4640 px and still dies, and $0011 never moves
(tools/probe_0011_series.py, 6 episodes: uniq=[8], 0 drops).  So 3592 is
just where the old scripted runner got stuck, not the arena.

This drives a cheat runner (no death, no starvation, axe always held) all
the way to the stage flag and reports:
  * the progress value where $003C (STAGE_COMPLETE) finally fires
  * whether $0011 ever moves, and where
  * the frozen / scrolling stretches (camera stops = arena)

    python tools/probe_14_end.py [--max-frames 40000]
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

OUT = os.path.join(_ROOT, "outputs")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-frames", type=int, default=60 * 60 * 8)
    args = ap.parse_args()

    env = AdventureIslandEnv(start_with_axe=True, start_round=3,
                             stage_index=3)
    env.reset(seed=0)
    nes = env._nes
    cam = env._cam

    best, stall = 0.0, 0
    hp_seen = {int(nes[0x0011]): 0}
    log = []
    complete_at = None
    prev_cam = int(nes[R.CAMERA])
    frozen_from = 0

    for f in range(args.max_frames):
        nes[R.DEATH] = 0
        nes[R.STAMINA] = 10
        nes[R.WEAPON] = 1
        # run right, hop, and throw now and then so we do not get stuck on
        # an enemy we cannot walk through
        ctl = NES_INPUT_RIGHT
        if f % 26 < 4:
            ctl |= NES_INPUT_A
        if f % 40 == 0:
            ctl |= NES_INPUT_B
        nes.controller = ctl
        nes.step()
        if f % 4 == 3:
            env._push_frame(np.asarray(nes.step(), dtype=np.uint8))

        p = cam.progress(nes)
        hp_seen[int(nes[0x0011])] = min(hp_seen.get(int(nes[0x0011]), f), f)
        if nes[R.STAGE_COMPLETE] != 0 and complete_at is None:
            complete_at = (p, f)
            print(f"  >>> STAGE_COMPLETE @ {p:.0f} px (frame {f})",
                  flush=True)
            break
        cam_now = int(nes[R.CAMERA])
        if cam_now != prev_cam:
            prev_cam = cam_now
        if p > best + 0.5:
            best, stall = p, 0
        else:
            stall += 1
        if f % 300 == 0:
            log.append((f, p, cam_now, int(nes[0x0011])))
            print(f"  f={f:6d} progress={p:7.1f} camera={cam_now:3d} "
                  f"$0011={int(nes[0x0011])}", flush=True)
        if stall > 45 * 60:
            print(f"  卡住 {best:.0f} px @ frame {f} "
                  f"(camera={cam_now}, $0011={int(nes[0x0011])})", flush=True)
            break

    print()
    print(f"最远 {best:.0f} px")
    print("$0011 出现过的取值 -> 首次出现帧:", hp_seen)
    if complete_at:
        print(f"通关发生在 {complete_at[0]:.0f} px (frame {complete_at[1]})")
    else:
        print("没有触发 STAGE_COMPLETE")


if __name__ == "__main__":
    main()
