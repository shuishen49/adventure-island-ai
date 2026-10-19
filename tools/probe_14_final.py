"""What actually ends 1-4?

Facts so far (all measured today):
  * 1-4 keeps scrolling well past the old "3592 px boss arena" claim --
    diag runs reach 4872 px and the camera is still moving.
  * STAGE_COMPLETE ($007A) has NEVER fired on 1-4 in any probe so far.
  * id 5 shows up only in the last ~1200 px, is lethal on contact and
    survived 900 frames of axe throwing.
  * id 59 survived 4000 frames of axe throwing (probably a platform).

So: run right forever under the invincibility cheat and watch (a) how far
the camera can go, (b) whether $007A ever fires, (c) which object ids
show up in the last stretch.  Then, un-cheated, see whether anything
there can actually be destroyed.

    python tools/probe_14_final.py
"""
from __future__ import annotations

import os
import sys

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
sys.path.insert(0, _ROOT)

import ram_map as R                                        # noqa: E402
from env import (AdventureIslandEnv, NES_INPUT_RIGHT,      # noqa: E402
                 NES_INPUT_A, NES_INPUT_B)

RAMP = " .:-=+*#%@"


def ascii_art(frame: np.ndarray, w: int = 76, h: int = 30) -> str:
    g = frame.astype(np.float32).mean(axis=2)
    H, W = g.shape
    ys = np.linspace(0, H, h + 1).astype(int)
    xs = np.linspace(0, W, w + 1).astype(int)
    return "\n".join(
        "".join(RAMP[min(len(RAMP) - 1,
                         int(g[ys[i]:ys[i + 1], xs[j]:xs[j + 1]].mean()
                             / 255.0 * len(RAMP)))]
                for j in range(w))
        for i in range(h))


def ids(nes):
    out = []
    for s in range(16):
        v = int(nes[R.OBJ.id + R.OBJ.STRIDE * s])
        if v:
            out.append((s, v))
    return out


def main() -> None:
    env = AdventureIslandEnv(start_with_axe=True, start_round=3,
                             stage_index=3)
    env.reset(seed=0)
    nes = env._nes
    cam = env._cam

    print("=== phase 1: invincible run to the far end (max 30000 frames) ===")
    prev = -1.0
    stall = 0
    last_frame = np.zeros((240, 256, 3), np.uint8)
    last_frame3 = np.zeros((240, 256, 3), np.uint8)
    snapshots = {}
    for f in range(30000):
        nes[R.DEATH] = 0
        nes[R.STAMINA] = 10
        nes[R.WEAPON] = 1
        nes.controller = NES_INPUT_RIGHT | (NES_INPUT_A if f % 22 < 4 else 0)
        last_frame = np.asarray(nes.step(1), dtype=np.uint8)
        p = cam.progress(nes)
        if f % 300 == 0:
            print(f"  f={f:6d} prog={p:7.0f} cam={int(nes[R.CAMERA]):4d} "
                  f"$007A={int(nes[R.STAGE_COMPLETE])} "
                  f"ids={sorted(set(v for _, v in ids(nes)))}", flush=True)
        if int(nes[R.STAGE_COMPLETE]) != 0:
            print(f"  >> STAGE_COMPLETE fired at f={f} prog={p:.0f}!",
                  flush=True)
            snapshots["complete"] = (f, p)
            break
        if p > prev + 0.5:
            prev = p
            stall = 0
        else:
            stall += 1
            if stall > 900:
                print(f"  >> progress stuck at {p:.0f} for 900 frames "
                      f"(f={f}); camera={int(nes[R.CAMERA])}", flush=True)
                snapshots["stuck"] = (f, p)
                break
    else:
        print("  >> ran 30000 frames without ending or stalling", flush=True)

    # --- dump the scene wherever we stopped ----------------------------
    for tag, (f, p) in snapshots.items():
        print(f"\n--- {tag}: f={f} progress={p:.0f} ---")
        print(f"    $007A={int(nes[R.STAGE_COMPLETE])} "
              f"$0076={int(nes[R.STAMINA])} $003F={int(nes[R.LIVES])} "
              f"$0037={int(nes[R.AREA])} $0038={int(nes[R.ROUND])}")
        print(f"    slots: {ids(nes)}")
        print(ascii_art(last_frame))

    # --- phase 2: how did 1-3 end?  (it DOES clear) --------------------
    print("\n=== phase 2: control -- the same walk on 1-3 (known to clear) ===")
    env3 = AdventureIslandEnv(start_with_axe=True, start_round=2,
                              stage_index=2)
    env3.reset(seed=0)
    nes3 = env3._nes
    cam3 = env3._cam
    fired = None
    for f in range(30000):
        nes3[R.DEATH] = 0
        nes3[R.STAMINA] = 10
        nes3[R.WEAPON] = 1
        nes3.controller = NES_INPUT_RIGHT | (NES_INPUT_A if f % 22 < 4 else 0)
        last_frame3 = np.asarray(nes3.step(1), dtype=np.uint8)
        if int(nes3[R.STAGE_COMPLETE]) != 0:
            fired = (f, cam3.progress(nes3))
            break
        if f % 1500 == 0:
            print(f"  f={f:6d} prog={cam3.progress(nes3):7.0f} "
                  f"$007A={int(nes3[R.STAGE_COMPLETE])}", flush=True)
    print("  1-3 result:",
          (f"$007A fired at f={fired[0]} progress={fired[1]:.0f}"
           if fired else "never fired"), flush=True)
    if fired:
        print(f"    slots at clear: {ids(nes3)}")
        print(ascii_art(last_frame3))


if __name__ == "__main__":
    main()
