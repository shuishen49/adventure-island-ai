"""What IS the 1-4 end-of-stage encounter, and can an axe beat it?

Open questions after $0011 turned out to be a timer (tools/
probe_0011_damage.py: standing still and throwing axes give the identical
7->6->5->4->3->2->1 series, one step every 256 frames):

  * is object id 59 the boss, or just a platform (ram_map calls it PLAT?)?
  * does the stage ever complete?  what has to happen?

This script walks to the end with the invulnerability cheat, prints the
frame as ASCII art (I cannot look at PNGs), then runs two long branches
from the same snapshot -- axes spam vs. standing still -- and reports what
changed: STAGE_COMPLETE, the object id set, $0011.

    python tools/probe_boss_truth.py
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


def ascii_art(frame: np.ndarray, w: int = 80, h: int = 34) -> str:
    g = frame.astype(np.float32).mean(axis=2)
    H, W = g.shape
    ys = np.linspace(0, H, h + 1).astype(int)
    xs = np.linspace(0, W, w + 1).astype(int)
    out = []
    for i in range(h):
        row = ""
        for j in range(w):
            blk = g[ys[i]:ys[i + 1], xs[j]:xs[j + 1]]
            v = blk.mean() / 255.0
            row += RAMP[min(len(RAMP) - 1, int(v * len(RAMP)))]
        out.append(row)
    return "\n".join(out)


def ids(nes):
    return {int(nes[R.OBJ.id + R.OBJ.STRIDE * s]) for s in range(1, 16)}


def branch(snap, mode: str, frames: int = 4000):
    env = AdventureIslandEnv(start_with_axe=True, start_round=3,
                             stage_index=3)
    env._reset_internal_state()
    env._nes.load(np.frombuffer(snap, dtype=np.uint8).copy())
    env._cam = R.CameraTracker(env._nes)
    nes = env._nes
    hp0 = int(nes[0x0011])
    prev = hp0
    hp_series, seen, complete_at = [], set(), None
    alive59 = 0
    for f in range(frames):
        nes[R.DEATH] = 0
        nes[R.STAMINA] = 10
        nes[R.WEAPON] = 1
        if mode == "throw":
            nes.controller = (NES_INPUT_B
                              | (NES_INPUT_A if f % 12 < 3 else 0))
        else:
            nes.controller = 0
        arr = np.asarray(nes.step(), dtype=np.uint8)
        cur = ids(nes)
        seen |= cur
        if 59 in cur:
            alive59 += 1
        hp = int(nes[0x0011])
        if hp != prev:
            hp_series.append((f, prev, hp))
            prev = hp
        if nes[R.STAGE_COMPLETE] != 0 and complete_at is None:
            complete_at = f
            print(f"      >>> STAGE_COMPLETE @ f={f} ($0011={hp})",
                  flush=True)
            break
    return {"hp0": hp0, "hp1": int(nes[0x0011]), "hp_series": hp_series,
            "seen": sorted(seen), "complete_at": complete_at,
            "alive59": alive59, "frames": frames,
            "progress": env._cam.progress(nes)}


def main() -> None:
    env = AdventureIslandEnv(start_with_axe=True, start_round=3,
                             stage_index=3)
    env.reset(seed=0)
    nes = env._nes
    cam = env._cam
    snap = None
    best, stall = 0.0, 0
    for f in range(60 * 60 * 10):
        nes[R.DEATH] = 0
        nes[R.STAMINA] = 10
        nes[R.WEAPON] = 1
        nes.controller = NES_INPUT_RIGHT | (NES_INPUT_A if f % 26 < 4 else 0)
        arr = np.asarray(nes.step(), dtype=np.uint8)
        p = cam.progress(nes)
        if p > best + 0.5:
            best, stall = p, 0
        else:
            stall += 1
        if 59 in ids(nes):
            snap = nes.save()
            print(f"id59 出现 f={f} progress={p:.0f} $0011={int(nes[0x0011])}",
                  flush=True)
            print("\n--- 画面 (ASCII) ---")
            print(ascii_art(arr))
            print("--------------------\n", flush=True)
            break
        if stall > 45 * 60:
            print(f"卡住 {best:.0f} px", flush=True)
            return

    for mode in ("stand", "throw"):
        r = branch(snap, mode)
        print(f"\n[{mode}] {r['frames']} 帧:  $0011 {r['hp0']} -> {r['hp1']}")
        print(f"    变化: {r['hp_series']}")
        print(f"    id59 在场 {r['alive59']}/{r['frames']} 帧   "
              f"出现过的 id: {r['seen']}")
        print(f"    STAGE_COMPLETE: {r['complete_at']}   "
              f"结束时 progress={r['progress']:.0f}")


if __name__ == "__main__":
    main()
