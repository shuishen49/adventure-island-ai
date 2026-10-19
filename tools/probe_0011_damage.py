"""$0011 drops while id 59 (the boss) is on screen -- but is that DAMAGE?

Two readings are still alive:

  (A) $0011 = boss HP, dropping only when an axe connects  -> safe to pay
  (B) $0011 = a phase / attack counter that ticks on its own, or that
      drops when the PLAYER is hit                         -> paying for it
      would reward getting hurt

Decisive test: snapshot the exact frame where the boss first appears with
$0011 still at 8, then run two branches from that same snapshot:

    stand   no input at all (cheat keeps us alive) for 1500 frames
    throw   jump+throw spam for 1500 frames

If "stand" also makes $0011 fall, it is NOT axe damage.

    python tools/probe_0011_damage.py
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

BOSS = 59
FRAMES = 1500


def ids(nes):
    return {int(nes[R.OBJ.id + R.OBJ.STRIDE * s]) for s in range(1, 16)}


def run(env, mode: str):
    nes = env._nes
    hp0 = int(nes[0x0011])
    series, boss_frames = [], 0
    prev = hp0
    for f in range(FRAMES):
        nes[R.DEATH] = 0
        nes[R.STAMINA] = 10
        nes[R.WEAPON] = 1
        if mode == "stand":
            nes.controller = 0
        else:                                  # throw: jump+throw spam
            nes.controller = NES_INPUT_B | (NES_INPUT_A if f % 12 < 3 else 0)
        nes.step()
        if BOSS in ids(nes):
            boss_frames += 1
        hp = int(nes[0x0011])
        if hp != prev:
            series.append((f, prev, hp))
            prev = hp
    return hp0, int(nes[0x0011]), series, boss_frames


def main() -> None:
    # --- get to the boss, without ever throwing -----------------------
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
        nes.step()
        p = cam.progress(nes)
        if p > best + 0.5:
            best, stall = p, 0
        else:
            stall += 1
        if BOSS in ids(nes):
            snap = nes.save()
            print(f"boss 出现: f={f} progress={p:.0f} $0011="
                  f"{int(nes[0x0011])} -> 已快照", flush=True)
            break
        if stall > 45 * 60:
            print(f"卡住 {best:.0f} px, 没找到满血 boss", flush=True)
            return

    for mode in ("stand", "throw"):
        e = AdventureIslandEnv(start_with_axe=True, start_round=3,
                               stage_index=3)
        e._reset_internal_state()
        e._nes.load(np.frombuffer(snap, dtype=np.uint8).copy())
        e._cam = type(e._cam)(e._nes) if False else None
        # rebuild the tracker so progress/stall state is fresh
        import ram_map as _R
        e._cam = _R.CameraTracker(e._nes)
        hp0, hp1, series, bf = run(e, mode)
        print(f"\n[{mode}] $0011 {hp0} -> {hp1}   "
              f"boss 在场帧数 {bf}/{FRAMES}")
        print(f"    变化: {series if series else '无'}")


if __name__ == "__main__":
    main()
