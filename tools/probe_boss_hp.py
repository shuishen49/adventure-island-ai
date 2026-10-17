"""Find the boss HP byte on the round-4 arena (stage 1-4).

Plan: god-mode bot runs to the arena (progress stops at ~2591 px because
the camera locks for the boss fight), then it keeps throwing axes at the
boss (object ids 198 + 67, two slots that always appear together).  We
snapshot the WHOLE object array region ($0564..$06A4 -- 13 parallel
arrays of 16 bytes, only 8 of which are named in ram_map.ObjSlot) every
decision, and afterwards look for a byte that ticks DOWN as the boss is
hit -- that is the HP / hit counter we can use as a dense reward.

    python tools/probe_boss_hp.py 1-4
"""
from __future__ import annotations

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import ram_map as R
from env import ACTIONS, AdventureIslandEnv

OID, ST = R.OBJ.id, R.OBJ.STRIDE
A_RIGHT = [i for i, a in enumerate(ACTIONS) if a[1] == "right"][0]
A_RJ = [i for i, a in enumerate(ACTIONS) if a[1] == "right+jump"][0]
A_RT = [i for i, a in enumerate(ACTIONS) if a[1] == "right+throw"][0]

BOSS_IDS = {198, 67}
LO, HI = 0x0564, 0x06A4


def snap(nes) -> np.ndarray:
    return np.frombuffer(bytes(nes[i] for i in range(LO, HI)), dtype=np.uint8).copy()


def main() -> None:
    name = sys.argv[1] if len(sys.argv) > 1 else "1-4"
    area, rnd = (int(p) for p in name.split("-"))
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    state = os.path.join(root, "saves", "stages", f"stage_{name}.state")
    env = AdventureIslandEnv(start_state=state, start_with_axe=True,
                             stage_index=(area - 1) * 4 + (rnd - 1))
    env.reset(seed=1)
    nes = env._nes

    # --- phase 1: reach the arena (the boss slots are the only proof) --
    prog = 0.0
    for t in range(3000):
        if int(nes[R.DEATH]) != 0:
            nes[R.DEATH] = 0
        nes[R.STAMINA] = 10
        prog = max(prog, env._cam.progress(nes))
        if any(int(nes[OID + ST * s]) in BOSS_IDS for s in range(16)):
            break
        env.step(A_RJ if t % 26 in (0, 1) else (A_RT if t % 11 == 0 else A_RIGHT))
    print(f"到达竞技场 t={t} prog={env._cam.progress(nes):.0f} (最远 {prog:.0f})")

    slots = [s for s in range(16) if int(nes[OID + ST * s]) in BOSS_IDS]
    print("boss slots:", slots,
          [int(nes[OID + ST * s]) for s in slots])

    # --- phase 2: pelt it with axes and watch every byte ---------------
    frames: list[np.ndarray] = []
    marks: list[tuple] = []
    hits = 0
    for t in range(900):
        if int(nes[R.DEATH]) != 0:
            nes[R.DEATH] = 0
        nes[R.STAMINA] = 10
        nes[R.WEAPON] = 1                      # always armed
        # face the boss and throw: alternate press / release so B re-triggers
        act = A_RT if t % 6 in (0, 1) else A_RIGHT
        env.step(act)
        frames.append(snap(nes))
        marks.append((int(nes[R.SCORE]),
                      int(nes[R.STAGE_COMPLETE]),
                      tuple(int(nes[OID + ST * s]) for s in slots)))
        if int(nes[R.STAGE_COMPLETE]) != 0:
            print(f"  过关! t={t}  score={marks[-1][0]}")
            break

    arr = np.stack(frames)
    print(f"记录 {len(frames)} 帧")
    # bytes that ever DECREASE by exactly 1..8 and never increase much:
    d = arr[1:].astype(int) - arr[:-1].astype(int)
    score = ((d < 0) & (d >= -8)).sum(axis=0)          # n decreasing steps
    jumps = (d > 8).sum(axis=0)                        # n big jumps (noise)
    rng = arr.max(axis=0) - arr.min(axis=0)
    cand = [(i, int(score[i]), int(rng[i]), int(arr[0, i]))
            for i in range(arr.shape[1])
            if score[i] >= 3 and jumps[i] <= score[i] and rng[i] <= 40]
    cand.sort(key=lambda c: -c[1])
    print("\n 地址   递减次数  变化幅度  起始值  (候选血量字节)")
    for i, n, rg, v0 in cand[:15]:
        print(f"  ${LO+i:04X}    {n:4d}      {rg:3d}      {v0:3d}")
    for i, n, rg, v0 in cand[:3]:
        seq = arr[:40, i].tolist()
        print(f"\n${LO+i:04X} 前40帧: {seq}")
    env.close()


if __name__ == "__main__":
    main()
